# Prefill (TTFT) Optimization Plan — Qwen3.8-27B

Measured on this checkout's running server (`incoai/Qwen3.8-27B-RichEngine`, port
8300, M5 Pro 48 GB), 2026-10-04.

## Baseline

| Request | Wall | Prompt tokens | Cached | GPU prefill |
| --- | ---: | ---: | ---: | ---: |
| Cold | 17.6 s | 8,058 | 0 | ~15.6 s (~515 tok/s) |
| Exact-prefix replay | 0.46 s | 8,058 | 8,032 | ~0.1 s |

`/status` latency stage totals across three requests: tokenization 18 ms,
template 2 ms, preparation 21 ms, images 0 ms, grammar 0 ms (no tools),
`native_queue` 1.7 s (GPU serialization), `http_ttft` 23.5 s. **~89% of cold
TTFT is `model_timing.prefill` GPU time.** Host-side stages are noise until
GPU time comes down.

Prefill runs in `ExecutionLimits::prefillTokenBudget = 2048`-row chunks
(`runtime/model/Model.hpp:379`); 8,058 tokens = 4 chunks × 64 layers at
~61 ms/layer. The staged/prefill tiles already run on the GPU's neural
accelerators (`matmul2d` / MPP tensor ops in `kernels/common/*tile.h` and
`gguf_prefill_<format>`), so prefill is **compute-bound, not
bandwidth-bound**: weight streaming (4 chunks × ~15 GB q4) has a ~0.2 s floor
at 273 GB/s.

## Levers, ranked

### 0. Chunked-parallel GDN scan — new highest-priority finding

The 27B is a hybrid model: **48 of 64 layers are Gated DeltaNet linear
attention**. RichEngine's prefill scan (`runtime/metal/kernels/prefill/gdn.metal`)
is the **serial per-token recurrence**: one threadgroup carries a value
head's 128×128 fp32 state through the whole 2048-token chunk
token-by-token (`S = d·S; m = S·k; S += k(v−m)β; o = S·q`), staged in
16-token blocks. Parallelism exists only across heads×state-rows, so the
sequential depth is the full chunk length on every GDN layer.

The literature is unambiguous that this is the wrong prefill form:

- The recurrence has a well-known **chunkwise-parallel WY/UT (DPLR) form**
  (Yang et al., Gated DeltaNet; Kimi Linear KDA): intra-chunk causal
  correction as a handful of GEMM-shaped ops + one triangular solve, plus a
  decayed state carry per chunk — cutting sequential depth by ~chunk-factor
  and turning memory-bound scan steps into NA-friendly matmuls.
- lattice (ohdearquant/lattice#235) measured **~1.55–1.65× end-to-end
  prefill** switching serial→chunked-parallel on a GDN Qwen, C=32, with
  argmax-parity gates; their follow-up scopes B=64–128 + `simdgroup_matrix`
  as further headroom.
- FlashQLA (Alibaba, TileLang) reports **2–3× on the GDN chunked-prefill
  forward** vs FLA Triton by fusing the memory-bound pieces (K,V,O staging,
  gate preprocessing) — directly relevant since RichEngine's scan similarly
  re-reads staged blocks.
- sglang-jax calls the per-token scan "the dominant cost on long-context
  prefill" for the same architecture family.

This likely beats lever 1 in total impact: GDN layers are 3/4 of the model,
and the serial scan depth grows with chunk length — it's also the piece that
degrades worst at 32K+ prompts. The pieces (`KS0^T`, `KK^T`, `QS0^T`,
`U^TK`, triangular solve) are GEMM-like at M=chunk — candidates for the same
`matmul2d` path the linear tiles already use.

Work items: implement chunkwise WY/UT scan behind a flag; parity gate =
argmax agreement + ≤1e-2 logit bound vs serial reference across a boundary
sweep (lattice's methodology); autotune chunk factor {32,64,128,256};
expect the existing `RICHENGINE_GDN_SCAN_*` geometry knobs to become the
intra-chunk tile shape.

### 1. fp8e4m3 activation-quantized prefill — biggest GEMM lever

TODO.md #3: "the only remaining NA-throughput play; gated on
activation-accuracy, not kernel work." The NA path is already in place; the
missing piece is feeding it fp8 activations (A-operand) instead of bf16,
which doubles matmul2d throughput on Apple10.

- Gate: measure activation error on the affine Q4 path (`affine-checkpoint`,
  `affine-source-oracle` engine tests) before kernel work.
- Kernel surface: `linear_q4.metal` prefill tile, `q4_mpp_tiles.h` A-stage,
  plus an fp8 quantize epilogue on each producer (norm, SiLU-gate, attention
  out) matching the existing `table64`/`table16` producer-consumer pattern.
- Fallback: keep bf16 activations for the first/last layer and any
  projection whose accuracy gate fails.

### 2. GPU-driven dispatch — biggest untapped host lever

TODO.md #1. Every layer's ~10 ops × 64 layers × 4 chunks are encoded on the
host. `last_wall_ms ≈ last_gpu_ms` shows encoding mostly overlaps GPU work,
but the accumulated per-chunk CPU cost is the wall–gpu gap and caps
chunk-pipelining gains. Metal 4's `MTL4MachineLearningCommandEncoder` /
on-device command generation is the target architecture. This is an
architecture project — schedule after lever 1 lands (it changes what the
dispatches look like).

### 3. Q4 packed-tile epilogue staging — small, independent

TODO.md #2: `q4_mpp_tiles.h` reads `scales[]`/`biases[]` per element per
group from device memory. Stage each group's scale+bias into threadgroup
memory once (~128 B/group). The 27B findings doc measured the register-cached
variant 21–43% slower — implement the *staged* form only, target m24/m32
tiles where the 30–40% gap lives. Expect single-digit % on affected shapes.

### 4. Prefill budget / chunk policy A/B

`prefillTokenBudget` is fixed at 2048 and `kPrefillCheckpointTokens` at 4096
(`runtime/engine/Engine.hpp:25`). Larger chunks amortize per-chunk dispatch,
checkpoint-eligibility passes, and draft-context planning; smaller chunks
improve `--decode-share` interleaving. Mechanical sweep via
`dev/tuning/TuningWorkloads` + `make benchmark-backend` at 8K/32K/128K.

### 5. Long-context attention

Quadratic in history; at 32K prompts the rate already drops to ~363 tok/s
(launch table). Verify-attention findings (`dev/benchmarks/Qwen 3.8 27B
Findings.md`) show fused attention at 52–56% of tile bandwidth — that work is
decode-side, but the same kernel-family gap applies to long prefill chunks.
Research directions if it stays hot after levers 0–3:

- **Block-sparse / block-union KV selection for chunked prefill**
  (CompactAttention, arXiv 2605.16839: up to 2.72× attention speedup at
  128K). Fits RichEngine's paged-KV model — selection becomes a block-table
  filter, no KV copies. Accuracy-gated; probably not acceptable for an
  engine that advertises byte-exact numerics.
- **ChunkAttention-style shared-prefix compute** (arXiv 2402.15220:
  3.2–4.8× when requests share system prompts). RichEngine already *caches*
  shared prefixes at page granularity; this would additionally *share the
  attention compute* across concurrent requests with common prefixes —
  relevant only under multi-tenant load. Decode-side variant scoped in
  DECODE_SPEED_PLAN.md §H.

Revisit after levers 0–3 so gains aren't double-counted.

### 6. Already-covered areas (no new work needed)

Research cross-check against standard serving techniques:

- **Chunked-prefill/hybrid batching** (Sarathi-Serve, vLLM): RichEngine already
  does this — 2048-token chunks interleaved with decode under
  `--decode-share`, plus mixed prefill+decode batches.
- **Prefix KV reuse** (RadixAttention-class): already done at page
  granularity, plus `--persistent-cache` and rolling checkpoints.
- **Paged KV**: already the design (`kv::kPageTokens`, page tier, fp8/int8
  KV formats).
- **Speculative prefill** is not a thing — drafting helps decode only
  (DFlash already covers that).
- **MoE expert prefetch**: N/A, the 27B is dense (the 35B-A3B path already
  has `moe_gguf`/`moe_expert` kernels).

## Apple Neural Engine offload — researched, not recommended

The standalone ANE (the Core ML "Neural Engine", distinct from the M5 GPU's
in-core neural accelerators RichEngine already uses via `matmul2d`) is not a
TTFT lever for this model:

- **Throughput**: ANE is a fixed-function fp16 engine, ~19 TFLOPS real peak
  (M5, ~16 NE cores; marketed TOPS double-counts INT8, which it upconverts).
  Measured LLM prefill via private API batching: ~268 tok/s on a 0.8B model
  (AtomGradient hybrid-ane-mlx-bench, M2 Ultra). RichEngine already sustains
  ~515 tok/s on a *27B*. On-chip SRAM ~32 MB cliffs throughput on large
  weight matrices; a 27B's projections blow past it.
- **Placement is expression-shaped**: CoreML decides ANE vs CPU/GPU by how
  ops are written, not what they compute (arXiv 2608.22110). On macOS 26.3
  `compute_units=ALL` routes LLM graphs to GPU anyway; genuine ANE requires
  the undocumented, version-fragile private dispatch route (arXiv
  2606.22283) — unshippable.
- **GDN/DeltaNet layers** (48 of the 27B's 64 layers are linear attention)
  have no ANE mapping; only the 16 full-attention layers' GEMMs could
  theoretically offload, fragmenting the graph and adding KV/state
  serialization (<30 ms per handoff in published work) between engines.
- **The real benefit is concurrency and power, not latency**: ANE prefill
  draws ~0.22 W GPU vs ~62 W, freeing the GPU to keep decoding — a
  scheduling win for multi-request throughput, not single-request TTFT.

Recommendation: do not pursue ANE offload for TTFT. Revisit only as a
power/thermal play (sustained throughput under thermally constrained
prefill) and only if a supported CoreML path ever batches it.

## Free wins (no kernel work)

- Exact-prefix caching already cuts replay TTFT to 0.46 s; keep client
  prompt prefixes byte-stable (tool/JSON ordering) so pages hit.
- `--persistent-cache` survives restarts; without it cold prefill recurs.
- `--decode-share` trades prefill for concurrent decode — prefill slows
  while other requests generate; tune or isolate latency-critical traffic.
- Grammar preparation was 0 here but appears on tool-bearing requests;
  check `/status` `grammar_cache` hit rate when tools are in play.

## Verification

- `make benchmark-backend MODEL=mlx-community/Qwen3.8-27B-4bit` — 2K–128K
  prefill characterization.
- `make test-performance-real MODEL=... BASELINE=<retained checkout>` —
  ABBA regression gate; keep cold prefill, cached TTFT, and decode separate
  per `DEVELOPMENT.md#local-benchmarks`.
- `/status` `model_timing.prefill.total_gpu_ms` delta around a fixed
  8,058-token request — the harness used for the baseline above.

## Measured outcomes (M5 Pro, 2026-10-05)

Every lever from the plan has now been measured on hardware. Results:

### Chunked-parallel GDN scan — implemented, slower, flag left off

`RICHENGINE_GDN_CHUNKED={32,64,128}` dispatches the WY/UT kernels
(`gdn_chunked.metal`, `PrefillTensor::GdnChunkScratch`, `GDNChunkedParams`).
2048-token chunk, best of 5 (`dev/benchmarks/gdn_chunked_bench.mm`):

| Shape | Serial | C=32 | C=64 | C=128 |
| --- | ---: | ---: | ---: | ---: |
| vh48 (27B) | 3.85 ms | 3.94 | 4.89 | 11.76 |
| vh32 (Ornith 9B) | 2.62 ms | 2.83 | 3.40 | 8.12 |

The prep phase round-trips ~200-380 MB fp32 scratch per (head, chunk); at
273 GB/s that alone is ~1.5 ms, and RichEngine's serial scan is already tiled
(128 threads x state rows), so the sequential depth is not the bottleneck
the literature assumes. C=128 also loses accuracy on strong-decay heads
(rowErr 0.08). Code stays behind the flag for a future device where the
tradeoff flips.

### fp8e4m3 prefill activations — dead at two levels

Accuracy gate (Ornith-1.5-9B-MLX-4bit, group-64 absmax): last-token argmax
agrees, KL 0.0003, cosine 0.9915, but greedy continuation diverges at
token 3 — fails the exactness contract.

And it is unreachable anyway: MPP `matmul2d` rejects the fp8e4m3-A x
uint4b-B operand pair (compile-time `static_assert "Unsupported type"`).
Available narrow formats are int2b/int4b/uint2b/uint4b/fp4_e2m1/fp8e4m3/
fp8e5m2 — no int8, so int8 activations are also impossible. fp8xfp8 would
need fp8-weight storage (2x weight bandwidth): net negative.

### Prefill chunk budget — flat

2048 vs 4096 on a 9,886-token cold prompt: 21.30 s vs 21.80 s. Compile-time
`RICHENGINE_PREFILL_TOKEN_BUDGET`; package manifest `execution_geometry`
validates it. Reverted to 2048.

### GPU-driven dispatch — no gap to recover

/status on a 9,862-token prefill: `total_gpu_ms` 30,913 vs `total_wall_ms`
30,929 — host encoding is 0.05% of prefill wall, already fully overlapped.
MTL4 queue/argument-table plumbing exists (`RICHENGINE_MTL4_AVAILABLE`), and
decode already uses baked ICB spans; prefill gains ~nothing.

### Per-dispatch attribution (decode-profile, 512-row prefill, 27B)

| Kernel class | Share |
| --- | ---: |
| `prefill_linear_q4_*_sg4` GEMMs (all) | ~93% |
| `prefill_gdn_scan` (48 layers) | 4.1% |
| attention split/reduce/store (q8) | ~1% |
| norms, gates, captures, embedding, head | ~2% |

The GEMMs run at 26-30 effective TFLOPS (`q4_prefill_profile`, rows=2048),
which is the bf16 x int4 tensor-op ceiling on this part. Prefill is
GEMM-bound at the hardware floor.

### Assembly-level paths — none exist

AGX ISA is private; `air-objdump` shows the kernels already lower to the
hardware tensor-op builtin (`__tensorops_impl_matmul2d_op_run_cooperative_
dv_b16_dv_ui4_f32_v2`). The authored floor is AIR/MSL; nothing below
`matmul2d` is authorable.

### Kernel policy tuning — decode wins only

`make tune-kernels` (88 keys, 27B; 72 keys, Ornith 9B): every prefill key
kept its default. Decode-side winners are encoded as exact-shape
`kMeasuredOverrides` in `runtime/ops/Linear.cpp` — +5.1%..+18.7% GPU on
the 27B, +6.7%..+24.0% on Ornith (incl. its {4096,32768} draft capture
projection).

### Acceptance-path flags on Ornith-1.5-9B — all neutral-to-negative

3-rep median decode tok/s, 256-token completions, cached prompt:

| Flag | Open-ended | Grounded |
| --- | ---: | ---: |
| baseline | 89.4 | 134.8 |
| `RICHENGINE_VERIFY_TREE=1` | 89.3 | 134.0 |
| `RICHENGINE_ADAPTIVE_PROPOSALS=1` | — | 121.7 |
| `RICHENGINE_NGRAM_PREDRAFT=1` | 88.6 | 120.3 |

Outputs byte-identical; no flag earns auto-enable on this model (plain
transformer draft has no tree worth verifying; n-gram traffic doesn't
overlap). Flags stay opt-in.

### Verified structure

Prefill attention is flash-chunked already (`paged_attention_tile.h`:
page-split online softmax, rescale, int8 KV). The ~1,700 -> ~1,080 tok/s
slope from 8K->60K context is linear KV-read growth, not recompute. The
vocab projection already runs only on per-sequence final rows
(`addHeadBatch`), not all chunk rows.

## Remaining levers

- fp4_e2m1 weight format: same 4 bits, drops the group scale/bias
  epilogue — low single-digit % at best; needs a new package format.
- Finer prefill chunks under `--decode-share` for queued-request TTFT
  under concurrency (no cold-TTFT change).
- Upstream context discipline: keep prompt prefixes byte-stable and batch
  tool-result appends — every uncached token is ~0.8-2 ms of TTFT.

### int8 activation x int4 weight — measured, fails exactness

MPP `matmul2d` supports `int8_t/uint8_t` A x `int4b/uint4b` B -> int32
(`__tensorops_impl_matmul2d_op_run_cooperative_*_i8_*_i4_i32`, all address
spaces). Microbenchmark, ffn_up shape M=2048 N=17408 K=5120: 29.1 TFLOPS
bf16xuint4 -> **119.4 TFLOPS uint8xuint4** (~4x the integer NA rate).

Accuracy gate on Ornith-1.5-9B-MLX-4bit (249 patched projections, per-64
group quant, 4 prompts x 256 greedy tokens): symmetric int8 diverges at
tokens 0/31/45/126; asymmetric (min/max + zero point) at 11/26/43/194.
KL up to 0.023. **Fails the greedy-identity contract.** Valid only as a
future opt-in approximate mode (~3x end-to-end TTFT cut if accepted).

NOTE on method: quantized-module patching must traverse
`model.named_modules()` — MLX module children are not visible via
`vars()`/`__dict__` traversal; an earlier gate that reported
"greedy-identical" had patched 0 modules.

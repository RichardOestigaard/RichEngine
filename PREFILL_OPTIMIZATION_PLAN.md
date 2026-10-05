# Prefill (TTFT) Optimization Plan — Qwen3.8-27B

Measured on this checkout's running server (`incoai/Qwen3.8-27B-Splash`, port
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
attention**. Splash's prefill scan (`runtime/metal/kernels/prefill/gdn.metal`)
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
  gate preprocessing) — directly relevant since Splash's scan similarly
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
expect the existing `SPLASH_GDN_SCAN_*` geometry knobs to become the
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
  128K). Fits Splash's paged-KV model — selection becomes a block-table
  filter, no KV copies. Accuracy-gated; probably not acceptable for an
  engine that advertises byte-exact numerics.
- **ChunkAttention-style shared-prefix compute** (arXiv 2402.15220:
  3.2–4.8× when requests share system prompts). Splash already *caches*
  shared prefixes at page granularity; this would additionally *share the
  attention compute* across concurrent requests with common prefixes —
  relevant only under multi-tenant load.

Revisit after levers 0–3 so gains aren't double-counted.

### 6. Already-covered areas (no new work needed)

Research cross-check against standard serving techniques:

- **Chunked-prefill/hybrid batching** (Sarathi-Serve, vLLM): Splash already
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
in-core neural accelerators Splash already uses via `matmul2d`) is not a
TTFT lever for this model:

- **Throughput**: ANE is a fixed-function fp16 engine, ~19 TFLOPS real peak
  (M5, ~16 NE cores; marketed TOPS double-counts INT8, which it upconverts).
  Measured LLM prefill via private API batching: ~268 tok/s on a 0.8B model
  (AtomGradient hybrid-ane-mlx-bench, M2 Ultra). Splash already sustains
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

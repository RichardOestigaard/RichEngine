# DiffusionGemma Optimization Plan (M5)

Analysis for serving `google/diffusiongemma-26B-A4B-it` efficiently on
Apple M5 hardware. Synthesis of two analysis passes: kernel
microarchitecture and denoising-loop structure.

Model: 25.2B MoE trunk (Q4 ~14.5 GB packed), hidden 2816, 30 layers,
128 experts top-8 + shared dense expert (2112). Canvas = 256 tokens,
≤48 denoise steps, entropy-bound accept + renoise, argmax commit,
encoder causal re-prefill per committed block.

## Cost model per denoise step (as implemented)

| Component | Traffic |
|---|---|
| All Q4 weights (all 128 experts touched at M=256) | ~14.5 GB (floor) |
| LM head write: 256×262144 fp32 logits | 268 MB |
| LM head weight re-reads (TileM=32 over M=256) | up to ~3 GB |
| `decode_logit_softcap` + `canvas_logits_scale` (r+w each) | ~1.07 GB |
| `canvas_row_stats` | ~565 MB |
| `canvas_soft_embed_topk` (32-iteration bisection over logits) | ~9.4 GB |
| Prefix KV (locals SWA-capped; globals grow) | 0.1–1 GB |
| **Total** | **~25–28 GB/step** |

Roofline at M5 Max ~614 GB/s: ~41–45 ms/step now vs ~25 ms after fixes
(M5 Pro ≈ ×2, base M5 ≈ ×4). At ~12 typical steps/canvas the logit tail
is ~40% of step time — it is nearly free compute, all bandwidth.

Upstream estimates: ~13–17 steps/256-token canvas, ~15–20 committed
tokens per forward; ~1150 tok/s on H200 FP8.

## Ranked optimizations

### Kernel microarchitecture (M5)

M5 facts: 8–40 GPU cores; 128 KiB per-core unified register/SRAM pool;
SLC ~4–8 MiB; NAX tensor path via `matmul2d` (fp16/bf16/int8/int4).

1. **Rewrite `canvas_soft_embed_topk`** (`decode/canvas.metal`):
   32-pass bisection → single-pass 256-bin histogram (tg atomics) +
   prefix-sum → one accumulation pass. 35 passes → ~3
   (~−8.5 GB/step). Compact qualifying vocab indices to a tg list so
   accumulation avoids ~2048 barriers/row. Adaptive K to ≥99% kept
   mass on early diffuse steps.
2. **Fuse logit transforms**: apply `cap*tanh(x/cap)*invT` in-register
   inside `canvas_row_stats` / soft-embed reads (or head-GEMM fp32
   epilogue) — removes two r+w passes (~1.07 GB/step).
3. **LM head at M=256**: TileM=32 re-reads the 415 MB Q4 head up to 8×.
   TileM=64 variant or one-pass N-tiled head → 1× (~3 GB/step).
4. **hd512 tile split M not D**: 128 fp32/thread PV accumulator consumes
   a whole core's 128 KB pool. Split M in two passes (fp8 precedent,
   `paged_attention_fp8_tile.h`), ~64 fp32/thread — applies to canvas
   and verify/prefill hd512 paths.
5. **`canvas_row_stats` single pass**: online softmax (rescale on max
   update) folds max+sum+argmax; float4 loads.
6. **Cooperative-tensor softmax** (Metal 4.1): keep scores/probs in CT
   registers in `canvas_tile.h`/`paged_attention_tile.h` — frees tg-mem,
   removes a barrier per page.
7. **MoE stays grouped**: at M=256 all experts route anyway;
   `moe_group_routes` + m32 indirect tiles already read each expert
   once. Dense all-experts GEMM would read the same bytes at 16× MACs —
   strictly worse. Optional m32 fused gate/up GeGLU tile: measure.
8. `canvas_entropy_accept` fine as-is (bitonic 256, µs-scale).
9. `canvas_soft_embed_exact` is ~106 GB/step as written — reference
   only; if exact A/B needed, use a real bf16 GEMM (~1.6 GB, 378 GFLOP).
10. Re-probe fp8 `matmul2d` on M5/NAX (emulated on M4) — could halve KV
    bytes for canvas/global tiers.

### Loop / structural

1. **Cross-request canvas batching (biggest serving lever)**: step cost
   is weight-bound at 256 rows — batch ≤4 requests' canvases into one
   forward (`maximumBatchWidth=4` precedent in Scheduler.cpp). ~4×
   throughput at ~1× step latency. Needs multi-sequence canvas plans +
   per-lane scratch extents/page tables.
2. **Submit-ahead step unrolling**: `canvas_entropy_accept` produces the
   next canvas on-device; host only needs stats for early exit. Unroll
   2–4 steps per submission with ping-pong canvas/argmax buffers
   (commit tokens don't change once argmax stable — waste is compute
   only). Removes per-step host turnaround.
3. **Prefix-stable early exit**: once argmax contains a stop at index j,
   tail positions can't affect output — exit on stable+low-entropy
   *prefix* rather than whole-canvas mean entropy.
4. **Speculative encoder re-prefill**: once argmax stabilizes (one step
   before exit), submit the committed block's causal re-prefill
   concurrently — hides ~8%/canvas of re-prefill weight traffic.
5. **Defer/skip dead work**: soft-embed computed for step k+1 is wasted
   on the exit step (mlx skips it); on the commit step only argmax is
   needed — sampling and soft-embed are dead.
6. **Schedule tuning (config-only, needs eval)**: max steps 48→24–32;
   anneal `entropy_bound` with temperature (aggressive early accept);
   optional confidence-threshold sampler A/B (upstream supports both).
7. **Adaptive canvas length**: `min(256, max(remaining, 64))` (mlx does
   this) — skips junk tail canvases; needs CanvasRows as param.
8. **Incremental canvas K/V** — speculative: unchanged positions' K/V
   could persist between steps (bidirectional still needs Q over all
   rows). Gated on per-step accept-rate data; upstream doesn't do it.
9. **Warm-start canvas** from n-gram predraft index / previous tail —
   speculative; risk of quality drift vs uniform init.

## Suggested order

1. Soft-embed histogram rewrite (k1) — largest single win, unblocks
   everything downstream.
2. Fused logit transforms (k2) + single-pass row_stats (k5).
3. LM-head M256 tiling (k3) + hd512 M-split (k4).
4. Submit-ahead unrolling (s2) — no accuracy risk.
5. Prefix-stable exit + dead-work skips (s3, s5).
6. Cross-request batching (s1) — biggest but most invasive.
7. Schedule/canvas-length tuning (s6, s7) after instrumentation data.

## Instrumentation needed

- Per-canvas: steps used, accepted/step, first-stop index, argmax
  stability point, per-pass GPU time (Metal counters).
- Per-request: tokens/forward, committed-canvas entropy profile.
- A/B harness for topk vs exact soft-embed and sampler variants.

## ANE verdict: NO

The Neural Engine path (`ops/AneFfn.cpp`, `ane_ffn.metal`) cannot help
this model:

- Geometry fails all three gates: hidden 2816 not 512/1024-block clean;
  intermediate 2112 (packed 2304) not 512-clean; expert width 768 same
  (`AneFfn::unsupported`, AneFfn.cpp:91-109).
- Row floor 512 > canvas M=256 (AneFfn.hpp:43).
- `aneFfnLayers` only collects `Qwen3_8LayerWeights` (Runtime.mm:1028) —
  Gemma/DiffusionGemma layers never offered.
- Economics are wrong anyway: a canvas step is DRAM-bound; ANE moves
  MACs not bytes and adds int8 staging traffic; 10–60 ms eval latency
  exceeds per-layer GPU time at M=256.

`--disable-ane` is a no-op for diffusion targets.

Additional detail: the ANE split is dense-SwiGLU W8A8 only (Hadamard
rotation, MIL text program via the private AppleNeuralEngine client,
MTLSharedEvent handoff inside one command buffer). Routed experts are
unmappable (fixed dense graph); the shared expert fails `inputSegment`
(2816%512), rotate group (2816%1024), and `kChannelUnit` (2304%512).
ANE eval cost is ~flat in rows (~1.7 ms per 512-channel unit at
2048 rows) — hopeless against ~0.4 ms GPU work at M=256.

## Memory residency / tiers

New per-canvas buffers to plan (needs a `canvasBytes`/scratch entry in
`EngineMemoryBreakdown` — absent from both arena plans today):
fp32 logits 256 MiB per active canvas (268 MB; ×4 lanes if concurrent),
soft-embed 2×1.4 MiB ping-pong, self-conditioning GeGLU ~4 MB,
canvas KV scratch (8 pages per lane — count like
`RICHENGINE_SPECULATIVE_SCRATCH_TOKENS` reserves), stats/argmax
rings <64 KB.

Fixed resident ≈ 15.3 GB (weights + arenas + logits + reserves).

| RAM | Verdict |
|---|---|
| 16 GB | Does not fit |
| 24 GB | Marginal — KV disk tier only, short context |
| 32 GB | Works — int4 KV, ~110–140K context |
| 48 GB | Comfortable — full 262K int4 / ~180K int8 |
| 64 GB+ | Full 262K int8 |
| 96 GB+ | Full 262K bf16 |

⚠️ KV extent granularity: dual geometry + quantized KV formats force
256-page extents (~0.5–0.9 GB per extent, PagedKv.hpp:234-275) — pool
the 8-page canvas scratch separately or into the request extent.

## Host-sync elimination

The canvas step's only host dependency is the early-exit read of
`stats` (`canvas_entropy_accept` already writes `{meanEntropy,
argmaxStable}` on-device). Two designs:

- **Whole-canvas single command (preferred)**: all ≤48 steps in ONE
  submitted command — per-step temperatures/seeds are constants at
  encode time; a device `done` flag in the stats buffer lets every
  kernel no-op after stability (~1–2 ms wasted dispatch vs 35–150 ms
  real work/step). Removes ~47 completion round-trips (~1–3 ms each).
  Fallback: chunk into 8-step commands to bound waste. Well under the
  120 s command watchdog.
- **Unroll K=2–4 steps + one-step-behind stats read** — the simpler
  variant if single-command proves fragile; backend already allows 2
  commands in flight (MetalEncode.mm:692,851).

Caveat: `MTLIndirectCommandBuffer` mis-executes GGUF kernels on the
M5 Pro driver (ISSUES.md) — the affine-Q4 packed path is unaffected.

## Logit-path fusion (preferred over per-pass fixes)

Rather than optimizing the fp32 logits round-trips pass-by-pass, fuse
the **head→stats path**: per-tile logits → softmax/entropy stats and
argmax/sample inside the head GEMM's epilogue, never materializing the
268 MB buffer at all (precedent: fused argmax head partials for
verify, `QwenTarget.hpp:431`). If that proves too invasive, bf16
logits halve every downstream pass — A/B for numerics.

## Thermal

Canvas loop is ~100% duty-cycle DRAM streaming plus high compute
density (M=256 GEMMs on NAX). Fanless base M5 may throttle ~15–25% on
sustained loads; Pro/Max expect ~10–20% droop in multi-canvas runs —
derate the roofline rows accordingly. The bandwidth fixes are also
~35% of per-step energy.

## Refined roofline (per committed token)

| Tier | Step now | Step after fixes | Committed tok/s after |
|---|---:|---:|---:|
| M5 (153 GB/s, ≤32GB) | ~130–168 ms | ~107 ms | ~130–175 |
| M5 Pro (307, ≤64GB) | ~65–84 ms | ~53 ms | ~260–350 |
| M5 Max 32c (460) | ~45–56 ms | ~35 ms | ~380–520 |
| M5 Max 40c (614, ≤128GB) | ~35–42 ms | ~27 ms | ~510–690 |

## Implemented optimizations — A/B results

Measured on this machine via GPU-timestamped native tests
(`canvas-kernels`, `hd512-attention`; MTL_SHADER_VALIDATION on, parity
vs CPU oracle):

| Change | Old | New | Result | Kept |
|---|---:|---:|---|---|
| soft-embed histogram vs 32-pass bisection | ~91–155 ms | ~7.2–15.9 ms | **~10–13×** | ON (`RICHENGINE_CANVAS_EMBED_HIST`) |
| fused single-pass `canvas_row_stats` | 2.7 ms | 1.35–1.6 ms | 1.7× | ON (`RICHENGINE_CANVAS_STATS_FUSED`) |
| standalone softcap + logits-scale passes | ~4.0 ms | 0 (fused on load) | removed | ON when both fused kernels active |
| hd512 attention M-split (`*_hd512_m2`) | — | — | 18–45% faster all formats | ON by default in `PagedAttention.cpp` dispatch |
| steps-per-command unroll (`RICHENGINE_CANVAS_STEPS_PER_CMD`) | per-step wait | 2-step commands | commit-equivalent | ON (default 2, max 4) |
| dead-work skip on commit step (`RICHENGINE_CANVAS_COMMIT_TAIL`) | sampler tail | argmax only | free | ON |
| prefix-stable early exit (`RICHENGINE_CANVAS_PREFIX_EXIT`) | whole-canvas mean | prefix to first stop | cuts steps on eager-eos | ON |
| speculative commit re-prefill (`RICHENGINE_CANVAS_SPECULATIVE_PREFILL`) | — | overlap ~8%/canvas | unmeasured on device | OFF pending device test |

Soft-embed parity vs fp64 reference: rel L2 ~2.7e-2 (top-64-kept
approximation, masses match); argmax/sample paths verified.

Wiring: `Canvas::fusedLogitTail()` gates the standalone passes in
`encodeCanvasStep` and forwards `{cap, invT}` (prev step's invT for
self-conditioning embeds — logits carry the producing step's
temperature).

## Status note

The canvas path is not yet wired end-to-end (ops::Canvas,
DiffusionSampler, prefillCanvasPlan exist; the step driver and buffer
allocation are in flight — runtime integration workstream).

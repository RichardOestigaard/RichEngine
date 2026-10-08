# DiffusionGemma Optimization Analysis

Companion to `GEMMA_DIFFUSION_OPTIMIZATION_PLAN.md`. Measured on M5 Pro (307 GB/s), model `google/diffusiongemma-26B-A4B-it` (Q4, int4 KV).

## Paper targets

arXiv 2608.00146, H100 FP8:

| Metric | Paper |
|---|---|
| Forward per step | 13.56ms |
| Effective steps/canvas | ~12 avg |
| TPF (tokens/forward) | ~20 |
| Output TPS | ~1450 |

## Measured

Instrumented via `RICHENGINE_CANVAS_TIMING` (env-gated, off by default).

| Request | Steps | Wall | ms/step | stop |
|---|---|---|---|---|
| "Capital of France?" | 25 | 3.58s | ~143 | 11 |
| "Autumn poem" | 48 (no exit) | 6.65s | ~138 | 69 |
| Short factual canvases | early exit | ~265ms | ~133 | 0–2 |

- GPU-bound: `gpuSeconds ≈ wallSeconds` per canvas.
- Early exit works. Fired at step 24 on factual prompt (entropy 0.0047 < 0.005).
- Creative prompt never hit threshold — ran all 48 steps.
- ~10× step-time vs paper is mostly hardware: 3.3TB/s HBM + FP8 vs 307GB/s + int4.

## Per-step budget

Weight-bound floor on M5 Pro:

| Component | Bytes | ms @307GB/s |
|---|---|---|
| Q4 weights (measured union ~35–75 of 128 experts; see below) | ~5.4GB | ~18 |
| LM head Q4 + TileM=32 re-reads | up to ~3GB | ~10 |
| bf16 logits 256×262144 write + stats/soft-embed reads | ~0.5GB | ~2 |
| Soft-embed histogram | — | ~8 |
| KV/activations/misc | ~1GB | ~5 |
| **Floor** | | **~43** |

`RICHENGINE_MOE_STATS=1` on a live canvas: per-layer expert union is
~34–75 (of 128), ~85–119 m32 tiles — the earlier "all experts touched"
assumption overcounted weight traffic ~2.5×. Routing concentrates on a
minority of experts per layer.

## Probe attribution (measured, M5 Pro)

`RICHENGINE_CANVAS_PROBE` stage skips on the same 48-step poem canvas:

| Probe | ms/step | Stage removed |
|---|---:|---|
| baseline | 139 | — |
| `notrunk` | 24 | 30-layer trunk |
| `notail` | ~125 | LM head + row stats + accept |
| `nosc` | 136 | self-conditioning chain |

⇒ trunk ≈ **115 ms/step** (~5.4GB at ~50 GB/s, ~17% of peak — the real
gap), logit tail ≈ 14, self-conditioning ≈ 3, residual (final norm,
noise, host) ≈ 7. gpu ≈ wall: host turnaround is ~0.4 ms/step, not the
problem.

## Kernel microbenchmark (measured, M5 Pro)

`make test-moe-prefill-bench` (`moe_prefill_bench_metal_test.mm` +
`moe_prefill_bench.metal` bench-only kernels) at the canvas shape —
256 rows, union-50 synthetic routing (~90 m32 tiles/layer, matching the
live `RICHENGINE_MOE_STATS` union of 34–75), medians of 20 GPU-timestamped
reps, **without** MTL_SHADER_VALIDATION (it inflates kernels ~10×):

| Pass | µs | GB/s(weights) |
|---|---:|---:|
| gate `q4_n256_indirect_m32` (prod) | 513 | 214 |
| up_gelu n256 (prod) | 742 | 148 |
| up_gelu n128 | 611 | 179 |
| up_gelu n64 | 897 | 122 |
| up_gelu n256 sg4 (128-thread tiles) | 1404 | 78 |
| **up_gelu n128 sg4** | **569** | **192** |
| **up_gelu m16 tiles** | **523** | **314 (≈ peak)** |
| up_gelu ksplit2 / ksplit4 | 629 / 782 | 174 / 140 |
| up_gelu n256 pipelined matmuls | 736 | 149 |
| down n256 (prod) | 448 | 244 |
| down n128 / n64 | 512 / 737 | 214 / 149 |
| MoE chain gate+up+down (prod) | 1641/layer | 200 |

Trunk decomposition (~133 ms/step now, with M16): **MoE block ≈ 45-69 ms
(35-52%)** — 49 ms isolated kernels, up to ~69 ms in situ counting
router/group/combine and serialization; dense projections ≈ 33 ms (25%,
on the already-optimal n128_sg4 tile); attention + ~15 small ops/layer ≈
25-30 ms (~20%); unaccounted dispatch/serialization spread across the
rest. The earlier "34 ms" dense figure was measured on the n256 sg8
tile; production plans use n128_sg4 on Apple10+.

Mechanism: at M=256 every trunk GEMM is latency-bound — a threadgroup
serially walks 44 quant groups (~1.5 µs each: staged input or weight
load, matmul, epilogue), and the grids are small (gate/up: 3×~90 TGs;
dense N768: 8×3 = 24 TGs → 66 µs for 1 MB). Wins come from more total
threadgroups (m16 tiles −30% on up_gelu; n128 sg4 −23%) — **not** from
per-TG pipelining (no gain) or K-split (partials traffic eats it at
expert sizes).

### Dense tile A/B (same bench, production-like layer shapes)

| Variant | N768 | N2304 | N2816·K2304 | N8192 | N10240 | N2816·K8192 |
|---|---:|---:|---:|---:|---:|---:|
| n256 sg8 | 316 | 132 | 142 | 485 | 511 | 505 |
| n128 sg8 | 76 | 161 | 148 | 496 | 612 | 553 |
| **n128 sg4** | **56** | **129** | **122** | **431** | **519** | **471** |
| n256 pipelined | 71 | 145 | 146 | 468 | 551 | 574 |
| n256 sg4 (device sums) | 190 | 362 | 455 | 1305 | 1450 | 1698 |
| m64 n256 sg4 | 476 | 655 | 715 | 2740 | 3709 | 2469 |

`prefill_linear_q4_n128_sg4` wins or ties at every shape — and Apple10+
prefill plans already select it (`Linear::baseline`, Linear.cpp). Dense
projections are already on the best tile; m64 (halved re-reads) loses
badly — device sums + fewer threadgroups cost more than the traffic
saves. Pipelined issue only helps the degenerate n256 sg8 tile.

### Landed: M16 prefill expert tiles

`MoeExpertTile::M16` + `_m16` kernels (prefill/moe.metal), selected by
`ExecutionPlans::moeConfig` for affine prefill plans at rows ≤ 256 —
the canvas trunk and any short chunk. e2e: 6.69 s → 6.38 s per 48-step
canvas (139 → 133 ms/step, −4.7%). `moe-metal` suite green (72 cases).

### Per-dispatch attribution (RICHENGINE_OP_TIMINGS, M5 Pro)

Encoder counter sampling is unsupported on this GPU family, so
OP_TIMINGS instead commits each dispatch as its own command buffer
(serial GPU, accurate per-dispatch GPU seconds; wall inflates). Request
deltas (poem canvas, ~50 steps):

| Kernel | ms/step | µs/dispatch | dispatches/step |
|---|---:|---:|---:|
| prefill_linear_q4_n128_sg4 (all dense) | 46.3 | 301 | ~154 |
| moe up_gelu_indirect_m16 | 19.1 | 663 | 29 |
| moe gate indirect_m16 | 18.3 | 635 | 29 |
| moe down_m16 | 17.7 | 615 | 29 |
| moe_route_scores_gemma | 7.9→4.1 (vec) | 262→142 | 30 |
| prefill_linear_q4_sums32 | 3.9 | 64 | 61 |
| norm_rms | 3.0 | 17 | 180 |
| attention (all canvas kernels) | ~7.5 | | ~50 |
| moe_combine | 2.1 | 69 | 30 |
| canvas_soft_embed_histogram | 2.6 | 2723 | ~1 |
| moe_combine | 2.1 | 69 | 30 |
| draft_residual_add | 1.3 | 15 | 90 |
| canvas_row_stats_fused | 1.2 | 1263 | ~1 |
| geglu + layer_scalar + misc | ~1.5 | | ~120 |

Corrections vs the earlier estimates: **dense is the largest block at 46
ms/step, not 33** — in-situ contention ~1.4× the isolated bench.
MoE experts = 55 ms in situ (vs 49 bench); router+combine ≈ 6.3 ms — the
262 µs `moe_route_scores_gemma` was a scalar bf16 strided-load loop;
the canvas-path `moe_route_scores_gemma_vec` (staged row + bfloat4 lanes,
canvas plan only) cuts it to ~142 µs. Attention ≈ 7.5 ms, not ~30 — the
hd512 canvas split DOES dispatch in production (the dev test's 34 KB
tg-mem failure is test-param-only). Norms/sums/small ops ≈ 12 ms across
~330 dispatches — per-dispatch overhead territory.

### Canvas vs normal-path isolation audit

The plan families now split on phase, not size:

| Path | Plan source | Effect |
|---|---|---|
| Canvas trunk MoE | `ExecutionPlans::moeCanvas` (MoePhase::Canvas) | M16 tiles + `moe_route_scores_gemma_vec`; unreachable from Prefill/Decode (ctor rejects non-canvas M16, canvas non-affine) |
| Normal/commit prefill MoE | `moePrefill` (Prefill) | M32 at any rows; commit prefill keeps M32 |
| Decode/verify MoE | `moeDecode` (Decode) | M8 fused tiles; canvas phase never enters verify |
| Canvas dense+head+SC | `LinearWorkload::canvas` via `prefillPlan(..., canvas)` | same tile selection today; hook for canvas-only variants |
| Canvas attention | `prefillCanvasPlan` | canvas split/reduce kernels (pre-existing) |
| Canvas probes | `RICHENGINE_GEMMA_SKIP_*` gated on `step.canvas` | no effect on normal prefill |
| Workspace | `moePrefillWorkspace` unions canvas bound (affine only) | GGUF MoE models skip the canvas bound |
| Batch guard | `addPrefill` throws on mixed canvas+causal sequences | prevents plan-family aliasing mid-batch |

Entry points: only `decodeDiffusion`'s trunk call sets
`sequence.canvas` (`RuntimeDiffusion.mm`); `encodeCanvasCommitPrefill`
and every other `addPrefill` caller leave it false.

### Skip-probe attribution (per-layer, in situ)

`RICHENGINE_GEMMA_SKIP_MOE=1` drops the routed block per layer:
133 → 64 ms/step → **MoE block ≈ 69 ms/step in situ** (vs ~49 ms of
isolated kernel time — ~20 ms is router/group/combine + serialization).
`RICHENGINE_GEMMA_SKIP_ATTN=1` is unusable: stale attention output
propagates denormals into every downstream pass and the step runs
*slowly*, not less (674→~140 ms/step). Attention+sweep of small ops is
the residual ~30 ms — per-kernel attribution needs a finer instrument.

## Ranked opportunities

### 1. Cross-request canvas batching — ~4× throughput

Step cost is weight-bound at M=256. Batching ≤4 canvases into one forward is ~4× TPS at ~1× latency. Most invasive. Biggest serving lever. (Plan doc #1.)

### 2. Adaptive canvas length — ~4× on short answers

`stop=11` ⇒ ~245 of 256 rows were eos padding. `min(256, max(remaining,64))` skips the tail. mlx does this.

### 3. Fewer effective steps — config

Factual prompts exit ~24, poems never exit; paper avg ~12. Levers: `maxSteps` 48→32, entropy-bound annealing, stability threshold. Needs an eval harness before touching — quality risk otherwise.

### 4. fp32 logits elimination — ~3–5ms/step

268MB write + ~800MB reads/step. bf16 logits, or fused head→stats epilogue that never materializes logits.

### 5. LM-head tiling — ~8ms/step

TileM=64 or N-tiled single pass removes up to ~3GB Q4 re-reads.

### 6. Whole-canvas single command — ~1–3ms×12

Device `done` flag in row stats; removes ~12 submit/wait round-trips and dead compute past the exit step. Plan doc marks this already done — verify.

### 7. fp8/NAX `matmul2d` re-probe

Halves KV bytes on canvas/global tiers if functional now.

### 8. Thought-channel eos waste

Canvases that open a thought channel then eos-pad burn ~6s for ~2 emitted tokens. Check whether prompt format can suppress thought preambles (the think-flag token exists in vocab). Reference repo behavior unknown — worth diffing chat templates.

## Already implemented

All ON in current build:

- Histogram soft-embed (10–13× on SC path)
- Fused canvas row stats (1.7×)
- Fused logit transforms, early-exit flag path
- hd512 M-split attention (18–45% on global layers)
- 2-step command unroll
- Commit-tail argmax-only step
- Prefix-stable exit, now gated on the [0, firstStop) argmax prefix *and* the
  prefix mean entropy from the per-slot entropy ring — a canvas that settles
  early but carries a stop mid-row no longer denoises its dead tail
- Speculative commit prefill

## Quick wins to verify first

1. Confirm whole-canvas single command actually landed (plan doc says done).
2. Profile one 48-step canvas with Metal counters — locate the missing ~65ms/step.
3. Check whether TileM is currently 32 or 64 in head dispatch.

## Instrumentation

`RICHENGINE_CANVAS_TIMING=1` prints per-canvas wall/GPU time, stop index, and per-step entropy/stability. Off by default.
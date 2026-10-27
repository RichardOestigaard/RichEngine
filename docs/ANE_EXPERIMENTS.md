# ANE experiments from maderix/ANE

What the [maderix/ANE](https://github.com/maderix/ANE) findings can add to
RichEngine, per model, as opt-in experiments. Context first: the repo proves
training on the Apple Neural Engine through the private `_ANEClient` /
`_ANECompiler` APIs and MIL. For inference it demonstrates a GPU-prefill /
ANE-decode hybrid, INT8 W8A8 (1.88x FP16), dynamic weights passed as inputs,
and documents the hardware's quirks.

## Already adopted

The engine's existing split already implements the repo's core techniques:

| maderix finding | RichEngine equivalent |
| --- | --- |
| Private API + in-memory MIL compile | `runtime/ane/Program.mm` |
| GPU<->ANE shared-event ordering | `ane::Handoff` + `metal::EventStep` |
| INT8 W8A8, ~1.9x over FP16 | `ops::AneFfn`: Hadamard-rotated int8 activations, per-row int8 weights |
| Weights as dynamic inputs (constants can't hot-swap) | Double-buffered weight surfaces staged one layer ahead |
| One program, many procedures share intermediates | 13 functions (512-2048 rows) in one program: 0.67 GB vs 3.25 GB |
| Lossless ANE offload (verify authoritative) | `RICHENGINE_ANE_MEDUSA` / `RICHENGINE_ANE_PREDRAFT` (CoreML route) |

## Experiment catalog

| Experiment | Models | Expected value |
| --- | --- | --- |
| ~~MoE routed-expert FFN on ANE~~ | Qwen3.6-35B-A3B, Ornith-35B-A3B, MiniCPM-V-35B-A3B, LFM2.5-24B, Gemma4-26B | **Measured: loses.** ~0.20 ms fixed cost/eval + GPU slower than the grouped pass per FLOP. See deep dive 2 |
| Padded-channel split | Granite-4.2-3B (hidden 2560), Gemma4 (2816) | Medium — unblocks models that fail `hidden % 1024` |
| Granite-8B enable | Granite-4.2-8B | Trivial — code review says it already passes shape checks; just run it |
| ~~Shared-expert FFN~~ | Qwen MoE family | **Measured: loses.** 2048-row whole-expert eval 1.03 ms vs ~0.62 ms of GPU time |
| ANE attention (primitive-built) | Qwen3.8-27B | Low — GPU prefill attention already strong; SDPA op ignores `attn_mask`, must build from matmul+add+softmax |
| Vision tower on ANE | SigLIP2 (d1-omni), Qwen-VL | Medium — dense conv workload, frees GPU for text; easy verify |
| ~~In-memory-MIL draft predictor~~ | DFlash/DSpark drafts | **Measured: loses.** MIL eval round-trip 0.31 ms vs CoreML predict() 0.25 ms. See deep dive 4 |
| ~~Chained full-forward (`_ANEChainingRequest` loopback)~~ | d1-3B judgments | **Measured: loses on latency.** Forward model ~1.9-2.0x slower than GPU at every prompt size; only value is GPU offload. See deep dive 3 |
|| ~~`perfStatsMask` hardware timing~~ | all ANE paths | **Measured: dead on macOS 27.** Setter + `_ANEPerformanceStats` factory exist; `_ANERequest perfStats:` wants a collection (validate sends `count`); array/object forms all evaluate fine but the driver never populates `hwExecutionTime` for masks {1, 0x3f, 0xff, 0xffff, 0xffffffff, ~0}. See deep dive 5 |
|| ~~Dispatch-floor probes~~ | all ANE paths | **Measured: chained wins.** Unchained copy 131.6 us wall (28 us call + ~104 us completion roundtrip) vs 102 us/eval chained — signals pipeline past the completion cost. `evaluateRealTimeWithModel:` + `beginRealTimeTask` runs but identical (143 us). No model-side sync eval exists. See deep dive 5 |
|| **Forward taps via concat outputs** | MEDUSA/PREDRAFT, hidden-state taps | **Measured: wins.** Expert tap of h (512 ch) costs -0.5 us (free); d1 tap (10752 ch, 5.5 MB) costs +38 us — vs +100-133 us for a second eval moving the same bytes. ~95-100 us saved per tapped eval. See deep dive 5 |
|| ~~RMSNorm into MIL~~ | dense split | **Measured: ~+0.1 ms/eval, not worth it.** norm+rotate+scale+quant in MIL costs +98 us (expert) / +100 us (d1) chained at 256 rows vs the GPU-side pack it replaces; fp16 x input also doubles input-surface bytes. See deep dive 5 |

## Constraints (maderix-verified, design around)

- ~119 compiles per process. On-disk compile cache covers restarts; keep
  program count per model small.
- Multi-input requests can fail 0x1d on some paths. Our `bind()` already
  passes multiple surfaces — probe before building multi-surface MoE programs.
- SDPA op ignores `attn_mask`. Never use it causally; compose matmul + add
  (-inf bias) + softmax.
- QoS has no measurable effect. Skip it.
- `weightsBuffer` does not override baked constants. Confirms the
  surfaces-as-inputs design; do not chase constant reload.
- fp16 surface I/O ~37% faster than fp32. Surfaces already fp16/int8.
- Weights are baked at compile, confirmed twice: `weightsBuffer` ignored
  and on-disk blob overwrite + unload/load returns identical output (M5
  `test_weight_reload` FAIL). No hot-swap path exists.
- M3 Pro compiles **only ch=512 convs** (52 channel values tested); M4/M5
  accept flexible channels. Our wide-channel functions likely fail on M3 —
  probe before claiming M3 support. M1 needs per-matrix weight blobs
  (moot: M3+ only).
- Spatial is the throughput dim: sp64->sp2048 doubled M3 Pro to 16.77
  TFLOPS (95% sustained over 5 s, no throttle). Large chunks mandatory;
  small prompts underutilize — calibration already gates this.
- M5 ANE is the same H16 family as M4 — channel-count tuning transfers.
  macOS 26.x broke `compileModelAtURL`; in-memory compile (our route) is
  the fix.
- Eval fixed cost decomposes (deep dives 3+5): ~0.11 ms/eval chained is
  *already optimal*. Unchained evals cost more (131.6 vs 102.0 us on the
  copy program) because each waits on a ~104 us completion roundtrip that
  chained signals pipeline past. The enqueue call itself is only ~26-48 us.
  Their ~0.06 ms figure is a 1-surface conv measured differently; our
  2-surface unchained copy is 131.6 us wall.
- `_ANERequest` signature (macOS 27, from type encodings):
  `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:
  perfStats:procedureIndex:sharedEvents:transactionHandle:` plus a short
  form without shared events. `perfStats:` must be a collection —
  `validate` sends it `count`. A request form that passes validate() but
  the driver cannot service **hangs `evaluateWithModel:` forever** inside
  `doEvaluateDirectWithModel:` (semaphore wait, no timeout): probe new
  argument shapes only in a sacrificial process.

## Likely losers

- ANE decode on >=2B models: decode is bandwidth-bound, weights stream from
  DRAM either way; Metal decode tiles (139-303 tok/s) win. Exception: d1-3B
  where the point is freeing the GPU, not speed.
- Per-expert MoE evals (measured 2026-10-26, M5 Pro): ~0.20 ms fixed cost
  per evaluation, ~0.4 us/row marginal, and a 512-channel expert only gets
  ~6 TFLOPS effective of the W8A8 peak — the GPU grouped pass is ~1.6x
  faster per FLOP. ANE needs thousands of output channels per eval, as the
  dense split's 8704-channel functions have; a MoE expert has 512.
- d1-3B chained full-forward (projected 2026-10-26, M5 Pro): a 30-eval
  forward at judgment sizes costs ~1.9-2.0x the GPU's latency (65-680 ms vs
  33-368 ms). The ~0.11 ms dispatch floor alone adds ~7 ms over 60+ evals.
  Only a GPU-offload play, and a weak one — the GPU finishes a d1 judgment
  in tens of ms regardless. Available anyway as an opt-in:
  `RICHENGINE_ANE_D1=<share>` gives the dense split to the ANE on d1
  decision models whatever calibration would say; the split log line warns
  it was measured slower and serves only to free the GPU.
- In-memory-MIL draft predictor (measured 2026-10-26, M5 Pro): replacing
  the CoreML predict() path with a hand-rolled MIL program over surfaces is
  ~1.25x SLOWER per call — the enqueue+event+report round-trip costs more
  than CoreML's internal dispatch saves. See deep dive 4.
- Anything training-related. Irrelevant.

---

## Deep dive 1: padded-channel split (Granite-3B, Gemma4)

### What blocks them today

`AneFfn::unsupported` (`runtime/ops/AneFfn.cpp`) imposes:

| Check | Rule | Granite-3B | Granite-8B | Gemma4-26B |
| --- | --- | --- | --- | --- |
| `inputSegment(hidden)` | hidden divisible by 512 | 2560 -> 512 OK | 4096 -> 2048 OK | 2816 fails |
| `hidden % kRotateGroup`, `/<=8` | kRotateGroup = 128x(256/32) = 1024 | 2560 % 1024 = 512 FAIL | 4096 OK (4 blocks) | 2816 % 1024 fails |
| `intermediate % kChannelUnit` | kChannelUnit = 512 | 8192 OK | 12800 OK | shared 2112 fails, expert 704 fails |

Granite-3B and Gemma4 fail the rotation-grid constraint. **Granite-8B passes
every static check** — README's "neither hidden size fits" is stale or a
no-gain calibration outcome. First experiment costs nothing: force
`--ane-ffn-share 0.4` on the 8B and read the verify/calibration result before
building anything.

### The padding trick

Round hidden up to the next 1024 multiple inside the split only:

- Granite-3B: 2560 -> 3072 (+20% ANE-side compute)
- Gemma4: 2816 -> 3072 (+9%)

Why it is safe — the math, not just the plumbing:

1. Activation rotation is per-128-block (`ANE_FFN_INPUT_BLOCK`). Pad blocks
   are all-zero rows; a rotated zero block stays zero. Nothing leaks into
   real channels. `ane_ffn_rotate` gets a `padFrom` parameter, or the pack
   kernel writes int8 0 for channels >= realHidden.
2. Gate/up weight surfaces are `[ane_channels x inputSegment]` int8, staged
   per layer by `ane_ffn_weights`. Pad columns get int8 zeros — zero-fill
   once per surface at init (kernel only writes real columns) or extend its
   bounds. GEMM over pad columns contributes exactly 0.
3. Token scale (`tx`) is an abs-max over the rotated row; zeros never win.
   Down projection untouched (its input is intermediate, not hidden).
4. The split boundary is already GPU=leading / ANE=trailing channels; the
   pad lands at the ANE's trailing edge, outside every GPU view.
5. `verify()` bounds the result against GPU-alone as usual — padding cannot
   slip past it since pad columns produce literal zeros.

Same mechanism rounds `intermediate` up to a 512 multiple when needed
(pad channels: gate/up zero rows -> SiLU(0)x0 = 0; down zero input
columns): Gemma shared expert 2112 -> 2560, Gemma routed expert 704 -> 1024,
any future model with awkward dims.

### Implementation sketch

- `Shape` gains `realHidden`/`realIntermediate` beside the padded values;
  `shapeOf` rounds up, `unsupported` accepts `hidden <= 8192` with
  `paddedHidden % 1024 == 0` and pads intermediate to 512.
- `Memory.inputs` surfaces sized at padded `inputSegment`; weight surfaces
  at padded width.
- Kernel touch points: `ane_ffn_pack` (zero channel >= realHidden),
  `ane_ffn_weights` (zero column >= realHidden, or one-time memset),
  `ane_ffn_rotate` (grid over padded blocks, early-out or zero-fill).
- Flag: `RICHENGINE_ANE_PAD=1` experiment; `unsupported()` reports
  "...padded to N" so the outcome line shows the waste.
- Cost bound: pad waste = (padded/real - 1) of ANE FLOPs, all on the ANE's
  otherwise-idle share. Memory surfaces grow by the same fraction.

### Order of work

1. Granite-8B: forced-share run today. If `no_gain`, padding is moot for
   speed but still relevant for KV headroom (split buffers come out of the
   KV cache — a losing split only costs context).
2. Granite-3B: hidden 2560->3072.
3. Gemma4: hidden 2816->3072 + intermediate pads, GeGLU instead of SiLU
   (compose gelu in MIL, or add a `gelu` op; verify bounds either way).

---

## Deep dive 2: MoE routed-expert FFN on ANE — measured, loses

`dev/benchmarks/ane_moe_benchmark.mm` (`make benchmark-ane-moe`) A/Bs one 35B
routed expert (hidden 2048, intermediate 512, affine Q4) through the dense
split's machinery — `ane_ffn_*` kernels, one multi-procedure MIL program,
`ane::Handoff`, double-buffered weight sets — against the GPU's own
`addPrefillSwiGlu`. Correctness was fine (int8 error ~2.0%, under the
split's 10% bound); performance was not.

M5 Pro, 2026-10-26:

| rows | gpu_ms | ane_ms | side_ms (rotate+pack+join) | stage_ms | err_% |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.781 | 0.234 | 0.145 | 0.166 | 2.04 |
| 256 | 0.262 | 0.273 | 0.156 | 0.167 | 2.01 |
| 512 | 0.306 | 0.427 | 0.179 | 0.169 | 2.04 |
| 1024 | 0.474 | 0.660 | 0.240 | 0.164 | 2.03 |
| 2048 | 0.749 | 1.026 | 0.391 | 0.166 | 2.04 |

Chained evals plateau at ~4800-5000 evals/s: **~0.20 ms fixed cost per
evaluation** regardless of rows, then ~0.4 us/row. The design above assumed
100-200 us/eval overlapping GPU work; the floor is real, and it is only
part of the loss:

- A whole expert at 2048 rows costs 1.03 ms on the ANE (~6 TFLOPS
  effective). The GPU's grouped pass does the same FLOPs in ~0.62 ms —
  **the GPU is ~1.6x faster per FLOP at this shape** because a 512-channel
  expert is 17x less compute per eval than the dense split's 8704-channel
  functions, and W8A8 needs bigger matrices to reach its 35-TOPS peak.
- GPU-side overhead per eval (rotate+pack+join+stage) is ~0.31 ms — taking
  an expert off the GPU costs more GPU time than the expert's grouped
  share saves.
- Projected layer time at 2048 rows under correlated routing: every
  hot-K choice is a regression (0.27-0.75x of the 6.75 ms GPU-alone layer).
- The shared expert — the only single-eval-per-layer variant — loses too:
  1.03 ms ANE vs ~0.62 ms GPU.

The one structural fix, packing several experts into one eval, needs
block-diagonal weights (ANE matmuls can't gather per-row); masking costs
K x redundant compute and loses harder. Per-expert MoE offload is dead at
this shape — the harness stays as the smallest complete
`Program`+`Handoff`+staging pipeline for future per-shape A/Bs.

### (Archived analysis — what the probe disproved)

### Why this was the candidate

`model::aneFfnLayers` collects only `DenseFfnWeights`; every MoE target
returns `dense=false` ("the target has no dense FFN layers") and the whole
prefill FFN runs on GPU. On a 14K prompt the 35B-A3B spends most of its
layer time in grouped expert GEMMs — exactly the matmul-shaped work the ANE
does at 18.6 (fp16) / 35.1 (int8) TOPS while the GPU runs its own share.

### Model shapes vs the existing kernels

| Model | hidden | experts | topK | expert int | shared | Shape fit |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 2048 | 256 | 8 | 512 | yes | Clean: hidden 2 blocks, int = 1 rotation block |
| Ornith-1.5-35B-A3B | 2048 | 256 | 8 | 512 | yes | Clean (same) |
| MiniCPM-V-35B-A3B | 2048 | 256 | 8 | 512 | yes | Clean (same) |
| LFM2.5-24B-A2B | 2048 | 64 | 4 | 1536 | no | Clean: int = 3 blocks |
| Gemma4-26B-A4B | 2816 | 128 | 8 | 704 | dense 2112 | Needs both pads + GeGLU |

### Design: `AneMoe` beside `AneFfn`

Per MoE layer, per prefill chunk:

1. **Route on GPU (unchanged).** `moe_route_scores`/`moe_group_routes` are
   tiny; they already sort rows into per-expert tiles (`groupedInput`,
   `routeRows`, `tileDescriptors`).
2. **Pick ANE experts.** Experts whose grouped rows cross a threshold —
   v1: all experts over `minRows`, capped at `K` surfaces staged; measure
   coverage vs K. At 14K rows x topK=8, ~440 rows/expert average across 256
   experts — heavily skewed, so top-32 typically covers a large share of
   routed rows. The shared expert (Qwen) sees every row — treat as expert
   `experts` at max bucket.
3. **Stage weights.** `ane_ffn_weights` variants read expert e's planes:
   GGUF `BlockExpertProjection.routed` is one segment of `experts*N` rows —
   expert e's tile starts at `e*N/256`, an offset plane view. Affine
   `ExpertProjection.packed` + `expertStrideBytes` gives the same view.
   int8 staging cost per expert ~3.1 MB write + ~1.7 MB Q4 read; K=32 costs
   ~155 MB traffic/layer (~0.4 ms) if done in-layer. v2 hides it: predict
   hot experts from the previous chunk's routing (sticky within a document)
   and stage one layer ahead like the dense split.
4. **Eval per expert.** One MIL function per row-bucket (128..2048), input
   surfaces = that expert's grouped rows packed int8 (the pack kernel reads
   `groupedInput` instead of `normalized`), weights = its staged surfaces.
   `Handoff` chains them back-to-back on the shared event, concurrent with
   the GPU's own expert tiles. Est. ~100-200 us/eval; 32 experts x 40
   layers = 1280 evals ~150-260 ms ANE-busy per 14K chunk, overlapped.
5. **Join.** ANE writes partial down per expert row-group; a combine kernel
   scatter-adds `routing_weight * partial` into the residual — extend
   `moe_combine` to take a second source. GPU-dispatched expert tiles simply
   skip the ANE-taken experts (filter tile descriptors).
6. **Correctness.** Any expert not staged runs the GPU grouped path — a
   staging miss costs latency, never correctness. `verify()` bounds the
   ANE share against the GPU-alone MoE per layer like the dense split does;
   breaker retires on loss.

### What it is NOT

- Not a grouped/block-diagonal GEMM on ANE — ANE convs can't gather
  per-token weights. Per-expert evals are the unit.
- Not decode: expert GEMMs at B1-B4 lanes are bandwidth-bound; ANE adds
  nothing the GPU isn't already doing.
- Not all-experts staging: 256 experts x ~3.1 MB int8 = ~800 MB/layer —
  defeats the point. Hot-subset only.

### Expected value — wrong, and why

The estimate assumed ~100-200 us/eval and per-eval work large enough to
reach W8A8 throughput. Both failed: the eval floor is ~200 us fixed
alone, and a 512-channel expert matmul never gets near the peak. The dense
split wins because it amortizes one eval over 8704 output channels of one
layer; MoE would need the inverse — many small evals per layer.

### Risks — resolution

- Multi-input evals work: ~11 surfaces per eval, 64+ chained in a command,
  zero 0x1d failures on this path (macOS 27, M5 Pro).
- Eval dispatch overhead measured: ~0.20 ms/eval floor, 4800-5000 evals/s
  ceiling — this alone kills 1000-eval/chunk designs (~200 ms/layer).
- Expert-hot prediction is moot: staging cost was not the bottleneck.

---

## Deep dive 3: eval-floor anatomy + d1-3B chained forward — both measured

`dev/benchmarks/ane_forward_benchmark.mm` (`make benchmark-ane-forward`)
runs two probes over one `ane::Handoff`:

### Floor anatomy — what the ~0.20 ms/eval is made of

Six programs of rising input count and compute, each chained 32 times in
one command with only shared-event signals between evals (plus one
`residency_kick` dispatch — a Metal command needs one). Rows 256, M5 Pro:

| program | us/eval | evals/s | delta vs previous |
| --- | ---: | ---: | --- |
| copy (fp16 pass-through, 1 input) | 113.5 | 8814 | — irreducible floor |
| mm_i8 (one int8 matmul, 2 in) | 146.0 | 6848 | +33: second surface + dequant + matmul |
| mm_fp (same matmul, x fp16, 2 in) | 144.1 | 6939 | int8-x dequant ~free |
| gate (2 matmuls + 2 scales, 6 in) | 225.1 | 4443 | +81: 4 more surfaces + second matmul |
| expert SwiGLU (512 ch W8A8, 11 in) | 583.2 | 1715 | +358: rotation conv + requantize + 5 surfaces |
| d1 FFN (10752 ch W8A8, 15 in) | 3163 | 316 | compute dominates |

Decomposition: **~0.11 ms irreducible** per evaluation (dispatch + shared-
event roundtrip), **~10-15 us per input surface**, the rest is compute
slope. The MoE bench's "~0.20 ms floor" = 113 us dispatch + ~11 surfaces.
Consequences:

- Any per-layer chained design pays >= 0.11 ms x evals in pure dispatch.
- Input count is second-order (10-15 us/surface), not the killer.
- Compute must be large per eval — the dense split's 8704 channels is the
  regime; 512-channel work never escapes the floor.
- The eval-rate ceiling seen before (4800-5000/s) is a chained-eval
  property; single-copy floor ~8800/s.

### d1-3B chained full-forward — loses on latency

d1-3B is the LFM2-family decision model: 30 layers (hidden 2048, FFN
intermediate 10752 — 22 conv + 8 attention layers), one forward per
judgment, no decode, no KV. The A/B models the forward as the chain it
would be: 30 real FFN evals (the dense split's W8A8 SwiGLU at the d1
shape), 22 conv-in_proj evals ([6144x2048] matmul), 8 attention evals
(QKV [3072x2048] + O [2048x2048] + a T^2 term scaled by the projections'
measured rate) — every eval a measured chain-median, GPU side through the
real `addPrefill`/`addPrefillSwiGlu` kernels at the same shapes.

M5 Pro, ms per eval and projected 30-layer forward:

| rows | ffn_ane | conv_ane | attn_ane | fwd_ane | ffn_gpu | conv_gpu | attn_gpu | fwd_gpu | ane/gpu |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.781 | 0.351 | 0.415 | 64.6 | 0.803 | 0.245 | 0.406 | 32.9 | 1.96x |
| 128 | 2.563 | 0.423 | 0.895 | 94.1 | 2.034 | 0.420 | 0.603 | 75.6 | 1.25x |
| 256 | 3.163 | 0.608 | 0.632 | 114.3 | 1.758 | 0.432 | 0.532 | 67.4 | 1.70x |
| 512 | 5.101 | 1.065 | 1.027 | 188.0 | 2.689 | 0.682 | 0.747 | 104.0 | 1.81x |
| 1024 | 9.471 | 1.854 | 1.692 | 349.3 | 4.919 | 1.090 | 1.120 | 187.7 | 1.86x |
| 2048 | 18.118 | 3.430 | 3.003 | 681.5 | 9.532 | 1.949 | 1.899 | 368.3 | 1.85x |

The FFN eval is the whole story: 10752 channels x 2048 int8 (~66 MB/layer)
is DRAM-bound both ways but the GPU reads faster; ANE lands ~1.9x the GPU's
per-layer time at every size. Chaining buys nothing — the dispatch floor
is per-eval, so 60 evals of floor (~7 ms) is already 20% of a 64-row GPU
forward. `_ANEChainingRequest` loopback would remove the per-eval event
roundtrip but not the weight streaming that dominates.

Verdict: do not build. The only win would have been freeing the GPU, and
the GPU does a whole d1 judgment in 33-368 ms — the offload saves little
and doubles latency. Revisit only if judgments ever saturate the GPU.

For a future revisit: `_ANEChainingRequest` exposes
`loopbackInputSymbolIndex`/`loopbackOutputSymbolIndex` (feed output back
as input), `memoryPoolId` (shared pool across chained ops),
`fwEnqueueDelay` (firmware enqueue timing), `outputSets`, and
`signalEvents`; setup calls are `prepareChainingWithModel:` and
`enqueueSetsWithModel:`, with `buffersReadyWithModel:` for async
input-ready. Loopback could collapse the per-eval event roundtrip — the
~0.11 ms piece — but weight streaming still dominates at these shapes.

---

## Deep dive 4: predraft transport — CoreML predict() vs in-memory MIL

`dev/benchmarks/ane_predraft_benchmark.mm` (`make benchmark-ane-predraft`,
model via `ANE_PREDRAFT_MODEL`) times both sides of the
`RICHENGINE_ANE_PREDRAFT` contract (4 lanes, `hidden [4,8,2048]` fp16,
`proposals [4,7]`): the production `AnePredictor::predict` — MLMultiArray
allocation, memcpy in, provider dict, `predictionFromFeatures:`, memcpy out
— against a same-shaped in-memory MIL program evaluated over preallocated
surfaces through `ane::Program` + `ane::Handoff`. The model is a stand-in
NeuralNetwork spec (two matmuls, gelu, ~34 MFLOP) small enough that
transport dominates.

M5 Pro, 50 rounds, median ms/call:

| path | ms/call |
|---|---:|
| A: CoreML predict() | 0.248 |
| B: MIL surfaces + eval | 0.31 |
| — surface writes (128 KB hidden + weights) | 0.002 |
| — eval round-trip (submit, event, report, wake) | 0.30 |

The transport hypothesis was wrong: IOSurface writes cost ~2 us — transport
was never the tax. What costs is the eval round-trip — a Metal command
submission, the shared-event handshake and the service's completion report
— which CoreML's single-process dispatch avoids. B/A ~1.25x consistently
(0.31-0.33 ms vs 0.25 ms).

Verdict: keep the CoreML `AnePredictor` path. In-memory MIL would buy no
.mlmodelc artifact and no CoreML compile budget, at a measurable latency
cost. The finding generalizes: for single-shot evals (not chained inside a
bigger command), the Handoff round-trip ~0.3 ms is the price of admission —
a predraft kicked once per step pays it in full.

Compiler note (informs any future MIL authoring): a `const()` BLOBFILE
tensor as a matmul operand makes `_ANECompiler` reject the program with
bare `InvalidMILProgram`; const() is accepted as a conv weight (the dense
split's rotation conv). Weight-surfaces-as-inputs — the `AneFfn` pattern —
is the working form. `transpose_y = bool(true)` on matmul is likewise
rejected; bake weights transposed instead.

## Deep dive 5: dispatch paths, perfStats, and norm-in-MIL — all measured

`dev/benchmarks/ane_forward_benchmark.mm`, M5 Pro, macOS 27, medians of 3
rounds, K=16 unchained / K=32 chained, rows 256.

### Dispatch variants — chained stays

| program | path | wall us | call us |
|---|---|---|---|
| copy (fp16 pass-through) | unchained | 131.6 | 28.0 |
| copy (fp16 pass-through) | client-rt | 143.5 | 30.1 |
| copy — chained, for reference | | 102.0/eval | — |
| mm_i8 | unchained | 154.3 | 29.5 |
| mm_i8 | client-rt | 146.2 | 26.0 |
| gate | unchained | 183.5 | 34.9 |
| expert SwiGLU | unchained | 348.2 | 40.6 |
| expert SwiGLU | client-rt | 337.9 | 38.4 |
| d1 FFN | unchained | 3213.1 | 47.0 |
| d1 FFN | client-rt | 3228.8 | 51.8 |

- The enqueue (`evaluateWithModel:`) call is only ~26-48 us. Everything
  else is the async completion roundtrip (~104 us on the copy program).
  Chained evals overlap that path through shared events, so the chained
  102 us/eval already beats unchained. Their ~0.06 ms unchained figure is
  a 1-surface conv, likely measured without the same completion wait.
- `evaluateRealTimeWithModel:options:request:error:` exists
  (`B48@0:8@16@24@32^@40`), runs after `beginRealTimeTask`, and lands
  within noise of the normal client path. No model-side synchronous eval
  (`evaluateWithQoS:options:request:error:`) exists on `_ANEModel`.
- Verdict: no production change. Shared-event chaining remains optimal.

### perfStatsMask + _ANEPerformanceStats — dead on macOS 27

- `setPerfStatsMask:` exists (`v20@0:8I16`, 32-bit mask);
  `+statsWithHardwareExecutionNS:` exists on `_ANEPerformanceStats`.
- `_ANERequest perfStats:` must be a collection — `validate` sends it
  `count` (crash on the bare stats object). Array/object forms all
  evaluate without error for masks {1, 0x3f, 0xff, 0xffff, 0xffffffff,
  ~0} — but `hwExecutionTime` is never populated and the array stays
  empty. Whatever shape the driver wants, it isn't this one on this OS.
- Hazard found probing this: a perfStats form that passes `validate()`
  but the driver cannot service **hangs `evaluateWithModel:` forever**
  inside `doEvaluateDirectWithModel:` — semaphore wait, no timeout.
  Future request-shape probes belong in a sacrificial process.

### RMSNorm into MIL — ~+0.1 ms/eval, regression

Two variants beside the existing int8-input programs: fp16 x input +
in-MIL `square → reduce_sum → mul(1/ch) → pow(-0.5) → mul(x) →
scale → rotate → scale-shift → quantize → round → cast-to-int8`
feeding the same matmul chain (multi-record weight blob for the norm
constants). Chained medians:

| program | baseline | +MILnorm | delta |
|---|---|---|---|
| expert SwiGLU (512 ch, 11→7 in) | 285.4 us | 383.0 us | **+97.6 us** |
| d1 FFN (10752 ch, 15→9 in) | 3165.1 us | 3265.2 us | **+100.1 us** |

Unchained shows the same +69-118 us. The in-MIL chain costs ~0.1 ms at
256 rows — roughly what the GPU-side norm+rotate+pack it would remove
costs (0.15-0.39 ms, deep dive 2, though that includes join+stage), while
the fp16 x surface doubles input-surface bytes vs int8. Net: break-even
at best, extra compile complexity guaranteed. Do not take; keep
norm+rotate+pack on the GPU.

### Forward taps via concat outputs — wins

`h` concat'd into the FFN's output surface behind `hscale` (output rows
grow by the tap's channels — no extra evaluation, no API change) vs a
second copy eval moving the same bytes. K=32 chained pairs, rows 256:

| program | base us | +tap us | tap cost | [FFN;copy] pair us | 2nd-eval cost | tap saves |
|---|---|---|---|---|---|---|
| expert SwiGLU (tap h: 512 ch, 0.26 MB) | 297.1 | 296.6 | -0.5 | 397.0 | +99.9 | ~100 us/eval |
| d1 FFN (tap h: 10752 ch, 5.5 MB) | 3164.3 | 3202.5 | +38.3 | 3297.1 | +132.8 | ~95 us/eval |

- A second eval always pays the ~100-133 us dispatch+completion floor.
  A concat tap pays only its surface write — free at 512 channels, ~38 us
  at 10752 (the full d1 hidden).
- Where intermediates are needed (MEDUSA/PREDRAFT hidden taps, draft-head
  inputs), tap via concat, never a second eval. Take when a use lands.

## Deep dive 6: CoreAI public framework — researched + probed

Apple's new public API (WWDC26, macOS 27+), purpose-built successor path
vs the private `_ANEClient`. Probed end-to-end on this machine.

### What it is

- `CoreAI.framework` re-exports `CoreAIDelegates`; execution lives in
  `CoreAIRuntime` (Swift-only; no ObjC API beyond `deinit`, C++ interop
  off in the module).
- Model format is `.aimodel` (MLIR bytecode bundle: `main.mlirb` +
  `main.hash` + `metadata.json`). NOT `.mlmodel`/`.mlmodelc` — those
  fail with `corruptedMetadata`/`Missing hash file`.
- Authoring: `pip install coreai-torch` (Python 3.12; no 3.15 wheel),
  `torch.export` → `run_decompositions(coreai_torch.get_decomp_table())`
  → `TorchConverter.add_exported_program()` → `to_coreai()` →
  `prog.save_asset(path.aimodel)`. Works; produced `/tmp/probe.aimodel`
  and `/tmp/predraft.aimodel` (33 MB fp32 MLP).
- AOT: `coreai-build compile` (in MetalToolchain cryptex) emits per-arch
  `.aimodelc` for all 22 archs with `--preferred-compute neural-engine`;
  `coreai-build package` repackages — loads fine.
- Public + documented: apple-docs coreai pages, `apple/coreai-models`
  GitHub repo with export recipes and Swift runtime helpers.

### API shape (verified in swiftinterface + live)

`AIModel(contentsOf:options:)` async → `model.loadFunction(named:)` →
`InferenceFunction` with:
- `run(inputs:states:outputViews:) async` — convenience path.
- `encode(inputs:states:outputViews:to: ComputeStream)` — synchronous
  encoding onto an existing `MTLCommandQueue`; inputs/outputs can be
  raw `MTLBuffer`s via `AsyncValue(unsafeBuffer:scalarType:shape:)` —
  true zero-copy, no MLMultiArray analog.
- `NDArray` also constructs directly on `IOSurface` and `MTLBuffer`;
  `ScalarType` spans int2…int128/float8/float16 — packed quant dtypes
  are first-class.
- `ComputeUnitKind.availableKinds` reported `[cpu, gpu, neuralEngine]`;
  specialization is real (same model: cpu 27 us, gpu 358 us, NE 204 us).

### C access — proven

No C API exists, but a ~60-line Swift shim exporting `@_cdecl`
functions (`coreai_load` / `coreai_eval` / `coreai_wait`, semaphore-
bridged async load) links into a plain C caller and runs `encode()`
on NE-specialized functions: **0.30 ms/call e2e** including per-call
completion wait. So "C option" = thin Swift dylib + C ABI, not direct.

### Latency vs existing paths

| path | ms/call |
|---|---|
| CoreML `AnePredictor::predict()` (predraft) | 0.248 |
| private MIL `ane::Program` + Handoff | 0.31 |
| CoreAI `run()` NE, tiny MLP | 0.214 |
| CoreAI `encode()` NE, tiny MLP | 0.217 |
| CoreAI `run()` NE, predraft-shaped (33 MB) | 0.394 |
| CoreAI `encode()` NE e2e, predraft-shaped | 0.396 |

Same ~0.2-0.4 ms band — the dispatch floor is hardware, not framework.
No measured latency win over CoreML at these shapes.

### Verdict

- Feasible: public API, zero-copy MTLBuffer I/O on our command queue,
  per-arch AOT artifacts, C-reachable via a small Swift shim.
- Not a drop-in: models must be re-authored through `coreai-torch`
  (no MIL ingestion, no `.mlmodel` compat).
- No latency advantage measured; benefit would be removing private-API
  risk and MLMultiArray copies, plus first-class packed dtypes.
- If adopted: only as an opt-in parallel path behind runtime gating,
  authored alongside existing artifacts — not a CoreML/ANE replacement.

### Predraft transport A/B/C — CoreAI loses, no production path

Extended `dev/benchmarks/ane_predraft_benchmark.mm` with a C side through
`dev/benchmarks/coreai_shim.swift` (@_cdecl, ~100 lines, links as a
dylib). Same contract both sides: anchor/position/retained [4] +
hidden [4,8,2048] fp16 -> proposals [4,7], 34 MFLOP, NE-specialized.
The .aimodel was authored via `coreai-torch` (torch.export ->
decompositions -> TorchConverter -> save_asset); ANE_PREDRAFT_AIMODEL
names it for `make benchmark-ane-predraft`.

| path | ms/call | vs A |
|---|---|---|
| A: CoreML predict() (MLMultiArray) | 0.244 | 1.0x |
| B: private MIL surfaces + eval | 0.315 | 1.29x |
| C1: CoreAI run() over preallocated NDArrays | 0.552 | 2.26x |
| C2: CoreAI encode() MTLBuffer + ComputeStream | 0.514 | 2.10x |
| CoreAI run(), pure Swift (no C shim) | 0.343 | 1.40x |

Unit check on the same asset: cpu 0.040, gpu 0.409, NE 0.343 ms/call —
the NE specialization is real; the gap is transport/dispatch overhead,
not fallback. The C shim adds ~0.2 ms/call over pure Swift (semaphore
bridge + Task per call + dict rebuild); even without it CoreAI loses.

Verdict: keep `AnePredictor` on CoreML. CoreAI's zero-copy encode() is
theoretically the cleanest transport we have measured, and it still
costs more than CoreML's MLMultiArray path at this workload. No CoreAI
production path was added; the shim + benchmark stay for re-checking on
future OS releases and bigger models.

### CoreAI quantized FFN — W8A16 reaches private-path parity

d1 FFN shape (rows 256, 2048->10752 SwiGLU ->2048) authored via coreai-torch,
NE-specialized, timed through the same run()/encode() probes:

| weights | asset | run() | encode() e2e |
|---|---|---|---|
| fp16 | 129 MB | 6.02 | 6.09 |
| int8 (constexpr_blockwise_shift_scale) | 64 MB | **3.03** | **3.09** |
| uint4 palettized (16-entry LUT) | 33 MB | 15.1 | 15.2 |
| int8 naive dequant (q.half()*scale) | 64 MB | 16.8 | 16.7 |
| private MIL W8A8 (ane::Program) | 66 MB | — | 3.16 |

- Weight-side int8 works only through the compression ops
  (`WeightDequantizeModule` -> `constexpr_blockwise_shift_scale`); a plain
  `int8.half()*scale` dequant gets `mps.cast` rejected by ANE validation
  and falls back (~5.5x slower than fp16).
- W8A16 = 3.03 ms is DRAM-bound parity with the private W8A8 path — the
  first workload where CoreAI matches `_ANEClient`.
- uint4 palettization packs correctly (33 MB) but the lut expansion costs
  ~5x; do not use.
- A8 int8 activations are unreachable: authoring `batch_matmul` rejects
  si8 operands. Moot — W8A16 already saturates.
- Caveat vs AneFfn: CoreAI bakes weights into the asset; the
  surfaces-as-inputs staging pattern is not exposed. A CoreAI FFN variant
  would need weight-bound inputs or per-layer asset specialization —
  unprobed.

### Weights-as-inputs works — encode() beats private MIL at FFN shape

Exporting the FFN with int8 weight tensors + fp16 scales declared as
function *inputs* (runtime `dequantize` op, 1-D scale at axis 0) produces
a 5 KB `.aimodel` — weights are NOT baked. Descriptors confirm runtime
inputs: `gate_q [10752,2048] int8`, `gate_s [10752] fp16`, all
`StorageKind.ioSurface`.

encode() e2e at rows 256, alternating between two weight-buffer sets
(simulating per-layer restaging):

| path | ms/eval |
|---|---|
| **CoreAI encode(), dynamic int8 weights** | **2.27** |
| CoreAI baked W8 | 3.03 |
| private MIL W8A8 | 3.16 |
| CoreAI fp16 | 6.02 |

Dynamic weights are FASTER than baked — likely better int8 streaming of
input-bound tensors vs the dequantized-constant path.

### Correctness + layout checks — all pass

Correctness vs manual dequant+linear reference (deterministic weights,
fp16-representable range, 64-row activation):

| path | result |
|---|---|
| Python runtime, NumPy inputs | max abs err 7.6e-5 (fp16 rounding) |
| Swift `encode()` + `MTLBuffer`, vs CPU reference | max err 2.2e-4 over probed cols |

Layout — `MTLBuffer` bindings satisfy every descriptor constraint
(`ioSurface` storage kind is a request, not a hard requirement; shared-mode
Metal buffers are accepted for all inputs/outputs):

- Arbitrary row strides accepted: `strides: [2176, 1]` padded x encodes and
  produces bit-identical output to packed `[2048, 1]`.
- Weight buffers: strides 2048 / 2112 / 2050 (unaligned) all accepted, same
  output — the runtime packs internally where needed.
- 1-D fp16 scale inputs match the per-output-channel contract of
  `runtime/ane` (axis 0 = output rows of each weight).

Dynamic rows — a single asset exported with `torch.export.Dim` covers every
row bucket (descriptor `x [-1,2048]` / `y [-1,2048]`); first call per shape
pays a one-time specialization (~30-100 ms), steady state:

| rows | ms/eval (min/med) |
|---|---|
| 64 | 1.87 / 2.70 |
| 256 | 2.50 / 2.87 |
| 1024 | 6.09 / 6.18 |
| 2048 | 11.53 / 11.77 |

R=256 median is ~0.3 ms slower than the fixed-shape asset (2.27) — per-shape
recompile noise. A fixed-256 asset is still fastest; dynamic covers the rest.

`coreai-build compile --expect-frequent-reshapes` removes the dynamic-shape
penalty entirely (per-arch `.aimodelc`, h17s on M5 Pro):

| rows | first call | min/med (AOT) | min/med (JIT .aimodel) |
|---|---|---|---|
| 64 | 266.9 (cold arch load) | 1.74 / 1.85 | 1.87 / 2.70 |
| 256 | 6.06 | **2.26 / 2.33** | 2.50 / 2.87 |
| 1024 | 9.80 | 5.40 / 5.45 | 6.09 / 6.18 |
| 2048 | 15.35 | 9.61 / 9.74 | 11.53 / 11.77 |

One AOT-compiled dynamic asset now MATCHES the fixed-shape benchmark
(2.26 vs 2.27 ms at R=256) while covering every row bucket; per-shape
first-call drops to 6-15 ms (amortizable at load or on first use).

Queue interleaving verified: a Metal `fill` command buffer writing x,
followed by `f.encode()` on a `ComputeStream` wrapping the same
`MTLCommandQueue`, produces correct output; changing the Metal-written x
changes CoreAI's y accordingly. `ComputeStream` is just queue ordering —
it composes with our existing dispatches with no extra sync.

### Production path — `RICHENGINE_COREAI_FFN` (opt-in)

`runtime/ops/CoreAiFfn.{hpp,cpp}` implements the same `FfnSplit` contract
as `ops::AneFfn`, so call sites select it without knowing which engine
serves the split. Selection: `RICHENGINE_COREAI_FFN=<share>`; the share
moves that fraction of intermediate channels (in whole
`max(QUANT_TILE_ROWS, ANE_FFN_INTERMEDIATE_BLOCK)` units) to CoreAI, the
rest stays on the GPU exactly as the private split's.

Moving parts:

- `runtime/ops/CoreAiShim.swift` builds `librichengine_coreai.dylib`
  beside the binary (dev/native.mk), a C ABI (`ops/CoreAiBridge.h`) over
  the Swift-only framework. The engine `dlopen`s it; where it or the
  `.aimodel` is absent the split reports unavailable and the GPU runs
  alone.
- `dev/tools/gen_coreai_ffn.py` authors one dynamic-weights `.aimodel`
  per model (int8 gate/up/down + fp16 scales as inputs, dynamic rows).
  Function names `ffn_<hidden>_<ane>` cover the shares a run can pick;
  `RICHENGINE_COREAI_ASSET` points at it.
- Staging reuses the `ane_ffn` Metal kernels in no-rotate variants
  (`kCoreAiFfnWeights`, `kCoreAiFfnRowScale`, cast + join), two weight
  sets ping-ponged per layer exactly like the private path's surfaces.

Synchronisation lessons (all measured, all in the shim):

- `encodeWaitForEvent` does NOT gate the evaluation: stream work is
  encoded when `start` runs — before the caller's GPU command buffer is
  even committed — so an eval can read weights the GPU has not staged
  yet (a chained two-layer verify read layer 2's weights from layer 1's
  set). The release must be CPU-side: `MTLSharedEvent.notify(listener,
  atValue:)` fires when the GPU's signal lands, and the eval starts in
  that callback.
- Completion is `AsyncValue.ndArray` — awaiting the returned y's array
  is the only reliable end-of-work signal; `currentWorkCompleted` and
  signal-command-buffer tricks either hang or fire early. The copied
  rows go into the caller's y buffer, then `signaledValue` goes up.
- Bound `MTLBuffer`s get hazard-tracked against the caller's still-open
  command buffer and deadlock; bind `bytesNoCopy` aliases with
  `hazardTrackingModeUntracked` — the shared event is the ordering.
- The x/y buffers are shared across jobs; a `DispatchSemaphore` (not a
  lock — the task resumes on arbitrary threads) keeps one eval at a
  time, and the first encode specializes the function (~2 s) so `open`
  burns a warm-up job before any bounded job runs.

Verified in-engine (LFM2 2.6B, share 0.52): two-layer chained split at
512 and 2048 rows, **1.86–1.87%** relative RMS error vs GPU alone —
under the 10% bound the private split also verifies against. Fallback
is unchanged: no flag, missing shim, missing asset, a failed verify or
a blown handoff bound all leave the whole FFN on the GPU.

### In-engine A/B — CoreAI loses where private runs, unlocks where it can't

2048-token prefill, same prompt, `RICH_STATUS` timings (M5 Pro).
Standalone parity (2.27 vs 3.16 ms) was a rows-256 artifact: engine
chunks are 512-2048 rows, where the CoreAI eval is DRAM-bound at
~5.4 ms/1024 rows and ~9.6 ms/2048 rows on the AOT asset — the shim
adds ~0; the eval itself costs more than the GPU's share it replaces.

| model | GPU alone | private split | CoreAI JIT | CoreAI AOT `.aimodelc` |
|---|---|---|---|---|
| LFM2.5-2.6B (h2048 i10752) | ~384 ms | **~323 ms** (cal. share 0.29, 5.2 ms/eval) | ~455 ms @0.52 (10.7 ms/eval) | ~408 ms @0.52 (9.5 ms/eval) |
| Ornith-1.5-9B (h4096 i12288) | ~1122 ms | **~959 ms** (cal. share 0.375, 14.3 ms/layer) | ~1181 ms @0.5 | ~1225 ms @0.5 |
| Granite-4.2-3B (h2560 i8192) | **~565 ms** | rejects (hidden %1024 fails) | ~676 ms @0.5 | ~598 ms @0.5 (1.6% err, 80 evals) |
| MiniCPM5-2B (h2048 i6144) | ~392 ms | **~335 ms** (cal. share 0.333, 3.5 ms/eval) | ~446 ms @0.333 (6.0 ms/eval) | ~412 ms @0.333 (5.6 ms/eval) |
| MiniCPM5-2B-MLX (affine, same dims) | ~373 ms | **~322 ms** (cal. share 0.333) | ~407 ms @0.333 (5.6 ms/eval) | ~398 ms @0.333 (5.6 ms/eval) |
| Qwen3.8-27B-4bit (mlx-community, h5120 i17408) | ~5027-5420 ms | **~3284-3472 ms** (cal. share 0.441, 25.3 ms/eval, 1.6% err) | ~5418 ms @0.441 (54.3 ms/eval, 1.4% err) | blocked: `memory_audit` runtime_reserve_exceeded (AOT delegates exceed the 768 MiB reserve on the 27B) |
| Granite-4.2-8B | registered, weights not fetched — untested | — | — | — |

Split-efficiency probe: the split layer's overhead above its GPU part is
~0.6 ms (staging + rotate + pack + join + event ops); the ANE eval is
fully hidden and both engines stream int8 weights at ~3.5 GB/s effective,
which is what parks the share at the calibration crossing. Fusing the
rotate's fp16 intermediate into an int8 one (pack is then a pure int8
transpose) cuts ~0.25 ms/layer of staging traffic — measured overhead
~0.35 ms/layer on a same-run recalibration — but the calibrated share
moved 8→6 units and total prefill stayed flat within noise. Commit-split
A/B (`RICHENGINE_MTL4_OFF`) measured ~0.

The `decode_warmup` blocker was a storage-lane bound, not an ANE path
issue: `encodeDraftBatchGraph` sized packed draft tensors from the
target's `decodeStorageLanes` alone, whose projection list is target-only.
An affine target (MiniCPM5-2B, Qwen3.8-27B) reports no padding, so three
lanes bound 24 rows while the draft's own AF4-block fused QKV staged the
32-row tile. `DecodeArena::draftStorageLanes` now bounds the draft's
shapes at its query stride and the caller takes the maximum; GGUF targets
masked it because their own projections already forced the bound.

CoreAI-only support required one fix: `RuntimeResources` gates
`model.unsupported`/`unavailable` to the private checks and evaluates
`CoreAiFfn::unsupported()` separately, so a `RICHENGINE_COREAI_FFN`
given can serve shapes the rotation path rejects (Granite-3B engages:
`state: split`, 1.6-2.4% verify error, 0.1-0.8 s setup).

**Verdict: ship as opt-in compatibility, not default.** Private ANE
wins every model it supports (1.7-2.1x faster per eval at real rows);
the win scales with FFN width — the 27B's split is 35% faster end to
end, the biggest measured. CoreAI's value is shapes the private path
structurally rejects — Granite-3B runs ANE-assisted for the first
time, ~6% slower than GPU alone but freeing GPU bandwidth during
prefill.

AOT delivery: `coreai-build compile` (in the MetalToolchain cryptex,
`.../Metal.xctoolchain/usr/bin/coreai-build`) with
`--preferred-compute neural-engine --architecture h17s
--expect-frequent-reshapes` emits `<name>.h17s.aimodelc`;
`RICHENGINE_COREAI_ASSET` accepts it directly. Setup drops to
0.1-0.8 s and per-eval time lands at the standalone AOT numbers —
still slower than private at engine row counts. Caveat: on the 27B
the `.aimodelc` delegates' driver-side allocations exceed the
256+512 MiB pipeline/runtime reserve and `memory_audit` refuses
bootstrap; JIT assets (which compile lazily per function) fit.

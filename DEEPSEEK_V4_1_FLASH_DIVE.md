# Metal 4 — Which Improvements Actually Pay

Ranking of the findings in the Metal 4 audit by expected end-to-end gain,
with the measurement that bounds each one.

Device: Apple M5 Pro, macOS 27.0.1, Metal 4, toolchain `-std=metal4.1`.
No models are installed on this machine, so every number below comes from
`dev/benchmarks/mtl4_benchmark.mm` and standalone probes against the same
driver — not from tok/s on a real model.

> Update: models were installed after all; the end-to-end A/B in the last
> section was run on Ornith-1.5-9B and MiniCPM5-2B. The estimates in this
> document were too optimistic about host work — the depth-2 ring hides it.

## Short answer

> Superseded by the A/B in the last section: items 1–3 were implemented and
> measured, and none of them moves end-to-end wall time on this hardware.
> The ranking below is the pre-implementation estimate, kept for the record;
> read "A/B results" for what actually happened.

1. **Span replay under the decode ring** (`#1`) — best gain per unit of work.
   Guaranteed 2× on host encode for span-covered dispatches, and it removes
   the barrier-analysis pass for those dispatches too.
2. **Bake span barriers from the dependency analysis** (`#2`) — best GPU-side
   gain, but conditional on the span holding independent dispatches.
3. **Events on the MTL4 path** (`#5`) — best prefill-only gain; ~2 command
   buffers per layer per chunk today, all on the conservative Metal 3 encoder.
4. Everything else is under ~1% of a decode step, or startup-only.

`#6` (GGUF under ICBs) is not a separate lever — it is the multiplier that
decides whether `#1` covers 30% or 90% of a decode command.

## The host budget these items compete for

Per decode token the submit thread pays roughly:

| Stage | Cost | Source |
|---|---|---|
| `CommandGraph` build | ~30 µs / 300 dispatches | measured, `/tmp/graph_build_bench` |
| `prepared()` cache hit | ~10 µs (vector copy + 300 ARC retains + key string) | estimate from code |
| `commit4` barrier analysis | 4.9 µs chained, up to 102 µs disjoint (300×6) | measured, `/tmp/barrier_analysis_bench` |
| encode + commit | 67 µs direct, 30 µs via spans (300 dispatches) | measured, `mtl4-benchmark` |

So ~110–210 µs of host time per token, of which encode is the largest single
term and the only one with a 2× lever already implemented.

Context for scale: a 4-bit 27B step is ~20–30 ms GPU, a 2B step ~3–5 ms. Host
work is therefore 0.4–1% on the big model and 2–7% on the small one. Nothing
in this list moves a large-model token time by more than a fraction of a
percent; the gains are real on small models, short contexts, and prefill-heavy
serving.

## 1. Span replay under the decode ring — best overall

`replaySpans = events.empty() && !asyncState->hasActiveSubmission()`
(`runtime/metal/MetalEncode.mm:1684`, same gate at `:1405`).

`RICHENGINE_DECODE_RING2` is default-on (`runtime/Tuning.hpp:143`,
`runtime/engine/Engine.cpp:1390`) and submits the next decode while the
retired ticket is still alive. `hasActiveSubmission()` is therefore true for
every ahead command, so **every baked span is direct-encoded in the default
decode configuration**.

Measured on the repo harness, 300 dispatches:

| Path | encode+commit | total |
|---|---|---|
| direct | 67.5 µs | 431.6 µs |
| spans (icb) | 29.6 µs | 538.5 µs |

The encode half of that is what the ring throws away. It also re-pays the
barrier analysis for every span dispatch, because a replayed span joins the
frontier as one unit instead of N.

Cause is the single params arena the ICB binds by address
(`MetalEncode.mm:560-572`, rewritten in place by `patchSpanParams`). Fix is
the one already used for `staging4`/`allocator4`: bank the arena by submission
parity and bake a second ICB per span (ICBs are `maxCommandCount × command
size`, tens of KB). Then drop the `hasActiveSubmission` clause.

Gain: ~30–40 µs per token for span-covered dispatches, plus the analysis pass.
Cost: ~2× ICB memory per cached span. Risk: low — the snapshot validation is
unchanged.

## 2. Bake span barriers from the dependency analysis — best GPU-side

`if (i) [command setBarrier];` (`MetalEncode.mm:634`) serializes every command
in a baked span. The span's bindings and byte ranges are fixed by the snapshot
that validated the replay, so the frontier analysis `commit4` already runs
(`:1799-1887`) can run once at bake time.

Measured, 256-command ICB on this driver:

| Shape | no barrier | barrier per command |
|---|---|---|
| disjoint buffers | 53.5–113.8 µs | 256.8–302.7 µs |
| chained (same buffer) | — (wrong) | 255.4–257.7 µs |

Two conclusions:

- A barrier costs ~0.8 µs of GPU time each, and costs the same whether the
  data actually flows. Independent commands lose 2.5–5× to a barrier they
  do not need.
- A fully chained span gains nothing — the barrier is required. So the gain
  is exactly the fraction of span dispatches that are independent.

That fraction is model-dependent and unmeasured here. Decode layers are mostly
chains (norm → linear → attention), but lanes, MoE experts, and split
attention branches are not. Confidence: medium. Measure with
`RICHENGINE_OP_TIMINGS` on a real model before committing.

The repo harness already shows the symptom: on a disjoint 200-dispatch graph,
span replay's total (417 µs) is worse than direct MTL4 (280 µs) even though
its encode is half.

## 3. Events on the MTL4 path — best prefill-only

Event-bearing commands still take the Metal 3 encoder
(`MetalEncode.mm:2154-2162`). The ANE FFN split signals twice per layer
(`runtime/ops/AneFfn.cpp:573-577`), so a 64-layer chunk creates ~128 command
buffers, all on the conservative hazard-tracking path.

Two costs: ~128 `newCommandBuffer`/commit cycles on the host per chunk, and the
whole chunk's dispatches lose dependency-scoped overlap. The Metal 4 split was
implemented and reverted for timing misattribution plus a `cold_output_match`
regression (CHANGELOG_UNPUSHED §7) — that is a bug to fix, not a limit.

Gain: prefill-only. Larger than the decode host items in absolute µs per
chunk, smaller as a share of chunk wall time. Confidence: medium-high, since
the encoder work is already written.

## 4. Cache the barrier plan per prepared command

`commit4` allocates `needsBarrier`/`frontier`/`ranges` and walks every binding
on every submission (`:1799-1887`). On a `commandMatches` hit the bindings and
byte ranges are provably identical, so the plan belongs in `CachedCommand`
(`snapshotCommand`, `:1276`).

Measured: 4.9 µs for a chained 300×6 graph, 102 µs for a disjoint one. The
disjoint shape is the one the ring forces today (no spans → every dispatch
analyzed). Cheap to implement, no behaviour change.

## 5. `prepared()` copies the dispatch vector per submission

`command.dispatches = candidate->prepared;` (`:1356`) copies 300
`PreparedDispatch` values and ARC-retains one pipeline object each. `commit`
and `commit4` take a `span`, so the cache-owned vector can be rebound in place
and borrowed. Estimated ~10 µs/token; low risk, purely mechanical.

## 6. GGUF kernels under ICB replay — multiplier for #1

`runtime/ops/Linear.cpp:859` and `runtime/ops/MoE.cpp:668` suspend every
Block32/GGUF run out of the span (ISSUES.md: driver produces no output from a
host-baked ICB). On a GGUF model that is most of the decode dispatches.

Effect on the ranking: for packaged affine-Q4 models spans likely cover
80–95% of a decode command, so `#1` is worth its full 2×. For GGUF models the
covered fraction may be 30–50%, and `#1`'s gain shrinks accordingly.
`RICHENGINE_GPU_SPANS` already replays these correctly; it is gated only by the
unresolved `MTL_SHADER_VALIDATION` tile anomaly.

Priority is therefore: `#1` first (it benefits both), then decide whether to
chase the driver bug or harden and default the GPU-span path.

## 7. Read-only bindings skip read/read barriers

The frontier treats every byte-range overlap as dependent
(`MetalEncode.mm:1864-1879`). Weights and tables are read-only by
construction; an access flag on `BufferBinding` lets read/read pairs skip the
barrier. Same 0.8 µs/barrier economics as `#2`, on the direct path. Unmeasured
fraction of avoidable barriers; medium confidence, low effort.

## 8. Per-commit allocation churn

`newCommandBuffer` per commit (`:1763`), `MTL4CommitOptions` per commit
(`:2080`), `make_shared<CommandTicket::State>` per commit. Command buffers are
reusable after `endCommandBuffer`. Microbench: 3–34% of a 9–16 µs submission —
small, free once banked, worth folding into `#1`'s edit.

## 9. Host graph construction

32 µs per 300-dispatch graph: one `vector<byte>` heap allocation per params
payload (`CommandGraph.hpp:47`), one `std::string` per pipeline name, shared_ptr
churn per binding. A per-command payload arena and `string_view` pipeline names
would cut most of it. Same order as encode savings, but touches a widely used
header — higher blast radius than `#1`–`#5`.

## 10. Startup-only (no steady-state effect)

- Cold pipeline compile is 22.8 ms per kernel, 0.41 ms warm (2369 functions
  measured). ~70 pipelines → ~1.6 s cold.
- `shouldMaximizeConcurrentCompilation` unused; `MTL4Archive`/
  `MTL4BinaryFunction` unused. Both cut cold start, neither touches decode.
- `threadGroupSizeIsMultipleOfThreadExecutionWidth` never set; almost all
  threadgroup widths are 32/128/256/1024. Might buy codegen on the hot
  kernels — cheap to test, unproven here.
- `indirectPipeline` compiles a second pipeline per span kernel only for
  `supportIndirectCommandBuffers`. Nearly free once the binary is cached.

## 11. Kernel-side Metal 4.1 (independent of the host path)

- 18 of 20 `simdgroup_barrier` sites still default to `seq_cst` (e.g.
  `shared/moe.metal:115`, `common/gguf_staged_tile.h:89`); `decode/gdn.metal:616`
  already uses `acq_rel` + `thread_scope_simdgroup`. Small, per-kernel, needs
  bitwise tests.
- `common/split_reduce.h:33-56` wait/notify is blocked by
  `SIMDGroupParallelForwardProgress` rejection on this family. Retry on newer
  silicon; it is the substrate for a persistent decode kernel.
- Packed operand gaps: Q2_K is native; Q5_K/Q6_K composites are not.
- No blit encoders and no `MTLHeap` in the tree. KV clears and row copies are
  compute dispatches; transient scratch (vision, MoE) is not aliased.

## Suggested order

Pre-implementation estimates. See "A/B results" for the measured outcome:
items 1–3 came out neutral end to end.

| Order | Item | Type | Expected | Confidence |
|---|---|---|---|---|
| 1 | Bank span params arenas, replay spans under the ring (`#1`) | host | ~30–40 µs/token + analysis | high |
| 2 | Cache the barrier plan (`#4`), borrow the prepared vector (`#5`), reuse command buffers (`#8`) | host | ~10–100 µs/token | high |
| 3 | Bake span barriers from the frontier (`#2`) | GPU | up to 2.5× on independent spans | medium |
| 4 | MTL4 event split for ANE prefill (`#3`) | host + GPU | prefill chunks only | medium-high |
| 5 | Read-only binding flags (`#7`) | GPU | 0.8 µs per avoided barrier | medium |
| 6 | GGUF span coverage (`#6`) | multiplier | scales `#1` for GGUF models | high, blocked upstream |

Everything from `#9` down is either one-time or under 1% of a decode step.

## How to verify, in order

```sh
make build/engine-tests/mtl4-benchmark build/engine-tests/metal-backend.metallib
./build/engine-tests/mtl4-benchmark build/engine-tests/metal-backend.metallib 300 150 --compare
RICHENGINE_OP_TIMINGS=1 ./build/richengine ...   # per-kernel attribution on a real model
RICHENGINE_BACKEND_INSTRUMENTATION=1 ...         # dispatch counts per step
```

`mtl4-benchmark` is scheduling-bound by design (tiny kernels, 200–500
dispatches), so it sizes host and barrier effects but not real GPU time. The
only trustworthy end-to-end gate is `test-performance-real` ABBA on a real
model, which cannot run on this machine as configured.

# A/B results: items 1, 2 and 3, implemented and measured

All three were implemented behind presence flags, built, and A/B'd against the
unchanged path in the same binary.

| Flag | Effect |
|---|---|
| `RICHENGINE_SPAN_BANK_OFF=1` | escape hatch: spans replay only on a GPU-idle backend, one bank |
| `RICHENGINE_SPAN_BARRIERS=1` | span commands barriered only where the byte-range frontier finds a dependency (opt-in) |
| `RICHENGINE_MTL4_EVENTS=1` | event-bearing commands encode on `commit4` (opt-in) |

Banking is now automatic and default-on: the parity bank's arena and ICB are
built the first time a span replays while another submission is in flight.
Nothing is built when nothing overlaps — verified: `decode-profile` and
`dfa-benchmark` build bank 0 only; `mtl4-benchmark --ring` builds bank 1 once
and reuses it.

The first attempt selected the bank by submission parity alone, which collided
when the first submission of a pair ran on a GPU-idle backend and therefore
took bank 0 while the second computed the same bank. It now takes the bank the
in-flight command is *not* using (the last submission's bank, flipped) and
reuses the last bank when nothing is in flight.

## End-to-end, real models through `Engine` (`dfa-benchmark`)

Ornith-1.5-9B-MLX-4bit, 96 steps, B1, best of 2 samples per config, 2 reps.
"base" is `RICHENGINE_SPAN_BANK_OFF=1` with the other two off:

| Config | wall ms | gpu ms | tok/s | vs base | tokens |
|---|---|---|---|---|---|
| base | 805.88 | 795.40 | 111.74 | — | — |
| bank | 805.56 | 795.13 | 111.65 | +0.04% | same |
| `SPAN_BARRIERS` | 818.41 | 808.19 | 110.12 | −1.55% | same |
| both | 807.21 | 796.52 | 111.24 | −0.17% | same |
| both + `MTL4_EVENTS` | 806.01 | 795.74 | 111.61 | −0.02% | same |
| `MTL4_OFF` | 815.21 | 804.44 | 110.58 | −1.16% | same |
| `MTL4_OFF` + both | 815.27 | 804.22 | 110.38 | −1.17% | same |

MiniCPM5-2B-MLX, 512 steps, B1, best of 3 samples per config, 3 reps:

| Config | wall ms | gpu ms | tok/s | vs base | tokens |
|---|---|---|---|---|---|
| base | 366.34 | 353.58 | 309.81 | — | — |
| bank | 368.54 | 355.53 | 307.74 | −0.60% | same |
| `SPAN_BARRIERS` | 365.92 | 353.32 | 309.63 | +0.11% | same |
| both | 366.94 | 354.34 | 308.94 | −0.16% | same |
| both + `MTL4_EVENTS` | 367.19 | 354.70 | 308.56 | −0.23% | same |

MiniCPM5-2B-MLX, B4, 192 steps: all configs within ±0.26%, tokens identical.

Re-run after banking became automatic (base = `SPAN_BANK_OFF`, auto = default):

| Case | base wall | auto wall | delta | tokens |
|---|---|---|---|---|
| 2B unconstrained B1 | 366.4 ms | 367.5 ms | −0.29% | same |
| 2B DFA B4 | 283.5 ms | 283.3 ms | +0.04% | same |
| 2B mask B4 | 282.5 ms | 283.5 ms | −0.38% | same |
| 9B unconstrained B1 | 811.1 ms | 809.8 ms | +0.16% | same |

**Verdict: none of the three moves end-to-end wall time.** Every delta is
inside run-to-run noise (±1%), and the token streams are bitwise identical
everywhere. Decode is 98.6–98.8% GPU time in both models, so host savings and
span scheduling changes do not reach the wall clock.

## Why the host work does not show

The ahead gate never fired in this harness: instrumenting `commit4` shows
`overlapping=0` on every commit in `dfa-benchmark`, for both the unconstrained
and the DFA scenarios. So the ring did not hide anything here — the host
budget is simply small, ~0.4 ms of a 10–30 ms batch. An earlier reading of
this document said the ring was hiding host work; that was wrong, and the
5.3% "agentic win" seen in one DFA run was GPU contention from another
benchmark on the machine, not the change.

`decode-profile` on the 9B, which drives `Runtime` without the ring, puts the
wall−GPU gap at 6.6 ms per cycle, but that gap is dominated by the
draft→verify host round trip, not encode: forcing direct encoding
(`RICHENGINE_ICB_OFF=1`) moves it by 0.06 ms.

The one clean host measurement is `mtl4-benchmark --ring`, which submits a
second command while the first is in flight — the exact condition the bank
auto-enable watches for:

| Graph, 300 dispatches | encode+commit |
|---|---|
| `SPAN_BANK_OFF` (no replay while in flight) | 61.3–62.5 µs |
| auto (bank 1 built on first overlap) | 19.6–20.1 µs |

3.2× on encode, ~42 µs per command. Real and repeatable, but 42 µs is
0.1–0.4% of a decode batch, which is why the end-to-end A/B cannot see it.

### Agentic workloads

Tool-call / JSON-constrained decode is the case the change was meant for, and
it is the case it does not help:

- `dfa-benchmark` `mask` and `dfa` scenarios: `overlapping=0`, bank 1 never
  built, ±0.4% wall (noise), tokens identical.
- The reason is structural: `Engine::trySubmitAheadDecode` excludes lanes with
  `ConstraintMode::TokenMask`, and the DFA path keeps a host round trip between
  commands, so no submission is ever in flight when a span commits.
- Free-form agentic output (no constraint) is plain decode: same 0.1–0.4%
  ceiling as above.

So the auto-enable is correct and free, but its expected gain on agentic
traffic is ~0 until something overlaps there — a merged constrained step on
the ring, or a device-driven decode loop.

## Why the barrier change does nothing

Real spans are chains. Barrier counts at bake time:

| Model / span | dispatches | barriers needed |
|---|---|---|
| 9B target forward | 290 | 289 |
| 35B MoE expert span | 10 | 9 |
| 35B MoE expert span | 13 | 12 |
| 35B attention span | 68 | 67 |

The frontier finds a dependency between every consecutive pair, so the
serialize-everything bake was already correct and already minimal. The
synthetic `mtl4-benchmark` graph does drop 299 → 149 barriers, and its GPU time
still does not improve, because span replay itself costs ~38 µs more GPU than
direct encoding at that dispatch size.

## Why events on Metal 4 do nothing

Prefill, 1024 rows, 9B, ANE FFN split at share 0.38 (`decode-profile`):

| Path | prefill fused gpu | prefill wall |
|---|---|---|
| Metal 3 encoder (base) | 556.71 ms | 557.44 ms |
| `MTL4_EVENTS=1` | 548.22 ms | 549.06 ms |
| `MTL4_OFF` | 549.77 ms | 550.47 ms |

Within noise, and wall ≈ GPU because prefill is GPU/ANE-bound. The 128 command
buffers the split creates per chunk cost a fraction of a millisecond against a
550 ms chunk. The old regression this path caused (prefix-hit prefill output,
`cold_output_match`) was not re-tested: `model-runtime-oracle` does not compile
in this tree (`decodeOne` arity mismatch at
`dev/tests/engine/model_runtime_oracle_test.mm:3037`), so the flag stays
opt-in.

## What the implementation is worth keeping for

- `SPAN_BANK` (automatic, default on): 3.2× encode on span-covered commands
  once a submission overlaps, and zero cost when none does — bank 1 is never
  built. It changes no result and no GPU time.
- `SPAN_BARRIERS`: strictly fewer barriers where ranges are disjoint, no
  measured effect on these models. Keep opt-in until a graph with independent
  branches (GGUF MoE with GPU spans, wide prefill tiles) shows a win.
- `MTL4_EVENTS`: no gain, one known unresolved correctness history. Leave off.

The honest conclusion is that the first-order Metal 4 lever on this hardware is
not in these three. It is the one already shipped: the dependency-scoped
barriers on the direct encoder. What is left is GPU-side — packed operand
coverage, tile shapes, and the GGUF-ICB driver bug that keeps 40–70% of a GGUF
decode command out of spans in the first place.

## Reproducing

```sh
make build/richengine build/engine-tests/dfa-benchmark \
     build/engine-tests/mtl4-benchmark build/engine-tests/decode-profile

# host encode under an in-flight command (the condition banking watches)
./build/engine-tests/mtl4-benchmark \
    build/engine-tests/metal-backend.metallib 300 80 --ring
RICHENGINE_SPAN_BANK_OFF=1 ./build/engine-tests/mtl4-benchmark \
    build/engine-tests/metal-backend.metallib 300 80 --ring

# end-to-end decode A/B
./build/engine-tests/dfa-benchmark build/richengine.metallib \
    install/models/ornith-ai/Ornith-1.5-9B-MLX-4bit \
    --pack build/dfa-bench-ornith9b.pack --scenario unconstrained \
    --steps 96 --samples 2
```

`mtl4-benchmark --ring` and the prefill wall line in `decode-profile` were
added for this A/B and are kept.


# Decode Speed — External Research Cross-Check & Plan

External findings checked against the codebase, then ranked. Complements
PROPOSED_PLAN.md (host path), SPEC_DECODE_BOOST.md (speculation), and
PERF_REPORT.md (survey). Sources: BaseRT (arXiv 2607.00501, 2607.19438),
Rigel (2606.12765), Macpaw NeurIPS ODI 2026, atomgradient studies,
mlx-lm #1450, ZMLX, llama.cpp Metal PRs, NunSpark, Flash-MoE.

## Cross-check: what we already do

| External finding | Status here |
|---|---|
| Few-row MMA verify kernels, weight dequantized once per row batch (llama.cpp) | **Done.** `q4sg::decode`/`gguf_sg` tiles share one weight load across all 8 verify rows (`linear_q4_sgmatrix.metal:56-96`). |
| Fused decode sampler, one dispatch for policy+draw (mlx-lm) | **Done.** Whole selector/policy/acceptance pipeline is GPU-resident (`sampling.metal`); argmax path never computes a softmax denominator. |
| Single fused GDN decode kernel: conv + norm + gates + scan (ZMLX +5–13%) | **Done.** `verify_gdn_fused*` fuses conv taps, Q/K norm, gating, recurrence, and optionally emits the out-projection input table (`decode/gdn.metal:648-676`). |
| fp8 `matmul2d` is emulated pre-M5 — footprint feature, not speed (Rigel) | **Confirmed rejection.** ISSUES.md records fp8 activations failing the numerics bound; Rigel removes the remaining speed motive on M3/M4. |
| Speculative decoding loses on Apple when draft isn't near-free (Macpaw, atomgradient) | **Architecture already answers it.** DFlash2 is one parallel pass conditioned on target hidden state; measured win is real (103 vs 74 target-only equivalent). |
| Decode is dispatch-bound more than compute-bound (ZMLX, BaseRT) | **Known** — PROPOSED_PLAN #1–#3 are exactly this; external data raises confidence, not the finding. |

## Applicable — not yet in tree

### A. Submit-ahead ring, depth 2 (PROPOSED_PLAN #1) — still the top lever

Every external engine result agrees the wall−gpu gap is the first-order
term on Apple; BaseRT's "custom dispatch logic" is its main decode margin
over MLX/llama.cpp. Our serialized tick() loop is the same gap. Design is
done (PROPOSED_PLAN:29-40); nothing external changes it.

**Implemented (prefill-only, opt-in `RICHENGINE_SUBMIT_AHEAD`).** The
dependency analysis narrowed the safe scope: decode can never submit
ahead — the next step's anchor token is decided by the running step's
acceptance — and a decode/prefill mix is unsafe because a prefill
command's commit tail reads decode-arena rows (logits, final hidden) the
other kind's command would overwrite. What pipelines is **prefill chunk
N+1 behind prefill chunk N**, whose inputs are fully deterministic:

- `BackendAsyncState` tracks a deque of in-flight commands (was a single
  slot); MTL3 `commit` and MTL4 `commit4` admit up to 2.
- MTL4 path banks `staging4`/`allocator4` by submission parity and skips
  baked-span replay while a command is in flight (the shared parameter
  arena is not banked); MTL3 path likewise direct-encodes.
- `PrefillArena` banks the three host-written tensors
  (InputTokens/TargetPositions/DraftPositions); `Request` counts
  unconsumed chunks so the encode binds the GDN parities the running
  chunk's pending swap will produce.
- `Scheduler` holds up to two active plans; `promptPlanned` lets a
  continuation chunk plan past the in-flight one's rows.
- `Engine` keeps `pending_` as an ordered deque, consumes strictly in
  submission order, and attempts `trySubmitAhead` whenever exactly one
  prefill is in flight and a bank is free.

Expected gain is small per step (~100–200 µs encode saved of a
multi-ms chunk) and only where prefill chunks are short or numerous;
the depth-2 decode ring the original plan wanted remains blocked on
GPU-driven next-step inputs.

### B. Device-buffer params → bake the attention section (PROPOSED_PLAN #2)

Unchanged. Stacks with A.

### C. Mixed-bit KV recipes — calibration-free variant of L4

`PagedKv.hpp:129` holds ONE global format per serve. SPEC_DECODE_BOOST L4
proposes asymmetric K/V and recency tiers (needs per-page format flags —
per-page flags exist via `KvPageTier`; per-layer does not). mlx-lm's
recipe result suggests a cheaper step: **per-layer-group KV bit widths by
measured argmax-flip sensitivity**, no calibration — measure each
attention layer's contribution to the int4 8-point acceptance tax
(`acceptedCount` delta under per-layer format override), then ship a
per-layer format table for the 27B. Kernels already select format per
layer; only `Layout::format` plumbing and a sensitivity sweep are new.

### D. matmul2d tile/B-operand sweep for prefill AND MoE decode

llama.cpp's standalone tensor-GEMM kernel (configurable NRA×NRB, 64×128
beating 64×32, B read straight from device, no threadgroup staging,
cooperative-tensor device writes) gave +26% geomean on M5 Max. Our
`matmul2d` B operand is already device-resident (`q4_mpp_tiles.h:160-165`,
`tensor_inline`), and decode doesn't use MPP tiles — but
`prefill/linear_q4.metal` TileM×TileN and the MoE expert tiles were tuned
on one shape (TODO #5). **Extend TODO #5's sweep with the llama.cpp grid
points and per-shape kernel-name selection.** Same harness
(`dev/benchmarks/mtl4_benchmark.mm`).

### E. Wired-budget + compressor guardrails for the disk tier

Residency wires all backend buffers (`Residency.hpp`); `SlotFile` sets
F_NOCACHE so the KV tier doesn't double-buffer RAM. The external numbers
add two specifics worth encoding:

- **~0.75× RAM wired ceiling** — past it the macOS compressor steals
  pages and throughput regresses (NunSpark measured the cliff). Check
  `MemoryGovernor`/`MemoryPlan` budgets count wired residency against
  this bound, not against total RAM.
- **Requantized/streamed expert pages served from page cache beat
  app-level caching** (Flash-MoE +38%). Our F_NOCACHE choice is correct
  for the KV tier (self-managed), but if MoE expert streaming ever lands,
  the opposite policy applies — record in KvPageTier notes.

### F. Adaptive per-lane proposal length (SPEC_DECODE_BOOST L3)

Still unimplemented; `AcceptedCount` readback exists (`Runtime.mm:2219`).
Cheap heuristic version needs only Scheduler/Runtime changes. Mostly
subsumed by a working tree, but the comb tree measured −12%/−32%
(TREE_VERIFY_DESIGN) — L3 is the low-risk fallback that recovers some of
that headroom on sampled/weak-draft lanes.

EdgeAgent (arXiv 2610.03394) adds the scheduling frame: on UMA the
decode phase is memory-bound, so every wasted proposal is wasted
bandwidth, and agentic workloads alternate between hard reasoning and
predictable structured output. The adaptive controller sizes a per-lane
draft budget from real-time predictability. Three differences shrink the
win here versus their 1.29x: the DFlash2 draft is one parallel pass
conditioned on target hidden state (near-free, not a serial bandwidth
competitor), verify reads weights once regardless of row count, and the
scheduler already suspends lanes on resource pressure.

Design, smallest first:

- **Draft on/off gating** (budget 0 vs 8). Skip the DFlash pass when the
  lane's EWMA acceptance sits below a floor (~30%) for a window, and when
  `applyNgramPredraft` is hitting — tool args, JSON, repeated diffs are
  exactly the high-predictability spans EdgeAgent targets. Costs no draft
  retraining; `draftQueryRows` stays the trained 8.
- **Per-lane proposal clamp** (2/4/7 of `draftProposalTokens`). At width
  4 one weak lane wastes verify rows for the whole packed command;
  clamping it recovers bandwidth the peers use.
- **Signals**: EWMA of `acceptedDraftTokens/draftedTokens` per lane
  (already in `ModelStepResult`, exported as
  `richengine_draft_acceptance_ratio`), plus the n-gram predraft hit rate
  to choose the proposal source. Hysteresis thresholds so the budget does
  not oscillate mid-sequence.

Expected: near-zero at B=1 (draft is near-free, verify rows are cheap),
a few % E2E at B>=2, plus robustness on sampled/weak-draft lanes where
int4 KV already taxes acceptance ~8 pts at long context (63% -> 55%,
TurboQuant_ANLYSIS). Gate it behind `RICHENGINE_ADAPTIVE_DRAFT=1`
(env-flag precedent: `RICHENGINE_NGRAM_PREDRAFT`), validate with
`dev/benchmarks/trainfree_spec_ab.py` ABBA on real agentic traces, then
consider a `--draft-budget=auto` serve option. Keep it behind the flag
until acceptance parity is proven — a mis-tuned floor silently disables
speculation on exactly the long outputs that need it.

### G. Global suffix index (L2) — unchanged, highest-leverage retrieval gap

SuffixDecoding/AgSpec numbers (up to 5.3× on agentic traces) stand.
`RICHENGINE_NGRAM_PREDRAFT` is per-lane 3-gram; the gap is a cross-request
suffix automaton over emitted tokens. Host-only, lossless.

### H. Shared-prefix attention reads (ChunkAttention TPP, arXiv 2402.15220)

ChunkAttention's prefix-aware KV cache (PAKV) is already the design — the
cache is a block-granular prefix tree with shared blocks and junction
boundaries. The open half is its two-phase partition: query rows of all
sequences sharing a prefix batch against each shared KV chunk, so the same
physical pages are read once per tile instead of once per lane.

Verify attention grids `(kv_head, split, lane)` and every lane walks its
own page table over identical physical pages below the shared boundary
(`paged_attention.metal`, `richengine_verify_attention_tile_at`). For 4
lanes sharing a 10K system prompt at int8 (~33 KB/token), that is ~1 GB of
redundant KV read per step — ~4 ms of a ~60 ms weight-bound step, growing
to ~10–15% of decode at 30K+ shared prefixes. Exact-numerics safe: a pure
data-locality rewrite.

Shape: scheduler groups lanes by shared prefix depth (cache already knows
junctions); verify attention gains a shared-span mode where splits below
the group's common boundary run M = lanes×rows fused rows with per-lane
limits and per-lane output slots; suffix pages keep the per-lane path.
Main kernel risk is threadgroup budget — `scores[M*N]`/`probabilities`
grow ~4× and may force smaller splits.

Dilutions: only the 16 full-attention layers benefit (48 GDN layers have
per-request recurrent state, unshareable); decode is weight-bound so the
win scales with shared-prefix length and lane count, not baseline speed.
Schedule only when real deployments show ≥3 concurrent lanes sharing >8K
prefixes and `CommandTiming` shows KV reads, not dispatch, dominating the
attention span.

## Not applicable / rejected

- **KV-cache pruning** (Macpaw 1.75×): lossy, violates exactness contract.
- **PEARL overlap, pre-verify, early-exit**: SPEC_DECODE_BOOST ruled out.
- **fp8 e4m3 KV→fp16 Q stage**: TODO #4 already tracks the quality fix.
- **Remote drafting over LAN**: ~50 ms/op RPC overhead — dead.
- **ANE in-step overlap**: no public precedent; keep serial-queue design.

## Plan

| # | Work item | Type | Expected | Gate |
|---|---|---|---|---|
| 1 | Submit-ahead ring depth 2 (double-buffered upload slots, per-slot tickets) | host | Largest; removes inter-token GPU idle | wall−gpu gap in `CommandTiming`; `test-performance-real` ABBA |
| 2 | Device-buffer params for verify params + `width`; bake attention span | host | ~48 dispatches/token un-encoded | `RICHENGINE_BACKEND_INSTRUMENTATION` dispatch count |
| 3 | Fuse `decode_linear_q4_prepare` into producing norm epilogue | kernel | ~100 dispatches/token | instrumentation + bitwise tests |
| 4 | Threadgroup-stage Q4 scales/biases (TODO #2) | kernel | single-digit % on m24/m32 tiles | tile benchmarks |
| 5 | Per-layer KV format table + acceptance-sensitivity sweep (C) | engine + experiment | recovers int4 acceptance at int8-ish quality for sensitive layers only | `acceptedCount` delta; `gguf_projection`-class tests |
| 6 | MPP tile sweep incl. llama.cpp grid points (D) | tuning | prefill + MoE decode | `mtl4_benchmark.mm` |
| 7 | Adaptive proposal length per lane (F) | host | few % at B≥2; fallback if tree stays gated | ABBA |
| 8 | MemoryGovernor wired-ceiling audit vs ~0.75× RAM (E) | config | prevents compressor regressions on small-RAM tiers | M6/24GB benchmark runs |
| 9 | Global suffix index over emitted tokens (G) | host | largest agentic-workload lever after #1 | acceptance on real coding-agent traces |
| 10 | Shared-prefix attention reads (H) — conditional | kernel + scheduler | ~5–15% decode under ≥3 lanes sharing >8K prefixes | `CommandTiming` attention-span KV-bound evidence on real multi-session load |

Order: 1→2→3 are the multiply-everything substrate. 4–6 are independent
kernel/tuning work. 7–10 are policy; 10 is additionally gated on
concurrency evidence, not just implementation cost. Measure the draft/verify/host split
first (`RICHENGINE_BACKEND_INSTRUMENTATION`, `benchmark-decode-profile`) —
it sizes every claim.

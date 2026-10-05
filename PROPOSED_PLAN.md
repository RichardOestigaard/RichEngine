# Proposed Performance Plan

Code-level findings from the submission path, graph encoding and kernels.
Complements TO_EXPLORE.md (Metal 4/4.1 features), TO_EXPLORE2.md (memory
ordering, indirect commands), PREFILL_OPTIMIZATION_PLAN.md (TTFT) and
TODO.md. Items already tracked there are noted; the rest are new.

## The serialization — biggest structural lever

`BackendAsyncState::beginSubmission` throws while a command is in flight
(`MetalBackend.mm:302`). The engine holds exactly one `pending_` ticket:
`tick()` waits `ticket->ready()` → `wait()` → `apply()` → then calls
`scheduler_.next()` → `model_.submit()` (`Engine.cpp:253–288`).

Per decode token the pipeline is fully serial:

    GPU token N → completion cb → finalizeDecode (reads output tokens) →
    scheduler → encode CommandGraph (~400–600 dispatches) → prepare() →
    encode/commit → GPU token N+1

The GPU is idle for the entire host turnaround. At 74 tok/s on the 27B the
budget is ~13.5 ms/token; GPU time is ~4–7 ms, so host turnaround is
plausibly 10–20% of decode throughput. The same gap recurs between the four
serial prefill-chunk submissions of a request.

This is the tractable end of TODO.md #1 ("GPU-driven dispatch"). Two
increments short of the full architecture project:

- **Ring depth 2.** `staging4`, `allocator4`, the watchdog model and
  `releaseMemory`'s `commandInFlight()` gate all assume one command in
  flight. Depth-2 rings for staging/allocator plus per-slot ticket tracking
  enable submit-ahead: encode token N+1 while N runs. The arena parity swap
  (`states.swapParity`) already alternates state buffers; the remaining
  hazard is host writes into shared upload buffers, solvable with
  double-buffered upload slots.
- **Cached PreparedCommand per batch shape.** The decode graph is
  structurally static per step — only bytes payloads change. Cache the
  prepared command per (width, constrained, draft shape) so `prepare()`,
  validation, sort/unique and span re-validation run once per shape instead
  of per token. A subset of the ring-2 win, standalone.

## Device-buffer params make the attention section bakeable

`PagedAttention::addVerify` suspends the span for store/split/reduce because
the param blocks carry `committed_tokens` and split counts that change every
step (`PagedAttention.cpp:402–407`). Each attention layer therefore splits
its baked span into two fragments plus three direct-encoded dispatches —
per token, for 16 attention layers, roughly 32 span fragments, ~48 direct
dispatches, ~32 `executeCommandsInBuffer` calls and ~300 `useResource`
calls.

The kernels already take `constant T *params` as a buffer binding
(`paged_attention.metal:67`, `[[buffer(7)]]`), not `setBytes`; only
`graph.add(..., attention, ...)` passing the `std::array` by value makes it
a `BytesBinding` that fails the snapshot match every step.

**Fix:** stage the params array into a persistent device buffer slot each
step (a shared-storage write, ~100 B) and bind the buffer. The binding is
static, so the dispatches bake. Per token this eliminates ~48 direct
encodes and merges the span fragments. The same trick applies to
`decode_linear_q4_prepare`'s `width` param and any other per-step
`BytesBinding` — and it is the host-side version of TO_EXPLORE2's GPU-side
`compute_command` patching, without the driver bug exposure.

## Per-token host encode cost

Per token `decodeAsync` rebuilds the CommandGraph (~400–600
`ComputeDispatch`, each with two heap vectors and a string), then
`prepare()`:

- **Retained-allocation dedupe** sorts + uniques every bound allocation
  (`MetalBackend.mm:1084–1099`): ~4,000 pointers sorted per token. A
  generation-stamp on `MetalAllocation` (mark visited in a per-submit
  epoch) makes it O(B) with no heap traffic.
- **Graph allocation churn**: ~1,500 heap allocations per token. A
  `CommandGraph` pool that resets `dispatches_` and `payloads_` and reuses
  their storage removes most of it.
- **`snapshotMatches`** does `view.allocation != saved.allocation.lock()` —
  a `weak_ptr::lock()` (atomic incref/decref) per binding per span per
  token. Store the raw `MetalAllocation *` alongside the weak pointer for
  compare-only.
- `bakedSpans` keys build a `std::string` per run per submission; a
  (begin, length, name-hash) struct key avoids the allocation.
- `pipeline()` hash lookups per dispatch are fine; they disappear entirely
  once the prepared command is cached.

## `commit4` frontier analysis — potential O(n²)

`MetalBackend.mm:1328–1376` scans each dispatch's bound ranges against the
whole frontier; the frontier clears only on a barrier. Mostly-dependent
decode chains keep it small, but independent per-lane buffers accumulate —
worst case tens of millions of range compares per submission on the MTL4
path. A `buffer → last-range` map answers the same question in O(ranges).

Also: the `stagingBytes` pre-pass (`:1244–1247`) counts bytes of
span-covered dispatches that never stage — `staging4` over-allocates.

## Kernel-level findings

1. **Q4 scale/bias epilogue loads** (TODO #2, confirmed):
   `q4_mpp_tiles.h:224–235` — `finish_group` issues `scales_0[parameter]`
   and `biases_0[parameter]` per accumulator element per quant group:
   scattered 2-byte device loads strided by `kQ4StorageColumns`. Staging
   each group's 2×TileN bf16 vector into threadgroup memory (~1 KB) turns
   them into contiguous loads. The m24/m32 shapes with the 30–40% gap are
   exactly the high-Rows variants.

2. **`decode_linear_q4_prepare` is a fuseable dispatch**
   (`linear_q4_sgmatrix.metal:137–148`): every simdgroup-path projection
   pays an extra dispatch to build the input table and row sums. The
   producing RMSNorm already touches every element — it can emit the table
   layout and sums as an epilogue. Removes ~1–2 dispatches per projection
   (~100+/token) plus a table write+read round trip.

3. **`split_arrive_last`** (`split_reduce.h:14–28`): two
   `threadgroup_barrier`s plus two device-scope `seq_cst` fences around one
   relaxed `fetch_add`. `memory_order_acq_rel` on the atomic plus one
   barrier gives the same publish ordering — the MSL 4.1 order/scope args
   from TO_EXPLORE2 apply here and in `sampling.metal`'s arrivals.

4. **`vocabulary_draw`** (`sampling.metal:746–794`): three
   `thread_index == 0` serial sections between barriers (kept-mass sum,
   cumulative range walk, residual over `kVocabularyRanges` +
   `kDraftCandidates`), each serializing ~256 threads for ~30 iterations.
   Parallelizable with `simd_sum`; sampling is a small share of token
   time — low priority.

5. **Attention split+reduce partials round-trip** through device memory
   (~200 KB/lane/layer at worst split count). Small traffic — the dispatch
   gap costs more. In-kernel split fusion becomes expressible once MSL 4.1
   device-scope ordering lands (TO_EXPLORE2 "Device-scope writes").

6. **Prefill GDN serial scan** — plan lever 0; chunked WY/UT was
   implemented then abandoned (`matmul2d` N>16 silent-wrong bug +
   latency-bound at T=2048). Remaining in-tree prefill levers: chunk-policy
   sweep (plan lever 4) and the Q4 epilogue staging above.

## Minor

- `Residency::use()` per commit — mutex + timestamp, fine.
- `sampleDeviceMemory()` in `release()` — safe today (GPU idle then) but
  can synchronize with an in-flight command on some GPUs; revisit if
  submissions pipeline.
- `submitProfiled` — one command buffer + wait per dispatch; instrumentation
  only.

## Ranked

| # | Change | Phase | Expected effect |
|---|---|---|---|
| 1 | Submit-ahead / depth-2 command ring | decode + prefill chunks | Removes GPU idle between tokens; largest |
| 2 | Device-buffer params → bake attention section | decode | ~48 dispatches/token un-encoded; spans merge |
| 3 | Fuse `decode_linear_q4_prepare` into norm epilogue | decode | ~100 dispatches/token plus round trips |
| 4 | Threadgroup-stage Q4 scales/biases | decode + prefill m24/m32 | Single-digit % on affected tiles |
| 5 | Frontier map in `commit4`; dedupe without sort; graph pool | host | Shortens the turnaround (feeds #1) |
| 6 | acq_rel atomics / wait-notify (TO_EXPLORE2 list) | decode | Small, hygiene-correct |
| 7 | Cached PreparedCommand per batch shape | decode | Medium; subset of #1 |

## Verification

- `make benchmark-backend` — decode tok/s and prefill characterization.
- `make test-performance-real BASELINE=<retained checkout>` — ABBA gate.
- `dev/benchmarks/mtl4_benchmark.mm` — ICB replay vs direct encode cost.
- `SPLASH_BACKEND_INSTRUMENTATION` dispatch profile — dispatches/token
  before and after items 2–3.
- wall−gpu gap on `CommandTiming` per step — the direct measure of item 1.

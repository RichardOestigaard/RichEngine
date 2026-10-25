# Batch width research

Where the four-lane cap lives, what it protects, and how it could become a
per-model value. Researched on the working tree of 2026-10-25 (post-Granite
`dfaMode` maker default); citations are `file:line` against that tree.

## The cap

Two mirrored constants, asserted equal:

- `ExecutionLimits::maximumBatchWidth = 4` — `runtime/model/Model.hpp:385`
  (host contract). Sibling limits in the same struct:
  `prefillTokenBudget = 2048`, `draftQueryRows = 16`,
  `targetVerifyRows = 16`, `maximumStepTokens = 17`.
- `RICHENGINE_MAXIMUM_BATCH_WIDTH 4u` — `runtime/metal/abi/ExecutionGeometry.h:25`
  (the value the precompiled metallib's param blocks and kernel signatures
  are built against).
- `static_assert(kMaximumLanes == 4)` — `runtime/ops/ExecutionPlans.cpp:14`:
  a deliberate tripwire so widening fails the build loudly.
- Aliases: `kLaneCount`, `kDecodeRows = targetVerifyRows` —
  `runtime/model/RuntimeGeometry.hpp:25-26`; `kPackedAttentionRows` grows
  the prefill arena's packed rows by `kLaneCount * (kTileRows - 1)`
  (:33-34) — `PrefillArena.mm:71-78,145-150`.

Consequence: the ABI value is **baked into the metallib at build time** —
a runtime or per-model cap can only shrink admission below 4, never grow
it without a recompile.

## What assumes 4 literally

These bind exactly `state0..3`/`table0..3` per lane or hardcode the count:

- Kernel signatures (4 buffer slots each):
  `runtime/metal/kernels/decode/gdn.metal` (:337, :365-371, :491-544,
  :553-572, :895-900 — layer-stride `layer * RICHENGINE_MAXIMUM_BATCH_WIDTH
  + batch`),
  `shared/lfm_conv.metal` (:155-172, :231, :382-438, :446-476, :545-547),
  `decode/draft.metal` (:119-169, :604-703),
  `decode/dfa.metal` (:34-47),
  `decode/paged_attention.metal` (:26-45, :61-72),
  `decode/fp8_attention.metal:44-46`,
  `decode/paged_attention_store.metal:29-31,223-225`,
  `decode/fp8_attention_store.metal:29-31`.
- Host literals that compile but silently drop lanes ≥4:
  `runtime/ops/PagedAttention.cpp:650-657,737` —
  `std::array<MetalBuffer, 4>` and `tables[0]..tables[3]` while neighboring
  code uses the macro.
- `runtime/model/Runtime.mm:1101` — warmup order
  `std::array<uint32_t, kLaneCount> stateLaneOrder{2,0,3,1}` — a baked
  4-entry permutation; at kLaneCount>4 the tail zero-inits to a duplicate
  lane 0 and `tryActivateLane` throws.
- `runtime/engine/wire/Status.cpp:291-296,325-331` — `b1..b4` /
  `decode_b1..decode_b4` telemetry keys; wider engines truncate counters.
- `server/server.py:1230-1231` — `preparation_capacity or 4` literal
  fallback in the shared-prefix fan-out.
- `install/autotune.py:59` — `BATCH_WIDTH = 4` (the sweep's full-width
  width; `dev/tests/test_tuning.py:128-133` asserts it against the parsed
  constant). `AGENTIC_WIDTHS = (2, BATCH_WIDTH)` (:63).
- `dev/tests/engine/model_package_test.cpp:68` — contract test pins
  `maximumBatchWidth == 4`; `dev/benchmarks/gguf_moe_benchmark.mm:244,335`,
  `dev/tests/moe_metal_test.mm:580,932` use literal `lanes <= 4`.

## What scales with width

| Resource | Scaling | Where |
|---|---|---|
| Decode arena | O(width) — every tensor stride × `kLaneCount`; per-lane tensors at `kDecodeRows`=16 rows. Largest: `DfaNextStates` ≈ 8.4 MB/lane at V=248k | `DecodeArena.mm:49-247`, `DecodeArena.hpp:185-191` |
| Per-lane composite state | O(width), allocated elastically on lane activation (GDN cells + draft ring) | `CompositeStateStorage.hpp:191-193,286`, `StateLayout.hpp:79-102` |
| KV pool | **Shared, not multiplied** — one elastic pool; per-request advertised context unchanged; more lanes just share it (suspend/preempt exists) | `RuntimeResources.mm:506-514`, `MemoryPlan.cpp:253-261,344-357`, `Engine.cpp:1250-1340` |
| KV runway | Fixed — `warmupKvPages` = 64 pages at pool construction | `Model.hpp:387-390`, `KvPool.cpp:14-28` |
| Draft/verify rows | O(width), hard ceiling 64 rows total (`16×4`) | `Linear.cpp:72-75,785-786`, `MoE.cpp:860-865`, `PagedAttention.cpp:387-389` |
| Attention | O(width) threadgroups, fully parametric — grid `(kvHeads, splits, lanes)` | `paged_attention.metal:35-57`, `PagedAttention.cpp:400-438` |
| Sampling/DFA | O(width) — `policies.size() ≤ kMaximumLanes` | `Sampling.cpp:13-366`, `DecodeArena.mm:201-209` |
| Host per-lane state | O(width) — ~50 `std::array<…,kLaneCount>` in `RuntimeImpl.hpp` | `:281,:425,:457-479,:530-532` and dispatch arrays |
| Lane masks | u32 `1u << lane` — **hard ceiling 32 lanes** | `metal/abi/Dfa.h`, `Sampling.h`, `RuntimeImpl.hpp:940-993`, `dfa.metal:169..632` |

Nothing material is O(width²).

## The real binding constraint: the 64-row decode tile budget

Decode linear work is capped at `TARGET_VERIFY_ROWS (16) × width` rows —
64 at width 4 (`Linear.cpp:72-75`). The fused-M kernel family only
instantiates to M64 (`KernelNames.hpp:224-271`; `Linear.cpp:363-369`
throws past 8 tiles). Width 8 needs either M96/M128 instantiations of ~15
kernel variants or the slower generic Simdgroup / GGUF-register paths
(already implemented — `Linear.cpp:629-632`, `LinearGguf.cpp:26-31,
277-279` — functional but slower).

This is also why **tree verify already halves effective width**: a tree
lane occupies two 16-row blocks, capping tree batches at `width/2`
(`RuntimeImpl.hpp:1134-1135`, `RuntimeEncodeVerify.mm:10-17`,
`TargetModel.cpp:654-664`). The invariant is the row budget, not the
lane count.

The LM head's fused argmax only handles ≤32 total rows
(`RuntimeEncodeVerify.mm:71-75`); verify is weight-bandwidth-bound
(~16 GB/step on a 27B Q4) with head compute-bound ~24-32 rows
(`docs/SPEC_DECODE_BOOST.md:45-51`, `docs/TREE_VERIFY_DESIGN.md:64-72`).

**Why 4 was chosen — performance, not capacity.** Decode lanes are nearly
free until the 64-row verify goes compute-bound; small models saturate
earlier (LFM2.5 flat by width 2 — `docs/performance.md:131-134`).

## The parameterization path (cap ≤ 4, no recompile)

`--max-context` is the precedent end-to-end: argv → engine → ReadyEvent →
server. The batch-width analogue:

1. **Declare** — `ModelTuning.maxBatchWidth` (`std::optional<uint32_t>`,
   `runtime/model/ModelTuning.hpp`, beside `proposalCap` ~:80). Family
   makers set `descriptor.tuning.maxBatchWidth` — same pattern as
   `tuning.dfaMode` (Granite.mm) and `Qwen.mm:91`'s
   `adaptiveProposals = false`.
2. **Resolve** — `ModelPolicy.maxBatchWidth`, `resolveModelPolicy` with
   `std::clamp(…, 1u, ExecutionLimits::maximumBatchWidth)`; env override
   `RICHENGINE_MAX_BATCH_WIDTH`. Policy installs in `Runtime`'s ctor
   (`Runtime.mm:54-56`), before `NativeRuntime`/`Engine` exist
   (`Bootstrap.mm:340,371`).
3. **Reach admission** — `EngineConfig.maxBatchWidth` (`Engine.hpp:36`)
   populated in `Bootstrap::start` after `createRuntime`
   (~`Bootstrap.mm:363`). Consumers: `Engine.cpp:543-546` (`lanesFull`),
   `Scheduler.cpp:324,417,425` + `kMaximumOvertakes` (:12-13), constructed
   beside `decodeShare` (`Scheduler.hpp:67`, `Engine.cpp:75`).
   Cheaper alternative: those sites read `model::modelPolicy()` directly —
   valid ordering, since the policy binds before the runtime exists.
4. **Announce** — `NativeRuntime.cpp:202` sends the configured value
   instead of the constant. `ReadyEvent.maxConcurrentRequests` is already
   the wire channel (`Protocol.hpp:187,267-275`; encoder `Protocol.cpp:
   591-595`; Python mirror `server/protocol.py:343-347`). No protocol
   change.
5. **Server** — zero changes: `readiness.max_concurrent_requests` →
   `Frontend(preparation_capacity)` → `BoundedSemaphore`
   (`model_host.py:341`, `frontend.py:586,1305,1314`); the relaunch
   consistency pin (`runtime.py:1155-1173`) already covers it.

### Gaps to fill

- **Family JSON → descriptor: no hop.** No family-JSON field is compiled
  into `ModelDescriptor` today — makers are the only native per-family
  channel. If JSON should be source of truth, a new key is needed in the
  assembly record (`RECORD_KEYS`, `install/assembly.py:51-61`) and the
  package manifest, parsed in `inspectSourceModel`/`inspectModelPackage`.
  (`pack.execution_geometry` is precedent for a manifest numeric but is
  validation-only — `ModelDescriptor.mm:242-267`.)
- **Warmup** — `Bootstrap.mm:208-224` warms widths 1..max; min the ladder
  with the cap or a capped model still warms B4.
- **Autotune** — `install/autotune.py:59` `BATCH_WIDTH = 4` is hardcoded;
  a `MAX_BATCH_WIDTH` knob in `_knobs.json` (`values: ["3","2","1"]`)
  would let the sweep tune it per model.
- **Floor** — `validateReady` rejects `maxConcurrentRequests == 0`
  (`Protocol.cpp:454`); clamp the resolved cap to `[1, 4]`.
- **Status** — `b1..b4` keys truncate above 4 (`Status.cpp:291-296,
  325-331`).

## Widening past 4 (recompile project, not a knob)

- Bump both constants; `ExecutionPlans.cpp:14` tripwire flips first.
- New kernel signatures or a redesign for the ~7 kernels with 4 per-lane
  buffer slots (listed above).
- M96/M128 fused-M instantiations across `KernelNames.hpp`'s ~15 decode
  variants — or accept the generic Simdgroup/GGUF-register paths.
- Regenerate `stateLaneOrder`; extend `decodeBatchesByWidth` and status
  keys; retune `DecodeGroupPolicy`.
- u32 lane masks hold until 32 lanes.
- Payoff is doubtful for decode: the 64-row verify budget goes
  compute-bound past B4 — but tree verify does gain (cap rises 2→4 tree
  lanes), and prefill packing benefits.

## Recommendation

Ship the per-model cap first (≈5 files, all plumbing exists, ABI
untouched). Candidates from today's data: models that saturate early
(LFM2.5 by B2) waste warmup time and arena bytes on lanes they never
fill; a `tuning.maxBatchWidth` lets a family declare its measured sweet
spot without touching the 64-row invariant.

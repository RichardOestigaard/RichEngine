# Untapped Metal 4.1 Pathways

Inventory of Metal 4.1 (MSL `-std=metal4.1`, macOS 27 toolchain) surface this
engine does not yet use, after the pathway work in this branch landed:

- MTL4 submission (`commit4`), banked allocators, argument table, commit
  feedback — still opt-in via `RICHENGINE_MTL4`.
- GPU-encoded spans on the MTL4 path (was a silent no-op — now replays
  `gpu_encode` + `executeCommandsInBuffer` under `RICHENGINE_GPU_SPANS`).
- Indirect dispatch grids (`ComputeDispatch::indirectGrid`,
  `dispatchThreadgroupsWithIndirectBuffer:`) and GPU-written span replay
  ranges (`ComputeDispatch::indirectRange`,
  `executeCommandsInBuffer:indirectBuffer:`), on both encoders; first
  decode consumer wired: `decode_dfa_mask` grid under
  `RICHENGINE_GPU_DECODE_GRID` (`decode_dfa_mask_grid` writes it).
- Event-split commands on the MTL4 path: `commit4` splits at EventSteps
  with queue-level `signalEvent:`/`waitForEvent:` (was: fell back to the
  Metal 3 encoder).
- `gdn.metal` intra-simdgroup barriers narrowed to acq_rel +
  simdgroup scope.
- GGUF Q2_K native packed tile (`q2k_mpp_tile.h`,
  `gguf_decode_q2km_*`), gated behind `nativeFormats()` +
  `RICHENGINE_GGUF_PACKED_ON` like its siblings — benchmarked only by
  construction here; no Q2_K model is installed locally.
- Tried and ruled out on toolchain 32023 / Apple family 10:
  `bf16 × fp8` matmul2d operands (no such intrinsic — fp8 right operands
  pair only with half or same-format fp8); `atomic_wait/notify` in
  split_reduce.h (marks kernels SIMDGroupParallelForwardProgress, rejected
  at pipeline creation on this device — kept opt-in under
  `-DRICHENGINE_SPLIT_WAIT_NOTIFY`); `resetCommandsInBuffer:` (the replay
  path never rewrites commands — host-patched parameter arenas carry all
  per-submission state, nothing to reset).
- Call-site function stitching (`entry[hook=impl:arity]` pipeline names via
  `MTLStitchedLibraryDescriptor`) and the MTL4 compiler pipeline path
  (`entry|persistent`, `|fp=weak|fp=simd`, `|relief=none` →
  `MTL4ComputePipelineDescriptor` options through `MTL4Compiler`).

## Remaining, ranked by expected decode value

### 1. Default-on MTL4

`RICHENGINE_MTL4` is presence-gated. `mtl4-benchmark --compare` now prints
the encode/commit delta; when it wins, flip the default and keep the env as
an off-switch. The remaining MTL4 coverage gap: commands with `EventStep`s
still take the Metal 3 path — `MTL4CommandQueue` has
`signalEvent:`/`waitForEvent:` and `MTL4CommandBuffer` has `waitFence:`, so
the split-at-signal encoding can be reproduced on MTL4 (each split becomes
one `MTL4CommandBuffer` with a queue-level signal between commits).

### 2. GPU-driven decode steps

The new `indirectGrid`/`indirectRange` fields are the substrate; the decode
loop does not produce them yet. The blocker named in
docs/DECODE_SPEED_PLAN.md ("decode can never submit ahead — the next step's
anchor token is decided by the running step's acceptance") becomes solvable:
a GPU-written grid/range lets the next verify submit while the current one
runs, with the sampler writing the span lengths. Design target: the
draft/verify dispatch count and the per-lane verify span tails are the two
quantities acceptance actually changes.

### 3. Device-side wait/notify (`__HAVE_ATOMIC_WAIT_NOTIFY__`, new in 4.1)

`atomic_wait_*`/`atomic_notify_*` at simdgroup and device scope, plus
`yield_simdgroup(t)`, `critical_section`, and `forwardProgressUsage` (now
reachable as `|fp=` pipeline options). Uses:

- Split-K merges: `common/split_reduce.h` polls a `coherent(device)`
  counter with seq_cst fences — `atomic_wait` + notify is cheaper and lets
  the reducer sleep rather than spin.
- Producer/consumer fusion inside one command: a consumer threadgroup can
  wait on a device atomic a producer kernel sets — dependency chaining
  finer than `barrierAfterEncoderStages`, which serializes whole
  dispatches.
- A persistent decode kernel is the big version: needs
  `optimizeForPersistentKernel` (`|persistent`) plus the wait/notify
  pair for its step loop.

### 4. Memory-order barriers and scoped atomics

`simdgroup_barrier(flags, memory_order, thread_scope)` (4.1) — every call
site uses the seq_cst default (`q4_mpp_tiles.h`, `paged_attention_tile.h`,
`gdn.metal`, `split_reduce.h`). Audit each barrier for acq_rel/relaxed +
narrower scope. New `atomic<T, thread_scope>` (partial-order atomics) and
threadgroup `atomic<float>` simplify the same code paths.

### 5. Packed-operand coverage gaps in matmul2d

Already native: q40/q41/q4k (uint4b), q80 (int8_t), pq20 (uint2b),
mxfp4m (fp4_e2m1 + ue8m0 multiplane). Still staged-dequantized:

- **Q2_K → `uint2b_format`** — the pq20 tile proves the pattern (data
  plane + per-superblock scale epilogue); Q2_K still runs the staged tile.
- **Q5_K / Q6_K composites** — 5-bit = uint4b low plane + a fifth-bit
  contribution; 6-bit = uint4b + uint2b as two operand matmuls. Both avoid
  staging at the cost of a second accumulation.
- **Stale assumption to re-test**: `decode/fp8_attention.metal` stages fp16
  because "matmul2d admits no bf16 operand" — the 4.1 operand table lists
  `bfloat × bfloat → bfloat/float`. Verify and drop the fp16 stage.
- **fp8 e4m3/e5m2 KV operands** — `metal_fp8_e4m3_format` right operands
  exist at 4.1; the KV path currently goes through int8.

### 6. MTL4MachineLearning + MTLTensor

`MTL4MachineLearningCommandEncoder` + `MTL4MachineLearningPipelineState` +
`MTLTensor` dispatch whole compiled networks (a heap of intermediates, one
dispatch). MTLTensorDataType gained Int4/UInt4 (26.4) and UE8M0/UInt2
(27.0). Whole-network dispatch is a poor fit for per-token decode kernels,
but the ML pipeline could cover fixed-shape draft passes or the head, where
Apple's compiled network does its own fusion. Untested hypothesis;
prototype before committing.

### 7. Smaller items

- `resetCommandsInBuffer:` / `optimizeIndirectCommandBuffer:` on the MTL4
  encoder — device-side ICB reset/optimization for baked spans instead of
  host patching; removes the params-arena sharing constraint that disables
  span replay under submit-ahead.
- `MTL4BinaryFunction`/`MTL4Archive`/`MTL4PipelineDataSetSerializer` —
  precompiled binary functions + archives for startup and per-shape
  specialization.
- `copyFromTensor:` on the compute encoder, if tensors ever back KV.
- `MTLTexture` block reads (`__HAVE_TEXTURE_BLOCK_READ__`) for weight
  streaming — speculative, likely minor for GEMM.
- `interleave_bits` intrinsics for quant repack/int4 KV packing.
- Imageblock on MTL4 compute encoders (`setImageblockWidth:`) — tile
  scratch beyond threadgroup-memory limits for the attention merge.

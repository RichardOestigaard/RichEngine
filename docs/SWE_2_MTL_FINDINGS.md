# SWE-2 Metal Findings

Audit of `runtime/metal/`, kernels, and docs — what Metal already does for us
and what the framework still offers.

## Already tapped

- Metal 4 encoders (`MTL4CommandQueue`, `MTL4CommandAllocator`,
  `MTL4ArgumentTable`, `MTL4CommitFeedback`, `MTL4ComputeCommandEncoder`),
  opt-in via `RICHENGINE_MTL4`.
- `metal::tensor` / `matmul2d` neural-accelerator ops in
  `runtime/metal/kernels/common/` tiles.
- simdgroup matrix ops, simd reductions, atomics throughout kernels.
- `MTLResidencySet` for weights + KV.
- Indirect command buffers (`MTLIndirectCommandTypeConcurrentDispatch`).
- `MTLSharedEvent` for CPU↔GPU ordering (CommandGraph event splits).
- Function constants, shared-storage buffers, `newBufferWithBytesNoCopy`.
- Private-framework ANE for the dense prefill FFN split
  (`runtime/ane/`, `runtime/ops/AneFfn*`).
- FP8 paged KV cache.

## High payoff — untapped

### 1. MTLIOCommandQueue (Metal IO) — model loading

Zero hits in repo. Streams files SSD→GPU buffer directly, no CPU memcpy,
with on-the-fly decompression (zlib/lz4/zstd/lzfse). ~15 GB weight loads
get faster and overlap GPU work. Could also stream MoE expert slabs on
demand — experts paged in per request instead of resident.

### 2. MTL4MachineLearningCommandEncoder + MTLTensor

Zero hits. Metal 4 runs a compiled ML network (Metal ML package) as a pass
*inside* an MTL4 command buffer, sharing the argument table and timeline
with compute passes. Two uses:

- Replace/augment the private-API ANE path with a supported one — a
  compiled FFN network dispatched next to GPU kernels, no CPU roundtrip.
- Fused subgraphs (matmul+bias+activation) Apple compiles itself, for
  dense-model prefill.

### 3. MTLHeap — transient scratch aliasing

Zero hits. Vision (`Vision.mm` scratch, 12 tensors) and MoE workspaces are
large short-lived allocations. A heap with
`MTLHazardTrackingModeUntracked` + aliasable resources lets disjoint passes
reuse the same physical pages. On a 36 GB budget that lowers peak RSS.
Also required as `intermediatesHeap` for the ML encoder.

### 4. MTLBinaryArchive — pipeline cache on disk

Zero hits. ~70 pipelines + function-constant variants all compile at
startup. Serialize to a per-version archive file; load once. Pairs with
`device.shouldMaximizeConcurrentCompilation` (also unused).

## Medium payoff

### 5. MTLBlitCommandEncoder

`fillBuffer`/`copyBuffer` for KV page clears, weight staging, RoPE table
fills. Replaces tiny compute dispatches (`row_copy.metal`) with hardware
DMA; blits run concurrent with compute in MTL4's unified
`MTL4CommandEncoder`.

### 6. GPU counters (MTLCounterSampleBuffer + device.counterSets)

Zero hits; `dev/tuning/` does black-box timing only. Timestamp pairs
in-stream give per-dispatch GPU time without commit boundaries. Apple
family counter sets are limited, but timestamps alone sharpen per-kernel
attribution inside one command buffer.

### 7. Runtime shader composition

Zero hits on all of:

- Function stitching (`newLibraryWithStitchedDescriptor`) — compose
  dequant × epilogue at runtime without template explosion.
- `MTLDynamicLibrary` + `MTLLinkedFunctions` — shared kernel code loaded
  once.
- Visible function tables — on-GPU dispatch; e.g., a MoE kernel calls a
  per-expert GEMM via table instead of host-side dispatch per expert.
- Function pointers in MSL (`device.supportsFunctionPointers`, Apple7+).

### 8. simd_permute / simd_rot / simdgroup_int8x8

Apple9+ arbitrary lane permutation; `simd_shuffle_xor` chains appear in
reductions — permute is one op. `simdgroup_int8x8` matrix ops (MSL 3.1+)
worth probing for the int8 KV / linear_i8 path on family-10 parts.

## Smaller / situational

- `MTLFence`/`MTLEvent` — zero hits; ordering is all shared-event splits
  and command-buffer boundaries. Fences order *within* a buffer across
  encoders — could merge some command-buffer splits.
- `MTLTexture` via `CVMetalTextureCache` — zero hits; vision input is
  CPU-preprocessed patch buffers. Texture path gives zero-copy from
  decoded video/HEIC plus hardware format conversion. Minor.
- Shader logging (`metal::os_log`, MTL 4) — print from inside kernels;
  useful for draft-acceptance debugging.
- `MTL4CommandEncoder` unified pass — one encoder does compute+blit+ML;
  fewer pass boundaries than compute-only encoding.

## Probably skip

- MPS/MPSGraph — our `matmul2d` tiles already beat what MPSGraph generates
  for quantized GEMM; useful only as a benchmark oracle.
- MetalFX, ray tracing, sparse textures — wrong domain.
- Mesh shaders — no analog for GEMM workloads.

## Suggested order

1. IOCommandQueue — measurable, self-contained.
2. MTLHeap scratch aliasing — memory headroom.
3. BinaryArchive — startup.
4. ML encoder spike — replaces private ANE dependency long-term.

## Session results (measured, not projected)

### MTL4 ML encoder spike — works, rejected for decode-FFN

Full pipeline runs end to end: coremltools (py3.12 venv — 3.14 lacks its
native modules) → `metal-package-builder -ml` → `MTL4MachineLearningPipelineState`
→ `MTL4MachineLearningCommandEncoder` dispatch in an MTL4 timeline
(`dev/tests/engine/ml_encoder_bench.mm`, kept as a dev tool). Facts learned:

- The compiler **does** attempt ANE placement — fp32 tensor I/O produced
  "Incompatible element type for ANE" diagnostics; the fp16 build compiled
  with no GPU adapters in the package manifest.
- `MTLTensorExtents` index 0 is the innermost (last) dimension — passing
  shape values in row-major order produces "incompatible shape" errors.
- `pipeline.reflection.bindings` can be empty; bind I/O tensors by buffer
  index (input 0, output 1).
- 8×5120→2×13824→5120 fp16 FFN: ~1.02 ms/dispatch (~417 GB/s effective on
  424 MB of baked weights).
- **Verdict: do not use for decode-FFN.** fp16 weights cost 2× the Q4 bytes;
  the ANE reads the same unified DRAM through the same controller, so
  decode's bandwidth wall doesn't move. The prefill ANE split wins because
  prefill is compute-bound (extra TOPS); decode gains nothing — zero-sum at
  best. Toolchain stays viable for compute-bound targets (vision tower,
  prefill), and the private-API ANE path stays for now.

### Verify attention splits — policy change, shipped

Swept `RICHENGINE_VERIFY_SPLITS_MAX` over histories {2K–131K} × lanes {1,4}
on M5 Pro. Optimum is ~32–64 pages per split everywhere — the old floor of
32/cap of 128 over-split at every regime. New policy
`verifyAttentionSplits = clamp(pages/64, 8, 64)` (`PagedAttention.hpp`):

| history | B1 old→new | B4 old→new |
|---|---|---|
| 2K | 0.165→0.111 (−33%) | — |
| 8K | 0.366→0.335 (−8.5%) | 1.300→1.090 (−16%) |
| 32K | 1.131→1.111 (−2%) | 4.710→4.179 (−11%) |
| 131K | 4.288→4.19 (−2.3%) | ~16.8→16.20 (−4%) |

### bf16 attention partials — shipped

Partials on the bus are bf16 (accumulator stays fp32, statistics stay
fp32): ~0.2% per-element error vs the 0.4% fp64 merge gate. fp16 rejected
(unbounded partial magnitudes overflow). Verify partial workspace halves
(~100 MB at B4 max splits). Measured: reduce kernel −40% at 131K
(0.110→0.068 ms), fused −0.7–3%; under the 5–8% projection because partial
writes were already largely L2-absorbed — the bytes saved land on the
reduce read path and arena memory. `q8-flash-attention` passes all gates
(cosine ~0.9999, top-1 ~1.0, bitwise repeat identity).

### In-kernel arrive-last reduce — analyzed, rejected

The merge runs today at `kvHeads×96×lanes` threadgroups; a last-arriver
merge has `kvHeads×passes×lanes` — same bytes, ~96× less parallelism.
Break-even at splits=8, regression at ≥32. The hot path is
`reduce_gate`/`reduce_gather`, so fusion needs ~3 epilogue variants × ~40
split entries to delete one dispatch + one barrier — <1% of token time
upside stacked on the unresolved q8_split flake family. Ticketed
last-R-arrivals restores parallelism but needs polling spin loops (none
exist in the tree). Not worth it.

### Environment findings

- `q8-flash-attention` was failing before this work: the test still
  dispatched a single prefill tile after `TARGET_VERIFY_ROWS` went 8→16 —
  second-tile fused rows stayed poison. Fixed (tiled dispatch + tiled
  partial layout, coverage-bounded poison check). Its earlier failure was
  test-side, not the documented q8_split flake.
- `fp8-flash-attention`/`hd512-attention` fail at pipeline creation on this
  device — kernels request ~34 KB threadgroup vs the 32 KB limit (stale
  binaries fail identically; pre-existing).
- `backend-benchmark` can't load the installed RichEngine packages
  (`draft_proposal_tokens` 7 package vs 15 runtime) and the GGUF path hits
  "physical width accounting mismatch" — bench harness is behind the
  draft-geometry change.

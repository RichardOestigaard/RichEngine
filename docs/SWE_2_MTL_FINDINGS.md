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

## Session results — simd probes, imageblock, _rP splits, OP_TIMINGS (M5 Pro)

### SIMD probe outcomes — measured, mostly dead ends

- `simd_permute`/`simd_rot` do not exist as named ops. The available
  primitives: `simd_shuffle` (dynamic index), `simd_shuffle_rotate_up/down`,
  `simd_shuffle_xor`, `simd_sum`.
- Dynamic-index `simd_shuffle`: **2.8× slower** than the xor chain — dead.
- `simd_sum`: slower than the xor chain — dead.
- `simd_shuffle_rotate_down`: **1.45× faster** than xor (671µs vs 973µs on
  the probe loop) — but valid only for full-32-lane reductions; it wraps
  values across sub-rows, which the attention softmax reductions cross.
  Candidate only where the reduced span is a full simdgroup.
- `simdgroup_matrix<int8_t>` / int8x8: **unsupported** — simdgroup matrix is
  floating-point only. INT8 stays on `metal::tensor` / MPP matmul2d, which
  already covers it.

### MTL4 imageblock — works, exceeds 32 KB scratch

`imageblock<T>` with explicit layout + `data()` compiles and runs under
Metal 4.1. A 48 KB imageblock dispatch at 16×16 succeeded (~196 KB reported
capacity) — an escape for the fp8 256-dim verify tile (34.4/45.8 KB) that
the 32 KB threadgroup limit blocks today.

### MTL4 OP_TIMINGS — shipped

`RICHENGINE_OP_TIMINGS` now works on the Metal 4 path: `commit4` brackets
each encoded unit with a `MTL4CounterHeap` timestamp pair
(`writeTimestampWithGranularity:intoHeap:atIndex:`, precise granularity),
resolved in the final commit's feedback handler. Unlike the MTL3 probe —
one buffer per dispatch, ~138µs/op including commit gaps — MTL4 keeps one
buffer: ~3.9µs/op on the probe graph, same serialized attribution
(timestamps order against prior work). GPU-encoded spans stay direct under
timings, matching commit(); baked span replays time as one unit under the
head pipeline name. Heaps bank on submission parity like allocator4/staging4.

### Verify `_rP` pass splits — measured winners per geometry

The `_r2` idea generalized to `_r<P>`: the fused M-row tile splits its
`Group×VRows` rows across P threadgroups (`group.x / P` selects KV head,
`group.x % P` the pass), cutting scratch P-fold. Emitted for INT8, INT4 and
BF16 wherever the per-pass MP stays a multiple of 8 (the PV matmul's M
constraint). A/B on the q8 harness, width 4, M5 Pro:

| geometry | history | base ms | _r2 ms | _r4 ms | winner |
|---|---:|---:|---:|---:|---|
| kv4/g6 (24/4/d256) | 4K | 1.026 | 0.746 | 0.991 | _r2 1.38× |
| | 32K | 5.434 | 3.998 | 4.880 | _r2 1.36× |
| | 131K | 20.72 | 16.65 | 22.43 | _r2 1.24× |
| kv4/g4 (16/4/d256) | 4K | 0.634 | 0.487 | 0.769 | _r2 1.30× |
| | 32K | 3.214 | 2.550 | 2.575 | _r2 1.26× |
| | 131K | 13.04 | 12.58 | 17.75 | _r2 1.04× |
| kv2/g8 (16/2/d256) | 4K | 0.601 | — | 0.476 | _r4 1.26× |
| | 32K | 3.448 | — | 2.534 | _r4 1.36× |
| | 131K | 12.63 | — | 11.60 | _r4 1.09× |

Outputs identical everywhere except kv4/g6's known 8-element/1-bf16-ULP
case at 4K. `_r4` loses at M≤96 — the extra same-page reads beat the
occupancy gain; the M=128 tiles need it (their _r2 pass still caps at one
group/core). Selected in `splitPipeline` from `fused = group×verifyRows`
(≥128 → 4 passes, else 2), gated by the same MP%8 rule the kernels compile
under. `RICHENGINE_VERIFY_SPLIT_R2_OFF` restores single-group splits;
`VERIFY_SPLIT_R2` is in the tuning sweep for the affected families.
Measured at VRows=16; at VRows=8 the chain is narrow (`_m8`) and the split
is unreachable — the instantiations stay compile-guarded either way.

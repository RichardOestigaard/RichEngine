#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace richengine::ops {

// The compiled Mamba-2 (SSD) kernel set a model's recurrent block binds,
// declared by the family maker. Nemotron64 is Nemotron 3.5's mixer: 64 heads
// of 64 dims in 8 groups of 128 states, a 6144-channel conv, a 10496-wide
// packed projection storage row (10304 logical).
enum class Mamba2Core : uint8_t { Unspecified, Nemotron64, Bolt128 };

// Tensor geometry mapped to a compiled Metal variant during graph
// construction. The packed projection row carves
// [gate innerSize | conv (x | B | C) convolutionDimension | dt heads],
// its storage rows padded up to a 256-element tile: packedWidth is the
// buffers' element stride — the padded tile, never the logical
// innerSize + convolutionDimension + heads. The conv region carves
// [x innerSize | B groups*stateSize | C groups*stateSize]:
// convolutionDimension == innerSize + 2 * groups * stateSize.
struct Mamba2Shape final {
  uint32_t heads = 0;
  uint32_t headDimension = 0;
  uint32_t groups = 0;
  uint32_t stateSize = 0;
  // The causal depthwise conv's tap count; the FIFO state holds
  // convolutionTaps - 1 inputs per channel.
  uint32_t convolutionTaps = 4;
  uint32_t innerSize = 0;
  uint32_t convolutionDimension = 0;
  uint32_t packedWidth = 0;
  // dt' = clamp(softplus(dt + dt_bias), dtMinimum, dtMaximum) — Nemotron's
  // time_step_limit (0.001, 0.1); an unbounded upper limit passes FLT_MAX.
  float dtMinimum = 0.0f;
  float dtMaximum = 0.0f;
  Mamba2Core core = Mamba2Core::Unspecified;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return heads && headDimension && groups && heads % groups == 0 &&
           stateSize && convolutionTaps >= 2 &&
           innerSize == heads * headDimension &&
           convolutionDimension == innerSize + 2 * groups * stateSize &&
           packedWidth == (innerSize + convolutionDimension + heads + 255) /
                              256 * 256 &&
           dtMinimum > 0.0f && dtMaximum > dtMinimum;
  }

  // The group's RMS-normalized width: one group's heads' dims (512 on
  // Nemotron64).
  [[nodiscard]] constexpr uint32_t groupWidth() const noexcept {
    return innerSize / groups;
  }

  bool operator==(const Mamba2Shape &) const = default;
};

// Per-lane state cell byte strides, the same contract as GdnStateStrides:
// every layer's convolution FIFO region first (each (convolutionTaps-1) *
// convolutionDimension bf16, aligned up), then every layer's recurrent
// matrix (heads * headDimension * stateSize fp32, aligned up); a lane's cell
// is convolutionBytes + recurrentBytes bytes. GdnStateLayout carries these
// as {layers, convolutionTaps-1, convolutionDimension, heads, headDimension,
// stateSize}.
struct Mamba2StateStrides final {
  uint64_t convolutionLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convolutionStateBytes = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return convolutionLayerBytes && recurrentLayerBytes &&
           convolutionStateBytes;
  }
};

struct Mamba2PrefillBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  // The conv's per-channel bias (GGUF ssm_conv1d.bias), [convolutionDimension]
  // bf16 — added to the accumulator before the SiLU.
  metal::MetalBuffer convolutionBias;
  metal::MetalBuffer convolutionIn;
  metal::MetalBuffer convolutionOut;
  // The prepare kernel's post-conv x|B|C rows, [tokens][convolutionDimension]
  // bf16 — the scan's operands and the commit layout's mixed form.
  metal::MetalBuffer postConvolution;
  // dt' per (token, head), [tokens][heads] fp32 — the scan recomputes
  // dA = exp(dt' * A[h]).
  metal::MetalBuffer stepSizes;
  // A = -exp(A_log) and D, per head, fp32 [heads].
  metal::MetalBuffer aWeights;
  metal::MetalBuffer skipWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer recurrentIn;
  metal::MetalBuffer recurrentOut;
  // The scan's ungated y rows, [tokens][innerSize] bf16 — the gate's input.
  metal::MetalBuffer recurrentRows;
  NormWeights mixerNorm;
  metal::MetalBuffer hidden;
};

struct Mamba2DecodeBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer convolutionBias;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  // The fused kernel's post-conv x|B|C rows, [verifyStride][convDim] bf16
  // per lane — the commit's operands.
  metal::MetalBuffer mixed;
  metal::MetalBuffer aWeights;
  metal::MetalBuffer timeBias;
  // dt' per (token, head), [verifyStride][heads] fp32 per lane.
  metal::MetalBuffer stepSizes;
  metal::MetalBuffer skipWeights;
  NormWeights mixerNorm;
  metal::MetalBuffer hidden;
  LinearScratch linearScratch{};
  // The lanes' decode tape (folded commit): one buffer, per-lane slots for
  // the previous step's retained rows and this step's — pending_tape and
  // step_tape in the parameters carry the offsets. Empty disables the fold.
  metal::MetalBuffer tape;
  // Bolt's split scan reuses the target's otherwise unused beta scratch for
  // the prologue's fp32 dA values, in stepSizes' [row][head] layout. The
  // fused kernels leave it empty and recompute nothing.
  metal::MetalBuffer decayValues{};
};

// One lane's tape offsets and previous retained count: pending/step are the
// byte offsets of the two parity slots, rows the retained count the next
// step folds in. layerBytes strides each layer's region inside the slot;
// tapeLayerBytes() derives it from the shape.
struct Mamba2TapeLanes final {
  std::array<uint64_t, RICHENGINE_MAXIMUM_BATCH_WIDTH> pending{};
  std::array<uint64_t, RICHENGINE_MAXIMUM_BATCH_WIDTH> step{};
  std::array<uint32_t, RICHENGINE_MAXIMUM_BATCH_WIDTH> rows{};
  uint32_t layerBytes = 0;
};

struct Mamba2CommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer stepSizes;
  metal::MetalBuffer aWeights;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer retainedCounts;
};

class Mamba2 final {
public:
  // `sums`, when present, is written by the gate's *_sums kernel: the
  // out-projection's input sums beside the gated rows, so the mixer output
  // needs no sums pass of its own.
  static void addPrefill(metal::CommandGraph &graph,
                         Mamba2PrefillBuffers buffers, Mamba2Shape shape,
                         uint32_t tokens, metal::MetalBuffer sums = {});
  // Also writes the out-projection's `input` table into
  // buffers.linearScratch when it is not Plain, and throws when the scratch
  // cannot hold it.
  // liveRows bounds each lane's serial verify scan (adaptive proposal
  // budgets); empty scans all RICHENGINE_TARGET_VERIFY_ROWS rows.
  // verifyStride is the lanes' packed row stride: draft.verifyRows() for a
  // trained-block draft — the packed, mixed, step-size and hidden tensors
  // are lane-major at that stride.
  static PreparedInput addDecode(metal::CommandGraph &graph,
                                 Mamba2DecodeBuffers buffers,
                                 Mamba2Shape shape, uint32_t lanes,
                                 uint32_t layer, Mamba2StateStrides state,
                                 LinearInput input,
                                 std::span<const uint32_t> liveRows = {},
                                 uint32_t verifyStride =
                                     RICHENGINE_TARGET_VERIFY_ROWS,
                                 const Mamba2TapeLanes *tapeLanes = nullptr);
  // The tree-verify variants: RICHENGINE_TREE_VERIFY_NODES rows per lane,
  // the node descriptors and emitted counts steering the conv taps and the
  // leaf-on-copy scan. addCommitTree replays the retained path.
  static PreparedInput addDecodeTree(metal::CommandGraph &graph,
                                     Mamba2DecodeBuffers buffers,
                                     metal::MetalBuffer treeNodes,
                                     metal::MetalBuffer treeCounts,
                                     Mamba2Shape shape, uint32_t lanes,
                                     uint32_t layer, Mamba2StateStrides state,
                                     LinearInput input);
  // Replays the retained verify rows of each lane over the incoming cells:
  // the convolution carry rewinds to the retained rows' raw conv inputs and
  // the recurrent matrix recomputes over the emitted post-conv rows and dt'.
  static void addCommit(metal::CommandGraph &graph,
                        Mamba2CommitBuffers buffers, Mamba2Shape shape,
                        uint32_t layers, uint32_t lanes,
                        Mamba2StateStrides state,
                        uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS);
  static void addCommitTree(metal::CommandGraph &graph,
                            Mamba2CommitBuffers buffers,
                            metal::MetalBuffer retainedPath,
                            Mamba2Shape shape, uint32_t layers, uint32_t lanes,
                            Mamba2StateStrides state);
  // One tape slot's per-layer bytes for `rows` verify rows:
  // rows * (2 * convDim * 2 + heads * 4) — the raw conv inputs, the
  // post-conv x|B|C and dt'.
  static constexpr uint64_t tapeLayerBytes(const Mamba2Shape &shape,
                                           uint32_t rows) {
    return uint64_t{rows} * (2 * uint64_t{shape.convolutionDimension} * 2 +
                             uint64_t{shape.heads} * 4);
  }
};

} // namespace richengine::ops

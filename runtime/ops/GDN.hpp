#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace richengine::ops {

// The compiled GDN kernel set a model's recurrent block binds, declared by
// the family maker (TargetModelGeometry::gdnCore). The plan validates the
// shape carries the core's compiled geometry instead of inferring it.
enum class GdnCore : uint8_t { Unspecified, Value48, Value32 };

// Tensor geometry mapped to a compiled Metal variant during graph construction.
struct GdnShape final {
  uint32_t keyHeads = 0;
  uint32_t valueHeads = 0;
  uint32_t headDimension = 0;
  uint32_t convolutionDimension = 0;
  uint32_t packedWidth = 0;
  GdnCore core = GdnCore::Unspecified;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return keyHeads && valueHeads && valueHeads % keyHeads == 0 &&
           headDimension && convolutionDimension && packedWidth &&
           convolutionDimension ==
               (uint64_t{2} * keyHeads + valueHeads) * headDimension &&
           packedWidth >= convolutionDimension +
                              uint64_t{valueHeads} * headDimension +
                              uint64_t{2} * valueHeads;
  }

  bool operator==(const GdnShape &) const = default;
};

// The value-head order of the GDN output, the out_proj input columns. Grouped
// keeps a key head's value heads adjacent; Tiled is llama.cpp's GGUF order,
// value head h at (h % heads per key) * key heads + h / heads per key.
enum class GdnHeadOrder : uint8_t { Grouped, Tiled };

struct GdnStateStrides final {
  uint64_t convolutionLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convolutionStateBytes = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return convolutionLayerBytes && recurrentLayerBytes &&
           convolutionStateBytes;
  }
};

struct GdnPrefillBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer convolutionIn;
  metal::MetalBuffer convolutionOut;
  metal::MetalBuffer queries;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  metal::MetalBuffer recurrentIn;
  metal::MetalBuffer recurrentOut;
  metal::MetalBuffer recurrentRows;
  NormWeights mixerNorm;
  metal::MetalBuffer hidden;
  // WY/UT scratch for the chunkwise-parallel scan (RICHENGINE_GDN_CHUNKED); an
  // empty buffer keeps the serial prefill_gdn_scan.
  metal::MetalBuffer chunkScratch;
};

struct GdnDecodeBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  NormWeights mixerNorm;
  metal::MetalBuffer hidden;
  LinearScratch linearScratch{};
  // The lanes' decode tape (folded commit): one buffer, per-lane slots for
  // the previous step's retained rows and this step's — pending_tape and
  // step_tape in the parameters carry the offsets. Empty disables the fold.
  metal::MetalBuffer tape;
};

// One lane's tape offsets and previous retained count: pending/step are the
// byte offsets of the two parity slots, rows the retained count the next
// step folds in. layerBytes strides each layer's region inside the slot.
struct GdnTapeLanes final {
  std::array<uint64_t, RICHENGINE_MAXIMUM_BATCH_WIDTH> pending{};
  std::array<uint64_t, RICHENGINE_MAXIMUM_BATCH_WIDTH> step{};
  std::array<uint32_t, RICHENGINE_MAXIMUM_BATCH_WIDTH> rows{};
  uint32_t layerBytes = 0;
};

struct GdnCommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer retainedCounts;
};

class GDN final {
public:
  // `sums`, when present, is written by the gate's *_sums kernel: the
  // out-projection's input sums beside the gated rows, so the mixer output
  // needs no sums pass of its own.
  // Floats of WY/UT scratch the chunked scan needs for `tokens` rows
  // (`factor` is the RICHENGINE_GDN_CHUNKED chunk size; 0 disables it).
  static uint64_t chunkScratchFloats(const GdnShape &shape, uint32_t tokens,
                                     uint32_t factor);
  // `flags` are the gate-phase flags of metal/abi/GDN.h beyond the head
  // order (which `order` still carries): GDN_SIGMOID_GATE gates the
  // normalized output by sigmoid(z) instead of silu(z) (qwen4exp).
  static void addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                         GdnShape shape, uint32_t tokens,
                         GdnHeadOrder order, metal::MetalBuffer sums = {},
                         uint32_t flags = 0);
  // Also writes the out-projection's `input` table into
  // buffers.linearScratch when it is not Plain, and throws when the scratch
  // cannot hold it.
  // liveRows bounds each lane's serial verify scan (adaptive proposal
  // budgets); empty scans all RICHENGINE_TARGET_VERIFY_ROWS rows.
  // verifyStride is the lanes' packed row stride: draft.verifyRows() for a
  // trained-block draft — the packed, mixed, decay, beta and hidden tensors
  // are lane-major at that stride.
  static PreparedInput addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                                 GdnShape shape, uint32_t lanes, uint32_t layer,
                                 GdnStateStrides state,
                                 GdnHeadOrder order,
                                 LinearInput input,
                                 std::span<const uint32_t> liveRows = {},
                                 uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS,
                                 const GdnTapeLanes *tapeLanes = nullptr,
                                 uint32_t flags = 0);
  // The tree-verify variants: RICHENGINE_TREE_VERIFY_NODES rows per lane, the
  // node descriptors and emitted counts steering the conv taps and the
  // leaf-on-copy scan. addCommitTree replays the retained path.
  static PreparedInput addDecodeTree(metal::CommandGraph &graph,
                                     GdnDecodeBuffers buffers,
                                     metal::MetalBuffer treeNodes,
                                     metal::MetalBuffer treeCounts,
                                     GdnShape shape, uint32_t lanes,
                                     uint32_t layer, GdnStateStrides state,
                                     GdnHeadOrder order, LinearInput input,
                                     uint32_t flags = 0);
  static void addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                        GdnShape shape, uint32_t layers, uint32_t lanes,
                        GdnStateStrides state,
                        uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS);
  static void addCommitTree(metal::CommandGraph &graph,
                            GdnCommitBuffers buffers,
                            metal::MetalBuffer retainedPath,
                            GdnShape shape, uint32_t layers, uint32_t lanes,
                            GdnStateStrides state);
};

} // namespace richengine::ops

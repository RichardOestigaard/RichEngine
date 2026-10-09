#pragma once

// The decode lane's packed scratch layout and its allocation owner.

#include "model/RuntimeGeometry.hpp"

#include "Checked.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace richengine::model {

enum class DecodeTensor : uint32_t {
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  GdnHidden,
  GdnOutput,
  Intermediate,
  FullPacked,
  FullQueries,
  AttentionPartials,
  AttentionStatistics,
  FullAttention,
  AttentionHidden,
  AttentionOutput,
  Positions,
  DraftPositions,
  RopeCos,
  RopeSin,
  ContextProjected,
  ContextHidden,
  ContextKv,
  CapturedTargetHidden,
  DraftQueryKeys,
  DraftQueryValues,
  DraftRopeCos,
  DraftRopeSin,
  FinalHidden,
  Logits,
  ArgmaxValues,
  ArgmaxIndices,
  // The fused greedy head's per-(selected row, 128-column tile) argmax
  // partials; sized rows * ceil(vocabulary / 128) each. Empty when the model
  // never takes the fused path.
  HeadArgmaxValues,
  HeadArgmaxIndices,
  TargetPartialMasses,
  TargetVocabularyRows,
  TargetVocabularyRanges,
  TargetVocabularyArrivals,
  SamplingUniforms,
  ConstraintMasks,
  OutputTokens,
  RetainedCount,
  AcceptedCount,
  DraftInputTokens,
  DraftHidden0,
  DraftHidden1,
  DraftNormalized,
  DraftDynamic,
  DraftConvolved,
  DraftProposalQkv,
  DraftAttention,
  DraftProjected,
  DraftResidual,
  DraftIntermediate,
  DraftFinalHidden,
  SelectorHidden,
  Candidates,
  Unary,
  TopPartialIds,
  TopPartialValues,
  ProposalProbs,
  ProposedTokens,
  // Verify-tree tables (draft_select_tree): packed node descriptors, node
  // tokens and node counts, RICHENGINE_TREE_VERIFY_NODES-stride per lane.
  TreeNodes,
  TreeTokens,
  TreeCounts,
  // Tree-verify operands (verify_input_tree_tokens/decode_accept_tree): one
  // ancestor bitmask per node, the retained path's DFS row indices, the
  // per-node argmax selections and the path-gathered captured hidden rows.
  TreeMasks,
  RetainedPath,
  TreeSelected,
  CapturedPath,
  // The alternate-geometry layers' rope tables (Gemma's globals).
  RopeCosAlt,
  RopeSinAlt,
  // Gemma's per-lane post-attention residual and the zeroed rows its routed
  // MoE combine adds to (memset once at arena creation).
  GemmaResidual,
  ZeroResidual,
  PageTable,
  // Indexed by state lane, like PageTable: a penalized request's penalty
  // words (ops::Sampling::rebuildPenaltyWords).
  PenaltyState,
  VerifyPackedBase,
  VerifyMixedBase,
  VerifyDecayBase,
  VerifyBetaBase,
  ChunkKeysBase,
  ChunkValuesBase,
  // One tensor per ops::kMoeScratchFields entry, in its order (moeScratchTensor).
  MoeScratch,
  MoeScratchLast = MoeScratch + ops::kMoeScratchFields.size() - 1,
  Count,
};

constexpr uint32_t decodeTensorCount =
    static_cast<uint32_t>(DecodeTensor::Count);

constexpr bool isGdnLayerTensor(DecodeTensor tensor) noexcept {
  return tensor == DecodeTensor::VerifyPackedBase ||
         tensor == DecodeTensor::VerifyMixedBase ||
         tensor == DecodeTensor::VerifyDecayBase ||
         tensor == DecodeTensor::VerifyBetaBase;
}

constexpr bool isAttentionLayerTensor(DecodeTensor tensor) noexcept {
  return tensor == DecodeTensor::ChunkKeysBase ||
         tensor == DecodeTensor::ChunkValuesBase;
}

constexpr bool isLayerMajorTensor(DecodeTensor tensor) noexcept {
  return isGdnLayerTensor(tensor) || isAttentionLayerTensor(tensor);
}

[[nodiscard]] std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators);
// A decode tensor is one packed M32 allocation.  B1/B2/B3/B4 are prefixes
// containing 8/16/24/32 rows. Lanes are never separated by arena-alignment
// holes; only whole tensor boundaries are aligned.
[[nodiscard]] uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                                            const ops::ExecutionPlans &operators);
[[nodiscard]] uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                                          const ops::ExecutionPlans &operators);

class DecodeArena final {
public:
  DecodeArena(metal::MetalBackend &backend, RuntimeGeometry geometry,
               const ops::ExecutionPlans &operators)
      : backend_(backend), geometry_(std::move(geometry)) {
    const uint64_t baseBytes = decodeArenaBaseBytes(geometry_, operators);
    auto sizes = decodeTensorBytes(geometry_, operators);
    base_ = backend_.allocateBuffer(baseBytes, metal::BufferStorage::Shared,
                                    "shared-decode");
    uint64_t cursor = 0;
    for (uint32_t tensor = 0; tensor < sizes.size(); ++tensor) {
      uint64_t stride = sizes[tensor];
      offsets_[tensor] = cursor;
      sizes_[tensor] = sizes[tensor];
      const DecodeTensor kind = static_cast<DecodeTensor>(tensor);
      if (sizes[tensor] && !isLayerMajorTensor(kind)) {
        for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
          tensors_[lane][tensor] = backend_.view(
              base_, cursor + uint64_t{lane} * stride, sizes[tensor]);
        }
      }
      cursor +=
          alignUp(checkedMultiply(stride, kLaneCount, "decode tensor"));
    }
    if (cursor != baseBytes)
      throw std::logic_error("decode arena mismatch");
    // Sampled rows' draws return their arrival counts to zero; they start
    // there.
    const metal::MetalBuffer arrivals =
        packed(DecodeTensor::TargetVocabularyArrivals, kLaneCount);
    std::memset(arrivals.contents(), 0, arrivals.sizeBytes());
    // Gemma's MoE combine residual: zeroed once at arena creation.
    if (const metal::MetalBuffer zero =
            packed(DecodeTensor::ZeroResidual, kLaneCount))
      std::memset(zero.contents(), 0, zero.sizeBytes());

    const uint64_t denseScratchBytes = gateScratchBytes(geometry_, operators);
    if (denseScratchBytes) {
      gateScratch_ = backend_.allocateBuffer(
          denseScratchBytes, metal::BufferStorage::Private, "gate-scratch");
    }
    // Each field exists only when some plan uses it (split-only plans have
    // partials and counters but no activation table).
    const auto linearSize = linearScratchSize(geometry_, operators);
    const auto allocate = [&](uint64_t bytes, metal::BufferStorage storage, const char *label) {
      return bytes ? backend_.allocateBuffer(bytes, storage, label) : metal::MetalBuffer{};
    };
    linearScratch_.input = allocate(linearSize.input, metal::BufferStorage::Private, "q4-input");
    linearScratch_.sums = allocate(linearSize.sums, metal::BufferStorage::Private, "q4-sums");
    linearScratch_.partials =
        allocate(linearSize.partials, metal::BufferStorage::Private, "q4-partials");
    linearScratch_.counters =
        allocate(linearSize.counters, metal::BufferStorage::Shared, "q4-counters");
    linearScratch_.rotated = allocate(linearSize.rotated, metal::BufferStorage::Private, "linear-rotated");
    if (linearSize.counters)
      std::memset(linearScratch_.counters.contents(), 0, linearSize.counters);
    bytes_ = plannedDecodeBytes(geometry_, operators);
  }

  [[nodiscard]] metal::MetalBuffer get(uint32_t lane, DecodeTensor tensor) const {
    if (lane >= kLaneCount)
      throw std::out_of_range("invalid decode lane");
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    return tensors_[lane][static_cast<uint32_t>(tensor)];
  }

  [[nodiscard]] metal::MetalBuffer packed(DecodeTensor tensor, uint32_t lanes) const {
    if (!lanes || lanes > kLaneCount)
      throw std::out_of_range("invalid packed decode width");
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    const uint32_t index = static_cast<uint32_t>(tensor);
    // Model-specific scratch is represented by an empty buffer.  The
    // selected target graph consumes either dense-FFN or MoE tensors, never
    // both, so no dummy allocation is needed for the inactive operator.
    if (!sizes_[index])
      return {};
    return backend_.view(base_, offsets_[index],
                         uint64_t{lanes} * sizes_[index]);
  }

  [[nodiscard]] ops::MoeScratch moeScratch(uint32_t lanes) const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer =
          packed(moeScratchTensor<DecodeTensor>(field), lanes);
    return scratch;
  }

  [[nodiscard]] ops::LinearScratch linearScratch() const { return linearScratch_; }
  static ops::LinearScratchSize linearScratchSize(const RuntimeGeometry &geometry,
                                                 const ops::ExecutionPlans &operators);

  [[nodiscard]] metal::MetalBuffer gateScratch() const { return gateScratch_; }

  static uint64_t gateScratchBytes(const RuntimeGeometry &geometry,
                                  const ops::ExecutionPlans &operators) {
    // One private gate buffer is reused serially by the target dense FFN (when
    // present) and the always-dense DFlash draft. Sparse target FFNs use their
    // own route-major arena tensors, but must not remove the draft's scratch.
    const uint64_t draft = operators.gateUpWorkspace(
        {geometry.draft.intermediateSize, geometry.draft.hiddenSize, ops::WeightLayout::Affine64});
    uint64_t target = 0;
    for (const auto &p : geometry.target.gateUpProjections)
      target = std::max(target, operators.gateUpWorkspace(p));
    return std::max(target, draft);
  }

  [[nodiscard]] metal::MetalBuffer gdnBatchSlice(DecodeTensor base, uint32_t gdnLayer,
                                          uint32_t lanes) const {
    const uint32_t layers = geometry_.target.stateLayout.layers;
    if (!lanes || lanes > kLaneCount || gdnLayer >= layers ||
        !isGdnLayerTensor(base))
      throw std::out_of_range("invalid batched GDN layer");
    return layerBatchSlice(base, layers, gdnLayer, lanes);
  }

  [[nodiscard]] metal::MetalBuffer gdnStorage(DecodeTensor base) const {
    if (!isGdnLayerTensor(base))
      throw std::invalid_argument("tensor is not GDN replay scratch");
    const uint32_t index = static_cast<uint32_t>(base);
    // A target without the tensor's field (an LFM2 target's decay/beta) has
    // a zero-sized entry; the unused binding is an empty buffer.
    if (!sizes_[index]) return {};
    return backend_.view(base_, offsets_[index], sizes_[index] * kLaneCount);
  }

  [[nodiscard]] metal::MetalBuffer attentionBatchSlice(DecodeTensor base,
                                                uint32_t attentionLayer,
                                                uint32_t lanes) const {
    const uint32_t layers = geometry_.target.kvLayout.attentionLayers;
    if (!lanes || lanes > kLaneCount || attentionLayer >= layers ||
        !isAttentionLayerTensor(base)) {
      throw std::out_of_range("invalid batched attention layer");
    }
    return layerBatchSlice(base, layers, attentionLayer, lanes);
  }

  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  // A layer-major tensor holds one stride per lane for each of its `layers`
  // layers, layer by layer; the planner sized it as layers x stride, so the
  // stride is recovered here, not supplied.
  [[nodiscard]] metal::MetalBuffer layerBatchSlice(DecodeTensor base, uint32_t layers,
                                                   uint32_t layer, uint32_t lanes) const {
    const uint32_t index = static_cast<uint32_t>(base);
    if (!sizes_[index]) return {};
    const uint64_t stride = sizes_[index] / layers;
    return backend_.view(base_, offsets_[index] + uint64_t{layer} * kLaneCount * stride,
                         uint64_t{lanes} * stride);
  }

  metal::MetalBackend &backend_;
  RuntimeGeometry geometry_;
  metal::MetalBuffer base_;
  std::array<std::array<metal::MetalBuffer, decodeTensorCount>, kLaneCount> tensors_{};
  std::array<uint64_t, decodeTensorCount> offsets_{};
  std::array<uint64_t, decodeTensorCount> sizes_{};
  metal::MetalBuffer gateScratch_;
  ops::LinearScratch linearScratch_;
  uint64_t bytes_ = 0;
};

} // namespace richengine::model

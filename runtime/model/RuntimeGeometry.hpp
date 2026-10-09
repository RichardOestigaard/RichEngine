#pragma once

// The runtime's model geometry and the scratch-layout helpers both arena
// domains share.

#include "model/DFlashDraft.hpp"
#include "model/Model.hpp"
#include "model/ModelFactory.hpp"
#include "model/ModelTuning.hpp"
#include "model/TargetModel.hpp"
#include "model/WeightLayout.hpp"

#include "metal/abi/Sampling.h"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>

namespace richengine::model {

inline constexpr uint32_t kLaneCount = ExecutionLimits::maximumBatchWidth;
inline constexpr uint32_t kDecodeRows = ExecutionLimits::targetVerifyRows;
// Per-lane sampling uniforms handed to the sampler each cycle, in the
// layout of metal/abi/Sampling.h.
inline constexpr uint32_t kSamplingUniformCount = RICHENGINE_SAMPLING_UNIFORMS;
inline constexpr uint32_t kDraftProposalTokens = ExecutionLimits::draftProposalTokens;
inline constexpr uint32_t kPrefillRows = ExecutionLimits::prefillTokenBudget;
inline constexpr uint32_t kTileRows = kv::kPageTokens;
inline constexpr uint32_t kPackedAttentionRows =
    kPrefillRows + kLaneCount * (kTileRows - 1);
inline constexpr uint32_t kDraftCacheStride = ExecutionLimits::draftRingCapacity;
inline constexpr uint32_t kMaximumPageTableEntries =
    (kv::kMaximumPhysicalTokens + kv::kPageTokens - 1) / kv::kPageTokens;

struct RuntimeGeometry final {
  TargetModelGeometry target;
  DFlashDraftLayout draft;
  // The descriptor's runtime policy, copied here so the model's makers —
  // not type probes on the weight variants — own the switches below.
  ModelTuning tuning;

  [[nodiscard]] static RuntimeGeometry from(const ModelPackage &package,
                                            kv::Format format) {
    RuntimeGeometry result;
    result.target = std::visit(
        [](const auto &weights) { return targetModelGeometry(weights); },
        package.target);
    result.target.kvLayout = package.targetKvLayout(format);
    result.draft = std::visit(
        [](const auto &weights) { return weights.layout; }, package.draft);
    result.tuning = package.descriptor.tuning;
    // A Null draft carries a placeholder layout only: it never encodes, so
    // it needs no hidden/vocabulary agreement with the target. Real drafts
    // still require a valid ring layout and matching sizes.
    const bool nullDraft = result.draft.kind == DraftKind::Null;
    if (!result.target.valid() || !result.draft.stateLayout().valid() ||
        (!nullDraft &&
         (result.target.hiddenSize != result.draft.hiddenSize ||
          result.target.vocabularySize != result.draft.vocabularySize ||
          result.target.capturedHiddenSize() != result.draft.targetHiddenSize))) {
      throw std::invalid_argument("invalid model runtime geometry");
    }
    return result;
  }

  [[nodiscard]] uint32_t draftRotaryPairs() const noexcept {
    return draft.attentionHeadDimension / 2;
  }
  [[nodiscard]] uint32_t maskWords() const noexcept {
    return maskWordsPerToken(target.vocabularySize);
  }
  [[nodiscard]] uint32_t projectionSumsWidth() const noexcept {
    uint32_t maximumInput = std::max(
        {target.hiddenSize, target.attentionWidth,
         target.capturedHiddenSize(), target.ffnScratchWidth(),
         draft.targetHiddenSize, draft.hiddenSize,
         draft.intermediateSize});
    return (maximumInput + kQ4GroupElements - 1) / kQ4GroupElements;
  }
};

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept {
  return elements * sizeof(T);
}

// The arena tensor of MoE scratch field `field` (ops::kMoeScratchFields).
template <class Tensor>
constexpr Tensor moeScratchTensor(size_t field) noexcept {
  return static_cast<Tensor>(static_cast<uint32_t>(Tensor::MoeScratch) + field);
}

} // namespace richengine::model

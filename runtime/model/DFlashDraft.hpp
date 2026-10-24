#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <cstdint>
#include <array>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace richengine::model {

struct DFlashDraftRingLayer final {
  // K is [head][ring_position][dimension].
  metal::MetalBuffer keys;
  // V is [head][dimension][ring_position].
  metal::MetalBuffer values;
};

// Draft-owned persistent context, paired with target state in composite caches.
class DFlashDraftRing final {
public:
  DFlashDraftRing(metal::MetalBackend &backend,
                  std::shared_ptr<StateAllocationTracker> tracker,
                  DraftStateLayout layout,
                  std::string_view label);
  ~DFlashDraftRing();
  DFlashDraftRing(const DFlashDraftRing &) = delete;
  DFlashDraftRing &operator=(const DFlashDraftRing &) = delete;

  [[nodiscard]] const std::vector<DFlashDraftRingLayer> &layers() const noexcept {
    return layers_;
  }

private:
  std::shared_ptr<StateAllocationTracker> tracker_;
  std::vector<DFlashDraftRingLayer> layers_;
  uint64_t actualAllocatedBytes_ = 0;
};

// Which draft architecture a layout describes: DFlash2 (dynamic convolutions,
// candidate-selector codebooks, block-bidirectional attention —
// DFlashTwoDraft.{hpp,cpp}), a plain transformer DFlash draft (no
// convolutions or selector; its sliding attention layers mask the block
// rows causally, its full layer does not) or DSpark (no convolutions or
// selector codebooks; block-bidirectional layers like DFlash2, plus a
// sequential Markov head that corrects each proposal position's logits with
// a low-rank bias of the previously sampled token and a confidence head the
// runtime loads but does not score).
enum class DraftKind : uint8_t { DFlashTwo, DFlashV1, DSpark, Null };

// The kind's user-facing name; Null reads "none" since the n-gram predraft
// still proposes for a draft-less family.
inline constexpr std::string_view draftKindName(DraftKind kind) {
  switch (kind) {
  case DraftKind::DFlashTwo: return "DFlash2";
  case DraftKind::DFlashV1: return "DFlash";
  case DraftKind::DSpark: return "DSpark";
  default: return "none";
  }
}

struct DFlashDraftLayout final {
  DraftKind kind = DraftKind::DFlashTwo;
  // Plain drafts: the bitmask of layers whose current block rows attend
  // causally (sliding_attention layers; bit i is layer i). Zero for DFlash2.
  uint32_t causalLayers = 0;
  uint32_t layers = 5;
  uint32_t hiddenSize = 5120;
  uint32_t vocabularySize = 248320;
  uint32_t dynamicSize = 1280;
  uint32_t qkvSize = 6144;
  uint32_t attentionSize = 4096;
  uint32_t intermediateSize = 17408;
  uint32_t attentionHeadDimension = 128;
  float rotaryTheta = 10'000'000.0F;
  uint32_t targetHiddenSize = 25600;
  uint32_t selectorRank = 256;
  uint32_t kvHeads = 8;
  // DSpark only: the Markov head's inner rank (markov_w1/markov_w2 rows of
  // vocabularySize x markovRank), the config's declared block_size (the
  // runtime dispatches the draft's verifyRows query rows — the trained
  // block's own span) and whether the rotary is the
  // interleaved (GPT-J) pairing rather than the half-split NeoX one.
  uint32_t markovRank = 0;
  uint32_t blockSize = 0;
  uint32_t ropeInterleaved = 0;
  // The trained attention window (the draft config's sliding_window): how far
  // back attention reads. The ring's physical capacity is
  // ExecutionLimits::draftRingCapacity; the effective window is clamped to it.
  uint32_t slidingWindow = ExecutionLimits::draftContextTokens;
  // The norms' epsilon: 1e-6 unless the family declares another (LFM2's
  // 1e-5, which also selects the _e5 draft kernels).
  float rmsEpsilon = 1e-6F;
  // The shared vocabulary head's logit softcap (Gemma's 30); 0 disables.
  float logitSoftcap = 0.0F;
  // The compiled kernels the draft binds (ops/DraftAttention.hpp). Makers
  // declare the pair; an Unspecified core or a shape that does not carry
  // the core's compiled geometry is rejected when the draft plans.
  ops::DraftHeadCore headCore = ops::DraftHeadCore::Unspecified;
  ops::DraftContextCore contextCore = ops::DraftContextCore::Unspecified;
  // DFlash2 only: the selector walks the whole 128-slot shard pool instead
  // of the merged top-16 when the batch has no sampled lanes (the residual
  // draw stages only kDraftCandidates). Off by default: the walk's argmax
  // was never the rejected position's target pick on the 35B-A3B decode
  // benchmark, and the wider edge table costs ~7% of decode throughput.
  bool poolSelector = false;

  // The trained block's usable proposal count. A plain or DFlash2
  // block_size counts the block's query rows (the anchor among them);
  // DSpark's counts the emitted proposals. The runtime dispatches the
  // draft's verifyRows — a shorter block's tail never runs.
  [[nodiscard]] constexpr uint32_t proposalLimit() const noexcept {
    if (kind == DraftKind::Null || !blockSize)
      return ExecutionLimits::draftProposalTokens;
    const uint32_t trained =
        kind == DraftKind::DSpark ? blockSize : blockSize - 1;
    return std::min(trained, ExecutionLimits::draftProposalTokens);
  }

  // The chain verify's packed lane stride: the anchor plus the trained
  // block's proposals, rounded up to the decode kernels' eight-row tile —
  // a block-8 draft verifies eight rows per lane instead of the compiled
  // sixteen, halving the verify projections' fused-M work. A deeper block
  // keeps sixteen.
  [[nodiscard]] constexpr uint32_t verifyRows() const noexcept {
    // FIXME(regression): the trained-block verify stride (block-8 → 8 rows)
    // is disabled — with it the draft's proposals stop matching after the
    // first committed step and acceptance collapses to ~44% with a wrong
    // output hash (1177270970566012317, 27B DFlash2). Audited-clean
    // surfaces: draft.metal params.query_rows everywhere (CurrentN=16 and
    // the RopeLaneStride=16 are masked-benign/host-fill-matched), arena
    // slot sizes, positions (16 stride matches rope build), verify_input
    // (separate verify/draft strides), context commit, retained readback.
    // Residual: draft_attention_split_impl's partials offset (queries +
    // lanes * Rows * Attention) and the queries region allocation — needs
    // a runtime buffer dump to pin the remaining 16-vs-8 stride mismatch.
    return ExecutionLimits::targetVerifyRows;
  }

  // The draft forward packs at the verify stride: upstream runs exactly
  // the trained block's rows, and dead rows risk leaking state.
  [[nodiscard]] constexpr uint32_t queryRows() const noexcept {
    return verifyRows();
  }

  [[nodiscard]] constexpr uint32_t draftWindow() const noexcept {
    return slidingWindow < ExecutionLimits::draftRingCapacity
               ? slidingWindow
               : ExecutionLimits::draftRingCapacity;
  }
  [[nodiscard]] constexpr DraftStateLayout stateLayout() const noexcept {
    return {layers, kvHeads, attentionHeadDimension, draftWindow()};
  }
  [[nodiscard]] constexpr ops::DraftAttentionShape attentionShape() const noexcept {
    return {hiddenSize, dynamicSize, qkvSize, attentionSize,
            attentionSize / attentionHeadDimension, kvHeads,
            attentionHeadDimension, ropeInterleaved, draftWindow(),
            headCore, contextCore};
  }
  // The key and value columns of the fused QKV projection, all the context
  // writers read.
  [[nodiscard]] constexpr uint32_t contextKvSize() const noexcept {
    return qkvSize - attentionSize;
  }

  bool operator==(const DFlashDraftLayout &) const = default;
};

struct DFlashDecodeBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer dynamic;
  metal::MetalBuffer convolved;
  metal::MetalBuffer proposalQkv;
  metal::MetalBuffer attention;
  metal::MetalBuffer projected;
  metal::MetalBuffer residual;
  metal::MetalBuffer intermediate;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer queryKeys;
  metal::MetalBuffer queryValues;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer gateScratch;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashContextBuffers final {
  ops::LinearScratch linearScratch{};
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer contextKv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer retainedCounts;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentKeys;
  std::vector<std::array<metal::MetalBuffer,
                         ExecutionLimits::maximumBatchWidth>> persistentValues;
};

struct DFlashPrefillSpan final {
  uint32_t compactRow = 0;
  uint32_t rows = 0;
  uint32_t startPosition = 0;
  std::span<const DFlashDraftRingLayer> ring;
};

struct DFlashPrefillBuffers final {
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer projected;
  metal::MetalBuffer hidden;
  metal::MetalBuffer contextKv;
  metal::MetalBuffer contextQkv;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  // RICHENGINE_PREFILL_FAST_INT8 operand buffers (empty when the flag is off).
  ops::LinearScratch linearScratch{};
};

} // namespace richengine::model

#pragma once

// The maxwell-moe internal MTP draft (blk.45 + nextn tensors of the target
// GGUF): one hyper-connected decoder layer over the target's final
// multistream residual, run fifteen times chained, each pass emitting one
// proposal through the target's shared vocabulary head. Its persistent
// state is a standard draft K/V ring of one layer whose slot s holds the
// normed+rotated key and value of the pair (h_s, x_{s+1}).

#include "model/DFlashDraft.hpp"
#include "model/GgufTarget.hpp"
#include "model/MaxwellMoe.hpp"
#include "ops/Linear.hpp"
#include "ops/Mhc.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <vector>

namespace richengine::model {

// The MTP block's weights: the nextn norms and fused eh projection, the
// dense layer's attention (GQA 32x4x128, no indexer), both mHC modules,
// the sigmoid MoE with its ungated shared expert, the nextn HC head and
// the layer output norm — plus a copy of the target's token table (the
// vocabulary head arrives as the draft call's parameter). The table is a
// copy, not a reference: the package's target variant relocates on the
// package's move, which would dangle a pointer into it.
struct MaxwellMtpDraftWeights final {
  DFlashDraftLayout layout;
  ops::NormWeights enorm;
  ops::NormWeights hnorm;
  ops::Projection ehProjection;
  ops::NormWeights inputNorm;
  ops::Projection queryProjection;
  ops::Projection keyProjection;
  ops::Projection valueProjection;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::Projection outputProjection;
  ops::NormWeights ffnNorm;
  MaxwellMhcWeights attentionMhc;
  ops::MoeWeights moe;
  MaxwellSharedFfn sharedFfn;
  MaxwellMhcWeights ffnMhc;
  MaxwellHcHeadWeights hcHead;
  ops::NormWeights outputNorm;
  ops::EmbeddingWeights tokenEmbedding;
  ops::MoeShape moeShape{};
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

// Reads the "mtp.bin" image's sections in planMtpImage's order; copies the
// target's token table out of `target`.
[[nodiscard]] MaxwellMtpDraftWeights
loadMaxwellMtpDraftWeights(metal::MetalBackend &backend,
                           const DFlashDraftLayout &layout, WeightFile mtp,
                           const MaxwellMoeWeights &target,
                           uint64_t allocationBaseline);

class MaxwellMtpDraft final {
public:
  MaxwellMtpDraft(MaxwellMtpDraftWeights weights,
                  metal::MetalBackend &backend,
                  const ops::ExecutionPlans &operators);
  [[nodiscard]] const MaxwellMtpDraftWeights &weights() const noexcept {
    return weights_;
  }

  // The chained passes: token gather and normed embedding, the eh rows
  // (pass 0's streams from the lane anchor, later passes the chained fp32
  // state), the eh projection, the attention mHC module, the normed and
  // rotated K/V ring write, the GQA flash, the out projection,
  // expand-mix, the FFN mHC module, the shared expert beside the routed
  // MoE, expand-mix, the HC head contraction, the layer output norm and
  // the shared vocabulary head — pass k's logits in the pass-major rows
  // [k * lanes, k * lanes + lanes). A nonempty passLimits caps each lane's
  // chain depth: the verify budget already truncates offers past a lane's
  // acceptance EWMA, and deeper passes' stale proposal slots are never
  // scored, so the draft skips their whole layer of compute too.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths,
                 std::span<const uint32_t> passLimits = {}) const;
  // The pass-major top-16 shard pass, then the plain chain select over its
  // lane-major partials (the MTP draft emits a chain; treeMask is unused).
  void addSelection(metal::CommandGraph &graph,
                    const ops::DraftSelectorBuffers &buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    uint32_t treeMask) const;
  // The retained rows' K/V commit: row r commits slot
  // startPosition[lane] + r the pair (captured row r, output token r)
  // through the eh projection and attention mHC module, and refreshes the
  // lane's pass-0 anchor from its last retained capture row.
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions) const;
  // Prompt context in kMtpPrefillChunk-row chunks: each captured row's
  // pair (the row's streams, the rolled prompt token) into its ring slot;
  // a tail span's last row also lands on the state lane's stream anchor.
  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t capturedRows,
                         std::span<const DFlashPrefillSpan> spans) const;

private:
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  MaxwellMtpDraftWeights weights_;
  ops::MhcShape mhcShape_{};
};

} // namespace richengine::model

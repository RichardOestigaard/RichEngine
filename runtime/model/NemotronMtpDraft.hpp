#pragma once

// The nemotron internal MTP draft (NemotronHMtpModel, the nextn block of a
// Nemotron-3.5-Lightning checkpoint): one NoPE GQA + ungated relu-squared
// MoE decoder layer over the trunk's captured 2688-wide hidden, run
// chained once per proposal, each pass emitting one token through the
// target's shared vocabulary head. Its persistent state is a standard
// draft K/V ring of one layer whose slot s holds the key and value of the
// pair (h_s, x_{s+1}). The block's projections ride F32 float segments —
// hidden 2688 is off every quantized tile — and its sigmoid MoE the
// dedicated nmtp_moe_* kernels (kernels/decode/nmtp.metal).

#include "model/DFlashDraft.hpp"
#include "model/TargetModel.hpp"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <variant>

namespace richengine::model {

class NemotronMtpCheckpointLoader;

// The draft's packed mtp.bin (a packaged draft/mtp.bin) or the checkpoint
// loader that writes it from root/draft's safetensors.
struct PackedNemotronMtpFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const DFlashDraftLayout &layout;
  [[nodiscard]] WeightFile mtp() const;
};

using NemotronMtpDraftFiles =
    std::variant<std::reference_wrapper<NemotronMtpCheckpointLoader>,
                 PackedNemotronMtpFiles>;

inline constexpr std::string_view kNemotronMtpDraftMagic = "MDFM0008";

// The MTP block's weights: the nextn norms and fused eh projection, the
// NoPE GQA attention, the sigmoid router and its selection bias, the
// bf16 expert and shared-expert slabs the nmtp_moe_* kernels read, the
// block output and shared-head norms — plus a copy of the target's token
// table (the vocabulary head arrives as the draft call's parameter). The
// table is a copy, not a reference, as the maxwell MTP draft keeps it.
struct NemotronMtpDraftWeights final {
  DFlashDraftLayout layout;
  ops::NormWeights enorm;
  ops::NormWeights hnorm;
  ops::Projection ehProjection;
  ops::NormWeights inputNorm;
  ops::Projection queryProjection;
  ops::Projection keyProjection;
  ops::Projection valueProjection;
  ops::Projection outputProjection;
  ops::NormWeights ffnNorm;
  metal::MetalBuffer router;
  metal::MetalBuffer expertBias;
  metal::MetalBuffer expertUp;
  metal::MetalBuffer expertDown;
  metal::MetalBuffer sharedUp;
  metal::MetalBuffer sharedDown;
  ops::NormWeights outputNorm;
  ops::NormWeights headNorm;
  ops::EmbeddingWeights tokenEmbedding;
  // The router's norm_topk_prob rescaling (the target layout's
  // moeRoutedScale — Nemotron's 2.5; zero reads as 1).
  float moeRoutedScale = 1.0F;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

// Reads the "mtp.bin" image's sections in nemotronMtpImage's order
// (model/DraftCheckpoint.cpp); copies the target's token table out of
// `target`.
[[nodiscard]] NemotronMtpDraftWeights
loadNemotronMtpDraftWeights(metal::MetalBackend &backend,
                            const NemotronMtpDraftFiles &files,
                            const DFlashDraftLayout &layout,
                            const TargetModelWeights &target);

class NemotronMtpDraft final {
public:
  NemotronMtpDraft(NemotronMtpDraftWeights weights,
                   metal::MetalBackend &backend,
                   const ops::ExecutionPlans &operators);
  [[nodiscard]] const NemotronMtpDraftWeights &weights() const noexcept {
    return weights_;
  }

  // The chained passes: token gather and normed embedding, the eh row
  // (pass 0's hidden from the lane anchor, later passes the chained fp32
  // state), the eh projection, the normed K/V ring write, the NoPE GQA
  // flash, the out projection and residual add, the normed input's MoE
  // (route, relu-squared up, down, weighted combine) and its residual
  // add, the chain state store, the output and shared-head norms and the
  // shared vocabulary head — pass k's logits in the pass-major rows
  // [k * lanes, k * lanes + lanes). passLimits is as MaxwellMtpDraft's.
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths,
                 std::span<const uint32_t> passLimits = {}) const;
  // The pass-major top-16 shard pass, then the plain chain select over
  // its lane-major partials (the MTP draft emits a chain; treeMask is
  // unused).
  void addSelection(metal::CommandGraph &graph,
                    const ops::DraftSelectorBuffers &buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    uint32_t treeMask) const;
  // The retained rows' K/V commit: row r commits slot
  // startPosition[lane] + r the pair (captured row r, output token r)
  // through the eh projection and the block's input norm, and refreshes
  // the lane's pass-0 anchor from its last retained capture row.
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions) const;
  // Prompt context in kMtpPrefillChunk-row chunks: each captured row's
  // pair (the row's hidden, the rolled prompt token) into its ring slot;
  // a tail span's last row also lands on the state lane's anchor.
  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t capturedRows,
                         std::span<const DFlashPrefillSpan> spans) const;

private:
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  NemotronMtpDraftWeights weights_;
};

} // namespace richengine::model

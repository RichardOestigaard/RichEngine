#pragma once

// MiniCPM-V 4.7 35B-A3B (openbmb/MiniCPM-V-4.7-35B-A3B): the qwen3_5_moe
// hybrid — 30 GDN layers with a full-attention layer every fourth — over 256
// routed experts plus a shared one. Unlike the Ornith/Qwen3.6 exports its
// checkpoint stores model.language_model.* tensors with the routed experts'
// gate and up projections fused in experts.gate_up_proj, the vocabulary is
// 248144 rows (the sections pad it to a 256-row tile) and its DSpark draft
// captures five hidden states — 3, 11, 19, 27 and 35.

#include "Qwen3_6Moe.hpp"
#include "QwenHybridLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

struct DFlashDraftLayout;

struct MiniCpmV47MoeLayout final : QwenHybridLayout<5> {
  static constexpr std::string_view layerMagic = "MCPM0001";
  static constexpr std::string_view headMagic = "MCPM0002";
  static constexpr FfnKind ffnKind = FfnKind::SparseMoe;

  uint32_t experts = 256;
  uint32_t expertsPerToken = 8;
  uint32_t expertIntermediateSize = 512;
  // The checkpoint's fused routed gate|up block: gate's 512 rows are the
  // first half of each expert's 1024-row slab, up the second.
  uint32_t fusedGateUpRows = 2 * expertIntermediateSize;

  constexpr MiniCpmV47MoeLayout()
      : QwenHybridLayout{.layers = 40,
                         .hiddenSize = 2048,
                         .vocabularySize = 248144,
                         .packedGdnWidth = 12544,
                         .packedFullWidth = 9216,
                         .convolutionDimension = 8192,
                         .gdnKeyHeads = 16,
                         .gdnValueHeads = 32,
                         .gdnHeadDimension = 128,
                         .attentionWidth = 4096,
                         .attentionQueryHeads = 16,
                         .attentionKvHeads = 2,
                         .attentionHeadDimension = 256,
                         .rotaryPairs = 32,
                         .rotaryTheta = 10'000'000.0F,
                         .fullAttentionPeriod = 4,
                         .maskToken = 248076,
                         .stopTokens = {248044, 248046},
                         .hiddenCaptureLayers = {3, 11, 19, 27, 35},
                         .attentionCore = kv::Core::Kv2Group8} {}
  bool operator==(const MiniCpmV47MoeLayout &) const = default;
};

using MiniCpmV47MoeWeights =
    TargetModelWeights<MiniCpmV47MoeLayout, Qwen3_6MoeLayerWeights>;

[[nodiscard]] MiniCpmV47MoeWeights
loadMiniCpmV47MoeWeights(metal::MetalBackend &backend, MiniCpmV47MoeLayout layout,
                         const TargetFiles<MiniCpmV47MoeLayout> &files);

// The DSpark draft of MiniCPM-V 4.7 (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark):
// five full-attention layers over hidden 2048, eight query heads over two KV
// heads of dimension 256 — a geometry the plain drafts do not share — reading
// the target's five capture layers, with a rank-256 Markov head over the
// target's 248144-row vocabulary and a 15-row trained block.
[[nodiscard]] DFlashDraftLayout minicpmV47DraftLayout();

} // namespace richengine::model

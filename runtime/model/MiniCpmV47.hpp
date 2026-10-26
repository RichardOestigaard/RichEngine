#pragma once

// MiniCPM-V 4.7 35B-A3B (openbmb/MiniCPM-V-4.7-35B-A3B): the qwen3_5_moe
// hybrid — 30 GDN layers with a full-attention layer every fourth — over 256
// routed experts plus a shared one. Unlike the Ornith/Qwen3.6 exports its
// checkpoint stores model.language_model.* tensors with the routed experts'
// gate and up projections fused in experts.gate_up_proj, the vocabulary is
// 248144 rows (the sections pad it to a 256-row tile) and its DSpark draft
// captures five hidden states — 3, 11, 19, 27 and 35.

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

struct DFlashDraftLayout;

// The MiniCPM-V 4.7 target's canonical layout. Its checkpoint's fused routed
// gate|up block — gate's 512 rows the first half of each expert's 1024-row
// slab, up the second — spans 2 * expertIntermediateSize rows where the
// other families keep separate tensors; the file planners derive that
// width from expertIntermediateSize.
[[nodiscard]] constexpr TargetLayout minicpmV47MoeTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "minicpm_v47_moe";
  layout.layerMagic = "MCPM0001";
  layout.headMagic = "MCPM0002";
  layout.ffnKind = FfnKind::SparseMoe;
  // The packed QKV rows interleave a gate row per query head ([q|gate]).
  layout.attentionQueryStride = 2;
  // The per-head RMS norms on the queries and keys.
  layout.attentionQkNorm = true;
  layout.packedOnly = false;
  layout.convolutionTaps = kGdnConvolutionTaps;
  layout.maximumContextTokens = kv::kMaximumLogicalTokens;
  layout.layers = 40;
  layout.hiddenSize = 2048;
  layout.vocabularySize = 248144;
  layout.packedGdnWidth = 12544;
  layout.packedFullWidth = 9216;
  layout.convolutionDimension = 8192;
  layout.gdnKeyHeads = 16;
  layout.gdnValueHeads = 32;
  layout.gdnHeadDimension = 128;
  layout.attentionWidth = 4096;
  layout.attentionQueryHeads = 16;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 256;
  layout.rotaryPairs = 32;
  layout.rotaryTheta = 10'000'000.0F;
  layout.fullAttentionPeriod = 4;
  layout.maskToken = 248076;
  layout.stopTokens = {248044, 248046};
  layout.hiddenCaptureLayers = {3, 11, 19, 27, 35};
  layout.captureLayerCount = 5;
  layout.attentionCore = kv::Core::Kv2Group8;
  layout.experts = 256;
  layout.expertsPerToken = 8;
  layout.expertIntermediateSize = 512;
  // The compiled GDN kernel set the recurrent blocks bind (ops/GDN.hpp).
  layout.gdnCore = ops::GdnCore::Value32;
  layout.moeSharedExpert = true;
  return layout;
}

using MiniCpmV47MoeWeights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadMiniCpmV47MoeWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                         const TargetFiles &files);

// The DSpark draft of MiniCPM-V 4.7 (openbmb/MiniCPM-V-4.7-35B-A3B-DSpark):
// five full-attention layers over hidden 2048, eight query heads over two KV
// heads of dimension 256 — a geometry the plain drafts do not share — reading
// the target's five capture layers, with a rank-256 Markov head over the
// target's 248144-row vocabulary and a 15-row trained block.
[[nodiscard]] DFlashDraftLayout minicpmV47DraftLayout();

} // namespace richengine::model

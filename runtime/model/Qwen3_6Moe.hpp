#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// The Qwen3.6-MoE target's canonical layout: a qwen3_5_moe hybrid of 40
// layers over hidden 2048, a full-attention layer every fourth, over 256
// routed experts plus a shared one. Its packed files carry the family's
// MDFM magics.
[[nodiscard]] constexpr TargetLayout qwen36MoeTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "qwen3_6moe";
  layout.layerMagic = "MDFM0001";
  layout.headMagic = "MDFM0002";
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
  layout.vocabularySize = 248320;
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
  layout.maskToken = 248077;
  layout.stopTokens = {248044, 248046};
  layout.hiddenCaptureLayers = {1, 6, 11, 16, 22, 27, 32, 37};
  layout.captureLayerCount = 8;
  layout.attentionCore = kv::Core::Kv2Group8;
  layout.experts = 256;
  layout.expertsPerToken = 8;
  layout.expertIntermediateSize = 512;
  // The compiled GDN kernel set the recurrent blocks bind (ops/GDN.hpp).
  layout.gdnCore = ops::GdnCore::Value32;
  layout.moeSharedExpert = true;
  return layout;
}

using Qwen3_6MoeWeights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                      const TargetFiles &files);

} // namespace richengine::model

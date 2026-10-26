#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// The Qwen3.8 target's canonical layout: a dense qwen3_5_text hybrid of 64
// layers over hidden 5120, 48 GDN value heads and the DFlash2 draft's five
// capture layers. Its packed files share the dense family's magics.
[[nodiscard]] constexpr TargetLayout qwen38TargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "qwen3_8";
  layout.layerMagic = "MDFL0006";
  layout.headMagic = "MDFL0002";
  layout.ffnKind = FfnKind::Dense;
  // The packed QKV rows interleave a gate row per query head ([q|gate]).
  layout.attentionQueryStride = 2;
  // The per-head RMS norms on the queries and keys.
  layout.attentionQkNorm = true;
  layout.packedOnly = false;
  layout.convolutionTaps = kGdnConvolutionTaps;
  layout.maximumContextTokens = kv::kMaximumLogicalTokens;
  layout.layers = 64;
  layout.hiddenSize = 5120;
  layout.vocabularySize = 248320;
  layout.packedGdnWidth = 16640;
  layout.packedFullWidth = 14336;
  layout.convolutionDimension = 10240;
  layout.gdnKeyHeads = 16;
  layout.gdnValueHeads = 48;
  layout.gdnHeadDimension = 128;
  layout.attentionWidth = 6144;
  layout.attentionQueryHeads = 24;
  layout.attentionKvHeads = 4;
  layout.attentionHeadDimension = 256;
  layout.rotaryPairs = 32;
  layout.rotaryTheta = 10'000'000.0F;
  layout.fullAttentionPeriod = 4;
  layout.maskToken = 248070;
  layout.stopTokens = {248044, 248046};
  layout.hiddenCaptureLayers = {5, 19, 33, 47, 61};
  layout.captureLayerCount = 5;
  layout.attentionCore = kv::Core::Kv4Group6;
  layout.intermediateSize = 17408;
  // The compiled GDN kernel set the recurrent blocks bind (ops/GDN.hpp).
  layout.gdnCore = ops::GdnCore::Value48;
  layout.moeSharedExpert = true;
  return layout;
}

using Qwen3_8Weights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadQwen3_8Weights(metal::MetalBackend &backend, const TargetLayout &layout,
                   const TargetFiles &files);

} // namespace richengine::model

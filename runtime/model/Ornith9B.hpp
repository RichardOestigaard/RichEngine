#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// Ornith 1.5 9B: a dense qwen3_5_text hybrid like the 27B, with 32 layers of
// hidden 4096, 32 GDN value heads and the DFlash2 draft's eight capture
// layers. Its packed files share the dense family's magics.
[[nodiscard]] constexpr TargetLayout ornith9TargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "ornith9";
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
  layout.layers = 32;
  layout.hiddenSize = 4096;
  layout.vocabularySize = 248320;
  layout.packedGdnWidth = 12544;
  layout.packedFullWidth = 10240;
  layout.convolutionDimension = 8192;
  layout.gdnKeyHeads = 16;
  layout.gdnValueHeads = 32;
  layout.gdnHeadDimension = 128;
  layout.attentionWidth = 4096;
  layout.attentionQueryHeads = 16;
  layout.attentionKvHeads = 4;
  layout.attentionHeadDimension = 256;
  layout.rotaryPairs = 32;
  layout.rotaryTheta = 10'000'000.0F;
  layout.fullAttentionPeriod = 4;
  layout.maskToken = 248077;
  layout.stopTokens = {248044, 248046};
  layout.hiddenCaptureLayers = {1, 5, 9, 13, 17, 21, 25, 29};
  layout.captureLayerCount = 8;
  layout.attentionCore = kv::Core::Kv4Group4;
  layout.intermediateSize = 12288;
  // The compiled GDN kernel set the recurrent blocks bind (ops/GDN.hpp).
  layout.gdnCore = ops::GdnCore::Value32;
  layout.moeSharedExpert = true;
  return layout;
}

// The dense FFN reads the same per-layer tensors as the 27B.
using Ornith9BWeights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadOrnith9BWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                    const TargetFiles &files);

} // namespace richengine::model

#pragma once

#include "Qwen3_8.hpp"
#include "QwenHybridLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// Ornith 1.5 9B: a dense qwen3_5_text hybrid like the 27B, with 32 layers of
// hidden 4096, 32 GDN value heads and the DFlash2 draft's eight capture
// layers. Its packed files share the dense family's magics.
struct Ornith9BLayout final : QwenHybridLayout<8> {
  static constexpr std::string_view layerMagic = "MDFL0006";
  static constexpr std::string_view headMagic = "MDFL0002";
  static constexpr FfnKind ffnKind = FfnKind::Dense;

  uint32_t intermediateSize = 12288;

  constexpr Ornith9BLayout()
      : QwenHybridLayout{.layers = 32,
                         .hiddenSize = 4096,
                         .vocabularySize = 248320,
                         .packedGdnWidth = 12544,
                         .packedFullWidth = 10240,
                         .convolutionDimension = 8192,
                         .gdnKeyHeads = 16,
                         .gdnValueHeads = 32,
                         .gdnHeadDimension = 128,
                         .attentionWidth = 4096,
                         .attentionQueryHeads = 16,
                         .attentionKvHeads = 4,
                         .attentionHeadDimension = 256,
                         .rotaryPairs = 32,
                         .rotaryTheta = 10'000'000.0F,
                         .fullAttentionPeriod = 4,
                         .maskToken = 248077,
                         .stopTokens = {248044, 248046},
                         .hiddenCaptureLayers = {1, 5, 9, 13, 17, 21, 25, 29}} {}
  bool operator==(const Ornith9BLayout &) const = default;
};

// The dense FFN reads the same per-layer tensors as the 27B.
using Ornith9BWeights = TargetModelWeights<Ornith9BLayout, DenseLayerWeights>;

[[nodiscard]] Ornith9BWeights
loadOrnith9BWeights(metal::MetalBackend &backend, Ornith9BLayout layout,
                    const TargetFiles<Ornith9BLayout> &files);

} // namespace richengine::model

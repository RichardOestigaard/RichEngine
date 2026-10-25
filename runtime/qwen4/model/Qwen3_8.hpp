#pragma once

#include "qwen4/model/QwenHybridLayout.hpp"
#include "qwen4/model/QwenTarget.hpp"
#include "qwen4/model/QwenTargetFiles.hpp"
#include "qwen4/metal/abi/RoPE.h"
#include "qwen4/ops/Linear.hpp"
#include "qwen4/ops/Normalization.hpp"
#include "qwen4/ops/Vision.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct Qwen3_8Layout final : QwenHybridLayout {
  // Its config's text model type and its GGUFs' architecture; a safetensors
  // checkpoint of it loads too.
  static constexpr std::string_view modelType = "qwen3_5_text";
  static constexpr std::string_view ggufArchitecture = "qwen35";
  static constexpr bool loadsSafetensors = true;

  // Qwen3.8-27B's, its first family (kFamilies).
  constexpr Qwen3_8Layout()
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .yarnContextTokens = 1'010'000,
                          .layers = 64,
                          .hiddenSize = 5120,
                          .vocabularySize = 248320,
                          .packedGdnWidth = 16640,
                          .packedFullWidth = 14336,
                          .convolutionDimension = 10240,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = 48,
                          .gdnHeadDimension = 128,
                          .attentionWidth = 6144,
                          .attentionQueryHeads = 24,
                          .attentionKvHeads = 4,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248070,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::Dense,
                          .intermediateSize = 17408},
                         "Qwen3.8-27B"} {}
  // A family of the architecture.
  constexpr explicit Qwen3_8Layout(const QwenHybridLayout &family) : QwenHybridLayout(family) {}
  bool operator==(const Qwen3_8Layout &) const = default;
};

struct Qwen3_8LayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

using Qwen3_8Weights = QwenTargetWeights<Qwen3_8Layout, Qwen3_8LayerWeights>;

[[nodiscard]] Qwen3_8Weights
loadQwen3_8Weights(metal::MetalBackend &backend, Qwen3_8Layout layout,
                   const QwenTargetFiles &files);

} // namespace splash::model

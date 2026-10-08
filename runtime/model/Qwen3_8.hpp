#pragma once

#include "QwenHybridLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

struct Qwen3_8Layout final : QwenHybridLayout<5> {
  static constexpr std::string_view layerMagic = "MDFL0006";
  static constexpr std::string_view headMagic = "MDFL0002";
  static constexpr FfnKind ffnKind = FfnKind::Dense;

  uint32_t intermediateSize = 17408;

  constexpr Qwen3_8Layout()
      : QwenHybridLayout{.layers = 64,
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
                         .rotaryPairs = 32,
                         .rotaryTheta = 10'000'000.0F,
                         .fullAttentionPeriod = 4,
                         .maskToken = 248070,
                         .stopTokens = {248044, 248046},
                         .hiddenCaptureLayers = {5, 19, 33, 47, 61}} {}
  bool operator==(const Qwen3_8Layout &) const = default;
};

struct DenseLayerWeights final {
  ops::NormWeights inputNorm;
  MixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

using Qwen3_8Weights = TargetModelWeights<Qwen3_8Layout, DenseLayerWeights>;

[[nodiscard]] Qwen3_8Weights
loadQwen3_8Weights(metal::MetalBackend &backend, Qwen3_8Layout layout,
                   const TargetFiles<Qwen3_8Layout> &files);

} // namespace richengine::model

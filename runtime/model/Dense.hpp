#pragma once

#include "Qwen3_8.hpp"
#include "QwenHybridLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// The pure dense transformer target (MiniCPM5-2B): 42 full-attention layers
// of hidden 2048, 16x2 heads of 128, a full rotary of 64 pairs at theta 5e6,
// no query gate and no per-head norms, and a gated FFN of intermediate
// 6144. It carries no recurrent state; its packed files are the family's
// own magics.
struct DenseLayout final {
  static constexpr std::string_view layerMagic = "MDFN0001";
  static constexpr std::string_view headMagic = "MDFN0002";
  static constexpr FfnKind ffnKind = FfnKind::Dense;
  // The packed QKV rows hold the query heads alone (no interleaved gate).
  static constexpr uint32_t attentionQueryStride = 1;
  static constexpr bool attentionQkNorm = false;

  uint32_t maximumContextTokens = 131072;
  uint32_t layers = 42;
  uint32_t hiddenSize = 2048;
  uint32_t vocabularySize = 130560;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 2560;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 2048;
  uint32_t attentionQueryHeads = 16;
  uint32_t attentionKvHeads = 2;
  uint32_t attentionHeadDimension = 128;
  uint32_t rotaryPairs = 64;
  float rotaryTheta = 5'000'000.0F;
  uint32_t fullAttentionPeriod = 1;
  // The DSpark draft's mask token id and hidden-state capture layers; the
  // draft config declares both (mask_token_id, target_layer_ids).
  uint32_t maskToken = 75982;
  std::array<uint32_t, 2> stopTokens = {1, 130073};
  std::array<uint32_t, 5> hiddenCaptureLayers = {1, 10, 20, 30, 39};
  uint32_t intermediateSize = 6144;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return layer < layers;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    return layers;
  }
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept { return 0; }
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {attentionLayerCount(), attentionKvHeads,
            attentionHeadDimension};
  }
  // No recurrent layers at all: the empty state layout.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr MixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize, packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads, gdnHeadDimension,
            attentionWidth, attentionHeadDimension};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  bool operator==(const DenseLayout &) const = default;
};

using DenseWeights = TargetModelWeights<DenseLayout, DenseLayerWeights>;

[[nodiscard]] DenseWeights
loadDenseWeights(metal::MetalBackend &backend, DenseLayout layout,
                 const TargetFiles<DenseLayout> &files);

} // namespace richengine::model

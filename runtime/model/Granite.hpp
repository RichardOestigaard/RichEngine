#pragma once

#include "Qwen3_8.hpp"
#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// The Granite 4.2 dense transformer targets (model_type "granite"): pure
// attention+SwiGLU layers like the dense target, plus three scaling quirks —
// a fixed softmax multiplier instead of 1/sqrt(d) (attentionScale), 1e-5 RMS
// norms and a full 1-axis rotary at theta 1e7. 3B: 40 layers of hidden 2560,
// 40x8 heads of 64. 8B: 40 layers of hidden 4096, 32x8 heads of 128.
// Neither carries recurrent state; no packed files exist (GGUF only), so the
// magics name a format that is never written.
struct GraniteLayout final {
  static constexpr std::string_view layerMagic = "MGRN0001";
  static constexpr std::string_view headMagic = "MGRN0002";
  static constexpr QwenFfnKind ffnKind = QwenFfnKind::Dense;
  static constexpr uint32_t attentionQueryStride = 1;
  static constexpr bool attentionQkNorm = false;

  uint32_t maximumContextTokens = 131072;
  uint32_t layers = 40;
  uint32_t hiddenSize = 2560;
  uint32_t vocabularySize = 100352;
  uint32_t packedGdnWidth = 0;
  // The packed QKV rows: query heads alone (no interleaved gate).
  uint32_t packedFullWidth = 3584;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 2560;
  uint32_t attentionQueryHeads = 40;
  uint32_t attentionKvHeads = 8;
  uint32_t attentionHeadDimension = 64;
  uint32_t rotaryPairs = 32;
  float rotaryTheta = 10'000'000.0F;
  uint32_t fullAttentionPeriod = 1;
  // Granite's attention_multiplier: the softmax scale the kernels apply to
  // q*k scores (1/head_dim), replacing the implicit 1/sqrt(d).
  float attentionScale = 0.015625F;
  float rmsEpsilon = 1e-5F;
  // eos and the template's <|end_of_text|>-style terminator; granite has no
  // draft mask token.
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens = {100257, 100283};
  // One capture layer satisfies the geometry's draft contract; the draft
  // never runs, so its captured rows are ignored.
  std::array<uint32_t, 1> hiddenCaptureLayers = {0};
  uint32_t intermediateSize = 8192;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return layer < layers;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    return layers;
  }
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept { return 0; }
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    kv::Layout layout{attentionLayerCount(), attentionKvHeads,
                      attentionHeadDimension};
    layout.scoreScale = attentionScale;
    return layout;
  }
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr QwenMixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize, packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads, gdnHeadDimension,
            attentionWidth, attentionHeadDimension};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  bool operator==(const GraniteLayout &) const = default;
};

// Granite 4.2 8B: same structure, hidden 4096, 32x8 heads of 128,
// intermediate 12800 and attention multiplier 1/128.
[[nodiscard]] constexpr GraniteLayout granite8BLayout() noexcept {
  GraniteLayout layout;
  layout.hiddenSize = 4096;
  layout.packedFullWidth = 4096 + 2 * 8 * 128;
  layout.attentionWidth = 4096;
  layout.attentionQueryHeads = 32;
  layout.attentionHeadDimension = 128;
  layout.rotaryPairs = 64;
  layout.attentionScale = 0.0078125F;
  layout.intermediateSize = 12800;
  return layout;
}

using GraniteWeights = QwenTargetWeights<GraniteLayout, Qwen3_8LayerWeights>;

[[nodiscard]] GraniteWeights
loadGraniteWeights(metal::MetalBackend &backend, GraniteLayout layout,
                   const QwenTargetFiles<GraniteLayout> &files);

} // namespace richengine::model

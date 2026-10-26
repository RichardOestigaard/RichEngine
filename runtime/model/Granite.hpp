#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// The Granite 4.2 dense transformer targets (model_type "granite"): pure
// attention+SwiGLU layers like the dense target, plus three scaling quirks —
// a fixed softmax multiplier instead of 1/sqrt(d) (attentionScale), 1e-5 RMS
// norms and a full 1-axis rotary at theta 1e7. 3B: 40 layers of hidden 2560,
// 40x8 heads of 64. 8B: 40 layers of hidden 4096, 32x8 heads of 128.
// Neither carries recurrent state; no packed files exist (GGUF only), so the
// magics name a format that is never written. The preset is the 3B;
// granite8BTargetLayout carries the 8B's deviations.
[[nodiscard]] constexpr TargetLayout graniteTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "granite";
  layout.ggufArch = "granite";
  layout.layerMagic = "MGRN0001";
  layout.headMagic = "MGRN0002";
  layout.ffnKind = FfnKind::Dense;
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = false;
  // Packed or GGUF sources only: no MLX image plan exists for the family.
  layout.mlxImageable = false;

  layout.maximumContextTokens = 131072;
  layout.layers = 40;
  layout.hiddenSize = 2560;
  layout.vocabularySize = 100352;
  layout.packedGdnWidth = 0;
  // The packed QKV rows: query heads alone (no interleaved gate).
  layout.packedFullWidth = 3584;
  layout.convolutionDimension = 0;
  layout.gdnKeyHeads = 0;
  layout.gdnValueHeads = 0;
  layout.gdnHeadDimension = 0;
  layout.attentionWidth = 2560;
  layout.attentionQueryHeads = 40;
  layout.attentionKvHeads = 8;
  layout.attentionHeadDimension = 64;
  layout.rotaryPairs = 32;
  layout.rotaryTheta = 10'000'000.0F;
  layout.fullAttentionPeriod = 1;
  // Granite's attention_multiplier: the softmax scale the kernels apply to
  // q*k scores (1/head_dim), replacing the implicit 1/sqrt(d).
  layout.attentionScale = 0.015625F;
  layout.rmsEpsilon = 1e-5F;
  // eos and the template's <|end_of_text|>-style terminator; granite has
  // no draft mask token.
  layout.maskToken = 0;
  layout.stopTokens = {100257, 100283};
  // One capture layer satisfies the geometry's draft contract; the draft
  // never runs, so its captured rows are ignored.
  layout.hiddenCaptureLayers = {0};
  layout.captureLayerCount = 1;
  layout.intermediateSize = 8192;
  // The compiled attention kernel the layers bind (ops/PagedKv.hpp);
  // granite8BTargetLayout swaps it for the 8B's 32x128 geometry.
  layout.attentionCore = kv::Core::Kv8Group5D64;
  return layout;
}

// Granite 4.2 8B: same structure, hidden 4096, 32x8 heads of 128,
// intermediate 12800 and attention multiplier 1/128.
[[nodiscard]] constexpr TargetLayout granite8BTargetLayout() noexcept {
  TargetLayout layout = graniteTargetLayout();
  layout.hiddenSize = 4096;
  layout.packedFullWidth = 4096 + 2 * 8 * 128;
  layout.attentionWidth = 4096;
  layout.attentionQueryHeads = 32;
  layout.attentionHeadDimension = 128;
  layout.attentionCore = kv::Core::Kv8Group4D128;
  layout.rotaryPairs = 64;
  layout.attentionScale = 0.0078125F;
  layout.intermediateSize = 12800;
  return layout;
}

using GraniteWeights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadGraniteWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                   const TargetFiles &files);

} // namespace richengine::model

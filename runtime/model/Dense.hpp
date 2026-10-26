#pragma once

#include "TargetLayout.hpp"
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
[[nodiscard]] constexpr TargetLayout denseTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "dense";
  layout.ggufArch = "llama";
  layout.layerMagic = "MDFN0001";
  layout.headMagic = "MDFN0002";
  layout.ffnKind = FfnKind::Dense;
  // The packed QKV rows hold the query heads alone (no interleaved gate).
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = false;
  // MLX sources may load as block images under preferBlockWeights.
  layout.mlxBlockWeights = true;

  layout.maximumContextTokens = 131072;
  layout.layers = 42;
  layout.hiddenSize = 2048;
  layout.vocabularySize = 130560;
  layout.packedGdnWidth = 0;
  layout.packedFullWidth = 2560;
  layout.convolutionDimension = 0;
  layout.gdnKeyHeads = 0;
  layout.gdnValueHeads = 0;
  layout.gdnHeadDimension = 0;
  layout.attentionWidth = 2048;
  layout.attentionQueryHeads = 16;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 128;
  layout.rotaryPairs = 64;
  layout.rotaryTheta = 5'000'000.0F;
  layout.fullAttentionPeriod = 1;
  // The DSpark draft's mask token id and hidden-state capture layers; the
  // draft config declares both (mask_token_id, target_layer_ids).
  layout.maskToken = 75982;
  layout.stopTokens = {1, 130073};
  layout.hiddenCaptureLayers = {1, 10, 20, 30, 39};
  layout.captureLayerCount = 5;
  layout.intermediateSize = 6144;
  // The compiled attention kernel the layers bind (ops/PagedKv.hpp).
  layout.attentionCore = kv::Core::Kv2Group8D128;
  return layout;
}

using DenseWeights = TargetModelWeights;

// preferBlockWeights reads an MLX source's images as AF4 block sections
// (AffineBlockTargetFormat); it must match the flag the AffineTargetLoader
// the files came from was built with, and leaves packed and GGUF files
// unchanged.
[[nodiscard]] TargetModelWeights
loadDenseWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                 const TargetFiles &files, bool preferBlockWeights = false);

} // namespace richengine::model

#pragma once

#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace richengine::model {

// Gemma 4 26B-A4B: 30 attention layers on every block — the sliding layers
// (all but layer % 6 == 5) hold 16 query heads over 8 KV heads of 256 with
// full rotary at theta 1e4 and a 1024-token window; the five global layers
// hold 16 query heads over 2 KV heads of 512, p-RoPE of 64 pairs at theta
// 1e6, and k_eq_v (no V projection: the value slot stores the scale-free
// RMS norm of the pre-norm K). Every layer ends in the MoE block: a bias-free
// bf16 router over the scaleless-normed, router-scaled post-attention
// residual, softmax over all 128 experts, top-8 renormalized by
// per_expert_scale, plus a dense GeGLU shared expert of width 2112. The two
// outputs are post-FFN-normed separately (norm_1 shared, norm_2 routed) and
// summed into the residual, then scaled by the per-layer layer_scalar.
// Not final: DiffusionGemmaLayout (DiffusionGemma.hpp) derives from it,
// adding the canvas schedule while keeping every trunk member.
struct Gemma4MoeLayout {
  static constexpr std::string_view layerMagic = "GEMM0001";
  static constexpr std::string_view headMagic = "GEMM0002";
  static constexpr FfnKind ffnKind = FfnKind::SparseMoe;
  // No MLX or GGUF loader maps this family's tensors; packages are packed at
  // installation (ModelFactory's packedOnly gate).
  static constexpr bool packedOnly = true;
  // Global layers are 5, 11, 17, 23, 29: every sixth, counting from 0.
  static constexpr uint32_t globalLayerPeriod = 6;
  static constexpr uint32_t captureLayerCount = 6;

  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
  uint32_t layers = 30;
  uint32_t hiddenSize = 2816;
  uint32_t vocabularySize = 262144;
  uint32_t attentionQueryHeads = 16;
  // The sliding layers' KV geometry.
  uint32_t attentionKvHeads = 8;
  uint32_t attentionHeadDimension = 256;
  // The global layers'.
  uint32_t globalKvHeads = 2;
  uint32_t globalHeadDimension = 512;
  // Full rotary of 256 / 2 pairs at theta 1e4 on the sliding layers; p-RoPE
  // of 64 pairs (the first 128 of 512 dimensions) at theta 1e6 on the
  // globals.
  uint32_t rotaryPairs = 128;
  float rotaryTheta = 10'000.0F;
  uint32_t globalRotaryPairs = 64;
  float globalRotaryTheta = 1'000'000.0F;
  uint32_t slidingWindowTokens = 1024;
  uint32_t experts = 128;
  uint32_t expertsPerToken = 8;
  // The routed experts' logical width. The Q4 expert tiles write whole
  // 128-column tiles, so the packed slabs pad each expert's gate/up rows and
  // down inputs to 768; the padding columns and rows hold zeros, which
  // contribute nothing to the GeGLU product or the down sums.
  uint32_t expertIntermediateSize = 704;
  uint32_t packedExpertWidth = 768;
  uint32_t sharedIntermediateSize = 2112;
  // The shared expert runs as a dense GeGLU triple through the affine
  // projections, whose tiles are 256 rows: its packed slabs pad 2112 to
  // 2304 (the padding rows and inputs hold zeros).
  uint32_t packedSharedWidth = 2304;
  float logitSoftcap = 30.0F;
  // Embedding scale sqrt(hidden); tied embeddings.
  float embeddingScale = 0.0F; // derived: sqrt(hiddenSize)
  uint32_t maskToken = 4;
  // Gemma's <eos> and <end_of_turn>.
  std::array<uint32_t, 2> stopTokens = {1, 106};
  std::array<uint32_t, captureLayerCount> hiddenCaptureLayers = {
      1, 6, 11, 17, 22, 27};

  [[nodiscard]] constexpr bool isGlobalAttentionLayer(uint32_t layer) const noexcept {
    return layer % globalLayerPeriod == globalLayerPeriod - 1;
  }
  // Every layer is attention; "full" names the globals for the packed files'
  // type tag.
  [[nodiscard]] constexpr bool isFullAttentionLayer(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer);
  }
  [[nodiscard]] constexpr uint64_t globalLayerMask() const noexcept {
    uint64_t mask = 0;
    for (uint32_t layer = 0; layer < layers; ++layer)
      mask |= uint64_t{isGlobalAttentionLayer(layer)} << layer;
    return mask;
  }
  [[nodiscard]] constexpr uint32_t kvHeadsAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? globalKvHeads : attentionKvHeads;
  }
  [[nodiscard]] constexpr uint32_t headDimensionAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? globalHeadDimension
                                         : attentionHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t rotaryPairsAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? globalRotaryPairs : rotaryPairs;
  }
  [[nodiscard]] constexpr float rotaryThetaAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? globalRotaryTheta : rotaryTheta;
  }
  [[nodiscard]] constexpr uint32_t attentionWidthAt(uint32_t layer) const noexcept {
    return attentionQueryHeads * headDimensionAt(layer);
  }
  // The fused QKV row width: queries, keys and — the locals only — values.
  [[nodiscard]] constexpr uint32_t packedWidthAt(uint32_t layer) const noexcept {
    const uint32_t values = isGlobalAttentionLayer(layer) ? 1 : 2;
    return attentionQueryHeads * headDimensionAt(layer) +
           values * kvHeadsAt(layer) * headDimensionAt(layer);
  }
  [[nodiscard]] constexpr uint32_t maximumPackedWidth() const noexcept {
    const uint32_t local = packedWidthAt(0);
    const uint32_t global = packedWidthAt(globalLayerPeriod - 1);
    return local > global ? local : global;
  }
  // The pool layout: local geometry primary, globals the alternate regions.
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {layers, attentionKvHeads, attentionHeadDimension,
            kv::Format::Int8, 1.0F, globalKvHeads, globalHeadDimension,
            globalLayerMask(), 0};
  }
  [[nodiscard]] constexpr kv::Layout kvLayout(kv::Format format) const noexcept {
    kv::Layout layout = kvLayout();
    layout.format = format;
    return layout;
  }
  // A stateless target: no recurrent or convolution state.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }

  bool operator==(const Gemma4MoeLayout &) const = default;
};

// One Gemma 4 layer's weights. The mixer is always a AttentionMixerWeights:
// the fused QKV projection (no V rows on the globals), the learned QK norms
// and the output projection; the scaleless V norm is inside the kernels.
struct Gemma4MoeLayerWeights final {
  ops::NormWeights inputNorm;
  MixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  // The MoE block: pre_ffn_norm feeds the shared expert and preFfnNormRouted
  // (the checkpoint's pre_feedforward_layernorm_2) the routed experts; the
  // router reads the post-attention residual through routerScale (learned
  // per-dimension scale, bf16) after a scaleless RMS norm.
  ops::NormWeights preFfnNorm;
  ops::NormWeights preFfnNormRouted;
  metal::MetalBuffer routerScale;
  // [experts][hidden] bf16, expert-major, bias-free.
  metal::MetalBuffer routerWeights;
  // f32[experts].
  metal::MetalBuffer perExpertScale;
  ops::ExpertProjection expertGate;
  ops::ExpertProjection expertUp;
  ops::ExpertProjection expertDown;
  ops::Projection sharedGate;
  ops::Projection sharedUp;
  ops::Projection sharedDown;
  ops::NormWeights postFfnNormShared;
  ops::NormWeights postFfnNormRouted;
  // The checkpoint's un-suffixed post_feedforward_layernorm: applied to the
  // routed + shared sum before the residual add.
  ops::NormWeights postFfnNorm;
  // layer_scalar: a host-known scalar bound to layer_scalar_scale.
  float layerScalar = 1.0F;
};

using Gemma4MoeWeights =
    TargetModelWeights<Gemma4MoeLayout, Gemma4MoeLayerWeights>;

[[nodiscard]] Gemma4MoeWeights
loadGemma4MoeWeights(metal::MetalBackend &backend, Gemma4MoeLayout layout,
                     const TargetFiles<Gemma4MoeLayout> &files);

} // namespace richengine::model

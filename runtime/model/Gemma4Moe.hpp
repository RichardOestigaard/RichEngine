#pragma once

#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <cstdint>

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
// DiffusionGemma's preset (DiffusionGemma.hpp) copies it, changing the arch
// while keeping every trunk member.
[[nodiscard]] constexpr TargetLayout gemma4TargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "gemma4";
  layout.layerMagic = "GEMM0001";
  layout.headMagic = "GEMM0002";
  layout.ffnKind = FfnKind::SparseMoe;
  // The learned per-head RMS norms on the queries and keys.
  layout.attentionQkNorm = true;
  // No MLX or GGUF loader maps this family's tensors; packages are packed at
  // installation (ModelFactory's packedOnly gate).
  layout.packedOnly = true;
  // Global layers are 5, 11, 17, 23, 29: every sixth, counting from 0.
  layout.globalLayerPeriod = 6;
  layout.maximumContextTokens = kv::kMaximumLogicalTokens;
  layout.layers = 30;
  layout.hiddenSize = 2816;
  layout.vocabularySize = 262144;
  layout.attentionQueryHeads = 16;
  // The sliding layers' KV geometry.
  layout.attentionKvHeads = 8;
  layout.attentionHeadDimension = 256;
  layout.attentionWidth = 16 * 256;
  // The global layers'.
  layout.altKvHeads = 2;
  layout.altHeadDimension = 512;
  // Full rotary of 256 / 2 pairs at theta 1e4 on the sliding layers; p-RoPE
  // of 64 pairs (the first 128 of 512 dimensions) at theta 1e6 on the
  // globals.
  layout.rotaryPairs = 128;
  layout.rotaryTheta = 10'000.0F;
  layout.altRotaryPairs = 64;
  layout.altRotaryTheta = 1'000'000.0F;
  layout.slidingWindowTokens = 1024;
  layout.experts = 128;
  layout.expertsPerToken = 8;
  // The routed experts' logical width. The Q4 expert tiles write whole
  // 128-column tiles, so the packed slabs pad each expert's gate/up rows and
  // down inputs to 768; the padding columns and rows hold zeros, which
  // contribute nothing to the GeGLU product or the down sums.
  layout.expertIntermediateSize = 704;
  layout.packedExpertWidth = 768;
  layout.sharedIntermediateSize = 2112;
  // The shared expert runs as a dense GeGLU triple through the affine
  // projections, whose tiles are 256 rows: its packed slabs pad 2112 to
  // 2304 (the padding rows and inputs hold zeros).
  layout.packedSharedWidth = 2304;
  layout.logitSoftcap = 30.0F;
  // Embedding scale sqrt(hidden); tied embeddings. 0 derives
  // sqrt(hiddenSize) in the geometry pass.
  layout.embeddingScale = 0.0F;
  layout.rmsEpsilon = 1e-6F;
  layout.maskToken = 4;
  // Gemma's <eos> and <end_of_turn>.
  layout.stopTokens = {1, 106};
  layout.hiddenCaptureLayers = {1, 6, 11, 17, 22, 27};
  layout.captureLayerCount = 6;
  // The compiled attention kernels: the sliding layers bind GemmaKv8H256,
  // the k_eq_v globals GemmaKv2H512 (ops/PagedKv.hpp).
  layout.attentionCore = kv::Core::GemmaKv8H256;
  layout.altAttentionCore = kv::Core::GemmaKv2H512;
  // The shared expert runs beside the router as a dense GeGLU block, not as
  // a routed shared-expert route.
  layout.moeSharedExpert = false;
  return layout;
}

// A layer's weights are the shared TargetLayerWeights: inputNorm, mixer and
// postAttentionNorm on the layer itself and every other Gemma tensor on its
// Gemma4FfnWeights FFN (TargetModel.hpp). The mixer is always a
// AttentionMixerWeights: the fused QKV projection (no V rows on the
// globals), the learned QK norms and the output projection; the scaleless V
// norm is inside the kernels.
using Gemma4MoeWeights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadGemma4MoeWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                     const TargetFiles &files);

} // namespace richengine::model

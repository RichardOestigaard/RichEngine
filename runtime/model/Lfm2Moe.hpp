#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/MoE.hpp"

#include <cstdint>
#include <string_view>
#include <variant>

namespace richengine::model {

// LiquidAI's LFM2.5-8B-A1B (arch "lfm2moe"): 24 layers of hidden 2048 — 18
// double-gated short convolutions over a taps-3 FIFO state and 6 full
// attention layers (mask below) of 32x8 heads of 64, a full rotary of 32
// pairs at theta 5e6, per-head q/k RMS norms of epsilon 1e-5, no query gate
// and tied embeddings. Every layer's FFN is dense (intermediate 7168) for
// the leading num_dense_layers 2 layers and a sparse MoE block after them:
// 32 routed experts of intermediate 1792, four per token, sigmoid gating
// with a per-expert selection bias (exp_probs_b) and weights normalized over
// the selected experts, no shared expert.
[[nodiscard]] constexpr TargetLayout lfm2MoeTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "lfm2_moe";
  layout.ggufArch = "lfm2moe";
  layout.tiedOutput = true;
  layout.layerMagic = "MDFH0003";
  layout.headMagic = "MDFH0004";
  layout.ffnKind = FfnKind::SparseMoe;
  // The packed QKV rows hold the query heads alone (no interleaved gate).
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = true;
  // The conv FIFO keeps taps - 1 rows of the convolution's B*x elements.
  layout.convolutionTaps = 3;
  // The leading dense-FFN layers (the config's num_dense_layers); every
  // later layer — conv or attention — carries the MoE block.
  layout.denseLayers = 2;

  // The full-attention layer indices {2,6,10,14,18,21} of the released
  // LFM2.5-8B-A1B config's layer_types.
  layout.attentionMask =
      (uint64_t{1} << 2) | (uint64_t{1} << 6) | (uint64_t{1} << 10) |
      (uint64_t{1} << 14) | (uint64_t{1} << 18) | (uint64_t{1} << 21);

  layout.maximumContextTokens = 128000;
  layout.layers = 24;
  layout.hiddenSize = 2048;
  layout.vocabularySize = 128000;
  layout.packedGdnWidth = 6144;  // the conv in_proj's [B|C|x] row
  layout.packedFullWidth = 3072; // 2048 query + 2*512 key/value
  layout.convolutionDimension = 2048;
  layout.gdnKeyHeads = 0;
  layout.gdnValueHeads = 0;
  layout.gdnHeadDimension = 0;
  layout.attentionWidth = 2048;
  layout.attentionQueryHeads = 32;
  layout.attentionKvHeads = 8;
  layout.attentionHeadDimension = 64;
  layout.rotaryPairs = 32;
  layout.rotaryTheta = 5'000'000.0F;
  // The DSpark draft's mask token id and hidden-state capture layers; the
  // draft config declares both under dflash_config.
  layout.maskToken = 125017;
  layout.stopTokens = {124900, 124900};
  layout.hiddenCaptureLayers = {2, 6, 10, 14, 18};
  layout.captureLayerCount = 5;
  // The leading dense layers' gated FFN intermediate size.
  layout.intermediateSize = 7168;
  // The MoE FFN of every layer from denseLayers on.
  layout.experts = 32;
  layout.expertsPerToken = 4;
  layout.expertIntermediateSize = 1792;
  // The norms' epsilon (norm_eps), which selects the _e5 kernel variants.
  layout.rmsEpsilon = 1e-5F;
  // No shared expert: the MoE blocks carry routed experts alone.
  layout.moeSharedExpert = false;
  // The compiled attention kernel the layers bind (ops/PagedKv.hpp).
  layout.attentionCore = kv::Core::Kv8Group4D64;
  return layout;
}

using Lfm2MoeWeights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadLfm2MoeWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                   const TargetFiles &files);

} // namespace richengine::model

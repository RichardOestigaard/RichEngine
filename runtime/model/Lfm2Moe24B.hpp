#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/MoE.hpp"

#include <cstdint>
#include <string_view>
#include <variant>

namespace richengine::model {

// LiquidAI's LFM2.5-24B-A2B (arch "lfm2moe"): 40 layers of hidden 2048 — 30
// double-gated short convolutions over a taps-3 FIFO state and 10 full
// attention layers (mask below) of 32x8 heads of 64, a full rotary of 32
// pairs at theta 1e7, per-head q/k RMS norms of epsilon 1e-5, no query gate
// and tied embeddings. Every layer's FFN is dense (intermediate 11776) for
// the leading num_dense_layers 2 layers and a sparse MoE block after them:
// 64 routed experts of intermediate 1536, four per token, sigmoid gating
// with a per-expert selection bias (exp_probs_b) and weights normalized over
// the selected experts, no shared expert. Its DSpark draft
// (LiquidAI/LFM2.5-24B-A2B-DSpark) taps the full-attention layers
// {2,10,18,26,34}'s hidden states.
[[nodiscard]] constexpr TargetLayout lfm2Moe24BTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "lfm2_moe_24b";
  layout.ggufArch = "lfm2moe";
  layout.tiedOutput = true;
  layout.layerMagic = "MDFH0005";
  layout.headMagic = "MDFH0006";
  layout.ffnKind = FfnKind::SparseMoe;
  // The packed QKV rows hold the query heads alone (no interleaved gate).
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = true;
  // The conv FIFO keeps taps - 1 rows of the convolution's B*x elements.
  layout.convolutionTaps = 3;
  // The leading dense-FFN layers (the config's num_dense_layers); every
  // later layer — conv or attention — carries the MoE block.
  layout.denseLayers = 2;

  // The full-attention layer indices {2,6,10,14,18,22,26,30,34,38} of the
  // released LFM2.5-24B-A2B config's layer_types.
  layout.attentionMask =
      (uint64_t{1} << 2) | (uint64_t{1} << 6) | (uint64_t{1} << 10) |
      (uint64_t{1} << 14) | (uint64_t{1} << 18) | (uint64_t{1} << 22) |
      (uint64_t{1} << 26) | (uint64_t{1} << 30) | (uint64_t{1} << 34) |
      (uint64_t{1} << 38);

  layout.maximumContextTokens = 131072;
  layout.layers = 40;
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
  layout.rotaryTheta = 10'000'000.0F;
  // The LFM2.5 tokenizer's mask and stop tokens; the DSpark draft's
  // mask_token_id is the mask.
  layout.maskToken = 125017;
  layout.stopTokens = {124900, 124900};
  // The DSpark draft's hidden-state capture layers (its dflash_config's
  // target_layer_ids), all full-attention layers.
  layout.hiddenCaptureLayers = {2, 10, 18, 26, 34};
  layout.captureLayerCount = 5;
  // The leading dense layers' gated FFN intermediate size.
  layout.intermediateSize = 11776;
  // The MoE FFN of every layer from denseLayers on.
  layout.experts = 64;
  layout.expertsPerToken = 4;
  layout.expertIntermediateSize = 1536;
  // The norms' epsilon (norm_eps), which selects the _e5 kernel variants.
  layout.rmsEpsilon = 1e-5F;
  // No shared expert: the MoE blocks carry routed experts alone.
  layout.moeSharedExpert = false;
  // The compiled attention kernel the layers bind (ops/PagedKv.hpp).
  layout.attentionCore = kv::Core::Kv8Group4D64;
  return layout;
}

using Lfm2Moe24BWeights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadLfm2Moe24BWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                      const TargetFiles &files);

} // namespace richengine::model

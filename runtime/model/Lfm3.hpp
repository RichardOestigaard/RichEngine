#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"

#include <cstdint>

namespace richengine::model {

// LFM3-1.7B's tri-hybrid schedule: thirteen conv layers, five
// sliding-window attention layers {4,8,12,16,20} and six full-attention
// (global) layers {2,6,10,14,18,22} of the released config's layer_types.
inline constexpr uint64_t kLfm3AttentionMask =
    (uint64_t{1} << 2) | (uint64_t{1} << 4) | (uint64_t{1} << 6) |
    (uint64_t{1} << 8) | (uint64_t{1} << 10) | (uint64_t{1} << 12) |
    (uint64_t{1} << 14) | (uint64_t{1} << 16) | (uint64_t{1} << 18) |
    (uint64_t{1} << 20) | (uint64_t{1} << 22);
inline constexpr uint64_t kLfm3GlobalMask =
    (uint64_t{1} << 2) | (uint64_t{1} << 6) | (uint64_t{1} << 10) |
    (uint64_t{1} << 14) | (uint64_t{1} << 18) | (uint64_t{1} << 22);

// LiquidAI's LFM3 target (LFM3-1.7B, model_type "lfm3"): 24 layers of
// hidden 2048 — 13 double-gated short convolutions over a taps-7 FIFO
// state, 5 sliding-window attention layers (window 256, full rotary of 64
// pairs at theta 1e4) and 6 global attention layers (partial rotary: the
// first 32 pairs at theta 1e6, the tail unrotated) — all attention of
// 16x4 heads of 128 with per-head q/k RMS norms of epsilon 1e-5 and the
// per-head sigmoid gate_proj on the attention output. Sandwich norms: the
// post-mixer and post-FFN outputs are normed before they join the
// residual. Gated FFN of intermediate 7680, tied embeddings, 262144-token
// context.
[[nodiscard]] constexpr TargetLayout lfm3TargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "lfm3";
  layout.ggufArch = "lfm3";
  // The output head shares the token embedding.
  layout.tiedOutput = true;
  layout.layerMagic = "MDFH0001";
  layout.headMagic = "MDFH0002";
  layout.ffnKind = FfnKind::Dense;
  // The packed QKV rows hold the query heads alone (no interleaved gate);
  // the head gate reads its own [row][head] buffer instead.
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = true;
  // The conv FIFO keeps taps - 1 rows of the convolution's B*x elements.
  layout.convolutionTaps = 7;
  layout.attentionMask = kLfm3AttentionMask;
  // The alt geometry slot carries the global layers: same KV sizes, a
  // theta-1e6 rope table whose tail past altRotaryRealPairs is zeroed (the
  // zero-tail partial rotary of partial_rotary_factor 0.5), V rows present.
  layout.globalAttentionMask = kLfm3GlobalMask;
  layout.altKvHeads = 4;
  layout.altHeadDimension = 128;
  layout.altRotaryPairs = 64;
  layout.altRotaryTheta = 1'000'000.0F;
  layout.altRotaryRealPairs = 32;
  layout.altCarriesValues = true;
  layout.slidingWindowTokens = 256;
  layout.sandwichNorm = true;
  layout.attentionHeadGate = true;
  // gate_proj is 16 rows, padded to one 256-row Q4 tile in the image.
  layout.attentionGateRows = 256;

  layout.maximumContextTokens = 262144;
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
  layout.attentionQueryHeads = 16;
  layout.attentionKvHeads = 4;
  layout.attentionHeadDimension = 128;
  layout.rotaryPairs = 64;
  layout.rotaryTheta = 10'000.0F;
  // No DSpark draft yet; the padding token id fills the mask slot and the
  // n-gram predraft supplies proposals.
  layout.maskToken = 124893;
  // LFM3 states a single end token; it fills both slots.
  layout.stopTokens = {124900, 124900};
  layout.hiddenCaptureLayers = {2, 6, 10, 14, 18};
  layout.captureLayerCount = 5;
  layout.intermediateSize = 7680;
  // The norms' epsilon (norm_eps), which selects the _e5 kernel variants.
  layout.rmsEpsilon = 1e-5F;

  layout.encoder = false;
  layout.decisionBlocks = 0;
  layout.altAttentionKvHeads = 0;
  layout.altIntermediateSize = 0;

  // The compiled attention kernel the layers bind (ops/PagedKv.hpp): both
  // layer kinds share the 4-KV-head, group-4, 128-dimension core.
  layout.attentionCore = kv::Core::Kv4Group4D128;
  layout.altAttentionCore = kv::Core::Kv4Group4D128;
  return layout;
}

using Lfm3Weights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadLfm3Weights(metal::MetalBackend &backend, const TargetLayout &layout,
                const TargetFiles &files);

} // namespace richengine::model

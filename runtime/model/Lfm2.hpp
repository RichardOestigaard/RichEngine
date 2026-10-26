#pragma once

#include "TargetLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// The full-attention layer indices {2,5,9,13,17,21,24,27} of the released
// LFM2.5-2.6B config's layer_types; a family maker overrides the preset's
// attentionMask for another schedule (d1-omni's {2,5,8,10,12,14,16,17}).
inline constexpr uint64_t kLfm2DefaultAttentionMask =
    (uint64_t{1} << 2) | (uint64_t{1} << 5) | (uint64_t{1} << 9) |
    (uint64_t{1} << 13) | (uint64_t{1} << 17) | (uint64_t{1} << 21) |
    (uint64_t{1} << 24) | (uint64_t{1} << 27);

// LiquidAI's LFM2 hybrid target (LFM2.5-2.6B): 30 layers of hidden 2048 —
// 22 double-gated short convolutions over a taps-3 FIFO state and 8 full
// attention layers (mask above) of 32x8 heads of 64, a full rotary of 32
// pairs at theta 1e7, per-head q/k RMS norms of epsilon 1e-5, no query
// gate, a gated FFN of intermediate 10752 and tied embeddings. Its conv
// state rolls back with speculative decoding like the GDN state does.
[[nodiscard]] constexpr TargetLayout lfm2TargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "lfm2";
  layout.ggufArch = "lfm2";
  // The output head shares the token embedding.
  layout.tiedOutput = true;
  layout.layerMagic = "MDFH0001";
  layout.headMagic = "MDFH0002";
  layout.ffnKind = FfnKind::Dense;
  // The packed QKV rows hold the query heads alone (no interleaved gate).
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = true;
  // The conv FIFO keeps taps - 1 rows of the convolution's B*x elements.
  layout.convolutionTaps = 3;
  layout.attentionMask = kLfm2DefaultAttentionMask;

  layout.maximumContextTokens = 131072;
  layout.layers = 30;
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
  // The DSpark draft's mask token id and hidden-state capture layers; the
  // draft config declares both under dflash_config.
  layout.maskToken = 125017;
  // LFM2.5 states a single end token; it fills both slots.
  layout.stopTokens = {124900, 124900};
  layout.hiddenCaptureLayers = {2, 9, 17, 21, 27};
  layout.captureLayerCount = 5;
  layout.intermediateSize = 10752;
  // The norms' epsilon (norm_eps), which selects the _e5 kernel variants.
  layout.rmsEpsilon = 1e-5F;

  // The d1-omni decision encoder's deviations from the causal LFM2 target,
  // recorded so the assembly's per-layer fields validate; the loader refuses
  // it — no compiled kernels cover its bidirectional attention, the widened
  // decision blocks' KV head count or their fused-QKV LayerNorm structure:
  //   encoder           — bidirectional attention (GGUF attention.causal
  //                       false); the model never decodes.
  //   decisionBlocks    — the trailing blocks that are bespoke decision
  //                       layers (lfm2.decision.block_count).
  //   altAttentionKvHeads — the decision blocks' KV head count where it
  //                       differs from attentionKvHeads.
  //   altIntermediateSize — the decision blocks' narrower FFN.
  layout.encoder = false;
  layout.decisionBlocks = 0;
  layout.altAttentionKvHeads = 0;
  layout.altIntermediateSize = 0;

  // The compiled attention kernel the layers bind (ops/PagedKv.hpp).
  layout.attentionCore = kv::Core::Kv8Group4D64;
  return layout;
}

using Lfm2Weights = TargetModelWeights;

[[nodiscard]] TargetModelWeights
loadLfm2Weights(metal::MetalBackend &backend, const TargetLayout &layout,
                const TargetFiles &files);

} // namespace richengine::model

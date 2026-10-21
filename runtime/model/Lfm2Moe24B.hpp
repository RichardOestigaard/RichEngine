#pragma once

#include "Lfm2Moe.hpp"
#include "Qwen3_8.hpp"
#include "QwenHybridLayout.hpp"
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
struct Lfm2Moe24BLayout final {
  static constexpr std::string_view layerMagic = "MDFH0005";
  static constexpr std::string_view headMagic = "MDFH0006";
  static constexpr FfnKind ffnKind = FfnKind::SparseMoe;
  // The packed QKV rows hold the query heads alone (no interleaved gate).
  static constexpr uint32_t attentionQueryStride = 1;
  static constexpr bool attentionQkNorm = true;
  // The conv FIFO keeps taps - 1 rows of the convolution's B*x elements.
  static constexpr uint32_t convolutionTaps = 3;
  // The leading dense-FFN layers (the config's num_dense_layers); every
  // later layer — conv or attention — carries the MoE block.
  static constexpr uint32_t denseLayers = 2;

  // The full-attention layer indices {2,6,10,14,18,22,26,30,34,38} of the
  // released LFM2.5-24B-A2B config's layer_types.
  static constexpr uint64_t attentionMask =
      (uint64_t{1} << 2) | (uint64_t{1} << 6) | (uint64_t{1} << 10) |
      (uint64_t{1} << 14) | (uint64_t{1} << 18) | (uint64_t{1} << 22) |
      (uint64_t{1} << 26) | (uint64_t{1} << 30) | (uint64_t{1} << 34) |
      (uint64_t{1} << 38);

  uint32_t maximumContextTokens = 131072;
  uint32_t layers = 40;
  uint32_t hiddenSize = 2048;
  uint32_t vocabularySize = 128000;
  uint32_t packedGdnWidth = 6144;  // the conv in_proj's [B|C|x] row
  uint32_t packedFullWidth = 3072; // 2048 query + 2*512 key/value
  uint32_t convolutionDimension = 2048;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 2048;
  uint32_t attentionQueryHeads = 32;
  uint32_t attentionKvHeads = 8;
  uint32_t attentionHeadDimension = 64;
  uint32_t rotaryPairs = 32;
  float rotaryTheta = 10'000'000.0F;
  // The LFM2.5 tokenizer's mask and stop tokens; the DSpark draft's
  // mask_token_id is the mask.
  uint32_t maskToken = 125017;
  std::array<uint32_t, 2> stopTokens = {124900, 124900};
  // The DSpark draft's hidden-state capture layers (its dflash_config's
  // target_layer_ids), all full-attention layers.
  std::array<uint32_t, 5> hiddenCaptureLayers = {2, 10, 18, 26, 34};
  // The leading dense layers' gated FFN intermediate size.
  uint32_t intermediateSize = 11776;
  // The MoE FFN of every layer from denseLayers on.
  uint32_t experts = 64;
  uint32_t expertsPerToken = 4;
  uint32_t expertIntermediateSize = 1536;
  // The norms' epsilon (norm_eps), which selects the _e5 kernel variants.
  float rmsEpsilon = 1e-5F;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return layer < 64 && (attentionMask >> layer) & 1;
  }
  // The dense-FFN layers come first; every later layer's FFN is the MoE.
  [[nodiscard]] constexpr bool isMoeLayer(uint32_t layer) const noexcept {
    return layer >= denseLayers;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    uint32_t count = 0;
    for (uint32_t layer = 0; layer < 64; ++layer)
      count += (attentionMask >> layer) & 1;
    return count;
  }
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept { return 0; }
  // The compiled attention kernel the layers bind (ops/PagedKv.hpp).
  kv::Core attentionCore = kv::Core::Kv8Group4D64;
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    kv::Layout layout{attentionLayerCount(), attentionKvHeads,
                      attentionHeadDimension};
    layout.attentionCore = attentionCore;
    return layout;
  }
  // Conv-state layers only: the FIFO of taps-1 rows, no recurrent matrix.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - attentionLayerCount(), convolutionTaps - 1,
            convolutionDimension, 0, 0, 0};
  }
  [[nodiscard]] constexpr MixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,     packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads,  gdnHeadDimension,
            attentionWidth, attentionHeadDimension, convolutionTaps};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  bool operator==(const Lfm2Moe24BLayout &) const = default;
};

using Lfm2Moe24BWeights =
    TargetModelWeights<Lfm2Moe24BLayout, Lfm2MoeLayerWeights>;

[[nodiscard]] Lfm2Moe24BWeights
loadLfm2Moe24BWeights(metal::MetalBackend &backend, Lfm2Moe24BLayout layout,
                      const TargetFiles<Lfm2Moe24BLayout> &files);

} // namespace richengine::model

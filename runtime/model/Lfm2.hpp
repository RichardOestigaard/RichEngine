#pragma once

#include "Qwen3_8.hpp"
#include "QwenHybridLayout.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"

#include <cstdint>
#include <string_view>

namespace richengine::model {

// LiquidAI's LFM2 hybrid target (LFM2.5-2.6B): 30 layers of hidden 2048 —
// 22 double-gated short convolutions over a taps-3 FIFO state and 8 full
// attention layers (mask below) of 32x8 heads of 64, a full rotary of 32
// pairs at theta 1e7, per-head q/k RMS norms of epsilon 1e-5, no query
// gate, a gated FFN of intermediate 10752 and tied embeddings. Its conv
// state rolls back with speculative decoding like the GDN state does.
struct Lfm2Layout final {
  static constexpr std::string_view layerMagic = "MDFH0001";
  static constexpr std::string_view headMagic = "MDFH0002";
  static constexpr FfnKind ffnKind = FfnKind::Dense;
  // The packed QKV rows hold the query heads alone (no interleaved gate).
  static constexpr uint32_t attentionQueryStride = 1;
  static constexpr bool attentionQkNorm = true;
  // The conv FIFO keeps taps - 1 rows of the convolution's B*x elements.
  static constexpr uint32_t convolutionTaps = 3;

  // The full-attention layer indices {2,5,9,13,17,21,24,27} of the released
  // LFM2.5-2.6B config's layer_types.
  static constexpr uint64_t attentionMask =
      (uint64_t{1} << 2) | (uint64_t{1} << 5) | (uint64_t{1} << 9) |
      (uint64_t{1} << 13) | (uint64_t{1} << 17) | (uint64_t{1} << 21) |
      (uint64_t{1} << 24) | (uint64_t{1} << 27);

  uint32_t maximumContextTokens = 131072;
  uint32_t layers = 30;
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
  // The DSpark draft's mask token id and hidden-state capture layers; the
  // draft config declares both under dflash_config.
  uint32_t maskToken = 125017;
  std::array<uint32_t, 2> stopTokens = {124900, 124900};
  std::array<uint32_t, 5> hiddenCaptureLayers = {2, 9, 17, 21, 27};
  uint32_t intermediateSize = 10752;
  // The norms' epsilon (norm_eps), which selects the _e5 kernel variants.
  float rmsEpsilon = 1e-5F;

  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    return layer < 64 && (attentionMask >> layer) & 1;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    uint32_t count = 0;
    for (uint32_t layer = 0; layer < 64; ++layer)
      count += (attentionMask >> layer) & 1;
    return count;
  }
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept { return 0; }
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {attentionLayerCount(), attentionKvHeads,
            attentionHeadDimension};
  }
  // Conv-state layers only: the FIFO of taps-1 rows, no recurrent matrix.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    return {layers - attentionLayerCount(), convolutionTaps - 1,
            convolutionDimension, 0, 0, 0};
  }
  [[nodiscard]] constexpr QwenMixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,     packedGdnWidth, packedFullWidth,
            convolutionDimension, gdnValueHeads,  gdnHeadDimension,
            attentionWidth, attentionHeadDimension, convolutionTaps};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * hiddenCaptureLayers.size();
  }
  bool operator==(const Lfm2Layout &) const = default;
};

using Lfm2Weights = TargetModelWeights<Lfm2Layout, DenseLayerWeights>;

[[nodiscard]] Lfm2Weights
loadLfm2Weights(metal::MetalBackend &backend, Lfm2Layout layout,
                const TargetFiles<Lfm2Layout> &files);

} // namespace richengine::model

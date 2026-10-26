#pragma once

// The canonical target layout: a POD superset of every family's old layout
// struct (the QwenHybridLayout derivatives, Dense, Granite,
// the LFM2 layouts, Gemma4Moe and DiffusionGemma). What used to be per-struct
// fields and static constexpr members are data members here; the family's
// preset in its model header fills them and the generic member functions
// below replace the per-family implementations the loaders and the geometry
// pass called through the template parameter. `arch` is the registry
// discriminator a ModelRegistry registration keys on.
//
// FfnKind, MixerGeometry and kEmbeddingMagic moved here from the deleted
// QwenHybridLayout.hpp template. kGdnConvolutionTaps stays in
// StateLayout.hpp, where the state accounting declares it.
#include "model/StateLayout.hpp"
#include "ops/GDN.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace richengine::model {

enum class FfnKind : uint8_t { Dense, SparseMoe };

// The magic of a hybrid target's packed embedding file, whatever its family.
inline constexpr std::string_view kEmbeddingMagic = "MDFE0001";

// Sizes of the mixer sections in a packed layer file.
struct MixerGeometry final {
  uint32_t hiddenSize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionHeadDimension = 0;
  uint32_t convolutionTaps = 0;
};

// The capture slots the geometry's captureLayerValues holds
// (TargetModelGeometry::maximumCaptureLayers); every family's hidden-state
// capture list fits it.
inline constexpr uint32_t kMaximumCaptureLayers = 8;

// DiffusionGemma's denoising schedule (moved from DiffusionGemma.hpp, whose
// copy Phase B deletes). A non-diffusion target leaves the default and the
// trunk never reads it.
struct DiffusionSchedule final {
  uint32_t canvasLength = 256;
  uint32_t maxDenoisingSteps = 48;
  // The per-step temperature interpolates t = tMin + (tMax - tMin) * step /
  // maxDenoisingSteps (0.4 -> 0.8 over 48 steps).
  float tMin = 0.4F;
  float tMax = 0.8F;
  // A position is accepted when the exclusive prefix sum of the ascendingly
  // sorted entropies at its rank stays within this budget.
  float entropyBound = 0.1F;
  // Early exit: the canvas argmax unchanged for stabilityThreshold
  // consecutive steps and mean entropy below confidenceThreshold.
  float confidenceThreshold = 0.005F;
  uint32_t stabilityThreshold = 1;
  // The token that fills canvas positions past the first stop token of a
  // committed canvas (Gemma's <pad>, id 0; the manifest may state it).
  uint32_t padToken = 0;

  [[nodiscard]] constexpr float temperature(uint32_t step) const noexcept {
    return tMin + (tMax - tMin) * static_cast<float>(step) /
                      static_cast<float>(maxDenoisingSteps);
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return canvasLength && !(canvasLength % kv::kPageTokens) &&
           maxDenoisingSteps && tMin > 0.0F && tMax >= tMin &&
           entropyBound >= 0.0F && confidenceThreshold >= 0.0F &&
           stabilityThreshold;
  }

  bool operator==(const DiffusionSchedule &) const = default;
};

struct TargetLayout final {
  // The registry discriminator: "qwen3_8", "ornith9", "qwen3_6moe",
  // "minicpm_v47_moe", "dense", "lfm2", "lfm2_moe", "lfm2_moe_24b",
  // "granite", "gemma4" or "diffusion_gemma".
  std::string_view arch{};
  // The GGUF schema arch the family's tensors follow ("llama", "granite",
  // "lfm2", "lfm2moe"); empty resolves to "qwen35"/"qwen35moe" on the MoE
  // split downstream.
  std::string_view ggufArch{};
  // The output head shares the token embedding (the LFM2 targets).
  bool tiedOutput = false;
  // The packed file magics the family's sections carry.
  std::string_view layerMagic{};
  std::string_view headMagic{};
  FfnKind ffnKind = FfnKind::Dense;
  // The packed QKV rows' per-head stride: the GDN hybrids interleave a gate
  // row per query head ([q|gate], stride 2); every other target packs the
  // query heads alone.
  uint32_t attentionQueryStride = 1;
  // The per-head RMS norms on the queries and keys (all but Dense and
  // Granite).
  bool attentionQkNorm = false;
  // No MLX or GGUF loader maps this family's tensors; packages are packed
  // at installation (Gemma 4's packedOnly).
  bool packedOnly = false;
  // Dense-layout targets opt into block-layout MLX images.
  bool mlxBlockWeights = false;
  // Whether an MLX or oQ safetensors source can plan this family's images:
  // Granite clears it (its sources are packed or GGUF only).
  bool mlxImageable = true;
  // The recurrent mixer's convolution taps: kGdnConvolutionTaps on the GDN
  // hybrids, the LFM2 conv_L_cache's 3, 0 on a target without convolutions.
  uint32_t convolutionTaps = 0;
  // The leading dense-FFN layers (LFM2-MoE's num_dense_layers); 0 puts the
  // MoE block on every layer.
  uint32_t denseLayers = 0;
  // The LFM2 families' explicit full-attention layer set; 0 selects the
  // period and global-schedule rules isFullAttentionLayer applies.
  uint64_t attentionMask = 0;
  uint32_t maximumContextTokens = kv::kMaximumLogicalTokens;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  // The GDN hybrids' schedule: every fullAttentionPeriod-th layer is full
  // attention. 1 or 0 where the mask or the all-attention rule decides.
  uint32_t fullAttentionPeriod = 0;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  // The layers whose hidden states the draft reads; the first
  // captureLayerCount entries are live.
  std::array<uint32_t, kMaximumCaptureLayers> hiddenCaptureLayers{};
  uint32_t captureLayerCount = 0;
  // The compiled attention kernels the layers bind (ops/PagedKv.hpp):
  // attentionCore the primary geometry, altAttentionCore the
  // globalLayerMask layers (Gemma's k_eq_v globals).
  kv::Core attentionCore = kv::Core::Unspecified;
  kv::Core altAttentionCore = kv::Core::Unspecified;
  // The FFN sizes: the dense blocks' intermediate and the MoE blocks'
  // routing.
  uint32_t intermediateSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  // Gemma 4's second attention geometry: every globalLayerPeriod-th layer
  // (the last of each period) is global — altKvHeads KV heads of
  // altHeadDimension, a rotary of altRotaryPairs at altRotaryTheta, k_eq_v
  // and full causal — while the rest attend over the trailing
  // slidingWindowTokens. Zero selects the single-geometry targets.
  uint32_t globalLayerPeriod = 0;
  uint32_t altKvHeads = 0;
  uint32_t altHeadDimension = 0;
  uint32_t altRotaryPairs = 0;
  float altRotaryTheta = 0.0F;
  uint32_t slidingWindowTokens = 0;
  // Gemma's expert slabs pad the logical expertIntermediateSize to the
  // tiles' column granularity; the shared expert's dense GeGLU pads its
  // intermediate to packedSharedWidth alike.
  uint32_t packedExpertWidth = 0;
  uint32_t sharedIntermediateSize = 0;
  uint32_t packedSharedWidth = 0;
  // The final logit softcap and the tied-embedding scale (0 derives
  // sqrt(hiddenSize) on a Gemma preset); each 0 disables.
  float logitSoftcap = 0.0F;
  float embeddingScale = 0.0F;
  // Granite's attention_multiplier: the fixed softmax scale the kernels
  // apply to q*k scores, replacing the implicit 1/sqrt(d).
  float attentionScale = 0.0F;
  // The norms' epsilon (norm_eps), which selects the _e5 kernel variants.
  float rmsEpsilon = 0.0F;
  // The LFM2 decision encoder's deviations from the causal target: a
  // bidirectional encoder, decisionBlocks trailing bespoke decision layers
  // whose KV head count and FFN width differ.
  bool encoder = false;
  uint32_t decisionBlocks = 0;
  uint32_t altAttentionKvHeads = 0;
  uint32_t altIntermediateSize = 0;
  // The compiled GDN kernel set the recurrent blocks bind (ops/GDN.hpp);
  // the family's maker declares it and geometryFor copies it through.
  ops::GdnCore gdnCore = ops::GdnCore::Unspecified;
  // Whether the MoE blocks carry a shared-expert route; LFM2-MoE and Gemma
  // clear it (Gemma's shared block runs as a dense GeGLU beside the router).
  bool moeSharedExpert = true;
  // The DiffusionGemma trunk's denoising schedule; inert on every other
  // target.
  DiffusionSchedule diffusion{};

  // The full-attention layer set: the LFM2 mask where set, else every
  // fullAttentionPeriod-th layer on the hybrids, else Gemma's globals,
  // else every layer (the dense and Granite targets).
  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    if (attentionMask) return layer < 64 && (attentionMask >> layer) & 1;
    if (fullAttentionPeriod > 1)
      return (layer + 1) % fullAttentionPeriod == 0;
    if (globalLayerPeriod) return isGlobalAttentionLayer(layer);
    return layer < layers;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    if (attentionMask) {
      uint32_t count = 0;
      for (uint32_t layer = 0; layer < 64; ++layer)
        count += (attentionMask >> layer) & 1;
      return count;
    }
    // Every layer is attention under a global schedule or a period of 1.
    return fullAttentionPeriod > 1 ? layers / fullAttentionPeriod : layers;
  }
  // Gemma's global layers: the last layer of each period.
  [[nodiscard]] constexpr bool
  isGlobalAttentionLayer(uint32_t layer) const noexcept {
    return globalLayerPeriod &&
           layer % globalLayerPeriod == globalLayerPeriod - 1;
  }
  [[nodiscard]] constexpr uint64_t globalLayerMask() const noexcept {
    uint64_t mask = 0;
    for (uint32_t layer = 0; layer < layers; ++layer)
      mask |= uint64_t{isGlobalAttentionLayer(layer)} << layer;
    return mask;
  }
  [[nodiscard]] constexpr uint32_t actualGdnWidth() const noexcept {
    return gdnHeadDimension
               ? convolutionDimension + attentionWidth + 2 * gdnValueHeads
               : 0;
  }
  // The pool layout: the primary geometry everywhere, the alt fields and
  // the global-layer mask only under a dual-geometry schedule. Gemma's
  // kernels bind an explicit scoreScale of 1; Granite's attentionScale
  // replaces the implicit 1/sqrt(d) on the others that declare one.
  [[nodiscard]] constexpr kv::Layout kvLayout() const noexcept {
    return {globalLayerPeriod ? layers : attentionLayerCount(),
            attentionKvHeads,
            attentionHeadDimension,
            kv::Format::Int8,
            globalLayerPeriod ? 1.0F : attentionScale,
            altKvHeads,
            altHeadDimension,
            globalLayerMask(),
            0,
            attentionCore,
            altAttentionCore};
  }
  [[nodiscard]] constexpr kv::Layout kvLayout(kv::Format format) const noexcept {
    kv::Layout layout = kvLayout();
    layout.format = format;
    return layout;
  }
  // The recurrent slots: the GDN hybrids keep the taps-1 convolution FIFO
  // and the value-head matrix; the LFM2 convolutions keep the FIFO alone; a
  // convolution-free target has no state at all.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    if (!convolutionTaps) return {};
    const uint32_t stateLayers = layers - attentionLayerCount();
    if (gdnHeadDimension)
      return {stateLayers, convolutionTaps - 1, convolutionDimension,
              gdnValueHeads, gdnHeadDimension, gdnHeadDimension};
    return {stateLayers, convolutionTaps - 1, convolutionDimension, 0, 0, 0};
  }
  [[nodiscard]] constexpr MixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,           packedGdnWidth,        packedFullWidth,
            convolutionDimension, gdnValueHeads,         gdnHeadDimension,
            attentionWidth,       attentionHeadDimension, convolutionTaps};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }
  // One of the trailing bespoke decision blocks (LFM2's d1-omni).
  [[nodiscard]] constexpr bool isDecisionLayer(uint32_t layer) const noexcept {
    return decisionBlocks && layer < layers &&
           layer >= layers - decisionBlocks;
  }
  [[nodiscard]] constexpr uint32_t kvHeadsAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? altKvHeads
           : isDecisionLayer(layer)      ? altAttentionKvHeads
                                         : attentionKvHeads;
  }
  [[nodiscard]] constexpr uint32_t headDimensionAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? altHeadDimension
                                         : attentionHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t rotaryPairsAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? altRotaryPairs : rotaryPairs;
  }
  [[nodiscard]] constexpr float rotaryThetaAt(uint32_t layer) const noexcept {
    return isGlobalAttentionLayer(layer) ? altRotaryTheta : rotaryTheta;
  }
  [[nodiscard]] constexpr uint32_t attentionWidthAt(uint32_t layer) const noexcept {
    return attentionQueryHeads * headDimensionAt(layer);
  }
  // The fused QKV row width: queries, keys and — the locals only — values
  // (Gemma's globals run k_eq_v, so their rows carry no V block).
  [[nodiscard]] constexpr uint32_t packedWidthAt(uint32_t layer) const noexcept {
    const uint32_t values = isGlobalAttentionLayer(layer) ? 1 : 2;
    return attentionQueryHeads * headDimensionAt(layer) +
           values * kvHeadsAt(layer) * headDimensionAt(layer);
  }
  // The wider of the two layer kinds' rows under a global schedule.
  [[nodiscard]] constexpr uint32_t maximumPackedWidth() const noexcept {
    if (!globalLayerPeriod) return packedWidthAt(0);
    const uint32_t local = packedWidthAt(0);
    const uint32_t global = packedWidthAt(globalLayerPeriod - 1);
    return local > global ? local : global;
  }
  [[nodiscard]] constexpr uint32_t intermediateSizeAt(uint32_t layer) const noexcept {
    return isDecisionLayer(layer) ? altIntermediateSize : intermediateSize;
  }
  // The dense-FFN layers come first; every later layer's FFN is the MoE.
  [[nodiscard]] constexpr bool isMoeLayer(uint32_t layer) const noexcept {
    return layer >= denseLayers;
  }

  bool operator==(const TargetLayout &) const = default;
};

} // namespace richengine::model

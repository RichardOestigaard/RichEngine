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
#include "ops/Mamba2.hpp"
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
  // The head-wise gate projection's padded storage rows (LFM3: 256).
  uint32_t attentionGateRows = 0;
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

// The layer-schedule masks (attentionMask, moeMask): one bit per block.
// 128 bits wide — nemotron_h_moe's Bolt targets 88 blocks.
using LayerMask = unsigned __int128;

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
  LayerMask attentionMask = 0;
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
  // captureLayerCount entries are live. captureStreams marks a hyper-
  // connected target whose capture is the layer's full streamWidth()
  // residual state rather than a hiddenSize hidden row — maxwell-moe's MTP
  // draft consumes the layer-44 streams.
  std::array<uint32_t, kMaximumCaptureLayers> hiddenCaptureLayers{};
  uint32_t captureLayerCount = 0;
  bool captureStreams = false;
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
  // LFM3's declared global-layer set: the alt geometry applies to these
  // attention layers where globalLayerPeriod's periodic rule does not
  // describe the schedule. altCarriesValues keeps a V block in the alt
  // layers' packed rows (LFM3; Gemma's k_eq_v globals carry none), and
  // altRotaryRealPairs zeroes the alt rope table's tail past them (the
  // zero-tail partial rotary).
  uint64_t globalAttentionMask = 0;
  // The head-wise gate projection's storage width: its 16 output rows pad
  // to one 256-row Q4 tile (gate weights outside row 15 are zeros the
  // kernels never read).
  uint32_t attentionGateRows = 0;
  bool altCarriesValues = false;
  uint32_t altRotaryRealPairs = 0;
  // LFM3's sandwich norm: each block also normalizes the mixer output and
  // the FFN output before they join the residual.
  bool sandwichNorm = false;
  // LFM3's head-wise gated attention: a per-head sigmoid gate from a
  // gate_proj the packed QKV row does not interleave, applied to the
  // attention output before the out-projection.
  bool attentionHeadGate = false;
  // Expert slabs pad the logical expertIntermediateSize to the packed
  // format's tile granularity — Gemma's affine Q4 tiles, and Qwen4Exp's
  // GGUF block segments whose 256-row planes and per-expert slab stride
  // need a whole tile multiple (640 -> 768); the shared expert's dense
  // GeGLU pads its intermediate to packedSharedWidth alike.
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
  // NVIDIA Nemotron's ungated relu-squared experts (ops::MoeShape's
  // ungatedExperts + sharedExpertUnitWeight): the blocks have no gate
  // slab and the shared expert runs at unit weight of
  // sharedIntermediateSize.
  bool moeUngatedExperts = false;
  // The routed experts' norm_topk_prob scaling factor (GGUF
  // expert_weights_scale; Nemotron's 2.5). Zero reads as 1.
  float moeRoutedScale = 0.0F;
  // The DiffusionGemma trunk's denoising schedule; inert on every other
  // target.
  DiffusionSchedule diffusion{};
  // The qwen4exp arch's additions, all zero on every other target
  // (upstream's QwenTargetDimensions): hyperConnections residual streams of
  // hiddenSize values and the mixes' low rank; the QSA block indexer every
  // full-attention layer carries, of indexerHeads heads of
  // indexerHeadDimension values scoring blocks of indexerBlockTokens tokens
  // (indexerTopBlocks kept per row); and the PLE n-gram embedding feeding
  // layer pleLayer — 2- to pleNgram-gram hashes, pleHeadsPerNgram heads per
  // n-gram of pleHeadDimension table values each, gated through a
  // dilated convolution of pleConvolutionTaps taps pleNgram rows apart.
  uint32_t hyperConnections = 0;
  uint32_t hyperRank = 0;
  uint32_t indexerHeads = 0;
  uint32_t indexerHeadDimension = 0;
  uint32_t indexerTopBlocks = 0;
  uint32_t indexerBlockTokens = 0;
  uint32_t pleLayer = 0;
  uint32_t pleNgram = 0;
  uint32_t pleHeadsPerNgram = 0;
  uint32_t pleHeadDimension = 0;
  uint32_t pleConvolutionTaps = 0;
  // The PLE hash's end-of-sequence token (an n-gram cuts there) and the
  // token it reads at an image's prompt positions (<|image_pad|>).
  uint32_t pleEosToken = 0;
  uint32_t pleImageToken = 0;
  // The maxwell_moe arch's additions, all zero on every other target. Its
  // indexer keeps every token's post-RoPE index key in the index plane
  // (indexerPerTokenKeys), not one pooled key per indexer block, so no
  // state cell keeps pending keys. Each row's route, one per KV group
  // (indexerHeads is the group count), selects indexerRemoteBlocks blocks
  // of indexerBlockTokens tokens by per-token block-max score inside the
  // eligible band — past the indexerPrefixBlocks prefix blocks and before
  // the indexerLocalBlocks local blocks ending at the row's own block,
  // both always attended. The manifold-constrained hyper-connection
  // (mhcMixWidth the mapping's width — streams*streams + 2*streams — and
  // mhcSinkhornIterations its Sinkhorn passes) replaces the qwen4exp
  // mixes' low-rank gate, and mhcHead the learned streams-to-one output
  // contraction the model runs before its final norm.
  uint32_t indexerRouteGroups = 0;
  uint32_t indexerPrefixBlocks = 0;
  uint32_t indexerLocalBlocks = 0;
  uint32_t indexerRemoteBlocks = 0;
  bool indexerPerTokenKeys = false;
  uint32_t mhcMixWidth = 0;
  uint32_t mhcSinkhornIterations = 0;
  bool mhcHead = false;
  // The opt-in DuoAttention policy (kv::DuoPolicy, metal/abi/DuoAttention.h):
  // the streaming KV heads' sink and trailing window in tokens, head_mask one
  // bit per (attention ordinal, KV head) pair at bit 4*ordinal + head.
  // kvLayout() forwards it; the all-zero default is dense attention.
  kv::DuoPolicy duoAttention{};
  // The nemotron_h_moe arch's additions, all zero on every other target. A
  // block carries exactly one sub-layer after its input norm — the
  // attentionMask bits name the attention blocks, moeMask the MoE blocks,
  // the rest are Mamba-2 mixers — with no post norms and no second
  // sub-layer. The Mamba-2 geometry: mambaHeads heads of
  // mambaHeadDimension values over an mambaStateSize-wide state,
  // mambaGroups shared B/C groups, a conv input of convolutionDimension
  // rows (x | B | C), a mambaInnerSize gate and a packed in_proj of
  // packedGdnWidth storage rows (gate | conv | dt | pad — the 10304
  // logical rows pad to a tile), of which mambaTimeStepRank are the dt
  // rows. mambaCore binds the compiled kernel set (ops/Mamba2.hpp).
  LayerMask moeMask = 0;
  // LatentMoE (Bolt): the routed experts read and write a latent of this
  // width — ffn_latent_down projects the block input into it before the
  // router-expert pass and ffn_latent_up projects the expert output back —
  // while the router and the shared expert keep the full hidden width.
  uint32_t moeLatentSize = 0;
  uint32_t mambaHeads = 0;
  uint32_t mambaHeadDimension = 0;
  uint32_t mambaGroups = 0;
  uint32_t mambaStateSize = 0;
  uint32_t mambaInnerSize = 0;
  uint32_t mambaTimeStepRank = 0;
  // dt' = clamp(softplus(dt + dt_bias), mambaDtMinimum, mambaDtMaximum) —
  // nemotron's time_step_limit (0.001, 0.1).
  float mambaDtMinimum = 0.0f;
  float mambaDtMaximum = 0.0f;
  ops::Mamba2Core mambaCore = ops::Mamba2Core::Unspecified;

  // The full-attention layer set: the LFM2 mask where set, else every
  // fullAttentionPeriod-th layer on the hybrids, else Gemma's globals,
  // else every layer (the dense and Granite targets).
  [[nodiscard]] constexpr bool
  isFullAttentionLayer(uint32_t layer) const noexcept {
    if (attentionMask) return (attentionMask >> layer) & 1;
    if (fullAttentionPeriod > 1)
      return (layer + 1) % fullAttentionPeriod == 0;
    if (globalLayerPeriod) return isGlobalAttentionLayer(layer);
    return layer < layers;
  }
  [[nodiscard]] constexpr uint32_t attentionLayerCount() const noexcept {
    if (attentionMask) {
      uint32_t count = 0;
      for (uint32_t layer = 0; layer < 128; ++layer)
        count += (attentionMask >> layer) & 1;
      return count;
    }
    // Every layer is attention under a global schedule or a period of 1.
    return fullAttentionPeriod > 1 ? layers / fullAttentionPeriod : layers;
  }
  // Gemma's global layers are the last layer of each period; LFM3's are
  // the declared globalAttentionMask.
  [[nodiscard]] constexpr bool
  isGlobalAttentionLayer(uint32_t layer) const noexcept {
    if (globalLayerPeriod)
      return layer % globalLayerPeriod == globalLayerPeriod - 1;
    return layer < 64 && (globalAttentionMask >> layer) & 1;
  }
  [[nodiscard]] constexpr uint64_t globalLayerMask() const noexcept {
    uint64_t mask = 0;
    if (!globalLayerPeriod) {
      // altLayerMask indexes attention ordinals, not absolute layers:
      // fold the declared absolute mask down over the attention layers.
      if (!globalAttentionMask) return 0;
      uint32_t ordinal = 0;
      for (uint32_t layer = 0; layer < layers && layer < 64; ++layer) {
        if (!isFullAttentionLayer(layer)) continue;
        if (isGlobalAttentionLayer(layer)) mask |= uint64_t{1} << ordinal;
        ++ordinal;
      }
      return mask;
    }
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
    kv::Layout result{globalLayerPeriod ? layers : attentionLayerCount(),
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
    // A sparse-indexed target's pages carry the index plane (one pooled
    // key of indexerHeadDimension bf16 values per indexer block, or every
    // token's index key on a per-token index) and token-major values,
    // which its sparse gathers read contiguously.
    result.indexDimension = indexerHeads ? indexerHeadDimension : 0;
    result.indexPerToken = indexerPerTokenKeys;
    result.valuesTokenMajor = indexerHeads != 0;
    result.duo = duoAttention;
    return result;
  }
  [[nodiscard]] constexpr kv::Layout kvLayout(kv::Format format) const noexcept {
    kv::Layout layout = kvLayout();
    layout.format = format;
    return layout;
  }
  // The qwen4exp helpers: the residual's streams' values per row, the PLE
  // table's heads and their gathered width, and the per-state-cell
  // auxiliary bytes the PLE history and the QSA pending keys take
  // (GdnStateLayout::auxiliaryBytes).
  [[nodiscard]] constexpr bool hyperConnected() const noexcept {
    return hyperConnections != 0;
  }
  [[nodiscard]] constexpr uint32_t streamWidth() const noexcept {
    return hyperConnections * hiddenSize;
  }
  [[nodiscard]] constexpr uint32_t pleHeads() const noexcept {
    return pleNgram ? (pleNgram - 1) * pleHeadsPerNgram : 0;
  }
  [[nodiscard]] constexpr uint32_t pleWidth() const noexcept {
    return pleHeads() * pleHeadDimension;
  }
  // The PLE convolution's history rows: (taps - 1) dilated by the n-gram.
  [[nodiscard]] constexpr uint32_t pleHistory() const noexcept {
    return pleConvolutionTaps ? (pleConvolutionTaps - 1) * pleNgram : 0;
  }
  [[nodiscard]] constexpr uint64_t pleHistoryBytes() const noexcept {
    return uint64_t{pleHistory()} * streamWidth() *
           GdnStateLayout::bfloat16Bytes;
  }
  // The QSA indexer's pending keys per state cell: per attention layer the
  // raw keys (bf16) of the open block's tokens but its last. A per-token
  // indexer (maxwell_moe) has none — every token's key lands in the index
  // plane as it is written and the open block is never eligible.
  [[nodiscard]] constexpr uint64_t indexerPendingBytes() const noexcept {
    return indexerBlockTokens && !indexerPerTokenKeys
               ? uint64_t{attentionLayerCount()} * (indexerBlockTokens - 1) *
                     indexerHeadDimension * GdnStateLayout::bfloat16Bytes
               : 0;
  }
  // nemotron_h_moe's MoE sub-blocks (moeMask): they carry neither KV nor
  // recurrent state, which the state-layer count excludes.
  [[nodiscard]] constexpr uint32_t moeLayerCount() const noexcept {
    uint32_t count = 0;
    for (uint32_t layer = 0; layer < 128; ++layer)
      count += (moeMask >> layer) & 1;
    return count;
  }
  // A recurrent mixer: every non-attention block of the hybrids whose
  // blocks are attention-or-recurrent, and nemotron's non-MoE ones.
  [[nodiscard]] constexpr bool isRecurrentLayer(uint32_t layer) const noexcept {
    return !isFullAttentionLayer(layer) && !isMoeLayer(layer);
  }
  // The recurrent slots: the GDN hybrids keep the taps-1 convolution FIFO
  // and the value-head matrix; the LFM2 convolutions keep the FIFO alone; a
  // convolution-free target has no state at all. nemotron's Mamba-2 mixers
  // keep the conv-input FIFO and the heads x headDim x state matrix.
  // qwen4exp's cells carry the PLE history and QSA pending keys after them.
  [[nodiscard]] constexpr GdnStateLayout gdnStateLayout() const noexcept {
    if (!convolutionTaps) return {};
    const uint32_t stateLayers =
        layers - attentionLayerCount() - moeLayerCount();
    GdnStateLayout state =
        mambaHeads
            ? GdnStateLayout{stateLayers, convolutionTaps - 1,
                             convolutionDimension, mambaHeads,
                             mambaHeadDimension, mambaStateSize}
        : gdnHeadDimension
            ? GdnStateLayout{stateLayers, convolutionTaps - 1,
                             convolutionDimension, gdnValueHeads,
                             gdnHeadDimension, gdnHeadDimension}
            : GdnStateLayout{stateLayers, convolutionTaps - 1,
                             convolutionDimension, 0, 0, 0};
    state.auxiliaryBytes = pleHistoryBytes() + indexerPendingBytes();
    return state;
  }
  [[nodiscard]] constexpr MixerGeometry mixerGeometry() const noexcept {
    return {hiddenSize,           packedGdnWidth,        packedFullWidth,
            convolutionDimension, gdnValueHeads,         gdnHeadDimension,
            attentionWidth,       attentionHeadDimension, convolutionTaps,
            attentionGateRows};
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return captureStreams ? streamWidth() * captureLayerCount
                          : hiddenSize * captureLayerCount;
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
    const uint32_t values =
        isGlobalAttentionLayer(layer) && !altCarriesValues ? 1 : 2;
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
  // The MoE sub-layers: nemotron's moeMask bits where set, else the
  // dense-FFN layers come first and every later layer's FFN is the MoE.
  [[nodiscard]] constexpr bool isMoeLayer(uint32_t layer) const noexcept {
    if (moeMask) return (moeMask >> layer) & 1;
    return layer >= denseLayers;
  }

  bool operator==(const TargetLayout &) const = default;
};

} // namespace richengine::model

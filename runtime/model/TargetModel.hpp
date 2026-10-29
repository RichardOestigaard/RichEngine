#pragma once

#include "Model.hpp"
#include "TargetLayout.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/LfmConv.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "metal/abi/BlockIndexer.h"
#include "metal/abi/Sampling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace richengine::ops {
class FfnSplit;
} // namespace richengine::ops

namespace richengine::model {

class TargetLayerEncoder;

// Both supported targets bind the same mixer tensors per hybrid layer; only
// the FFN differs between them.
struct GdnMixerWeights final {
  ops::Projection inputProjection;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer decay;
  metal::MetalBuffer timeBias;
  ops::NormWeights mixerNorm;
  ops::Projection outputProjection;
  // The value-head order of outputProjection's input columns, in which the
  // GDN writes its output.
  ops::GdnHeadOrder outputHeadOrder = ops::GdnHeadOrder::Grouped;
};

struct AttentionMixerWeights final {
  ops::Projection inputProjection;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::Projection outputProjection;
  // LFM3's head-wise gate: hidden -> queryHeads sigmoid scalars on the
  // attention output. Empty on every other target.
  ops::Projection gateProjection;
};

// LFM2's double-gated short convolution: in_proj produces [B|C|x], the
// causal depthwise convolution runs over the FIFO state, and the result is
// C*conv fed to out_proj. `convolutionTapsMajor` records the weight's
// stored order ([tap][dim] in a GGUF, [dim][tap] otherwise).
struct LfmConvWeights final {
  ops::Projection inputProjection;
  metal::MetalBuffer convolutionWeights;
  ops::Projection outputProjection;
  bool convolutionTapsMajor = false;
};

using MixerWeights =
    std::variant<GdnMixerWeights, AttentionMixerWeights, LfmConvWeights>;

// A target's weights outside its layers and the record of every file
// its weights were read from.
struct TargetModelWeightsBase {
  ops::NormWeights finalNorm;
  ops::Projection logitsProjection;
  ops::EmbeddingWeights tokenEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

// A layer's dense FFN (SwiGLU/GeGLU): the gate, up and down projections.
struct DenseFfnWeights final {
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

// A leading dense layer's gated FFN of an LFM2-MoE target (moved from
// Lfm2Moe.hpp; the family header's copy goes away in Phase B).
struct Lfm2DenseFfn final {
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

// LFM2-MoE's per-layer FFN: dense for the leading denseLayers, the sparse
// MoE block after them.
struct Lfm2FfnWeights final {
  std::variant<Lfm2DenseFfn, ops::MoeWeights> ffn;
};

// Gemma 4's layer tensors besides the shared record's norms and mixer: the
// pre_ffn_norm feeds the shared expert and preFfnNormRouted (the
// checkpoint's pre_feedforward_layernorm_2) the routed experts; the router
// reads the post-attention residual through routerScale (learned
// per-dimension scale, bf16) after a scaleless RMS norm.
struct Gemma4FfnWeights final {
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

// The FFN kinds a target layer can carry: the plain dense triple, a routed
// MoE block (Qwen3.6-MoE, MiniCPM-V), LFM2-MoE's dense/MoE mix and Gemma's
// routed-plus-shared-expert block.
using TargetLayerFfn = std::variant<DenseFfnWeights, ops::MoeWeights,
                                    Lfm2FfnWeights, Gemma4FfnWeights>;

// One target layer: the mixer norm, the mixer, the FFN norm and the FFN.
// Gemma's extra norms and tensors live in its Gemma4FfnWeights.
struct TargetLayerWeights final {
  ops::NormWeights inputNorm;
  MixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  TargetLayerFfn ffn;
  // LFM3's sandwich norms: the mixer output's and the FFN output's, applied
  // before each joins the residual. Empty on every other target.
  ops::NormWeights postMixerNorm;
  ops::NormWeights postFfnNorm;
};

// The weights of a target: its canonical layout and every layer. Not final:
// DiffusionGemmaWeights derives from it to add the self-conditioning block
// and encoder scalars.
struct TargetModelWeights : TargetModelWeightsBase {
  TargetLayout layout;
  std::vector<TargetLayerWeights> layers;
};

// Runtime-visible tensor geometry shared by the supported hybrid
// targets. It describes semantics only; operators remain responsible for
// choosing device-specific Metal pipelines and compute tiles.
struct TargetModelGeometry final {
  static constexpr uint32_t maximumCaptureLayers = kMaximumCaptureLayers;

  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  // The tokenizer's vocabulary: the logits argmax's bound. The head section
  // and the logits buffers span paddedVocabularySize rows: a vocabulary that
  // is not a storage-tile multiple pads to one (MiniCPM-V 4.7's 248144).
  uint32_t vocabularySize = 0;
  uint32_t paddedVocabularySize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  // The compiled GDN kernel set the recurrent blocks bind (ops/GDN.hpp);
  // the family maker declares it, geometryFor copies it here.
  ops::GdnCore gdnCore = ops::GdnCore::Unspecified;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  // The position axes a target row carries: 3 for the M-RoPE families
  // (dim % 3 chooses the axis), 1 for the dense and LFM2 targets.
  uint32_t ropeAxes = 3;
  uint32_t denseIntermediateSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  FfnKind ffnKind = FfnKind::Dense;
  // The stateLayout.layers recurrent slots are gdnLayers GDN layers plus
  // convLayers LFM2 convolution layers; a pure dense target has neither.
  uint32_t gdnLayers = 0;
  uint32_t convLayers = 0;
  // The convolution FIFO's kernel taps (kGdnConvolutionTaps for GDN, the
  // LFM2 conv_L_cache for conv layers).
  uint32_t convolutionTaps = 0;
  // The attention rows' [query|gate] packing and the per-head RMS norms:
  // the GDN hybrids keep both, the dense target neither, LFM2 only the
  // norms.
  bool attentionQueryGate = true;
  bool attentionQkNorm = true;
  // The weight layout every sparse MoE block of the target shares, and in a
  // GGUF the format of most of its routed expert weights. A target whose
  // MoE blocks have no shared expert (LFM2-MoE) clears moeSharedExpert.
  ops::WeightLayout moeLayout = ops::WeightLayout::Affine64;
  uint32_t moeExpertFormat = GGUF_FMT_COUNT;
  bool moeSharedExpert = true;
  // Collection-time marker: a mixed-FFN target's first MoE block seeds
  // moeLayout, not its first layer.
  bool moeSeeded = false;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  std::array<uint32_t, maximumCaptureLayers> captureLayerValues{};
  uint32_t captureLayerCount = 0;
  kv::Layout kvLayout{};
  GdnStateLayout stateLayout{};
  // Gemma 4's second attention geometry: layers whose bit is set in
  // altAttentionMask are the global layers — altKvHeads KV heads of
  // altHeadDimension, rotary of altRotaryPairs at altRotaryTheta, k_eq_v and
  // full causal. The rest run the primary geometry masked to the trailing
  // slidingWindowTokens. kvLayout carries the same mask and alternate sizes.
  uint64_t altAttentionMask = 0;
  // altAttentionMask is attention-ordinal space (it must equal
  // kvLayout.altLayerMask); hybrid schedules like LFM3's — where attention
  // ordinals and absolute layer indices differ — additionally carry the
  // absolute-index mask the layer-kind checks read.
  uint64_t altAbsoluteMask = 0;
  uint32_t altKvHeads = 0;
  uint32_t altHeadDimension = 0;
  uint32_t altRotaryPairs = 0;
  float altRotaryTheta = 0.0F;
  uint32_t slidingWindowTokens = 0;
  // The alt layers' packed rows carry a V block (LFM3's globals — Gemma's
  // k_eq_v alt layers store keys only).
  bool altCarriesValues = false;
  // LFM3's additions: the sandwich norms between each sublayer and the
  // residual, the per-head sigmoid gate on the attention output, and the
  // alt rope table's real prefix (its tail carries zeros — the zero-tail
  // partial rotary of the global layers).
  bool sandwichNorm = false;
  bool attentionHeadGate = false;
  // The gate projection's padded storage rows — the gate buffer's stride.
  uint32_t attentionGateRows = 0;
  uint32_t altRotaryRealPairs = 0;
  // Gemma 4's MoE routes from the post-attention residual and runs its
  // shared expert as a dense GeGLU block beside the routed experts, whose
  // slab width is the padded packed width (768), not the logical 704.
  bool gemmaMoe = false;
  // sqrt(hiddenSize) embedding scale and the final logit softcap; each 0
  // disables.
  float embeddingScale = 0.0F;
  float logitSoftcap = 0.0F;
  // qwen4exp's PLE n-gram length (0 elsewhere): the runtime sizes each
  // lane's prior-token span by it.
  uint32_t pleNgram = 0;
  // Distinct operator requirements, collected from the loaded weights.
  std::vector<ops::ProjectionShape> prefillProjections;
  std::vector<ops::ProjectionShape> decodeProjections;
  std::vector<ops::ProjectionShape> gateUpProjections;

  [[nodiscard]] constexpr uint32_t gdnKeyWidth() const noexcept {
    return gdnKeyHeads * gdnHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }
  [[nodiscard]] constexpr ops::MoeShape moeShape() const noexcept {
    return {hiddenSize, experts, expertsPerToken, expertIntermediateSize, moeLayout, moeExpertFormat,
            moeSharedExpert};
  }
  // The widest FFN intermediate of the target: a mixed target (LFM2-MoE)
  // carries both a dense intermediate for its leading layers and the MoE
  // blocks' expert width.
  [[nodiscard]] constexpr uint32_t ffnScratchWidth() const noexcept {
    return std::max(denseIntermediateSize,
                    ffnKind == FfnKind::SparseMoe ? expertIntermediateSize : uint32_t{0});
  }
  [[nodiscard]] constexpr std::span<const uint32_t>
  captureLayers() const noexcept {
    return {captureLayerValues.data(), captureLayerCount};
  }
  // The capture slot of `layer`, whose output the draft reads.
  [[nodiscard]] constexpr std::optional<uint32_t> captureSlot(uint32_t layer) const noexcept {
    const auto layers = captureLayers();
    const auto found = std::find(layers.begin(), layers.end(), layer);
    if (found == layers.end()) return std::nullopt;
    return static_cast<uint32_t>(found - layers.begin());
  }
  [[nodiscard]] constexpr ops::GdnShape gdnShape() const noexcept {
    return {gdnKeyHeads, gdnValueHeads, gdnHeadDimension,
            convolutionDimension, packedGdnWidth, gdnCore};
  }
  [[nodiscard]] constexpr ops::LfmConvShape convShape() const noexcept {
    return {convolutionDimension, convolutionTaps, packedGdnWidth};
  }
  // The dual-geometry accessors: without altAttentionMask every layer takes
  // the primary geometry. `layer` is the absolute index; a hybrid target's
  // altAbsoluteMask names it directly, a dense one shares ordinals.
  [[nodiscard]] constexpr bool isAltAttentionLayer(uint32_t layer) const noexcept {
    const uint64_t mask = altAbsoluteMask ? altAbsoluteMask : altAttentionMask;
    return layer < 64 && (mask >> layer) & 1;
  }
  // The per-layer kernel layout: attentionLayers 1, the layer kind's KV
  // geometry and, on the sliding layers, the window. Kernel names and
  // prefill/verify plans derive from it; it never addresses pages.
  [[nodiscard]] constexpr kv::Layout layerKvLayout(bool alt) const noexcept {
    kv::Layout layout = kvLayout;
    layout.attentionLayers = 1;
    layout.altLayerMask = 0;
    layout.altKvHeads = 0;
    layout.altHeadDimension = 0;
    layout.windowTokens = 0;
    // The per-kind declared core: the alt geometry carries its own binding.
    layout.attentionCore = alt ? kvLayout.altAttentionCore
                               : kvLayout.attentionCore;
    layout.altAttentionCore = kv::Core::Unspecified;
    if (alt) {
      layout.kvHeads = altKvHeads;
      layout.headDimension = altHeadDimension;
    } else {
      layout.windowTokens = slidingWindowTokens;
    }
    return layout;
  }
  [[nodiscard]] constexpr uint32_t layerHeadDimension(uint32_t layer) const noexcept {
    return isAltAttentionLayer(layer) ? altHeadDimension : attentionHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t layerKvHeads(uint32_t layer) const noexcept {
    return isAltAttentionLayer(layer) ? altKvHeads : attentionKvHeads;
  }
  [[nodiscard]] constexpr uint32_t rotaryPairsAt(uint32_t layer) const noexcept {
    return isAltAttentionLayer(layer) ? altRotaryPairs : rotaryPairs;
  }
  // The packed QKV row width of `layer`: the globals' rows have no V block
  // unless the target declares they carry values (LFM3's globals do).
  [[nodiscard]] constexpr uint32_t layerPackedWidth(uint32_t layer) const noexcept {
    if (isAltAttentionLayer(layer))
      return attentionQueryHeads * altHeadDimension +
             altKvHeads * altHeadDimension * (altCarriesValues ? 2 : 1);
    return packedFullWidth;
  }
  [[nodiscard]] constexpr uint32_t layerAttentionWidth(uint32_t layer) const noexcept {
    return attentionQueryHeads * layerHeadDimension(layer);
  }
  // The buffer widths a dual-geometry target needs: the wider of the two
  // layer kinds'.
  [[nodiscard]] constexpr uint32_t maximumAttentionWidth() const noexcept {
    return altAttentionMask
               ? std::max(attentionWidth, attentionQueryHeads * altHeadDimension)
               : attentionWidth;
  }
  [[nodiscard]] constexpr uint32_t maximumPackedWidth() const noexcept {
    const uint32_t alt =
        altAttentionMask
            ? attentionQueryHeads * altHeadDimension +
                  altKvHeads * altHeadDimension * (altCarriesValues ? 2 : 1)
            : 0;
    return std::max(packedFullWidth, alt);
  }
  [[nodiscard]] constexpr uint32_t maximumHeadDimension() const noexcept {
    return std::max(attentionHeadDimension, altHeadDimension);
  }
  // One chunk's per-layer KV staging elements: max(kvHeads, altKvHeads
  // scaled by the head widths).
  [[nodiscard]] constexpr uint32_t chunkLayerWidth() const noexcept {
    return std::max(attentionKvHeads * attentionHeadDimension,
                    altAttentionMask ? altKvHeads * altHeadDimension : 0u);
  }
  // The layout itself was checked by requireTargetLayout when the target
  // loaded. The projection lists hold every projection the weights dispatch,
  // which each have sizes.
  [[nodiscard]] bool valid() const noexcept {
    const auto sized = [](const std::vector<ops::ProjectionShape> &shapes) {
      return !shapes.empty() && std::all_of(shapes.begin(), shapes.end(), [](const auto &shape) {
        return shape.outputSize && shape.inputSize;
      });
    };
    const bool recurrentShape =
        gdnLayers ? gdnShape().valid()
                  : convLayers ? convShape().valid() : true;
    return captureLayerCount && captureLayerCount <= maximumCaptureLayers &&
           stateLayout.layers + kvLayout.attentionLayers == layers &&
           stateLayout.layers == gdnLayers + convLayers &&
           (!convLayers || attentionQueryHeads / attentionKvHeads * attentionKvHeads == attentionQueryHeads) &&
           recurrentShape &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           kvLayout.altLayerMask == altAttentionMask &&
           // The alternate regions exist exactly for the marked layers.
           (!altAttentionMask ||
            (altKvHeads && altHeadDimension && altRotaryPairs &&
             altRotaryTheta > 0.0F && slidingWindowTokens &&
             kvLayout.altKvHeads == altKvHeads &&
             kvLayout.altHeadDimension == altHeadDimension)) &&
           sized(prefillProjections) && sized(decodeProjections) &&
           (ffnKind == FfnKind::Dense
                ? denseIntermediateSize && sized(gateUpProjections)
                : moeShape().valid() &&
                      // A mixed target's leading dense FFN layers.
                      (!denseIntermediateSize ||
                       sized(gateUpProjections)));
  }
};

struct TargetModelPrefillCapture final {
  uint32_t sourceStart = 0;
  uint32_t destinationStart = 0;
  uint32_t rows = 0;
};

struct TargetModelPrefillSequence final {
  uint32_t rowBegin = 0;
  uint32_t rows = 0;
  uint32_t attentionStride = 0;
  uint64_t queryOffset = 0;
  uint64_t kvOffset = 0;
  kv::ChunkedPrefillParams chunk;
  metal::MetalBuffer pageTable;
  std::span<const metal::MetalBuffer> convolutionIn;
  std::span<const metal::MetalBuffer> convolutionOut;
  std::span<const metal::MetalBuffer> recurrentIn;
  std::span<const metal::MetalBuffer> recurrentOut;
  std::array<TargetModelPrefillCapture, 2> captures{};
  uint32_t captureCount = 0;
  // The sequence's auxiliary state cell region (the tail of its GDN cell
  // past the convolution and recurrent bytes, GdnStateLayout::auxiliaryBytes):
  // the views into the cells its rows read and write. Only targets whose
  // layout reserves auxiliary bytes bind them — qwen4exp's PLE convolution
  // history (first pleHistoryBytes) and QSA pending keys; empty elsewhere.
  metal::MetalBuffer auxiliaryIn;
  metal::MetalBuffer auxiliaryOut;
  // Its pages' block index, for a layout with one (ops::BlockIndex::tables):
  // qwen4exp's QSA. Written beside the page table (synchronizedPageTable).
  metal::MetalBuffer indexTable;
  // The QSA block records the sequence's rows complete (one per block whose
  // last token is a row of the sequence), computed host-side where the RoPE
  // positions live. Empty on targets without a block index.
  std::span<const BlockIndexerBlock> qsaBlocks{};
  // The row ids the sequence's layer encoder gathers — qwen4exp's PLE table
  // rows: pleHeads() ids per row, computed on the CPU by qwen4ExpPleRows
  // during the packed prefill's input preparation. Empty on targets without
  // a host-computed row table.
  std::span<const uint32_t> pleRowIds{};
  // A canvas (DiffusionGemma decoder) sequence: its chunk's committed_tokens
  // is the read-only prefix length, its rows the canvas positions, and its
  // page table the prefix pages followed by the canvas scratch pages. The
  // bidirectional canvas splits replace the causal prefill splits; every
  // other dispatch is the ordinary prefill layer's.
  bool canvas = false;
};

struct TargetModelPrefillBuffers final {
  // Split projections of chunks of up to 32 rows (LinearGguf.cpp).
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer captured;
  metal::MetalBuffer gdnPacked;
  metal::MetalBuffer gdnQueries;
  metal::MetalBuffer gdnKeys;
  metal::MetalBuffer gdnValues;
  metal::MetalBuffer gdnDecay;
  metal::MetalBuffer gdnBeta;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseGateScratch;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer downProjectionSums;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  // The global layers' rope tables (altRotaryPairs/altRotaryTheta); empty on
  // single-geometry targets.
  metal::MetalBuffer ropeCosAlt;
  metal::MetalBuffer ropeSinAlt;
  // Gemma's MoE combine binds a zeroed hidden-width residual so the routed
  // sum lands unmixed; empty elsewhere.
  metal::MetalBuffer zeroResidual;
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  // WY/UT scratch for the chunked GDN scan; empty keeps the serial scan.
  metal::MetalBuffer gdnChunkScratch;
  ops::MoeScratch moe;

  // The dense FFN's buffers among these.
  [[nodiscard]] ops::PrefillFfnBuffers ffn() const {
    return {normalized, projectionSums, denseGateScratch, denseIntermediate, downProjectionSums, linearScratch};
  }
};

struct TargetModelVerifyBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  // The global layers' rope tables (altRotaryPairs/altRotaryTheta); empty on
  // single-geometry targets.
  metal::MetalBuffer ropeCosAlt;
  metal::MetalBuffer ropeSinAlt;
  // Gemma's layer scratch: the post-attention residual the FFN's router and
  // experts both read, and the zeroed rows the routed combine adds to.
  metal::MetalBuffer gemmaResidual;
  metal::MetalBuffer zeroResidual;
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer denseGateScratch;
  std::span<const metal::MetalBuffer> gdnPacked;
  std::span<const metal::MetalBuffer> gdnMixed;
  // The lanes' GDN decode tape (folded commit): two parity slots per lane,
  // each one step's rows of every GDN layer. Empty when the fold is off.
  metal::MetalBuffer gdnTape;
  std::span<const metal::MetalBuffer> gdnDecay;
  std::span<const metal::MetalBuffer> gdnBeta;
  std::span<const metal::MetalBuffer> chunkKeys;
  std::span<const metal::MetalBuffer> chunkValues;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      pageTables;
  // Each lane's pages' block index, for a layout with one (qwen4exp's QSA;
  // ops::BlockIndex::tables). Empty elsewhere.
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      indexTables{};
  // Each lane's QSA block records (lane-major, laneRecords per lane), the
  // blocks its verify rows complete; the layer encoder copies them into its
  // own scratch in beginVerify. Empty without a block index.
  std::span<const BlockIndexerBlock> qsaBlocks{};
  // Fused greedy head (Ornith-9B vocabulary shape): when fusedHead is set and
  // every lane is greedy and unconstrained, the head writes per-tile argmax
  // partials into these instead of materializing logits.
  bool fusedHead = false;
  metal::MetalBuffer headArgmaxValues;
  metal::MetalBuffer headArgmaxIndices;
  HeadArgmaxParams headArgs{};
  // Tree verify only: the selector's node tables, the input pass's ancestor
  // masks and the capture staging the post-acceptance gather reads. Empty in
  // a chain batch.
  metal::MetalBuffer treeNodes;
  metal::MetalBuffer treeCounts;
  metal::MetalBuffer treeMasks;
  metal::MetalBuffer capturedPath;
  ops::MoeScratch moe;
  // Each lane's verify rows (ops::DraftSelector::addVerifyRows), which a
  // sparse-MoE target's steps route alone (ops::MoeBuffers::verifyRows).
  // Empty on a dense target or a tree batch.
  metal::MetalBuffer verifyRows;
};

struct TargetModelCommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextStates;
  metal::MetalBuffer retainedCounts;
};

[[nodiscard]] TargetModelGeometry
targetModelGeometry(const TargetModelWeights &weights);

// Builds the shared GDN/attention layer graph with the target's dense
// or sparse-MoE FFN. Architecture-specific loaders supply the package tensors.
class TargetModel final {
public:
  // The pass step records the layer encoders walk; their fields are shared
  // internals, defined only in TargetModelGemmaHelpers.hpp.
  struct PrefillStep;
  struct VerifyStep;

  TargetModel(const TargetModelWeights &weights, const TargetModelGeometry &geometry,
             metal::MetalBackend &backend, const ops::ExecutionPlans &operators);
  ~TargetModel();
  // The encoder references this object; a copy would bind it to the wrong
  // model, so copies are out and the move rebuilds the encoder instead of
  // moving the pointer.
  TargetModel(const TargetModel &) = delete;
  TargetModel &operator=(const TargetModel &) = delete;
  TargetModel(TargetModel &&);

  [[nodiscard]] const ops::Projection &
  vocabularyProjection() const noexcept;
  // Lanes of storage the tensors of a decode step of `lanes` lanes bind: a
  // linear tile may hold more rows than the step (LinearPlan::storageRows;
  // a three-lane GGUF step on the staged tile runs its 32-row tile over four
  // lanes). Every op still processes the step's lanes; padding rows read
  // stale activations and write results no active row reads.
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;

  // Returns the hidden buffer that holds the last layer's output rows. The
  // dense FFN of a chunk the split takes (FfnSplit::splits) runs split with the
  // Neural Engine on `aneFfn`, when given.
  // layerScalars, when nonempty, replaces each layer's packed layer_scalar —
  // the DiffusionGemma encoder pass runs the trunk with the separately
  // packed encoder scalars while its canvas pass keeps the files' (decoder)
  // values.
  [[nodiscard]] metal::MetalBuffer addPrefill(
      metal::CommandGraph &graph, TargetModelPrefillBuffers buffers,
      std::span<const TargetModelPrefillSequence> sequences, uint32_t rows,
      std::span<const RichKvLayer> kvLayers, ops::FfnSplit *aneFfn = nullptr,
      std::span<const float> layerScalars = {}) const;
  // liveRows, when nonempty, gives each chain lane's live verify row count
  // for the GDN scan (adaptive proposal budgets); ignored for tree batches.
  // liveNodes bounds a tree batch's live node count when the caller already
  // knows the emitted counts — the attention splits and verify stores then
  // size for the real rows instead of the scratch's full capacity.
  // verifyStride is a chain lane's packed row stride: the draft's trained
  // verify depth (draft.verifyRows()), at most targetVerifyRows. The verify
  // tensors are lane-major at that stride, so the fused-M projections run
  // lanes * verifyStride rows instead of the full capacity.
  void addVerify(
      metal::CommandGraph &graph, TargetModelVerifyBuffers buffers,
      std::span<const RichKvLayer> kvLayers,
      std::span<const kv::ChunkedPrefillParams> chunks,
      uint32_t lanes, bool tree = false,
      std::span<const uint32_t> liveRows = {}, uint32_t liveNodes = 0,
      uint32_t verifyStride = ExecutionLimits::targetVerifyRows,
      const ops::GdnTapeLanes *tapeLanes = nullptr) const;
  // The final norm and LM head over `lanes` lanes of targetVerifyRows rows,
  // as verify ends: one sweep of the vocabulary projection for every lane.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                    metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
                    uint32_t lanes, ops::LinearScratch scratch,
                    const TargetModelVerifyBuffers *fused = nullptr,
                    uint32_t verifyStride = ExecutionLimits::targetVerifyRows) const;
  // The score-gather head (RICHENGINE_SCORE_GATHER): the final norm, then
  // only the quant tiles of the vocabulary projection holding one of
  // `quantTiles` (sorted, unique). Returns false — the caller encodes
  // addHeadBatch — where the projection cannot take the restricted
  // dispatch: an affine or rotated head, several segments, a logit softcap.
  [[nodiscard]] bool addScoreHeadColumns(
      metal::CommandGraph &graph, metal::MetalBuffer hidden,
      metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
      uint32_t lanes, ops::LinearScratch scratch,
      std::span<const uint32_t> quantTiles) const;
  // The score head's single gather dispatch (RICHENGINE_SCORE_TENSOR and
  // the fused RICHENGINE_SCORE_FUSED): `columns` holds `columnCount`
  // absolute vocabulary columns; the fused variant runs the final norm in
  // the same dispatch. Same return contract as addScoreHeadColumns.
  [[nodiscard]] bool addScoreHeadGather(
      metal::CommandGraph &graph, metal::MetalBuffer hidden,
      metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
      uint32_t lanes, ops::LinearScratch scratch,
      metal::MetalBuffer columns, uint32_t columnCount, bool fused) const;
  // The verify input tokens addEmbedding then gathers: each lane's anchor,
  // row 0 of its draft input, and the draft's proposals. liveRows[lane]
  // bounds the live rows (anchor included); rows past it repeat the anchor,
  // keeping dead rows' expert routes on the anchor's paths. plePriorTokens
  // holds, for a target whose verify rows hash their context (qwen4exp's
  // PLE), the tokens before each lane's first row — pleNgram - 1 per lane,
  // nearest first — that the layer encoder's onVerifyInput hook hashes with.
  void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                      metal::MetalBuffer proposals,
                      metal::MetalBuffer verifyInput, uint32_t lanes,
                      uint32_t verifyStride = ExecutionLimits::targetVerifyRows,
                      uint32_t draftStride = ExecutionLimits::draftQueryRows,
                      std::span<const uint32_t> liveRows = {},
                      metal::MetalBuffer plePriorTokens = {}) const;
  // A tree batch's verify inputs: each lane's RICHENGINE_TREE_VERIFY_NODES nodes
  // supply the input tokens, rope positions and ancestor masks. `base` is
  // each lane's (t, h, w) rotary position at its committed length.
  void addVerifyTreeInput(metal::CommandGraph &graph,
                          metal::MetalBuffer treeTokens,
                          metal::MetalBuffer treeNodes,
                          metal::MetalBuffer treeCounts,
                          metal::MetalBuffer verifyInput,
                          metal::MetalBuffer positions,
                          metal::MetalBuffer masks,
                          const uint32_t base[][3], uint32_t lanes) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  void addStateCommit(metal::CommandGraph &graph,
                      TargetModelCommitBuffers buffers, uint32_t lanes,
                      uint32_t verifyStride = ExecutionLimits::targetVerifyRows) const;
  // The tree batch's commit: the retained path replays its DFS rows through
  // each layer's states instead of a chain's prefix rows.
  void addStateCommitTree(metal::CommandGraph &graph,
                          TargetModelCommitBuffers buffers,
                          metal::MetalBuffer retainedPath,
                          uint32_t lanes) const;

private:
  // A layer's parts in dispatch order: the mixer normalizes its input and
  // returns the residual rows the FFN normalizes and adds to into `output`.
  void addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm,
                      ops::WeightLayout consumer) const;
  void addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                        metal::MetalBuffer input, metal::MetalBuffer output) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const GdnMixerWeights &mixer, const ops::NormWeights &norm,
                                     metal::MetalBuffer input) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const AttentionMixerWeights &mixer,
                                     const ops::NormWeights &norm, metal::MetalBuffer input) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const LfmConvWeights &mixer,
                                     const ops::NormWeights &norm, metal::MetalBuffer input) const;
  // The FFN of layer `index`; the dense FFN runs on `step.aneFfn` when set.
  void addPrefillFfn(PrefillStep &step, uint32_t index, const TargetLayerWeights &layer,
                     metal::MetalBuffer residual, metal::MetalBuffer output) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const GdnMixerWeights &mixer, const ops::NormWeights &norm,
                                    metal::MetalBuffer input) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const AttentionMixerWeights &mixer,
                                    const ops::NormWeights &norm, metal::MetalBuffer input) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const LfmConvWeights &mixer,
                                    const ops::NormWeights &norm, metal::MetalBuffer input) const;
  void addVerifyDenseFfn(VerifyStep &step, const ops::NormWeights &norm,
                         const ops::Projection &gate, const ops::Projection &up,
                         const ops::Projection &down, metal::MetalBuffer residual,
                         metal::MetalBuffer output) const;
  void addVerifyFfn(VerifyStep &step, const TargetLayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;
  // The elementwise helpers Gemma's layer epilogue shares with the draft
  // kernels: a flat bf16 residual add and the in-place scalar multiply.
  static void addResidualAdd(metal::CommandGraph &graph, metal::MetalBuffer input,
                             metal::MetalBuffer residual, metal::MetalBuffer output,
                             uint32_t elements);
  static void addLayerScalar(metal::CommandGraph &graph, metal::MetalBuffer values,
                             float scale, uint32_t count);

  // The arch seam's encoders (TargetLayerEncoder.hpp) call the layer
  // helpers and reach the weights, geometry, backend and plans through the
  // model.
  friend class StandardLayerEncoder;
  friend class GemmaLayerEncoder;
  friend class Qwen4LayerEncoder;
  friend class Lfm3LayerEncoder;

  const TargetModelWeights *weights_;
  const TargetModelWeightsBase &weightsBase_;
  TargetModelGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  // The arch's layer encoder — the standard one where the registration
  // leaves makeEncoder null. The factory is kept so a move rebuilds the
  // encoder against the moved-to object.
  std::unique_ptr<TargetLayerEncoder> encoder_;
  std::unique_ptr<TargetLayerEncoder> (*encoderFactory_)(const TargetModel &);
};

} // namespace richengine::model

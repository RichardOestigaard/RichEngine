#pragma once

#include "Model.hpp"
#include "QwenHybridLayout.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/LfmConv.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace splash::model {

struct Qwen3_8Layout;
struct Qwen3_8LayerWeights;
struct Ornith9BLayout;
struct Qwen3_6MoeLayout;
struct Qwen3_6MoeLayerWeights;
struct DenseLayout;
struct Lfm2Layout;
struct Lfm2MoeLayout;
struct Lfm2MoeLayerWeights;

// Both supported targets bind the same mixer tensors per hybrid layer; only
// the FFN differs between them.
struct QwenGdnWeights final {
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

struct QwenAttentionWeights final {
  ops::Projection inputProjection;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::Projection outputProjection;
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

using QwenMixerWeights =
    std::variant<QwenGdnWeights, QwenAttentionWeights, LfmConvWeights>;

// A Qwen target's weights outside its layers and the record of every file
// its weights were read from.
struct QwenTargetWeightsBase {
  ops::NormWeights finalNorm;
  ops::Projection logitsProjection;
  ops::EmbeddingWeights tokenEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
};

// The weights of a target of Layout, whose layers the family keeps in Layer.
template <class Layout, class Layer> struct QwenTargetWeights final : QwenTargetWeightsBase {
  Layout layout;
  std::vector<Layer> layers;
};

// Runtime-visible tensor geometry shared by the supported Qwen hybrid
// targets. It describes semantics only; operators remain responsible for
// choosing device-specific Metal pipelines and compute tiles.
struct QwenTargetGeometry final {
  static constexpr uint32_t maximumCaptureLayers = 8;

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
  // The position axes a target row carries: 3 for the Qwen3.5 M-RoPE
  // families (dim % 3 chooses the axis), 1 for the dense and LFM2 targets.
  uint32_t ropeAxes = 3;
  uint32_t denseIntermediateSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  QwenFfnKind ffnKind = QwenFfnKind::Dense;
  // The stateLayout.layers recurrent slots are gdnLayers GDN layers plus
  // convLayers LFM2 convolution layers; a pure dense target has neither.
  uint32_t gdnLayers = 0;
  uint32_t convLayers = 0;
  // The convolution FIFO's kernel taps (kGdnConvolutionTaps for GDN, the
  // LFM2 conv_L_cache for conv layers).
  uint32_t convolutionTaps = 0;
  // The attention rows' [query|gate] packing and the per-head RMS norms:
  // both are Qwen features; the dense target has neither, LFM2 keeps only
  // the norms.
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
                    ffnKind == QwenFfnKind::SparseMoe ? expertIntermediateSize : uint32_t{0});
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
            convolutionDimension, packedGdnWidth};
  }
  [[nodiscard]] constexpr ops::LfmConvShape convShape() const noexcept {
    return {convolutionDimension, convolutionTaps, packedGdnWidth};
  }
  // The layout itself was checked by requireQwenLayout when the target
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
           sized(prefillProjections) && sized(decodeProjections) &&
           (ffnKind == QwenFfnKind::Dense
                ? denseIntermediateSize && sized(gateUpProjections)
                : moeShape().valid() &&
                      // A mixed target's leading dense FFN layers.
                      (!denseIntermediateSize ||
                       sized(gateUpProjections)));
  }
};

struct QwenTargetPrefillCapture final {
  uint32_t sourceStart = 0;
  uint32_t destinationStart = 0;
  uint32_t rows = 0;
};

struct QwenTargetPrefillSequence final {
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
  std::array<QwenTargetPrefillCapture, 2> captures{};
  uint32_t captureCount = 0;
};

struct QwenTargetPrefillBuffers final {
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
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  // WY/UT scratch for the chunked GDN scan; empty keeps the serial scan.
  metal::MetalBuffer gdnChunkScratch;
  ops::MoeScratch moe;
};

struct QwenTargetVerifyBuffers final {
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
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer denseGateScratch;
  std::span<const metal::MetalBuffer> gdnPacked;
  std::span<const metal::MetalBuffer> gdnMixed;
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
  // Tree verify only: the selector's node tables, the input pass's ancestor
  // masks and the capture staging the post-acceptance gather reads. Empty in
  // a chain batch.
  metal::MetalBuffer treeNodes;
  metal::MetalBuffer treeCounts;
  metal::MetalBuffer treeMasks;
  metal::MetalBuffer capturedPath;
  ops::MoeScratch moe;
};

struct QwenTargetCommitBuffers final {
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

template <class Layout, class Layer>
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights);

// Builds the shared Qwen GDN/attention layer graph with the target's dense
// or sparse-MoE FFN. Architecture-specific loaders supply the package tensors.
class QwenTarget final {
public:
  template <class Layout, class Layer>
  QwenTarget(const QwenTargetWeights<Layout, Layer> &weights, const QwenTargetGeometry &geometry,
             metal::MetalBackend &backend, const ops::ExecutionPlans &operators);

  [[nodiscard]] const ops::Projection &
  vocabularyProjection() const noexcept;
  // Lanes of storage the tensors of a decode step of `lanes` lanes bind: a
  // linear tile may hold more rows than the step (LinearPlan::storageRows;
  // a three-lane GGUF step on the staged tile runs its 32-row tile over four
  // lanes). Every op still processes the step's lanes; padding rows read
  // stale activations and write results no active row reads.
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;

  // Returns the hidden buffer that holds the last layer's output rows.
  [[nodiscard]] metal::MetalBuffer addPrefill(
      metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const SplashKvLayer> kvLayers) const;
  // liveRows, when nonempty, gives each chain lane's live verify row count
  // for the GDN scan (adaptive proposal budgets); ignored for tree batches.
  void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      std::span<const SplashKvLayer> kvLayers,
      std::span<const kv::ChunkedPrefillParams> chunks,
      uint32_t lanes, bool tree = false,
      std::span<const uint32_t> liveRows = {}) const;
  // The final norm and LM head over `lanes` lanes of targetVerifyRows rows,
  // as verify ends: one sweep of the vocabulary projection for every lane.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                    metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
                    uint32_t lanes, ops::LinearScratch scratch) const;
  // The verify input tokens addEmbedding then gathers: each lane's anchor,
  // row 0 of its draft input, and the draft's proposals.
  void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                      metal::MetalBuffer proposals,
                      metal::MetalBuffer verifyInput, uint32_t lanes) const;
  // A tree batch's verify inputs: each lane's SPLASH_TREE_VERIFY_NODES nodes
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
                      QwenTargetCommitBuffers buffers, uint32_t lanes) const;
  // The tree batch's commit: the retained path replays its DFS rows through
  // each layer's states instead of a chain's prefix rows.
  void addStateCommitTree(metal::CommandGraph &graph,
                          QwenTargetCommitBuffers buffers,
                          metal::MetalBuffer retainedPath,
                          uint32_t lanes) const;

private:
  using WeightView =
      std::variant<const QwenTargetWeights<Qwen3_8Layout, Qwen3_8LayerWeights> *,
                   const QwenTargetWeights<Ornith9BLayout, Qwen3_8LayerWeights> *,
                   const QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights> *,
                   const QwenTargetWeights<DenseLayout, Qwen3_8LayerWeights> *,
                   const QwenTargetWeights<Lfm2Layout, Qwen3_8LayerWeights> *,
                   const QwenTargetWeights<Lfm2MoeLayout, Lfm2MoeLayerWeights> *>;
  struct PrefillStep;
  struct VerifyStep;

  // A layer's parts in dispatch order: the mixer normalizes its input and
  // returns the residual rows the FFN normalizes and adds to into `output`.
  void addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm,
                      ops::WeightLayout consumer) const;
  void addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                        metal::MetalBuffer input, metal::MetalBuffer output) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, const ops::NormWeights &norm,
                                     metal::MetalBuffer input) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                     const ops::NormWeights &norm, metal::MetalBuffer input) const;
  metal::MetalBuffer addPrefillMixer(PrefillStep &step, const LfmConvWeights &mixer,
                                     const ops::NormWeights &norm, metal::MetalBuffer input) const;
  // The dense gated FFN of three projections; the LFM2-MoE target's leading
  // layers run the same one.
  void addPrefillDenseFfn(PrefillStep &step, const ops::NormWeights &norm,
                          const ops::Projection &gate, const ops::Projection &up,
                          const ops::Projection &down, metal::MetalBuffer residual,
                          metal::MetalBuffer output) const;
  void addPrefillFfn(PrefillStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                     metal::MetalBuffer output) const;
  void addPrefillFfn(PrefillStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                     metal::MetalBuffer output) const;
  void addPrefillFfn(PrefillStep &step, const Lfm2MoeLayerWeights &layer, metal::MetalBuffer residual,
                     metal::MetalBuffer output) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer, const ops::NormWeights &norm,
                                    metal::MetalBuffer input) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                    const ops::NormWeights &norm, metal::MetalBuffer input) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const LfmConvWeights &mixer,
                                    const ops::NormWeights &norm, metal::MetalBuffer input) const;
  void addVerifyDenseFfn(VerifyStep &step, const ops::NormWeights &norm,
                         const ops::Projection &gate, const ops::Projection &up,
                         const ops::Projection &down, metal::MetalBuffer residual,
                         metal::MetalBuffer output) const;
  void addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;
  void addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;
  void addVerifyFfn(VerifyStep &step, const Lfm2MoeLayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;

  WeightView weights_;
  const QwenTargetWeightsBase &weightsBase_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

} // namespace splash::model

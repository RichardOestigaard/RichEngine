#include "model/TargetModel.hpp"
#include "model/TargetModelGemmaHelpers.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/ModelRegistry.hpp"
#include "Tuning.hpp"
#include "model/ModelPolicy.hpp"

#include "metal/abi/GDN.h"
#include "model/WeightStore.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace richengine::model {
namespace {

TargetModelGeometry commonGeometry(const TargetLayout &layout) {
  if (layout.captureLayerCount > TargetModelGeometry::maximumCaptureLayers)
    throw std::invalid_argument(
        "target captures more hidden layers than the geometry holds");
  TargetModelGeometry result;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
  result.paddedVocabularySize = model::paddedVocabularySize(layout.vocabularySize);
  result.packedGdnWidth = layout.packedGdnWidth;
  result.packedFullWidth = layout.packedFullWidth;
  result.convolutionDimension = layout.convolutionDimension;
  result.attentionWidth = layout.attentionWidth;
  result.attentionQueryHeads = layout.attentionQueryHeads;
  result.attentionKvHeads = layout.attentionKvHeads;
  result.attentionHeadDimension = layout.attentionHeadDimension;
  result.rotaryPairs = layout.rotaryPairs;
  result.rotaryTheta = layout.rotaryTheta;
  result.gdnKeyHeads = layout.gdnKeyHeads;
  result.gdnValueHeads = layout.gdnValueHeads;
  result.gdnHeadDimension = layout.gdnHeadDimension;
  result.maskToken = layout.maskToken;
  result.stopTokens = layout.stopTokens;
  result.ffnKind = layout.ffnKind;
  result.kvLayout = layout.kvLayout();
  result.stateLayout = layout.gdnStateLayout();
  // The stateLayout.layers recurrent slots are the GDN hybrids' gdnLayers,
  // the LFM2 targets' convLayers; a target without recurrent state has
  // neither.
  result.gdnLayers = layout.gdnHeadDimension ? result.stateLayout.layers : 0;
  result.convLayers =
      !layout.gdnHeadDimension && layout.convolutionTaps
          ? result.stateLayout.layers
          : 0;
  result.convolutionTaps = layout.convolutionTaps;
  // The M-RoPE families carry three position axes; everything else one.
  result.ropeAxes = layout.gdnHeadDimension ? 3 : 1;
  result.attentionQueryGate = layout.attentionQueryStride == 2;
  result.attentionQkNorm = layout.attentionQkNorm;
  result.moeSharedExpert = layout.moeSharedExpert;
  result.gdnCore = layout.gdnCore;
  result.pleNgram = layout.pleNgram;
  result.captureLayerCount = layout.captureLayerCount;
  std::copy_n(layout.hiddenCaptureLayers.begin(), layout.captureLayerCount,
              result.captureLayerValues.begin());
  return result;
}

// The execution geometry of the canonical layout: the dual-geometry
// (Gemma/DiffusionGemma) targets take the alt-attention branch, the rest
// the common fields plus their FFN sizes.
TargetModelGeometry geometryFor(const TargetLayout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  if (layout.globalLayerPeriod) {
    // Gemma 4 26B-A4B (and the DiffusionGemma trunk, which shares it): the
    // sliding layers' geometry is primary, the k_eq_v globals' the
    // alternate; tied embeddings scale by sqrt(hidden) and the head takes
    // the logit softcap.
    result.attentionWidth =
        layout.attentionQueryHeads * layout.attentionHeadDimension;
    result.packedFullWidth = layout.packedWidthAt(0);
    result.altAttentionMask = layout.globalLayerMask();
    result.altKvHeads = layout.altKvHeads;
    result.altHeadDimension = layout.altHeadDimension;
    result.altRotaryPairs = layout.altRotaryPairs;
    result.altRotaryTheta = layout.altRotaryTheta;
    result.slidingWindowTokens = layout.slidingWindowTokens;
    result.gemmaMoe = true;
    // The expert tiles' padded width; the logical width rides inside it.
    result.expertIntermediateSize = layout.packedExpertWidth;
    // The shared expert is a dense GeGLU of the padded shared width.
    result.denseIntermediateSize = layout.packedSharedWidth;
    result.embeddingScale =
        layout.embeddingScale
            ? layout.embeddingScale
            : std::sqrt(static_cast<float>(layout.hiddenSize));
    result.logitSoftcap = layout.logitSoftcap;
  } else {
    result.denseIntermediateSize = layout.intermediateSize;
    result.expertIntermediateSize = layout.expertIntermediateSize;
  }
  if (layout.ffnKind == FfnKind::SparseMoe) {
    result.experts = layout.experts;
    result.expertsPerToken = layout.expertsPerToken;
  }
  return result;
}

void requireWeights(const TargetModelWeights &weights,
                    const TargetModelGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<AttentionMixerWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "target weights do not match execution geometry");
  }
}

} // namespace

// The default layer encoder: today's mixer visit plus the FFN dispatch —
// the layer body of every arch whose registration leaves makeEncoder null.
// At namespace scope, not the anonymous one, because TargetModel's friend
// declaration names it.
class StandardLayerEncoder final : public TargetLayerEncoder {
public:
  explicit StandardLayerEncoder(const TargetModel &model) : model_(model) {}

  metal::MetalBuffer addPrefillLayer(TargetModel::PrefillStep &step,
                                     uint32_t index, metal::MetalBuffer input,
                                     metal::MetalBuffer output) const override {
    const TargetLayerWeights &layer = model_.weights_->layers[index];
    const metal::MetalBuffer residual = std::visit(
        [&](const auto &mixer) {
          return model_.addPrefillMixer(step, mixer, layer.inputNorm, input);
        },
        layer.mixer);
    model_.addPrefillFfn(step, index, layer, residual, output);
    return output;
  }

  void addVerifyLayer(TargetModel::VerifyStep &step, uint32_t index,
                      metal::MetalBuffer input,
                      metal::MetalBuffer output) const override {
    const TargetLayerWeights &layer = model_.weights_->layers[index];
    const metal::MetalBuffer residual = std::visit(
        [&](const auto &mixer) {
          return model_.addVerifyMixer(step, mixer, layer.inputNorm, input);
        },
        layer.mixer);
    model_.addVerifyFfn(step, layer, residual, output);
  }

private:
  const TargetModel &model_;
};

namespace {

// The arch registration's encoder factory; unregistered or unset takes the
// standard encoder.
std::unique_ptr<TargetLayerEncoder> (*encoderFactoryFor(
    const TargetLayout &layout))(const TargetModel &) {
  const TargetArchRegistration *registration = findTargetArch(layout.arch);
  return registration ? registration->makeEncoder : nullptr;
}

} // namespace

TargetModel::TargetModel(const TargetModelWeights &weights,
                       const TargetModelGeometry &geometry,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), weightsBase_(weights), geometry_(geometry),
      backend_(backend), operators_(operators),
      encoderFactory_(encoderFactoryFor(weights.layout)) {
  requireWeights(weights, geometry_);
  if (encoderFactory_) encoder_ = encoderFactory_(*this);
  if (!encoder_) encoder_ = std::make_unique<StandardLayerEncoder>(*this);
}

TargetModel::~TargetModel() = default;

TargetModel::TargetModel(TargetModel &&other)
    : weights_(other.weights_), weightsBase_(other.weightsBase_),
      geometry_(std::move(other.geometry_)), backend_(other.backend_),
      operators_(other.operators_), encoderFactory_(other.encoderFactory_) {
  // The encoder references the model it was built against — rebuild it for
  // the moved-to object instead of moving the pointer.
  if (encoderFactory_) encoder_ = encoderFactory_(*this);
  if (!encoder_) encoder_ = std::make_unique<StandardLayerEncoder>(*this);
}

namespace {

void includeProjection(TargetModelGeometry &geometry, const ops::Projection &projection) {
  geometry.decodeProjections.push_back(projection.shape());
}

// The projections a dense FFN dispatches: the fused gate/up pair (matching
// shapes and layouts) and the down projection.
void includeDenseFfn(TargetModelGeometry &geometry, const ops::Projection &gate,
                     const ops::Projection &up, const ops::Projection &down) {
  if (gate.shape() != up.shape())
    throw WeightStoreError("fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, gate);
  includeProjection(geometry, up);
  includeProjection(geometry, down);
  geometry.gateUpProjections.push_back(up.shape());
}

// No source mixes MoE layouts, so one plan runs every block of a step; the
// first MoE block seen seeds the shared layout (a mixed-FFN target's leading
// dense layers never reach here).
void includeMoeFfn(TargetModelGeometry &geometry, const ops::MoeWeights &ffn) {
  if (!geometry.moeSeeded) {
    geometry.moeSeeded = true;
    geometry.moeLayout = ffn.layout();
  }
  if (ffn.layout() != geometry.moeLayout)
    throw WeightStoreError("the MoE blocks of a target must share one weight layout");
}

// The projections each layer's FFN dispatches. Gemma's block binds the
// routed expert slabs (padded width) plus the shared expert's dense GeGLU
// projections.
void includeFfn(TargetModelGeometry &geometry, const TargetLayerWeights &layer) {
  std::visit(
      [&](const auto &ffn) {
        using Ffn = std::decay_t<decltype(ffn)>;
        if constexpr (std::is_same_v<Ffn, DenseFfnWeights>) {
          includeDenseFfn(geometry, ffn.gateProjection, ffn.upProjection,
                          ffn.downProjection);
        } else if constexpr (std::is_same_v<Ffn, ops::MoeWeights>) {
          includeMoeFfn(geometry, ffn);
        } else if constexpr (std::is_same_v<Ffn, Lfm2FfnWeights>) {
          std::visit(
              [&](const auto &inner) {
                using Inner = std::decay_t<decltype(inner)>;
                if constexpr (std::is_same_v<Inner, Lfm2DenseFfn>)
                  includeDenseFfn(geometry, inner.gateProjection,
                                  inner.upProjection, inner.downProjection);
                else
                  includeMoeFfn(geometry, inner);
              },
              ffn.ffn);
        } else {
          static_assert(std::is_same_v<Ffn, Gemma4FfnWeights>);
          // Gemma's expert slabs are always affine; the first block seeds
          // the shared MoE layout the same way.
          if (!geometry.moeSeeded) {
            geometry.moeSeeded = true;
            geometry.moeLayout = ops::WeightLayout::Affine64;
          }
          if (ffn.sharedGate.shape() != ffn.sharedUp.shape() ||
              ffn.expertGate.outputSize != ffn.expertUp.outputSize)
            throw WeightStoreError(
                "fused gate/up projections must have matching shapes and layouts");
          includeProjection(geometry, ffn.sharedGate);
          includeProjection(geometry, ffn.sharedUp);
          includeProjection(geometry, ffn.sharedDown);
          geometry.gateUpProjections.push_back(ffn.sharedUp.shape());
        }
      },
      layer.ffn);
}

// The format of most routed expert weights of a GGUF target's MoE blocks,
// GGUF_FMT_COUNT for none (ops::MoeShape::expertFormat). Layers without a
// routed block — a dense FFN or Gemma's always-affine slabs — contribute
// nothing.
uint32_t routedExpertFormat(std::span<const TargetLayerWeights> layers) {
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  for (const TargetLayerWeights &layer : layers) {
    const ops::MoeWeights *moe = std::get_if<ops::MoeWeights>(&layer.ffn);
    if (!moe) {
      if (const auto *lfm2 = std::get_if<Lfm2FfnWeights>(&layer.ffn))
        moe = std::get_if<ops::MoeWeights>(&lfm2->ffn);
      if (!moe) continue;
    }
    if (moe->layout() != ops::WeightLayout::Block32) return GGUF_FMT_COUNT;
    const ops::BlockMoeWeights &block = moe->blocks();
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  return *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
}

} // namespace

TargetModelGeometry targetModelGeometry(const TargetModelWeights &weights) {
  auto geometry = geometryFor(weights.layout);
  for (const auto &layer : weights.layers) {
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    includeFfn(geometry, layer);
  }
  geometry.moeExpertFormat = routedExpertFormat(weights.layers);
  geometry.prefillProjections = geometry.decodeProjections;
  includeProjection(geometry, weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections,
                       &geometry.gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

const ops::Projection &TargetModel::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

uint32_t TargetModel::decodeStorageLanes(uint32_t lanes) const {
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  uint32_t storageRows = rows;
  for (const auto &shape : geometry_.decodeProjections)
    storageRows = std::max(storageRows, operators_.linear().decodeStorageRows(rows, shape));
  return storageRows / ExecutionLimits::targetVerifyRows;
}

namespace {

void requireLayerPartition(const TargetModelGeometry &geometry, uint32_t gdnLayers, uint32_t attentionLayers) {
  if (gdnLayers != geometry.stateLayout.layers || attentionLayers != geometry.kvLayout.attentionLayers)
    throw std::logic_error("target layer partition mismatch");
}

// Copies `rows` rows of a capture layer's output, from row `sourceRow`, into
// capture slot `slot` of the captured hidden rows from row `destinationRow`.
void addCapture(metal::CommandGraph &graph, const TargetModelGeometry &geometry, uint32_t slot,
                metal::MetalBuffer output, uint32_t sourceRow, metal::MetalBuffer captured,
                uint32_t destinationRow, uint32_t rows) {
  const uint32_t width = geometry.hiddenSize, capturedWidth = geometry.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("target capture slot past the captured hidden rows");
  ops::RowCopy::add(graph, std::move(output), {sourceRow, width, 0}, std::move(captured),
                    {destinationRow, capturedWidth, slot * width}, rows, width);
}

} // namespace

metal::MetalBuffer TargetModel::addPrefill(
    metal::CommandGraph &graph, TargetModelPrefillBuffers buffers,
    std::span<const TargetModelPrefillSequence> sequences, uint32_t rows,
    std::span<const RichKvLayer> kvLayers, ops::FfnSplit *aneFfn,
    std::span<const float> layerScalars) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      (!layerScalars.empty() && layerScalars.size() != geometry_.layers)) {
    throw std::invalid_argument("invalid packed prefill batch");
  }
  for (const TargetModelPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("prefill state layer mismatch");
    }
  }
  PrefillStep step{graph, buffers, sequences, rows, kvLayers};
  // The pass inputs routed to the active layer encoder (the DiffusionGemma
  // encoder's per-layer scalars); its beginPrefill replaces the seed with
  // its own state record.
  step.extension = TargetPrefillInputs{layerScalars};
  // A canvas pass is homogeneous: either every sequence spans canvas rows or
  // none does (the commit pass encodes separately).
  const bool canvas = sequences.front().canvas;
  for (const TargetModelPrefillSequence &sequence : sequences)
    if (sequence.canvas != canvas)
      throw std::invalid_argument("mixed canvas and causal prefill sequences");
  for (const TargetModelPrefillSequence &sequence : sequences) {
    step.attention.push_back(
        sequence.canvas
            ? ops::PagedAttention::prefillCanvasPlan(
                  sequence.rows, geometry_.attentionQueryHeads,
                  geometry_.layerKvLayout(false))
            : operators_.prefillAttention(
                  sequence.rows, geometry_.attentionQueryHeads,
                  geometry_.layerKvLayout(false)));
  }
  encoder_->beginPrefill(step);
  if (geometry_.ffnKind == FfnKind::SparseMoe)
    step.moe = canvas ? operators_.moeCanvas(geometry_.moeShape(), rows)
                      : operators_.moePrefill(geometry_.moeShape(), rows);
  if (aneFfn && aneFfn->splits(rows)) step.aneFfn = aneFfn;
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const metal::MetalBuffer input = buffers.hidden[index & 1];
    const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
    const metal::MetalBuffer layerOutput =
        encoder_->addPrefillLayer(step, index, input, output);
    if (const auto slot = geometry_.captureSlot(index);
        slot && !encoder_->capturesHidden())
      for (const TargetModelPrefillSequence &sequence : sequences)
        for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
          const TargetModelPrefillCapture &c = sequence.captures[capture];
          addCapture(graph, geometry_, *slot, layerOutput, c.sourceStart, buffers.captured, c.destinationStart,
                     c.rows);
        }
  }
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  return buffers.hidden[geometry_.layers & 1];
}

// An affine prefill projection reads the Q4 input sums of its rows, which the
// norm writes beside them; a block projection reads none.
void TargetModel::addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm,
                                ops::WeightLayout consumer) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  if (consumer == ops::WeightLayout::Affine64)
    ops::Normalization::addRmsWithQ4Sums(step.graph, input, norm, b.normalized, b.projectionSums,
                                         geometry_.hiddenSize, step.rows);
  else
    ops::Normalization::addRms(step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows);
}

// The mixer output projection adds the mixer's rows to `input`. Its affine
// input sums are in `projectionSums` already: the mixer gates wrote them
// beside the rows they stored (the prefill_*_gate*_sums kernels).
void TargetModel::addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                                  metal::MetalBuffer input, metal::MetalBuffer output) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  operators_.linear().addPrefillResidual(step.graph, hidden, projection, input, output, b.projectionSums,
                                         step.rows, b.linearScratch);
}

metal::MetalBuffer TargetModel::addPrefillMixer(PrefillStep &step, const AttentionMixerWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const TargetModelPrefillSequence &sequence = step.sequences[index];
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const uint64_t headBytes = uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
    const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
    const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
    const metal::MetalBuffer queries = backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer attentionRows = backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
    const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
    ops::PagedAttention::addPrefillProjection(
        step.graph, u16(b.fullPacked, geometry_.packedFullWidth), mixer.queryNorm, mixer.keyNorm,
        f32(b.ropeCos, geometry_.rotaryPairs), f32(b.ropeSin, geometry_.rotaryPairs), queries, keys, values,
        sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads, geometry_.kvLayout);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
                                         sequence.chunk, geometry_.kvLayout);
    ops::PagedAttention::addPrefill(
        step.graph, step.kvLayers[layer], queries, attentionRows, b.attentionPartials, b.attentionStatistics,
        sequence.pageTable, sequence.chunk, step.attention[index]);
    const metal::MetalBuffer hiddenRows =
        u16(b.attentionHidden, geometry_.attentionWidth);
    if (geometry_.attentionQueryGate) {
      // The sums-emitting gate writes the out-projection's input sums beside
      // the gated rows; the mixer output needs no sums pass of its own.
      ops::PagedAttention::addPrefillGateSums(
          step.graph, u16(b.fullPacked, geometry_.packedFullWidth),
          attentionRows, hiddenRows,
          prefillSums(backend_, b.projectionSums, sequence.rowBegin,
                      sequence.rows, geometry_.attentionWidth),
          sequence.rows, sequence.attentionStride,
          geometry_.attentionQueryHeads, geometry_.kvLayout);
    } else {
      // The no-gate targets gather the attention rows into the out-
      // projection's input; its affine sums are the generic pass.
      ops::PagedAttention::addPrefillGate(
          step.graph, u16(b.fullPacked, geometry_.packedFullWidth),
          attentionRows, hiddenRows, sequence.rows, sequence.attentionStride,
          geometry_.attentionQueryHeads, geometry_.kvLayout, false);
      if (mixer.outputProjection.layout() == ops::WeightLayout::Affine64) {
        // The sums pass walks whole 32-row tiles: the input view must
        // cover the padded rows, not just this sequence's.
        const uint32_t paddedRows = (sequence.rows + 31) / 32 * 32;
        operators_.linear().addPrefillSums(
            step.graph,
            rowsOf<uint16_t>(backend_, b.attentionHidden, sequence.rowBegin,
                             paddedRows, geometry_.attentionWidth),
            prefillSums(backend_, b.projectionSums, sequence.rowBegin,
                        paddedRows, geometry_.attentionWidth),
            mixer.outputProjection, sequence.rows);
      }
    }
  }
  addPrefillOutput(step, b.attentionHidden, mixer.outputProjection, input, b.attentionOutput);
  return b.attentionOutput;
}

// The layer's FFN: the dense triple (on `step.aneFfn` when the split takes
// it), the routed MoE block, LFM2-MoE's per-layer dense/MoE mix. Gemma's
// FFN never reaches here — its whole layer encodes in the Gemma layer
// encoder.
void TargetModel::addPrefillFfn(PrefillStep &step, uint32_t index, const TargetLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  const auto dense = [&](const ops::Projection &gate, const ops::Projection &up,
                         const ops::Projection &down) {
    addPrefillNorm(step, residual, layer.postAttentionNorm, gate.layout());
    const ops::PrefillFfnBuffers ffn = b.ffn();
    if (step.aneFfn)
      step.aneFfn->add(step.graph, index, ffn, residual, output, step.rows);
    else
      operators_.linear().addPrefillSwiGlu(step.graph, {&gate, &up, &down},
                                           ffn, residual, output, step.rows);
  };
  const auto moe = [&](const ops::MoeWeights &weights) {
    ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm,
                             b.normalized, geometry_.hiddenSize, step.rows);
    ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe, {}, {}},
                  weights, *step.moe);
  };
  std::visit(
      [&](const auto &ffn) {
        using Ffn = std::decay_t<decltype(ffn)>;
        if constexpr (std::is_same_v<Ffn, DenseFfnWeights>) {
          dense(ffn.gateProjection, ffn.upProjection, ffn.downProjection);
        } else if constexpr (std::is_same_v<Ffn, ops::MoeWeights>) {
          moe(ffn);
        } else if constexpr (std::is_same_v<Ffn, Lfm2FfnWeights>) {
          std::visit(
              [&](const auto &inner) {
                using Inner = std::decay_t<decltype(inner)>;
                if constexpr (std::is_same_v<Inner, Lfm2DenseFfn>)
                  dense(inner.gateProjection, inner.upProjection,
                        inner.downProjection);
                else
                  moe(inner);
              },
              ffn.ffn);
        } else {
          throw std::logic_error("Gemma FFN outside a Gemma layer encode");
        }
      },
      layer.ffn);
}

void TargetModel::addVerify(
    metal::CommandGraph &graph, TargetModelVerifyBuffers buffers,
    std::span<const RichKvLayer> kvLayers,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
    bool tree, std::span<const uint32_t> liveRows, uint32_t liveNodes,
    uint32_t verifyStride, const ops::GdnTapeLanes *tapeLanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      (!liveRows.empty() && liveRows.size() != lanes) ||
      !verifyStride || verifyStride > ExecutionLimits::targetVerifyRows ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid verify batch");
  }
  // A tree lane doubles its row block: RICHENGINE_TREE_VERIFY_NODES rows, of
  // which nodes 1..15 hold the emitted tree. Tree mode supports the GDN,
  // attention and LFM2 conv mixers.
  if (tree &&
      (lanes * 2 > ExecutionLimits::maximumBatchWidth ||
       !buffers.treeNodes || !buffers.treeCounts || !buffers.treeMasks ||
       !buffers.capturedPath)) {
    throw std::invalid_argument("invalid tree verify batch");
  }
  const uint32_t rowCapacity =
      tree ? uint32_t{RICHENGINE_TREE_VERIFY_NODES}
           : verifyStride;
  const uint32_t rows = lanes * rowCapacity;
  const uint32_t planLanes = tree ? 2 * lanes : lanes;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, chunks, lanes, rows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.layerKvLayout(false),
                                             std::span(histories).first(lanes), tree, liveNodes, rowCapacity)};
  step.planLanes = planLanes;
  step.tapeLanes = tapeLanes;
  step.rowCapacity = rowCapacity;
  step.tree = tree;
  step.liveRows = liveRows;
  step.liveNodes = liveNodes;
  if (geometry_.ffnKind == FfnKind::SparseMoe)
    step.moe = operators_.moeDecode(geometry_.moeShape(), planLanes,
                                    rowCapacity);
  encoder_->beginVerify(step);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const metal::MetalBuffer input = buffers.hidden[index & 1];
    const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
    encoder_->addVerifyLayer(step, index, input, output);
    if (const auto slot = geometry_.captureSlot(index);
        slot && !encoder_->capturesHidden()) {
      // The row copy's buffers and geometry are static per batch width. A
      // tree batch captures the DFS rows into its staging; the path gather
      // that produces the committed rows runs after acceptance.
      graph.beginBakedSpan();
      addCapture(graph, geometry_, *slot, output, 0,
                 tree ? buffers.capturedPath : buffers.capturedTargetHidden,
                 0, rows);
      graph.endBakedSpan();
    }
  }
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  addHeadBatch(graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, planLanes,
               buffers.linearScratch, &buffers,
               tree ? uint32_t{RICHENGINE_TREE_VERIFY_NODES} : verifyStride);
}

metal::MetalBuffer TargetModel::addVerifyMixer(VerifyStep &step, const AttentionMixerWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.planLanes, ops::LinearEpilogue::None, nullptr, step.rowCapacity);
  // Norm, input projection, QKV projection, gate and residual output
  // projection replay from baked indirect command buffers; the paged
  // store/split/reduce between them suspends the span — their parameter
  // blocks carry each lane's committed history, which changes every step.
  step.graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.fullPacked, .scratch = b.linearScratch, .prepared = normalized},
             mixer.inputProjection, inputPlan);
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes,
                                           step.rowCapacity);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.planLanes, ops::LinearEpilogue::Residual, nullptr, step.rowCapacity);
  ops::PreparedInput hidden;
  static const bool noFusedGate = modelPolicy().noFusedGate;
  // A Plain-input out-projection folds its operand pass into the attention
  // reduce: the query gate of the gated targets, the hidden-row gather of
  // the no-gate ones (chain verifies only — tree plans keep the two-pass
  // path, the fused gather kernels are chain-shaped).
  const bool fusedReduce =
      !noFusedGate && outputPlan.input() == ops::LinearInput::Plain &&
      (geometry_.attentionQueryGate ||
       // verify_attention_reduce_gather_* exists for hd128 and hd64 only.
       (!step.tree && (geometry_.attentionHeadDimension == 128 ||
                       geometry_.attentionHeadDimension == 64)));
  if (fusedReduce) {
    ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                   {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries,
                                    b.attentionPartials, b.attentionStatistics, b.fullAttention,
                                    b.pageTables, b.treeMasks},
                                   step.chunks, step.attention,
                                   geometry_.attentionQueryGate
                                       ? b.fullPacked
                                       : metal::MetalBuffer{},
                                   b.attentionHidden);
  } else {
    ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                   {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries, b.attentionPartials,
                                    b.attentionStatistics, b.fullAttention, b.pageTables, b.treeMasks},
                                   step.chunks, step.attention);
    hidden = ops::PagedAttention::addVerifyGate(
        step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
        geometry_.kvLayout, step.lanes, b.linearScratch, outputPlan.input(),
        geometry_.attentionQueryGate, step.rowCapacity);
  }
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = b.attentionOutput, .residual = input,
              .scratch = b.linearScratch, .prepared = hidden},
             mixer.outputProjection, outputPlan);
  step.graph.endBakedSpan();
  return b.attentionOutput;
}

void TargetModel::addVerifyDenseFfn(VerifyStep &step, const ops::NormWeights &norm,
                                   const ops::Projection &gate, const ops::Projection &up,
                                   const ops::Projection &down, metal::MetalBuffer residual,
                                   metal::MetalBuffer output) const {
  const TargetModelVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  // Norm, gate/up and residual down projection replay from a baked indirect
  // command buffer; all parameters and buffers are static per batch width.
  step.graph.beginBakedSpan();
  const ops::LinearPlan gateUpPlan =
      linear.decodePlan(up, step.planLanes, ops::LinearEpilogue::GateUp, &gate, step.rowCapacity);
  const ops::LinearPlan downPlan =
      linear.decodePlan(down, step.planLanes, ops::LinearEpilogue::Residual, nullptr, step.rowCapacity);
  // The gate/up tile emits its output as the down input's Table64 into
  // denseIntermediate plus paired half sums, sparing the prepare pass —
  // only when both plans run the simdgroup tile.
  const bool emitTable = gateUpPlan.usesSimdgroup() && downPlan.usesSimdgroup();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, residual, norm, b.normalized, geometry_.hiddenSize, step.rows,
      b.linearScratch, gateUpPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .prepared = normalized, .emitTable64 = emitTable},
             up, gateUpPlan, &gate);
  ops::LinearScratch downScratch = b.linearScratch;
  downScratch.input = b.denseIntermediate;
  linear.add(step.graph,
             {.input = b.denseIntermediate, .output = output, .residual = residual,
              .scratch = emitTable ? downScratch : b.linearScratch,
              .prepared = emitTable ? ops::PreparedInput{b.denseIntermediate, ops::LinearInput::Table64Paired}
                                    : ops::PreparedInput{}},
             down, downPlan);
  step.graph.endBakedSpan();
}

// The same layer FFN in a verify step: the dense triple through
// addVerifyDenseFfn, the routed MoE block replayed from a baked span (MoE::add
// nests its own), LFM2-MoE's mix between them. Gemma's FFN never reaches
// here — its whole layer encodes in the Gemma layer encoder.
void TargetModel::addVerifyFfn(VerifyStep &step, const TargetLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const TargetModelVerifyBuffers &b = step.buffers;
  const auto moe = [&](const ops::MoeWeights &weights) {
    // One layer's norm plus MoE block is replayed from a baked indirect
    // command buffer when the submission revalidates it; MoE::add nests its
    // own span.
    step.graph.beginBakedSpan();
    ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm,
                             b.normalized, geometry_.hiddenSize, step.rows);
    ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe, {}, b.verifyRows},
                  weights, *step.moe);
    step.graph.endBakedSpan();
  };
  std::visit(
      [&](const auto &ffn) {
        using Ffn = std::decay_t<decltype(ffn)>;
        if constexpr (std::is_same_v<Ffn, DenseFfnWeights>) {
          addVerifyDenseFfn(step, layer.postAttentionNorm, ffn.gateProjection,
                            ffn.upProjection, ffn.downProjection, residual,
                            output);
        } else if constexpr (std::is_same_v<Ffn, ops::MoeWeights>) {
          moe(ffn);
        } else if constexpr (std::is_same_v<Ffn, Lfm2FfnWeights>) {
          std::visit(
              [&](const auto &inner) {
                using Inner = std::decay_t<decltype(inner)>;
                if constexpr (std::is_same_v<Inner, Lfm2DenseFfn>)
                  addVerifyDenseFfn(step, layer.postAttentionNorm,
                                    inner.gateProjection, inner.upProjection,
                                    inner.downProjection, residual, output);
                else
                  moe(inner);
              },
              ffn.ffn);
        } else {
          throw std::logic_error("Gemma FFN outside a Gemma layer encode");
        }
      },
      layer.ffn);
}

void TargetModel::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                              metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                              ops::LinearScratch scratch,
                              const TargetModelVerifyBuffers *fused,
                              uint32_t verifyStride) const {
  const ops::Linear &linear = operators_.linear();
  const ops::Projection &head = vocabularyProjection();
  // Fused greedy head: the Ornith-9B {248320, 4096} affine projection runs
  // decode_head_argmax_q4, which argmaxes its logits tiles in-kernel and
  // writes partials the sampling reduce turns into tokens — the fp32 logits
  // buffer is never touched on an all-greedy chain-verify step.
  if (fused && fused->fusedHead) {
    const ops::NormWeights &norm = weightsBase_.finalNorm;
    graph.beginBakedSpan();
    ops::Normalization::addRms(graph, std::move(hidden), norm, finalHidden,
                               geometry_.hiddenSize,
                               lanes * verifyStride,
                               scratch, ops::LinearInput::Plain);
    if (head.layout() == ops::WeightLayout::Block32) {
      // The GGUF head's staged decode argmaxes per 64-column tile; the
      // logits buffer is never written.
      if (linear.addGgufHeadArgmax(graph, finalHidden, head, lanes,
                                   fused->headArgmaxValues,
                                   fused->headArgmaxIndices, fused->headArgs,
                                   scratch, verifyStride)) {
        graph.endBakedSpan();
        return;
      }
      // Not a fused-capable segment: the span keeps the norm and the logits
      // path adds the projection below.
    } else {
      const ops::AffineWeights &w = head.affine();
      // The fused-M kernels size their logits tile to the live rows: the
      // 8- and 16-row variants skip the 32-row tile's dead MMA work.
      const uint32_t liveRows = lanes * verifyStride;
      const std::string_view kernel =
          liveRows <= 8    ? "decode_head_argmax_q4_m8"
          : liveRows <= 16 ? "decode_head_argmax_q4_m16"
                           : "decode_head_argmax_q4";
      graph.add(std::string(kernel),
                {finalHidden, w.weights, w.scales, w.biases,
                 fused->headArgmaxValues, fused->headArgmaxIndices},
                fused->headArgs, {head.outputSize / 128, 1, 1}, {256, 1, 1});
      graph.endBakedSpan();
      return;
    }
    graph.endBakedSpan();
  }
  const ops::LinearPlan logitsPlan = linear.decodePlan(
      vocabularyProjection(), lanes, ops::LinearEpilogue::None, nullptr,
      verifyStride);
  // The final norm and vocabulary projection replay from a baked span.
  graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), weightsBase_.finalNorm, finalHidden, geometry_.hiddenSize,
      lanes * verifyStride, scratch, logitsPlan.input());
  linear.add(graph,
             {.input = std::move(finalHidden), .output = logits, .scratch = scratch,
              .prepared = normalized},
             vocabularyProjection(), logitsPlan);
  // Gemma's final softcap runs on the materialized logits only — the map is
  // monotonic, so the fused argmax head never needs it.
  if (geometry_.logitSoftcap > 0.0F) {
    const uint32_t count =
        lanes * verifyStride * geometry_.vocabularySize;
    graph.addTail("decode_logit_softcap", {logits}, geometry_.logitSoftcap,
                  count, {(count + 255) / 256, 1, 1}, {256, 1, 1});
  }
  graph.endBakedSpan();
}

bool TargetModel::addScoreHeadColumns(
    metal::CommandGraph &graph, metal::MetalBuffer hidden,
    metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
    ops::LinearScratch scratch, std::span<const uint32_t> quantTiles) const {
  const ops::Linear &linear = operators_.linear();
  const ops::Projection &head = vocabularyProjection();
  // The restricted dispatch covers whole quant tiles of one GGUF segment; a
  // rotated head's input pass this path does not encode, and a softcap maps
  // the materialized logits, which no uncovered column run would apply.
  if (head.layout() != ops::WeightLayout::Block32 || head.rotation ||
      geometry_.logitSoftcap > 0.0F || quantTiles.empty())
    return false;
  const ops::LinearPlan logitsPlan = linear.decodePlan(
      head, lanes, ops::LinearEpilogue::None, nullptr,
      ExecutionLimits::targetVerifyRows);
  graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), weightsBase_.finalNorm, finalHidden,
      geometry_.hiddenSize, lanes * ExecutionLimits::targetVerifyRows,
      scratch, logitsPlan.input());
  const bool done = linear.addGgufHeadColumns(
      graph, backend_,
      {.input = finalHidden,
       .output = logits,
       .scratch = scratch,
       .prepared = normalized},
      head, logitsPlan, quantTiles);
  graph.endBakedSpan();
  return done;
}

bool TargetModel::addScoreHeadGather(
    metal::CommandGraph &graph, metal::MetalBuffer hidden,
    metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
    ops::LinearScratch scratch, metal::MetalBuffer columns,
    uint32_t columnCount, bool fused) const {
  const ops::Linear &linear = operators_.linear();
  const ops::Projection &head = vocabularyProjection();
  const ops::NormWeights &norm = weightsBase_.finalNorm;
  if (head.layout() != ops::WeightLayout::Block32 || head.rotation ||
      geometry_.logitSoftcap > 0.0F || !columnCount ||
      (fused && !norm.buffer))
    return false;
  graph.beginBakedSpan();
  // The fused kernel norms `hidden` in-dispatch; the unfused one takes the
  // same norm as the restricted head, in plain rows it gathers from.
  if (!fused)
    ops::Normalization::addRms(graph, hidden, norm, finalHidden,
                               geometry_.hiddenSize,
                               lanes * ExecutionLimits::targetVerifyRows,
                               scratch, ops::LinearInput::Plain);
  const bool done = linear.addGgufHeadGather(
      graph, fused ? hidden : finalHidden, finalHidden, norm.buffer,
      norm.float32, norm.rmsEpsilon, logits, columns, columnCount, head,
      lanes, ExecutionLimits::targetVerifyRows, fused);
  graph.endBakedSpan();
  return done;
}

void TargetModel::addVerifyInput(metal::CommandGraph &graph,
                                metal::MetalBuffer draftInput,
                                metal::MetalBuffer proposals,
                                metal::MetalBuffer verifyInput,
                                uint32_t lanes, uint32_t verifyStride,
                                uint32_t draftStride,
                                std::span<const uint32_t> liveRows,
                                metal::MetalBuffer plePriorTokens) const {
  const metal::MetalBuffer tokens = verifyInput;
  ops::Embedding::addVerifyInput(graph, std::move(draftInput),
                                 std::move(proposals), std::move(verifyInput),
                                 geometry_.vocabularySize, lanes, verifyStride,
                                 draftStride, liveRows);
  // The arch encoder's input hook (qwen4exp's GPU PLE hash of the rows).
  encoder_->onVerifyInput(graph, std::move(tokens),
                          std::move(plePriorTokens), lanes, verifyStride);
}

void TargetModel::addVerifyTreeInput(metal::CommandGraph &graph,
                                    metal::MetalBuffer treeTokens,
                                    metal::MetalBuffer treeNodes,
                                    metal::MetalBuffer treeCounts,
                                    metal::MetalBuffer verifyInput,
                                    metal::MetalBuffer positions,
                                    metal::MetalBuffer masks,
                                    const uint32_t base[][3],
                                    uint32_t lanes) const {
  ops::Embedding::addVerifyTreeInput(
      graph, std::move(treeTokens), std::move(treeNodes),
      std::move(treeCounts), std::move(verifyInput), std::move(positions),
      std::move(masks), base, geometry_.vocabularySize, geometry_.maskToken,
      lanes);
}

void TargetModel::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows, geometry_.embeddingScale);
}

} // namespace richengine::model

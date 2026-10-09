#include "model/TargetModel.hpp"
#include "model/TargetModelGemmaHelpers.hpp"
#include "Tuning.hpp"

#include "metal/abi/GDN.h"
#include "model/Dense.hpp"
#include "model/DiffusionGemma.hpp"
#include "model/Gemma4Moe.hpp"
#include "model/Lfm2.hpp"
#include "model/Granite.hpp"
#include "model/Lfm2Moe.hpp"
#include "model/Ornith9B.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
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
#include <utility>
#include <variant>

namespace richengine::model {
namespace {

template <class Layout>
TargetModelGeometry commonGeometry(const Layout &layout) {
  static_assert(std::tuple_size_v<decltype(Layout::hiddenCaptureLayers)> <=
                TargetModelGeometry::maximumCaptureLayers);
  TargetModelGeometry result;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
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
  result.ffnKind = Layout::ffnKind;
  result.kvLayout = layout.kvLayout();
  result.stateLayout = layout.gdnStateLayout();
  result.gdnLayers = layout.gdnStateLayout().layers;
  result.convolutionTaps = layout.mixerGeometry().convolutionTaps;
  result.attentionQueryGate = Layout::attentionQueryStride == 2;
  result.attentionQkNorm = Layout::attentionQkNorm;
  result.captureLayerCount =
      static_cast<uint32_t>(layout.hiddenCaptureLayers.size());
  std::copy(layout.hiddenCaptureLayers.begin(),
            layout.hiddenCaptureLayers.end(),
            result.captureLayerValues.begin());
  return result;
}

TargetModelGeometry geometryFor(const Qwen3_8Layout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  return result;
}

TargetModelGeometry geometryFor(const Ornith9BLayout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  return result;
}

TargetModelGeometry geometryFor(const Qwen3_6MoeLayout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  result.experts = layout.experts;
  result.expertsPerToken = layout.expertsPerToken;
  result.expertIntermediateSize = layout.expertIntermediateSize;
  return result;
}

// The pure dense target: every layer full attention, no recurrent slots.
TargetModelGeometry geometryFor(const DenseLayout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.gdnLayers = 0;
  result.convLayers = 0;
  result.convolutionTaps = 0;
  result.ropeAxes = 1;
  return result;
}

// Granite 4.2: the same all-attention target; its kvLayout carries the
// attention_multiplier into the kernels' softmax scale.
TargetModelGeometry geometryFor(const GraniteLayout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.gdnLayers = 0;
  result.convLayers = 0;
  result.convolutionTaps = 0;
  result.ropeAxes = 1;
  return result;
}

// LFM2's conv layers are its recurrent slots (convLayers), gated by the
// double-gated short convolution instead of the GDN.
TargetModelGeometry geometryFor(const Lfm2Layout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.gdnLayers = 0;
  result.convLayers = layout.gdnStateLayout().layers;
  result.convolutionTaps = Lfm2Layout::convolutionTaps;
  result.ropeAxes = 1;
  return result;
}

// LFM2-MoE: LFM2's conv/attention mixers, a dense FFN on the leading
// denseLayers and the shared-expert-free sigmoid MoE block after them.
TargetModelGeometry geometryFor(const Lfm2MoeLayout &layout) {
  TargetModelGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.experts = layout.experts;
  result.expertsPerToken = layout.expertsPerToken;
  result.expertIntermediateSize = layout.expertIntermediateSize;
  result.moeSharedExpert = false;
  result.gdnLayers = 0;
  result.convLayers = layout.gdnStateLayout().layers;
  result.convolutionTaps = Lfm2MoeLayout::convolutionTaps;
  result.ropeAxes = 1;
  return result;
}

// Gemma 4 26B-A4B: 30 attention layers — 25 sliding (KV8x256, full rotary
// at 1e4, 1024-token window) and five globals (KV2x512, p-RoPE of 64 at 1e6,
// k_eq_v) — each followed by the GeGLU MoE block; no recurrent state, tied
// embeddings scaled by sqrt(hidden), and the 30 logit softcap.
TargetModelGeometry geometryFor(const Gemma4MoeLayout &layout) {
  TargetModelGeometry result;
  result.layers = layout.layers;
  result.hiddenSize = layout.hiddenSize;
  result.vocabularySize = layout.vocabularySize;
  result.attentionQueryHeads = layout.attentionQueryHeads;
  result.attentionKvHeads = layout.attentionKvHeads;
  result.attentionHeadDimension = layout.attentionHeadDimension;
  result.rotaryPairs = layout.rotaryPairs;
  result.rotaryTheta = layout.rotaryTheta;
  result.attentionWidth =
      layout.attentionQueryHeads * layout.attentionHeadDimension;
  result.packedFullWidth = layout.packedWidthAt(0);
  result.altAttentionMask = layout.globalLayerMask();
  result.altKvHeads = layout.globalKvHeads;
  result.altHeadDimension = layout.globalHeadDimension;
  result.altRotaryPairs = layout.globalRotaryPairs;
  result.altRotaryTheta = layout.globalRotaryTheta;
  result.slidingWindowTokens = layout.slidingWindowTokens;
  result.maskToken = layout.maskToken;
  result.stopTokens = layout.stopTokens;
  result.ffnKind = FfnKind::SparseMoe;
  result.experts = layout.experts;
  result.expertsPerToken = layout.expertsPerToken;
  // The expert tiles' padded width; the logical 704 rides inside it.
  result.expertIntermediateSize = layout.packedExpertWidth;
  result.moeSharedExpert = false;
  result.gemmaMoe = true;
  // The shared expert is a dense GeGLU of the padded shared width.
  result.denseIntermediateSize = layout.packedSharedWidth;
  result.kvLayout = layout.kvLayout();
  result.stateLayout = layout.gdnStateLayout();
  result.gdnLayers = 0;
  result.convLayers = 0;
  result.convolutionTaps = 0;
  result.ropeAxes = 1;
  // Gemma keeps the queries' gate-free packing and the per-head norms.
  result.attentionQueryGate = false;
  result.attentionQkNorm = true;
  result.embeddingScale = layout.embeddingScale
                              ? layout.embeddingScale
                              : std::sqrt(static_cast<float>(layout.hiddenSize));
  result.logitSoftcap = layout.logitSoftcap;
  result.captureLayerCount =
      static_cast<uint32_t>(layout.hiddenCaptureLayers.size());
  std::copy(layout.hiddenCaptureLayers.begin(),
            layout.hiddenCaptureLayers.end(),
            result.captureLayerValues.begin());
  return result;
}

// DiffusionGemma's trunk is the Gemma 4 MoE trunk — the decoder runs the
// same layers; only its pass over the canvas differs, and that is a
// per-sequence flag (TargetModelPrefillSequence::canvas), not a geometry.
TargetModelGeometry geometryFor(const DiffusionGemmaLayout &layout) {
  return geometryFor(static_cast<const Gemma4MoeLayout &>(layout));
}

template <class Weights>
void requireWeights(const Weights &weights,
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

template <class Layout, class Layer>
TargetModel::TargetModel(const TargetModelWeights<Layout, Layer> &weights,
                       const TargetModelGeometry &geometry,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), weightsBase_(weights), geometry_(geometry),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
}

template TargetModel::TargetModel(const Qwen3_8Weights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const Ornith9BWeights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const Qwen3_6MoeWeights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const DenseWeights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const Lfm2Weights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const Lfm2MoeWeights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const Gemma4MoeWeights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const DiffusionGemmaTrunkWeights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template TargetModel::TargetModel(const GraniteWeights &, const TargetModelGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);

namespace {

void includeProjection(TargetModelGeometry &geometry, const ops::Projection &projection) {
  geometry.decodeProjections.push_back(projection.shape());
}

// The projections each layer's FFN dispatches, from the first layer on.
void includeFfn(TargetModelGeometry &geometry, const DenseLayerWeights &layer, bool) {
  if (layer.gateProjection.shape() != layer.upProjection.shape())
    throw WeightStoreError("fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, layer.gateProjection);
  includeProjection(geometry, layer.upProjection);
  includeProjection(geometry, layer.downProjection);
  geometry.gateUpProjections.push_back(layer.upProjection.shape());
}
// A dense FFN carried as a variant member (LFM2-MoE's leading layers).
void includeFfn(TargetModelGeometry &geometry, const Lfm2DenseFfn &ffn) {
  if (ffn.gateProjection.shape() != ffn.upProjection.shape())
    throw WeightStoreError("fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, ffn.gateProjection);
  includeProjection(geometry, ffn.upProjection);
  includeProjection(geometry, ffn.downProjection);
  geometry.gateUpProjections.push_back(ffn.upProjection.shape());
}
// No source mixes MoE layouts, so one plan runs every block of a step.
void includeFfn(TargetModelGeometry &geometry, const Qwen3_6MoeLayerWeights &layer, bool first) {
  if (first) geometry.moeLayout = layer.ffn.layout();
  if (layer.ffn.layout() != geometry.moeLayout)
    throw WeightStoreError("the MoE blocks of a target must share one weight layout");
}
// LFM2-MoE's per-layer FFN: dense below denseLayers, MoE from there on.
void includeFfn(TargetModelGeometry &geometry, const Lfm2MoeLayerWeights &layer, bool) {
  std::visit(
      [&](const auto &ffn) {
        using Ffn = std::decay_t<decltype(ffn)>;
        if constexpr (std::is_same_v<Ffn, Lfm2DenseFfn>) {
          includeFfn(geometry, ffn);
        } else {
          if (!geometry.moeSeeded) {
            geometry.moeSeeded = true;
            geometry.moeLayout = ffn.layout();
          }
          if (ffn.layout() != geometry.moeLayout)
            throw WeightStoreError("the MoE blocks of a target must share one weight layout");
        }
      },
      layer.ffn);
}
// Gemma's block: the routed expert slabs (padded width) plus the shared
// expert's dense GeGLU projections.
void includeFfn(TargetModelGeometry &geometry, const Gemma4MoeLayerWeights &layer, bool first) {
  if (first) {
    geometry.moeSeeded = true;
    geometry.moeLayout = ops::WeightLayout::Affine64;
  }
  if (layer.sharedGate.shape() != layer.sharedUp.shape() ||
      layer.expertGate.outputSize != layer.expertUp.outputSize)
    throw WeightStoreError(
        "fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, layer.sharedGate);
  includeProjection(geometry, layer.sharedUp);
  includeProjection(geometry, layer.sharedDown);
  geometry.gateUpProjections.push_back(layer.sharedUp.shape());
}

// The format of most routed expert weights of a GGUF target's MoE blocks,
// GGUF_FMT_COUNT for none (ops::MoeShape::expertFormat).
uint32_t routedExpertFormat(std::span<const DenseLayerWeights>) { return GGUF_FMT_COUNT; }
uint32_t routedExpertFormat(std::span<const Qwen3_6MoeLayerWeights> layers) {
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  for (const Qwen3_6MoeLayerWeights &layer : layers) {
    if (layer.ffn.layout() != ops::WeightLayout::Block32) return GGUF_FMT_COUNT;
    const ops::BlockMoeWeights &block = layer.ffn.blocks();
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  return *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
}
// The same over LFM2-MoE's per-layer dense/MoE FFN variant.
uint32_t routedExpertFormat(std::span<const Lfm2MoeLayerWeights> layers) {
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  for (const Lfm2MoeLayerWeights &layer : layers) {
    const auto *moe = std::get_if<ops::MoeWeights>(&layer.ffn);
    if (!moe) continue;
    if (moe->layout() != ops::WeightLayout::Block32) return GGUF_FMT_COUNT;
    const ops::BlockMoeWeights &block = moe->blocks();
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  return *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
}
// Gemma's slabs are always affine.
uint32_t routedExpertFormat(std::span<const Gemma4MoeLayerWeights>) { return GGUF_FMT_COUNT; }

} // namespace

template <class Layout, class Layer>
TargetModelGeometry targetModelGeometry(const TargetModelWeights<Layout, Layer> &weights) {
  auto geometry = geometryFor(weights.layout);
  for (const auto &layer : weights.layers) {
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    includeFfn(geometry, layer, &layer == &weights.layers.front());
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

template TargetModelGeometry targetModelGeometry(const Qwen3_8Weights &);
template TargetModelGeometry targetModelGeometry(const Ornith9BWeights &);
template TargetModelGeometry targetModelGeometry(const Qwen3_6MoeWeights &);
template TargetModelGeometry targetModelGeometry(const DenseWeights &);
template TargetModelGeometry targetModelGeometry(const Lfm2Weights &);
template TargetModelGeometry targetModelGeometry(const Lfm2MoeWeights &);
template TargetModelGeometry targetModelGeometry(const Gemma4MoeWeights &);
template TargetModelGeometry targetModelGeometry(const DiffusionGemmaTrunkWeights &);
template TargetModelGeometry targetModelGeometry(const GraniteWeights &);

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
    std::span<const RichKvLayer> kvLayers, ops::AneFfn *aneFfn,
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
  step.layerScalars = layerScalars;
  // A canvas pass is homogeneous: either every sequence spans canvas rows or
  // none does (the commit pass encodes separately).
  step.canvas = sequences.front().canvas;
  for (const TargetModelPrefillSequence &sequence : sequences)
    if (sequence.canvas != step.canvas)
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
    if (geometry_.altAttentionMask)
      step.altAttention.push_back(
          sequence.canvas
              ? ops::PagedAttention::prefillCanvasPlan(
                    sequence.rows, geometry_.attentionQueryHeads,
                    geometry_.layerKvLayout(true))
              : operators_.prefillAttention(
                    sequence.rows, geometry_.attentionQueryHeads,
                    geometry_.layerKvLayout(true)));
  }
  if (geometry_.ffnKind == FfnKind::SparseMoe)
    step.moe = step.canvas ? operators_.moeCanvas(geometry_.moeShape(), rows)
                           : operators_.moePrefill(geometry_.moeShape(), rows);
  if (aneFfn && aneFfn->splits(rows)) step.aneFfn = aneFfn;
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      if constexpr (std::is_same_v<std::decay_t<decltype(layer)>,
                                   Gemma4MoeLayerWeights>) {
        // The Gemma layer keeps the post-attention residual inside its own
        // encode: its router reads it, so the mixer cannot return a fused
        // residual.
        addPrefillGemmaLayer(step, layer, index, input, output);
      } else {
        const metal::MetalBuffer residual = std::visit(
            [&](const auto &mixer) { return addPrefillMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
        addPrefillFfn(step, index, layer, residual, output);
      }
      if (const auto slot = geometry_.captureSlot(index))
        for (const TargetModelPrefillSequence &sequence : sequences)
          for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
            const TargetModelPrefillCapture &c = sequence.captures[capture];
            addCapture(graph, geometry_, *slot, output, c.sourceStart, buffers.captured, c.destinationStart,
                       c.rows);
          }
    }
  }, weights_);
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

void TargetModel::addPrefillFfn(PrefillStep &step, uint32_t index, const DenseLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  addPrefillNorm(step, residual, layer.postAttentionNorm, layer.gateProjection.layout());
  const ops::PrefillFfnBuffers ffn = step.buffers.ffn();
  if (step.aneFfn)
    step.aneFfn->add(step.graph, index, ffn, residual, output, step.rows);
  else
    operators_.linear().addPrefillSwiGlu(step.graph, {&layer.gateProjection, &layer.upProjection,
                                                      &layer.downProjection},
                                         ffn, residual, output, step.rows);
}

void TargetModel::addVerify(
    metal::CommandGraph &graph, TargetModelVerifyBuffers buffers,
    std::span<const RichKvLayer> kvLayers,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
    bool tree, std::span<const uint32_t> liveRows, uint32_t liveNodes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      (!liveRows.empty() && liveRows.size() != lanes) ||
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
  // which nodes 1..15 hold the emitted tree. Tree mode supports the GDN and
  // attention mixers only — LFM2 convolutions stay on the chain path.
  if (tree &&
      (lanes * 2 > ExecutionLimits::maximumBatchWidth ||
       (geometry_.convLayers && !geometry_.gdnLayers) ||
       !buffers.treeNodes || !buffers.treeCounts || !buffers.treeMasks ||
       !buffers.capturedPath)) {
    throw std::invalid_argument("invalid tree verify batch");
  }
  const uint32_t rowCapacity =
      tree ? uint32_t{RICHENGINE_TREE_VERIFY_NODES}
           : ExecutionLimits::targetVerifyRows;
  const uint32_t rows = lanes * rowCapacity;
  const uint32_t planLanes = tree ? 2 * lanes : lanes;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, chunks, lanes, rows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.layerKvLayout(false),
                                             std::span(histories).first(lanes), tree, liveNodes)};
  if (geometry_.altAttentionMask)
    step.altAttention.emplace(operators_.verifyAttention(
        lanes, geometry_.attentionQueryHeads, geometry_.layerKvLayout(true),
        std::span(histories).first(lanes), tree, liveNodes));
  step.planLanes = planLanes;
  step.rowCapacity = rowCapacity;
  step.tree = tree;
  step.liveRows = liveRows;
  if (geometry_.ffnKind == FfnKind::SparseMoe) step.moe = operators_.moeDecode(geometry_.moeShape(), planLanes);
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      if constexpr (std::is_same_v<std::decay_t<decltype(layer)>,
                                   Gemma4MoeLayerWeights>) {
        addVerifyGemmaLayer(step, layer, index, input, output);
        if (const auto slot = geometry_.captureSlot(index)) {
          graph.beginBakedSpan();
          addCapture(graph, geometry_, *slot, output, 0,
                     tree ? buffers.capturedPath : buffers.capturedTargetHidden,
                     0, rows);
          graph.endBakedSpan();
        }
        continue;
      } else {
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addVerifyMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addVerifyFfn(step, layer, residual, output);
      }
      if (const auto slot = geometry_.captureSlot(index)) {
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
  }, weights_);
  addHeadBatch(graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, planLanes,
               buffers.linearScratch, &buffers);
}

metal::MetalBuffer TargetModel::addVerifyMixer(VerifyStep &step, const AttentionMixerWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.planLanes);
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
      linear.decodePlan(mixer.outputProjection, step.planLanes, ops::LinearEpilogue::Residual);
  ops::PreparedInput hidden;
  static const bool noFusedGate = tuning().noFusedGate;
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
      linear.decodePlan(up, step.planLanes, ops::LinearEpilogue::GateUp, &gate);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, residual, norm, b.normalized, geometry_.hiddenSize, step.rows,
      b.linearScratch, gateUpPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .prepared = normalized},
             up, gateUpPlan, &gate);
  linear.add(step.graph,
             {.input = b.denseIntermediate, .output = output, .residual = residual, .scratch = b.linearScratch},
             down,
             linear.decodePlan(down, step.planLanes, ops::LinearEpilogue::Residual));
  step.graph.endBakedSpan();
}

void TargetModel::addVerifyFfn(VerifyStep &step, const DenseLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  addVerifyDenseFfn(step, layer.postAttentionNorm, layer.gateProjection, layer.upProjection,
                    layer.downProjection, residual, output);
}

void TargetModel::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                              metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                              ops::LinearScratch scratch,
                              const TargetModelVerifyBuffers *fused) const {
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
                               lanes * ExecutionLimits::targetVerifyRows,
                               scratch, ops::LinearInput::Plain);
    if (head.layout() == ops::WeightLayout::Block32) {
      // The GGUF head's staged decode argmaxes per 64-column tile; the
      // logits buffer is never written.
      if (linear.addGgufHeadArgmax(graph, finalHidden, head, lanes,
                                   fused->headArgmaxValues,
                                   fused->headArgmaxIndices, fused->headArgs,
                                   scratch)) {
        graph.endBakedSpan();
        return;
      }
      // Not a fused-capable segment: the span keeps the norm and the logits
      // path adds the projection below.
    } else {
      const ops::AffineWeights &w = head.affine();
      graph.add("decode_head_argmax_q4",
                {finalHidden, w.weights, w.scales, w.biases,
                 fused->headArgmaxValues, fused->headArgmaxIndices},
                fused->headArgs, {head.outputSize / 128, 1, 1}, {256, 1, 1});
      graph.endBakedSpan();
      return;
    }
    graph.endBakedSpan();
  }
  const ops::LinearPlan logitsPlan = linear.decodePlan(vocabularyProjection(), lanes);
  // The final norm and vocabulary projection replay from a baked span.
  graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), weightsBase_.finalNorm, finalHidden, geometry_.hiddenSize,
      lanes * ExecutionLimits::targetVerifyRows, scratch, logitsPlan.input());
  linear.add(graph,
             {.input = std::move(finalHidden), .output = logits, .scratch = scratch,
              .prepared = normalized},
             vocabularyProjection(), logitsPlan);
  // Gemma's final softcap runs on the materialized logits only — the map is
  // monotonic, so the fused argmax head never needs it.
  if (geometry_.logitSoftcap > 0.0F) {
    const uint32_t count =
        lanes * ExecutionLimits::targetVerifyRows * geometry_.vocabularySize;
    graph.addTail("decode_logit_softcap", {logits}, geometry_.logitSoftcap,
                  count, {(count + 255) / 256, 1, 1}, {256, 1, 1});
  }
  graph.endBakedSpan();
}

void TargetModel::addVerifyInput(metal::CommandGraph &graph,
                                metal::MetalBuffer draftInput,
                                metal::MetalBuffer proposals,
                                metal::MetalBuffer verifyInput,
                                uint32_t lanes) const {
  ops::Embedding::addVerifyInput(graph, std::move(draftInput),
                                 std::move(proposals), std::move(verifyInput),
                                 geometry_.vocabularySize, lanes);
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

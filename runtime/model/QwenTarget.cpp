#include "model/QwenTarget.hpp"
#include "Env.hpp"

#include "metal/abi/GDN.h"
#include "model/Dense.hpp"
#include "model/Lfm2.hpp"
#include "model/Lfm2Moe.hpp"
#include "model/Ornith9B.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/WeightStore.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

template <class Layout>
QwenTargetGeometry commonGeometry(const Layout &layout) {
  static_assert(std::tuple_size_v<decltype(Layout::hiddenCaptureLayers)> <=
                QwenTargetGeometry::maximumCaptureLayers);
  QwenTargetGeometry result;
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

QwenTargetGeometry geometryFor(const Qwen3_8Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  return result;
}

QwenTargetGeometry geometryFor(const Ornith9BLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  return result;
}

QwenTargetGeometry geometryFor(const Qwen3_6MoeLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.experts = layout.experts;
  result.expertsPerToken = layout.expertsPerToken;
  result.expertIntermediateSize = layout.expertIntermediateSize;
  return result;
}

// The pure dense target: every layer full attention, no recurrent slots.
QwenTargetGeometry geometryFor(const DenseLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.gdnLayers = 0;
  result.convLayers = 0;
  result.convolutionTaps = 0;
  result.ropeAxes = 1;
  return result;
}

// LFM2's conv layers are its recurrent slots (convLayers), gated by the
// double-gated short convolution instead of the GDN.
QwenTargetGeometry geometryFor(const Lfm2Layout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
  result.denseIntermediateSize = layout.intermediateSize;
  result.gdnLayers = 0;
  result.convLayers = layout.gdnStateLayout().layers;
  result.convolutionTaps = Lfm2Layout::convolutionTaps;
  result.ropeAxes = 1;
  return result;
}

// LFM2-MoE: LFM2's conv/attention mixers, a dense FFN on the leading
// denseLayers and the shared-expert-free sigmoid MoE block after them.
QwenTargetGeometry geometryFor(const Lfm2MoeLayout &layout) {
  QwenTargetGeometry result = commonGeometry(layout);
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

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers != geometry.kvLayout.attentionLayers) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

} // namespace

template <class Layout, class Layer>
QwenTarget::QwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                       const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(&weights), weightsBase_(weights), geometry_(geometry),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
}

template QwenTarget::QwenTarget(const Qwen3_8Weights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template QwenTarget::QwenTarget(const Ornith9BWeights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template QwenTarget::QwenTarget(const Qwen3_6MoeWeights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template QwenTarget::QwenTarget(const DenseWeights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template QwenTarget::QwenTarget(const Lfm2Weights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);
template QwenTarget::QwenTarget(const Lfm2MoeWeights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &);

namespace {

void includeProjection(QwenTargetGeometry &geometry, const ops::Projection &projection) {
  geometry.decodeProjections.push_back(projection.shape());
}

// The projections each layer's FFN dispatches, from the first layer on.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_8LayerWeights &layer, bool) {
  if (layer.gateProjection.shape() != layer.upProjection.shape())
    throw WeightStoreError("fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, layer.gateProjection);
  includeProjection(geometry, layer.upProjection);
  includeProjection(geometry, layer.downProjection);
  geometry.gateUpProjections.push_back(layer.upProjection.shape());
}
// A dense FFN carried as a variant member (LFM2-MoE's leading layers).
void includeFfn(QwenTargetGeometry &geometry, const Lfm2DenseFfn &ffn) {
  if (ffn.gateProjection.shape() != ffn.upProjection.shape())
    throw WeightStoreError("fused gate/up projections must have matching shapes and layouts");
  includeProjection(geometry, ffn.gateProjection);
  includeProjection(geometry, ffn.upProjection);
  includeProjection(geometry, ffn.downProjection);
  geometry.gateUpProjections.push_back(ffn.upProjection.shape());
}
// No source mixes MoE layouts, so one plan runs every block of a step.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_6MoeLayerWeights &layer, bool first) {
  if (first) geometry.moeLayout = layer.ffn.layout();
  if (layer.ffn.layout() != geometry.moeLayout)
    throw WeightStoreError("the MoE blocks of a target must share one weight layout");
}
// LFM2-MoE's per-layer FFN: dense below denseLayers, MoE from there on.
void includeFfn(QwenTargetGeometry &geometry, const Lfm2MoeLayerWeights &layer, bool) {
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

// The format of most routed expert weights of a GGUF target's MoE blocks,
// GGUF_FMT_COUNT for none (ops::MoeShape::expertFormat).
uint32_t routedExpertFormat(std::span<const Qwen3_8LayerWeights>) { return GGUF_FMT_COUNT; }
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

} // namespace

template <class Layout, class Layer>
QwenTargetGeometry qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights) {
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

template QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &);
template QwenTargetGeometry qwenTargetGeometry(const Ornith9BWeights &);
template QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &);
template QwenTargetGeometry qwenTargetGeometry(const DenseWeights &);
template QwenTargetGeometry qwenTargetGeometry(const Lfm2Weights &);
template QwenTargetGeometry qwenTargetGeometry(const Lfm2MoeWeights &);

const ops::Projection &QwenTarget::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

uint32_t QwenTarget::decodeStorageLanes(uint32_t lanes) const {
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  uint32_t storageRows = rows;
  for (const auto &shape : geometry_.decodeProjections)
    storageRows = std::max(storageRows, operators_.linear().decodeStorageRows(rows, shape));
  return storageRows / ExecutionLimits::targetVerifyRows;
}

namespace {

// Rows [begin, begin + count) of a row-major buffer of `width` values of T.
template <class T>
metal::MetalBuffer rowsOf(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                          uint32_t count, uint32_t width) {
  return backend.view(buffer, uint64_t{begin} * width * sizeof(T), uint64_t{count} * width * sizeof(T));
}

void requireLayerPartition(const QwenTargetGeometry &geometry, uint32_t gdnLayers, uint32_t attentionLayers) {
  if (gdnLayers != geometry.stateLayout.layers || attentionLayers != geometry.kvLayout.attentionLayers)
    throw std::logic_error("Qwen target layer partition mismatch");
}

// The out-projection's input sums of the rows [begin, begin + count): one
// float per 64-wide group of the `width`-element rows, in the flat
// [row][group] layout of prefill_linear_q4_sums32.
metal::MetalBuffer prefillSums(metal::MetalBackend &backend, const metal::MetalBuffer &sums, uint32_t begin,
                               uint32_t count, uint32_t width) {
  return backend.view(sums, uint64_t{begin} * (width / 64) * sizeof(float),
                      uint64_t{count} * (width / 64) * sizeof(float));
}

// Copies `rows` rows of a capture layer's output, from row `sourceRow`, into
// capture slot `slot` of the captured hidden rows from row `destinationRow`.
void addCapture(metal::CommandGraph &graph, const QwenTargetGeometry &geometry, uint32_t slot,
                metal::MetalBuffer output, uint32_t sourceRow, metal::MetalBuffer captured,
                uint32_t destinationRow, uint32_t rows) {
  const uint32_t width = geometry.hiddenSize, capturedWidth = geometry.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::RowCopy::add(graph, std::move(output), {sourceRow, width, 0}, std::move(captured),
                    {destinationRow, capturedWidth, slot * width}, rows, width);
}

} // namespace

// The state a prefill command's layers share: its inputs and the next GDN
// and attention layer of the step.
struct QwenTarget::PrefillStep {
  metal::CommandGraph &graph;
  const QwenTargetPrefillBuffers &buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows;
  std::span<const SplashKvLayer> kvLayers;
  // Each sequence's attention plan, which every attention layer runs.
  std::vector<ops::PrefillAttentionPlan> attention{};
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

struct QwenTarget::VerifyStep {
  metal::CommandGraph &graph;
  const QwenTargetVerifyBuffers &buffers;
  std::span<const SplashKvLayer> kvLayers;
  std::span<const kv::ChunkedPrefillParams> chunks;
  uint32_t lanes;
  uint32_t rows;
  ops::VerifyAttentionPlan attention;
  std::optional<ops::MoePlan> moe{};
  // A tree batch keeps `lanes` real lanes of `rowCapacity` node rows, but the
  // row-space operators (linear plans, MoE, the head) see planLanes virtual
  // lanes of targetVerifyRows rows — the same tiles as a wider chain batch.
  uint32_t planLanes = 0;
  uint32_t rowCapacity = 0;
  bool tree = false;
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
  std::span<const uint32_t> liveRows{};
};

metal::MetalBuffer QwenTarget::addPrefill(
    metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
    std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
    std::span<const SplashKvLayer> kvLayers) const {
  if (sequences.empty() ||
      sequences.size() > ExecutionLimits::maximumBatchWidth || !rows ||
      rows > ExecutionLimits::prefillTokenBudget ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen packed prefill batch");
  }
  for (const QwenTargetPrefillSequence &sequence : sequences) {
    if (sequence.convolutionIn.size() != geometry_.stateLayout.layers ||
        sequence.convolutionOut.size() != geometry_.stateLayout.layers ||
        sequence.recurrentIn.size() != geometry_.stateLayout.layers ||
        sequence.recurrentOut.size() != geometry_.stateLayout.layers) {
      throw std::invalid_argument("Qwen prefill state layer mismatch");
    }
  }
  PrefillStep step{graph, buffers, sequences, rows, kvLayers};
  for (const QwenTargetPrefillSequence &sequence : sequences)
    step.attention.push_back(operators_.prefillAttention(
        sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout));
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moePrefill(geometry_.moeShape(), rows);
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addPrefillMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addPrefillFfn(step, layer, residual, output);
      if (const auto slot = geometry_.captureSlot(index))
        for (const QwenTargetPrefillSequence &sequence : sequences)
          for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
            const QwenTargetPrefillCapture &c = sequence.captures[capture];
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
void QwenTarget::addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm,
                                ops::WeightLayout consumer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (consumer == ops::WeightLayout::Affine64)
    ops::Normalization::addRmsWithQ4Sums(step.graph, input, norm, b.normalized, b.projectionSums,
                                         geometry_.hiddenSize, step.rows);
  else
    ops::Normalization::addRms(step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows);
}

// The mixer output projection adds the mixer's rows to `input`. Its affine
// input sums are in `projectionSums` already: the mixer gates wrote them
// beside the rows they stored (the prefill_*_gate*_sums kernels).
void QwenTarget::addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                                  metal::MetalBuffer input, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  operators_.linear().addPrefillResidual(step.graph, hidden, projection, input, output, b.projectionSums,
                                         step.rows, b.linearScratch);
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    ops::GDN::addPrefill(
        step.graph,
        {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights, sequence.convolutionIn[layer],
         sequence.convolutionOut[layer], u16(b.gdnQueries, geometry_.gdnKeyWidth()),
         u16(b.gdnKeys, geometry_.gdnKeyWidth()), u16(b.gdnValues, geometry_.attentionWidth), mixer.decay,
         mixer.timeBias, f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
         sequence.recurrentIn[layer], sequence.recurrentOut[layer], u16(b.recurrent, geometry_.attentionWidth),
         mixer.mixerNorm, u16(b.gdnHidden, geometry_.attentionWidth), b.gdnChunkScratch},
        geometry_.gdnShape(), sequence.rows, mixer.outputHeadOrder,
        prefillSums(backend_, b.projectionSums, sequence.rowBegin, sequence.rows, geometry_.attentionWidth));
  }
  addPrefillOutput(step, b.gdnHidden, mixer.outputProjection, input, b.gdnOutput);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.attentionLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
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
      if (mixer.outputProjection.layout() == ops::WeightLayout::Affine64)
        operators_.linear().addPrefillSums(
            step.graph, hiddenRows,
            prefillSums(backend_, b.projectionSums, sequence.rowBegin,
                        sequence.rows, geometry_.attentionWidth),
            mixer.outputProjection, sequence.rows);
    }
  }
  addPrefillOutput(step, b.attentionHidden, mixer.outputProjection, input, b.attentionOutput);
  return b.attentionOutput;
}

// LFM2's conv layer: input norm, the [B|C|x] in_proj, the double-gated
// causal convolution from the layer's FIFO state into the out-projection's
// input, then the residual output projection.
metal::MetalBuffer QwenTarget::addPrefillMixer(PrefillStep &step, const LfmConvWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const metal::MetalBuffer convRows = u16(b.gdnHidden, geometry_.convolutionDimension);
    ops::LfmConv::addPrefill(step.graph,
                             {u16(b.gdnPacked, geometry_.packedGdnWidth),
                              mixer.convolutionWeights, sequence.convolutionIn[layer],
                              sequence.convolutionOut[layer], convRows},
                             geometry_.convShape(), sequence.rows, mixer.convolutionTapsMajor);
    if (mixer.outputProjection.layout() == ops::WeightLayout::Affine64)
      operators_.linear().addPrefillSums(
          step.graph, convRows,
          prefillSums(backend_, b.projectionSums, sequence.rowBegin, sequence.rows,
                      geometry_.convolutionDimension),
          mixer.outputProjection, sequence.rows);
  }
  addPrefillOutput(step, b.gdnHidden, mixer.outputProjection, input, b.gdnOutput);
  return b.gdnOutput;
}

void QwenTarget::addPrefillDenseFfn(PrefillStep &step, const ops::NormWeights &norm,
                                    const ops::Projection &gate, const ops::Projection &up,
                                    const ops::Projection &down, metal::MetalBuffer residual,
                                    metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  addPrefillNorm(step, residual, norm, gate.layout());
  linear.addPrefill(step.graph, b.normalized, gate, b.denseGateScratch, b.projectionSums,
                    step.rows, b.linearScratch);
  linear.addPrefillUpWithGate(step.graph, b.normalized, up, b.denseGateScratch,
                              b.denseIntermediate, b.projectionSums, b.downProjectionSums, step.rows,
                              b.linearScratch);
  linear.addPrefillResidual(step.graph, b.denseIntermediate, down, residual, output,
                            b.downProjectionSums, step.rows, b.linearScratch);
}

void QwenTarget::addPrefillFfn(PrefillStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                               metal::MetalBuffer output) const {
  addPrefillDenseFfn(step, layer.postAttentionNorm, layer.gateProjection, layer.upProjection,
                     layer.downProjection, residual, output);
}

void QwenTarget::addPrefillFfn(PrefillStep &step, const Qwen3_6MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
}

// LFM2-MoE's per-layer FFN: dense below denseLayers, the MoE block after.
void QwenTarget::addPrefillFfn(PrefillStep &step, const Lfm2MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  std::visit(
      [&](const auto &ffn) {
        using Ffn = std::decay_t<decltype(ffn)>;
        if constexpr (std::is_same_v<Ffn, Lfm2DenseFfn>) {
          addPrefillDenseFfn(step, layer.postAttentionNorm, ffn.gateProjection, ffn.upProjection,
                             ffn.downProjection, residual, output);
        } else {
          ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized,
                                     geometry_.hiddenSize, step.rows);
          ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, ffn, *step.moe);
        }
      },
      layer.ffn);
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    std::span<const SplashKvLayer> kvLayers,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
    bool tree, std::span<const uint32_t> liveRows) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      (!liveRows.empty() && liveRows.size() != lanes) ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.gdnPacked.size() != geometry_.stateLayout.layers ||
      buffers.gdnMixed.size() != geometry_.stateLayout.layers ||
      buffers.gdnDecay.size() != geometry_.stateLayout.layers ||
      buffers.gdnBeta.size() != geometry_.stateLayout.layers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  // A tree lane doubles its row block: SPLASH_TREE_VERIFY_NODES rows, of
  // which nodes 1..15 hold the emitted tree. Tree mode supports the GDN and
  // attention mixers only — LFM2 convolutions stay on the chain path.
  if (tree &&
      (lanes * 2 > ExecutionLimits::maximumBatchWidth || geometry_.convLayers ||
       !buffers.treeNodes || !buffers.treeCounts || !buffers.treeMasks ||
       !buffers.capturedPath)) {
    throw std::invalid_argument("invalid Qwen tree verify batch");
  }
  const uint32_t rowCapacity =
      tree ? uint32_t{SPLASH_TREE_VERIFY_NODES}
           : ExecutionLimits::targetVerifyRows;
  const uint32_t rows = lanes * rowCapacity;
  const uint32_t planLanes = tree ? 2 * lanes : lanes;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, chunks, lanes, rows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(lanes), tree)};
  step.planLanes = planLanes;
  step.rowCapacity = rowCapacity;
  step.tree = tree;
  step.liveRows = liveRows;
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moeDecode(geometry_.moeShape(), planLanes);
  std::visit([&](const auto *weights) {
    for (uint32_t index = 0; index < geometry_.layers; ++index) {
      const auto &layer = weights->layers[index];
      const metal::MetalBuffer input = buffers.hidden[index & 1];
      const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
      const metal::MetalBuffer residual = std::visit(
          [&](const auto &mixer) { return addVerifyMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
      addVerifyFfn(step, layer, residual, output);
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
               buffers.linearScratch);
}

// Each producer emits the table (if any) its consumer's plan reads.
metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.planLanes);
  // Norm, input projection, fused GDN and residual output projection replay
  // from a baked indirect command buffer: every parameter is per-layer
  // geometry and the bound buffers are the decode arena's (the GDN state
  // pair's current/next swap alternates between the span cache's two slots).
  step.graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked[layer], .scratch = b.linearScratch,
              .prepared = normalized},
             mixer.inputProjection, inputPlan);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.planLanes, ops::LinearEpilogue::Residual);
  const ops::PreparedInput hidden =
      step.tree
          ? ops::GDN::addDecodeTree(
                step.graph,
                {b.gdnPacked[layer], mixer.convolutionWeights,
                 b.currentGdnStates, b.nextGdnStates, b.gdnMixed[layer],
                 mixer.decay, mixer.timeBias, b.gdnDecay[layer],
                 b.gdnBeta[layer], mixer.mixerNorm, b.gdnHidden,
                 b.linearScratch},
                b.treeNodes, b.treeCounts, geometry_.gdnShape(), step.lanes,
                layer,
                {geometry_.stateLayout.convolutionLayerBytes(),
                 geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()},
                mixer.outputHeadOrder, outputPlan.input())
          : ops::GDN::addDecode(
                step.graph,
                {b.gdnPacked[layer], mixer.convolutionWeights,
                 b.currentGdnStates, b.nextGdnStates, b.gdnMixed[layer],
                 mixer.decay, mixer.timeBias, b.gdnDecay[layer],
                 b.gdnBeta[layer], mixer.mixerNorm, b.gdnHidden,
                 b.linearScratch},
                geometry_.gdnShape(), step.lanes, layer,
                {geometry_.stateLayout.convolutionLayerBytes(),
                 geometry_.stateLayout.recurrentLayerBytes(),
                 geometry_.stateLayout.convolutionBytes()},
                mixer.outputHeadOrder, outputPlan.input(), step.liveRows);
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = b.gdnOutput, .residual = input, .scratch = b.linearScratch,
              .prepared = hidden},
             mixer.outputProjection, outputPlan);
  step.graph.endBakedSpan();
  return b.gdnOutput;
}

// LFM2's conv layer in a verify step: the same norm and in_proj, then the
// causal convolution over every lane's FIFO state — which also writes the
// bx rows the state commit rolls forward — and the residual out_proj.
metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const LfmConvWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  if (step.tree)
    throw std::invalid_argument("LFM2 convolutions have no tree verify path");
  const uint32_t layer = step.gdnLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.planLanes);
  step.graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked[layer], .scratch = b.linearScratch,
              .prepared = normalized},
             mixer.inputProjection, inputPlan);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.planLanes, ops::LinearEpilogue::Residual);
  // The conv reads every lane's current state and writes its bx rows for the
  // commit; its output is the out-projection's input. The consumer packs its
  // own operand table, so no prepared input is passed.
  ops::LfmConv::addVerify(
      step.graph,
      {b.gdnPacked[layer], mixer.convolutionWeights, b.currentGdnStates,
       b.gdnMixed[layer], b.gdnHidden},
      geometry_.convShape(), step.lanes, ExecutionLimits::targetVerifyRows,
      layer, geometry_.stateLayout.convolutionLayerBytes(), mixer.convolutionTapsMajor);
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = b.gdnOutput, .residual = input, .scratch = b.linearScratch},
             mixer.outputProjection, outputPlan);
  step.graph.endBakedSpan();
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
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
  static const bool noFusedGate = envFlag("SPLASH_NO_FUSED_GATE");
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

void QwenTarget::addVerifyDenseFfn(VerifyStep &step, const ops::NormWeights &norm,
                                   const ops::Projection &gate, const ops::Projection &up,
                                   const ops::Projection &down, metal::MetalBuffer residual,
                                   metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
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

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  addVerifyDenseFfn(step, layer.postAttentionNorm, layer.gateProjection, layer.upProjection,
                    layer.downProjection, residual, output);
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  // One layer's norm plus MoE block is replayed from a baked indirect command
  // buffer when the submission revalidates it; MoE::add nests its own span.
  step.graph.beginBakedSpan();
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
  step.graph.endBakedSpan();
}

// LFM2-MoE's per-layer FFN in a verify step: dense below denseLayers, the
// MoE block after.
void QwenTarget::addVerifyFfn(VerifyStep &step, const Lfm2MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  std::visit(
      [&](const auto &ffn) {
        using Ffn = std::decay_t<decltype(ffn)>;
        if constexpr (std::is_same_v<Ffn, Lfm2DenseFfn>) {
          addVerifyDenseFfn(step, layer.postAttentionNorm, ffn.gateProjection, ffn.upProjection,
                            ffn.downProjection, residual, output);
        } else {
          step.graph.beginBakedSpan();
          ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized,
                                     geometry_.hiddenSize, step.rows);
          ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, ffn, *step.moe);
          step.graph.endBakedSpan();
        }
      },
      layer.ffn);
}

void QwenTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                              metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                              ops::LinearScratch scratch) const {
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan logitsPlan = linear.decodePlan(vocabularyProjection(), lanes);
  // The final norm and vocabulary projection replay from a baked span.
  graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), weightsBase_.finalNorm, finalHidden, geometry_.hiddenSize,
      lanes * ExecutionLimits::targetVerifyRows, scratch, logitsPlan.input());
  linear.add(graph,
             {.input = std::move(finalHidden), .output = std::move(logits), .scratch = scratch,
              .prepared = normalized},
             vocabularyProjection(), logitsPlan);
  graph.endBakedSpan();
}

void QwenTarget::addVerifyInput(metal::CommandGraph &graph,
                                metal::MetalBuffer draftInput,
                                metal::MetalBuffer proposals,
                                metal::MetalBuffer verifyInput,
                                uint32_t lanes) const {
  ops::Embedding::addVerifyInput(graph, std::move(draftInput),
                                 std::move(proposals), std::move(verifyInput),
                                 geometry_.vocabularySize, lanes);
}

void QwenTarget::addVerifyTreeInput(metal::CommandGraph &graph,
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

void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph,
                                QwenTargetCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (!geometry_.gdnLayers) {
    // LFM2's conv FIFOs: every conv layer rolls its lanes' states forward by
    // the retained rows' bx, kept in VerifyMixedBase like the GDN's. The
    // layer's slice of it is lane-major with the decode row count's stride.
    if (!geometry_.convLayers) return;
    const uint32_t laneStride = uint32_t{ExecutionLimits::targetVerifyRows} *
                                geometry_.convolutionDimension * 2;
    const uint64_t layerBytes =
        uint64_t{ExecutionLimits::maximumBatchWidth} * laneStride;
    // One dispatch rolls every conv layer's lanes: the kernel derives each
    // layer's state slot and its `mixed` block from the layer's grid slice.
    ops::LfmConv::addCommit(
        graph,
        {buffers.mixed, buffers.retainedCounts, buffers.currentStates,
         buffers.nextStates},
        geometry_.convShape(), lanes, ExecutionLimits::targetVerifyRows,
        /*layer*/ 0, geometry_.stateLayout.convolutionLayerBytes(),
        geometry_.convLayers, layerBytes, /*tapsMajor*/ false);
    return;
  }
  ops::GDN::addCommit(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      geometry_.gdnShape(), geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

void QwenTarget::addStateCommitTree(metal::CommandGraph &graph,
                                    QwenTargetCommitBuffers buffers,
                                    metal::MetalBuffer retainedPath,
                                    uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || !retainedPath)
    throw std::invalid_argument("invalid Qwen tree state commit batch");
  if (!geometry_.gdnLayers) {
    if (!geometry_.convLayers) return;
    throw std::invalid_argument(
        "LFM2 convolutions have no tree commit path");
  }
  ops::GDN::addCommitTree(
      graph,
      {std::move(buffers.packed), std::move(buffers.mixed),
       std::move(buffers.decay), std::move(buffers.beta), buffers.currentStates,
       buffers.nextStates, std::move(buffers.retainedCounts)},
      std::move(retainedPath), geometry_.gdnShape(),
      geometry_.stateLayout.layers, lanes,
      {geometry_.stateLayout.convolutionLayerBytes(),
       geometry_.stateLayout.recurrentLayerBytes(),
       geometry_.stateLayout.convolutionBytes()});
}

} // namespace splash::model

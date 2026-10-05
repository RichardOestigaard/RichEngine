#include "DSparkDraft.hpp"
#include "Checked.hpp"
#include "DraftCheckpoint.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace richengine::model {
namespace {

void requireLayout(const DFlashDraftLayout &layout) {
  if (layout.kind != DraftKind::DSpark)
    throw WeightStoreError("a DSpark draft requires a DSpark layout");
  if (!layout.layers || layout.layers > 32 || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) || !layout.targetHiddenSize ||
      !layout.kvHeads || !layout.markovRank) {
    throw WeightStoreError("DSpark draft layout contains a zero dimension");
  }
  if (layout.markovRank != RICHENGINE_DRAFT_SELECTOR_RANK) {
    throw WeightStoreError("the DSpark selector kernel is compiled for rank " +
                           std::to_string(RICHENGINE_DRAFT_SELECTOR_RANK));
  }
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
}

// The key and value rows of each layer's fused QKV projection, without a
// copy: affine planes store whole tiles of kQ4StorageN rows in row order
// (AffinePreparation), so rows from a tile boundary on are one range of
// each plane.
std::vector<ops::Projection> contextKvRows(metal::MetalBackend &backend,
                                           const DSparkDraftWeights &weights) {
  const DFlashDraftLayout &layout = weights.layout;
  if (layout.attentionSize >= layout.qkvSize ||
      layout.attentionSize % kQ4StorageN) {
    throw std::invalid_argument(
        "draft key and value rows do not start at a storage tile");
  }
  std::vector<ops::Projection> result;
  result.reserve(weights.layers.size());
  for (const PlainDraftLayerWeights &layer : weights.layers) {
    const ops::AffineWeights &fused = layer.qkvProjection.affine();
    const auto rows = [&](const metal::MetalBuffer &plane) {
      const uint64_t rowBytes = plane.sizeBytes() / layout.qkvSize;
      return backend.view(plane, uint64_t{layout.attentionSize} * rowBytes,
                          uint64_t{layout.contextKvSize()} * rowBytes);
    };
    result.emplace_back(layout.contextKvSize(), layout.hiddenSize,
                        ops::AffineWeights{rows(fused.weights),
                                           rows(fused.scales),
                                           rows(fused.biases)});
  }
  return result;
}

} // namespace

DSparkDraft::DSparkDraft(const DSparkDraftWeights &weights,
                         metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(weights.layout.vocabularySize),
      contextKvProjections_(contextKvRows(backend, weights)) {}

void DSparkDraft::addSelection(
    metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies,
    uint32_t /*treeMask*/) const {
  selector_.addDSpark(graph, buffers,
                      {weights_.markovEmbedding, weights_.markovProjection},
                      anchors, policies);
}

void DSparkDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
  operators_.linear().addPrefillSums(graph, buffers.capturedTargetHidden, buffers.projectionSums,
                                     weights_.contextProjection, rows);
  operators_.linear().addPrefill(graph, buffers.capturedTargetHidden, weights_.contextProjection,
                                 buffers.projected, buffers.projectionSums, rows,
                                 buffers.linearScratch);
  ops::Normalization::addRmsWithQ4Sums(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden,
      buffers.projectionSums, layout.hiddenSize, rows);

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    operators_.linear().addPrefill(graph, buffers.hidden,
                      contextKvProjections_[layer],
                      buffers.contextKv, buffers.projectionSums, rows,
                      buffers.linearScratch);
    for (const DFlashPrefillSpan &span : spans) {
      const uint64_t kvOffset =
          uint64_t{span.compactRow} * layout.contextKvSize() * sizeof(uint16_t);
      const uint64_t ropeOffset =
          uint64_t{span.compactRow} * (layout.attentionHeadDimension / 2) * sizeof(float);
      ops::DraftAttention::addContextPrefill(
          graph,
          backend_.view(buffers.contextKv, kvOffset,
                        uint64_t{span.rows} * layout.contextKvSize() *
                            sizeof(uint16_t)),
          weights_.layers[layer].keyNorm,
          backend_.view(buffers.ropeCos, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          backend_.view(buffers.ropeSin, ropeOffset,
                        uint64_t{span.rows} * (layout.attentionHeadDimension / 2) * sizeof(float)),
          span.ring[layer].keys, span.ring[layer].values, span.rows,
          span.startPosition, layout.attentionShape());
    }
  }
}

void DSparkDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths) const {
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::draftQueryRows;
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes);
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;

  // The plain draft's layer schedule with the DFlash2 blocks' bidirectional
  // attention (causalLayers is zero for a DSpark layout): every dispatch is
  // static per batch width except the attention split/reduce, so the layers
  // and head replay as one baked indirect command buffer span.
  graph.beginBakedSpan();
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const bool causal = (layout.causalLayers >> layer) & 1;
    const PlainDraftLayerWeights &weights = weights_.layers[layer];
    const ops::LinearPlan qkvPlan = linear.decodePlan(weights.qkvProjection, lanes);
    const ops::PreparedInput attentionNormalized = ops::Normalization::addRms(
        graph, buffers.hidden[current], weights.inputNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, qkvPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.proposalQkv, .scratch = scratch,
                .prepared = attentionNormalized},
               weights.qkvProjection, qkvPlan);
    ops::DraftAttention::addPrepare(
        graph,
        {buffers.proposalQkv, buffers.attention, weights.queryNorm,
         weights.keyNorm, buffers.ropeCos, buffers.ropeSin, buffers.queryKeys,
         buffers.queryValues},
        attentionPlan);
    ops::DraftAttention::addDecode(
        graph,
        {buffers.attention, buffers.persistentKeys[layer],
         buffers.persistentValues[layer], buffers.queryKeys,
         buffers.queryValues},
        cacheLengths, attentionPlan, causal);
    ops::DraftAttention::addReorder(graph, buffers.attention,
                                    buffers.proposalQkv, attentionPlan);
    linear.add(graph, {.input = buffers.proposalQkv, .output = buffers.residual,
                       .residual = buffers.hidden[current], .scratch = scratch},
               weights.outputProjection,
               linear.decodePlan(weights.outputProjection, lanes,
                                 ops::LinearEpilogue::Residual));
    const ops::LinearPlan mlpPlan = linear.decodePlan(weights.upProjection, lanes);
    const ops::PreparedInput mlpNormalized = ops::Normalization::addRms(
        graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, mlpPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
                .scratch = scratch,
                .prepared = mlpNormalized},
               weights.upProjection,
               linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::GateUp, &weights.gateProjection),
               &weights.gateProjection);
    linear.add(graph, {.input = buffers.intermediate, .output = buffers.hidden[next],
                       .residual = buffers.residual, .scratch = scratch},
               weights.downProjection,
               linear.decodePlan(weights.downProjection, lanes,
                                 ops::LinearEpilogue::Residual));
  }

  // The final norm feeds the shared vocabulary head; the DSpark selector
  // works on the head's logits directly and needs no projection of its own.
  const ops::LinearPlan headPlan = linear.decodePlan(vocabularyProjection, lanes);
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[weights_.layout.layers & 1], weights_.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, scratch, headPlan.input());
  linear.add(
      graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
      vocabularyProjection, headPlan);
  graph.endBakedSpan();
}

void DSparkDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions) const {
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  linear.add(graph, {.input = buffers.capturedTargetHidden, .output = buffers.projected, .scratch = scratch},
             weights_.contextProjection, linear.decodePlan(weights_.contextProjection, lanes));
  // Every layer's key and value projection reads the same normalized rows,
  // prepared for the first layer's plan.
  ops::LinearPlan kvPlan = linear.decodePlan(contextKvProjections_[0], lanes);
  ops::PreparedInput hidden = ops::Normalization::addRms(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize, rows,
      scratch, kvPlan.input());

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &projection = contextKvProjections_[layer];
    if (layer) kvPlan = linear.decodePlan(projection, lanes);
    hidden = linear.add(graph,
                        {.input = buffers.hidden, .output = buffers.contextKv, .scratch = scratch,
                         .prepared = hidden},
                        projection, kvPlan);
    ops::DraftAttention::addContextCommit(
        graph, buffers.contextKv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.attentionShape());
  }
}

namespace {

// Reads a DSpark draft's files in their section order: each layer (the plain
// draft's sections), then model.bin (fc, the norms, the Markov head and the
// confidence head).
template <class Files>
DSparkDraftWeights readDraft(metal::MetalBackend &backend, Files &files,
                             const DFlashDraftLayout &layout) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DSparkDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t headNormBytes = checkedMultiply<WeightStoreError>(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = files.layer(layerIndex);
    PlainDraftLayerWeights layer;
    layer.inputNorm = readNorm(file, layout.hiddenSize, false, "input-norm");
    layer.inputNorm.rmsEpsilon = layout.rmsEpsilon;
    layer.qkvProjection = readAffineProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readAffineProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.postAttentionNorm.rmsEpsilon = layout.rmsEpsilon;
    layer.gateProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readAffineProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readAffineProjection(
        file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file = files.model();
    result.contextProjection = readAffineProjection(
        file, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.hiddenNorm = readNorm(file, layout.hiddenSize, false, "hidden-norm");
    result.hiddenNorm.rmsEpsilon = layout.rmsEpsilon;
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    result.finalNorm.rmsEpsilon = layout.rmsEpsilon;
    const uint64_t markovBytes = checkedMultiply<WeightStoreError>(
        checkedMultiply<WeightStoreError>(layout.vocabularySize,
                                          layout.markovRank,
                                          "draft markov elements"),
        kBFloat16Bytes, "draft markov bytes");
    result.markovEmbedding = file.section(markovBytes, "markov-embedding");
    result.markovProjection = file.section(markovBytes, "markov-projection");
    result.confidenceWeight = file.section(
        checkedMultiply<WeightStoreError>(layout.hiddenSize + layout.markovRank,
                                          kBFloat16Bytes,
                                          "draft confidence weight bytes"),
        "confidence-weight");
    result.confidenceBias =
        file.section(kBFloat16Bytes, "confidence-bias");
    file.finish();
    result.files.push_back(file.record());
  }

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace

WeightFile PackedDSparkDraftFiles::layer(uint32_t index) const {
  const std::string filename = "layer-" + std::to_string(index) + ".bin";
  return images.load(packedImage(directory / filename, "draft/" + filename,
                                 kDSparkDraftMagic, index, 0));
}

WeightFile PackedDSparkDraftFiles::model() const {
  return images.load(packedImage(directory / "model.bin", "draft/model.bin",
                                 kDSparkDraftMagic, layout.layers, 1));
}

DSparkDraftWeights loadDSparkDraftWeights(metal::MetalBackend &backend,
                                          const DSparkDraftFiles &files,
                                          DFlashDraftLayout layout) {
  requireLayout(layout);
  if (const auto *checkpoint =
          std::get_if<std::reference_wrapper<DSparkCheckpointLoader>>(&files))
    return readDraft(backend, checkpoint->get(), layout);
  return readDraft(backend, std::get<PackedDSparkDraftFiles>(files), layout);
}

} // namespace richengine::model

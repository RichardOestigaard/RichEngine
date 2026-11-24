#include "DFlashV1Draft.hpp"
#include "Checked.hpp"
#include "DraftCheckpoint.hpp"
#include "metal/abi/Vision.h"

#include "ops/KernelNames.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace richengine::model {
namespace {

void requireLayout(const DFlashDraftLayout &layout) {
  if (layout.kind != DraftKind::DFlashV1)
    throw WeightStoreError("a plain draft requires a plain layout");
  if (!layout.layers || layout.layers > 32 || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) || !layout.targetHiddenSize ||
      !layout.kvHeads) {
    throw WeightStoreError("plain draft layout contains a zero dimension");
  }
  // A draft whose dims are off the quantized tiles loads F32 float
  // segments instead — DraftCheckpoint.cpp's floatProjections makes the
  // same decision when it plans the files.
  const auto q4 = [](uint32_t rows, uint32_t columns) {
    return rows % kQ4StorageN == 0 && columns % kQ4GroupElements == 0;
  };
  if (q4(layout.qkvSize, layout.hiddenSize) &&
      q4(layout.hiddenSize, layout.attentionSize) &&
      q4(layout.intermediateSize, layout.hiddenSize) &&
      q4(layout.hiddenSize, layout.intermediateSize) &&
      q4(layout.hiddenSize, layout.targetHiddenSize)) {
    validateQ4Layout(layout.qkvSize, layout.hiddenSize);
    validateQ4Layout(layout.hiddenSize, layout.attentionSize);
    validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
    validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
    validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
  }
}

// The key and value rows of each layer's fused QKV projection, without a
// copy: affine planes store whole tiles of kQ4StorageN rows in row order
// (AffinePreparation), so rows from a tile boundary on are one range of
// each plane.
std::vector<ops::Projection> contextKvRows(metal::MetalBackend &backend,
                                           const DFlashV1DraftWeights &weights) {
  const DFlashDraftLayout &layout = weights.layout;
  if (layout.attentionSize >= layout.qkvSize ||
      layout.attentionSize % kQ4StorageN) {
    throw std::invalid_argument(
        "draft key and value rows do not start at a storage tile");
  }
  std::vector<ops::Projection> result;
  result.reserve(weights.layers.size());
  for (const DFlashV1DraftLayerWeights &layer : weights.layers) {
    if (layer.qkvProjection.layout() == ops::WeightLayout::Block32) {
      // The block (AF4) fused QKV's context K/V rows: a whole-tile row slice
      // of the segment's planes — 256-row tiles of 16-byte plane0 units per
      // 32-column group and 4-byte meta units per 64-column block.
      const ops::QuantizedSegment &fused =
          layer.qkvProjection.blocks().segments.at(0);
      if (fused.isFloat()) {
        // F32 float segments: the K/V rows are the segments covering the
        // tail, renumbered to lead the slice projection's columns — each
        // plane0 holds [rows][inputs] floats.
        ops::BlockWeights slice;
        uint32_t covered = 0;
        for (const ops::QuantizedSegment &s :
             layer.qkvProjection.blocks().segments) {
          if (s.columnOffset < layout.attentionSize) continue;
          ops::QuantizedSegment tail = s;
          tail.columnOffset = covered;
          covered += s.outputSize;
          slice.segments.push_back(std::move(tail));
        }
        if (covered != layout.contextKvSize())
          throw std::invalid_argument(
              "draft float QKV does not slice the context K/V");
        result.emplace_back(layout.contextKvSize(), layout.hiddenSize,
                            std::move(slice));
        continue;
      }
      constexpr uint32_t kTileRows = 256;
      if (layout.attentionSize % kTileRows ||
          layout.contextKvSize() % kTileRows || fused.plane1)
        throw std::invalid_argument(
            "draft block QKV rows are not tile-aligned for the context slice");
      const uint64_t tile0 = layout.attentionSize / kTileRows;
      const uint64_t tiles = layout.contextKvSize() / kTileRows;
      const uint64_t plane0Tile =
          uint64_t(layout.hiddenSize / 32) * kTileRows * 16;
      const uint64_t metaTile =
          uint64_t(layout.hiddenSize / 64) * kTileRows * 4;
      result.emplace_back(
          layout.contextKvSize(), layout.hiddenSize,
          ops::BlockWeights{{ops::QuantizedSegment::planes(
              fused.formatId, layout.contextKvSize(), layout.hiddenSize,
              backend.view(fused.plane0, tile0 * plane0Tile,
                           tiles * plane0Tile),
              {},
              backend.view(fused.meta, tile0 * metaTile, tiles * metaTile))}});
      continue;
    }
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

DFlashV1Draft::DFlashV1Draft(const DFlashV1DraftWeights &weights,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(weights.layout.vocabularySize,
                paddedVocabularySize(weights.layout.vocabularySize)),
      contextKvProjections_(contextKvRows(backend, weights)) {}

void DFlashV1Draft::addSelection(
    metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies,
    uint32_t treeMask) const {
  if (treeMask)
    selector_.addPlainTree(graph, buffers, anchors, policies, treeMask,
                           weights_.layout.queryRows());
  else
    selector_.addPlain(graph, buffers, anchors, policies,
                     weights_.layout.queryRows());
}

void DFlashV1Draft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
  // An off-tile draft's context projection is a float segment the GGUF
  // block plans run; the affine input sums and their norm variant do not
  // apply.
  if (weights_.contextProjection.layout() == ops::WeightLayout::Block32) {
    operators_.linear().addPrefill(graph, buffers.capturedTargetHidden,
                                   weights_.contextProjection, buffers.projected,
                                   buffers.projectionSums, rows,
                                   buffers.linearScratch);
    ops::Normalization::addRms(graph, buffers.projected, weights_.hiddenNorm,
                               buffers.hidden, layout.hiddenSize, rows);
  } else {
    operators_.linear().addPrefillSums(graph, buffers.capturedTargetHidden, buffers.projectionSums,
                                       weights_.contextProjection, rows);
    operators_.linear().addPrefill(graph, buffers.capturedTargetHidden, weights_.contextProjection,
                                   buffers.projected, buffers.projectionSums, rows,
                                   buffers.linearScratch);
    ops::Normalization::addRmsWithQ4Sums(
        graph, buffers.projected, weights_.hiddenNorm, buffers.hidden,
        buffers.projectionSums, layout.hiddenSize, rows);
  }

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    // The affine prefill tile miscomputes a block projection, so the
    // context K/V runs through the decode plan — the commit path's proven
    // kernel — in verify-row tiles.
    if (contextKvProjections_[layer].layout() == ops::WeightLayout::Block32) {
      for (uint32_t begin = 0; begin < rows; begin += layout.verifyRows()) {
      // Decode tiles run verifyRows rows; the tail's extra rows write past
      // the used span into the padded arena buffer.
      const ops::LinearPlan plan = operators_.linear().decodePlan(
          contextKvProjections_[layer], 1, ops::LinearEpilogue::None, nullptr,
          layout.verifyRows());
      operators_.linear().add(
          graph,
          {.input = backend_.view(buffers.hidden,
                                  uint64_t{begin} * layout.hiddenSize * 2,
                                  uint64_t{layout.verifyRows()} *
                                      layout.hiddenSize * 2),
           .output = backend_.view(buffers.contextKv,
                                   uint64_t{begin} * layout.contextKvSize() * 2,
                                   uint64_t{layout.verifyRows()} *
                                       layout.contextKvSize() * 2),
           .scratch = buffers.linearScratch},
          contextKvProjections_[layer], plan);
      }
    } else {
      operators_.linear().addPrefill(graph, buffers.hidden,
                        contextKvProjections_[layer],
                        buffers.contextKv, buffers.projectionSums, rows,
                        buffers.linearScratch);
    }
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

void DFlashV1Draft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths,
    const metal::MetalBuffer &devicePack) const {
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft decode batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * layout.queryRows();
  const auto attentionPlan =
      operators_.draftAttention(layout.attentionShape(), lanes, layout.queryRows());
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  // F32 float segments (a draft off the quantized tiles): the fp32
  // kernels take no epilogue, so the residual adds dispatch
  // draft_residual_add and the gate/up pair a standalone silu-mul.
  const bool floats =
      !weights_.layers.empty() &&
      weights_.layers.front().qkvProjection.layout() ==
          ops::WeightLayout::Block32 &&
      weights_.layers.front().qkvProjection.blocks().segments.front().isFloat();

  // Every draft decode dispatch is static per batch width except the
  // attention split/reduce, whose parameters carry each lane's cache length;
  // the pair's payloads are patchable, so the layers and head replay as one
  // baked indirect command buffer span.
  graph.beginBakedSpan();
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const bool causal = (layout.causalLayers >> layer) & 1;
    const DFlashV1DraftLayerWeights &weights = weights_.layers[layer];
    const ops::LinearPlan qkvPlan = linear.decodePlan(weights.qkvProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
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
         buffers.queryValues, {}},
        cacheLengths, attentionPlan, causal, devicePack);
    if (floats) {
      linear.add(graph, {.input = buffers.attention, .output = buffers.residual,
                         .scratch = scratch},
                 weights.outputProjection,
                 linear.decodePlan(weights.outputProjection, lanes,
                                   ops::LinearEpilogue::None, nullptr,
                                   layout.queryRows()));
      ops::DraftAttention::addResidual(graph, buffers.residual,
                                       buffers.hidden[current],
                                       buffers.residual, attentionPlan);
    } else {
      linear.add(graph, {.input = buffers.attention, .output = buffers.residual,
                         .residual = buffers.hidden[current], .scratch = scratch},
                 weights.outputProjection,
                 linear.decodePlan(weights.outputProjection, lanes,
                                   ops::LinearEpilogue::Residual, nullptr,
                                   layout.queryRows()));
    }
    const ops::LinearPlan mlpPlan = linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
    const ops::PreparedInput mlpNormalized = ops::Normalization::addRms(
        graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, mlpPlan.input());
    if (floats) {
      // gate into the scratch, up over it, then the elementwise
      // silu(gate) * up — the vision encoder's is the kernel that takes it.
      linear.add(graph, {.input = buffers.normalized, .output = buffers.gateScratch,
                         .scratch = scratch, .prepared = mlpNormalized},
                 weights.gateProjection, mlpPlan);
      linear.add(graph, {.input = buffers.normalized, .output = buffers.intermediate,
                         .scratch = scratch, .prepared = mlpNormalized},
                 weights.upProjection, mlpPlan);
      const uint32_t elements = rows * layout.intermediateSize;
      graph.add(std::string(ops::kLfmVisionSiluMul),
                {buffers.gateScratch, buffers.intermediate,
                 buffers.intermediate},
                VisionNormParams{elements}, {(elements + 255) / 256, 1, 1},
                {256, 1, 1});
    } else {
      linear.add(graph,
                 {.input = buffers.normalized, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
                  .scratch = scratch,
                  .prepared = mlpNormalized},
                 weights.upProjection,
                 linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::GateUp, &weights.gateProjection, layout.queryRows()),
                 &weights.gateProjection);
    }
    if (floats) {
      linear.add(graph, {.input = buffers.intermediate, .output = buffers.hidden[next],
                         .scratch = scratch},
                 weights.downProjection,
                 linear.decodePlan(weights.downProjection, lanes,
                                   ops::LinearEpilogue::None, nullptr,
                                   layout.queryRows()));
      ops::DraftAttention::addResidual(graph, buffers.hidden[next],
                                       buffers.residual,
                                       buffers.hidden[next], attentionPlan);
    } else {
      linear.add(graph, {.input = buffers.intermediate, .output = buffers.hidden[next],
                         .residual = buffers.residual, .scratch = scratch},
                 weights.downProjection,
                 linear.decodePlan(weights.downProjection, lanes,
                                   ops::LinearEpilogue::Residual, nullptr,
                                   layout.queryRows()));
    }
  }

  // The final norm feeds the shared vocabulary head; the plain selector
  // works on the head's logits directly and needs no projection of its own.
  const ops::LinearPlan headPlan = linear.decodePlan(vocabularyProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[weights_.layout.layers & 1], weights_.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, scratch, headPlan.input());
  linear.add(
      graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
      vocabularyProjection, headPlan);
  // Gemma's softcap applies to the draft's proposal logits too.
  if (layout.logitSoftcap > 0.0F) {
    const uint32_t count = rows * layout.vocabularySize;
    graph.addTail(std::string(ops::kDecodeLogitSoftcap), {buffers.logits},
                  layout.logitSoftcap, count,
                  {(count + 255) / 256, 1, 1}, {256, 1, 1});
  }
  graph.endBakedSpan();
}

void DFlashV1Draft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions) const {
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth ||
      buffers.persistentKeys.size() != weights_.layout.layers ||
      buffers.persistentValues.size() != weights_.layout.layers) {
    throw std::invalid_argument("invalid draft context batch");
  }
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t rows = lanes * layout.verifyRows();
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  linear.add(graph, {.input = buffers.capturedTargetHidden, .output = buffers.projected, .scratch = scratch},
             weights_.contextProjection, linear.decodePlan(weights_.contextProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.verifyRows()));
  // Every layer's key and value projection reads the same normalized rows,
  // prepared for the first layer's plan.
  ops::LinearPlan kvPlan = linear.decodePlan(contextKvProjections_[0], lanes, ops::LinearEpilogue::None, nullptr, layout.verifyRows());
  ops::PreparedInput hidden = ops::Normalization::addRms(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize, rows,
      scratch, kvPlan.input());

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &projection = contextKvProjections_[layer];
    if (layer) kvPlan = linear.decodePlan(projection, lanes, ops::LinearEpilogue::None, nullptr, layout.verifyRows());
    hidden = linear.add(graph,
                        {.input = buffers.hidden, .output = buffers.contextKv, .scratch = scratch,
                         .prepared = hidden},
                        projection, kvPlan);
    ops::DraftAttention::addContextCommit(
        graph, buffers.contextKv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.attentionShape(), layout.verifyRows());
  }
}

namespace {

// Reads a plain draft's files in their section order: each layer, then
// model.bin.
template <class Files>
DFlashV1DraftWeights readDraft(metal::MetalBackend &backend, Files &files,
                            const DFlashDraftLayout &layout) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DFlashV1DraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t headNormBytes = checkedMultiply<WeightStoreError>(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = files.layer(layerIndex);
    DFlashV1DraftLayerWeights layer;
    layer.inputNorm = readNorm(file, layout.hiddenSize, false, "input-norm");
    // Block files read the fused QKV through the oQ loop: it covers both
    // the single AF4 segment of a quantized draft and the three F32 float
    // segments (q, k, v) of an off-tile one.
    layer.qkvProjection = file.record().type & kBlockWeightsImageType
                              ? readOqFused(file, layout.qkvSize,
                                            layout.hiddenSize, "qkv")
                              : readProjection(file, layout.qkvSize,
                                               layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.gateProjection = readProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection = readProjection(
        file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection = readProjection(
        file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    file.finish();
    result.files.push_back(file.record());
    result.layers.push_back(std::move(layer));
  }

  {
    WeightFile file = files.model();
    result.contextProjection = readProjection(
        file, layout.hiddenSize, layout.targetHiddenSize,
        "context-projection");
    result.hiddenNorm = readNorm(file, layout.hiddenSize, false, "hidden-norm");
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    // The draft's own embed_tokens trails the norms when the layout
    // declares one (has_embed_tokens): F32 native rows the block's input
    // gather reads instead of the target's table.
    if (layout.tokenEmbedding)
      result.tokenEmbedding = readOqEmbedding(
          file, layout.vocabularySize, layout.hiddenSize, "token-embedding");
    file.finish();
    result.files.push_back(file.record());
  }

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace

WeightFile PackedDFlashV1DraftFiles::layer(uint32_t index) const {
  const std::string filename = "layer-" + std::to_string(index) + ".bin";
  return images.load(packedImage(directory / filename, "draft/" + filename,
                                 kDFlashV1DraftMagic, index, 0));
}

WeightFile PackedDFlashV1DraftFiles::model() const {
  return images.load(packedImage(directory / "model.bin", "draft/model.bin",
                                 kDFlashV1DraftMagic, layout.layers, 1));
}

DFlashV1DraftWeights loadDFlashV1DraftWeights(metal::MetalBackend &backend,
                                        const DFlashV1DraftFiles &files,
                                        DFlashDraftLayout layout) {
  requireLayout(layout);
  if (const auto *checkpoint =
          std::get_if<std::reference_wrapper<DFlashV1DraftCheckpointLoader>>(&files))
    return readDraft(backend, checkpoint->get(), layout);
  return readDraft(backend, std::get<PackedDFlashV1DraftFiles>(files), layout);
}

} // namespace richengine::model

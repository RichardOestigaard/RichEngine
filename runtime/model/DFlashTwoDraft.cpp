#include "DFlashTwoDraft.hpp"
#include "Checked.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/LiLiCorr.h"
#include "metal/abi/Mtp.h"
#include "DraftCheckpoint.hpp"
#include "Tuning.hpp"
#include "model/ModelPolicy.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace richengine::model {
namespace {

void requireLayout(const DFlashDraftLayout &layout) {
  if (!layout.layers || !layout.hiddenSize || !layout.vocabularySize ||
      !layout.dynamicSize || !layout.qkvSize || !layout.attentionSize ||
      !layout.intermediateSize || !layout.attentionHeadDimension ||
      !(layout.rotaryTheta > 0.0F) ||
      !layout.targetHiddenSize || !layout.kvHeads ||
      (layout.kind != DraftKind::LiLiCorr && !layout.selectorRank)) {
    throw WeightStoreError("DFlash draft layout contains a zero dimension");
  }
  if (layout.kind != DraftKind::LiLiCorr &&
      layout.selectorRank != RICHENGINE_DRAFT_SELECTOR_RANK) {
    throw WeightStoreError("draft selector kernels are compiled for rank " +
                           std::to_string(RICHENGINE_DRAFT_SELECTOR_RANK));
  }
  // The conv kernels derive the dynamic width from the hidden size
  // (Hidden/4): hidden 2560's 640 rows sit off the affine 256-row tile, so
  // such a draft's packed files carry the dynamic projections unquantized
  // and only the per-file readers enforce their format's alignment.
  if (layout.dynamicSize * 4 != layout.hiddenSize)
    throw WeightStoreError("draft dynamic width is not hiddenSize / 4");
  validateQ4Layout(layout.qkvSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionSize);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.hiddenSize, layout.targetHiddenSize);
  if (layout.kind != DraftKind::LiLiCorr)
    validateQ4Layout(layout.selectorRank, layout.hiddenSize);
}

// The key and value rows of each layer's fused QKV projection, without a
// copy: affine planes store whole tiles of kQ4StorageN rows in row order
// (AffinePreparation), so rows from a tile boundary on are one range of
// each plane.
std::vector<ops::Projection> contextKvRows(metal::MetalBackend &backend,
                                           const DFlashDraftWeights &weights) {
  const DFlashDraftLayout &layout = weights.layout;
  if (layout.attentionSize >= layout.qkvSize ||
      layout.attentionSize % kQ4StorageN) {
    throw std::invalid_argument(
        "draft key and value rows do not start at a storage tile");
  }
  std::vector<ops::Projection> result;
  result.reserve(weights.layers.size());
  for (const DFlashDraftLayerWeights &layer : weights.layers) {
    if (layer.qkvProjection.layout() == ops::WeightLayout::Block32) {
      // The block (AF4) fused QKV's context K/V rows: a whole-tile row slice
      // of the segment's planes — 256-row tiles of 16-byte plane0 units per
      // 32-column group and 4-byte meta units per 64-column block.
      const ops::QuantizedSegment &fused =
          layer.qkvProjection.blocks().segments.at(0);
      if (fused.isFloat()) {
        // LiLiCorr's GGUF stores q, k and v separately. The loader fuses
        // them as bf16 float segments at their output-column offsets, so the
        // context slice is simply the tail segments renumbered to zero.
        ops::BlockWeights slice;
        uint32_t covered = 0;
        for (const ops::QuantizedSegment &segment : layer.qkvProjection.blocks().segments) {
          if (segment.columnOffset < layout.attentionSize) continue;
          ops::QuantizedSegment tail = segment;
          tail.columnOffset = covered;
          covered += tail.outputSize;
          slice.segments.push_back(std::move(tail));
        }
        if (covered != layout.contextKvSize())
          throw std::invalid_argument("draft float QKV does not slice the context K/V");
        result.emplace_back(layout.contextKvSize(), layout.hiddenSize, std::move(slice));
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

metal::DispatchSize elementsGrid(uint64_t elements, uint32_t threads = 256) {
  return {(elements + threads - 1) / threads, 1, 1};
}

LiLiCorrGemmParams gemmParams(uint32_t rows, uint32_t rowsPerLane,
                              uint32_t inputSize, uint32_t outputSize,
                              uint32_t inputOffset, uint32_t outputOffset,
                              uint32_t epilogue = 0, float scale = 1.0F) {
  return {rows, inputSize, outputSize, rowsPerLane,
          LILICORR_SCRATCH_STRIDE, inputOffset, outputOffset, epilogue, scale};
}

bool gemmMmaShape(const LiLiCorrGemmParams &params) {
  return params.rows_per_lane && params.rows_per_lane <= LILICORR_NODES &&
         params.rows % params.rows_per_lane == 0 &&
         params.input_size % 8 == 0 && params.output_size % 8 == 0;
}

void addGemmF32(metal::CommandGraph &graph, const metal::MetalBuffer &scratch,
                const metal::MetalBuffer &weight, const metal::MetalBuffer &bias,
                const LiLiCorrGemmParams &params) {
  if (gemmMmaShape(params)) {
    graph.add("lilicorr_gemm_f32_mma", {scratch, weight, bias, scratch}, params,
              {params.output_size / 8, params.rows / params.rows_per_lane, 1},
              {256, 1, 1});
    return;
  }
  graph.add("lilicorr_gemm_f32", {scratch, weight, bias, scratch}, params,
            elementsGrid(uint64_t{params.rows} * params.output_size));
}

void addGemmNvfp4(metal::CommandGraph &graph, const metal::MetalBuffer &scratch,
                  const metal::MetalBuffer &weight, const metal::MetalBuffer &bias,
                  const LiLiCorrGemmParams &params) {
  if (gemmMmaShape(params)) {
    graph.add("lilicorr_gemm_nvfp4_mma", {scratch, weight, bias, scratch}, params,
              {params.output_size / 8, params.rows / params.rows_per_lane, 1},
              {256, 1, 1});
    return;
  }
  graph.add("lilicorr_gemm_nvfp4", {scratch, weight, bias, scratch}, params,
            elementsGrid(uint64_t{params.rows} * params.output_size));
}

LiLiCorrNormParams normParams(uint32_t rows, uint32_t rowsPerLane,
                              uint32_t width, uint32_t inputOffset,
                              uint32_t outputOffset, float epsilon) {
  return {rows, width, rowsPerLane, LILICORR_SCRATCH_STRIDE,
          inputOffset, outputOffset, epsilon};
}

void addRmsNorm(metal::CommandGraph &graph, const metal::MetalBuffer &scratch,
                const metal::MetalBuffer &weight,
                const LiLiCorrNormParams &params) {
  graph.add("lilicorr_rmsnorm", {weight, scratch}, params,
            {params.rows, 1, 1}, {256, 1, 1});
}

void addResidual(metal::CommandGraph &graph, const metal::MetalBuffer &scratch,
                 uint32_t rows, uint32_t deltaOffset) {
  const LiLiCorrGemmParams params =
      gemmParams(rows, LILICORR_NODES, LILICORR_HIDDEN, 0, deltaOffset,
                 LILICORR_SCRATCH_X);
  graph.add("lilicorr_residual", {scratch}, params,
            elementsGrid(uint64_t{rows} * LILICORR_HIDDEN));
}

void addL2Norm(metal::CommandGraph &graph, const metal::MetalBuffer &scratch,
               uint32_t rows, uint32_t rowsPerLane, uint32_t offset,
               float epsilon) {
  const LiLiCorrNormParams params =
      normParams(rows, rowsPerLane, LILICORR_HIDDEN, offset, offset, epsilon);
  graph.add("lilicorr_l2norm", {scratch}, params, {rows, 1, 1}, {256, 1, 1});
}

void addLiLiCorrScorer(metal::CommandGraph &graph,
                       const ops::DraftSelectorBuffers &buffers,
                       const LiLiCorrScorerWeights &scorer,
                       uint32_t lanes, uint32_t queryRows,
                       uint32_t vocabulary, uint32_t logitsStride) {
  const ops::DraftLiLiCorrBuffers &input = *buffers.lilicorr;
  if (input.tokenEmbeddingFormat != GGUF_FMT_Q80)
    throw std::invalid_argument("LiLiCorr scorer requires a q8_0 token table");
  const metal::MetalBuffer &scratch = input.scratch;
  const uint32_t nodeRows = lanes * LILICORR_NODES;
  const uint32_t passRows = lanes * LILICORR_SLOTS;

  for (uint32_t lane = 0; lane < lanes; ++lane) {
    graph.add("lilicorr_stage_anchor", {input.anchors[lane], scratch},
              LiLiCorrCopyParams{LILICORR_TARGET_HIDDEN, lane},
              elementsGrid(LILICORR_TARGET_HIDDEN));
  }
  const LiLiCorrFeatureParams features{lanes, vocabulary, logitsStride, queryRows};
  graph.add("lilicorr_features",
            {buffers.logits, buffers.candidates, buffers.proposalProbabilities, scratch},
            features, {lanes * LILICORR_SLOTS, 1, 1}, {256, 1, 1});
  graph.add("lilicorr_gather_inputs",
            {input.tokenEmbedding, input.finalHidden, scratch}, features,
            elementsGrid(uint64_t{lanes} *
                         (LILICORR_NODES + LILICORR_SLOTS) *
                         LILICORR_TARGET_HIDDEN));

  addGemmNvfp4(graph, scratch, scorer.tokenProjection, scorer.tokenBias,
               gemmParams(nodeRows, LILICORR_NODES, LILICORR_TARGET_HIDDEN,
                          LILICORR_HIDDEN, LILICORR_SCRATCH_TOKEN_INPUT,
                          LILICORR_SCRATCH_TOKEN_STATES, 0, scorer.tokenScale));
  addGemmNvfp4(graph, scratch, scorer.passProjection, scorer.passBias,
               gemmParams(passRows, LILICORR_SLOTS, LILICORR_TARGET_HIDDEN,
                          LILICORR_HIDDEN, LILICORR_SCRATCH_PASS_INPUT,
                          LILICORR_SCRATCH_PASS_STATES, 0, scorer.passScale));
  addGemmNvfp4(graph, scratch, scorer.contextProjection, scorer.contextBias,
               gemmParams(lanes, 1, LILICORR_TARGET_HIDDEN, LILICORR_HIDDEN,
                          LILICORR_SCRATCH_ANCHOR_INPUT,
                          LILICORR_SCRATCH_ANCHOR_STATE, 0, scorer.contextScale));

  graph.add("lilicorr_feature_norm",
            {scorer.featureNorm, scorer.featureNormBias, scratch},
            normParams(nodeRows, LILICORR_NODES, LILICORR_FEATURE_WIDTH,
                       LILICORR_SCRATCH_FEATURES,
                       LILICORR_SCRATCH_FEATURE_NORMED, 1e-5F),
            elementsGrid(nodeRows));
  addGemmF32(graph, scratch, scorer.featureUp, scorer.featureUpBias,
             gemmParams(nodeRows, LILICORR_NODES, LILICORR_FEATURE_WIDTH,
                        LILICORR_HIDDEN, LILICORR_SCRATCH_FEATURE_NORMED,
                        LILICORR_SCRATCH_FEATURE_UP, 1));
  addGemmF32(graph, scratch, scorer.featureDown, scorer.featureDownBias,
             gemmParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                        LILICORR_HIDDEN, LILICORR_SCRATCH_FEATURE_UP,
                        LILICORR_SCRATCH_FEATURE_STATES));
  graph.add("lilicorr_assemble_nodes",
            {scorer.slotEmbedding, scorer.rankEmbedding, scratch},
            normParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN, 0, 0, 0),
            elementsGrid(uint64_t{nodeRows} * LILICORR_HIDDEN));

  for (const LiLiCorrScorerLayerWeights &layer : scorer.layers) {
    addRmsNorm(graph, scratch, layer.attentionNorm,
               normParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                          LILICORR_SCRATCH_X, LILICORR_SCRATCH_NORMED,
                          scorer.rmsEpsilon));
    addGemmF32(graph, scratch, layer.attentionIn, layer.attentionInBias,
               gemmParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                          3 * LILICORR_HIDDEN, LILICORR_SCRATCH_NORMED,
                          LILICORR_SCRATCH_QKV));
    graph.add("lilicorr_attention", {scorer.attentionBias, scratch},
              LiLiCorrAttentionParams{lanes, LILICORR_SCRATCH_STRIDE,
                                      1.0F / 11.313708499F},
              {lanes * LILICORR_HEADS * LILICORR_NODES, 1, 1},
              {LILICORR_HEAD_DIM, 1, 1});
    addGemmNvfp4(graph, scratch, layer.attentionOut, layer.attentionOutBias,
                 gemmParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                            LILICORR_HIDDEN, LILICORR_SCRATCH_ATTENTION_OUT,
                            LILICORR_SCRATCH_MLP, 0, layer.attentionOutScale));
    addResidual(graph, scratch, nodeRows, LILICORR_SCRATCH_MLP);
    addRmsNorm(graph, scratch, layer.mlpNorm,
               normParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                          LILICORR_SCRATCH_X, LILICORR_SCRATCH_NORMED,
                          scorer.rmsEpsilon));
    addGemmNvfp4(graph, scratch, layer.mlpUp, layer.mlpUpBias,
                 gemmParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                            LILICORR_MLP_HIDDEN, LILICORR_SCRATCH_NORMED,
                            LILICORR_SCRATCH_MLP, 1, layer.mlpUpScale));
    addGemmNvfp4(graph, scratch, layer.mlpDown, layer.mlpDownBias,
                 gemmParams(nodeRows, LILICORR_NODES, LILICORR_MLP_HIDDEN,
                            LILICORR_HIDDEN, LILICORR_SCRATCH_MLP,
                            LILICORR_SCRATCH_ATTENTION_OUT, 0,
                            layer.mlpDownScale));
    addResidual(graph, scratch, nodeRows, LILICORR_SCRATCH_ATTENTION_OUT);
  }

  addRmsNorm(graph, scratch, scorer.outputNorm,
             normParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                        LILICORR_SCRATCH_X, LILICORR_SCRATCH_X,
                        scorer.rmsEpsilon));
  addRmsNorm(graph, scratch, scorer.anchorNorm,
             normParams(lanes, 1, LILICORR_HIDDEN,
                        LILICORR_SCRATCH_ANCHOR_STATE,
                        LILICORR_SCRATCH_ANCHOR_NORMED, scorer.rmsEpsilon));
  graph.add("lilicorr_factor_input", {scratch},
            normParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN, 0, 0, 0),
            elementsGrid(uint64_t{nodeRows} * LILICORR_HIDDEN));
  addGemmF32(graph, scratch, scorer.factorInput, scorer.factorInputBias,
             gemmParams(nodeRows, LILICORR_NODES, 3 * LILICORR_HIDDEN,
                        LILICORR_HIDDEN, LILICORR_SCRATCH_FACTOR_INPUT,
                        LILICORR_SCRATCH_FACTORS, 1));
  addGemmF32(graph, scratch, scorer.outHead, scorer.outHeadBias,
             gemmParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                        LILICORR_HIDDEN, LILICORR_SCRATCH_FACTORS,
                        LILICORR_SCRATCH_OUT));
  addGemmF32(graph, scratch, scorer.inHead, scorer.inHeadBias,
             gemmParams(nodeRows, LILICORR_NODES, LILICORR_HIDDEN,
                        LILICORR_HIDDEN, LILICORR_SCRATCH_FACTORS,
                        LILICORR_SCRATCH_IN));
  addGemmF32(graph, scratch, scorer.anchorOutHead, scorer.anchorOutHeadBias,
             gemmParams(lanes, 1, LILICORR_HIDDEN, LILICORR_HIDDEN,
                        LILICORR_SCRATCH_ANCHOR_NORMED,
                        LILICORR_SCRATCH_ANCHOR_OUT));
  addL2Norm(graph, scratch, nodeRows, LILICORR_NODES,
            LILICORR_SCRATCH_OUT, scorer.vectorEpsilon);
  addL2Norm(graph, scratch, nodeRows, LILICORR_NODES,
            LILICORR_SCRATCH_IN, scorer.vectorEpsilon);
  addL2Norm(graph, scratch, lanes, 1,
            LILICORR_SCRATCH_ANCHOR_OUT, scorer.vectorEpsilon);
  graph.add("lilicorr_walk",
            {scratch, buffers.proposedTokens, buffers.confidence},
            LiLiCorrWalkParams{lanes, LILICORR_SCRATCH_STRIDE,
                               scorer.vectorEpsilon, scorer.logitScale},
            {lanes, 1, 1}, {1, 1, 1});
}

} // namespace

DFlashDraft::DFlashDraft(const DFlashDraftWeights &weights,
                         metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators)
    : weights_(weights), backend_(backend), operators_(operators),
      selector_(weights.layout.vocabularySize,
                paddedVocabularySize(weights.layout.vocabularySize)),
      poolSelector_(tuning().dflashPoolSet
                        ? tuning().dflashPool
                        : (modelPolicy().dflashPool ||
                           weights.layout.poolSelector)),
      contextKvProjections_(contextKvRows(backend, weights)) {}

void DFlashDraft::addSelection(
    metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies, uint32_t treeMask) const {
  if (weights_.layout.kind == DraftKind::LiLiCorr) {
    if (treeMask)
      selector_.addPlainTree(graph, buffers, anchors, policies, treeMask,
                             weights_.layout.queryRows(), 1);
    else
      selector_.addPlain(graph, buffers, anchors, policies,
                         weights_.layout.queryRows(), 1);
    const bool sampled =
        std::any_of(policies.begin(), policies.end(),
                    [](const ops::SamplingPolicy &policy) {
                      return policy.samples();
                    });
    if (!treeMask && !sampled && buffers.lilicorr &&
        weights_.lilicorr.present) {
      addLiLiCorrScorer(graph, buffers, weights_.lilicorr,
                        static_cast<uint32_t>(anchors.size()),
                        weights_.layout.queryRows(),
                        weights_.layout.vocabularySize,
                        paddedVocabularySize(weights_.layout.vocabularySize));
    }
    return;
  }
  // The pool walk is greedy-only: a sampled lane's residual draw stages
  // kDraftCandidates entries, so any sampling policy falls the whole batch
  // back to the merged-16 path.
  const bool pool =
      poolSelector_ &&
      std::none_of(policies.begin(), policies.end(),
                   [](const ops::SamplingPolicy &policy) {
                     return policy.samples();
                   });
  if (pool) {
    const ops::DraftCodebooks codebooks{weights_.predecessorCodebook,
                                        weights_.successorCodebook};
    if (treeMask) {
      selector_.addPoolTree(graph, buffers, codebooks, anchors, policies,
                            treeMask, weights_.layout.queryRows());
      return;
    }
    selector_.addPool(graph, buffers, codebooks, anchors, policies,
                      weights_.layout.queryRows());
    return;
  }
  if (treeMask) {
    selector_.addTree(graph, buffers,
                      {weights_.predecessorCodebook,
                       weights_.successorCodebook},
                      anchors, policies, treeMask, weights_.layout.queryRows());
    return;
  }
  selector_.add(graph, buffers,
                {weights_.predecessorCodebook, weights_.successorCodebook},
                anchors, policies, weights_.layout.queryRows());
}

void DFlashDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers, uint32_t rows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid draft context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  for (const DFlashPrefillSpan &span : spans) {
    if (span.ring.size() != layout.layers)
      throw std::invalid_argument("draft prefill ring layer mismatch");
  }
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

  if (layout.kind == DraftKind::LiLiCorr) {
    for (const DFlashPrefillSpan &span : spans) {
      if (span.anchorRow == ~0u || !span.anchor)
        continue;
      const metal::MetalBuffer row = backend_.view(
          buffers.hidden,
          uint64_t{span.anchorRow} * layout.hiddenSize * sizeof(uint16_t),
          uint64_t{layout.hiddenSize} * sizeof(uint16_t));
      graph.add("mtp_anchor_prefill", {row, span.anchor},
                MtpAnchorParams{layout.hiddenSize, 0, 1},
                {1, 1, 1}, {256, 1, 1});
    }
  }

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

void DFlashDraft::addDecode(
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

  // Every draft decode dispatch is static per batch width except the
  // attention split/reduce, whose parameters carry each lane's cache length;
  // the pair's payloads are patchable, so the layers and head replay as one
  // baked indirect command buffer span.
  graph.beginBakedSpan();
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const uint32_t current = layer & 1;
    const uint32_t next = current ^ 1;
    const DFlashDraftLayerWeights &weights = weights_.layers[layer];
    const ops::LinearPlan attentionDynamicPlan = linear.decodePlan(weights.attentionDynamic, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
    const ops::PreparedInput attentionNormalized = ops::Normalization::addRms(
        graph, buffers.hidden[current], weights.inputNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, attentionDynamicPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.dynamic, .scratch = scratch,
                .prepared = attentionNormalized},
               weights.attentionDynamic, attentionDynamicPlan);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    linear.add(graph, {.input = buffers.convolved, .output = buffers.proposalQkv, .scratch = scratch},
               weights.qkvProjection, linear.decodePlan(weights.qkvProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows()));
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
         buffers.queryValues, weights.attentionSinks},
        cacheLengths, attentionPlan, /*causal=*/false, devicePack);
    linear.add(graph, {.input = buffers.attention, .output = buffers.projected, .scratch = scratch},
               weights.outputProjection, linear.decodePlan(weights.outputProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows()));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.attentionConvolution,
         buffers.hidden[current], buffers.residual},
        attentionPlan, ops::DraftConvolutionStage::Residual);
    const ops::LinearPlan mlpDynamicPlan = linear.decodePlan(weights.mlpDynamic, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
    const ops::PreparedInput mlpNormalized = ops::Normalization::addRms(
        graph, buffers.residual, weights.postAttentionNorm, buffers.normalized,
        layout.hiddenSize, rows, scratch, mlpDynamicPlan.input());
    linear.add(graph,
               {.input = buffers.normalized, .output = buffers.dynamic, .scratch = scratch,
                .prepared = mlpNormalized},
               weights.mlpDynamic, mlpDynamicPlan);
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.normalized, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.convolved},
        attentionPlan, ops::DraftConvolutionStage::Prepare);
    linear.add(graph,
               {.input = buffers.convolved, .output = buffers.intermediate, .gateScratch = buffers.gateScratch,
                .scratch = scratch},
               weights.upProjection,
               linear.decodePlan(weights.upProjection, lanes, ops::LinearEpilogue::GateUp, &weights.gateProjection, layout.queryRows()),
               &weights.gateProjection);
    linear.add(graph, {.input = buffers.intermediate, .output = buffers.projected, .scratch = scratch},
               weights.downProjection, linear.decodePlan(weights.downProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows()));
    ops::DraftAttention::addConvolution(
        graph,
        {buffers.projected, buffers.dynamic, weights.mlpConvolution,
         buffers.residual, buffers.hidden[next]},
        attentionPlan, ops::DraftConvolutionStage::Residual);
  }

  // The final norm feeds the shared vocabulary head and then the selector;
  // the selector reuses whatever table the head leaves when the layouts match.
  const ops::LinearPlan headPlan = linear.decodePlan(vocabularyProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows());
  std::optional<ops::LinearPlan> selectorPlan;
  bool sharedTable = false;
  if (layout.kind != DraftKind::LiLiCorr) {
    selectorPlan.emplace(linear.decodePlan(weights_.selectorProjection, lanes, ops::LinearEpilogue::None, nullptr, layout.queryRows()));
    // The selector reuses the head's table only when the layouts match; a
    // mismatch makes it re-prepare from the bf16 rows, which must then exist.
    sharedTable = headPlan.input() != ops::LinearInput::Plain &&
                  headPlan.input() == selectorPlan->input();
  }
  const ops::PreparedInput finalHidden = ops::Normalization::addRms(
      graph, buffers.hidden[weights_.layout.layers & 1], weights_.finalNorm,
      buffers.finalHidden, layout.hiddenSize, rows, scratch, headPlan.input(), !sharedTable);
  const ops::PreparedInput afterHead = linear.add(
      graph, {.input = buffers.finalHidden, .output = buffers.logits, .scratch = scratch, .prepared = finalHidden},
      vocabularyProjection, headPlan);
  if (layout.kind != DraftKind::LiLiCorr) {
    linear.add(graph,
               {.input = buffers.finalHidden, .output = buffers.selectorHidden, .scratch = scratch,
                .prepared = afterHead},
               weights_.selectorProjection, *selectorPlan);
  }
  graph.endBakedSpan();
}

void DFlashDraft::addContextCommit(
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
  // prepared for the first layer's plan. The bf16 rows are dead only when
  // every layer's plan takes the same table layout — a mismatch makes a
  // later layer re-prepare from them.
  std::vector<ops::LinearPlan> kvPlans;
  kvPlans.reserve(layout.layers);
  ops::LinearInput kvInput = ops::LinearInput::Plain;
  bool kvUniform = true;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    ops::LinearPlan plan = linear.decodePlan(contextKvProjections_[layer], lanes,
                                             ops::LinearEpilogue::None, nullptr, layout.verifyRows());
    if (layer == 0) kvInput = plan.input();
    else if (plan.input() != kvInput) kvUniform = false;
    kvPlans.push_back(std::move(plan));
  }
  const bool kvRowsDead = kvUniform && kvInput != ops::LinearInput::Plain;
  ops::PreparedInput hidden = ops::Normalization::addRms(
      graph, buffers.projected, weights_.hiddenNorm, buffers.hidden, layout.hiddenSize, rows,
      scratch, kvInput, !kvRowsDead || layout.kind == DraftKind::LiLiCorr);

  if (layout.kind == DraftKind::LiLiCorr) {
    const auto anchor = [&](uint32_t lane) {
      return buffers.mtpAnchors[std::min(lane, lanes - 1)];
    };
    graph.add("mtp_anchor_commit",
              {buffers.hidden, buffers.retainedCounts,
               anchor(0), anchor(1), anchor(2), anchor(3)},
              MtpAnchorParams{layout.hiddenSize, layout.verifyRows(), lanes},
              {lanes, 1, 1}, {256, 1, 1});
  }

  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const ops::Projection &projection = contextKvProjections_[layer];
    hidden = linear.add(graph,
                        {.input = buffers.hidden, .output = buffers.contextKv, .scratch = scratch,
                         .prepared = hidden},
                        projection, kvPlans[layer]);
    ops::DraftAttention::addContextCommit(
        graph, buffers.contextKv, weights_.layers[layer].keyNorm, buffers.ropeCos,
        buffers.ropeSin, buffers.persistentKeys[layer],
        buffers.persistentValues[layer], buffers.retainedCounts,
        startPositions, layout.attentionShape(), layout.verifyRows());
  }
}

namespace {

// Reads a draft's files in their section order: each layer, then model.bin.
template <class Files>
DFlashDraftWeights readDraft(metal::MetalBackend &backend, Files &files,
                             const DFlashDraftLayout &layout) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DFlashDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t convolutionBytes = checkedMultiply<WeightStoreError>(
      checkedMultiply<WeightStoreError>(4, layout.hiddenSize,
                                        "draft convolution elements"),
      kBFloat16Bytes, "draft convolution bytes");
  const uint64_t headNormBytes = checkedMultiply<WeightStoreError>(
      layout.attentionHeadDimension, kBFloat16Bytes,
      "draft head norm bytes");

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = files.layer(layerIndex);
    DFlashDraftLayerWeights layer;
    layer.inputNorm = readNorm(file, layout.hiddenSize, false, "input-norm");
    layer.attentionConvolution =
        file.section(convolutionBytes, "attention-convolution");
    // The dynamic width is hiddenSize / 4 — 640 at 2560, off the affine
    // 256-row tile — so both paths carry these two as block projections:
    // packed files as F32 GGUF segments (validateQ4Layout would reject
    // 640 rows), checkpoints as AF4 segments.
    layer.attentionDynamic = readBlockProjection(
        file, layout.dynamicSize, layout.hiddenSize,
        "attention-dynamic");
    layer.qkvProjection = readProjection(
        file, layout.qkvSize, layout.hiddenSize, "qkv");
    layer.queryNorm = file.section(headNormBytes, "query-norm");
    layer.keyNorm = file.section(headNormBytes, "key-norm");
    layer.outputProjection = readProjection(
        file, layout.hiddenSize, layout.attentionSize,
        "attention-output");
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, false, "post-attention-norm");
    layer.mlpConvolution = file.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readBlockProjection(
        file, layout.dynamicSize, layout.hiddenSize, "mlp-dynamic");
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
    result.selectorProjection = readProjection(
        file, layout.selectorRank, layout.hiddenSize, "selector");
    const uint64_t codebookBytes = checkedMultiply<WeightStoreError>(
        checkedMultiply<WeightStoreError>(layout.vocabularySize,
                                          layout.selectorRank,
                                          "draft codebook elements"),
        kBFloat16Bytes, "draft codebook bytes");
    result.predecessorCodebook =
        file.section(codebookBytes, "predecessor-codebook");
    result.successorCodebook =
        file.section(codebookBytes, "successor-codebook");
    file.finish();
    result.files.push_back(file.record());
  }

  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace

WeightFile PackedDraftFiles::layer(uint32_t index) const {
  const std::string filename = "layer-" + std::to_string(index) + ".bin";
  return images.load(packedImage(directory / filename, "draft/" + filename,
                                 kDFlashLayerMagic, index, 0));
}

WeightFile PackedDraftFiles::model() const {
  return images.load(packedImage(directory / "model.bin", "draft/model.bin",
                                 kDFlashLayerMagic, layout.layers, 1));
}

DFlashDraftWeights loadDFlashDraftWeights(metal::MetalBackend &backend,
                                          const DraftFiles &files,
                                          DFlashDraftLayout layout) {
  requireLayout(layout);
  if (const auto *checkpoint =
          std::get_if<std::reference_wrapper<DFlash2CheckpointLoader>>(&files))
    return readDraft(backend, checkpoint->get(), layout);
  return readDraft(backend, std::get<PackedDraftFiles>(files), layout);
}

} // namespace richengine::model

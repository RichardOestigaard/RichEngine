// The recurrent-state mixer encoders of the hybrid targets: the Qwen
// families' fused GDN block and LFM2's double-gated short convolution, and
// the composite-state commits the verify step's bx rows feed. The layers'
// FFN encodes live with the rest in TargetModel.cpp; the step records and
// the row/sum view helpers the target translation units share live in
// TargetModelGemmaHelpers.hpp.
#include "model/TargetModelGemmaHelpers.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "Tuning.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/QuantFormat.h"
#include "ops/AneFfn.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace richengine::model {

namespace {

// RICHENGINE_GDN_ZSPLIT: the packed GDN input projection's columns carve into
// three contiguous runs — the ConvDim conv/qkv inputs, the
// ValueHeads*HeadDim z-gate columns (contiguous: gdn.metal's gate reads
// packed[ConvDim + head*HeadDim + dim]) and the beta/decay tail — each a
// whole run of QUANT_TILE_ROWS plane tiles, so Projection::rowRange views
// them and three decode plans reproduce the fused projection's columns
// bitwise (same tile, splits and simdgroups; K order unchanged).
struct GdnZSplitDims final {
  uint32_t conv = 0;
  uint32_t z = 0;
  uint32_t tail = 0;
};

[[nodiscard]] std::optional<GdnZSplitDims> gdnZSplitDims(
    const TargetModelGeometry &geometry, const ops::Projection &packed) {
  // Affine Q4 planes only: a GGUF's segmented projections and its gpu-span
  // decode ordering stay on the fused path, as does any non-bf16 destination.
  if (packed.layout() != ops::WeightLayout::Affine64 ||
      !packed.takesPlaneViews() ||
      packed.destination != ops::FloatOutput::BFloat16 ||
      packed.outputSize != geometry.packedGdnWidth)
    return std::nullopt;
  const ops::GdnShape shape = geometry.gdnShape();
  const uint32_t conv = shape.convolutionDimension;
  const uint32_t z = shape.valueHeads * shape.headDimension;
  const uint32_t tail = shape.packedWidth - conv - z;
  if (!tail || conv % QUANT_TILE_ROWS || z % QUANT_TILE_ROWS ||
      tail % QUANT_TILE_ROWS)
    return std::nullopt;
  return GdnZSplitDims{conv, z, tail};
}

// The column tiles of the affine decode tiles a split plan can inherit from
// the full projection's configuration (the GGUF tiles never reach here).
[[nodiscard]] constexpr uint32_t gdnZSplitTileColumns(ops::LinearTile tile) noexcept {
  switch (tile) {
  case ops::LinearTile::Simdgroup: return 64;
  case ops::LinearTile::N256:
  case ops::LinearTile::Paired256: return 256;
  default: return 128;
  }
}

struct GdnZSplit final {
  GdnZSplitDims dims;
  ops::Projection conv, z, tail;
  ops::LinearPlan convPlan, zPlan, tailPlan;
};

// The carved projections and their plans on the full projection's decode
// configuration — identical tile, splits and simdgroups keep every output
// column's accumulation order, hence bitwise — or nullopt when the
// projection or plan cannot take the split. splits must be one: the carved
// GEMVs share LinearScratch's split partials and counters otherwise.
[[nodiscard]] std::optional<GdnZSplit> gdnZSplitPlans(
    const metal::MetalBackend &backend,
    const TargetModelGeometry &geometry, const ops::Projection &packed,
    const ops::LinearPlan &inputPlan) {
  if (!tuning().gdnZSplit)
    return std::nullopt;
  const std::optional<GdnZSplitDims> dims = gdnZSplitDims(geometry, packed);
  if (!dims || inputPlan.configuration().splits != 1)
    return std::nullopt;
  try {
    const ops::LinearConfig config = inputPlan.configuration();
    const ops::LinearWorkload full = inputPlan.workload();
    const auto plan = [&](const ops::Projection &p) {
      ops::LinearConfig c = config;
      // Persistent grids size to the full projection's tiles; clamp to the
      // carved projection's (a scheduling bound only, not the math).
      c.groups = std::min(c.groups, p.outputSize / gdnZSplitTileColumns(c.tile));
      return ops::Linear::plan(
          {{p.outputSize, p.inputSize}, full.rows, ops::LinearPhase::Decode,
           ops::LinearEpilogue::None, ops::WeightLayout::Affine64, false,
           full.lanes},
          c, inputPlan.destination());
    };
    const ops::Projection conv = packed.rowRange(backend, 0, dims->conv);
    const ops::Projection z = packed.rowRange(backend, dims->conv, dims->z);
    const ops::Projection tail =
        packed.rowRange(backend, dims->conv + dims->z, dims->tail);
    auto convPlan = plan(conv);
    auto zPlan = plan(z);
    auto tailPlan = plan(tail);
    return GdnZSplit{*dims, std::move(conv), std::move(z), std::move(tail),
                     std::move(convPlan), std::move(zPlan),
                     std::move(tailPlan)};
  } catch (...) {
    return std::nullopt;
  }
}

// Whether a chain commit of `lanes` lanes at `verifyStride` follows
// z-split verifies: every GDN layer's input projection must take the split
// (they share one shape, so the first layer's plans speak for all) — the
// commit then reads the carved conv rows through verify_gdn_commit_split.
[[nodiscard]] bool gdnZSplitCommitted(
    const TargetModelWeights &weights, const TargetModelGeometry &geometry,
    const ops::Linear &linear, const metal::MetalBackend &backend,
    uint32_t lanes, uint32_t verifyStride) {
  if (!tuning().gdnZSplit || !geometry.gdnLayers)
    return false;
  const ops::Projection *first = nullptr;
  for (const TargetLayerWeights &layer : weights.layers) {
    const auto *gdn = std::get_if<GdnMixerWeights>(&layer.mixer);
    if (!gdn)
      continue;
    if (!gdnZSplitDims(geometry, gdn->inputProjection))
      return false;
    first = first ? first : &gdn->inputProjection;
  }
  if (!first)
    return false;
  const ops::LinearPlan inputPlan = linear.decodePlan(
      *first, lanes, ops::LinearEpilogue::None, nullptr, verifyStride);
  return gdnZSplitPlans(backend, geometry, *first, inputPlan)
      .has_value();
}

} // namespace


metal::MetalBuffer TargetModel::addPrefillMixer(PrefillStep &step, const GdnMixerWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (const TargetModelPrefillSequence &sequence : step.sequences) {
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

// LFM2's conv layer: input norm, the [B|C|x] in_proj, the double-gated
// causal convolution from the layer's FIFO state into the out-projection's
// input, then the residual output projection.
metal::MetalBuffer TargetModel::addPrefillMixer(PrefillStep &step, const LfmConvWeights &mixer,
                                               const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addPrefillNorm(step, input, norm, mixer.inputProjection.layout());
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, b.projectionSums,
                                 step.rows, b.linearScratch);
  for (const TargetModelPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const metal::MetalBuffer convRows = u16(b.gdnHidden, geometry_.convolutionDimension);
    ops::LfmConv::addPrefill(step.graph,
                             {u16(b.gdnPacked, geometry_.packedGdnWidth),
                              mixer.convolutionWeights, sequence.convolutionIn[layer],
                              sequence.convolutionOut[layer], convRows},
                             geometry_.convShape(), sequence.rows, mixer.convolutionTapsMajor);
    if (mixer.outputProjection.layout() == ops::WeightLayout::Affine64) {
      const uint32_t paddedRows = (sequence.rows + 31) / 32 * 32;
      operators_.linear().addPrefillSums(
          step.graph,
          rowsOf<uint16_t>(backend_, b.gdnHidden, sequence.rowBegin, paddedRows,
                           geometry_.convolutionDimension),
          prefillSums(backend_, b.projectionSums, sequence.rowBegin, paddedRows,
                      geometry_.convolutionDimension),
          mixer.outputProjection, sequence.rows);
    }
  }
  addPrefillOutput(step, b.gdnHidden, mixer.outputProjection, input, b.gdnOutput);
  return b.gdnOutput;
}

// Each producer emits the table (if any) its consumer's plan reads.
metal::MetalBuffer TargetModel::addVerifyMixer(VerifyStep &step, const GdnMixerWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.planLanes, ops::LinearEpilogue::None, nullptr, step.rowCapacity);
  // Norm, input projection, fused GDN and residual output projection replay
  // from a baked indirect command buffer: every parameter is per-layer
  // geometry and the bound buffers are the decode arena's (the GDN state
  // pair's current/next swap alternates between the span cache's two slots).
  step.graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input(), false);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.planLanes, ops::LinearEpilogue::Residual, nullptr, step.rowCapacity);
  const ops::GdnStateStrides state{
      geometry_.stateLayout.convolutionLayerBytes(),
      geometry_.stateLayout.recurrentLayerBytes(),
      geometry_.stateLayout.convolutionBytes()};
  ops::GdnDecodeBuffers gdn{b.gdnPacked[layer], mixer.convolutionWeights,
                            b.currentGdnStates, b.nextGdnStates,
                            b.gdnMixed[layer], mixer.decay, mixer.timeBias,
                            b.gdnDecay[layer], b.gdnBeta[layer],
                            mixer.mixerNorm, b.gdnHidden, b.linearScratch,
                            b.gdnTape};
  // RICHENGINE_GDN_ZSPLIT (chain verifies only): the input projection's
  // columns carve into conv, z and tail GEMVs — the z dispatch sits between
  // the recurrent scan and the gate apply so it overlaps the scan where
  // RICHENGINE_SPAN_BARRIERS lets siblings run un-barriered. The packed
  // buffer's carved regions are disjoint byte ranges of the layer's rows.
  if (!step.tree) {
    if (std::optional<GdnZSplit> split =
            gdnZSplitPlans(backend_, geometry_,
                           mixer.inputProjection, inputPlan)) {
      const uint64_t rows = inputPlan.workload().rows;
      const auto carve = [&](uint64_t first, uint64_t count) {
        return backend_.view(b.gdnPacked[layer], first * 2, count * 2);
      };
      const metal::MetalBuffer convView =
          carve(0, rows * split->dims.conv);
      const metal::MetalBuffer zView =
          carve(rows * split->dims.conv, rows * split->dims.z);
      const metal::MetalBuffer tailView =
          carve(rows * (split->dims.conv + split->dims.z),
                rows * split->dims.tail);
      gdn.packed = convView;
      ops::PreparedInput prepared = normalized;
      prepared = linear.add(
          step.graph,
          {.input = b.normalized, .output = convView,
           .scratch = b.linearScratch, .prepared = prepared},
          split->conv, split->convPlan);
      prepared = linear.add(
          step.graph,
          {.input = b.normalized, .output = tailView,
           .scratch = b.linearScratch, .prepared = prepared},
          split->tail, split->tailPlan);
      ops::GDN::addDecodeScan(
          step.graph, gdn, tailView, geometry_.gdnShape(), step.lanes, layer,
          state, mixer.outputHeadOrder, step.liveRows, step.rowCapacity,
          step.tapeLanes);
      prepared = linear.add(
          step.graph,
          {.input = b.normalized, .output = zView,
           .scratch = b.linearScratch, .prepared = prepared},
          split->z, split->zPlan);
      const ops::PreparedInput hidden = ops::GDN::addDecodeGate(
          step.graph, zView, gdn, geometry_.gdnShape(), step.lanes,
          step.rowCapacity, mixer.outputHeadOrder, outputPlan.input());
      linear.add(step.graph,
                 {.input = b.gdnHidden, .output = b.gdnOutput,
                  .residual = input, .scratch = b.linearScratch,
                  .prepared = hidden},
                 mixer.outputProjection, outputPlan);
      step.graph.endBakedSpan();
      return b.gdnOutput;
    }
  }
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked[layer], .scratch = b.linearScratch,
              .prepared = normalized},
             mixer.inputProjection, inputPlan);
  const ops::PreparedInput hidden =
      step.tree
          ? ops::GDN::addDecodeTree(
                step.graph, gdn, b.treeNodes, b.treeCounts,
                geometry_.gdnShape(), step.lanes, layer, state,
                mixer.outputHeadOrder, outputPlan.input())
          : ops::GDN::addDecode(
                step.graph, gdn, geometry_.gdnShape(), step.lanes, layer,
                state, mixer.outputHeadOrder, outputPlan.input(),
                step.liveRows, step.rowCapacity, step.tapeLanes);
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
metal::MetalBuffer TargetModel::addVerifyMixer(VerifyStep &step, const LfmConvWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.planLanes, ops::LinearEpilogue::None, nullptr, step.rowCapacity);
  step.graph.beginBakedSpan();
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input(), false);
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked[layer], .scratch = b.linearScratch,
              .prepared = normalized},
             mixer.inputProjection, inputPlan);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.planLanes, ops::LinearEpilogue::Residual, nullptr, step.rowCapacity);
  // The conv reads every lane's current state and writes its bx rows for the
  // commit; its output is the out-projection's input. A tree step's taps
  // follow the nodes' ancestor paths and commit through the retained path.
  ops::PreparedInput hidden;
  if (step.tree) {
    ops::LfmConv::addVerifyTree(
        step.graph,
        {b.gdnPacked[layer], mixer.convolutionWeights, b.currentGdnStates,
         b.gdnMixed[layer], b.gdnHidden, b.treeNodes, b.treeCounts},
        geometry_.convShape(), step.lanes, step.rowCapacity,
        layer, geometry_.stateLayout.convolutionLayerBytes(), mixer.convolutionTapsMajor);
  } else {
    // The conv may write the out-projection's operand table itself
    // (RICHENGINE_CONV_FUSE); `hidden` then describes it.
    hidden = ops::LfmConv::addVerify(
        step.graph,
        {b.gdnPacked[layer], mixer.convolutionWeights, b.currentGdnStates,
         b.gdnMixed[layer], b.gdnHidden, b.linearScratch},
        geometry_.convShape(), step.lanes, step.rowCapacity,
        layer, geometry_.stateLayout.convolutionLayerBytes(),
        outputPlan.input(), mixer.convolutionTapsMajor);
  }
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = b.gdnOutput, .residual = input, .scratch = b.linearScratch,
              .prepared = hidden},
             mixer.outputProjection, outputPlan);
  step.graph.endBakedSpan();
  return b.gdnOutput;
}

void TargetModel::addStateCommit(metal::CommandGraph &graph,
                                TargetModelCommitBuffers buffers,
                                uint32_t lanes, uint32_t verifyStride) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || !verifyStride)
    throw std::invalid_argument("invalid state commit batch");
  // The arch encoder's auxiliary commit (qwen4exp's PLE history and QSA
  // pending keys) reads the lane cells and retained counts the shared
  // commit below consumes; keep them for it.
  const TargetModelCommitBuffers commit = buffers;
  if (!geometry_.gdnLayers) {
    // LFM2's conv FIFOs: every conv layer rolls its lanes' states forward by
    // the retained rows' bx, kept in VerifyMixedBase like the GDN's. The
    // layer's slice of it is lane-major with the decode row count's stride.
    if (!geometry_.convLayers) return;
    const uint32_t laneStride = verifyStride *
                                geometry_.convolutionDimension * 2;
    const uint64_t layerBytes =
        uint64_t{ExecutionLimits::maximumBatchWidth} * laneStride;
    // One dispatch rolls every conv layer's lanes: the kernel derives each
    // layer's state slot and its `mixed` block from the layer's grid slice.
    ops::LfmConv::addCommit(
        graph,
        {buffers.mixed, buffers.retainedCounts, buffers.currentStates,
         buffers.nextStates},
        geometry_.convShape(), lanes, verifyStride,
        /*layer*/ 0, geometry_.stateLayout.convolutionLayerBytes(),
        geometry_.convLayers, layerBytes, /*tapsMajor*/ false);
    encoder_->commitState(graph, commit, lanes, verifyStride);
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
       geometry_.stateLayout.convolutionBytes()},
      verifyStride,
      gdnZSplitCommitted(*weights_, geometry_, operators_.linear(), backend_,
                         lanes, verifyStride));
  encoder_->commitState(graph, commit, lanes, verifyStride);
}

void TargetModel::addStateCommitTree(metal::CommandGraph &graph,
                                    TargetModelCommitBuffers buffers,
                                    metal::MetalBuffer retainedPath,
                                    uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || !retainedPath)
    throw std::invalid_argument("invalid tree state commit batch");
  if (!geometry_.gdnLayers) {
    // LFM2's conv FIFOs: one dispatch rolls every conv layer's lanes forward
    // by the retained path's bx rows — the tree counterpart of the chain
    // commit's retained prefix.
    if (!geometry_.convLayers) return;
    const uint32_t laneStride =
        uint32_t{RICHENGINE_TREE_VERIFY_NODES} *
        geometry_.convolutionDimension * 2;
    const uint64_t layerBytes =
        uint64_t{ExecutionLimits::maximumBatchWidth} * laneStride;
    ops::LfmConv::addCommitTree(
        graph,
        {buffers.mixed, buffers.retainedCounts, std::move(retainedPath),
         buffers.currentStates, buffers.nextStates},
        geometry_.convShape(), lanes,
        RICHENGINE_TREE_VERIFY_NODES, /*layer*/ 0,
        geometry_.stateLayout.convolutionLayerBytes(),
        geometry_.convLayers, layerBytes, /*tapsMajor*/ false);
    return;
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

} // namespace richengine::model

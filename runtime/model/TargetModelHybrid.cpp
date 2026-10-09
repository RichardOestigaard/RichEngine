// The recurrent-state mixer encoders of the hybrid targets: the Qwen
// families' fused GDN block and LFM2's double-gated short convolution, the
// hybrid families' per-layer FFNs (Qwen3.6-MoE's routed block, LFM2-MoE's
// dense/MoE variant), and the composite-state commits the verify step's bx
// rows feed. Split out of TargetModel.cpp; the step records and the row/sum
// view helpers the target translation units share live in
// TargetModelGemmaHelpers.hpp.
#include "model/TargetModelGemmaHelpers.hpp"

#include "model/Lfm2Moe.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "ops/AneFfn.hpp"

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace richengine::model {

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

void TargetModel::addPrefillFfn(PrefillStep &step, uint32_t, const Qwen3_6MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe, {}}, layer.ffn, *step.moe);
}

// LFM2-MoE's per-layer FFN: dense below denseLayers, the MoE block after.
void TargetModel::addPrefillFfn(PrefillStep &step, uint32_t index, const Lfm2MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const TargetModelPrefillBuffers &b = step.buffers;
  std::visit(
      [&](const auto &ffn) {
        using Ffn = std::decay_t<decltype(ffn)>;
        if constexpr (std::is_same_v<Ffn, Lfm2DenseFfn>) {
          addPrefillNorm(step, residual, layer.postAttentionNorm, ffn.gateProjection.layout());
          const ops::PrefillFfnBuffers buffers = b.ffn();
          if (step.aneFfn)
            step.aneFfn->add(step.graph, index, buffers, residual, output, step.rows);
          else
            operators_.linear().addPrefillSwiGlu(
                step.graph, {&ffn.gateProjection, &ffn.upProjection, &ffn.downProjection},
                buffers, residual, output, step.rows);
        } else {
          ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized,
                                     geometry_.hiddenSize, step.rows);
          ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe, {}}, ffn, *step.moe);
        }
      },
      layer.ffn);
}

// Each producer emits the table (if any) its consumer's plan reads.
metal::MetalBuffer TargetModel::addVerifyMixer(VerifyStep &step, const GdnMixerWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelVerifyBuffers &b = step.buffers;
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
metal::MetalBuffer TargetModel::addVerifyMixer(VerifyStep &step, const LfmConvWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const TargetModelVerifyBuffers &b = step.buffers;
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

void TargetModel::addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const TargetModelVerifyBuffers &b = step.buffers;
  // One layer's norm plus MoE block is replayed from a baked indirect command
  // buffer when the submission revalidates it; MoE::add nests its own span.
  step.graph.beginBakedSpan();
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe, {}}, layer.ffn, *step.moe);
  step.graph.endBakedSpan();
}

// LFM2-MoE's per-layer FFN in a verify step: dense below denseLayers, the
// MoE block after.
void TargetModel::addVerifyFfn(VerifyStep &step, const Lfm2MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const TargetModelVerifyBuffers &b = step.buffers;
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
          ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe, {}}, ffn, *step.moe);
          step.graph.endBakedSpan();
        }
      },
      layer.ffn);
}

void TargetModel::addStateCommit(metal::CommandGraph &graph,
                                TargetModelCommitBuffers buffers,
                                uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid state commit batch");
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

void TargetModel::addStateCommitTree(metal::CommandGraph &graph,
                                    TargetModelCommitBuffers buffers,
                                    metal::MetalBuffer retainedPath,
                                    uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || !retainedPath)
    throw std::invalid_argument("invalid tree state commit batch");
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

} // namespace richengine::model

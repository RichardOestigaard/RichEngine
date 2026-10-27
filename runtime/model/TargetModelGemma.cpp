// The Gemma 4 MoE target's layer encoder: the dual-geometry attention
// mixer, the post-attention residual both the router and the FFN-normed
// experts read, and the routed plus shared-expert GeGLU block whose normed
// outputs join the residual before layer_scalar — plus the per-pass state
// (alternate-geometry plans, canvas flag, encoder scalars) its step
// extension carries. DiffusionGemma runs the same encoder; its trunk's
// canvas and encoder-scalar state is the extension. Split out of
// TargetModel.cpp; the step records and the row/sum view helpers the
// target translation units share live in TargetModelGemmaHelpers.hpp.
#include "model/TargetModelGemmaHelpers.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "Tuning.hpp"
#include "model/ModelPolicy.hpp"

#include <any>
#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <variant>

namespace richengine::model {

namespace {

// The Gemma layer encoder's per-pass prefill state, carried in
// PrefillStep::extension: the same sequences' plans in the alternate
// (global) geometry, the caller's per-layer scalar overrides and the
// canvas flag of a DiffusionGemma trunk pass — whose plan selection takes
// the canvas family (MoePhase::Canvas, canvas-marked linear workloads).
struct GemmaPrefillExtension {
  std::vector<ops::PrefillAttentionPlan> altAttention{};
  std::span<const float> layerScalars{};
  bool canvas = false;
};

// The same for a verify step: the alternate-geometry plan; empty on a
// single-geometry pass.
struct GemmaVerifyExtension {
  std::optional<ops::VerifyAttentionPlan> altAttention{};
};

} // namespace

// The gemma4/diffusion_gemma layer encoder. Its member references alias
// the model's internals so the moved layer bodies read as they did as
// TargetModel methods.
class GemmaLayerEncoder final : public TargetLayerEncoder {
public:
  explicit GemmaLayerEncoder(const TargetModel &model)
      : model_(model), weights_(*model.weights_), geometry_(model.geometry_),
        backend_(model.backend_), operators_(model.operators_) {}

  void beginPrefill(TargetModel::PrefillStep &step) const override {
    GemmaPrefillExtension extension;
    if (const auto *inputs =
            std::any_cast<TargetPrefillInputs>(&step.extension))
      extension.layerScalars = inputs->layerScalars;
    extension.canvas =
        !step.sequences.empty() && step.sequences.front().canvas;
    if (geometry_.altAttentionMask)
      for (const TargetModelPrefillSequence &sequence : step.sequences)
        extension.altAttention.push_back(
            sequence.canvas
                ? ops::PagedAttention::prefillCanvasPlan(
                      sequence.rows, geometry_.attentionQueryHeads,
                      geometry_.layerKvLayout(true))
                : operators_.prefillAttention(
                      sequence.rows, geometry_.attentionQueryHeads,
                      geometry_.layerKvLayout(true)));
    step.extension = std::move(extension);
  }

  void beginVerify(TargetModel::VerifyStep &step) const override {
    GemmaVerifyExtension extension;
    if (geometry_.altAttentionMask) {
      std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
      for (uint32_t lane = 0; lane < step.lanes; ++lane)
        histories[lane] = step.chunks[lane].committed_tokens;
      extension.altAttention.emplace(operators_.verifyAttention(
          step.lanes, geometry_.attentionQueryHeads,
          geometry_.layerKvLayout(true),
          std::span(histories).first(step.lanes), step.tree, step.liveNodes,
          step.rowCapacity));
    }
    step.extension = std::move(extension);
  }

  // One Gemma 4 layer in a prefill chunk: attention (the sliding or the
  // k_eq_v global kernels by index), the post-attention residual the
  // router reads, and the routed experts beside the dense shared expert,
  // each post-normed before they join the residual under layer_scalar.
  metal::MetalBuffer addPrefillLayer(TargetModel::PrefillStep &step,
                                     uint32_t index, metal::MetalBuffer input,
                                     metal::MetalBuffer output) const override {
    // Timing probes (garbage output downstream, like RICHENGINE_GEGLU_OFF):
    // SKIP_ATTN drops the four paged-attention dispatches per sequence;
    // SKIP_MOE drops the routed block (route, group, gate/up/down, combine).
    static const bool skipAttentionEnv = tuning().gemmaSkipAttn;
    static const bool skipMoeEnv = tuning().gemmaSkipMoe;
    const TargetModelPrefillBuffers &b = step.buffers;
    const GemmaPrefillExtension &ext =
        std::any_cast<const GemmaPrefillExtension &>(step.extension);
    const ops::Linear &linear = operators_.linear();
    const bool skipAttention = ext.canvas && skipAttentionEnv;
    const bool skipMoe = ext.canvas && skipMoeEnv;
    const uint32_t attentionLayer = step.attentionLayer++;
    const bool global = geometry_.isAltAttentionLayer(index);
    const kv::Layout layerKv = geometry_.layerKvLayout(global);
    const uint32_t headDim = layerKv.headDimension;
    const uint32_t kvHeads = layerKv.kvHeads;
    const uint32_t packedWidth = geometry_.layerPackedWidth(index);
    const uint32_t attentionWidth = geometry_.layerAttentionWidth(index);
    const uint32_t hidden = geometry_.hiddenSize;
    const TargetLayerWeights &layer = weights_.layers[index];
    const AttentionMixerWeights &mixer = std::get<AttentionMixerWeights>(layer.mixer);
    // The FFN half of the layer record: the extra norms, the router and
    // both expert blocks the shared TargetLayerWeights does not carry.
    const Gemma4FfnWeights &ffn = std::get<Gemma4FfnWeights>(layer.ffn);
    const metal::MetalBuffer ropeCos = global ? b.ropeCosAlt : b.ropeCos;
    const metal::MetalBuffer ropeSin = global ? b.ropeSinAlt : b.ropeSin;
    const uint32_t pairs = geometry_.rotaryPairsAt(index);

    model_.addPrefillNorm(step, input, layer.inputNorm, mixer.inputProjection.layout());
    linear.addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked,
                      b.projectionSums, step.rows, b.linearScratch, {},
                      ext.canvas);
    // The packed queries/K/V slabs lay out per sequence by this layer kind's
    // strides; the offsets accumulate here, per kind, not in
    // TargetModelPrefillSequence (which keeps the primary geometry's).
    uint64_t queryOffset = 0, kvOffset = 0;
    for (size_t s = 0; s < step.sequences.size(); ++s) {
      const TargetModelPrefillSequence &sequence = step.sequences[s];
      const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
      };
      const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
        return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
      };
      const uint64_t headBytes = uint64_t{sequence.attentionStride} * headDim * sizeof(uint16_t);
      const metal::MetalBuffer queries =
          backend_.view(b.fullQueries, queryOffset, geometry_.attentionQueryHeads * headBytes);
      const metal::MetalBuffer attentionRows =
          backend_.view(b.fullAttention, queryOffset, geometry_.attentionQueryHeads * headBytes);
      const metal::MetalBuffer keys =
          backend_.view(b.chunkKeys, kvOffset, kvHeads * headBytes);
      const metal::MetalBuffer values =
          backend_.view(b.chunkValues, kvOffset, kvHeads * headBytes);
      if (!skipAttention) {
      ops::PagedAttention::addPrefillProjection(
          step.graph, u16(b.fullPacked, packedWidth), mixer.queryNorm, mixer.keyNorm,
          f32(ropeCos, pairs), f32(ropeSin, pairs), queries, keys, values,
          sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads, layerKv);
      ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[attentionLayer], keys, values,
                                           sequence.pageTable, sequence.chunk, layerKv);
      ops::PagedAttention::addPrefill(
          step.graph, step.kvLayers[attentionLayer], queries, attentionRows,
          b.attentionPartials, b.attentionStatistics, sequence.pageTable,
          sequence.chunk,
          global ? ext.altAttention[s] : step.attention[s]);
      const metal::MetalBuffer hiddenRows = u16(b.attentionHidden, attentionWidth);
      // The no-gate gather; the out-projection's affine input sums take the
      // generic pass.
      ops::PagedAttention::addPrefillGate(
          step.graph, u16(b.fullPacked, packedWidth), attentionRows, hiddenRows,
          sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads,
          layerKv, false);
      }
      if (mixer.outputProjection.layout() == ops::WeightLayout::Affine64) {
        // The sums pass walks whole 32-row tiles: the input view must cover
        // the padded rows, not just this sequence's.
        const uint32_t paddedRows = (sequence.rows + 31) / 32 * 32;
        linear.addPrefillSums(
            step.graph,
            rowsOf<uint16_t>(backend_, b.attentionHidden, sequence.rowBegin,
                             paddedRows, attentionWidth),
            prefillSums(backend_, b.projectionSums, sequence.rowBegin, paddedRows, attentionWidth),
            mixer.outputProjection, sequence.rows);
      }
      queryOffset += geometry_.attentionQueryHeads * headBytes;
      kvOffset += kvHeads * headBytes;
    }
    // The output projection does not add the residual: the post-attention
    // norm sits between it and the residual add.
    linear.addPrefill(step.graph, b.attentionHidden, mixer.outputProjection,
                      b.attentionOutput, b.projectionSums, step.rows, b.linearScratch,
                      {}, ext.canvas);
    // R = input + postAttentionNorm(o(x)); the router and the FFN input
    // both read R.
    ops::Normalization::addRms(step.graph, b.attentionOutput, layer.postAttentionNorm,
                               b.normalized, hidden, step.rows);
    TargetModel::addResidualAdd(step.graph, input, b.normalized, b.attentionOutput, step.rows * hidden);
    // Router input: scale-free RMS norm times the learned per-dimension
    // scale (rms with weight = routerScale) times hidden^-0.5.
    ops::Normalization::addRms(step.graph, b.attentionOutput,
                               ops::NormWeights{ffn.routerScale}, b.gdnOutput,
                               hidden, step.rows);
    TargetModel::addLayerScalar(step.graph, b.gdnOutput,
                   1.0F / std::sqrt(static_cast<float>(hidden)),
                   step.rows * hidden);
    // The shared expert's input norm also emits its affine input sums; the
    // routed experts take the checkpoint's second pre-FFN norm.
    ops::Normalization::addRmsWithQ4Sums(step.graph, b.attentionOutput, ffn.preFfnNorm,
                                         b.normalized, b.projectionSums, hidden, step.rows);
    ops::Normalization::addRms(step.graph, b.attentionOutput, ffn.preFfnNormRouted,
                               b.gdnHidden, hidden, step.rows);
    // Routed experts: the raw routed sum lands in fullPacked, post-normed
    // into gdnOutput.
    if (!skipMoe)
      ops::MoE::addGemma(step.graph,
                       {b.gdnHidden, b.zeroResidual, b.fullPacked, b.moe, b.gdnOutput, {}},
                       {ffn.routerWeights, ffn.perExpertScale, ffn.expertGate,
                        ffn.expertUp, ffn.expertDown},
                       *step.moe);
    ops::Normalization::addRms(step.graph, b.fullPacked, ffn.postFfnNormRouted,
                               b.gdnOutput, hidden, step.rows);
    // The shared expert's dense GeGLU.
    linear.addPrefill(step.graph, b.normalized, ffn.sharedGate, b.denseGateScratch,
                      b.projectionSums, step.rows, b.linearScratch, {}, ext.canvas);
    linear.addPrefill(step.graph, b.normalized, ffn.sharedUp, b.denseIntermediate,
                      b.projectionSums, step.rows, b.linearScratch, {},
                      ext.canvas);
    const uint32_t intermediate = step.rows * geometry_.denseIntermediateSize;
    // RICHENGINE_GEGLU_OFF: debug gate that leaves the raw sharedUp output
    // in denseIntermediate, separating the GEMM from the geglu_multiply
    // pass.
    static const bool gegluOff = modelPolicy().gegluOff;
    if (!gegluOff)
      step.graph.add("geglu_multiply", {b.denseGateScratch, b.denseIntermediate, b.denseIntermediate},
                     intermediate, {(intermediate + 255) / 256, 1, 1}, {256, 1, 1});
    linear.addPrefillSums(step.graph, b.denseIntermediate,
                          b.projectionSums, ffn.sharedDown, step.rows);
    linear.addPrefill(step.graph, b.denseIntermediate, ffn.sharedDown,
                      b.attentionHidden, b.projectionSums, step.rows, b.linearScratch,
                      {}, ext.canvas);
    ops::Normalization::addRms(step.graph, b.attentionHidden, ffn.postFfnNormShared,
                               b.gdnHidden, hidden, step.rows);
    // The routed + shared sum takes the checkpoint's (un-suffixed) post-FFN
    // norm before it joins the residual.
    TargetModel::addResidualAdd(step.graph, b.gdnOutput, b.gdnHidden, b.attentionHidden,
                   step.rows * hidden);
    ops::Normalization::addRms(step.graph, b.attentionHidden, ffn.postFfnNorm,
                               b.gdnOutput, hidden, step.rows);
    TargetModel::addResidualAdd(step.graph, b.attentionOutput, b.gdnOutput, output,
                   step.rows * hidden);
    // The DiffusionGemma encoder pass overrides the packed (decoder)
    // layer_scalar with the encoder's.
    const float layerScalar = ext.layerScalars.empty()
                                  ? ffn.layerScalar
                                  : ext.layerScalars[index];
    TargetModel::addLayerScalar(step.graph, output, layerScalar, step.rows * hidden);
    return output;
  }

  // The same layer in a verify step. The attention dispatches suspend no
  // span: the store/split/reduce parameters are patchable, and everything
  // else is static per batch width — the whole layer replays from the ICB.
  void addVerifyLayer(TargetModel::VerifyStep &step, uint32_t index,
                      metal::MetalBuffer input,
                      metal::MetalBuffer output) const override {
    const TargetModelVerifyBuffers &b = step.buffers;
    const GemmaVerifyExtension &ext =
        std::any_cast<const GemmaVerifyExtension &>(step.extension);
    const ops::Linear &linear = operators_.linear();
    const uint32_t attentionLayer = step.attentionLayer++;
    const bool global = geometry_.isAltAttentionLayer(index);
    const kv::Layout layerKv = geometry_.layerKvLayout(global);
    const uint32_t hidden = geometry_.hiddenSize;
    const TargetLayerWeights &layer = weights_.layers[index];
    const AttentionMixerWeights &mixer = std::get<AttentionMixerWeights>(layer.mixer);
    const Gemma4FfnWeights &ffn = std::get<Gemma4FfnWeights>(layer.ffn);
    const metal::MetalBuffer ropeCos = global ? b.ropeCosAlt : b.ropeCos;
    const metal::MetalBuffer ropeSin = global ? b.ropeSinAlt : b.ropeSin;

    step.graph.beginBakedSpan();
    const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.planLanes);
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        step.graph, input, layer.inputNorm, b.normalized, hidden, step.rows,
        b.linearScratch, inputPlan.input());
    linear.add(step.graph,
               {.input = b.normalized, .output = b.fullPacked, .scratch = b.linearScratch,
                .prepared = normalized},
               mixer.inputProjection, inputPlan);
    ops::PagedAttention::addVerifyProjection(
        step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, ropeCos, ropeSin,
        b.fullQueries, b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer],
        geometry_.attentionQueryHeads, layerKv, step.lanes, step.rowCapacity);
    const ops::LinearPlan outputPlan = linear.decodePlan(mixer.outputProjection, step.planLanes);
    static const bool noFusedGate = modelPolicy().noFusedGate;
    const ops::VerifyAttentionPlan &plan =
        global ? *ext.altAttention : step.attention;
    const bool fusedReduce = !noFusedGate && !step.tree &&
                             outputPlan.input() == ops::LinearInput::Plain;
    if (fusedReduce) {
      // verify_attention_reduce_gather_gemma_*: the hidden layout directly.
      ops::PagedAttention::addVerify(step.graph, step.kvLayers[attentionLayer],
                                     {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer],
                                      b.fullQueries, b.attentionPartials, b.attentionStatistics,
                                      b.fullAttention, b.pageTables, b.treeMasks},
                                     step.chunks, plan, {}, b.attentionHidden);
    } else {
      ops::PagedAttention::addVerify(step.graph, step.kvLayers[attentionLayer],
                                     {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer],
                                      b.fullQueries, b.attentionPartials, b.attentionStatistics,
                                      b.fullAttention, b.pageTables, b.treeMasks},
                                     step.chunks, plan);
      ops::PagedAttention::addVerifyGate(
          step.graph, b.fullPacked, b.fullAttention, b.attentionHidden,
          geometry_.attentionQueryHeads, layerKv, step.lanes, b.linearScratch,
          outputPlan.input(), false, step.rowCapacity);
    }
    linear.add(step.graph,
               {.input = b.attentionHidden, .output = b.attentionOutput,
                .scratch = b.linearScratch},
               mixer.outputProjection, outputPlan);
    // R = input + postAttentionNorm(o(x)) in gemmaResidual.
    ops::Normalization::addRms(step.graph, b.attentionOutput, layer.postAttentionNorm,
                               b.normalized, hidden, step.rows);
    TargetModel::addResidualAdd(step.graph, input, b.normalized, b.gemmaResidual, step.rows * hidden);
    // The router input: scale-free rms times the learned scale times
    // hidden^-0.5.
    ops::Normalization::addRms(step.graph, b.gemmaResidual,
                               ops::NormWeights{ffn.routerScale}, b.gdnOutput,
                               hidden, step.rows);
    TargetModel::addLayerScalar(step.graph, b.gdnOutput,
                   1.0F / std::sqrt(static_cast<float>(hidden)),
                   step.rows * hidden);
    const ops::LinearPlan gatePlan = linear.decodePlan(ffn.sharedGate, step.planLanes);
    ops::Normalization::addRms(step.graph, b.gemmaResidual, ffn.preFfnNorm,
                               b.normalized, hidden, step.rows, b.linearScratch,
                               gatePlan.input());
    // The routed experts take the checkpoint's second pre-FFN norm.
    ops::Normalization::addRms(step.graph, b.gemmaResidual, ffn.preFfnNormRouted,
                               b.gdnHidden, hidden, step.rows);
    // Routed experts: raw sum into fullPacked, post-normed into gdnOutput.
    ops::MoE::addGemma(step.graph,
                       {b.gdnHidden, b.zeroResidual, b.fullPacked, b.moe, b.gdnOutput, b.verifyRows},
                       {ffn.routerWeights, ffn.perExpertScale, ffn.expertGate,
                        ffn.expertUp, ffn.expertDown},
                       *step.moe);
    ops::Normalization::addRms(step.graph, b.fullPacked, ffn.postFfnNormRouted,
                               b.gdnOutput, hidden, step.rows);
    // The dense shared expert's GeGLU.
    linear.add(step.graph,
               {.input = b.normalized, .output = b.denseGateScratch,
                .scratch = b.linearScratch},
               ffn.sharedGate, gatePlan);
    linear.add(step.graph,
               {.input = b.normalized, .output = b.denseIntermediate,
                .scratch = b.linearScratch},
               ffn.sharedUp, gatePlan);
    const uint32_t intermediate = step.rows * geometry_.denseIntermediateSize;
    step.graph.add("geglu_multiply", {b.denseGateScratch, b.denseIntermediate, b.denseIntermediate},
                   intermediate, {(intermediate + 255) / 256, 1, 1}, {256, 1, 1});
    linear.add(step.graph,
               {.input = b.denseIntermediate, .output = b.attentionHidden,
                .scratch = b.linearScratch},
               ffn.sharedDown, linear.decodePlan(ffn.sharedDown, step.planLanes));
    ops::Normalization::addRms(step.graph, b.attentionHidden, ffn.postFfnNormShared,
                               b.normalized, hidden, step.rows);
    // The routed + shared sum takes the checkpoint's (un-suffixed) post-FFN
    // norm before it joins the residual.
    TargetModel::addResidualAdd(step.graph, b.gdnOutput, b.normalized, b.attentionHidden,
                   step.rows * hidden);
    ops::Normalization::addRms(step.graph, b.attentionHidden, ffn.postFfnNorm,
                               b.normalized, hidden, step.rows);
    TargetModel::addResidualAdd(step.graph, b.gemmaResidual, b.normalized, output,
                   step.rows * hidden);
    TargetModel::addLayerScalar(step.graph, output, ffn.layerScalar, step.rows * hidden);
    step.graph.endBakedSpan();
  }

private:
  const TargetModel &model_;
  const TargetModelWeights &weights_;
  const TargetModelGeometry &geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

// A flat bf16 residual add (the draft's kernel, batch 1 over the whole row
// block).
void TargetModel::addResidualAdd(metal::CommandGraph &graph, metal::MetalBuffer input,
                                metal::MetalBuffer residual, metal::MetalBuffer output,
                                uint32_t elements) {
  graph.add("draft_residual_add",
            {std::move(input), std::move(residual), std::move(output)},
            elements, {(elements + 255) / 256, 1, 1}, {256, 1, 1});
}

// layer_scalar_scale: values *= scale, in place (Gemma's layer scalar and
// the router input's hidden^-0.5).
void TargetModel::addLayerScalar(metal::CommandGraph &graph, metal::MetalBuffer values,
                                float scale, uint32_t count) {
  graph.addTail("layer_scalar_scale", {std::move(values)}, scale, count,
                {(count + 255) / 256, 1, 1}, {256, 1, 1});
}

std::unique_ptr<TargetLayerEncoder>
makeGemmaLayerEncoder(const TargetModel &model) {
  return std::make_unique<GemmaLayerEncoder>(model);
}

} // namespace richengine::model

// The LFM3 target's layer encoder: the tri-hybrid schedule's conv,
// sliding-window and global attention mixers, the sandwich norms between
// each sublayer's output and the residual, and the head-wise sigmoid gate
// on the attention output — plus the per-pass state (the global layers'
// attention plans and rope tables) its step extension carries. The conv
// halves reuse LFM2's LfmConv operators and FIFO commit unchanged; only
// the residual ordering differs.
#include "model/TargetModelGemmaHelpers.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "ops/LfmConv.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/Normalization.hpp"

#include <any>
#include <memory>
#include <utility>
#include <vector>
#include <optional>

namespace richengine::model {

namespace {

// The LFM3 encoder's per-pass prefill state, carried in
// PrefillStep::extension: the same sequences' plans in the global layers'
// geometry (altAttentionMask's layers — full causal, the theta-1e6
// zero-tail rope table).
struct Lfm3PrefillExtension {
  std::vector<ops::PrefillAttentionPlan> altAttention{};
};

// The same for a verify step: the alternate-geometry plan; empty on a
// single-geometry pass.
struct Lfm3VerifyExtension {
  std::optional<ops::VerifyAttentionPlan> altAttention{};
};

} // namespace

// The lfm3 layer encoder. Its member references alias the model's
// internals so the moved layer bodies read as they did as TargetModel
// methods.
class Lfm3LayerEncoder final : public TargetLayerEncoder {
public:
  explicit Lfm3LayerEncoder(const TargetModel &model)
      : model_(model), weights_(*model.weights_), geometry_(model.geometry_),
        backend_(model.backend_), operators_(model.operators_) {}

  void beginPrefill(TargetModel::PrefillStep &step) const override {
    Lfm3PrefillExtension extension;
    if (geometry_.altAttentionMask)
      for (const TargetModelPrefillSequence &sequence : step.sequences)
        extension.altAttention.push_back(
            operators_.prefillAttention(
                sequence.rows, geometry_.attentionQueryHeads,
                geometry_.layerKvLayout(true)));
    step.extension = std::move(extension);
  }

  void beginVerify(TargetModel::VerifyStep &step) const override {
    Lfm3VerifyExtension extension;
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

  // The rows' head-gate scalar buffer: [rows][queryHeads] of bf16 carved
  // out of the FFN's gate scratch, which is idle until the layer's FFN.
  metal::MetalBuffer gateBuffer(metal::MetalBuffer scratch,
                                uint64_t rows) const {
    return backend_.view(
        scratch, 0,
        rows * geometry_.attentionGateRows * sizeof(uint16_t));
  }

  // One LFM3 layer in a prefill chunk: the conv or attention mixer —
  // sliding or global by index — then the sandwich norms around each
  // sublayer's residual join.
  metal::MetalBuffer addPrefillLayer(TargetModel::PrefillStep &step,
                                     uint32_t index, metal::MetalBuffer input,
                                     metal::MetalBuffer output) const override {
    const TargetModelPrefillBuffers &b = step.buffers;
    const Lfm3PrefillExtension &ext =
        std::any_cast<const Lfm3PrefillExtension &>(step.extension);
    const ops::Linear &linear = operators_.linear();
    const uint32_t hidden = geometry_.hiddenSize;
    const TargetLayerWeights &layer = weights_.layers[index];
    metal::MetalBuffer residual = b.attentionOutput;

    if (const auto *conv = std::get_if<LfmConvWeights>(&layer.mixer)) {
      const uint32_t convLayer = step.gdnLayer++;
      model_.addPrefillNorm(step, input, layer.inputNorm,
                            conv->inputProjection.layout());
      linear.addPrefill(step.graph, b.normalized, conv->inputProjection,
                        b.gdnPacked, b.projectionSums, step.rows,
                        b.linearScratch, {});
      for (const TargetModelPrefillSequence &sequence : step.sequences) {
        const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
          return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin,
                                  sequence.rows, width);
        };
        const metal::MetalBuffer convRows =
            u16(b.gdnHidden, geometry_.convolutionDimension);
        ops::LfmConv::addPrefill(
            step.graph,
            {u16(b.gdnPacked, geometry_.packedGdnWidth),
             conv->convolutionWeights, sequence.convolutionIn[convLayer],
             sequence.convolutionOut[convLayer], convRows},
            geometry_.convShape(), sequence.rows, conv->convolutionTapsMajor);
        if (conv->outputProjection.layout() == ops::WeightLayout::Affine64) {
          const uint32_t paddedRows = (sequence.rows + 31) / 32 * 32;
          linear.addPrefillSums(
              step.graph,
              rowsOf<uint16_t>(backend_, b.gdnHidden, sequence.rowBegin,
                               paddedRows, geometry_.convolutionDimension),
              prefillSums(backend_, b.projectionSums, sequence.rowBegin,
                          paddedRows, geometry_.convolutionDimension),
              conv->outputProjection, sequence.rows);
        }
      }
      // The sandwich norm sits between the mixer's out-projection and the
      // residual add, so the projection runs without its residual epilogue.
      linear.addPrefill(step.graph, b.gdnHidden, conv->outputProjection,
                        residual, b.projectionSums, step.rows,
                        b.linearScratch, {});
    } else {
      const AttentionMixerWeights &mixer =
          std::get<AttentionMixerWeights>(layer.mixer);
      const uint32_t attentionLayer = step.attentionLayer++;
      const bool global = geometry_.isAltAttentionLayer(index) &&
                          !envFlag("RICHENGINE_NO_ALT_LAYERS");
      const kv::Layout layerKv = geometry_.layerKvLayout(global);
      const uint32_t pairs = weights_.layout.rotaryPairsAt(index);
      const metal::MetalBuffer ropeCos = global ? b.ropeCosAlt : b.ropeCos;
      const metal::MetalBuffer ropeSin = global ? b.ropeSinAlt : b.ropeSin;
      model_.addPrefillNorm(step, input, layer.inputNorm,
                            mixer.inputProjection.layout());
      linear.addPrefill(step.graph, b.normalized, mixer.inputProjection,
                        b.fullPacked, b.projectionSums, step.rows,
                        b.linearScratch, {});
      // gate_proj reads the same normalized input into the per-head scalar
      // buffer; it is idle scratch until the FFN runs.
      const metal::MetalBuffer gates = b.denseIntermediate;
      linear.addPrefill(step.graph, b.normalized, mixer.gateProjection, gates,
                        b.projectionSums, step.rows, b.linearScratch, {});
      for (size_t s = 0; s < step.sequences.size(); ++s) {
        const TargetModelPrefillSequence &sequence = step.sequences[s];
        const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
          return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin,
                                  sequence.rows, width);
        };
        const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
          return rowsOf<float>(backend_, buffer, sequence.rowBegin,
                               sequence.rows, width);
        };
        const uint64_t headBytes = uint64_t{sequence.attentionStride} *
                                   geometry_.attentionHeadDimension *
                                   sizeof(uint16_t);
        const uint64_t queryBytes =
            geometry_.attentionQueryHeads * headBytes;
        const uint64_t kvBytes = layerKv.kvHeads * headBytes;
        const metal::MetalBuffer queries =
            backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
        const metal::MetalBuffer attentionRows =
            backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
        const metal::MetalBuffer keys =
            backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
        const metal::MetalBuffer values =
            backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
        ops::PagedAttention::addPrefillProjection(
            step.graph, u16(b.fullPacked, geometry_.packedFullWidth),
            mixer.queryNorm, mixer.keyNorm, f32(ropeCos, pairs),
            f32(ropeSin, pairs), queries, keys, values, sequence.rows,
            sequence.attentionStride, geometry_.attentionQueryHeads, layerKv);
        if (std::getenv("RICHENGINE_VERIFY_DUMP2")) {
          const auto fnv = [](const void *p, size_t n) {
            const auto *by = static_cast<const unsigned char *>(p);
            uint64_t h = 1469598103934665603ull;
            const size_t st = n / 32 + 1;
            for (size_t i = 0; i < n; i += st)
              h = (h * 1099511628211ull) ^ by[i];
            return h;
          };
          fprintf(stderr,
                  "dump2 kvview L=%u seq=%zu global=%d kvoff=%llu "
                  "kp=%p vp=%p kb=%llu vb=%llu "
                  "kh=%016llx vh=%016llx\n",
                  attentionLayer, s, global ? 1 : 0,
                  (unsigned long long)sequence.kvOffset,
                  keys.contents(), values.contents(),
                  (unsigned long long)keys.sizeBytes(),
                  (unsigned long long)values.sizeBytes(),
                  (unsigned long long)fnv(keys.contents(),
                                          keys.sizeBytes()),
                  (unsigned long long)fnv(values.contents(),
                                          values.sizeBytes()));
        }
        ops::PagedAttention::addPrefillStore(
            step.graph, step.kvLayers[attentionLayer], keys, values,
            sequence.pageTable, sequence.chunk, layerKv);
        ops::PagedAttention::addPrefill(
            step.graph, step.kvLayers[attentionLayer], queries,
            attentionRows, b.attentionPartials, b.attentionStatistics,
            sequence.pageTable, sequence.chunk,
            global ? ext.altAttention[s] : step.attention[s]);
        const metal::MetalBuffer hiddenRows =
            u16(b.attentionHidden, geometry_.attentionWidth);
        const metal::MetalBuffer sequenceGates = backend_.view(
            gates, uint64_t{sequence.rowBegin} * geometry_.attentionGateRows *
                       sizeof(uint16_t),
            uint64_t{sequence.rows} * geometry_.attentionGateRows *
                sizeof(uint16_t));
        // The sums-emitting head gate writes the out-projection's input
        // sums beside the gated rows; a block-layout out-projection takes
        // the plain variant plus the generic sums pass.
        if (mixer.outputProjection.layout() == ops::WeightLayout::Affine64) {
          ops::PagedAttention::addPrefillGateSums(
              step.graph, sequenceGates, attentionRows, hiddenRows,
              prefillSums(backend_, b.projectionSums, sequence.rowBegin,
                          sequence.rows, geometry_.attentionWidth),
              sequence.rows, sequence.attentionStride,
              geometry_.attentionQueryHeads, layerKv, /*headGate*/ true);
        } else {
          ops::PagedAttention::addPrefillGate(
              step.graph, sequenceGates, attentionRows, hiddenRows,
              sequence.rows, sequence.attentionStride,
              geometry_.attentionQueryHeads, layerKv, /*queryGate*/ false,
              /*headGate*/ true);
        }
      }
      linear.addPrefill(step.graph, b.attentionHidden, mixer.outputProjection,
                        residual, b.projectionSums, step.rows,
                        b.linearScratch, {});
    }
    // R = input + postMixerNorm(mixer(x)); the FFN reads R.
    ops::Normalization::addRms(step.graph, residual, layer.postMixerNorm,
                               b.normalized, hidden, step.rows);
    TargetModel::addResidualAdd(step.graph, input, b.normalized, residual,
                                step.rows * hidden);
    // The FFN: norm, the SwiGLU triple and a non-residual down projection,
    // then the post-FFN sandwich norm before it joins the residual.
    const DenseFfnWeights &ffn = std::get<DenseFfnWeights>(layer.ffn);
    const ops::PrefillFfnBuffers fb = b.ffn();
    model_.addPrefillNorm(step, residual, layer.postAttentionNorm,
                          ffn.gateProjection.layout());
    const ops::PreparedInput prepared =
        linear.addPrefill(step.graph, fb.normalized, ffn.gateProjection,
                          fb.gateScratch, fb.sums, step.rows, fb.scratch, {});
    linear.addPrefillUpWithGate(step.graph, fb.normalized, ffn.upProjection,
                                fb.gateScratch, fb.intermediate, fb.sums,
                                fb.downSums, step.rows, fb.scratch, prepared);
    linear.addPrefill(step.graph, fb.intermediate, ffn.downProjection,
                      b.attentionHidden, fb.downSums, step.rows,
                      fb.scratch, {});
    ops::Normalization::addRms(step.graph, b.attentionHidden,
                               layer.postFfnNorm, b.normalized, hidden,
                               step.rows);
    TargetModel::addResidualAdd(step.graph, residual, b.normalized, output,
                                step.rows * hidden);
    return output;
  }

  // The same layer in a verify step. The conv branch replays LFM2's
  // dispatch (state read, bx rows for the commit) without the out-
  // projection's residual epilogue; the attention branch binds the layer
  // kind's rope table and plan, then the head gate — fused into the chain
  // reduce where the out-projection reads Plain input.
  void addVerifyLayer(TargetModel::VerifyStep &step, uint32_t index,
                      metal::MetalBuffer input,
                      metal::MetalBuffer output) const override {
    const TargetModelVerifyBuffers &b = step.buffers;
    const Lfm3VerifyExtension &ext =
        std::any_cast<const Lfm3VerifyExtension &>(step.extension);
    const ops::Linear &linear = operators_.linear();
    const uint32_t hidden = geometry_.hiddenSize;
    const TargetLayerWeights &layer = weights_.layers[index];
    metal::MetalBuffer residual = b.gemmaResidual;

    if (const auto *conv = std::get_if<LfmConvWeights>(&layer.mixer)) {
      const uint32_t convLayer = step.gdnLayer++;
      const ops::LinearPlan inputPlan =
          linear.decodePlan(conv->inputProjection, step.planLanes,
                            ops::LinearEpilogue::None, nullptr,
                            step.rowCapacity);
      step.graph.beginBakedSpan();
      const ops::PreparedInput normalized = ops::Normalization::addRms(
          step.graph, input, layer.inputNorm, b.normalized, hidden,
          step.rows, b.linearScratch, inputPlan.input());
      linear.add(step.graph,
                 {.input = b.normalized, .output = b.gdnPacked[convLayer],
                  .scratch = b.linearScratch, .prepared = normalized},
                 conv->inputProjection, inputPlan);
      const ops::LinearPlan outputPlan =
          linear.decodePlan(conv->outputProjection, step.planLanes,
                            ops::LinearEpilogue::None, nullptr,
                            step.rowCapacity);
      ops::PreparedInput convOut;
      if (step.tree) {
        ops::LfmConv::addVerifyTree(
            step.graph,
            {b.gdnPacked[convLayer], conv->convolutionWeights,
             b.currentGdnStates, b.gdnMixed[convLayer], b.gdnHidden,
             b.treeNodes, b.treeCounts},
            geometry_.convShape(), step.lanes, step.rowCapacity, convLayer,
            geometry_.stateLayout.convolutionLayerBytes(),
            conv->convolutionTapsMajor);
      } else {
        convOut = ops::LfmConv::addVerify(
            step.graph,
            {b.gdnPacked[convLayer], conv->convolutionWeights,
             b.currentGdnStates, b.gdnMixed[convLayer], b.gdnHidden,
             b.linearScratch},
            geometry_.convShape(), step.lanes, step.rowCapacity, convLayer,
            geometry_.stateLayout.convolutionLayerBytes(), outputPlan.input(),
            conv->convolutionTapsMajor);
      }
      linear.add(step.graph,
                 {.input = b.gdnHidden, .output = residual,
                  .scratch = b.linearScratch, .prepared = convOut},
                 conv->outputProjection, outputPlan);
      step.graph.endBakedSpan();
    } else {
      const AttentionMixerWeights &mixer =
          std::get<AttentionMixerWeights>(layer.mixer);
      const uint32_t attentionLayer = step.attentionLayer++;
      const bool global = geometry_.isAltAttentionLayer(index) &&
                          !envFlag("RICHENGINE_NO_ALT_LAYERS");
      const kv::Layout layerKv = geometry_.layerKvLayout(global);
      const metal::MetalBuffer ropeCos = global ? b.ropeCosAlt : b.ropeCos;
      const metal::MetalBuffer ropeSin = global ? b.ropeSinAlt : b.ropeSin;
      step.graph.beginBakedSpan();
      const ops::LinearPlan inputPlan =
          linear.decodePlan(mixer.inputProjection, step.planLanes,
                            ops::LinearEpilogue::None, nullptr,
                            step.rowCapacity);
      const ops::PreparedInput normalized = ops::Normalization::addRms(
          step.graph, input, layer.inputNorm, b.normalized, hidden,
          step.rows, b.linearScratch, inputPlan.input());
      linear.add(step.graph,
                 {.input = b.normalized, .output = b.fullPacked,
                  .scratch = b.linearScratch, .prepared = normalized},
                 mixer.inputProjection, inputPlan);
      const ops::LinearPlan gatePlan =
          linear.decodePlan(mixer.gateProjection, step.planLanes,
                            ops::LinearEpilogue::None, nullptr,
                            step.rowCapacity);
      if (envFlag("RICHENGINE_VERIFY_DUMP2")) {
        const auto iw = inputPlan.workload(), gw = gatePlan.workload();
        fprintf(stderr,
                "dump2 lfm3-verify L=%u planLanes=%u rowCap=%u "
                "iw={r=%u ph=%u ep=%u wl=%u ln=%u} "
                "gw={r=%u ph=%u ep=%u wl=%u ln=%u} "
                "inputIn=%u gateIn=%u\n",
                attentionLayer, step.planLanes, step.rowCapacity,
                iw.rows, uint32_t(iw.phase), uint32_t(iw.epilogue),
                uint32_t(iw.weightLayout), iw.lanes,
                gw.rows, uint32_t(gw.phase), uint32_t(gw.epilogue),
                uint32_t(gw.weightLayout), gw.lanes,
                uint32_t(inputPlan.input()), uint32_t(gatePlan.input()));
        fprintf(stderr,
                "dump2 lfm3-tile L=%u inputTile=%u inputGroups=%u "
                "inputThreads=%u gateTile=%u gateGroups=%u "
                "gateThreads=%u inDest=%u gateDest=%u\n",
                attentionLayer,
                uint32_t(inputPlan.configuration().tile),
                inputPlan.groups(),
                inputPlan.threadsPerThreadgroup(),
                uint32_t(gatePlan.configuration().tile),
                gatePlan.groups(), gatePlan.threadsPerThreadgroup(),
                uint32_t(inputPlan.destination()),
                uint32_t(gatePlan.destination()));
        fprintf(stderr,
                "dump2 lfm3-src L=%u normSrc=%p bNorm=%p normLay=%u "
                "iScratch=%llu gScratch=%llu\n",
                attentionLayer, normalized.source.contents(),
                b.normalized.contents(), uint32_t(normalized.layout),
                (unsigned long long)inputPlan.scratchSize().bytes(),
                (unsigned long long)gatePlan.scratchSize().bytes());
      }
      const metal::MetalBuffer gates = b.denseIntermediate;
      linear.add(step.graph,
                 {.input = b.normalized, .output = gates,
                  .scratch = b.linearScratch, .prepared = normalized},
                 mixer.gateProjection, gatePlan);
      ops::PagedAttention::addVerifyProjection(
          step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, ropeCos,
          ropeSin, b.fullQueries, b.chunkKeys[attentionLayer],
          b.chunkValues[attentionLayer], geometry_.attentionQueryHeads,
          layerKv, step.lanes, step.rowCapacity);
      const ops::LinearPlan outputPlan =
          linear.decodePlan(mixer.outputProjection, step.planLanes,
                            ops::LinearEpilogue::None, nullptr,
                            step.rowCapacity);
      const ops::VerifyAttentionPlan &plan =
          global ? *ext.altAttention : step.attention;
      const bool fusedReduce =
          !step.tree && outputPlan.input() == ops::LinearInput::Plain &&
          !envFlag("RICHENGINE_NO_FUSED_REDUCE");
      if (fusedReduce) {
        // verify_attention_reduce_headgate_*: the hidden layout directly.
        ops::PagedAttention::addVerify(
            step.graph, step.kvLayers[attentionLayer],
            {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer],
             b.fullQueries, b.attentionPartials, b.attentionStatistics,
             b.fullAttention, b.pageTables, b.treeMasks},
            step.chunks, plan, gates, b.attentionHidden, {},
            /*headGate*/ true);
      } else {
        ops::PagedAttention::addVerify(
            step.graph, step.kvLayers[attentionLayer],
            {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer],
             b.fullQueries, b.attentionPartials, b.attentionStatistics,
             b.fullAttention, b.pageTables, b.treeMasks},
            step.chunks, plan);
        ops::PagedAttention::addVerifyGate(
            step.graph, gates, b.fullAttention, b.attentionHidden,
            geometry_.attentionQueryHeads, layerKv, step.lanes,
            b.linearScratch, outputPlan.input(), /*queryGate*/ false,
            step.rowCapacity, /*headGate*/ true);
      }
      linear.add(step.graph,
                 {.input = b.attentionHidden, .output = residual,
                  .scratch = b.linearScratch},
                 mixer.outputProjection, outputPlan);
      step.graph.endBakedSpan();
    }
    // R = input + postMixerNorm(mixer(x)).
    ops::Normalization::addRms(step.graph, residual, layer.postMixerNorm,
                               b.normalized, hidden, step.rows);
    TargetModel::addResidualAdd(step.graph, input, b.normalized, residual,
                                step.rows * hidden);
    // The FFN: norm, the fused gate/up and a non-residual down, then the
    // post-FFN sandwich norm before it joins the residual.
    const DenseFfnWeights &ffn = std::get<DenseFfnWeights>(layer.ffn);
    step.graph.beginBakedSpan();
    const ops::LinearPlan gateUpPlan =
        linear.decodePlan(ffn.upProjection, step.planLanes,
                          ops::LinearEpilogue::GateUp, &ffn.gateProjection,
                          step.rowCapacity);
    const ops::LinearPlan downPlan =
        linear.decodePlan(ffn.downProjection, step.planLanes,
                          ops::LinearEpilogue::None, nullptr,
                          step.rowCapacity);
    const bool emitTable =
        gateUpPlan.usesSimdgroup() && downPlan.usesSimdgroup();
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        step.graph, residual, layer.postAttentionNorm, b.normalized, hidden,
        step.rows, b.linearScratch, gateUpPlan.input());
    linear.add(step.graph,
               {.input = b.normalized, .output = b.denseIntermediate,
                .gateScratch = b.denseGateScratch,
                .scratch = b.linearScratch, .prepared = normalized,
                .emitTable64 = emitTable},
               ffn.upProjection, gateUpPlan, &ffn.gateProjection);
    ops::LinearScratch downScratch = b.linearScratch;
    downScratch.input = b.denseIntermediate;
    linear.add(step.graph,
               {.input = b.denseIntermediate, .output = b.attentionOutput,
                .scratch = emitTable ? downScratch : b.linearScratch,
                .prepared = emitTable
                               ? ops::PreparedInput{b.denseIntermediate,
                                                    ops::LinearInput::Table64Paired}
                               : ops::PreparedInput{}},
               ffn.downProjection, downPlan);
    ops::Normalization::addRms(step.graph, b.attentionOutput,
                               layer.postFfnNorm, b.normalized, hidden,
                               step.rows);
    TargetModel::addResidualAdd(step.graph, residual, b.normalized, output,
                                step.rows * hidden);
    step.graph.endBakedSpan();
  }

private:
  const TargetModel &model_;
  const TargetModelWeights &weights_;
  const TargetModelGeometry &geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

std::unique_ptr<TargetLayerEncoder>
makeLfm3LayerEncoder(const TargetModel &model) {
  return std::make_unique<Lfm3LayerEncoder>(model);
}

} // namespace richengine::model

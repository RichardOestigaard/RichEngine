// The maxwell_moe target's packed loader and its layer encoder
// (model/MaxwellMoe.hpp). The HF reference is the repo's bundled
// modeling_maxwell_moe.py; the vendored qwen4exp port under
// runtime/qwen4/ shows the same hyper-connected MoE skeleton, whose
// kernels this target's mHC and MBSA pieces replace.
#include "model/GgufTarget.hpp"
#include "model/MaxwellMoe.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetModelGemmaHelpers.hpp"
#include "model/TargetWeights.hpp"
#include "metal/abi/Mbsa.h"
#include "ops/HyperConnection.hpp"
#include "ops/Mbsa.hpp"
#include "ops/Mhc.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <any>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace richengine::model {
namespace {

// The maxwell_moe additions to requireTargetLayout's checks: every layer
// carries the MBSA indexer, whose groups, windows and per-token keys are
// set and consistent, and the mHC geometry agrees with the streams.
void requireMaxwellLayout(const TargetLayout &layout) {
  requireTargetLayout(layout);
  const uint32_t n = layout.hyperConnections;
  const bool mhc = n && layout.mhcMixWidth == n * n + 2 * n &&
                   layout.mhcSinkhornIterations && layout.mhcHead;
  const bool indexer = layout.indexerHeads == layout.indexerRouteGroups &&
                       layout.indexerHeads == layout.attentionKvHeads &&
                       layout.indexerHeadDimension ==
                           layout.attentionHeadDimension &&
                       layout.indexerBlockTokens &&
                       layout.indexerPrefixBlocks &&
                       layout.indexerLocalBlocks &&
                       layout.indexerRemoteBlocks &&
                       layout.indexerPerTokenKeys;
  if (layout.layers != 45 || layout.hiddenSize != 2048 ||
      layout.vocabularySize != 200704 || layout.packedFullWidth != 5120 ||
      layout.attentionWidth != 4096 || layout.attentionQueryHeads != 32 ||
      layout.attentionKvHeads != 4 || layout.attentionHeadDimension != 128 ||
      layout.rotaryPairs != 64 || !(layout.rotaryTheta > 0.0F) ||
      !layout.maximumContextTokens || layout.intermediateSize != 6144 ||
      layout.experts != 256 || layout.expertsPerToken != 12 ||
      layout.expertIntermediateSize != 512 ||
      layout.sharedIntermediateSize != 1024 || !layout.attentionQkNorm ||
      !mhc || !indexer)
    throw WeightStoreError("maxwell_moe target layout is inconsistent");
}

// One mHC module's sections: the fp32 mapping projection, the three F32
// gating factors and the mhcMixWidth F32 bias.
MaxwellMhcWeights readMhc(WeightFile &file, const BlockTargetFormat &format,
                          const TargetLayout &layout,
                          std::string_view label) {
  MaxwellMhcWeights mhc;
  mhc.mapping = format.projection(file, layout.mhcMixWidth,
                                  layout.streamWidth(), label);
  mhc.mapping.destination = ops::FloatOutput::Float32;
  mhc.alphas = file.section(3 * sizeof(float), label);
  mhc.bias = file.section(uint64_t{layout.mhcMixWidth} * sizeof(float), label);
  return mhc;
}

// A layer's MBSA indexer after its mixer's sections: the fp32 group query
// and key projections and their per-head norms.
MaxwellIndexerWeights readIndexer(WeightFile &file,
                                  const BlockTargetFormat &format,
                                  const TargetLayout &layout) {
  MaxwellIndexerWeights indexer;
  indexer.query =
      format.projection(file, layout.indexerHeads * layout.indexerHeadDimension,
                        layout.hiddenSize, "indexer-query");
  indexer.key = format.projection(file, layout.indexerHeadDimension,
                                  layout.hiddenSize, "indexer-key");
  // The indexer scores in fp32.
  indexer.query.destination = ops::FloatOutput::Float32;
  indexer.key.destination = ops::FloatOutput::Float32;
  indexer.queryNorm =
      format.norm(file, layout.indexerHeadDimension, "indexer-query-norm");
  indexer.keyNorm =
      format.norm(file, layout.indexerHeadDimension, "indexer-key-norm");
  return indexer;
}

// The attention mixer of a full-attention layer (every layer): the fused
// QKV, its per-head q/k RMS norms and the out projection.
MixerWeights readMaxwellMixer(WeightFile &file, const BlockTargetFormat &format,
                              const MixerGeometry &geometry) {
  AttentionMixerWeights attention;
  attention.inputProjection =
      format.fused(file, geometry.packedFullWidth, geometry.hiddenSize,
                   "attention-input", {"attn-q", "attn-k", "attn-v"});
  attention.queryNorm =
      format.norm(file, geometry.attentionHeadDimension, "query-norm");
  attention.keyNorm =
      format.norm(file, geometry.attentionHeadDimension, "key-norm");
  attention.outputProjection = format.projection(
      file, geometry.hiddenSize, geometry.attentionWidth, "attention-output");
  return attention;
}

// A leading dense layer's SwiGLU FFN (w1 gate, w3 up, w2 down).
DenseFfnWeights readDenseFfn(WeightFile &file, const BlockTargetFormat &format,
                             const TargetLayout &layout) {
  return {format.projection(file, layout.intermediateSize, layout.hiddenSize,
                            "mlp-gate"),
          format.projection(file, layout.intermediateSize, layout.hiddenSize,
                            "mlp-up"),
          format.projection(file, layout.hiddenSize, layout.intermediateSize,
                            "mlp-down")};
}

// The routed MoE block: the F32 router, the F32 per-expert selection bias
// and the quantized expert slabs. The shared expert is not a route —
// it runs beside the block (MaxwellSharedFfn).
ops::MoeWeights readMoeBlock(WeightFile &file) {
  ops::BlockMoeWeights block;
  block.router = readQuantizedSegment(file, "router");
  block.expertBias = readQuantizedSegment(file, "expert-bias");
  block.gate.routed = readQuantizedSegment(file, "experts-gate");
  block.up.routed = readQuantizedSegment(file, "experts-up");
  block.down.routed = readQuantizedSegment(file, "experts-down");
  return ops::MoeWeights{std::move(block)};
}

// The ungated shared expert's SwiGLU after the routed block's sections.
MaxwellSharedFfn readSharedFfn(WeightFile &file,
                               const BlockTargetFormat &format,
                               const TargetLayout &layout) {
  return {format.projection(file, layout.sharedIntermediateSize,
                            layout.hiddenSize, "shared-gate"),
          format.projection(file, layout.sharedIntermediateSize,
                            layout.hiddenSize, "shared-up"),
          format.projection(file, layout.hiddenSize,
                            layout.sharedIntermediateSize, "shared-down")};
}

// The head's HC contraction before the final norm and logits: the fp32
// streams-by-streamWidth projection, the F32 base and scale.
MaxwellHcHeadWeights readHcHead(WeightFile &file,
                                const BlockTargetFormat &format,
                                const TargetLayout &layout) {
  MaxwellHcHeadWeights head;
  head.fn = format.projection(file, layout.hyperConnections,
                              layout.streamWidth(), "hc-head");
  head.fn.destination = ops::FloatOutput::Float32;
  head.base =
      file.section(uint64_t{layout.hyperConnections} * sizeof(float), "hc-head");
  head.scale = file.section(sizeof(float), "hc-head");
  return head;
}

} // namespace

// The maxwell_moe layer encoder. The residual it encodes is
// hyperConnections fp32 streams per row; the mHC and MBSA pieces whose
// kernels richengine does not have yet throw named errors at encode time
// rather than silently approximating (the qwen4exp convention). Named at
// namespace scope: TargetModel declares it a friend.
// The encoder's per-step tensors: the four fp32 residual streams, the mHC
// mapping's logits and gated rows, the stream aggregate and branch rows,
// the dense-layer zeros and MoE shared expert rows, the MBSA projections
// (fp32 queries/key in, bf16 rotated queries out) and per-group route
// lists, and the HC head's contraction weights.
struct MaxwellStepScratch final {
  metal::MetalBuffer streams;
  metal::MetalBuffer logits;
  metal::MetalBuffer mappings;
  metal::MetalBuffer aggregated;
  metal::MetalBuffer branch;
  metal::MetalBuffer zeros;
  metal::MetalBuffer shared;
  metal::MetalBuffer mbsaQueries;
  metal::MetalBuffer routeLists;
  metal::MetalBuffer routeCounts;
  metal::MetalBuffer headWeights;
};

MaxwellStepScratch stepScratch(metal::MetalBackend &backend,
                               const TargetLayout &layout, uint64_t rows,
                               uint64_t tiles) {
  const auto buffer = [&](uint64_t bytes, const char *label) {
    return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  };
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t streams = layout.streamWidth();
  const uint32_t mixWidth = layout.mhcMixWidth;
  const uint32_t indexWidth =
      layout.indexerHeads * layout.indexerHeadDimension;
  MaxwellStepScratch scratch;
  scratch.streams =
      buffer(rows * streams * sizeof(float), "maxwell-streams");
  scratch.logits =
      buffer(rows * mixWidth * sizeof(float), "maxwell-mhc-logits");
  scratch.mappings =
      buffer(rows * mixWidth * sizeof(float), "maxwell-mhc-mappings");
  scratch.aggregated =
      buffer(rows * hidden * sizeof(uint16_t), "maxwell-aggregated");
  scratch.branch =
      buffer(rows * hidden * sizeof(uint16_t), "maxwell-branch");
  scratch.zeros = buffer(rows * hidden * sizeof(uint16_t), "maxwell-zeros");
  scratch.shared =
      buffer(rows * hidden * sizeof(uint16_t), "maxwell-shared");
  scratch.mbsaQueries =
      buffer(rows * indexWidth * sizeof(uint16_t), "maxwell-mbsa-queries");
  // (tiles * groups) lists of the worst-case route length.
  scratch.routeLists =
      buffer(tiles * layout.indexerRouteGroups *
                 ops::Mbsa::kRouteListEntries * sizeof(uint32_t),
             "maxwell-route-lists");
  scratch.routeCounts =
      buffer(tiles * layout.indexerRouteGroups * sizeof(uint32_t),
             "maxwell-route-counts");
  scratch.headWeights = buffer(
      rows * layout.hyperConnections * sizeof(float), "maxwell-head-weights");
  std::memset(scratch.zeros.contents(), 0, scratch.zeros.sizeBytes());
  return scratch;
}

// Each prefill sequence's sparse plan (every sequence routes all its
// blocks; there is no dense-window shortcut under MBSA).
struct MaxwellPrefillExtension final {
  std::vector<ops::PrefillAttentionPlan> sparseAttention;
};

// The lanes' shared verify plan (its rowTiles counts the route lists).
struct MaxwellVerifyExtension final {
  ops::VerifyAttentionPlan sparseAttention;
};

// The single fp32 plane of an mHC mapping or HC head projection.
metal::MetalBuffer floatPlane(const ops::Projection &projection,
                              const char *what) {
  const auto &segments = projection.blocks().segments;
  if (segments.size() != 1 || !segments.front().isFloat())
    throw std::invalid_argument(what);
  return segments.front().plane0;
}

// RICHENGINE_MHC_UNFUSED: parity harness — runs the unfused mhc chain the
// fused mhc_module kernel replaced, for side-by-side verification dumps.
bool unfusedMhc() {
  static const bool flag = std::getenv("RICHENGINE_MHC_UNFUSED") != nullptr;
  return flag;
}

class MaxwellLayerEncoder final : public TargetLayerEncoder {
public:
  explicit MaxwellLayerEncoder(const TargetModel &model)
      : model_(model),
        weights_(static_cast<const MaxwellMoeWeights &>(*model.weights_)),
        // lane * RICHENGINE_TREE_VERIFY_NODES covers a tree batch's rows
        // and its two tiles of route lists per lane.
        verifyTiles_(static_cast<uint32_t>(
            uint64_t{ExecutionLimits::maximumBatchWidth} *
            ((RICHENGINE_TREE_VERIFY_NODES + kv::kSparseTileRows - 1) /
             kv::kSparseTileRows))),
        prefill_(stepScratch(
            model_.backend_, weights_.layout,
            ExecutionLimits::prefillTokenBudget,
            kv::prefillAttentionTiles(ExecutionLimits::prefillTokenBudget))),
        verify_(stepScratch(
            model_.backend_, weights_.layout,
            uint64_t{ExecutionLimits::maximumBatchWidth} *
                RICHENGINE_TREE_VERIFY_NODES,
            verifyTiles_)) {}

  // The streams start as the embedding's rows — stream 0 the row, the
  // rest zero — and each sequence's sparse plan lands in the extension.
  void beginPrefill(TargetModel::PrefillStep &step) const override {
    ops::HyperConnection::addInit(step.graph, step.buffers.hidden[0],
                                  prefill_.streams, hyperShape(), step.rows);
    MaxwellPrefillExtension extension;
    for (const TargetModelPrefillSequence &sequence : step.sequences) {
      if (!sequence.indexTable)
        throw std::invalid_argument(
            "maxwell_moe prefill sequence lacks its index table");
      extension.sparseAttention.push_back(ops::PagedAttention::prefillPlan(
          sequence.rows, model_.geometry_.attentionQueryHeads,
          model_.geometry_.layerKvLayout(false), /*sparse*/ true));
    }
    step.extension = std::move(extension);
  }

  void beginVerify(TargetModel::VerifyStep &step) const override {
    ops::HyperConnection::addInit(step.graph, step.buffers.hidden[0],
                                  verify_.streams, hyperShape(), step.rows);
    std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      histories[lane] = step.chunks[lane].committed_tokens;
    MaxwellVerifyExtension extension{
        ops::PagedAttention::verifyPlan(
            step.lanes, model_.geometry_.attentionQueryHeads,
            model_.geometry_.layerKvLayout(false),
            std::span(histories).first(step.lanes), step.tree,
            step.liveNodes, step.rowCapacity, /*sparse*/ true)};
    if (uint64_t{step.lanes} * extension.sparseAttention.rowTiles >
        verifyTiles_)
      throw std::logic_error("maxwell_moe verify route lists overflow");
    step.extension = std::move(extension);
  }

  // The encoder writes the capture layers' rows itself (the stream mean).
  [[nodiscard]] bool capturesHidden() const override { return true; }

  // One layer of a prefill chunk, in reference order: the attention mHC
  // (mapping, aggregate, input norm), the MBSA indexer and the sparse
  // attention branch, the attention expand-mix, the FFN mHC (mapping,
  // aggregate, post-attention norm), the dense or routed FFN branch, the
  // FFN expand-mix and — on the last layer — the HC head contraction.
  metal::MetalBuffer addPrefillLayer(TargetModel::PrefillStep &step,
                                     uint32_t index,
                                     metal::MetalBuffer /*input*/,
                                     metal::MetalBuffer output) const override {
    const MaxwellLayerExtras &extra = weights_.extras[index];
    const TargetLayerWeights &layer = weights_.layers[index];
    const AttentionMixerWeights &mixer =
        std::get<AttentionMixerWeights>(layer.mixer);
    addPrefillModule(step, extra.attentionMhc, layer.inputNorm,
                     mixer.inputProjection.layout());
    addPrefillMbsa(step, extra.indexer, mixer);
    ops::Mhc::addExpandMix(step.graph, prefill_.streams, prefill_.branch,
                           prefill_.mappings, mhcShape(), step.rows);
    const ops::Projection *ffnConsumer = nullptr;
    if (const auto *dense = std::get_if<DenseFfnWeights>(&layer.ffn))
      ffnConsumer = &dense->gateProjection;
    addPrefillModule(step, extra.ffnMhc, layer.postAttentionNorm,
                     ffnConsumer ? ffnConsumer->layout()
                                 : ops::WeightLayout::Block32);
    addPrefillFfn(step, layer, extra);
    ops::Mhc::addExpandMix(step.graph, prefill_.streams, prefill_.branch,
                           prefill_.mappings, mhcShape(), step.rows);
    if (index + 1 == weights_.layout.layers)
      addPrefillHead(step, output);
    addPrefillCapture(step, index);
    return output;
  }

  void addVerifyLayer(TargetModel::VerifyStep &step, uint32_t index,
                      metal::MetalBuffer /*input*/,
                      metal::MetalBuffer output) const override {
    const MaxwellLayerExtras &extra = weights_.extras[index];
    const TargetLayerWeights &layer = weights_.layers[index];
    const ops::Linear &linear = model_.operators_.linear();
    const AttentionMixerWeights &mixer =
        std::get<AttentionMixerWeights>(layer.mixer);
    const ops::LinearPlan inputPlan =
        linear.decodePlan(mixer.inputProjection, step.planLanes,
                          ops::LinearEpilogue::None, nullptr,
                          step.rowCapacity);
    const ops::PreparedInput mixInput = addVerifyModule(
        step, extra.attentionMhc, layer.inputNorm, inputPlan.input());
    addVerifyMbsa(step, extra.indexer, mixer, inputPlan, mixInput);
    ops::Mhc::addExpandMix(step.graph, verify_.streams, verify_.branch,
                           verify_.mappings, mhcShape(), step.rows);
    // The FFN's first consumer: the dense gate/up pair's plan input, or
    // the router's plain bf16 rows.
    if (const auto *dense = std::get_if<DenseFfnWeights>(&layer.ffn)) {
      const ops::LinearPlan gateUpPlan =
          linear.decodePlan(dense->upProjection, step.planLanes,
                            ops::LinearEpilogue::GateUp,
                            &dense->gateProjection, step.rowCapacity);
      const ops::PreparedInput normOut =
          addVerifyModule(step, extra.ffnMhc, layer.postAttentionNorm,
                          gateUpPlan.input());
      addVerifySwiGlu(step, dense->gateProjection, dense->upProjection,
                      dense->downProjection, gateUpPlan, normOut,
                      verify_.branch);
    } else {
      // The shared expert, when the layer has one, is the norm's first
      // consumer: the router reads the same rows plain.
      const std::optional<ops::LinearPlan> sharedPlan =
          extra.sharedFfn
              ? std::optional{linear.decodePlan(
                    extra.sharedFfn->up, step.planLanes,
                    ops::LinearEpilogue::GateUp, &extra.sharedFfn->gate,
                    step.rowCapacity)}
              : std::nullopt;
      const ops::PreparedInput normOut = addVerifyModule(
          step, extra.ffnMhc, layer.postAttentionNorm,
          sharedPlan ? sharedPlan->input() : ops::LinearInput::Plain);
      addVerifyMoe(step, layer, extra, sharedPlan, normOut);
    }
    ops::Mhc::addExpandMix(step.graph, verify_.streams, verify_.branch,
                           verify_.mappings, mhcShape(), step.rows);
    if (index + 1 == weights_.layout.layers)
      addVerifyHead(step, output);
    addVerifyCapture(step, index);
  }

private:
  [[nodiscard]] ops::MhcShape mhcShape() const {
    const TargetLayout &layout = weights_.layout;
    return {layout.hiddenSize, layout.hyperConnections,
            layout.mhcSinkhornIterations};
  }
  [[nodiscard]] ops::HyperShape hyperShape() const {
    const TargetLayout &layout = weights_.layout;
    return {layout.hiddenSize, layout.hyperConnections, 0};
  }

  // A module's mHC before its norm: the fp32 mapping projection of the
  // streams into logits, the gated mappings and the aggregate's h_pre
  // contraction — then the layer norm into b.normalized for `consumer`.
  void addPrefillModule(TargetModel::PrefillStep &step,
                        const MaxwellMhcWeights &mhc,
                        const ops::NormWeights &norm,
                        ops::WeightLayout consumer) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const ops::MhcShape shape = mhcShape();
    if (unfusedMhc()) {
      ops::Mhc::addProject(step.graph, prefill_.streams,
                           floatPlane(mhc.mapping, "maxwell_moe mHC mapping"),
                           weights_.layout.mhcMixWidth, prefill_.logits,
                           shape, step.rows);
      ops::Mhc::addMappings(step.graph, prefill_.streams, prefill_.logits,
                            mhc.alphas, mhc.bias, prefill_.mappings, shape,
                            step.rows);
      ops::Mhc::addWeightedSum(step.graph, prefill_.streams,
                               prefill_.mappings, shape.mixWidth(), 0,
                               prefill_.aggregated, shape, step.rows);
    } else {
      ops::Mhc::addModule(step.graph, prefill_.streams,
                          floatPlane(mhc.mapping, "maxwell_moe mHC mapping"),
                          mhc.alphas, mhc.bias, prefill_.mappings,
                          prefill_.aggregated, shape, step.rows);
    }
    if (consumer == ops::WeightLayout::Affine64)
      ops::Normalization::addRmsWithQ4Sums(step.graph, prefill_.aggregated,
                                           norm, b.normalized,
                                           b.projectionSums,
                                           model_.geometry_.hiddenSize,
                                           step.rows);
    else
      ops::Normalization::addRms(step.graph, prefill_.aggregated, norm,
                                 b.normalized, model_.geometry_.hiddenSize,
                                 step.rows);
  }

  // The same in a verify step: the returned PreparedInput is the
  // consumer's table form of b.normalized.
  ops::PreparedInput addVerifyModule(TargetModel::VerifyStep &step,
                                     const MaxwellMhcWeights &mhc,
                                     const ops::NormWeights &norm,
                                     ops::LinearInput consumer) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    // The fused module wins at prefill widths but starves the GPU at verify
    // row counts: one threadgroup per row is eight groups where the unfused
    // chain spreads its projection over rows * mixWidth of them.
    const ops::MhcShape shape = mhcShape();
    if (step.rows >= 64 && !unfusedMhc()) {
      ops::Mhc::addModule(step.graph, verify_.streams,
                          floatPlane(mhc.mapping, "maxwell_moe mHC mapping"),
                          mhc.alphas, mhc.bias, verify_.mappings,
                          verify_.aggregated, shape, step.rows);
    } else {
      ops::Mhc::addProject(step.graph, verify_.streams,
                           floatPlane(mhc.mapping, "maxwell_moe mHC mapping"),
                           weights_.layout.mhcMixWidth, verify_.logits, shape,
                           step.rows);
      ops::Mhc::addMappings(step.graph, verify_.streams, verify_.logits,
                            mhc.alphas, mhc.bias, verify_.mappings, shape,
                            step.rows);
      ops::Mhc::addWeightedSum(step.graph, verify_.streams, verify_.mappings,
                               shape.mixWidth(), 0, verify_.aggregated, shape,
                               step.rows);
    }
    return ops::Normalization::addRms(
        step.graph, verify_.aggregated, norm, b.normalized,
        model_.geometry_.hiddenSize, step.rows, b.linearScratch, consumer,
        false);
  }

  // The layer's index-plane offset in every index entry's plane: layer *
  // extentPages * a page's index bytes — per-token keys of
  // indexerHeadDimension bf16 values.
  [[nodiscard]] uint32_t indexOffset(const TargetModel::PrefillStep &step,
                                     uint32_t layer) const {
    return indexOffset(step.kvLayers[layer].extent_pages, layer);
  }
  [[nodiscard]] uint32_t indexOffset(const TargetModel::VerifyStep &step,
                                     uint32_t layer) const {
    return indexOffset(step.kvLayers[layer].extent_pages, layer);
  }
  [[nodiscard]] uint32_t indexOffset(uint64_t extentPages,
                                     uint32_t layer) const {
    const uint64_t indexPageBytes =
        uint64_t{kv::kPageTokens} * weights_.layout.indexerHeadDimension * 2;
    return static_cast<uint32_t>(layer * extentPages * indexPageBytes);
  }

  // A layer's MBSA in a prefill chunk: the indexer projections and
  // per-token key store, each sequence's per-group route lists, and the
  // QKV attention over them into the branch rows.
  void addPrefillMbsa(TargetModel::PrefillStep &step,
                      const MaxwellIndexerWeights &indexer,
                      const AttentionMixerWeights &mixer) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const MaxwellStepScratch &h = prefill_;
    const auto &ext =
        std::any_cast<const MaxwellPrefillExtension &>(step.extension);
    const ops::Linear &linear = model_.operators_.linear();
    const TargetLayout &layout = weights_.layout;
    const TargetModelGeometry &geometry = model_.geometry_;
    const uint32_t layer = step.attentionLayer++;
    const uint32_t indexOffset = this->indexOffset(step, layer);
    const uint32_t queryWidth =
        layout.indexerHeads * layout.indexerHeadDimension;
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                         uint32_t count, uint32_t width) {
      return rowsOf<uint16_t>(model_.backend_, buffer, begin, count, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t begin,
                         uint32_t count, uint32_t width) {
      return rowsOf<float>(model_.backend_, buffer, begin, count, width);
    };
    std::vector<uint32_t> firstPositions;
    for (const TargetModelPrefillSequence &sequence : step.sequences)
      firstPositions.push_back(
          static_cast<uint32_t>(sequence.chunk.committed_tokens));
    uint32_t lists = 0;
    for (size_t at = 0; at < step.sequences.size(); ++at) {
      const TargetModelPrefillSequence &sequence = step.sequences[at];
      const ops::MbsaRows mbsa{sequence.rows,
                               std::span(firstPositions).subspan(at, 1),
                               {{&sequence.indexTable, 1}, indexOffset}};
      ops::Mbsa::addProject(
          step.graph,
          u16(b.normalized, sequence.rowBegin, sequence.rows,
              layout.hiddenSize),
          floatPlane(indexer.query, "indexer-query"),
          floatPlane(indexer.key, "indexer-key"), indexer.queryNorm,
          indexer.keyNorm,
          f32(b.ropeCos, sequence.rowBegin, sequence.rows,
              geometry.rotaryPairs),
          f32(b.ropeSin, sequence.rowBegin, sequence.rows,
              geometry.rotaryPairs),
          u16(h.mbsaQueries, sequence.rowBegin, sequence.rows, queryWidth),
          mbsa, layout.hiddenSize);
      const uint32_t tiles = kv::prefillAttentionTiles(sequence.rows);
      const uint32_t activeRows[1] = {sequence.rows};
      ops::Mbsa::addRoute(
          step.graph,
          u16(h.mbsaQueries, sequence.rowBegin, sequence.rows, queryWidth),
          h.routeLists, h.routeCounts, lists, ops::Mbsa::kRouteListEntries,
          tiles, std::span<const uint32_t>(activeRows, 1), mbsa);
      lists += tiles;
    }
    // The attention itself: QKV projection, store, the sparse sweep over
    // each sequence's route lists, the gather into the out-projection's
    // input rows, and the projection into the branch.
    linear.addPrefill(step.graph, b.normalized, mixer.inputProjection,
                      b.fullPacked, b.projectionSums, step.rows,
                      b.linearScratch);
    lists = 0;
    for (size_t at = 0; at < step.sequences.size(); ++at) {
      const TargetModelPrefillSequence &sequence = step.sequences[at];
      const uint64_t headBytes = uint64_t{sequence.attentionStride} *
                                 geometry.attentionHeadDimension *
                                 sizeof(uint16_t);
      const metal::MetalBuffer queries = model_.backend_.view(
          b.fullQueries, sequence.queryOffset,
          uint64_t{geometry.attentionQueryHeads} * headBytes);
      const metal::MetalBuffer attentionRows = model_.backend_.view(
          b.fullAttention, sequence.queryOffset,
          uint64_t{geometry.attentionQueryHeads} * headBytes);
      const metal::MetalBuffer keys = model_.backend_.view(
          b.chunkKeys, sequence.kvOffset,
          uint64_t{geometry.attentionKvHeads} * headBytes);
      const metal::MetalBuffer values = model_.backend_.view(
          b.chunkValues, sequence.kvOffset,
          uint64_t{geometry.attentionKvHeads} * headBytes);
      ops::PagedAttention::addPrefillProjection(
          step.graph, u16(b.fullPacked, sequence.rowBegin, sequence.rows,
                          geometry.packedFullWidth),
          mixer.queryNorm, mixer.keyNorm,
          f32(b.ropeCos, sequence.rowBegin, sequence.rows,
              geometry.rotaryPairs),
          f32(b.ropeSin, sequence.rowBegin, sequence.rows,
              geometry.rotaryPairs),
          queries, keys, values, sequence.rows, sequence.attentionStride,
          geometry.attentionQueryHeads, geometry.kvLayout);
      ops::PagedAttention::addPrefillStore(
          step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
          sequence.chunk, geometry.kvLayout);
      const ops::AttentionBlocks blocks{h.routeLists, h.routeCounts,
                                        ops::Mbsa::kRouteListEntries, lists};
      ops::PagedAttention::addPrefill(
          step.graph, step.kvLayers[layer], queries, attentionRows,
          b.attentionPartials, b.attentionStatistics, sequence.pageTable,
          sequence.chunk, ext.sparseAttention[at], blocks);
      ops::PagedAttention::addPrefillGate(
          step.graph, u16(b.fullPacked, sequence.rowBegin, sequence.rows,
                          geometry.packedFullWidth),
          attentionRows,
          u16(b.attentionHidden, sequence.rowBegin, sequence.rows,
              geometry.attentionWidth),
          sequence.rows, sequence.attentionStride,
          geometry.attentionQueryHeads, geometry.kvLayout, false);
      lists += kv::prefillAttentionTiles(sequence.rows);
    }
    linear.addPrefill(step.graph, b.attentionHidden, mixer.outputProjection,
                      h.branch, b.projectionSums, step.rows,
                      b.linearScratch);
  }

  // The layer's MBSA in a verify step: all lanes in one dispatch of each
  // kernel, the sparse plan's rowTiles covering the route lists.
  void addVerifyMbsa(TargetModel::VerifyStep &step,
                     const MaxwellIndexerWeights &indexer,
                     const AttentionMixerWeights &mixer,
                     const ops::LinearPlan &inputPlan,
                     const ops::PreparedInput &mixInput) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const MaxwellStepScratch &h = verify_;
    const auto &ext =
        std::any_cast<const MaxwellVerifyExtension &>(step.extension);
    const ops::Linear &linear = model_.operators_.linear();
    const TargetModelGeometry &geometry = model_.geometry_;
    const TargetLayout &layout = weights_.layout;
    const uint32_t layer = step.attentionLayer++;
    const uint32_t indexOffset = this->indexOffset(step, layer);
    const uint32_t queryWidth =
        layout.indexerHeads * layout.indexerHeadDimension;
    std::array<uint32_t, ExecutionLimits::maximumBatchWidth> firstPositions{};
    std::array<uint32_t, ExecutionLimits::maximumBatchWidth> activeRows{};
    for (uint32_t lane = 0; lane < step.lanes; ++lane) {
      firstPositions[lane] = step.chunks[lane].committed_tokens;
      activeRows[lane] = step.chunks[lane].chunk_tokens;
    }
    const ops::MbsaRows rows{step.rowCapacity,
                             std::span(firstPositions).first(step.lanes),
                             {std::span(b.indexTables).first(step.lanes),
                              indexOffset}};
    ops::Mbsa::addProject(
        step.graph, b.normalized, floatPlane(indexer.query, "indexer-query"),
        floatPlane(indexer.key, "indexer-key"), indexer.queryNorm,
        indexer.keyNorm,
        rowsOf<float>(model_.backend_, b.ropeCos, 0, step.rows,
                      geometry.rotaryPairs),
        rowsOf<float>(model_.backend_, b.ropeSin, 0, step.rows,
                      geometry.rotaryPairs),
        rowsOf<uint16_t>(model_.backend_, h.mbsaQueries, 0, step.rows,
                         queryWidth),
        rows, layout.hiddenSize);
    ops::Mbsa::addRoute(
        step.graph,
        rowsOf<uint16_t>(model_.backend_, h.mbsaQueries, 0, step.rows,
                         queryWidth),
        h.routeLists, h.routeCounts, /*list*/ 0, ops::Mbsa::kRouteListEntries,
        ext.sparseAttention.rowTiles,
        std::span(activeRows).first(step.lanes), rows);
    const ops::AttentionBlocks blocks{h.routeLists, h.routeCounts,
                                      ops::Mbsa::kRouteListEntries, 0};
    step.graph.beginBakedSpan();
    linear.add(step.graph,
               {.input = b.normalized, .output = b.fullPacked,
                .scratch = b.linearScratch, .prepared = mixInput},
               mixer.inputProjection, inputPlan);
    ops::PagedAttention::addVerifyProjection(
        step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
        b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
        geometry.attentionQueryHeads, geometry.kvLayout, step.lanes,
        step.rowCapacity);
    const ops::LinearPlan outputPlan =
        linear.decodePlan(mixer.outputProjection, step.planLanes,
                          ops::LinearEpilogue::None, nullptr,
                          step.rowCapacity);
    ops::PagedAttention::addVerify(
        step.graph, step.kvLayers[layer],
        {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries,
         b.attentionPartials, b.attentionStatistics, b.fullAttention,
         b.pageTables, b.treeMasks},
        step.chunks, ext.sparseAttention, {}, {}, blocks);
    const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
        step.graph, b.fullPacked, b.fullAttention, b.attentionHidden,
        geometry.attentionQueryHeads, geometry.kvLayout, step.lanes,
        b.linearScratch, outputPlan.input(), /*gated*/ false,
        step.rowCapacity);
    linear.add(step.graph,
               {.input = b.attentionHidden, .output = h.branch,
                .scratch = b.linearScratch, .prepared = hidden},
               mixer.outputProjection, outputPlan);
    step.graph.endBakedSpan();
  }

  // The FFN branch of a prefill layer: the dense SwiGLU of the first
  // denseLayers layers — or the MoE layers' shared expert beside the
  // routed block — reading b.normalized and writing the branch.
  void addPrefillFfn(TargetModel::PrefillStep &step,
                     const TargetLayerWeights &layer,
                     const MaxwellLayerExtras &extra) const {
    const TargetModelPrefillBuffers &b = step.buffers;
    const MaxwellStepScratch &h = prefill_;
    if (const auto *dense = std::get_if<DenseFfnWeights>(&layer.ffn)) {
      model_.operators_.linear().addPrefillSwiGlu(
          step.graph,
          {&dense->gateProjection, &dense->upProjection,
           &dense->downProjection},
          b.ffn(), h.zeros, h.branch, step.rows);
      return;
    }
    const ops::MoeWeights &moe = std::get<ops::MoeWeights>(layer.ffn);
    if (extra.sharedFfn) {
      model_.operators_.linear().addPrefillSwiGlu(
          step.graph,
          {&extra.sharedFfn->gate, &extra.sharedFfn->up,
           &extra.sharedFfn->down},
          b.ffn(), h.zeros, h.shared, step.rows);
    }
    ops::MoE::add(step.graph,
                  {b.normalized, extra.sharedFfn ? h.shared : h.zeros,
                   h.branch, b.moe, {}, {}},
                  moe, *step.moe);
  }

  // A verify step's dense SwiGLU into `output` (no residual epilogue; the
  // mHC expand-mix injects the branch).
  void addVerifySwiGlu(TargetModel::VerifyStep &step,
                       const ops::Projection &gate, const ops::Projection &up,
                       const ops::Projection &down,
                       const ops::LinearPlan &gateUpPlan,
                       const ops::PreparedInput &normOut,
                       metal::MetalBuffer output) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const ops::Linear &linear = model_.operators_.linear();
    step.graph.beginBakedSpan();
    const ops::LinearPlan downPlan =
        linear.decodePlan(down, step.planLanes, ops::LinearEpilogue::Residual,
                          nullptr, step.rowCapacity);
    const bool emitTable =
        gateUpPlan.usesSimdgroup() && downPlan.usesSimdgroup();
    linear.add(step.graph,
               {.input = b.normalized, .output = b.denseIntermediate,
                .gateScratch = b.denseGateScratch,
                .scratch = b.linearScratch, .prepared = normOut,
                .emitTable64 = emitTable},
               up, gateUpPlan, &gate);
    ops::LinearScratch downScratch = b.linearScratch;
    downScratch.input = b.denseIntermediate;
    linear.add(step.graph,
               {.input = b.denseIntermediate, .output = output,
                .residual = verify_.zeros,
                .scratch = emitTable ? downScratch : b.linearScratch,
                .prepared = emitTable
                                ? ops::PreparedInput{b.denseIntermediate,
                                                     ops::LinearInput::Table64Paired}
                                : ops::PreparedInput{}},
               down, downPlan);
    step.graph.endBakedSpan();
  }

  // A verify MoE layer: the shared expert (the same SwiGLU; `sharedPlan`
  // its gate/up decode plan from addVerifyLayer) into h.shared, then the
  // routed block adding into the branch.
  void addVerifyMoe(TargetModel::VerifyStep &step,
                    const TargetLayerWeights &layer,
                    const MaxwellLayerExtras &extra,
                    const std::optional<ops::LinearPlan> &sharedPlan,
                    const ops::PreparedInput &normOut) const {
    const TargetModelVerifyBuffers &b = step.buffers;
    const MaxwellStepScratch &h = verify_;
    const ops::Linear &linear = model_.operators_.linear();
    const ops::MoeWeights &moe = std::get<ops::MoeWeights>(layer.ffn);
    metal::MetalBuffer residual = h.zeros;
    if (extra.sharedFfn) {
      step.graph.beginBakedSpan();
      const ops::LinearPlan &gateUpPlan = *sharedPlan;
      const ops::LinearPlan downPlan =
          linear.decodePlan(extra.sharedFfn->down, step.planLanes,
                            ops::LinearEpilogue::Residual, nullptr,
                            step.rowCapacity);
      const bool emitTable =
          gateUpPlan.usesSimdgroup() && downPlan.usesSimdgroup();
      linear.add(step.graph,
                 {.input = b.normalized, .output = b.denseIntermediate,
                  .gateScratch = b.denseGateScratch,
                  .scratch = b.linearScratch, .prepared = normOut,
                  .emitTable64 = emitTable},
                 extra.sharedFfn->up, gateUpPlan, &extra.sharedFfn->gate);
      ops::LinearScratch downScratch = b.linearScratch;
      downScratch.input = b.denseIntermediate;
      linear.add(step.graph,
                 {.input = b.denseIntermediate, .output = h.shared,
                  .residual = h.zeros,
                  .scratch = emitTable ? downScratch : b.linearScratch,
                  .prepared = emitTable
                                  ? ops::PreparedInput{b.denseIntermediate,
                                                       ops::LinearInput::Table64Paired}
                                  : ops::PreparedInput{}},
                 extra.sharedFfn->down, downPlan);
      step.graph.endBakedSpan();
      residual = h.shared;
    }
    step.graph.beginBakedSpan();
    ops::MoE::add(step.graph, {b.normalized, residual, h.branch, b.moe,
                               {}, b.verifyRows},
                  moe, *step.moe);
    step.graph.endBakedSpan();
  }

  // The HC head of the last layer: the fp32 contraction projection into
  // the logits scratch's front, the gated weights and the weighted sum
  // into the step's hidden output (the head input addHeadBatch reads).
  void addPrefillHead(TargetModel::PrefillStep &step,
                      metal::MetalBuffer output) const {
    const ops::MhcShape shape = mhcShape();
    const uint32_t streams = weights_.layout.hyperConnections;
    ops::Mhc::addProject(step.graph, prefill_.streams,
                         floatPlane(weights_.hcHead.fn,
                                    "maxwell_moe HC head projection"),
                         streams, prefill_.logits, shape, step.rows);
    ops::Mhc::addHeadWeights(step.graph, prefill_.streams, prefill_.logits,
                             weights_.hcHead.base, weights_.hcHead.scale,
                             prefill_.headWeights, shape, step.rows);
    ops::Mhc::addWeightedSum(step.graph, prefill_.streams,
                             prefill_.headWeights, streams, 0, output, shape,
                             step.rows);
  }
  void addVerifyHead(TargetModel::VerifyStep &step,
                     metal::MetalBuffer output) const {
    const ops::MhcShape shape = mhcShape();
    const uint32_t streams = weights_.layout.hyperConnections;
    ops::Mhc::addProject(step.graph, verify_.streams,
                         floatPlane(weights_.hcHead.fn,
                                    "maxwell_moe HC head projection"),
                         streams, verify_.logits, shape, step.rows);
    ops::Mhc::addHeadWeights(step.graph, verify_.streams, verify_.logits,
                             weights_.hcHead.base, weights_.hcHead.scale,
                             verify_.headWeights, shape, step.rows);
    ops::Mhc::addWeightedSum(step.graph, verify_.streams,
                             verify_.headWeights, streams, 0, output, shape,
                             step.rows);
  }

  // The draft's read of a capture layer's output: the stream mean into
  // the capture slot (upstream addStreamCapture), or — for the MTP draft,
  // whose TargetLayout flags captureStreams — the full stream state.
  void addPrefillCapture(TargetModel::PrefillStep &step,
                         uint32_t index) const {
    const auto slot = model_.geometry_.captureSlot(index);
    if (!slot) return;
    const uint32_t width = model_.geometry_.hiddenSize;
    const uint32_t captured = model_.geometry_.capturedHiddenSize();
    for (const TargetModelPrefillSequence &sequence : step.sequences)
      for (uint32_t c = 0; c < sequence.captureCount; ++c) {
        const TargetModelPrefillCapture &capture = sequence.captures[c];
        const metal::MetalBuffer rows = rowsOf<float>(
            model_.backend_, prefill_.streams, capture.sourceStart,
            capture.rows, weights_.layout.streamWidth());
        if (model_.geometry_.captureStreamWidth)
          ops::HyperConnection::addStreamCapture(
              step.graph, rows, step.buffers.captured,
              {capture.destinationStart, captured, *slot * width},
              hyperShape(), capture.rows);
        else
          ops::HyperConnection::addStreamMean(
              step.graph, rows, step.buffers.captured,
              {capture.destinationStart, captured, *slot * width},
              hyperShape(), capture.rows);
      }
  }
  void addVerifyCapture(TargetModel::VerifyStep &step,
                        uint32_t index) const {
    const auto slot = model_.geometry_.captureSlot(index);
    if (!slot) return;
    const uint32_t width = model_.geometry_.hiddenSize;
    const metal::MetalBuffer captured =
        step.tree ? step.buffers.capturedPath
                  : step.buffers.capturedTargetHidden;
    if (model_.geometry_.captureStreamWidth)
      ops::HyperConnection::addStreamCapture(
          step.graph, verify_.streams, captured,
          {0, model_.geometry_.capturedHiddenSize(), *slot * width},
          hyperShape(), step.rows);
    else
      ops::HyperConnection::addStreamMean(
          step.graph, verify_.streams, captured,
          {0, model_.geometry_.capturedHiddenSize(), *slot * width},
          hyperShape(), step.rows);
  }

  const TargetModel &model_;
  const MaxwellMoeWeights &weights_;
  // Declared before the scratches: verify_'s route lists are sized by it
  // (member initialization order).
  const uint32_t verifyTiles_;
  // The step scratch at the prefill chunk's and the verify step's most
  // rows (lane * RICHENGINE_TREE_VERIFY_NODES covers a tree batch too).
  MaxwellStepScratch prefill_;
  MaxwellStepScratch verify_;
};

MaxwellMoeWeights loadMaxwellMoeWeights(metal::MetalBackend &backend,
                                        const TargetLayout &layout,
                                        const TargetFiles &files) {
  requireMaxwellLayout(layout);
  // Packed files (a safetensors install's package) or a maxwell-moe
  // GGUF's images — both vend the same sections; there is no affine or
  // oQ tensor map for this tensor set.
  const auto *packed = std::get_if<PackedTargetFiles>(&files);
  const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files);
  if (!packed && !gguf)
    throw WeightStoreError("maxwell_moe loads from packed or GGUF files");
  const auto source = [&](uint32_t index) -> WeightFile {
    return packed ? packed->layer(index) : gguf->get().layer(index);
  };
  const auto headFile = [&]() -> WeightFile {
    return packed ? packed->head() : gguf->get().head();
  };
  const auto embeddingFile = [&]() -> WeightFile {
    return packed ? packed->embedding() : gguf->get().embedding();
  };
  const BlockTargetFormat format;
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  MaxwellMoeWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  result.extras.reserve(layout.layers);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    WeightFile file = source(index);
    TargetLayerWeights &layer = result.layers.emplace_back();
    MaxwellLayerExtras &extra = result.extras.emplace_back();
    layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
    layer.mixer = readMaxwellMixer(file, format, layout.mixerGeometry());
    extra.indexer = readIndexer(file, format, layout);
    layer.postAttentionNorm =
        format.norm(file, layout.hiddenSize, "post-attention-norm");
    extra.attentionMhc = readMhc(file, format, layout, "mhc-attn");
    if (layout.isMoeLayer(index)) {
      layer.ffn = readMoeBlock(file);
      extra.sharedFfn = readSharedFfn(file, format, layout);
    } else {
      layer.ffn = readDenseFfn(file, format, layout);
    }
    extra.ffnMhc = readMhc(file, format, layout, "mhc-ffn");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = headFile();
    result.hcHead = readHcHead(file, format, layout);
    result.finalNorm = format.norm(file, layout.hiddenSize, "final-norm");
    result.logitsProjection = format.projection(
        file, paddedVocabularySize(layout.vocabularySize), layout.hiddenSize,
        "logits");
    // bf16 logits would round near-ties together: their spacing is 0.125
    // at logits of 16 to 32.
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = embeddingFile();
    result.tokenEmbedding =
        format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

TargetModelGeometry targetModelGeometry(const MaxwellMoeWeights &weights) {
  TargetModelGeometry geometry =
      targetModelGeometry(static_cast<const TargetModelWeights &>(weights));
  // The extras' Linear-dispatched projections the base pass does not see:
  // every layer's indexer and each shared expert's SwiGLU. The mHC
  // mappings and the HC head's contraction are no Linear workloads — the
  // mhc kernels read their float planes directly, and their 24- and
  // 4-row outputs are under a tile anyway.
  const auto include = [&](const ops::Projection &projection) {
    geometry.prefillProjections.push_back(projection.shape());
    geometry.decodeProjections.push_back(projection.shape());
  };
  for (const MaxwellLayerExtras &extra : weights.extras) {
    include(extra.indexer.query);
    include(extra.indexer.key);
    if (extra.sharedFfn) {
      if (extra.sharedFfn->gate.shape() != extra.sharedFfn->up.shape())
        throw WeightStoreError(
            "fused gate/up projections must have matching shapes and layouts");
      include(extra.sharedFfn->gate);
      include(extra.sharedFfn->up);
      include(extra.sharedFfn->down);
      geometry.gateUpProjections.push_back(extra.sharedFfn->up.shape());
    }
  }
  for (auto *shapes :
       {&geometry.prefillProjections, &geometry.decodeProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()),
                  shapes->end());
  }
  return geometry;
}

std::unique_ptr<TargetLayerEncoder>
makeMaxwellLayerEncoder(const TargetModel &model) {
  return std::make_unique<MaxwellLayerEncoder>(model);
}

} // namespace richengine::model

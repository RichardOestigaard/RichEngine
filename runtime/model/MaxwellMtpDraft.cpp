// The maxwell-moe internal MTP draft (model/MaxwellMtpDraft.hpp): its
// fifteen chained passes over the target's captured multistream state,
// the context commit's retained-row K/V writes and the prompt prefill's
// rolled-token context, all through the dedicated mtp.metal kernels plus
// the shared Linear, mHC, MoE and norm operators.

#include "model/MaxwellMtpDraft.hpp"

#include "Env.hpp"
#include "model/Model.hpp"
#include "metal/abi/Mtp.h"
#include "metal/abi/Sampling.h"
#include "model/TargetLoader.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/Embedding.hpp"
#include "ops/RowCopy.hpp"
#include "ops/KernelNames.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace richengine::model {
namespace {

constexpr uint32_t kHidden = 2048;
constexpr uint32_t kStreams = 4;
// The decode linear's activation tile (ops/Linear.cpp
// kDecodeActivationRows): every draft-decode projection runs an
// eight-row plan whose pad rows carry garbage.
constexpr uint32_t kDecodeTileRows = 8;
constexpr uint32_t kMixWidth = 24;
// The ring slot s holds the pair (h_s, x_{s+1}): a decode pass k commits
// slot positions[lane] - 1 + k, a commit row r slot startPosition + r and
// a prefill row its own captured position.

// The single fp32 plane of an mHC mapping or HC head projection.
metal::MetalBuffer floatPlane(const ops::Projection &projection,
                              const char *what) {
  const auto &segments = projection.blocks().segments;
  if (segments.size() != 1 || !segments.front().isFloat())
    throw std::invalid_argument(what);
  return segments.front().plane0;
}

// One hyper-connection module (the MTP attention's or FFN's): the fp32
// mapping [mhcMixWidth][streamWidth], the three gating factors and the
// mhcMixWidth bias, all F32 sections.
MaxwellMhcWeights readMtpMhc(WeightFile &file,
                             const BlockTargetFormat &format,
                             std::string_view label) {
  MaxwellMhcWeights mhc;
  mhc.mapping = format.projection(file, kMixWidth, kStreams * kHidden, label);
  mhc.mapping.destination = ops::FloatOutput::Float32;
  mhc.alphas = file.section(3 * sizeof(float), label);
  mhc.bias = file.section(uint64_t{kMixWidth} * sizeof(float), label);
  return mhc;
}

} // namespace

MaxwellMtpDraftWeights
loadMaxwellMtpDraftWeights(metal::MetalBackend &backend,
                           const DFlashDraftLayout &layout, WeightFile mtp,
                           const MaxwellMoeWeights &target,
                           uint64_t allocationBaseline) {
  if (layout.kind != DraftKind::Mtp || layout.hiddenSize != kHidden ||
      layout.kvHeads != target.layout.attentionKvHeads ||
      layout.attentionSize != target.layout.attentionWidth ||
      layout.targetHiddenSize != target.layout.streamWidth())
    throw WeightStoreError("maxwell MTP draft layout mismatch");
  const BlockTargetFormat format;
  MaxwellMtpDraftWeights result;
  result.layout = layout;
  result.enorm = format.norm(mtp, kHidden, "mtp-enorm");
  result.hnorm = format.norm(mtp, kHidden, "mtp-hnorm");
  result.ehProjection =
      format.projection(mtp, kHidden, 2 * kHidden, "mtp-eh-proj");
  result.inputNorm = format.norm(mtp, kHidden, "mtp-attn-norm");
  result.queryProjection = format.projection(
      mtp, layout.attentionSize, kHidden, "mtp-attn-q");
  const uint32_t kv = layout.kvHeads * layout.attentionHeadDimension;
  result.keyProjection = format.projection(mtp, kv, kHidden, "mtp-attn-k");
  result.valueProjection = format.projection(mtp, kv, kHidden, "mtp-attn-v");
  result.queryNorm = format.norm(mtp, layout.attentionHeadDimension,
                                 "mtp-attn-q-norm");
  result.keyNorm = format.norm(mtp, layout.attentionHeadDimension,
                               "mtp-attn-k-norm");
  result.outputProjection = format.projection(
      mtp, kHidden, layout.attentionSize, "mtp-attn-output");
  result.ffnNorm = format.norm(mtp, kHidden, "mtp-ffn-norm");
  result.attentionMhc = readMtpMhc(mtp, format, "mtp-hc-attn");
  {
    ops::BlockMoeWeights block;
    block.router = readQuantizedSegment(mtp, "mtp-router");
    block.expertBias = readQuantizedSegment(mtp, "mtp-expert-bias");
    block.gate.routed = readQuantizedSegment(mtp, "mtp-experts-gate");
    block.up.routed = readQuantizedSegment(mtp, "mtp-experts-up");
    block.down.routed = readQuantizedSegment(mtp, "mtp-experts-down");
    result.moe = ops::MoeWeights{std::move(block)};
  }
  result.sharedFfn = {
      format.projection(mtp, target.layout.sharedIntermediateSize, kHidden,
                        "mtp-shared-gate"),
      format.projection(mtp, target.layout.sharedIntermediateSize, kHidden,
                        "mtp-shared-up"),
      format.projection(mtp, kHidden, target.layout.sharedIntermediateSize,
                        "mtp-shared-down")};
  result.ffnMhc = readMtpMhc(mtp, format, "mtp-hc-ffn");
  result.hcHead.fn = format.projection(mtp, kStreams, kStreams * kHidden,
                                       "mtp-hc-head");
  result.hcHead.fn.destination = ops::FloatOutput::Float32;
  result.hcHead.base = mtp.section(kStreams * sizeof(float), "mtp-hc-head");
  result.hcHead.scale = mtp.section(sizeof(float), "mtp-hc-head");
  result.outputNorm = format.norm(mtp, kHidden, "mtp-layer-output-norm");
  result.tokenEmbedding = target.tokenEmbedding;
  result.moeShape = {kHidden, target.layout.experts,
                     target.layout.expertsPerToken,
                     target.layout.expertIntermediateSize,
                     ops::WeightLayout::Block32, GGUF_FMT_COUNT, false};
  mtp.finish();
  result.files.push_back(mtp.record());
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

MaxwellMtpDraft::MaxwellMtpDraft(MaxwellMtpDraftWeights weights,
                                 metal::MetalBackend &backend,
                                 const ops::ExecutionPlans &operators)
    : backend_(backend), operators_(operators), weights_(std::move(weights)),
      mhcShape_{kHidden, kStreams, 20} {
  if (!weights_.moeShape.valid())
    throw std::invalid_argument("invalid maxwell MTP MoE shape");
}

void MaxwellMtpDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths,
    std::span<const uint32_t> passLimits) const {
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  const DFlashDraftLayout &layout = weights_.layout;
  // RICHENGINE_MTP_PASSES caps the chain for offline bisection: with 1,
  // every pass scratch in the dump holds the first pass's state.
  static const uint32_t envPasses =
      envUint("RICHENGINE_MTP_PASSES", ~uint32_t{0});
  const uint32_t envCap = std::min(layout.proposalLimit(), envPasses);
  if (!passLimits.empty() && passLimits.size() < lanes)
    throw std::invalid_argument("invalid MTP pass-limit batch");
  const uint32_t stride = layout.verifyRows();
  const uint32_t logitsStride =
      paddedVocabularySize(layout.vocabularySize);
  const uint32_t mix = mhcShape_.mixWidth();
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || !envCap)
    throw std::invalid_argument("invalid MTP draft decode batch");
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  const metal::MetalBuffer attnPlane =
      floatPlane(weights_.attentionMhc.mapping, "mtp attention mHC");
  const metal::MetalBuffer ffnPlane =
      floatPlane(weights_.ffnMhc.mapping, "mtp ffn mHC");
  const metal::MetalBuffer headPlane =
      floatPlane(weights_.hcHead.fn, "mtp HC head");
  const uint32_t ring = layout.draftWindow();
  const uint32_t inter = layout.intermediateSize;

  // The decode linear's tile consumes eight activation rows. The MTP
  // chain produces one real row per lane per pass, so each lane's
  // scratch slice is eight rows deep — the tile's pad rows read and
  // write garbage the downstream kernels never touch — and every lane
  // runs its own section, its kernels and shared ops all seeing one
  // row. A lane's shared-op tables land in the reused linear scratch in
  // front of the linear that consumes them.
  const auto rows8 = [&](metal::MetalBuffer buffer, uint32_t lane,
                         uint64_t elements) {
    return backend_.view(buffer, uint64_t{lane} * 8 * elements * 2,
                         uint64_t{8} * elements * 2);
  };
  const auto rows8f = [&](metal::MetalBuffer buffer, uint32_t lane,
                          uint64_t elements) {
    return backend_.view(buffer,
                         uint64_t{lane} * 8 * elements * sizeof(float),
                         uint64_t{8} * elements * sizeof(float));
  };
  const auto half = [&](metal::MetalBuffer buffer, uint32_t lane,
                        uint64_t elements) {
    return backend_.view(buffer, uint64_t{lane} * elements * 2,
                         elements * 2);
  };
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const metal::MetalBuffer tokenIn =
        backend_.view(buffers.mtp.tokenIn, lane * sizeof(uint32_t),
                      sizeof(uint32_t));
    const metal::MetalBuffer inputTokens =
        backend_.view(buffers.inputTokens,
                      uint64_t{lane} * layout.queryRows() * sizeof(uint32_t),
                      uint64_t{layout.queryRows()} * sizeof(uint32_t));
    // The packed DraftPositions stride is the verify-row count per lane.
    const metal::MetalBuffer positions =
        backend_.view(buffers.positions,
                      uint64_t{lane} * stride * sizeof(uint32_t),
                      sizeof(uint32_t));
    const metal::MetalBuffer embed = half(buffers.mtp.embed, lane, kHidden);
    const metal::MetalBuffer embedNormed =
        half(buffers.mtp.embedNormed, lane, kHidden);
    const metal::MetalBuffer eh = rows8(buffers.mtp.eh, lane, 2 * kHidden);
    const metal::MetalBuffer ehOut = rows8(buffers.mtp.ehOut, lane, kHidden);
    const metal::MetalBuffer normed =
        rows8(buffers.mtp.normed, lane, kHidden);
    const metal::MetalBuffer queries =
        rows8(buffers.mtp.queries, lane, layout.attentionSize);
    const metal::MetalBuffer keys =
        rows8(buffers.mtp.keys, lane, layout.contextKvSize() / 2);
    const metal::MetalBuffer values =
        rows8(buffers.mtp.values, lane, layout.contextKvSize() / 2);
    const metal::MetalBuffer attention =
        rows8(buffers.mtp.attention, lane, layout.attentionSize);
    const metal::MetalBuffer branch =
        rows8(buffers.mtp.branch, lane, kHidden);
    const metal::MetalBuffer shared =
        rows8(buffers.mtp.shared, lane, kHidden);
    const metal::MetalBuffer intermediate =
        rows8(buffers.mtp.intermediate, lane, inter);
    const metal::MetalBuffer head = rows8(buffers.mtp.head, lane, kHidden);
    const metal::MetalBuffer streams = backend_.view(
        buffers.mtp.streams,
        uint64_t{lane} * kStreams * kHidden * sizeof(float),
        uint64_t{kStreams} * kHidden * sizeof(float));
    const metal::MetalBuffer plane = backend_.view(
        buffers.mtp.plane, uint64_t{lane} * mix * sizeof(float),
        uint64_t{mix} * sizeof(float));
    const metal::MetalBuffer mappings = backend_.view(
        buffers.mtp.mappings, uint64_t{lane} * mix * sizeof(float),
        uint64_t{mix} * sizeof(float));
    const metal::MetalBuffer headWeights = backend_.view(
        buffers.mtp.headWeights, uint64_t{lane} * mix * sizeof(float),
        uint64_t{mix} * sizeof(float));
    // The table-writing norms read a whole eight-row slice: row 0 is
    // the mHC aggregate, the rest pad.
    const metal::MetalBuffer aggregate =
        rows8(buffers.mtp.aggregate, lane, kHidden);
    const metal::MetalBuffer anchor = buffers.mtpAnchors[lane];
    const ops::MoePlan moePlan =
        operators_.moeDecode(weights_.moeShape, 1, 1);

    // Pass 0's token is the packed draft input's anchor row; later
    // passes chain on mtp_argmax's greedy pick.
    graph.add("mtp_gather_tokens", {inputTokens, tokenIn},
              MtpGatherTokensParams{0, layout.queryRows(), 0, 0, 1},
              {1, 1, 1}, {32, 1, 1});
    // The lane's chain depth: the acceptance budget when the caller
    // supplies one, otherwise the full layout limit — envCap always wins.
    const uint32_t passes =
        std::min(envCap, passLimits.empty() ? layout.proposalLimit()
                                            : std::max(passLimits[lane], 1u));
    for (uint32_t pass = 0; pass < passes; ++pass) {
      ops::Embedding::add(graph, tokenIn, weights_.tokenEmbedding, embed, 1);
      ops::Normalization::addRms(graph, embed, weights_.enorm, embedNormed,
                                 kHidden, 1);
      if (!pass) {
        // Source 1 reads the lane's anchor row — every row is the a0
        // slot on a one-row dispatch, so the anchor binds four times.
        graph.add("mtp_ehtoken",
                  {eh, embedNormed, weights_.hnorm.buffer, anchor, anchor,
                   anchor, anchor, eh},
                  MtpEhTokenParams{kHidden, kStreams, 1, 1, 0},
                  {1, kStreams, 1}, {1024, 1, 1});
      } else {
        graph.add("mtp_ehtoken_chain",
                  {streams, embedNormed, weights_.hnorm.buffer, eh},
                  MtpEhTokenParams{kHidden, kStreams, 1, 0, 0},
                  {1, kStreams, 1}, {1024, 1, 1});
      }
      linear.add(graph,
                 {.input = eh, .output = ehOut, .scratch = scratch},
                 weights_.ehProjection,
                 linear.decodePlan(weights_.ehProjection, 1,
                                   ops::LinearEpilogue::None, nullptr,
                                   kDecodeTileRows));
      graph.add("mtp_stream_init", {ehOut, streams},
                MtpStreamInitParams{kStreams * kHidden, 1},
                {(uint64_t{kStreams} * kHidden + 255) / 256, 1, 1},
                {256, 1, 1});
      ops::Mhc::addModule(graph, streams, attnPlane,
                          weights_.attentionMhc.alphas,
                          weights_.attentionMhc.bias, mappings, aggregate,
                          mhcShape_, 1);
      const ops::LinearPlan qPlan =
          linear.decodePlan(weights_.queryProjection, 1,
                            ops::LinearEpilogue::None, nullptr,
                            kDecodeTileRows);
      const ops::LinearPlan kPlan =
          linear.decodePlan(weights_.keyProjection, 1,
                            ops::LinearEpilogue::None, nullptr,
                            kDecodeTileRows);
      const ops::LinearPlan vPlan =
          linear.decodePlan(weights_.valueProjection, 1,
                            ops::LinearEpilogue::None, nullptr,
                            kDecodeTileRows);
      const ops::LinearInput qkvInput = qPlan.input();
      const bool uniform = kPlan.input() == qkvInput &&
                           vPlan.input() == qkvInput;
      const ops::PreparedInput prepared = ops::Normalization::addRms(
          graph, aggregate, weights_.inputNorm, normed, kHidden,
          kDecodeTileRows, scratch, qkvInput,
          !uniform || qkvInput == ops::LinearInput::Plain);
      linear.add(graph,
                 {.input = normed, .output = queries, .scratch = scratch,
                  .prepared = prepared},
                 weights_.queryProjection, qPlan);
      linear.add(graph,
                 {.input = normed, .output = keys, .scratch = scratch,
                  .prepared = prepared},
                 weights_.keyProjection, kPlan);
      linear.add(graph,
                 {.input = normed, .output = values, .scratch = scratch,
                  .prepared = prepared},
                 weights_.valueProjection, vPlan);
      MtpKvStoreParams kv{};
      kv.rows = 1;
      kv.lanes = 1;
      kv.heads = layout.attentionSize / layout.attentionHeadDimension;
      kv.kvHeads = layout.kvHeads;
      kv.headDim = layout.attentionHeadDimension;
      kv.rotaryPairs = layout.attentionHeadDimension / 2;
      kv.ropeTheta = layout.rotaryTheta;
      kv.ring = ring;
      kv.mode = 0;
      kv.posStride = stride;
      kv.pass = pass;
      kv.rmsEps = weights_.queryNorm.rmsEpsilon;
      graph.add("mtp_kv_store",
                {queries, keys, values, weights_.queryNorm.buffer,
                 weights_.keyNorm.buffer, positions, positions, queries,
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane]},
                kv, {1, uint32_t{kv.heads + kv.kvHeads}, 1}, {32, 1, 1});
      graph.add("mtp_attention",
                {queries, buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane], positions, attention},
                MtpAttentionParams{1, kv.heads, kv.kvHeads, kv.headDim,
                                   ring, stride, pass},
                {1, kv.heads, 1}, {32, 1, 1});
      linear.add(graph,
                 {.input = attention, .output = branch,
                  .scratch = scratch},
                 weights_.outputProjection,
                 linear.decodePlan(weights_.outputProjection, 1,
                                   ops::LinearEpilogue::None, nullptr,
                                   kDecodeTileRows));
      ops::Mhc::addExpandMix(graph, streams, branch, mappings, mhcShape_,
                             1);
      ops::Mhc::addModule(graph, streams, ffnPlane, weights_.ffnMhc.alphas,
                          weights_.ffnMhc.bias, mappings, aggregate,
                          mhcShape_, 1);
      const ops::LinearPlan upPlan =
          linear.decodePlan(weights_.sharedFfn.up, 1,
                            ops::LinearEpilogue::GateUp,
                            &weights_.sharedFfn.gate, kDecodeTileRows);
      const ops::LinearPlan downPlan =
          linear.decodePlan(weights_.sharedFfn.down, 1,
                            ops::LinearEpilogue::None, nullptr,
                            kDecodeTileRows);
      const bool emitTable =
          upPlan.usesSimdgroup() && downPlan.usesSimdgroup();
      const ops::PreparedInput ffnPrepared = ops::Normalization::addRms(
          graph, aggregate, weights_.ffnNorm, normed, kHidden,
          kDecodeTileRows, scratch, upPlan.input(),
          upPlan.input() == ops::LinearInput::Plain);
      ops::PreparedInput interPlan = linear.add(
          graph,
          {.input = normed, .output = intermediate,
           .gateScratch = buffers.gateScratch, .scratch = scratch,
           .prepared = ffnPrepared, .emitTable64 = emitTable},
          weights_.sharedFfn.up, upPlan, &weights_.sharedFfn.gate);
      static_cast<void>(interPlan);
      ops::LinearScratch downScratch = scratch;
      downScratch.input = intermediate;
      linear.add(graph,
                 {.input = intermediate, .output = shared,
                  .scratch = emitTable ? downScratch : scratch,
                  .prepared = emitTable
                                  ? ops::PreparedInput{
                                        intermediate,
                                        ops::LinearInput::Table64Paired}
                                  : ops::PreparedInput{}},
                 weights_.sharedFfn.down, downPlan);
      ops::MoE::add(graph, {normed, shared, branch, buffers.mtp.moe, {}, {}},
                    weights_.moe, moePlan);
      ops::Mhc::addExpandMix(graph, streams, branch, mappings, mhcShape_,
                             1);
      ops::Mhc::addProject(graph, streams, headPlane, kStreams, plane,
                           mhcShape_, 1);
      ops::Mhc::addHeadWeights(graph, streams, plane, weights_.hcHead.base,
                               weights_.hcHead.scale, headWeights,
                               mhcShape_, 1);
      ops::Mhc::addWeightedSum(graph, streams, headWeights, kStreams, 0,
                               head, mhcShape_, 1);
      ops::Normalization::addRms(graph, head, weights_.outputNorm, normed,
                                 kHidden, 1);
      // The head's eight-row tile lands in the lane's logits scratch;
      // row 0 is the pass's logits, copied into its selector slot
      // (fp32 rows copied as pairs of bf16 columns).
      const metal::MetalBuffer logitsWide =
          rows8f(buffers.mtp.logits, lane, logitsStride);
      linear.add(graph,
                 {.input = normed, .output = logitsWide,
                  .scratch = scratch},
                 vocabularyProjection,
                 linear.decodePlan(vocabularyProjection, 1,
                                   ops::LinearEpilogue::None, nullptr,
                                   kDecodeTileRows));
      const metal::MetalBuffer logits = backend_.view(
          buffers.logits,
          (uint64_t{pass} * lanes + lane) * logitsStride * sizeof(float),
          uint64_t{logitsStride} * sizeof(float));
      ops::RowCopy::add(graph, logitsWide, {.row = 0, .stride = 2 * logitsStride},
                        logits, {.row = 0, .stride = 2 * logitsStride}, 1,
                        2 * logitsStride);
      graph.add("mtp_argmax", {logits, tokenIn},
                MtpArgmaxParams{1, layout.vocabularySize, logitsStride},
                {1, 1, 1}, {256, 1, 1});
    }
  }
}

void MaxwellMtpDraft::addSelection(
    metal::CommandGraph &graph, const ops::DraftSelectorBuffers &buffers,
    std::span<const uint32_t> anchors,
    std::span<const ops::SamplingPolicy> policies,
    uint32_t treeMask) const {
  static_cast<void>(treeMask); // The MTP draft emits a chain, never a tree.
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid MTP selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = weights_.layout.queryRows();
  params.vocabulary = weights_.layout.vocabularySize;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  constexpr uint32_t kPositions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr uint32_t kShards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  graph.add("mtp_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            MtpTop16Params{weights_.layout.vocabularySize,
                           paddedVocabularySize(weights_.layout.vocabularySize),
                           lanes},
            {uint64_t{lanes} * kPositions * kShards, 1, 1}, {256, 1, 1});
  graph.add(std::string(ops::kDraftSelectPlain),
            {buffers.partialIds, buffers.partialValues, buffers.uniforms,
             buffers.candidates, buffers.proposalProbabilities,
             buffers.proposedTokens, buffers.confidence},
            params, {uint64_t{lanes} * kPositions, 1, 1}, {32, 1, 1});
}

void MaxwellMtpDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions) const {
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t stride = layout.verifyRows();
  const uint32_t rows = lanes * stride;
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid MTP context commit");
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  const metal::MetalBuffer attnPlane =
      floatPlane(weights_.attentionMhc.mapping, "mtp attention mHC");

  // The retained rows' eh tokens: the verify step's output tokens are the
  // rows' x_{s+1} — captured row r pairs with output token r.
  ops::Embedding::add(graph, buffers.outputTokens, weights_.tokenEmbedding,
                      buffers.mtp.embed, rows);
  ops::Normalization::addRms(graph, buffers.mtp.embed, weights_.enorm,
                             buffers.mtp.embedNormed, kHidden, rows);
  // Anchors for lanes beyond the batch alias lane 0: the kernel only
  // reads a row's own lane, so the extra bindings are never touched.
  const auto anchor = [&](uint32_t lane) {
    return buffers.mtpAnchors[std::min(lane, lanes - 1)];
  };
  graph.add("mtp_ehtoken",
            {buffers.capturedTargetHidden, buffers.mtp.embedNormed,
             weights_.hnorm.buffer, anchor(0), anchor(1), anchor(2),
             anchor(3), buffers.mtp.eh},
            MtpEhTokenParams{kHidden, kStreams, rows, 0, 0},
            {rows, kStreams, 1}, {1024, 1, 1});
  // The eh rows are stream-expanded — each verify row's four [e|h]
  // halves — so a lane's stride * kStreams projection rows stand in the
  // buffer where one decode plan covers stride. Run a tile per
  // (lane, stream block) pair: each lane's 32 eh rows are contiguous.
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t tile = 0; tile < kStreams; ++tile) {
      const uint64_t block = uint64_t{lane} * stride * kStreams +
                             uint64_t{tile} * stride;
      const metal::MetalBuffer ehIn = backend_.view(
          buffers.mtp.eh, block * 2 * kHidden * sizeof(uint16_t),
          uint64_t{stride} * 2 * kHidden * sizeof(uint16_t));
      const metal::MetalBuffer ehTile = backend_.view(
          buffers.mtp.ehOut, block * kHidden * sizeof(uint16_t),
          uint64_t{stride} * kHidden * sizeof(uint16_t));
      linear.add(graph,
                 {.input = ehIn, .output = ehTile, .scratch = scratch},
                 weights_.ehProjection,
                 linear.decodePlan(weights_.ehProjection, 1,
                                   ops::LinearEpilogue::None, nullptr,
                                   stride));
    }
  graph.add("mtp_stream_init",
            {buffers.mtp.ehOut, buffers.mtp.streams},
            MtpStreamInitParams{kStreams * kHidden, rows},
            {(uint64_t{rows} * kStreams * kHidden + 255) / 256, 1, 1},
            {256, 1, 1});
  ops::Mhc::addModule(graph, buffers.mtp.streams, attnPlane,
                      weights_.attentionMhc.alphas,
                      weights_.attentionMhc.bias, buffers.mtp.mappings,
                      buffers.projected, mhcShape_, rows);
  const ops::LinearPlan kPlan =
      linear.decodePlan(weights_.keyProjection, lanes,
                        ops::LinearEpilogue::None, nullptr, stride);
  const ops::LinearPlan vPlan =
      linear.decodePlan(weights_.valueProjection, lanes,
                        ops::LinearEpilogue::None, nullptr, stride);
  const ops::LinearInput kvInput = kPlan.input();
  const ops::PreparedInput prepared = ops::Normalization::addRms(
      graph, buffers.projected, weights_.inputNorm, buffers.hidden, kHidden,
      rows, scratch, kvInput,
      vPlan.input() != kvInput || kvInput == ops::LinearInput::Plain);
  linear.add(graph,
             {.input = buffers.hidden, .output = buffers.mtp.keys,
              .scratch = scratch, .prepared = prepared},
             weights_.keyProjection, kPlan);
  linear.add(graph,
             {.input = buffers.hidden, .output = buffers.mtp.values,
              .scratch = scratch, .prepared = prepared},
             weights_.valueProjection, vPlan);
  MtpKvStoreParams kv{};
  kv.rows = rows;
  kv.lanes = lanes;
  kv.heads = layout.attentionSize / layout.attentionHeadDimension;
  kv.kvHeads = layout.kvHeads;
  kv.headDim = layout.attentionHeadDimension;
  kv.rotaryPairs = layout.attentionHeadDimension / 2;
  kv.ropeTheta = layout.rotaryTheta;
  kv.ring = layout.draftWindow();
  kv.mode = 1;
  kv.stride = stride;
  kv.rmsEps = weights_.keyNorm.rmsEpsilon;
  std::copy_n(startPositions.begin(), lanes, kv.startPositions);
  graph.add("mtp_kv_store",
            {buffers.mtp.keys, buffers.mtp.keys, buffers.mtp.values,
             weights_.queryNorm.buffer, weights_.keyNorm.buffer,
             buffers.retainedCounts, buffers.retainedCounts,
             buffers.mtp.keys,
             buffers.persistentKeys[0][0], buffers.persistentKeys[0][1],
             buffers.persistentKeys[0][2], buffers.persistentKeys[0][3],
             buffers.persistentValues[0][0], buffers.persistentValues[0][1],
             buffers.persistentValues[0][2], buffers.persistentValues[0][3]},
            kv, {rows, uint32_t{kv.kvHeads}, 1}, {32, 1, 1});
  // The pass-0 anchor of the next decode: each lane's last retained
  // capture row.
  graph.add("mtp_anchor_commit",
            {buffers.capturedTargetHidden, buffers.retainedCounts,
             anchor(0), anchor(1), anchor(2), anchor(3)},
            MtpAnchorParams{kStreams * kHidden, stride, lanes},
            {lanes, 1, 1}, {256, 1, 1});
}

void MaxwellMtpDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers,
    uint32_t capturedRows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!capturedRows || spans.empty())
    throw std::invalid_argument("invalid MTP context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t ring = layout.draftWindow();
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  const metal::MetalBuffer attnPlane =
      floatPlane(weights_.attentionMhc.mapping, "mtp attention mHC");
  for (const DFlashPrefillSpan &span : spans) {
    for (uint32_t begin = 0; begin < span.rows; begin += kMtpPrefillChunk) {
      const uint32_t rows =
          std::min(kMtpPrefillChunk, span.rows - begin);
      const uint32_t rowBase = span.compactRow + begin;
      // The rows' rolled prompt tokens: row r's pair token is
      // input[inputRow + r + 1], clamped at the sequence's end.
      graph.add("mtp_gather_tokens",
                {buffers.inputTokens, buffers.mtp.tokens},
                MtpGatherTokensParams{2, 0, span.inputRow + begin,
                                      span.inputEnd, rows},
                {(rows + 31) / 32, 1, 1}, {32, 1, 1});
      ops::Embedding::add(graph, buffers.mtp.tokens,
                          weights_.tokenEmbedding, buffers.mtp.embed,
                          rows);
      ops::Normalization::addRms(graph, buffers.mtp.embed, weights_.enorm,
                                 buffers.mtp.embedNormed, kHidden, rows);
      const metal::MetalBuffer captured = backend_.view(
          buffers.capturedTargetHidden,
          uint64_t{rowBase} * kStreams * kHidden * sizeof(uint16_t),
          uint64_t{rows} * kStreams * kHidden * sizeof(uint16_t));
      graph.add("mtp_ehtoken",
                {captured, buffers.mtp.embedNormed, weights_.hnorm.buffer,
                 captured, captured, captured, captured, buffers.mtp.eh},
                MtpEhTokenParams{kHidden, kStreams, rows, 0, 0},
                {rows, kStreams, 1}, {1024, 1, 1});
      linear.addPrefill(graph, buffers.mtp.eh, weights_.ehProjection,
                        buffers.mtp.ehOut, {}, rows * kStreams, scratch);
      graph.add("mtp_stream_init",
                {buffers.mtp.ehOut, buffers.mtp.streams},
                MtpStreamInitParams{kStreams * kHidden, rows},
                {(uint64_t{rows} * kStreams * kHidden + 255) / 256, 1, 1},
                {256, 1, 1});
      ops::Mhc::addModule(graph, buffers.mtp.streams, attnPlane,
                          weights_.attentionMhc.alphas,
                          weights_.attentionMhc.bias, buffers.mtp.mappings,
                          buffers.projected, mhcShape_, rows);
      ops::Normalization::addRms(graph, buffers.projected,
                                 weights_.inputNorm, buffers.hidden, kHidden,
                                 rows, scratch);
      linear.addPrefill(graph, buffers.hidden, weights_.keyProjection,
                        buffers.mtp.keys, {}, rows, scratch);
      linear.addPrefill(graph, buffers.hidden, weights_.valueProjection,
                        buffers.mtp.values, {}, rows, scratch);
      graph.add("mtp_kv_store_prefill",
                {buffers.mtp.keys, buffers.mtp.values,
                 weights_.keyNorm.buffer, span.ring[0].keys,
                 span.ring[0].values},
                MtpPrefillKvParams{rows, layout.kvHeads,
                                   layout.attentionHeadDimension,
                                   layout.attentionHeadDimension / 2,
                                   layout.rotaryTheta, ring,
                                   span.startPosition + begin,
                                   span.inputEnd - span.inputRow +
                                       span.startPosition,
                                   weights_.keyNorm.rmsEpsilon},
                {rows, layout.kvHeads, 1}, {32, 1, 1});
      // A tail span's last captured row anchors the state lane's first
      // decode pass.
      if (span.anchorRow != ~0u && begin + rows > span.anchorRow - rowBase) {
        const uint32_t anchor = span.anchorRow;
        if (anchor >= rowBase && anchor < rowBase + rows) {
          const metal::MetalBuffer row = backend_.view(
              buffers.capturedTargetHidden,
              uint64_t{anchor} * kStreams * kHidden * sizeof(uint16_t),
              uint64_t{kStreams} * kHidden * sizeof(uint16_t));
          graph.add("mtp_anchor_prefill", {row, span.anchor},
                    MtpAnchorParams{kStreams * kHidden, 0, 1},
                    {1, 1, 1}, {256, 1, 1});
        }
      }
    }
  }
}

} // namespace richengine::model

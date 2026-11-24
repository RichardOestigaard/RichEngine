// The nemotron internal MTP draft (model/NemotronMtpDraft.hpp): its
// chained passes over the target's captured hidden, the context commit's
// retained-row K/V writes and the prompt prefill's rolled-token context.
// The eh, attention and output projections run the F32 float segments of
// mtp.bin through the shared Linear operator; the NoPE K/V stores, the
// sigmoid MoE's route/relu-squared-up/down/combine and the eh-row norms
// run the dedicated nmtp.metal kernels; the gather, attention flash,
// stream store, top-16 select, argmax and anchor commits reuse
// mtp.metal's.

#include "model/NemotronMtpDraft.hpp"

#include "Env.hpp"
#include "model/Model.hpp"
#include "metal/abi/Mtp.h"
#include "metal/abi/Nmtp.h"
#include "metal/abi/Sampling.h"
#include "model/DraftCheckpoint.hpp"
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

constexpr uint32_t kHidden = 2688;
constexpr uint32_t kExperts = 128;
constexpr uint32_t kTopK = 6;
constexpr uint32_t kIntermediate = 1856;
constexpr uint32_t kSharedIntermediate = 3712;
// The up intermediate's per-slot stride: the shared expert's 3712 rounds
// up to a 256-aligned plane.
constexpr uint32_t kInterStride = 4096;
constexpr uint32_t kMoeSlots = kTopK + 1;
// The decode linear's activation tile (ops/Linear.cpp
// kDecodeActivationRows): every draft-decode projection runs an
// eight-row plan whose pad rows carry garbage.
constexpr uint32_t kDecodeTileRows = 8;
// The ring slot s holds the pair (h_s, x_{s+1}): a decode pass k commits
// slot positions[lane] - 1 + k, a commit row r slot startPosition + r and
// a prefill row its own captured position.

WeightFile mtpFile(const NemotronMtpDraftFiles &files) {
  if (const auto *checkpoint =
          std::get_if<std::reference_wrapper<NemotronMtpCheckpointLoader>>(&files))
    return checkpoint->get().mtp();
  return std::get<PackedNemotronMtpFiles>(files).mtp();
}

NemotronMtpDraftWeights readDraft(metal::MetalBackend &backend,
                                  const NemotronMtpDraftFiles &files,
                                  const DFlashDraftLayout &layout,
                                  const TargetModelWeights &target) {
  // The descriptor declares the nextn block DraftKind::Mtp today; the
  // dedicated NemotronMtp value stays an accepted alias should the
  // dispatch want it.
  if ((layout.kind != DraftKind::Mtp &&
       layout.kind != DraftKind::NemotronMtp) ||
      layout.hiddenSize != kHidden ||
      layout.kvHeads != target.layout.attentionKvHeads ||
      layout.attentionSize != target.layout.attentionWidth ||
      layout.attentionHeadDimension != target.layout.attentionHeadDimension ||
      layout.vocabularySize != target.layout.vocabularySize ||
      layout.targetHiddenSize != target.layout.hiddenSize ||
      target.layout.experts != kExperts ||
      target.layout.expertsPerToken != kTopK ||
      target.layout.expertIntermediateSize != kIntermediate ||
      target.layout.sharedIntermediateSize != kSharedIntermediate ||
      !target.layout.moeSharedExpert || !target.layout.moeUngatedExperts)
    throw WeightStoreError("nemotron MTP draft layout mismatch");
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  WeightFile mtp = mtpFile(files);
  NemotronMtpDraftWeights result;
  result.layout = layout;
  const float eps = layout.rmsEpsilon;
  result.enorm = readNorm(mtp, kHidden, false, "mtp-enorm");
  result.enorm.rmsEpsilon = eps;
  result.hnorm = readNorm(mtp, kHidden, false, "mtp-hnorm");
  result.hnorm.rmsEpsilon = eps;
  result.ehProjection =
      readOqProjection(mtp, kHidden, 2 * kHidden, "mtp-eh-proj");
  result.inputNorm = readNorm(mtp, kHidden, false, "mtp-attn-norm");
  result.inputNorm.rmsEpsilon = eps;
  result.queryProjection = readOqProjection(
      mtp, layout.attentionSize, kHidden, "mtp-attn-q");
  const uint32_t kv = layout.kvHeads * layout.attentionHeadDimension;
  result.keyProjection = readOqProjection(mtp, kv, kHidden, "mtp-attn-k");
  result.valueProjection = readOqProjection(mtp, kv, kHidden, "mtp-attn-v");
  result.outputProjection = readOqProjection(
      mtp, kHidden, layout.attentionSize, "mtp-attn-output");
  result.ffnNorm = readNorm(mtp, kHidden, false, "mtp-ffn-norm");
  result.ffnNorm.rmsEpsilon = eps;
  result.router = mtp.section(
      uint64_t{kExperts} * kHidden * sizeof(uint16_t), "mtp-router");
  result.expertBias =
      mtp.section(uint64_t{kExperts} * sizeof(uint16_t), "mtp-expert-bias");
  result.expertUp = mtp.section(
      uint64_t{kExperts} * kIntermediate * kHidden * sizeof(uint16_t),
      "mtp-experts-up");
  result.expertDown = mtp.section(
      uint64_t{kExperts} * kHidden * kIntermediate * sizeof(uint16_t),
      "mtp-experts-down");
  result.sharedUp = mtp.section(
      uint64_t{kSharedIntermediate} * kHidden * sizeof(uint16_t),
      "mtp-shared-up");
  result.sharedDown = mtp.section(
      uint64_t{kHidden} * kSharedIntermediate * sizeof(uint16_t),
      "mtp-shared-down");
  result.outputNorm = readNorm(mtp, kHidden, false, "mtp-output-norm");
  result.outputNorm.rmsEpsilon = eps;
  result.headNorm = readNorm(mtp, kHidden, false, "mtp-head-norm");
  result.headNorm.rmsEpsilon = eps;
  result.tokenEmbedding = target.tokenEmbedding;
  result.moeRoutedScale = target.layout.moeRoutedScale
                              ? target.layout.moeRoutedScale
                              : 1.0F;
  mtp.finish();
  result.files.push_back(mtp.record());
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace

WeightFile PackedNemotronMtpFiles::mtp() const {
  return images.load(packedImage(directory / "mtp.bin", "draft/mtp.bin",
                                 kNemotronMtpDraftMagic, layout.layers,
                                 1 | kBlockWeightsImageType));
}

NemotronMtpDraftWeights
loadNemotronMtpDraftWeights(metal::MetalBackend &backend,
                            const NemotronMtpDraftFiles &files,
                            const DFlashDraftLayout &layout,
                            const TargetModelWeights &target) {
  return readDraft(backend, files, layout, target);
}

NemotronMtpDraft::NemotronMtpDraft(NemotronMtpDraftWeights weights,
                                   metal::MetalBackend &backend,
                                   const ops::ExecutionPlans &operators)
    : backend_(backend), operators_(operators), weights_(std::move(weights)) {}

void NemotronMtpDraft::addDecode(
    metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
    const ops::Projection &vocabularyProjection,
    std::span<const uint32_t> cacheLengths,
    std::span<const uint32_t> passLimits) const {
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  const DFlashDraftLayout &layout = weights_.layout;
  static const uint32_t envPasses =
      envUint("RICHENGINE_MTP_PASSES", ~uint32_t{0});
  const uint32_t envCap = std::min(layout.proposalLimit(), envPasses);
  if (!passLimits.empty() && passLimits.size() < lanes)
    throw std::invalid_argument("invalid MTP pass-limit batch");
  const uint32_t stride = layout.verifyRows();
  const uint32_t logitsStride =
      paddedVocabularySize(layout.vocabularySize);
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || !envCap)
    throw std::invalid_argument("invalid nemotron MTP draft decode batch");
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
  const uint32_t ring = layout.draftWindow();
  const uint32_t kv = layout.contextKvSize() / 2;

  // As the maxwell MTP draft's: the decode linear's tile consumes eight
  // activation rows, so each lane's scratch slice is eight rows deep and
  // every lane's kernels and shared ops see one row.
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
  const NmtpMoeParams moe{kHidden, kIntermediate, kSharedIntermediate,
                        kInterStride, kExperts, kTopK};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const metal::MetalBuffer tokenIn =
        backend_.view(buffers.mtp.tokenIn, lane * sizeof(uint32_t),
                      sizeof(uint32_t));
    const metal::MetalBuffer inputTokens =
        backend_.view(buffers.inputTokens,
                      uint64_t{lane} * layout.queryRows() * sizeof(uint32_t),
                      uint64_t{layout.queryRows()} * sizeof(uint32_t));
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
    const metal::MetalBuffer keys = rows8(buffers.mtp.keys, lane, kv);
    const metal::MetalBuffer values = rows8(buffers.mtp.values, lane, kv);
    const metal::MetalBuffer attention =
        rows8(buffers.mtp.attention, lane, layout.attentionSize);
    const metal::MetalBuffer branch =
        rows8(buffers.mtp.branch, lane, kHidden);
    const metal::MetalBuffer shared =
        rows8(buffers.mtp.shared, lane, kHidden);
    const metal::MetalBuffer intermediate = rows8(
        buffers.mtp.intermediate, lane, uint64_t{kMoeSlots} * kInterStride);
    const metal::MetalBuffer head = rows8(buffers.mtp.head, lane, kHidden);
    const metal::MetalBuffer aggregate =
        rows8(buffers.mtp.aggregate, lane, kHidden);
    const metal::MetalBuffer streams = backend_.view(
        buffers.mtp.streams, uint64_t{lane} * kHidden * sizeof(float),
        uint64_t{kHidden} * sizeof(float));
    const metal::MetalBuffer routes = backend_.view(
        buffers.mtp.moeRoutes,
        uint64_t{lane} * 8 * 16 * sizeof(uint32_t),
        uint64_t{8} * 16 * sizeof(uint32_t));
    const metal::MetalBuffer partials = backend_.view(
        buffers.mtp.moePartials,
        uint64_t{lane} * 8 * uint64_t{kMoeSlots} * kHidden * sizeof(float),
        8 * uint64_t{kMoeSlots} * kHidden * sizeof(float));
    const metal::MetalBuffer anchor = buffers.mtpAnchors[lane];

    // Pass 0's token is the packed draft input's anchor row; later
    // passes chain on mtp_argmax's greedy pick.
    graph.add("mtp_gather_tokens", {inputTokens, tokenIn},
              MtpGatherTokensParams{0, layout.queryRows(), 0, 0, 1},
              {1, 1, 1}, {32, 1, 1});
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
        graph.add("nmtp_ehtoken",
                  {eh, embedNormed, weights_.hnorm.buffer, anchor, anchor,
                   anchor, anchor, eh},
                  NmtpEhTokenParams{kHidden, 1, 1, 0,
                                    weights_.hnorm.rmsEpsilon},
                  {1, 1, 1}, {1024, 1, 1});
      } else {
        graph.add("nmtp_ehtoken_chain",
                  {streams, embedNormed, weights_.hnorm.buffer, eh},
                  NmtpEhTokenParams{kHidden, 1, 0, 0,
                                    weights_.hnorm.rmsEpsilon},
                  {1, 1, 1}, {1024, 1, 1});
      }
      linear.add(graph,
                 {.input = eh, .output = ehOut, .scratch = scratch},
                 weights_.ehProjection,
                 linear.decodePlan(weights_.ehProjection, 1,
                                   ops::LinearEpilogue::None, nullptr,
                                   kDecodeTileRows));
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
          graph, ehOut, weights_.inputNorm, normed, kHidden,
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
      NmtpKvStoreParams kvParams{};
      kvParams.rows = 1;
      kvParams.kvHeads = layout.kvHeads;
      kvParams.headDim = layout.attentionHeadDimension;
      kvParams.ring = ring;
      kvParams.mode = 0;
      kvParams.posStride = stride;
      kvParams.pass = pass;
      graph.add("nmtp_kv_store",
                {keys, values, positions, positions,
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane]},
                kvParams, {1, uint32_t{kvParams.kvHeads}, 1}, {32, 1, 1});
      graph.add("mtp_attention",
                {queries, buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentKeys[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane],
                 buffers.persistentValues[0][lane], positions, attention},
                MtpAttentionParams{
                    1, layout.attentionSize / layout.attentionHeadDimension,
                    kvParams.kvHeads, kvParams.headDim, ring, stride, pass},
                {1, layout.attentionSize / layout.attentionHeadDimension, 1},
                {32, 1, 1});
      linear.add(graph,
                 {.input = attention, .output = shared,
                  .scratch = scratch},
                 weights_.outputProjection,
                 linear.decodePlan(weights_.outputProjection, 1,
                                   ops::LinearEpilogue::None, nullptr,
                                   kDecodeTileRows));
      // The attention residual: float segments take no epilogue, so the
      // draft residual kernel adds the projection output onto the eh
      // row — the pass's post-attention hidden.
      graph.add(std::string(ops::kDraftResidualAdd),
                {shared, ehOut, branch}, kHidden,
                {(kHidden + 255) / 256, 1, 1}, {256, 1, 1});
      ops::Normalization::addRms(graph, branch, weights_.ffnNorm, normed,
                                 kHidden, kDecodeTileRows);
      // The ungated sigmoid MoE: route, relu-squared up, down and the
      // weighted combine onto the residual — slot kTopK the shared
      // expert at unit weight.
      graph.add("nmtp_moe_route",
                {normed, weights_.router, weights_.expertBias, routes},
                NmtpMoeRouteParams{kHidden, kExperts, kTopK,
                                   weights_.moeRoutedScale},
                {1, 1, 1}, {1024, 1, 1});
      graph.add("nmtp_moe_up",
                {normed, weights_.expertUp, weights_.sharedUp, routes,
                 intermediate},
                moe,
                {(kSharedIntermediate + 255) / 256, kMoeSlots, 1},
                {256, 1, 1});
      graph.add("nmtp_moe_down",
                {intermediate, weights_.expertDown, weights_.sharedDown,
                 routes, partials},
                moe,
                {(kHidden + 255) / 256, kMoeSlots, 1}, {256, 1, 1});
      graph.add("nmtp_moe_combine", {branch, partials, routes, head}, moe,
                {(kHidden + 255) / 256, 1, 1}, {256, 1, 1});
      // The chained hidden of the next pass is the block's residual
      // stream output, pre norms — bf16 to the fp32 chain state.
      graph.add("mtp_stream_init", {head, streams},
                MtpStreamInitParams{kHidden, 1},
                {(uint64_t{kHidden} + 255) / 256, 1, 1}, {256, 1, 1});
      ops::Normalization::addRms(graph, head, weights_.outputNorm, normed,
                                 kHidden, kDecodeTileRows);
      ops::Normalization::addRms(graph, normed, weights_.headNorm,
                                 aggregate, kHidden, kDecodeTileRows);
      // The head's eight-row tile lands in the lane's logits scratch;
      // row 0 is the pass's logits, copied into its selector slot.
      const metal::MetalBuffer logitsWide =
          rows8f(buffers.mtp.logits, lane, logitsStride);
      linear.add(graph,
                 {.input = aggregate, .output = logitsWide,
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

void NemotronMtpDraft::addSelection(
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

void NemotronMtpDraft::addContextCommit(
    metal::CommandGraph &graph, DFlashContextBuffers buffers,
    std::span<const uint32_t> startPositions) const {
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t stride = layout.verifyRows();
  const uint32_t rows = lanes * stride;
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid nemotron MTP context commit");
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;

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
  graph.add("nmtp_ehtoken",
            {buffers.capturedTargetHidden, buffers.mtp.embedNormed,
             weights_.hnorm.buffer, anchor(0), anchor(1), anchor(2),
             anchor(3), buffers.mtp.eh},
            NmtpEhTokenParams{kHidden, rows, 0, 0,
                              weights_.hnorm.rmsEpsilon},
            {rows, 1, 1}, {1024, 1, 1});
  linear.add(graph,
             {.input = buffers.mtp.eh, .output = buffers.mtp.ehOut,
              .scratch = scratch},
             weights_.ehProjection,
             linear.decodePlan(weights_.ehProjection, lanes,
                               ops::LinearEpilogue::None, nullptr, stride));
  const ops::LinearPlan kPlan =
      linear.decodePlan(weights_.keyProjection, lanes,
                        ops::LinearEpilogue::None, nullptr, stride);
  const ops::LinearPlan vPlan =
      linear.decodePlan(weights_.valueProjection, lanes,
                        ops::LinearEpilogue::None, nullptr, stride);
  const ops::LinearInput kvInput = kPlan.input();
  const ops::PreparedInput prepared = ops::Normalization::addRms(
      graph, buffers.mtp.ehOut, weights_.inputNorm, buffers.hidden, kHidden,
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
  NmtpKvStoreParams kv{};
  kv.rows = rows;
  kv.kvHeads = layout.kvHeads;
  kv.headDim = layout.attentionHeadDimension;
  kv.ring = layout.draftWindow();
  kv.mode = 1;
  kv.stride = stride;
  std::copy_n(startPositions.begin(), lanes, kv.startPositions);
  graph.add("nmtp_kv_store",
            {buffers.mtp.keys, buffers.mtp.values,
             buffers.retainedCounts, buffers.retainedCounts,
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
            MtpAnchorParams{kHidden, stride, lanes},
            {lanes, 1, 1}, {256, 1, 1});
}

void NemotronMtpDraft::addContextPrefill(
    metal::CommandGraph &graph, DFlashPrefillBuffers buffers,
    uint32_t capturedRows,
    std::span<const DFlashPrefillSpan> spans) const {
  if (!capturedRows || spans.empty())
    throw std::invalid_argument("invalid nemotron MTP context prefill");
  const DFlashDraftLayout &layout = weights_.layout;
  const uint32_t ring = layout.draftWindow();
  const ops::Linear &linear = operators_.linear();
  const ops::LinearScratch &scratch = buffers.linearScratch;
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
          uint64_t{rowBase} * kHidden * sizeof(uint16_t),
          uint64_t{rows} * kHidden * sizeof(uint16_t));
      graph.add("nmtp_ehtoken",
                {captured, buffers.mtp.embedNormed, weights_.hnorm.buffer,
                 captured, captured, captured, captured, buffers.mtp.eh},
                NmtpEhTokenParams{kHidden, rows, 0, 0,
                                  weights_.hnorm.rmsEpsilon},
                {rows, 1, 1}, {1024, 1, 1});
      linear.addPrefill(graph, buffers.mtp.eh, weights_.ehProjection,
                        buffers.mtp.ehOut, {}, rows, scratch);
      ops::Normalization::addRms(graph, buffers.mtp.ehOut,
                                 weights_.inputNorm, buffers.hidden, kHidden,
                                 rows, scratch);
      linear.addPrefill(graph, buffers.hidden, weights_.keyProjection,
                        buffers.mtp.keys, {}, rows, scratch);
      linear.addPrefill(graph, buffers.hidden, weights_.valueProjection,
                        buffers.mtp.values, {}, rows, scratch);
      graph.add("nmtp_kv_store_prefill",
                {buffers.mtp.keys, buffers.mtp.values,
                 span.ring[0].keys, span.ring[0].values},
                NmtpPrefillKvParams{rows, layout.kvHeads,
                                    layout.attentionHeadDimension, ring,
                                    span.startPosition + begin,
                                    span.inputEnd - span.inputRow +
                                        span.startPosition},
                {rows, layout.kvHeads, 1}, {32, 1, 1});
      // A tail span's last captured row anchors the state lane's first
      // decode pass.
      if (span.anchorRow != ~0u && begin + rows > span.anchorRow - rowBase) {
        const uint32_t anchor = span.anchorRow;
        if (anchor >= rowBase && anchor < rowBase + rows) {
          const metal::MetalBuffer row = backend_.view(
              buffers.capturedTargetHidden,
              uint64_t{anchor} * kHidden * sizeof(uint16_t),
              uint64_t{kHidden} * sizeof(uint16_t));
          graph.add("mtp_anchor_prefill", {row, span.anchor},
                    MtpAnchorParams{kHidden, 0, 1},
                    {1, 1, 1}, {256, 1, 1});
        }
      }
    }
  }
}

} // namespace richengine::model

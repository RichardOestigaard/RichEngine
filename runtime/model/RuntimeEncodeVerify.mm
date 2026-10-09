#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  void Runtime::Impl::encodeTargetVerifyBatchForward(CommandGraph &graph,
                                      std::span<Request *const> entries,
                                      std::span<const ModelBatchItem> items,
                                      bool tree,
                                      std::span<const uint32_t> liveRows) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size() ||
        (tree && entries.size() > kLaneCount / 2)) {
      throw std::invalid_argument("invalid target verify batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    const uint32_t storage =
        targetModel.decodeStorageLanes(tree ? 2 * lanes : lanes);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, storage);
    };

    std::array<ChunkedPrefillParams, kLaneCount> chunks{};
    const uint32_t gdnLayers = geometry.target.stateLayout.layers;
    const uint32_t attentionLayers =
        geometry.target.kvLayout.attentionLayers;
    std::vector<MetalBuffer> gdnPacked(gdnLayers);
    std::vector<MetalBuffer> gdnMixed(gdnLayers);
    std::vector<MetalBuffer> gdnDecay(gdnLayers);
    std::vector<MetalBuffer> gdnBeta(gdnLayers);
    std::vector<MetalBuffer> chunkKeys(attentionLayers);
    std::vector<MetalBuffer> chunkValues(attentionLayers);
    TargetModelVerifyBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.hidden = {d(DecodeTensor::Hidden0), d(DecodeTensor::Hidden1)};
    buffers.normalized = d(DecodeTensor::Normalized);
    buffers.gdnHidden = d(DecodeTensor::GdnHidden);
    buffers.gdnOutput = d(DecodeTensor::GdnOutput);
    buffers.denseIntermediate = d(DecodeTensor::Intermediate);
    buffers.fullPacked = d(DecodeTensor::FullPacked);
    buffers.fullQueries = d(DecodeTensor::FullQueries);
    buffers.attentionPartials = d(DecodeTensor::AttentionPartials);
    buffers.attentionStatistics = d(DecodeTensor::AttentionStatistics);
    buffers.fullAttention = d(DecodeTensor::FullAttention);
    buffers.attentionHidden = d(DecodeTensor::AttentionHidden);
    buffers.attentionOutput = d(DecodeTensor::AttentionOutput);
    buffers.ropeCos = d(DecodeTensor::RopeCos);
    buffers.ropeSin = d(DecodeTensor::RopeSin);
    buffers.ropeCosAlt = d(DecodeTensor::RopeCosAlt);
    buffers.ropeSinAlt = d(DecodeTensor::RopeSinAlt);
    buffers.gemmaResidual = d(DecodeTensor::GemmaResidual);
    buffers.zeroResidual = d(DecodeTensor::ZeroResidual);
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = d(DecodeTensor::FinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    // The fused greedy head runs when every lane is greedy, unconstrained
    // and unpenalized on a chain batch — the vocabulary projection then
    // argmaxes in-kernel and the logits buffer is never written.
    const ops::Projection &headShape = targetModel.vocabularyProjection();
    // The affine fused head takes a 128-column tile, the single-segment GGUF
    // one (gguf_decode_*_m*_amax) a 64-column tile; both need whole tiles and
    // 64-input spans.
    const bool ggufHead =
        headShape.layout() == ops::WeightLayout::Block32 &&
        headShape.blocks().segments.size() == 1 &&
        !headShape.blocks().segments.front().isFloat() &&
        headShape.outputSize % 64 == 0 &&
        headShape.inputSize % 64 == 0;
    static const bool headFusedOff = envFlag("RICHENGINE_HEAD_FUSED_OFF");
    // The fused head's argmax tile holds 32 rows; banding wider batches
    // through it measured no better than the ordinary projection.
    buffers.fusedHead = !headFusedOff && !tree &&
        uint64_t{lanes} * ExecutionLimits::targetVerifyRows <= 32 &&
        ((headShape.layout() == ops::WeightLayout::Affine64 &&
          headShape.outputSize % 128 == 0 && headShape.inputSize % 64 == 0) ||
         ggufHead);
    buffers.headArgs.output_size = geometry.target.vocabularySize;
    buffers.headArgs.input_size = geometry.target.hiddenSize;
    buffers.headArgs.rows = ExecutionLimits::targetVerifyRows;
    buffers.headArgs.stop_token_0 = geometry.target.stopTokens[0];
    buffers.headArgs.stop_token_1 = geometry.target.stopTokens[1];
    for (uint32_t lane = 0; lane < lanes && buffers.fusedHead; ++lane) {
      const ops::SamplingPolicy policy = samplingPolicy(*entries[lane]);
      if (policy.samples() || policy.constrained || policy.penalties.active()) {
        buffers.fusedHead = false;
        break;
      }
      if (policy.excludesStopTokens)
        buffers.headArgs.exclude_stop_mask |= uint32_t{1} << lane;
      buffers.headArgs.live_rows[lane] =
          lane < liveRows.size()
              ? std::clamp(liveRows[lane], uint32_t{1},
                           ExecutionLimits::targetVerifyRows)
              : ExecutionLimits::targetVerifyRows;
    }
    if (buffers.fusedHead) {
      buffers.headArgmaxValues = d(DecodeTensor::HeadArgmaxValues);
      buffers.headArgmaxIndices = d(DecodeTensor::HeadArgmaxIndices);
    }
    buffers.denseGateScratch = decodeArena->gateScratch();
    buffers.gdnPacked = gdnPacked;
    buffers.gdnMixed = gdnMixed;
    buffers.gdnDecay = gdnDecay;
    buffers.gdnBeta = gdnBeta;
    buffers.chunkKeys = chunkKeys;
    buffers.chunkValues = chunkValues;
    buffers.moe = decodeArena->moeScratch(storage);
    if (tree) {
      buffers.treeNodes = decodeArena->packed(DecodeTensor::TreeNodes, lanes);
      buffers.treeCounts =
          decodeArena->packed(DecodeTensor::TreeCounts, lanes);
      buffers.treeMasks = decodeArena->packed(DecodeTensor::TreeMasks, lanes);
      buffers.capturedPath =
          decodeArena->packed(DecodeTensor::CapturedPath, lanes);
    }
    // A host-emitted comb's node count is already known; a kernel-emitted
    // tree's is on the GPU, so it keeps the scratch's full capacity.
    const uint32_t liveNodes =
        ngram_.predraftedTree_ ? ngram_.predraftedTreeNodes_
                               : RICHENGINE_TREE_VERIFY_NODES;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      chunks[lane] =
          tree ? ops::PagedAttention::verifyTreeParams(
                     items[lane].logicalPosition,
                     static_cast<uint32_t>(items[lane].pageTable.size()),
                     liveNodes)
               : ops::PagedAttention::verifyParams(
                     items[lane].logicalPosition,
                     static_cast<uint32_t>(items[lane].pageTable.size()));
    bindPageTables(entries, buffers.pageTables);
    if (gdnLayers)
      bindGdnStates(entries, buffers.currentGdnStates,
                    buffers.nextGdnStates);
    for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
      gdnPacked[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyPackedBase, layer, storage);
      gdnMixed[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyMixedBase, layer, storage);
      gdnDecay[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyDecayBase, layer, storage);
      gdnBeta[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyBetaBase, layer, storage);
    }
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      chunkKeys[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkKeysBase, layer, storage);
      chunkValues[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkValuesBase, layer, storage);
    }
    targetModel.addVerify(graph, std::move(buffers), kvPages.layers(),
                          std::span(chunks).first(lanes), lanes, tree,
                          liveRows, liveNodes);
  }

  // The sampling buffers of a tree batch: lane-semantic tensors keep their
  // lanes' strides, while the row-indexed logits and the argmax output take
  // the tree's RICHENGINE_TREE_VERIFY_NODES stride.
  ops::SamplingBuffers Runtime::Impl::samplingTreeBuffers(uint32_t lanes) const {
    ops::SamplingBuffers buffers = samplingBuffers(lanes);
    buffers.logits =
        decodeArena->packed(DecodeTensor::Logits, 2 * lanes);
    buffers.inputTokens =
        decodeArena->packed(DecodeTensor::InputTokens, 2 * lanes);
    buffers.outputTokens =
        decodeArena->packed(DecodeTensor::TreeSelected, lanes);
    return buffers;
  }

  void Runtime::Impl::encodeTargetVerifyBatchPolicy(CommandGraph &graph,
                                     std::span<Request *const> entries,
                                     bool tree,
                                     std::span<const uint32_t> liveRows) {
    if (entries.empty() || entries.size() > kLaneCount)
      throw std::invalid_argument("invalid target policy batch");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint32_t, kLaneCount> stateLanes{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!entries[lane])
        throw std::invalid_argument("empty target policy lane");
      policies[lane] = samplingPolicy(*entries[lane]);
      stateLanes[lane] = entries[lane]->stateLane;
    }
    if (tree) {
      sampling.addVerifyTree(graph, std::span(policies).first(lanes),
                             samplingTreeBuffers(lanes),
                             geometry.target.stopTokens[0],
                             geometry.target.stopTokens[1]);
      return;
    }
    const ops::Projection &headShape = targetModel.vocabularyProjection();
    static const bool headFusedOff = envFlag("RICHENGINE_HEAD_FUSED_OFF");
    const bool ggufHead =
        !headFusedOff &&
        headShape.layout() == ops::WeightLayout::Block32 &&
        headShape.blocks().segments.size() == 1 &&
        !headShape.blocks().segments.front().isFloat() &&
        headShape.outputSize % 64 == 0 &&
        headShape.inputSize % 64 == 0;
    // The fused head path must match addHeadBatch's 32-row bound: taking
    // partials for a head that never ran would reduce stale scratch — the
    // logits it skipped are never written.
    uint32_t fusedHead = headFusedOff ||
            uint64_t{lanes} * ExecutionLimits::targetVerifyRows > 32 ? 0 :
        headShape.layout() == ops::WeightLayout::Affine64 &&
                headShape.outputSize % 128 == 0 && headShape.inputSize % 64 == 0
            ? 1
            : ggufHead ? 2 : 0;
    for (uint32_t lane = 0; lane < lanes && fusedHead; ++lane) {
      if (policies[lane].samples() || policies[lane].constrained ||
          policies[lane].penalties.active()) {
        fusedHead = false;
        break;
      }
    }
    auto vBuffers = samplingBuffers(lanes);
    if (fusedHead) {
      vBuffers.headArgmaxValues =
          decodeArena->packed(DecodeTensor::HeadArgmaxValues,
                              targetModel.decodeStorageLanes(lanes));
      vBuffers.headArgmaxIndices =
          decodeArena->packed(DecodeTensor::HeadArgmaxIndices,
                              targetModel.decodeStorageLanes(lanes));
    }
    sampling.addVerify(graph, std::span(policies).first(lanes),
                       vBuffers, geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1],
                       {penaltyTable, std::span(stateLanes).first(lanes)},
                       liveRows, fusedHead);
  }

  void Runtime::Impl::encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained,
                             bool tree,
                             std::span<const uint32_t> proposals) {
    if (lanes.empty() || lanes.size() > kLaneCount ||
        lanes.size() != maximumRetained.size()) {
      throw std::invalid_argument("invalid DFlash acceptance batch");
    }
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      if (!lanes[lane] || !maximumRetained[lane] ||
          maximumRetained[lane] > kDecodeRows) {
        throw std::invalid_argument("invalid DFlash acceptance lane");
      }
      policies[lane] = samplingPolicy(*lanes[lane]);
    }
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    if (tree) {
      sampling.addTreeAcceptance(
          graph, decodeArena->packed(DecodeTensor::TreeTokens, width),
          decodeArena->packed(DecodeTensor::TreeNodes, width),
          decodeArena->packed(DecodeTensor::TreeCounts, width),
          decodeArena->packed(DecodeTensor::TreeSelected, width),
          decodeArena->packed(DecodeTensor::OutputTokens, width),
          decodeArena->packed(DecodeTensor::RetainedCount, width),
          decodeArena->packed(DecodeTensor::AcceptedCount, width),
          decodeArena->packed(DecodeTensor::RetainedPath, width),
          maximumRetained, geometry.target.stopTokens[0],
          geometry.target.stopTokens[1]);
      return;
    }
    // DSpark emits its proposals from the merged shard pool, so a sampled
    // lane's draft-distribution lookup reads the pool's ids and
    // probabilities at the pool stride instead of the merged top-16's.
    const bool dspark = std::holds_alternative<DSparkDraft>(draftModel);
    sampling.addAcceptance(
        graph,
        {decodeArena->packed(DecodeTensor::ProposedTokens, width),
         dspark ? decodeArena->packed(DecodeTensor::TopPartialIds, width)
                : decodeArena->packed(DecodeTensor::Candidates, width),
         decodeArena->packed(DecodeTensor::ProposalProbs, width),
         decodeArena->packed(DecodeTensor::TargetVocabularyRows, width),
         decodeArena->packed(DecodeTensor::SamplingUniforms, width),
         decodeArena->packed(DecodeTensor::OutputTokens, width),
         decodeArena->packed(DecodeTensor::RetainedCount, width),
         decodeArena->packed(DecodeTensor::AcceptedCount, width)},
        maximumRetained, std::span(policies).first(width),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1],
        proposals,
        dspark ? RICHENGINE_DSPARK_POOL : RICHENGINE_DRAFT_CANDIDATES);
  }

  void Runtime::Impl::encodeBatchVerifyInput(CommandGraph &graph, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid verify-input batch width");
    targetModel.addVerifyInput(
        graph, decodeArena->packed(DecodeTensor::DraftInputTokens, lanes),
        decodeArena->packed(DecodeTensor::ProposedTokens, lanes),
        decodeArena->packed(DecodeTensor::InputTokens, lanes), lanes);
  }

  // A tree batch's verify inputs: the selector's tables carry every lane's
  // emitted nodes; the pass emits the input tokens, rotary positions and
  // ancestor masks, two decode rows' storage per lane.
  void Runtime::Impl::encodeBatchVerifyTreeInput(CommandGraph &graph,
                                  std::span<Request *const> entries,
                                  std::span<const ModelBatchItem> items,
                                  uint32_t lanes) {
    if (!lanes || lanes > kLaneCount / 2)
      throw std::invalid_argument("invalid verify-tree input batch width");
    uint32_t base[RICHENGINE_MAXIMUM_BATCH_WIDTH][3]{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const std::array<uint32_t, 3> rotary =
          ropePosition(*entries[lane], items[lane].logicalPosition);
      std::copy(rotary.begin(), rotary.end(), base[lane]);
    }
    targetModel.addVerifyTreeInput(
        graph, decodeArena->packed(DecodeTensor::TreeTokens, lanes),
        decodeArena->packed(DecodeTensor::TreeNodes, lanes),
        decodeArena->packed(DecodeTensor::TreeCounts, lanes),
        decodeArena->packed(DecodeTensor::InputTokens, 2 * lanes),
        decodeArena->packed(DecodeTensor::Positions, 2 * lanes),
        decodeArena->packed(DecodeTensor::TreeMasks, lanes), base, lanes);
  }

  // The tree batch's KV tail: after acceptance, the retained path's slabs
  // move to their committed positions in each attention layer's pages.
  void Runtime::Impl::encodeBatchTreeKvCompact(
      CommandGraph &graph, std::span<Request *const> entries,
      std::span<const ModelBatchItem> items) {
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    const uint32_t attentionLayers =
        geometry.target.kvLayout.attentionLayers;
    if (!attentionLayers)
      return;
    std::array<MetalBuffer, kLaneCount> pageTables;
    std::array<kv::ChunkedPrefillParams, kLaneCount> chunks;
    bindPageTables(entries, pageTables);
    const uint32_t liveNodes =
        ngram_.predraftedTree_ ? ngram_.predraftedTreeNodes_
                               : RICHENGINE_TREE_VERIFY_NODES;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      chunks[lane] = ops::PagedAttention::verifyTreeParams(
          items[lane].logicalPosition,
          static_cast<uint32_t>(items[lane].pageTable.size()), liveNodes);
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      ops::PagedAttention::addVerifyTreeCompact(
          graph, kvPages.layers()[layer], pageTables,
          decodeArena->packed(DecodeTensor::RetainedPath, lanes),
          decodeArena->packed(DecodeTensor::RetainedCount, lanes),
          std::span(chunks).first(lanes), lanes,
          geometry.target.kvLayout);
    }
  }

  void Runtime::Impl::encodeBatchGdnCommit(CommandGraph &graph,
                            std::span<Request *const> lanes,
                            bool tree) {
    if (lanes.empty() || lanes.size() > kLaneCount)
      throw std::invalid_argument("invalid GDN commit batch");
    // A stateless target (the pure dense family) commits nothing.
    if (!geometry.target.stateLayout.layers)
      return;
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    std::array<MetalBuffer, kLaneCount> currentStates;
    std::array<MetalBuffer, kLaneCount> nextStates;
    bindGdnStates(lanes, currentStates, nextStates);
    TargetModelCommitBuffers buffers{
        decodeArena->gdnStorage(DecodeTensor::VerifyPackedBase),
        decodeArena->gdnStorage(DecodeTensor::VerifyMixedBase),
        decodeArena->gdnStorage(DecodeTensor::VerifyDecayBase),
        decodeArena->gdnStorage(DecodeTensor::VerifyBetaBase), currentStates,
        nextStates, decodeArena->packed(DecodeTensor::RetainedCount, width)};
    if (tree) {
      targetModel.addStateCommitTree(
          graph, std::move(buffers),
          decodeArena->packed(DecodeTensor::RetainedPath, width), width);
      return;
    }
    targetModel.addStateCommit(graph, std::move(buffers), width);
  }

} // namespace richengine::model

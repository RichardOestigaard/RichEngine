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
    buffers.gdnTape = gdnTape_;
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
        uint64_t{lanes} * geometry.draft.verifyRows() <= 32 &&
        ((headShape.layout() == ops::WeightLayout::Affine64 &&
          headShape.outputSize % 128 == 0 && headShape.inputSize % 64 == 0) ||
         ggufHead);
    buffers.headArgs.output_size = geometry.target.vocabularySize;
    buffers.headArgs.output_stride = geometry.target.paddedVocabularySize;
    buffers.headArgs.input_size = geometry.target.hiddenSize;
    buffers.headArgs.rows = geometry.draft.verifyRows();
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
                           geometry.draft.verifyRows())
              : geometry.draft.verifyRows();
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
    // A chain batch's verify rows, which a sparse-MoE target's steps route
    // alone; a tree batch routes every live node.
    if (!tree)
      buffers.verifyRows = decodeArena->packed(DecodeTensor::VerifyRows, lanes);
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
                     static_cast<uint32_t>(items[lane].pageTable.size()),
                     geometry.draft.verifyRows());
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
                          liveRows, liveNodes, geometry.draft.verifyRows());
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
                                     std::span<const uint32_t> liveRows,
                                     uint32_t dfaLaneSkip) {
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
            uint64_t{lanes} * geometry.draft.verifyRows() > 32 ? 0 :
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
    // Full-mode DFA lanes read their next-state admission rows; a skipped
    // lane (an abandoned mask wait) keeps the host-filled bitmask, as do
    // all lanes under RICHENGINE_DFA_MODE=forced. A greedy DFA lane instead
    // argmaxes unmasked here: its proposal check compares the proposal to
    // that argmax, and the single masked correction row comes from the
    // commit graph's truncation pass (decode_dfa_truncate).
    const uint32_t unmasked = greedyDfaLaneMask(entries, dfaLaneSkip);
    vBuffers.dfaLanes =
        (gpuDfaLaneMask(entries) & ~dfaLaneSkip) & ~unmasked;
    vBuffers.unmaskedLanes = unmasked;
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
                       liveRows, fusedHead, geometry.draft.verifyRows());
  }

  void Runtime::Impl::encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained,
                             bool tree,
                             std::span<const uint32_t> proposals,
                             uint32_t truncLanes) {
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
         decodeArena->packed(DecodeTensor::AcceptedCount, width),
         decodeArena->packed(DecodeTensor::DfaTrunc, width),
         truncLanes,
         decodeArena->packed(DecodeTensor::VerifyRows, width)},
        maximumRetained, std::span(policies).first(width),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1],
        proposals,
        dspark ? RICHENGINE_DSPARK_POOL : RICHENGINE_DRAFT_CANDIDATES,
        geometry.draft.verifyRows());
  }

  void Runtime::Impl::encodeBatchVerifyInput(
      CommandGraph &graph, std::span<Request *const> entries,
      std::span<const ModelBatchItem> items, uint32_t lanes,
      std::span<const uint32_t> liveRows) {
    if (!lanes || lanes > kLaneCount ||
        entries.size() < lanes || items.size() < lanes)
      throw std::invalid_argument("invalid verify-input batch width");
    const MetalBuffer draftInput =
        decodeArena->packed(DecodeTensor::DraftInputTokens, lanes);
    const MetalBuffer proposals =
        decodeArena->packed(DecodeTensor::ProposedTokens, lanes);
    const MetalBuffer verifyInput =
        decodeArena->packed(DecodeTensor::InputTokens, lanes);
    if (deviceVerifyParams_) {
      // The device variant binds the step pack: each lane takes its
      // anchor and live count from the pack when it proves to be that
      // lane's own predecessor's. The staged payload still carries the
      // host's values — they are the proof's reference and the fallback.
      // device_lanes stays zero on the readback path: the equality proof
      // alone decides.
      std::array<uint64_t, kLaneCount> requestIds{};
      std::array<uint32_t, kLaneCount> positionBases{};
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        requestIds[lane] = entries[lane]->id;
        positionBases[lane] =
            static_cast<uint32_t>(items[lane].logicalPosition);
      }
      ops::Embedding::addVerifyInputDevice(
          graph, draftInput, proposals, verifyInput, stepPackBuffer(),
          {requestIds.data(), lanes}, {positionBases.data(), lanes},
          /*deviceLanes=*/0, geometry.target.vocabularySize, lanes,
          geometry.draft.verifyRows(), geometry.draft.queryRows(),
          liveRows);
      return;
    }
    targetModel.addVerifyInput(
        graph, draftInput, proposals, verifyInput, lanes,
        geometry.draft.verifyRows(), geometry.draft.queryRows(), liveRows);
  }

  void Runtime::Impl::encodeBatchStepPack(
      CommandGraph &graph, std::span<Request *const> entries,
      std::span<const ModelBatchItem> items,
      std::span<const uint32_t> budgets, uint32_t outputStride) {
    if (!deviceVerifyParams_)
      return;
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size() || entries.size() != budgets.size())
      throw std::invalid_argument("invalid step-pack batch");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    DecodeStepPackParams params{};
    params.lanes = lanes;
    params.output_stride = outputStride;
    params.decode_rows = geometry.draft.verifyRows();
    params.verify_stride = geometry.draft.verifyRows();
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const Request &entry = *entries[lane];
      params.request_id[lane] = entry.id;
      params.position_base[lane] =
          static_cast<uint32_t>(items[lane].logicalPosition);
      params.remaining[lane] =
          entry.maxNewTokens - entry.generatedTokens;
      params.budget[lane] = budgets[lane];
    }
    graph.add(std::string(ops::kDecodeStepPack),
              {decodeArena->packed(DecodeTensor::RetainedCount, lanes),
               decodeArena->packed(DecodeTensor::AcceptedCount, lanes),
               decodeArena->packed(DecodeTensor::OutputTokens, lanes),
               stepPackBuffer()},
              params, {lanes, 1, 1}, {1, 1, 1});
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

  // The DFA lanes' per-lane transition tables, accepting bitsets and
  // marker masks: lanes without a DFA rebind the first one's buffers and
  // are masked out.
  std::array<MetalBuffer, kLaneCount>
  Runtime::Impl::dfaBuffers(std::span<Request *const> entries,
                            DfaBufferKind kind) {
    const auto pick = [](const RequestDfa &dfa, DfaBufferKind kind)
        -> const MetalBuffer & {
      switch (kind) {
      case DfaBufferKind::Table:
        return dfa.tableBuffer;
      case DfaBufferKind::Accept:
        return dfa.acceptBuffer;
      case DfaBufferKind::Marker:
        return dfa.markerBuffer;
      }
      return dfa.tableBuffer;
    };
    std::array<MetalBuffer, kLaneCount> buffers{};
    const RequestDfa *first = nullptr;
    for (Request *const entry : entries)
      if (entry->dfa)
        first = &*entry->dfa;
    for (uint32_t lane = 0; lane < entries.size(); ++lane) {
      const RequestDfa &dfa = entries[lane]->dfa ? *entries[lane]->dfa : *first;
      buffers[lane] = pick(dfa, kind);
    }
    // The kernels take all four table slots; lanes past the live batch are
    // masked out, so any valid buffer satisfies their bindings.
    for (uint32_t lane = entries.size(); lane < kLaneCount; ++lane)
      buffers[lane] = pick(*first, kind);
    return buffers;
  }

  // A lane's grammar-EOS set into a params struct's per-lane fields.
  template <class Params>
  static void dfaLaneEos(Params &params, uint32_t lane,
                         const RequestDfa &dfa) {
    params.eos_count[lane] = dfa.eosCount;
    std::copy(dfa.eosIds.begin(),
              dfa.eosIds.begin() + dfa.eosCount,
              params.eos_ids[lane]);
  }

  void Runtime::Impl::encodeDfaMasks(
      CommandGraph &graph, std::span<Request *const> entries, uint32_t lanes,
      std::span<const uint32_t> liveRows, const DfaGridLanes *grid) {
    // The GPU-grid path's span consumer: decode_dfa_mask_grid packs the
    // mask pass's launch size from the same live-row arithmetic the host
    // applies and clears each consumed span on-device, so it dispatches
    // even when no mask pass follows — a skipped dispatch would leave the
    // span pending for a later round to reapply.
    if (grid && grid->gpuSpanMask) {
      if (!packedVocab_)
        throw std::logic_error("GPU-DFA lane has no packed vocabulary");
      if (!dfaMaskGrid_)
        dfaMaskGrid_ = backend.allocateBuffer(16, BufferStorage::Shared,
                                              "dfa-mask-grid");
      DfaMaskGridParams gridParams{};
      gridParams.vocabulary_groups =
          static_cast<uint32_t>((packedVocab_->vocab + 255) / 256);
      gridParams.lanes = lanes;
      gridParams.gpu_span_lanes = grid->gpuSpanMask;
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        gridParams.state_lanes[lane] = entries[lane]->stateLane;
        gridParams.maximum_retained[lane] = grid->maximumRetained[lane];
        gridParams.live_rows[lane] =
            lane < liveRows.size() ? liveRows[lane]
                                   : geometry.draft.verifyRows();
      }
      graph.add(std::string(ops::kDecodeDfaMaskGrid),
                {decodeArena->packed(DecodeTensor::DfaForced, kLaneCount),
                 dfaMaskGrid_},
                gridParams, {1, 1, 1}, {1, 1, 1});
    }
    // Forced mode keeps the host mask round trip: the chain's row states
    // feed only the mask kernel, so both stay unencoded.
    if (dfaMode_ != DfaMode::Full)
      return;
    const uint32_t laneMask = dfaLaneMask(entries);
    if (!laneMask)
      return;
    if (!packedVocab_)
      throw std::logic_error("GPU-DFA lane has no packed vocabulary");
    const auto tables = dfaBuffers(entries, DfaBufferKind::Table);
    const auto accepts = dfaBuffers(entries, DfaBufferKind::Accept);
    const auto markers = dfaBuffers(entries, DfaBufferKind::Marker);
    DfaWalkParams walk{};
    walk.verify_stride = geometry.draft.verifyRows();
    walk.lane_mask = laneMask;
    walk.vocabulary = packedVocab_->vocab;
    DfaMaskParams mask{};
    mask.vocabulary = packedVocab_->vocab;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      // liveRows bounds the chain's input walk; the mask pass writes one
      // next-state row per live row plus the anchor's row 0.
      const uint32_t live =
          lane < liveRows.size() ? liveRows[lane]
                                 : geometry.draft.verifyRows();
      walk.live_rows[lane] = live;
      mask.live_rows[lane] =
          std::min(live + 1, geometry.draft.verifyRows() + 1);
      mask.rows = std::max(mask.rows, mask.live_rows[lane]);
      if (entries[lane]->dfa) {
        walk.states[lane] = entries[lane]->dfa->states;
        mask.states[lane] = entries[lane]->dfa->states;
        dfaLaneEos(walk, lane, *entries[lane]->dfa);
        dfaLaneEos(mask, lane, *entries[lane]->dfa);
        // A greedy lane's proposals validate against the unmasked argmax
        // and its single correction row comes from the commit graph's
        // truncation pass — only a sampled lane's selection still reads
        // the full next-state rows this round.
        if (samplingPolicy(*entries[lane]).samples())
          mask.lane_mask |= uint32_t{1} << lane;
      }
      walk.state_lanes[lane] = entries[lane]->stateLane;
      if (entries[lane]->dfa)
        syncDfaState(*entries[lane]);
    }
    const MetalBuffer nextStates =
        decodeArena->packed(DecodeTensor::DfaNextStates, lanes);
    graph.add(std::string(ops::kDecodeDfaChain),
              {tables[0], tables[1], tables[2], tables[3],
               packedVocab_->bytes, packedVocab_->offsets,
               decodeArena->packed(DecodeTensor::InputTokens, lanes),
               decodeArena->packed(DecodeTensor::DfaState, kLaneCount),
               decodeArena->packed(DecodeTensor::DfaRowStates, lanes),
               accepts[0], accepts[1], accepts[2], accepts[3],
               markers[0], markers[1], markers[2], markers[3],
               packedVocab_->special},
              walk, {lanes, 1, 1}, {1, 1, 1});
    if (!mask.lane_mask)
      return;
    const std::vector<MetalBuffer> maskBuffers{
        tables[0], tables[1], tables[2], tables[3],
        accepts[0], accepts[1], accepts[2], accepts[3],
        packedVocab_->bytes, packedVocab_->offsets,
        decodeArena->packed(DecodeTensor::DfaRowStates, lanes), nextStates,
        decodeArena->packed(DecodeTensor::DfaTrunc, lanes),
        markers[0], markers[1], markers[2], markers[3],
        packedVocab_->special};
    if (grid && grid->gpuSpanMask) {
      // The grid kernel's MTLDispatchThreadgroupsIndirectArguments sizes
      // the launch: {vocabulary groups, mask.rows, lanes}.
      graph.addIndirect(std::string(ops::kDecodeDfaMask), maskBuffers, mask,
                        backend.view(dfaMaskGrid_, 0, 12), {256, 1, 1});
      return;
    }
    graph.add(std::string(ops::kDecodeDfaMask), maskBuffers, mask,
              {(uint64_t{packedVocab_->vocab} + 255) / 256, mask.rows,
               lanes},
              {256, 1, 1});
  }

  // The greedy DFA lanes' truncation pass, encoded between the unmasked
  // selection and the correction mask of the same commit graph: each lane
  // names the first proposal row the byte-walk or the argmax rejects and
  // writes its GPU forced-span prefix over the unmasked picks.
  void Runtime::Impl::encodeDfaTruncation(
      CommandGraph &graph, uint32_t lanes, uint32_t laneMask,
      std::span<const uint32_t> proposals,
      std::span<const uint32_t> gpuForced) {
    if (!laneMask)
      return;
    DfaTruncParams trunc{};
    trunc.verify_stride = geometry.draft.verifyRows();
    trunc.lane_mask = laneMask;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      trunc.proposals[lane] =
          lane < proposals.size() ? proposals[lane]
                                  : RICHENGINE_DRAFT_PROPOSAL_TOKENS;
      trunc.forced[lane] = lane < gpuForced.size() ? gpuForced[lane] : 0;
    }
    graph.add(std::string(ops::kDecodeDfaTrunc),
              {decodeArena->packed(DecodeTensor::DfaRowStates, lanes),
               decodeArena->packed(DecodeTensor::ProposedTokens, lanes),
               decodeArena->packed(DecodeTensor::OutputTokens, lanes),
               decodeArena->packed(DecodeTensor::DfaTrunc, lanes)},
              trunc, {lanes, 1, 1}, {1, 1, 1});
  }

  // The greedy DFA lanes' single next-state row each, written at the
  // truncation row's mask row: the masked argmax over it is the round's
  // correction (or bonus) token. Every other mask row stays unwritten —
  // no selection reads it this round.
  void Runtime::Impl::encodeDfaCorrectionMasks(
      CommandGraph &graph, std::span<Request *const> entries, uint32_t lanes,
      uint32_t laneMask) {
    if (!laneMask)
      return;
    if (!packedVocab_)
      throw std::logic_error("GPU-DFA lane has no packed vocabulary");
    const auto tables = dfaBuffers(entries, DfaBufferKind::Table);
    const auto accepts = dfaBuffers(entries, DfaBufferKind::Accept);
    const auto markers = dfaBuffers(entries, DfaBufferKind::Marker);
    DfaMaskParams mask{};
    mask.vocabulary = packedVocab_->vocab;
    mask.rows = 1;
    mask.lane_mask = laneMask;
    mask.indexed = 1;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      if (entries[lane]->dfa) {
        mask.states[lane] = entries[lane]->dfa->states;
        dfaLaneEos(mask, lane, *entries[lane]->dfa);
      }
    graph.add(std::string(ops::kDecodeDfaMask),
              {tables[0], tables[1], tables[2], tables[3],
               accepts[0], accepts[1], accepts[2], accepts[3],
               packedVocab_->bytes, packedVocab_->offsets,
               decodeArena->packed(DecodeTensor::DfaRowStates, lanes),
               decodeArena->packed(DecodeTensor::DfaNextStates, lanes),
               decodeArena->packed(DecodeTensor::DfaTrunc, lanes),
               markers[0], markers[1], markers[2], markers[3],
               packedVocab_->special},
              mask,
              {(uint64_t{packedVocab_->vocab} + 255) / 256, 1, lanes},
              {256, 1, 1});
  }

  // The masked argmax at each greedy DFA lane's truncation row, over the
  // next-state row the correction mask just wrote: the selection kernels
  // read it in place of a bitmask exactly as the full-mask path did.
  void Runtime::Impl::encodeDfaCorrections(
      CommandGraph &graph, std::span<Request *const> entries, uint32_t lanes,
      uint32_t laneMask, std::span<const uint32_t> liveRows) {
    if (!laneMask)
      return;
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      policies[lane] = samplingPolicy(*entries[lane]);
    ops::SamplingBuffers buffers = samplingBuffers(lanes);
    buffers.dfaLanes = laneMask;
    sampling.addDfaCorrections(
        graph, std::span(policies).first(lanes), buffers,
        decodeArena->packed(DecodeTensor::DfaTrunc, lanes),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1],
        liveRows, geometry.draft.verifyRows());
  }

  void Runtime::Impl::encodeDfaAdvance(
      CommandGraph &graph, std::span<Request *const> entries, uint32_t lanes) {
    const uint32_t laneMask = dfaLaneMask(entries);
    if (!laneMask)
      return;
    if (!packedVocab_)
      throw std::logic_error("GPU-DFA lane has no packed vocabulary");
    const auto tables = dfaBuffers(entries, DfaBufferKind::Table);
    const auto accepts = dfaBuffers(entries, DfaBufferKind::Accept);
    const auto markers = dfaBuffers(entries, DfaBufferKind::Marker);
    DfaWalkParams walk{};
    walk.verify_stride = geometry.draft.verifyRows();
    walk.lane_mask = laneMask;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (entries[lane]->dfa) {
        walk.states[lane] = entries[lane]->dfa->states;
        dfaLaneEos(walk, lane, *entries[lane]->dfa);
        // Forced mode runs no chain pass, so this is where a newer mirror
        // (provideDfa, a resume) reaches the state-lane slot; in Full mode
        // the mask encode already cleared the dirty flag.
        syncDfaState(*entries[lane]);
      }
      walk.state_lanes[lane] = entries[lane]->stateLane;
    }
    graph.add(std::string(ops::kDecodeDfaAdvance),
              {tables[0], tables[1], tables[2], tables[3],
               packedVocab_->bytes, packedVocab_->offsets,
               decodeArena->packed(DecodeTensor::InputTokens, lanes),
               decodeArena->packed(DecodeTensor::OutputTokens, lanes),
               decodeArena->packed(DecodeTensor::RetainedCount, lanes),
               decodeArena->packed(DecodeTensor::DfaState, kLaneCount),
               accepts[0], accepts[1], accepts[2], accepts[3],
               markers[0], markers[1], markers[2], markers[3],
               packedVocab_->special},
              walk, {lanes, 1, 1}, {1, 1, 1});
  }

  // The commit-stage forced-span derivation: after the state advanced, each
  // DFA lane's threadgroup follows the single-live-byte chain from the new
  // anchor's state into the state lane's DfaForcedSpan, then greedily
  // tokenizes it into the next round's proposal tokens.
  void Runtime::Impl::encodeDfaForced(
      CommandGraph &graph, std::span<Request *const> entries, uint32_t lanes) {
    const uint32_t laneMask = dfaLaneMask(entries);
    if (!laneMask)
      return;
    if (!packedVocab_)
      throw std::logic_error("GPU-DFA lane has no packed vocabulary");
    const auto tables = dfaBuffers(entries, DfaBufferKind::Table);
    const auto accepts = dfaBuffers(entries, DfaBufferKind::Accept);
    const auto markers = dfaBuffers(entries, DfaBufferKind::Marker);
    DfaWalkParams walk{};
    walk.verify_stride = geometry.draft.verifyRows();
    walk.lane_mask = laneMask;
    walk.vocabulary = packedVocab_->vocab;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (entries[lane]->dfa) {
        walk.states[lane] = entries[lane]->dfa->states;
        dfaLaneEos(walk, lane, *entries[lane]->dfa);
      }
      walk.state_lanes[lane] = entries[lane]->stateLane;
    }
    const MetalBuffer forced =
        decodeArena->packed(DecodeTensor::DfaForced, kLaneCount);
    graph.add(std::string(ops::kDecodeDfaForcedBytes),
              {tables[0], tables[1], tables[2], tables[3],
               packedVocab_->bytes, packedVocab_->offsets,
               decodeArena->packed(DecodeTensor::OutputTokens, lanes),
               decodeArena->packed(DecodeTensor::RetainedCount, lanes),
               decodeArena->packed(DecodeTensor::DfaState, kLaneCount),
               forced,
               accepts[0], accepts[1], accepts[2], accepts[3],
               markers[0], markers[1], markers[2], markers[3],
               packedVocab_->special},
              walk, {lanes, 1, 1}, {1024, 1, 1});
    graph.add(std::string(ops::kDecodeDfaForcedTokens),
              {tables[0], tables[1], tables[2], tables[3],
               packedVocab_->bytes, packedVocab_->offsets, forced,
               decodeArena->packed(DecodeTensor::ProposedTokens, lanes),
               markers[0], markers[1], markers[2], markers[3],
               packedVocab_->special},
              walk, {lanes, 1, 1}, {1024, 1, 1});
  }

  // The merged constrained step's on-device span apply
  // (RICHENGINE_DFA_MERGE): between the draft pass and the verify input of
  // the same command, each named lane copies its state lane's pending
  // DfaForcedSpan tokens over the draft's own proposals and clears the
  // span — what the split path's host apply writes after the draft command
  // completes.
  void Runtime::Impl::encodeDfaSpanApply(
      CommandGraph &graph, std::span<Request *const> entries, uint32_t lanes,
      uint32_t laneMask, std::span<const uint32_t> maximumRetained) {
    if (!laneMask)
      return;
    DfaSpanApplyParams params{};
    params.lane_mask = laneMask;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      params.state_lanes[lane] = entries[lane]->stateLane;
      params.maximum_retained[lane] =
          lane < maximumRetained.size() ? maximumRetained[lane]
                                        : geometry.draft.verifyRows();
    }
    graph.add(std::string(ops::kDecodeDfaSpanApply),
              {decodeArena->packed(DecodeTensor::DfaForced, kLaneCount),
               decodeArena->packed(DecodeTensor::ProposedTokens, lanes)},
              params, {lanes, 1, 1}, {1, 1, 1});
  }

  void Runtime::Impl::encodeDfaInitialMasks(
      CommandGraph &graph, std::span<Request *const> entries, uint32_t lanes) {
    // Forced mode's initial masks come from the host like any
    // host-constrained lane's.
    if (dfaMode_ != DfaMode::Full)
      return;
    const uint32_t laneMask = dfaLaneMask(entries);
    if (!laneMask)
      return;
    if (!packedVocab_)
      throw std::logic_error("GPU-DFA lane has no packed vocabulary");
    // The chain kernel's row-0 input is the mirrored state; deeper rows are
    // unused by the initial selection's single mask row.
    uint32_t *rowStates = contents<uint32_t>(
        decodeArena->packed(DecodeTensor::DfaRowStates, lanes),
        "dfa row states");
    const auto tables = dfaBuffers(entries, DfaBufferKind::Table);
    const auto accepts = dfaBuffers(entries, DfaBufferKind::Accept);
    const auto markers = dfaBuffers(entries, DfaBufferKind::Marker);
    DfaMaskParams mask{};
    mask.vocabulary = packedVocab_->vocab;
    mask.rows = 1;
    mask.lane_mask = laneMask;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!entries[lane] || !entries[lane]->dfa)
        continue;
      mask.states[lane] = entries[lane]->dfa->states;
      mask.live_rows[lane] = 1;
      dfaLaneEos(mask, lane, *entries[lane]->dfa);
      rowStates[lane * (kDecodeRows + 1)] = entries[lane]->dfaState;
    }
    const MetalBuffer nextStates =
        decodeArena->packed(DecodeTensor::DfaNextStates, lanes);
    graph.add(std::string(ops::kDecodeDfaMask),
              {tables[0], tables[1], tables[2], tables[3],
               accepts[0], accepts[1], accepts[2], accepts[3],
               packedVocab_->bytes, packedVocab_->offsets,
               decodeArena->packed(DecodeTensor::DfaRowStates, lanes),
               nextStates,
               decodeArena->packed(DecodeTensor::DfaTrunc, lanes),
               markers[0], markers[1], markers[2], markers[3],
               packedVocab_->special},
              mask,
              {(uint64_t{packedVocab_->vocab} + 255) / 256, 1, lanes},
              {256, 1, 1});
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
    targetModel.addStateCommit(graph, std::move(buffers), width,
                               geometry.draft.verifyRows());
  }

} // namespace richengine::model

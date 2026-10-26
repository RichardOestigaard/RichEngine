#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  void Runtime::Impl::prepareDecodeLane(Request &entry, const ModelBatchItem &item,
                         uint32_t lane) {
    if (!entry.resident || !entry.promptComplete || !entry.pendingToken) {
      throw std::logic_error("decode request is not ready");
    }
    const LaneMetadata &metadata = states.metadata(entry.stateLane);
    if (metadata.lengths.targetTokens != item.logicalPosition ||
        !metadata.lengths.hasCompleteDraftWindow(geometry.draft.draftWindow())) {
      throw std::logic_error("decode state length is not exact");
    }
    static_cast<void>(synchronizedPageTable(entry, item));
    // The draft input packs at the draft's query stride, so its embedding
    // gathers draft.queryRows() rows per lane; the anchor occupies row 0.
    const uint32_t queryRows = geometry.draft.queryRows();
    auto *draftInput = contents<uint32_t>(
        decodeArena->packed(DecodeTensor::DraftInputTokens, kLaneCount),
        "draft input tokens") +
        lane * queryRows;
    draftInput[0] = *entry.pendingToken;
    std::fill(draftInput + 1, draftInput + queryRows,
              geometry.target.maskToken);

    auto *positions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Positions),
                           "decode RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::DraftPositions),
                           "decode draft RoPE positions");
    for (uint32_t row = 0; row < kDecodeRows; ++row) {
      const std::array<uint32_t, 3> rotary =
          ropePosition(entry, item.logicalPosition + row);
      std::copy(rotary.begin(), rotary.end(), positions + row * 3);
      // The draft is a text model over logical positions.
      draftPositions[row] = static_cast<uint32_t>(item.logicalPosition + row);
    }
  }

  void Runtime::Impl::bindDraftRings(
      std::span<Request *const> entries,
      std::vector<std::array<MetalBuffer, kLaneCount>> &keys,
      std::vector<std::array<MetalBuffer, kLaneCount>> &values) const {
    keys.resize(geometry.draft.layers);
    values.resize(geometry.draft.layers);
    for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer) {
      for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
        const auto &ring =
            states.draft(laneEntry(entries, lane).stateLane)[layer];
        keys[layer][lane] = ring.keys;
        values[layer][lane] = ring.values;
      }
    }
  }

  void Runtime::Impl::encodeDraftBatchGraph(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const uint64_t> logicalPositions) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != logicalPositions.size()) {
      throw std::invalid_argument("invalid draft decode batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    // The draft shares the target's vocabulary head and its storage rows.
    const uint32_t storage = targetModel.decodeStorageLanes(lanes);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, storage);
    };
    std::array<uint32_t, kLaneCount> cacheLengths{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      cacheLengths[lane] = static_cast<uint32_t>(logicalPositions[lane]);

    DFlashDecodeBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    for (uint32_t hidden = 0; hidden < buffers.hidden.size(); ++hidden) {
      buffers.hidden[hidden] = d(static_cast<DecodeTensor>(
          static_cast<uint32_t>(DecodeTensor::DraftHidden0) + hidden));
    }
    buffers.normalized = d(DecodeTensor::DraftNormalized);
    buffers.dynamic = d(DecodeTensor::DraftDynamic);
    buffers.convolved = d(DecodeTensor::DraftConvolved);
    buffers.proposalQkv = d(DecodeTensor::DraftProposalQkv);
    buffers.attention = d(DecodeTensor::DraftAttention);
    buffers.projected = d(DecodeTensor::DraftProjected);
    buffers.residual = d(DecodeTensor::DraftResidual);
    buffers.intermediate = d(DecodeTensor::DraftIntermediate);
    buffers.finalHidden = d(DecodeTensor::DraftFinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.selectorHidden = d(DecodeTensor::SelectorHidden);
    buffers.queryKeys = d(DecodeTensor::DraftQueryKeys);
    buffers.queryValues = d(DecodeTensor::DraftQueryValues);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.gateScratch = decodeArena->gateScratch();
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    std::visit([&](const auto &model) {
      model.addDecode(graph, std::move(buffers),
                      targetModel.vocabularyProjection(),
                      std::span(cacheLengths).first(lanes));
    }, draftModel);
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    uint32_t treeMask = 0;
    const bool treeCapable = treeDraftCapable();
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      Request &entry = laneEntry(entries, lane);
      if (!entry.pendingToken)
        throw std::invalid_argument("draft batch lane has no anchor");
      anchors[lane] = *entry.pendingToken;
      policies[lane] = samplingPolicy(entry);
      if (treeCapable && !policies[lane].samples() &&
          entry.constraint != ConstraintMode::TokenMask &&
          !policies[lane].penalties.active())
        treeMask |= uint32_t{1} << lane;
    }
    std::visit([&](const auto &model) {
      model.addSelection(
          graph,
          {d(DecodeTensor::Logits), d(DecodeTensor::TopPartialIds),
           d(DecodeTensor::TopPartialValues), d(DecodeTensor::Candidates),
           d(DecodeTensor::Unary), d(DecodeTensor::SelectorHidden),
           d(DecodeTensor::SamplingUniforms), d(DecodeTensor::ProposedTokens),
           d(DecodeTensor::ProposalProbs), d(DecodeTensor::VerifyConfidence),
           d(DecodeTensor::TreeNodes),
           d(DecodeTensor::TreeTokens), d(DecodeTensor::TreeCounts)},
          std::span(anchors).first(lanes), std::span(policies).first(lanes),
          treeMask);
    }, draftModel);
    // RICHENGINE_DRAFT_CONF_MIN: the DSpark confidence head rewrites the
    // lane's proposal confidences from the selector's top-candidate mass
    // to the head's own sigmoid, which draft_verify_rows's conf_min
    // prunes on. The GPU predraft's graph scores its proposals the same
    // way, so an adopted predraft's confidences stay valid.
    if (tuning().draftConfMin > 0.0) {
      if (const auto *dspark = std::get_if<DSparkDraft>(&draftModel)) {
        const DSparkDraftWeights &w = dspark->weights();
        graph.add(
            std::string(ops::kDraftConfScore),
            {d(DecodeTensor::DraftFinalHidden), d(DecodeTensor::ProposedTokens),
             d(DecodeTensor::DraftInputTokens), w.markovEmbedding,
             w.confidenceWeight, w.confidenceBias,
             d(DecodeTensor::VerifyConfidence)},
            DraftConfScoreParams{lanes, w.layout.hiddenSize, w.layout.markovRank,
                                 w.layout.vocabularySize,
                                 w.layout.queryRows()},
            {uint64_t{lanes} * RICHENGINE_DRAFT_PROPOSAL_TOKENS, 1, 1},
            {32, 1, 1});
      }
    }
  }

  // The speculative next-step draft: the step's own commit has already run,
  // so the rings hold every committed context row and the lanes' next
  // anchors and positions are host-known. The draft's own graph replays
  // unchanged — embedding, the baked draft span and the selector — writing
  // ProposedTokens the next decode adopts or overwrites.
  void Runtime::Impl::launchDraftPredraft(
      std::span<const DecodeLaneResult> lanes,
      std::span<const ModelBatchItem> items) {
    if (!gpuPredraftOn_ || std::holds_alternative<NullDraft>(draftModel))
      return;
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    GpuPredraft predraft;
    std::array<Request *, kLaneCount> entries{};
    std::array<uint64_t, kLaneCount> logicalPositions{};
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = *lanes[lane].request;
      const ops::SamplingPolicy policy = samplingPolicy(entry);
      // Any lane that cannot take a chain draft vetoes the batch: finished
      // or terminal anchors leave no next step to draft, a failed lane
      // committed nothing to draft from, and a sampled, constrained or
      // penalized policy could change before adoption.
      if (!lanes[lane].failure.empty() || !entry.resident ||
          !entry.pendingToken ||
          entry.generatedTokens >= entry.maxNewTokens ||
          isStopToken(geometry, *entry.pendingToken) || policy.samples() ||
          policy.constrained || policy.penalties.active())
        return;
      predraft.requestIds[lane] = entry.id;
      predraft.anchors[lane] = *entry.pendingToken;
      predraft.positions[lane] =
          items[lane].logicalPosition + lanes[lane].retained;
      entries[lane] = &entry;
      logicalPositions[lane] = predraft.positions[lane];
      packDraftPredraftLane(lane, *entry.pendingToken,
                            predraft.positions[lane]);
    }
    static const bool pdTiming = envFlag("RICHENGINE_DECODE_TIMING");
    const auto p0 = std::chrono::steady_clock::now();
    CommandGraph graph;
    addRopeTables(
        graph, decodeArena->packed(DecodeTensor::Positions, width), 0,
        decodeArena->packed(DecodeTensor::DraftPositions, width),
        width * kDecodeRows,
        decodeArena->packed(DecodeTensor::RopeCos, width),
        decodeArena->packed(DecodeTensor::RopeSin, width),
        decodeArena->packed(DecodeTensor::DraftRopeCos, width),
        decodeArena->packed(DecodeTensor::DraftRopeSin, width));
    encodeBatchEmbedding(graph, DecodeTensor::DraftInputTokens,
                         DecodeTensor::DraftHidden0, width, 1,
                         geometry.draft.queryRows());
    encodeDraftBatchGraph(graph, {entries.data(), width},
                          {logicalPositions.data(), width});
    const auto p1 = std::chrono::steady_clock::now();
    predraft.epoch = arenaEpoch_;
    predraft.lanes = width;
    predraft.ticket = backend.submitCommandAsync(graph.dispatches());
    if (pdTiming) {
      const auto p2 = std::chrono::steady_clock::now();
      fprintf(stderr, "predraft-detail build=%.3fms submit=%.3fms\n",
              std::chrono::duration<double, std::milli>(p1 - p0).count(),
              std::chrono::duration<double, std::milli>(p2 - p1).count());
    }
    predraft.submitted = true;
    gpuPredraft_ = std::move(predraft);
  }

  bool Runtime::Impl::adoptGpuPredraft(std::span<Request *const> entries,
                                       std::span<const ModelBatchItem> items,
                                       uint32_t width, bool allowed) {
    // armed, not submitted: the ticket may already be consumed by a
    // settleGpuPredraft on this decode path — the lane bookkeeping is the
    // proposal proof either way.
    if (!gpuPredraft_.lanes)
      return false;
    settleGpuPredraft();
    GpuPredraft predraft = std::move(gpuPredraft_);
    gpuPredraft_ = GpuPredraft{};
    // The epoch proves no other decode wrote the arena's proposal buffers
    // since the predraft ran; the per-lane match proves the same requests
    // continue at the anchors and positions it assumed.
    if (!allowed || width != predraft.lanes ||
        arenaEpoch_ != predraft.epoch)
      return false;
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = *entries[lane];
      const ops::SamplingPolicy policy = samplingPolicy(entry);
      if (entry.id != predraft.requestIds[lane] || !entry.pendingToken ||
          *entry.pendingToken != predraft.anchors[lane] ||
          items[lane].logicalPosition != predraft.positions[lane] ||
          policy.samples() || policy.constrained || policy.penalties.active()) {
        if (draftDebug_)
          fprintf(stderr, "gpu-predraft miss lane=%u id=%llu/%llu\n", lane,
                  (unsigned long long)entry.id,
                  (unsigned long long)predraft.requestIds[lane]);
        return false;
      }
    }
    if (draftDebug_)
      fprintf(stderr, "gpu-predraft adopted lanes=%u\n", width);
    return true;
  }

  void Runtime::Impl::encodeDraftStateCommitBatch(CommandGraph &graph,
                                   std::span<Request *const> entries,
                                   std::span<const ModelBatchItem> items,
                                   bool tree) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid draft state commit batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, lanes);
    };
    // A tree batch captured the emitted rows in DFS order; the context
    // commit consumes the retained path's rows in path order.
    if (tree) {
      if (treeSkip_.find('x') == std::string::npos)
        ops::Embedding::addTreeCaptureGather(
            graph, d(DecodeTensor::CapturedPath),
            d(DecodeTensor::RetainedPath), d(DecodeTensor::RetainedCount),
            d(DecodeTensor::CapturedTargetHidden),
            geometry.draft.targetHiddenSize, lanes);
    }

    std::array<uint32_t, kLaneCount> startPositions{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      startPositions[lane] = static_cast<uint32_t>(items[lane].logicalPosition);
    DFlashContextBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.projected = d(DecodeTensor::ContextProjected);
    buffers.hidden = d(DecodeTensor::ContextHidden);
    buffers.contextKv = d(DecodeTensor::ContextKv);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.retainedCounts = d(DecodeTensor::RetainedCount);
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    std::visit([&](const auto &model) {
      model.addContextCommit(graph, std::move(buffers),
                             std::span(startPositions).first(lanes));
    }, draftModel);
  }

} // namespace richengine::model

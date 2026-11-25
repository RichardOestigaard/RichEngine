#include "model/RuntimeImpl.hpp"
#include "AwakeClock.hpp"

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
    entry.stateCapture = item.stateCapture;
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
    // A draft-less geometry's ring has no layers: the layout still
    // reports its default count, so stop at what the ring holds.
    const uint32_t layers = std::min<uint32_t>(
        geometry.draft.layers,
        states.draft(laneEntry(entries, 0).stateLane).size());
    keys.resize(layers);
    values.resize(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
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
                             std::span<const uint64_t> logicalPositions,
                             const MetalBuffer &devicePack) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != logicalPositions.size()) {
      throw std::invalid_argument("invalid draft decode batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    // The draft shares the target's vocabulary head and its storage rows;
    // its own AF4-block projections pad to the staged tile the same way
    // the target's do, which a target-only bound misses on affine targets.
    const uint32_t storage = std::max(
        targetModel.decodeStorageLanes(lanes),
        DecodeArena::draftStorageLanes(geometry, operators, lanes));
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
    buffers.inputTokens = d(DecodeTensor::DraftInputTokens);
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
    if (geometry.draft.kind == DraftKind::Mtp) {
      // The MTP draft's positions, per-state-lane stream anchors and
      // pass scratch; the MoE fields take the draft block's own shape.
      buffers.positions = d(DecodeTensor::DraftPositions);
      for (uint32_t lane = 0; lane < lanes; ++lane)
        buffers.mtpAnchors[lane] = decodeArena->get(
            laneEntry(entries, lane).stateLane, DecodeTensor::MtpAnchor);
      buffers.mtp.moe = decodeArena->moeScratch(storage);
      buffers.mtp.streams = d(DecodeTensor::MtpStreams);
      buffers.mtp.eh = d(DecodeTensor::MtpEh);
      buffers.mtp.ehOut = d(DecodeTensor::MtpEhOut);
      buffers.mtp.plane = d(DecodeTensor::MtpPlane);
      buffers.mtp.mappings = d(DecodeTensor::MtpMappings);
      buffers.mtp.headWeights = d(DecodeTensor::MtpHeadWeights);
      buffers.mtp.embed = d(DecodeTensor::MtpEmbed);
      buffers.mtp.embedNormed = d(DecodeTensor::MtpEmbedNormed);
      buffers.mtp.aggregate = d(DecodeTensor::MtpAggregate);
      buffers.mtp.normed = d(DecodeTensor::MtpNormed);
      buffers.mtp.queries = d(DecodeTensor::MtpQueries);
      buffers.mtp.keys = d(DecodeTensor::MtpKeys);
      buffers.mtp.values = d(DecodeTensor::MtpValues);
      buffers.mtp.attention = d(DecodeTensor::MtpAttention);
      buffers.mtp.branch = d(DecodeTensor::MtpBranch);
      buffers.mtp.shared = d(DecodeTensor::MtpShared);
      buffers.mtp.intermediate = d(DecodeTensor::MtpIntermediate);
      buffers.mtp.moeRoutes = d(DecodeTensor::MtpMoeRoutes);
      buffers.mtp.moePartials = d(DecodeTensor::MtpMoePartials);
      buffers.mtp.head = d(DecodeTensor::MtpHead);
      buffers.mtp.logits = d(DecodeTensor::MtpLogits);
      buffers.mtp.tokenIn = d(DecodeTensor::MtpTokenIn);
    }
    std::visit([&](const auto &model) {
      using DraftModelType = std::decay_t<decltype(model)>;
      // Every GPU draft whose decode binds the step pack: the V1/DSpark
      // variants and the DFlash2 width subclasses (its selectors take the
      // anchor from the draft-input rows the pack staged, like DSpark's).
      if constexpr (std::is_same_v<DraftModelType, DFlashV1Draft> ||
                    std::is_same_v<DraftModelType, DSparkDraft> ||
                    std::is_same_v<DraftModelType, DSplineDraft> ||
                    std::is_base_of_v<DFlashDraft, DraftModelType>) {
        model.addDecode(graph, std::move(buffers),
                        targetModel.vocabularyProjection(),
                        std::span(cacheLengths).first(lanes), devicePack);
      } else if constexpr (std::is_same_v<DraftModelType,
                                          MaxwellMtpDraft> ||
                           std::is_same_v<DraftModelType,
                                          NemotronMtpDraft>) {
        // The MTP chain costs a full decoder layer per pass, so each lane
        // runs only as deep as its accepted-token EWMA earned — the same
        // proposalBudget the verify row cap already follows.
        std::array<uint32_t, kLaneCount> passLimits{};
        for (uint32_t lane = 0; lane < lanes; ++lane)
          passLimits[lane] =
              adaptiveProposals_
                  ? std::clamp(laneEntry(entries, lane).proposalBudget, 1u,
                               geometry.draft.proposalLimit())
                  : geometry.draft.proposalLimit();
        model.addDecode(graph, std::move(buffers),
                        targetModel.vocabularyProjection(),
                        std::span(cacheLengths).first(lanes),
                        std::span(passLimits).first(lanes));
      } else {
        model.addDecode(graph, std::move(buffers),
                        targetModel.vocabularyProjection(),
                        std::span(cacheLengths).first(lanes));
      }
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
           d(DecodeTensor::TreeTokens), d(DecodeTensor::TreeCounts),
           d(DecodeTensor::DraftInputTokens)},
          std::span(anchors).first(lanes), std::span(policies).first(lanes),
          treeMask);
    }, draftModel);
    // RICHENGINE_DRAFT_CONF_MIN: the DSpark confidence head rewrites the
    // lane's proposal confidences from the selector's top-candidate mass
    // to the head's own sigmoid, which draft_verify_rows's conf_min
    // prunes on. The GPU predraft's graph scores its proposals the same
    // way, so an adopted predraft's confidences stay valid.
    if (modelPolicy().draftConfMin > 0.0) {
      if (const auto *dspark = std::get_if<DSparkDraft>(&draftModel)) {
        const DSparkDraftWeights &w = dspark->weights();
        // A draft without a confidence head (enable_confidence_head off —
        // Nemotron's DSpark draft) leaves the proposals' selector-mass
        // confidences in place; nothing rewrites them.
        if (!w.layout.confidenceHead)
          return;
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
    const auto p0 = richengine::AwakeClock::now();
    CommandGraph graph;
    addRopeTables(
        graph, decodeArena->packed(DecodeTensor::Positions, width), 0,
        decodeArena->packed(DecodeTensor::DraftPositions, width),
        width * kDecodeRows,
        decodeArena->packed(DecodeTensor::RopeCos, width),
        decodeArena->packed(DecodeTensor::RopeSin, width),
        decodeArena->packed(DecodeTensor::DraftRopeCos, width),
        decodeArena->packed(DecodeTensor::DraftRopeSin, width));
    encodeDraftInputEmbedding(graph, DecodeTensor::DraftInputTokens,
                         DecodeTensor::DraftHidden0, width, 1,
                         geometry.draft.queryRows());
    encodeDraftBatchGraph(graph, {entries.data(), width},
                          {logicalPositions.data(), width});
    const auto p1 = richengine::AwakeClock::now();
    predraft.epoch = arenaEpoch_;
    predraft.lanes = width;
    // The overlap variant raises a queue-level event at the tail: the next
    // decode's verify submits behind this command and waits on the signal
    // — Metal 4 commits are not implicitly serialized — while its host
    // encode already ran during this draft's execution.
    if (predraftOverlap_) {
      if (!predraftEvent_)
        predraftEvent_ = backend.newSharedEvent();
      const EventStep step{.before = graph.dispatches().size(),
                           .event = predraftEvent_,
                           .value = ++predraftEventValue_,
                           .kind = EventStep::Kind::Signal};
      const EventStep steps[1] = {step};
      const metal::Command command{graph.dispatches(),
                                   std::span<const EventStep>(steps)};
      predraft.ticket = backend.submitCommandAsync(command);
      predraft.waitValue = predraftEventValue_;
    } else {
      predraft.ticket = backend.submitCommandAsync(graph.dispatches());
    }
    if (pdTiming) {
      const auto p2 = richengine::AwakeClock::now();
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
    // Armed, maybe still in flight (RICHENGINE_PREDRAFT_OVERLAP keeps the
    // ticket pending through this cycle's encode and submit) or already
    // drained by a settleGpuPredraft on this decode path — the adoption
    // proof is host facts only: the epoch, the request ids and the anchors
    // and positions it assumed. Its writes arrive through queue order,
    // never through the host's wait.
    if (!gpuPredraft_.lanes)
      return false;
    const GpuPredraft &predraft = gpuPredraft_;
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
    if (geometry.draft.kind == DraftKind::Mtp) {
      // The MTP commit's verify output tokens, anchors and row scratch.
      buffers.outputTokens = d(DecodeTensor::OutputTokens);
      for (uint32_t lane = 0; lane < lanes; ++lane)
        buffers.mtpAnchors[lane] = decodeArena->get(
            laneEntry(entries, lane).stateLane, DecodeTensor::MtpAnchor);
      // The commit's projections stage lanes×verifyRows rows to the
      // GGUF decode tile (LinearGguf.cpp stagedTileRows): bind the slots
      // the staged plan covers, which a three-lane batch under-packs.
      const uint32_t commitRows = lanes * ExecutionLimits::targetVerifyRows;
      const uint32_t staged = commitRows <= 8   ? 8
                              : commitRows <= 16 ? 16
                              : commitRows <= 32 ? 32
                                                 : 64;
      const uint32_t ctxLanes =
          (staged + ExecutionLimits::targetVerifyRows - 1) /
          ExecutionLimits::targetVerifyRows;
      auto dc = [&](DecodeTensor tensor) {
        return decodeArena->packed(tensor, ctxLanes);
      };
      buffers.mtp.streams = dc(DecodeTensor::MtpCtxStreams);
      buffers.mtp.eh = dc(DecodeTensor::MtpCtxEh);
      buffers.mtp.ehOut = dc(DecodeTensor::MtpCtxEhOut);
      buffers.mtp.plane = dc(DecodeTensor::MtpCtxPlane);
      buffers.mtp.mappings = dc(DecodeTensor::MtpCtxMappings);
      buffers.mtp.embed = dc(DecodeTensor::MtpCtxEmbed);
      buffers.mtp.embedNormed = dc(DecodeTensor::MtpCtxEmbedNormed);
      buffers.mtp.keys = dc(DecodeTensor::MtpCtxKeys);
      buffers.mtp.values = dc(DecodeTensor::MtpCtxValues);
    }
    std::visit([&](const auto &model) {
      model.addContextCommit(graph, std::move(buffers),
                             std::span(startPositions).first(lanes));
    }, draftModel);
  }

  // The chained predraft (B-prime): appended to the committed step's
  // command after decode_step_pack, the draft input pack, rope tables,
  // embedding, forward and selector all read the pack lanes the step just
  // wrote — no host readback stands between this step and the next
  // draft. Bookkeeping arms here; the facts the encode could not know
  // (anchors, positions) land at completeGpuPredraft.
  void Runtime::Impl::encodeChainedPredraft(
      CommandGraph &graph, std::span<Request *const> entries,
      std::span<const ModelBatchItem> items) {
    static_cast<void>(items);
    // The chainable drafts: variants whose decode binds the step pack for
    // its cache lengths and whose selector takes the anchor from the
    // pack-staged draft-input rows instead of the encode's host constants.
    // The DFlash2 width subclasses cover the base via is_base_of.
    const bool chainable = std::visit([](const auto &model) {
      using DraftModelType = std::decay_t<decltype(model)>;
      return std::is_same_v<DraftModelType, DFlashV1Draft> ||
             std::is_same_v<DraftModelType, DSparkDraft> ||
             std::is_same_v<DraftModelType, DSplineDraft> ||
             std::is_base_of_v<DFlashDraft, DraftModelType>;
    }, draftModel);
    if (!gpuPredraftOn_ || !chainPredraft_ || !chainable ||
        entries.empty() || entries.size() > kLaneCount)
      return;
    const uint32_t width = static_cast<uint32_t>(entries.size());
    GpuPredraft predraft;
    for (uint32_t lane = 0; lane < width; ++lane) {
      const Request &entry = *entries[lane];
      const ops::SamplingPolicy policy = samplingPolicy(entry);
      // The launch path's policy vetoes, evaluated one step earlier: a
      // lane that cannot take a chain draft vetoes the batch. Anchors,
      // positions and terminal status are the pack's job — the host
      // proves or drops the arm at finalize.
      if (!entry.resident || !entry.pendingToken || policy.samples() ||
          policy.constrained || policy.penalties.active())
        return;
      predraft.requestIds[lane] = entry.id;
    }
    const DraftPredraftPackParams params{
        width, geometry.draft.queryRows(), kDecodeRows,
        geometry.target.maskToken};
    graph.add(std::string(ops::kDraftPredraftPack),
              {stepPackBuffer(),
               decodeArena->packed(DecodeTensor::DraftInputTokens,
                                   kLaneCount),
               decodeArena->packed(DecodeTensor::DraftPositions,
                                   kLaneCount)},
              params,
              {std::max(params.query_rows, params.decode_rows), width, 1});
    // The target's positions are stale here — the draft forward reads
    // only its own tables, so the target rope slots stay untouched (a
    // zero target row count keeps the pass draft-only the way
    // launchDraftPredraft calls it).
    addRopeTables(
        graph, decodeArena->packed(DecodeTensor::Positions, width), 0,
        decodeArena->packed(DecodeTensor::DraftPositions, width),
        width * kDecodeRows,
        decodeArena->packed(DecodeTensor::RopeCos, width),
        decodeArena->packed(DecodeTensor::RopeSin, width),
        decodeArena->packed(DecodeTensor::DraftRopeCos, width),
        decodeArena->packed(DecodeTensor::DraftRopeSin, width));
    encodeDraftInputEmbedding(graph, DecodeTensor::DraftInputTokens,
                         DecodeTensor::DraftHidden0, width, 1,
                         geometry.draft.queryRows());
    std::array<uint64_t, kLaneCount> zeroPositions{};
    encodeDraftBatchGraph(graph, {entries.data(), width},
                          {zeroPositions.data(), width}, stepPackBuffer());
    predraft.epoch = arenaEpoch_;
    predraft.lanes = width;
    predraft.chained = true;
    gpuPredraft_ = std::move(predraft);
    if (draftDebug_)
      fprintf(stderr, "chained predraft armed lanes=%u\n", width);
  }

  void Runtime::Impl::completeGpuPredraft(
      std::span<const DecodeLaneResult> lanes,
      std::span<const ModelBatchItem> items) {
    if (!gpuPredraft_.chained) {
      launchDraftPredraft(lanes, items);
      return;
    }
    GpuPredraft predraft = std::move(gpuPredraft_);
    gpuPredraft_ = GpuPredraft{};
    if (lanes.size() != predraft.lanes || items.size() != lanes.size())
      return;
    // The chained pass ran on the pack's device values; the launch
    // path's vetoes — terminal anchors, failed lanes, policy changes —
    // apply at readback, and a dropped arm leaves nothing to adopt.
    for (uint32_t lane = 0; lane < predraft.lanes; ++lane) {
      Request &entry = *lanes[lane].request;
      const ops::SamplingPolicy policy = samplingPolicy(entry);
      if (!lanes[lane].failure.empty() || !entry.resident ||
          !entry.pendingToken ||
          entry.generatedTokens >= entry.maxNewTokens ||
          isStopToken(geometry, *entry.pendingToken) || policy.samples() ||
          policy.constrained || policy.penalties.active() ||
          entry.id != predraft.requestIds[lane])
        return;
      predraft.anchors[lane] = *entry.pendingToken;
      predraft.positions[lane] =
          items[lane].logicalPosition + lanes[lane].retained;
    }
    gpuPredraft_ = std::move(predraft);
  }

} // namespace richengine::model

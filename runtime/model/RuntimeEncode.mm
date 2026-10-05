#include "model/RuntimeImpl.hpp"

namespace splash::model {

  void Runtime::Impl::addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin) const {
    ops::RoPE::addTables(
        graph, std::move(targetPositions), std::move(draftPositions),
        prefillArena->get(PrefillTensor::TargetInverseFrequencies),
        prefillArena->get(PrefillTensor::DraftInverseFrequencies),
        std::move(targetCos), std::move(targetSin), std::move(draftCos),
        std::move(draftSin),
        {targetRows, draftRows, geometry.target.rotaryPairs,
         geometry.target.ropeAxes},
        kPrefillRows);
  }

  void Runtime::Impl::prepareDecodeLane(Request &entry, const ModelBatchItem &item,
                         uint32_t lane) {
    if (!entry.resident || !entry.promptComplete || !entry.pendingToken) {
      throw std::logic_error("decode request is not ready");
    }
    const QwenLaneMetadata &metadata = states.metadata(entry.stateLane);
    if (metadata.lengths.targetTokens != item.logicalPosition ||
        !metadata.lengths.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::logic_error("decode state length is not exact");
    }
    static_cast<void>(synchronizedPageTable(entry, item));
    auto *draftInput = contents<uint32_t>(
        decodeArena->get(lane, DecodeTensor::DraftInputTokens),
        "draft input tokens");
    draftInput[0] = *entry.pendingToken;
    std::fill(draftInput + 1, draftInput + kDecodeRows,
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
    const bool treeCapable =
        verifyTreeEnabled &&
        std::holds_alternative<DFlashDraft>(draftModel);
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
           d(DecodeTensor::ProposalProbs), d(DecodeTensor::TreeNodes),
           d(DecodeTensor::TreeTokens), d(DecodeTensor::TreeCounts)},
          std::span(anchors).first(lanes), std::span(policies).first(lanes),
          treeMask);
    }, draftModel);
  }

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
    QwenTargetVerifyBuffers buffers;
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
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = d(DecodeTensor::FinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
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
    for (uint32_t lane = 0; lane < lanes; ++lane)
      chunks[lane] =
          tree ? ops::PagedAttention::verifyTreeParams(
                     items[lane].logicalPosition,
                     static_cast<uint32_t>(items[lane].pageTable.size()))
               : ops::PagedAttention::verifyParams(
                     items[lane].logicalPosition,
                     static_cast<uint32_t>(items[lane].pageTable.size()));
    bindPageTables(entries, buffers.pageTables);
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
                          liveRows);
  }

  // The sampling buffers of a tree batch: lane-semantic tensors keep their
  // lanes' strides, while the row-indexed logits and the argmax output take
  // the tree's SPLASH_TREE_VERIFY_NODES stride.
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
    sampling.addVerify(graph, std::span(policies).first(lanes),
                       samplingBuffers(lanes), geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1],
                       {penaltyTable, std::span(stateLanes).first(lanes)},
                       liveRows);
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
    sampling.addAcceptance(
        graph,
        {decodeArena->packed(DecodeTensor::ProposedTokens, width),
         decodeArena->packed(DecodeTensor::Candidates, width),
         decodeArena->packed(DecodeTensor::ProposalProbs, width),
         decodeArena->packed(DecodeTensor::TargetVocabularyRows, width),
         decodeArena->packed(DecodeTensor::SamplingUniforms, width),
         decodeArena->packed(DecodeTensor::OutputTokens, width),
         decodeArena->packed(DecodeTensor::RetainedCount, width),
         decodeArena->packed(DecodeTensor::AcceptedCount, width)},
        maximumRetained, std::span(policies).first(width),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1],
        proposals);
  }

  void Runtime::Impl::encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes,
                            uint32_t rowFactor) {
    if (!lanes || lanes * rowFactor > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t storage = lanes * rowFactor;
    const uint32_t rows = lanes * rowFactor * kDecodeRows;
    targetModel.addEmbedding(graph, decodeArena->packed(tokens, storage),
                             decodeArena->packed(output, storage), rows);
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
    uint32_t base[SPLASH_MAXIMUM_BATCH_WIDTH][3]{};
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
    for (uint32_t lane = 0; lane < lanes; ++lane)
      chunks[lane] = ops::PagedAttention::verifyTreeParams(
          items[lane].logicalPosition,
          static_cast<uint32_t>(items[lane].pageTable.size()));
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
    QwenTargetCommitBuffers buffers{
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

} // namespace splash::model

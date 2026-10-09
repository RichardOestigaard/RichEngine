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
           d(DecodeTensor::ProposalProbs), d(DecodeTensor::TreeNodes),
           d(DecodeTensor::TreeTokens), d(DecodeTensor::TreeCounts)},
          std::span(anchors).first(lanes), std::span(policies).first(lanes),
          treeMask);
    }, draftModel);
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

#include "model/RuntimeImpl.hpp"
#include "model/Qwen4Exp.hpp"
#include "metal/abi/BlockIndexer.h"
#include "metal/abi/NgramEmbedding.h"

namespace richengine::model {

namespace {
// A verify lane's record slots: the blocks its verifyRows can complete.
constexpr uint32_t kQsaVerifyLaneRecords =
    (RICHENGINE_TARGET_VERIFY_ROWS + BLOCK_INDEXER_BLOCK - 1) /
    BLOCK_INDEXER_BLOCK;
}  // namespace

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
    // The fused head's argmax tile holds 48 affine rows (the m48 kernel's
    // 24 KB tile stays under the threadgroup cap) or 32 GGUF rows (the gguf
    // amax family is instantiated to m32); beyond that the ordinary
    // projection plus argmax pass wins.
    const bool affine48 =
        headShape.layout() == ops::WeightLayout::Affine64 &&
        headShape.outputSize % 128 == 0 && headShape.inputSize % 64 == 0;
    // m64h is opt-in: its half-precision tile can flip argmax near-ties.
    static const bool headM64 = envFlag("RICHENGINE_HEAD_M64");
    const uint32_t affineBound = headM64 ? 64 : 48;
    // RICHENGINE_HEAD_PER_LANE: a lane that is not clean-greedy keeps its
    // logits rows, which the fused head's *_logits variant writes beside the
    // argmax partials — sampled and penalized selection and masked argmax
    // read them, and a greedy GPU-DFA lane's correction row does too.
    static const bool perLaneHead = tuning().headPerLane;
    uint32_t logitsLaneMask = 0, coverableMask = 0;
    const uint32_t gpuDfa = perLaneHead ? gpuDfaLaneMask(entries) : 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const ops::SamplingPolicy policy = samplingPolicy(*entries[lane]);
      const bool clean = !policy.samples() && !policy.constrained &&
                         !policy.penalties.active();
      if (!clean)
        logitsLaneMask |= uint32_t{1} << lane;
      // A lane the fused argmax answers: clean greedy, or a greedy GPU-DFA
      // lane whose proposal check argmaxes unmasked anyway. Penalized lanes
      // stay out — their logits change after the head wrote them. A skipped
      // DFA lane counts coverable here although the policy pass excludes
      // it: the fused head then only wastes its partials.
      if (clean || (perLaneHead && !policy.samples() &&
                    !policy.penalties.active() &&
                    (gpuDfa & (uint32_t{1} << lane))))
        coverableMask |= uint32_t{1} << lane;
    }
    // The GGUF amax family has no logits writeback variant: a batch with a
    // logits-reading lane on a GGUF head keeps the all-or-nothing rule.
    const bool fusedShape =
        affine48 ? (perLaneHead ? coverableMask != 0 : !logitsLaneMask)
                 : ggufHead && !logitsLaneMask;
    buffers.fusedHead = !headFusedOff && !tree && fusedShape &&
        uint64_t{lanes} * geometry.draft.verifyRows() <=
            (affine48 ? affineBound : 32);
    buffers.headArgs.output_size = geometry.target.vocabularySize;
    buffers.headArgs.output_stride = geometry.target.paddedVocabularySize;
    buffers.headArgs.input_size = geometry.target.hiddenSize;
    buffers.headArgs.rows = geometry.draft.verifyRows();
    buffers.headArgs.stop_token_0 = geometry.target.stopTokens[0];
    buffers.headArgs.stop_token_1 = geometry.target.stopTokens[1];
    buffers.headArgs.logits_lane_mask =
        buffers.fusedHead ? logitsLaneMask : 0;
    if (buffers.fusedHead) {
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const ops::SamplingPolicy policy = samplingPolicy(*entries[lane]);
        if (policy.excludesStopTokens)
          buffers.headArgs.exclude_stop_mask |= uint32_t{1} << lane;
        buffers.headArgs.live_rows[lane] =
            lane < liveRows.size()
                ? std::clamp(liveRows[lane], uint32_t{1},
                             geometry.draft.verifyRows())
                : geometry.draft.verifyRows();
      }
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
    bindIndexTables(entries, buffers.indexTables);
    // Each lane's QSA block records (upstream prepareVerifyLane's
    // writeBlocks): one per block whose last token is a verify row, its
    // first token's position and M-RoPE coordinates.
    std::array<BlockIndexerBlock, kLaneCount * kQsaVerifyLaneRecords>
        qsaBlocks{};
    if (geometry.target.kvLayout.indexDimension) {
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint64_t position = items[lane].logicalPosition;
        uint32_t count = 0;
        for (uint64_t last = position;
             last < position + geometry.draft.verifyRows(); ++last) {
          if (last % BLOCK_INDEXER_BLOCK != BLOCK_INDEXER_BLOCK - 1)
            continue;
          const uint64_t first = last - (BLOCK_INDEXER_BLOCK - 1);
          const std::array<uint32_t, 3> rotary =
              ropePosition(*entries[lane], first);
          qsaBlocks[lane * kQsaVerifyLaneRecords + count++] =
              {static_cast<uint32_t>(first),
               {rotary[0], rotary[1], rotary[2]}};
        }
      }
      buffers.qsaBlocks =
          std::span<const BlockIndexerBlock>(qsaBlocks.data(),
                                             kLaneCount * kQsaVerifyLaneRecords);
    }
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
    // The GDN decode tape: each lane's previous step's retained rows fold
    // into this step's scan; the step's own rows go to its tape slot for the
    // next. Empty pending rows run the standalone commit instead.
    ops::GdnTapeLanes tapeLanes;
    const ops::GdnTapeLanes *tape = nullptr;
    if (gdnTape_ && !tree) {
      const uint64_t layerBytes = gdnTapeLayerBytes();
      const uint64_t slotBytes =
          uint64_t{geometry.target.stateLayout.layers} * layerBytes;
      tapeLanes.layerBytes = layerBytes;
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t sl = entries[lane]->stateLane;
        const uint64_t base = uint64_t{sl} * 2 * slotBytes;
        const uint32_t parity = gdnTapeParity_[sl];
        tapeLanes.pending[lane] = base + (parity ^ 1) * slotBytes;
        tapeLanes.step[lane] = base + parity * slotBytes;
        tapeLanes.rows[lane] = gdnPendingRows_[sl];
      }
      tape = &tapeLanes;
    }
    targetModel.addVerify(graph, std::move(buffers), kvPages.layers(),
                          std::span(chunks).first(lanes), lanes, tree,
                          liveRows, liveNodes, geometry.draft.verifyRows(),
                          tape);
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
    // Affine heads fuse up to 48 rows (the m48 argmax tile), GGUF heads up
    // to 32 (their amax kernels stop at m32); the t256 variant additionally
    // wants a 256-divisible vocabulary.
    const uint32_t headRows = lanes * geometry.draft.verifyRows();
    const bool affineHead =
        headShape.layout() == ops::WeightLayout::Affine64 &&
        headShape.outputSize % 128 == 0 && headShape.inputSize % 64 == 0;
    static const bool headM64 = envFlag("RICHENGINE_HEAD_M64");
    uint32_t fusedHead =
        headFusedOff || headRows > (affineHead ? (headM64 ? 64u : 48u) : 32u)
            ? 0
            : affineHead
                  ? (headRows <= 16 && headShape.outputSize % 256 == 0 &&
                             geometry.target.paddedVocabularySize % 256 == 0
                         ? 3
                         : 1)
                  : ggufHead ? 2 : 0;
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
    // RICHENGINE_HEAD_PER_LANE: the fused argmax answers clean-greedy lanes
    // and greedy GPU-DFA lanes (their proposal check argmaxes unmasked
    // anyway); every other lane reads the logits rows the head's *_logits
    // variant wrote. Penalized lanes stay out of fusedLaneMask — the
    // penalties rewrite their logits after the head ran. The mask matches
    // the forward pass's coverable set minus lanes skipped here: a
    // coverable-only-skip batch ran the fused head needlessly, never the
    // reverse (a fused reduce over unwritten partials).
    uint32_t logitsLaneMask = 0, fusedLaneMask = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const bool clean = !policies[lane].samples() &&
                         !policies[lane].constrained &&
                         !policies[lane].penalties.active();
      if (clean ||
          (tuning().headPerLane && !policies[lane].penalties.active() &&
           (unmasked & (uint32_t{1} << lane)))) {
        fusedLaneMask |= uint32_t{1} << lane;
      } else {
        logitsLaneMask |= uint32_t{1} << lane;
      }
    }
    // The GGUF amax family has no logits writeback variant: it keeps the
    // all-or-nothing rule under the flag too.
    if (tuning().headPerLane
            ? (fusedHead == 2 ? logitsLaneMask : !fusedLaneMask)
            : logitsLaneMask)
      fusedHead = 0;
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
                       liveRows, fusedHead, geometry.draft.verifyRows(),
                       fusedLaneMask);
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
    // The staged anchors: draft_predraft_pack fills draft_input only for
    // the chained-predraft path, so a target without it (LFM3) would leave
    // the verify input's proof fallback reading zeros. The kernels stride
    // draft_input by queryRows — the arena's kDecodeRows allocation is one
    // row wider — so stride by the same, never by the buffer's lane size.
    {
      uint32_t *staged =
          contents<uint32_t>(draftInput, "verify staged anchors");
      const uint64_t stride = geometry.draft.queryRows();
      for (uint32_t lane = 0; lane < lanes; ++lane)
        staged[lane * stride] = *entries[lane]->pendingToken;
    }
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
    // qwen4exp's PLE hash reads each lane's pleNgram - 1 tokens before its
    // first verify row (upstream Qwen4Steps::prepareVerifyLane's
    // plePriorTokens); underflow lanes get the none id.
    MetalBuffer plePriorTokens;
    if (const auto *qwen4 =
            std::get_if<Qwen4ExpWeights>(&package.target)) {
      const uint32_t priors = qwen4->layout.pleNgram - 1;
      plePriorTokens =
          decodeArena->packed(DecodeTensor::PlePriorTokens, lanes);
      auto *priorTokens =
          contents<uint32_t>(plePriorTokens, "PLE prior tokens");
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint64_t position = items[lane].logicalPosition;
        const std::span<const uint32_t> request =
            items[lane].requestTokens;
        for (uint32_t back = 1; back <= priors; ++back)
          priorTokens[lane * priors + back - 1] =
              position >= back && position - back < request.size()
                  ? request[position - back]
                  : NGRAM_EMBEDDING_NONE;
      }
    }
    targetModel.addVerifyInput(
        graph, draftInput, proposals, verifyInput, lanes,
        geometry.draft.verifyRows(), geometry.draft.queryRows(), liveRows,
        std::move(plePriorTokens));
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
      case DfaBufferKind::Cost:
        return dfa.costBuffer;
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
    const auto costs = dfaBuffers(entries, DfaBufferKind::Cost);
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
      mask.tokens_left[lane] = dfaTokensLeft(*entries[lane]);
      walk.tokens_left[lane] = dfaTokensLeft(*entries[lane]);
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
        packedVocab_->special,
        costs[0], costs[1], costs[2], costs[3]};
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
    const auto costs = dfaBuffers(entries, DfaBufferKind::Cost);
    DfaMaskParams mask{};
    mask.vocabulary = packedVocab_->vocab;
    mask.rows = 1;
    mask.lane_mask = laneMask;
    mask.indexed = 1;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      mask.tokens_left[lane] = dfaTokensLeft(*entries[lane]);
      if (entries[lane]->dfa) {
        mask.states[lane] = entries[lane]->dfa->states;
        dfaLaneEos(mask, lane, *entries[lane]->dfa);
      }
    }
    graph.add(std::string(ops::kDecodeDfaMask),
              {tables[0], tables[1], tables[2], tables[3],
               accepts[0], accepts[1], accepts[2], accepts[3],
               packedVocab_->bytes, packedVocab_->offsets,
               decodeArena->packed(DecodeTensor::DfaRowStates, lanes),
               decodeArena->packed(DecodeTensor::DfaNextStates, lanes),
               decodeArena->packed(DecodeTensor::DfaTrunc, lanes),
               markers[0], markers[1], markers[2], markers[3],
               packedVocab_->special,
               costs[0], costs[1], costs[2], costs[3]},
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
      walk.tokens_left[lane] = dfaTokensLeft(*entries[lane]);
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
      walk.tokens_left[lane] = dfaTokensLeft(*entries[lane]);
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
    const auto costs = dfaBuffers(entries, DfaBufferKind::Cost);
    DfaMaskParams mask{};
    mask.vocabulary = packedVocab_->vocab;
    mask.rows = 1;
    mask.lane_mask = laneMask;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!entries[lane] || !entries[lane]->dfa)
        continue;
      mask.tokens_left[lane] = dfaTokensLeft(*entries[lane]);
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
               packedVocab_->special,
               costs[0], costs[1], costs[2], costs[3]},
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
    // The folded path folds retained rows into the next step's scan — the
    // standalone commit only runs when the tape is off or tree-verify did.
    if (!gdnTape_)
      targetModel.addStateCommit(graph, std::move(buffers), width,
                                 geometry.draft.verifyRows());
  }

} // namespace richengine::model

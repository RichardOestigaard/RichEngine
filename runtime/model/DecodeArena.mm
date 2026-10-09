#include "model/DecodeArena.hpp"

#include "ops/DraftSelector.hpp"
#include "ops/Sampling.hpp"

namespace richengine::model {

static uint64_t gdnPackedStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<uint16_t>(uint64_t{kDecodeRows} *
                            geometry.target.packedGdnWidth);
}
static uint64_t gdnMixedStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<uint16_t>(uint64_t{kDecodeRows} *
                            geometry.target.convolutionDimension);
}
static uint64_t gdnDecayStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<float>(uint64_t{kDecodeRows} *
                         geometry.target.gdnValueHeads);
}
static uint64_t gdnBetaStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<uint16_t>(uint64_t{kDecodeRows} *
                            geometry.target.gdnValueHeads);
}
static uint64_t decodeChunkLayerBytes(const RuntimeGeometry &geometry) noexcept {
  // The layer slab covers the wider of the target's two KV geometries.
  return bytesFor<uint16_t>(uint64_t{geometry.target.chunkLayerWidth()} *
                            kv::kVerifyChunkStride);
}

std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators) {
  std::array<uint64_t, decodeTensorCount> result{};
  const auto draftWorkspace =
      geometry.draft.kind == DraftKind::Null
          ? ops::DraftAttentionWorkspace{}
          : operators.draftAttentionWorkspacePerLane(
                geometry.draft.attentionShape());
  // The sampling workspaces cover a tree lane's node count so the same
  // tensors serve chain (8-row) and tree (16-row) verify batches.
  const auto samplingWorkspace =
      ops::Sampling::workspace(RICHENGINE_TREE_VERIFY_NODES);
  const auto selectorWorkspace = ops::DraftSelector::workspace(kDraftProposalTokens);
  auto put = [&](DecodeTensor tensor, uint64_t bytes) {
    auto &size = result[static_cast<uint32_t>(tensor)];
    size = std::max(size, bytes);
  };
  const uint64_t r = kDecodeRows;
  put(DecodeTensor::Hidden0,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Hidden1,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::InputTokens, bytesFor<uint32_t>(r));
  put(DecodeTensor::Normalized,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::GdnHidden,
      bytesFor<uint16_t>(r * geometry.target.attentionWidth));
  put(DecodeTensor::GdnOutput,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Intermediate,
      bytesFor<uint16_t>(r * geometry.target.denseIntermediateSize));
  put(DecodeTensor::FullPacked,
      bytesFor<uint16_t>(r * geometry.target.maximumPackedWidth()));
  put(DecodeTensor::FullQueries,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kv::kVerifyChunkStride *
                         geometry.target.maximumHeadDimension()));
  ops::AttentionWorkspace attentionWorkspace =
      operators.verifyAttentionWorkspacePerLane(
          geometry.target.attentionQueryHeads,
          geometry.target.layerKvLayout(false));
  if (geometry.target.altAttentionMask) {
    const ops::AttentionWorkspace alt =
        operators.verifyAttentionWorkspacePerLane(
            geometry.target.attentionQueryHeads,
            geometry.target.layerKvLayout(true));
    attentionWorkspace.partialsBytes =
        std::max(attentionWorkspace.partialsBytes, alt.partialsBytes);
    attentionWorkspace.statisticsBytes =
        std::max(attentionWorkspace.statisticsBytes, alt.statisticsBytes);
  }
  put(DecodeTensor::AttentionPartials, attentionWorkspace.partialsBytes);
  put(DecodeTensor::AttentionStatistics, attentionWorkspace.statisticsBytes);
  put(DecodeTensor::FullAttention,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kv::kVerifyChunkStride *
                         geometry.target.maximumHeadDimension()));
  put(DecodeTensor::AttentionHidden,
      bytesFor<uint16_t>(r * geometry.target.maximumAttentionWidth()));
  put(DecodeTensor::AttentionOutput,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Positions, bytesFor<uint32_t>(r * 3));
  put(DecodeTensor::DraftPositions, bytesFor<uint32_t>(r));
  put(DecodeTensor::RopeCos,
      bytesFor<float>(r * geometry.target.rotaryPairs));
  put(DecodeTensor::RopeSin,
      bytesFor<float>(r * geometry.target.rotaryPairs));
  put(DecodeTensor::RopeCosAlt,
      bytesFor<float>(r * geometry.target.altRotaryPairs));
  put(DecodeTensor::RopeSinAlt,
      bytesFor<float>(r * geometry.target.altRotaryPairs));
  put(DecodeTensor::ContextProjected,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::ContextHidden,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::ContextKv,
      bytesFor<uint16_t>(r * geometry.draft.contextKvSize()));
  put(DecodeTensor::CapturedTargetHidden,
      bytesFor<uint16_t>(r * geometry.draft.targetHiddenSize));
  put(DecodeTensor::DraftQueryKeys, draftWorkspace.queryKeysBytes);
  put(DecodeTensor::DraftQueryValues, draftWorkspace.queryValuesBytes);
  // Proposal attention and accepted target-hidden injection use the same
  // eight absolute positions, so one RoPE table per lane is sufficient.
  put(DecodeTensor::DraftRopeCos,
      bytesFor<float>(r * geometry.draftRotaryPairs()));
  put(DecodeTensor::DraftRopeSin,
      bytesFor<float>(r * geometry.draftRotaryPairs()));
  put(DecodeTensor::FinalHidden,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Logits,
      bytesFor<float>(r * geometry.target.vocabularySize));
  put(DecodeTensor::ArgmaxValues, samplingWorkspace.argmaxValuesBytes);
  put(DecodeTensor::ArgmaxIndices, samplingWorkspace.argmaxIndicesBytes);
  // The fused greedy head's per-(row, column tile) partials: the affine
  // kernel's tiles are 128 wide, the GGUF one's 64 — reserve the latter's.
  const uint64_t headTiles = geometry.target.vocabularySize / 64;
  put(DecodeTensor::HeadArgmaxValues,
      bytesFor<float>(uint64_t{r} * headTiles));
  put(DecodeTensor::HeadArgmaxIndices,
      bytesFor<uint32_t>(uint64_t{r} * headTiles));
  put(DecodeTensor::TargetPartialMasses, samplingWorkspace.partialMassesBytes);
  put(DecodeTensor::TargetVocabularyRows, samplingWorkspace.vocabularyRowsBytes);
  put(DecodeTensor::TargetVocabularyRanges,
      samplingWorkspace.vocabularyRangesBytes);
  put(DecodeTensor::TargetVocabularyArrivals,
      samplingWorkspace.vocabularyArrivalsBytes);
  put(DecodeTensor::SamplingUniforms, bytesFor<float>(kSamplingUniformCount));
  put(DecodeTensor::ConstraintMasks,
      bytesFor<uint32_t>(uint64_t{ExecutionLimits::maximumStepTokens} *
                         geometry.maskWords()));
  put(DecodeTensor::OutputTokens, bytesFor<uint32_t>(r));
  put(DecodeTensor::RetainedCount, sizeof(uint32_t));
  put(DecodeTensor::AcceptedCount, sizeof(uint32_t));
  put(DecodeTensor::DraftInputTokens, bytesFor<uint32_t>(r));
  for (uint32_t index = 0; index < 2; ++index) {
    put(static_cast<DecodeTensor>(
            static_cast<uint32_t>(DecodeTensor::DraftHidden0) + index),
        bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  }
  put(DecodeTensor::DraftNormalized,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftDynamic,
      bytesFor<uint16_t>(r * geometry.draft.dynamicSize));
  put(DecodeTensor::DraftConvolved, draftWorkspace.convolutionBytes);
  put(DecodeTensor::DraftProposalQkv, draftWorkspace.qkvBytes);
  put(DecodeTensor::DraftAttention, draftWorkspace.groupedQueriesBytes);
  put(DecodeTensor::DraftProjected,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftResidual,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftIntermediate,
      bytesFor<uint16_t>(r * geometry.draft.intermediateSize));
  put(DecodeTensor::DraftFinalHidden,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::SelectorHidden,
      bytesFor<uint16_t>(r * geometry.draft.selectorRank));
  put(DecodeTensor::Candidates, selectorWorkspace.candidatesBytes);
  put(DecodeTensor::Unary, selectorWorkspace.unaryBytes);
  put(DecodeTensor::TopPartialIds, selectorWorkspace.partialIdsBytes);
  put(DecodeTensor::TopPartialValues, selectorWorkspace.partialValuesBytes);
  put(DecodeTensor::ProposalProbs, selectorWorkspace.proposalProbabilitiesBytes);
  put(DecodeTensor::ProposedTokens, bytesFor<uint32_t>(kDraftProposalTokens));
  put(DecodeTensor::TreeNodes,
      bytesFor<uint32_t>(RICHENGINE_TREE_VERIFY_NODES));
  put(DecodeTensor::TreeTokens,
      bytesFor<uint32_t>(RICHENGINE_TREE_VERIFY_NODES));
  put(DecodeTensor::TreeCounts, bytesFor<uint32_t>(1));
  put(DecodeTensor::TreeMasks,
      bytesFor<uint32_t>(RICHENGINE_TREE_VERIFY_NODES));
  put(DecodeTensor::RetainedPath,
      bytesFor<uint32_t>(RICHENGINE_TARGET_VERIFY_ROWS));
  put(DecodeTensor::TreeSelected,
      bytesFor<uint32_t>(RICHENGINE_TREE_VERIFY_NODES));
  put(DecodeTensor::CapturedPath,
      bytesFor<uint16_t>(uint64_t{RICHENGINE_TREE_VERIFY_NODES} *
                         geometry.draft.targetHiddenSize));
  // Gemma's layer scratch: the post-attention residual and the zeroed rows
  // the routed combine adds to.
  put(DecodeTensor::GemmaResidual,
      geometry.target.gemmaMoe
          ? bytesFor<uint16_t>(r * geometry.target.hiddenSize)
          : 0);
  put(DecodeTensor::ZeroResidual,
      geometry.target.gemmaMoe
          ? bytesFor<uint16_t>(r * geometry.target.hiddenSize)
          : 0);
  put(DecodeTensor::PageTable, bytesFor<RichKvPage>(kMaximumPageTableEntries));
  put(DecodeTensor::PenaltyState,
      bytesFor<uint32_t>(geometry.target.vocabularySize));
  put(DecodeTensor::VerifyPackedBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnPackedStride(geometry));
  put(DecodeTensor::VerifyMixedBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnMixedStride(geometry));
  put(DecodeTensor::VerifyDecayBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnDecayStride(geometry));
  put(DecodeTensor::VerifyBetaBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnBetaStride(geometry));
  put(DecodeTensor::ChunkKeysBase,
      uint64_t{geometry.target.kvLayout.attentionLayers} *
          decodeChunkLayerBytes(geometry));
  put(DecodeTensor::ChunkValuesBase,
      uint64_t{geometry.target.kvLayout.attentionLayers} *
          decodeChunkLayerBytes(geometry));
  if (geometry.target.ffnKind == FfnKind::SparseMoe) {
    const ops::MoeWorkspace workspace =
        operators.moeDecodeWorkspacePerLane(geometry.target.moeShape());
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      put(moeScratchTensor<DecodeTensor>(field),
          workspace.*ops::kMoeScratchFields[field].bytes);
  }
  return result;
}

uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                             const ops::ExecutionPlans &operators) {
  uint64_t bytes = 0;
  for (uint64_t value : decodeTensorBytes(geometry, operators)) {
    bytes = checkedAdd(
        bytes, alignUp(checkedMultiply(value, kLaneCount, "decode tensor")),
        "decode arena");
  }
  return bytes;
}

ops::LinearScratchSize DecodeArena::linearScratchSize(
    const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators) {
  const auto &d = geometry.draft;
  ops::LinearScratchSize result;
  // Includes the vocabulary head shared with the draft, whose own
  // projections are affine.
  for (const auto &p : geometry.target.decodeProjections) result.include(operators.linear().decodeScratchSize(p));
  for (const ops::ProjectionShape shape : {ops::ProjectionShape{d.dynamicSize, d.hiddenSize},
       {d.qkvSize, d.hiddenSize}, {d.contextKvSize(), d.hiddenSize},
       {d.hiddenSize, d.attentionSize},
       {d.intermediateSize, d.hiddenSize}, {d.hiddenSize, d.intermediateSize},
       {d.selectorRank, d.hiddenSize}, {d.hiddenSize, d.targetHiddenSize}}) {
    // A plain draft's dynamic and selector projections are absent (size 0).
    if (!shape.outputSize || !shape.inputSize) continue;
    result.include(operators.linear().decodeScratchSize(shape));
  }
  return result;
}

uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                           const ops::ExecutionPlans &operators) {
  return checkedAdd(decodeArenaBaseBytes(geometry, operators),
                    checkedAdd(DecodeArena::gateScratchBytes(geometry, operators),
                               DecodeArena::linearScratchSize(geometry, operators).bytes(),
                               "Q4 decode scratch"),
                    "planned gate scratch");
}

} // namespace richengine::model

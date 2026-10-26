#include "model/PrefillArena.hpp"

#include "Tuning.hpp"
#include "model/ModelPolicy.hpp"
#include "ops/GDN.hpp"

namespace richengine::model {
std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators) {
  std::array<uint64_t, prefillTensorCount> result{};
  auto put = [&](PrefillTensor tensor, uint64_t bytes) {
    auto &size = result[static_cast<uint32_t>(tensor)];
    size = std::max(size, bytes);
  };
  put(PrefillTensor::Hidden0,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::Hidden1,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::InputTokens, bytesFor<uint32_t>(kPrefillRows));
  put(PrefillTensor::Normalized,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::Captured,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.capturedHiddenSize()));
  put(PrefillTensor::GdnPacked,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.packedGdnWidth));
  put(PrefillTensor::GdnQueries,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.gdnKeyWidth()));
  put(PrefillTensor::GdnKeys,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.gdnKeyWidth()));
  put(PrefillTensor::GdnValues,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.attentionWidth));
  put(PrefillTensor::GdnDecay,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.target.gdnValueHeads));
  put(PrefillTensor::GdnBeta,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.gdnValueHeads));
  {
    // RICHENGINE_GDN_CHUNKED (32/64/128) sizes the WY/UT scratch at this
    // geometry's GDN shape; the serial scan binds nothing here.
    const uint32_t factor = modelPolicy().gdnChunked;
    if (factor == 32 || factor == 64 || factor == 128) {
      put(PrefillTensor::GdnChunkScratch,
          bytesFor<float>(ops::GDN::chunkScratchFloats(
              geometry.target.gdnShape(), kPrefillRows, factor)));
    }
  }
  put(PrefillTensor::Recurrent,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.attentionWidth));
  put(PrefillTensor::GdnHidden,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.attentionWidth));
  put(PrefillTensor::GdnOutput,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::GateIntermediate,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.denseIntermediateSize));
  put(PrefillTensor::Intermediate,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.denseIntermediateSize));
  put(PrefillTensor::FullPacked,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.maximumPackedWidth()));
  put(PrefillTensor::FullQueries,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kPackedAttentionRows *
                         geometry.target.maximumHeadDimension()));
  put(PrefillTensor::FullAttention,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kPackedAttentionRows *
                         geometry.target.maximumHeadDimension()));
  // A dual-geometry target needs the larger of its layer kinds' workspaces.
  ops::AttentionWorkspace attentionWorkspace =
      operators.prefillAttentionWorkspace(
          kPrefillRows, geometry.target.attentionQueryHeads,
          geometry.target.layerKvLayout(false));
  if (geometry.target.altAttentionMask) {
    const ops::AttentionWorkspace alt =
        operators.prefillAttentionWorkspace(
            kPrefillRows, geometry.target.attentionQueryHeads,
            geometry.target.layerKvLayout(true));
    attentionWorkspace.partialsBytes =
        std::max(attentionWorkspace.partialsBytes, alt.partialsBytes);
    attentionWorkspace.statisticsBytes =
        std::max(attentionWorkspace.statisticsBytes, alt.statisticsBytes);
  }
  put(PrefillTensor::AttentionPartials, attentionWorkspace.partialsBytes);
  put(PrefillTensor::AttentionStatistics, attentionWorkspace.statisticsBytes);
  put(PrefillTensor::AttentionHidden,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.maximumAttentionWidth()));
  put(PrefillTensor::AttentionOutput,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::ProjectionSums,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.projectionSumsWidth()));
  put(PrefillTensor::DownProjectionSums,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.projectionSumsWidth()));
  // Three rotary axes per row (M-RoPE); text rows repeat one value.
  put(PrefillTensor::TargetPositions,
      bytesFor<uint32_t>(uint64_t{kPrefillRows} * 3));
  put(PrefillTensor::DraftPositions, bytesFor<uint32_t>(kPrefillRows));
  put(PrefillTensor::TargetInverseFrequencies,
      bytesFor<float>(geometry.target.rotaryPairs));
  put(PrefillTensor::DraftInverseFrequencies,
      bytesFor<float>(geometry.draftRotaryPairs()));
  put(PrefillTensor::RopeCos,
      bytesFor<float>(uint64_t{kPrefillRows} * geometry.target.rotaryPairs));
  put(PrefillTensor::RopeSin,
      bytesFor<float>(uint64_t{kPrefillRows} * geometry.target.rotaryPairs));
  put(PrefillTensor::TargetInverseFrequencies2,
      bytesFor<float>(geometry.target.altRotaryPairs));
  put(PrefillTensor::RopeCosAlt,
      bytesFor<float>(uint64_t{kPrefillRows} * geometry.target.altRotaryPairs));
  put(PrefillTensor::RopeSinAlt,
      bytesFor<float>(uint64_t{kPrefillRows} * geometry.target.altRotaryPairs));
  // Gemma's routed combine residual: zeroed hidden rows.
  put(PrefillTensor::ZeroResidual,
      geometry.target.gemmaMoe
          ? bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize)
          : 0);
  put(PrefillTensor::ContextProjected,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextHidden,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextKv,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.draft.contextKvSize()));
  put(PrefillTensor::ContextQkv,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.draft.qkvSize));
  put(PrefillTensor::DraftRopeCos,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.draftRotaryPairs()));
  put(PrefillTensor::DraftRopeSin,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.draftRotaryPairs()));
  put(PrefillTensor::ChunkKeys,
      bytesFor<uint16_t>(uint64_t{geometry.target.chunkLayerWidth()} *
                         kPackedAttentionRows));
  put(PrefillTensor::ChunkValues,
      bytesFor<uint16_t>(uint64_t{geometry.target.chunkLayerWidth()} *
                         kPackedAttentionRows));
  // The split partials and counters and the rotated rows of the largest
  // prefill plan.
  for (const auto &projection : geometry.target.prefillProjections) {
    const ops::LinearScratchSize linear = operators.linear().prefillScratchSize(projection);
    put(PrefillTensor::LinearPartials, linear.partials);
    put(PrefillTensor::LinearCounters, linear.counters);
    put(PrefillTensor::LinearRotated, linear.rotated);
    put(PrefillTensor::LinearPacked, linear.input);
    put(PrefillTensor::LinearExponents, linear.sums);
  }
  // The draft's context K/V commits run decode plans (DSparkDraft::addContextPrefill).
  if (geometry.draft.contextKvSize()) {
    ops::ProjectionShape draftShape{geometry.draft.contextKvSize(),
                                    geometry.draft.hiddenSize};
    ops::LinearScratchSize draft =
        operators.linear().decodeScratchSize(draftShape);
    // The context K/V rows come off the fused QKV's AF4 block segment now:
    // reserve the GGUF decode bound as well.
    draftShape.layout = ops::WeightLayout::Block32;
    draftShape.registerFormats = draftShape.fourBits = true;
    draft.include(operators.linear().decodeScratchSize(draftShape));
    // DFlash2's context prefill runs prefill plans on the same block
    // segment (DFlashDraft::addContextPrefill).
    draft.include(operators.linear().prefillScratchSize(draftShape));
    // The context projection's prefill plan runs on the same block layout
    // and is much wider than the context K/V rows.
    ops::ProjectionShape contextBlock{geometry.draft.hiddenSize,
                                      geometry.draft.targetHiddenSize};
    contextBlock.layout = ops::WeightLayout::Block32;
    contextBlock.registerFormats = contextBlock.fourBits = true;
    draft.include(operators.linear().prefillScratchSize(contextBlock));
    put(PrefillTensor::LinearPartials, draft.partials);
    put(PrefillTensor::LinearCounters, draft.counters);
    put(PrefillTensor::LinearPacked, draft.input);
    put(PrefillTensor::LinearExponents, draft.sums);
  }
  if (modelPolicy().prefillFastInt8) {
    const uint64_t inputWidth =
        uint64_t{geometry.projectionSumsWidth()} * kQ4GroupElements;
    put(PrefillTensor::I8Codes, uint64_t{kPrefillRows} * inputWidth);
    put(PrefillTensor::I8CodesLo, uint64_t{kPrefillRows} * inputWidth);
    put(PrefillTensor::I8Params,
        uint64_t{kPrefillRows} * geometry.projectionSumsWidth() * 16);
    put(PrefillTensor::I8ParamsLo,
        uint64_t{kPrefillRows} * geometry.projectionSumsWidth() * 16);
  }
  if (geometry.target.ffnKind == FfnKind::SparseMoe) {
    const ops::MoeWorkspace workspace =
        operators.moePrefillWorkspace(geometry.target.moeShape(), kPrefillRows);
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      put(moeScratchTensor<PrefillTensor>(field),
          workspace.*ops::kMoeScratchFields[field].bytes);
  }
  return result;
}

uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                            const ops::ExecutionPlans &operators) {
  uint64_t bytes = 0;
  for (uint64_t value : prefillTensorBytes(geometry, operators)) {
    bytes = checkedAdd(bytes, alignUp(value), "prefill arena");
  }
  return bytes;
}

} // namespace richengine::model

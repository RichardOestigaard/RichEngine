#include "model/PrefillArena.hpp"

#include "Tuning.hpp"
#include "model/ModelPolicy.hpp"
#include "ops/GDN.hpp"

namespace richengine::model {
namespace {

// The prefill tensors of a command of `rows` rows (prefillTensorBytes).
std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators, uint32_t rows) {
  std::array<uint64_t, prefillTensorCount> result{};
  auto put = [&](PrefillTensor tensor, uint64_t bytes) {
    auto &size = result[static_cast<uint32_t>(tensor)];
    size = std::max(size, bytes);
  };
  put(PrefillTensor::Hidden0,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.target.hiddenSize));
  put(PrefillTensor::Hidden1,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.target.hiddenSize));
  put(PrefillTensor::InputTokens, bytesFor<uint32_t>(rows));
  put(PrefillTensor::Normalized,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.target.hiddenSize));
  put(PrefillTensor::Captured,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.capturedHiddenSize()));
  put(PrefillTensor::GdnPacked,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.packedGdnWidth));
  put(PrefillTensor::GdnQueries,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.gdnKeyWidth()));
  put(PrefillTensor::GdnKeys,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.gdnKeyWidth()));
  put(PrefillTensor::GdnValues,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.attentionWidth));
  // nemotron's Mamba-2 layers keep the per-row fp32 dt where a GDN
  // layer's decay logits land.
  const uint32_t recurrentValues = geometry.target.gdnValueHeads
                                       ? geometry.target.gdnValueHeads
                                       : geometry.target.mambaHeads;
  put(PrefillTensor::GdnDecay,
      bytesFor<float>(uint64_t{rows} * recurrentValues));
  put(PrefillTensor::GdnBeta,
      bytesFor<uint16_t>(uint64_t{rows} * recurrentValues));
  // nemotron's Mamba-2 prepare emits the post-conv x|B|C rows the scan
  // reads; sized by the conv width, zero on non-mamba targets.
  put(PrefillTensor::GdnPostConv,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.convolutionDimension));
  {
    // RICHENGINE_GDN_CHUNKED (32/64/128) sizes the WY/UT scratch at this
    // geometry's GDN shape; the serial scan binds nothing here.
    const uint32_t factor = modelPolicy().gdnChunked;
    if (factor == 32 || factor == 64 || factor == 128) {
      put(PrefillTensor::GdnChunkScratch,
          bytesFor<float>(ops::GDN::chunkScratchFloats(
              geometry.target.gdnShape(), rows, factor)));
    }
  }
  // nemotron's Mamba-2 scan writes mambaInnerSize rows where a GDN
  // layer's value rows land (attentionWidth on the hybrids).
  const uint32_t mixerRows = std::max(geometry.target.attentionWidth,
                                      geometry.target.mambaInnerSize);
  put(PrefillTensor::Recurrent,
      bytesFor<uint16_t>(uint64_t{rows} * mixerRows));
  put(PrefillTensor::GdnHidden,
      bytesFor<uint16_t>(uint64_t{rows} * mixerRows));
  put(PrefillTensor::GdnOutput,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.target.hiddenSize));
  put(PrefillTensor::GateIntermediate,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.denseIntermediateSize));
  put(PrefillTensor::Intermediate,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.denseIntermediateSize));
  put(PrefillTensor::FullPacked,
      bytesFor<uint16_t>(uint64_t{rows} *
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
          rows, geometry.target.attentionQueryHeads,
          geometry.target.layerKvLayout(false));
  if (geometry.target.altAttentionMask) {
    const ops::AttentionWorkspace alt =
        operators.prefillAttentionWorkspace(
            rows, geometry.target.attentionQueryHeads,
            geometry.target.layerKvLayout(true));
    attentionWorkspace.partialsBytes =
        std::max(attentionWorkspace.partialsBytes, alt.partialsBytes);
    attentionWorkspace.statisticsBytes =
        std::max(attentionWorkspace.statisticsBytes, alt.statisticsBytes);
  }
  put(PrefillTensor::AttentionPartials, attentionWorkspace.partialsBytes);
  put(PrefillTensor::AttentionStatistics, attentionWorkspace.statisticsBytes);
  put(PrefillTensor::AttentionHidden,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.maximumAttentionWidth()));
  put(PrefillTensor::AttentionOutput,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.target.hiddenSize));
  put(PrefillTensor::ProjectionSums,
      bytesFor<float>(uint64_t{rows} *
                      geometry.projectionSumsWidth()));
  put(PrefillTensor::DownProjectionSums,
      bytesFor<float>(uint64_t{rows} *
                      geometry.projectionSumsWidth()));
  // Three rotary axes per row (M-RoPE); text rows repeat one value.
  put(PrefillTensor::TargetPositions,
      bytesFor<uint32_t>(uint64_t{rows} * 3));
  put(PrefillTensor::DraftPositions, bytesFor<uint32_t>(rows));
  put(PrefillTensor::TargetInverseFrequencies,
      bytesFor<float>(geometry.target.rotaryPairs));
  put(PrefillTensor::DraftInverseFrequencies,
      bytesFor<float>(geometry.draftRotaryPairs()));
  put(PrefillTensor::RopeCos,
      bytesFor<float>(uint64_t{rows} * geometry.target.rotaryPairs));
  put(PrefillTensor::RopeSin,
      bytesFor<float>(uint64_t{rows} * geometry.target.rotaryPairs));
  put(PrefillTensor::TargetInverseFrequencies2,
      bytesFor<float>(geometry.target.altRotaryPairs));
  put(PrefillTensor::RopeCosAlt,
      bytesFor<float>(uint64_t{rows} * geometry.target.altRotaryPairs));
  put(PrefillTensor::RopeSinAlt,
      bytesFor<float>(uint64_t{rows} * geometry.target.altRotaryPairs));
  // Gemma's routed combine residual: zeroed hidden rows.
  put(PrefillTensor::ZeroResidual,
      geometry.target.gemmaMoe
          ? bytesFor<uint16_t>(uint64_t{rows} * geometry.target.hiddenSize)
          : 0);
  // Bolt's latent rows. The shared expert's 5376-wide intermediate reuses
  // GdnHidden (rows x mambaInnerSize 8192), idle on every MoE-only layer.
  const uint64_t latentBytes =
      geometry.target.moeLatentSize
          ? bytesFor<uint16_t>(uint64_t{rows} * geometry.target.moeLatentSize)
          : 0;
  put(PrefillTensor::MoeLatentRows, latentBytes);
  put(PrefillTensor::MoeLatentOutput, latentBytes);
  put(PrefillTensor::MoeZeroLatent, latentBytes);
  put(PrefillTensor::ContextProjected,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextHidden,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextKv,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.draft.contextKvSize()));
  put(PrefillTensor::ContextQkv,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.draft.qkvSize));
  put(PrefillTensor::DraftRopeCos,
      bytesFor<float>(uint64_t{rows} *
                      geometry.draftRotaryPairs()));
  put(PrefillTensor::DraftRopeSin,
      bytesFor<float>(uint64_t{rows} *
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
  // The MTP draft's context rows run prefill plans on its own block
  // shapes: the per-stream eh projection and the K/V rows.
  if (geometry.draft.kind == DraftKind::Mtp) {
    ops::LinearScratchSize draft;
    for (ops::ProjectionShape shape :
         {ops::ProjectionShape{geometry.draft.hiddenSize,
                               2 * geometry.draft.hiddenSize},
          {geometry.draft.contextKvSize() / 2, geometry.draft.hiddenSize}}) {
      shape.layout = ops::WeightLayout::Block32;
      shape.registerFormats = shape.fourBits = true;
      draft.include(operators.linear().prefillScratchSize(shape));
      draft.include(operators.linear().decodeScratchSize(shape));
    }
    put(PrefillTensor::LinearPartials, draft.partials);
    put(PrefillTensor::LinearCounters, draft.counters);
    put(PrefillTensor::LinearPacked, draft.input);
    put(PrefillTensor::LinearExponents, draft.sums);
  }
  // The draft's context K/V commits run decode plans (DSparkDraft::addContextPrefill).
  else if (geometry.draft.contextKvSize()) {
    ops::ProjectionShape draftShape{geometry.draft.contextKvSize(),
                                    geometry.draft.hiddenSize};
    // Nemotron drafts hold no affine segments (their 2688-wide hidden is
    // off the affine tile): only the GGUF block bound applies.
    const bool nemotronDraft = geometry.target.mambaLayers != 0;
    ops::LinearScratchSize draft =
        nemotronDraft ? ops::LinearScratchSize{}
                      : operators.linear().decodeScratchSize(draftShape);
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
    put(PrefillTensor::I8Codes, uint64_t{rows} * inputWidth);
    put(PrefillTensor::I8CodesLo, uint64_t{rows} * inputWidth);
    put(PrefillTensor::I8Params,
        uint64_t{rows} * geometry.projectionSumsWidth() * 16);
    put(PrefillTensor::I8ParamsLo,
        uint64_t{rows} * geometry.projectionSumsWidth() * 16);
  }
  if (geometry.target.ffnKind == FfnKind::SparseMoe) {
    const ops::MoeWorkspace workspace =
        operators.moePrefillWorkspace(geometry.target.moeShape(), rows);
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      put(moeScratchTensor<PrefillTensor>(field),
          workspace.*ops::kMoeScratchFields[field].bytes);
  }
  if (geometry.draft.kind == DraftKind::Mtp) {
    // The MTP layout's targetHiddenSize is the streams' width,
    // dynamicSize the mHC mix width; DFlashDraft.hpp's kMtpPrefillChunk
    // rows at a time.
    const uint32_t mtpRows = 256;
    const uint32_t h = geometry.draft.hiddenSize;
    const uint32_t streams = geometry.draft.targetHiddenSize;
    const uint32_t mix = geometry.draft.dynamicSize;
    const uint32_t kv = geometry.draft.contextKvSize() / 2;
    put(PrefillTensor::MtpStreams, uint64_t{mtpRows} * streams * sizeof(float));
    put(PrefillTensor::MtpEh, uint64_t{mtpRows} * streams * 2 * sizeof(uint16_t));
    put(PrefillTensor::MtpEhOut, uint64_t{mtpRows} * streams * sizeof(uint16_t));
    put(PrefillTensor::MtpPlane, uint64_t{mtpRows} * mix * sizeof(float));
    put(PrefillTensor::MtpMappings, uint64_t{mtpRows} * mix * sizeof(float));
    put(PrefillTensor::MtpTokens, uint64_t{mtpRows} * sizeof(uint32_t));
    put(PrefillTensor::MtpEmbed, uint64_t{mtpRows} * h * sizeof(uint16_t));
    put(PrefillTensor::MtpEmbedNormed, uint64_t{mtpRows} * h * sizeof(uint16_t));
    put(PrefillTensor::MtpKeys, uint64_t{mtpRows} * kv * sizeof(uint16_t));
    put(PrefillTensor::MtpValues, uint64_t{mtpRows} * kv * sizeof(uint16_t));
  }
  return result;
}

} // namespace

std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators) {
  std::array<uint64_t, prefillTensorCount> result =
      prefillTensorBytes(geometry, operators, kPrefillRows);
  // A command of two steps' step tensors (prefillStepTensor) hold each
  // step's rows in turn: prefillChunks of prefillChunkRows rows each, and
  // its scratch cell beside them.
  if (geometry.prefillChunks > 1) {
    const std::array<uint64_t, prefillTensorCount> each =
        prefillTensorBytes(geometry, operators, geometry.prefillChunkRows);
    for (uint32_t index = 0; index < result.size(); ++index)
      if (result[index] &&
          prefillStepTensor(static_cast<PrefillTensor>(index)))
        result[index] = alignUp(each[index]) * geometry.prefillChunks;
    result[static_cast<uint32_t>(PrefillTensor::GroupCell)] =
        groupCellBytes(geometry.target.stateLayout);
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

uint64_t groupCellBytes(const GdnStateLayout &layout) noexcept {
  if (!layout.layers) return 0;
  return layout.convolutionLayerBytes() + layout.recurrentLayerBytes() +
         layout.auxiliaryBytes;
}

GdnParityBuffers groupCellBuffers(metal::MetalBackend &backend,
                                  metal::MetalBuffer base,
                                  const GdnStateLayout &layout) {
  if (base.sizeBytes() < groupCellBytes(layout))
    throw std::invalid_argument("a group cell is larger than its buffer");
  GdnParityBuffers cell;
  cell.stateBase = std::move(base);
  const uint64_t convolution =
      layout.layers ? layout.convolutionLayerBytes() : 0;
  const uint64_t recurrent =
      layout.layers ? layout.recurrentLayerBytes() : 0;
  // One layer's state serves every layer: a part writes it at a layer and
  // the other step's part reads it there before any part's next layer.
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    if (convolution)
      cell.convolutionLayers.push_back(
          backend.view(cell.stateBase, 0, convolution));
    else
      cell.convolutionLayers.push_back(metal::MetalBuffer{});
    if (recurrent)
      cell.recurrentLayers.push_back(
          backend.view(cell.stateBase, convolution, recurrent));
    else
      cell.recurrentLayers.push_back(metal::MetalBuffer{});
  }
  return cell;
}

} // namespace richengine::model

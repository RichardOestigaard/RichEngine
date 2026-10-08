#include "model/Gemma4Moe.hpp"
#include "Checked.hpp"
#include "model/WeightLayout.hpp"
#include "model/TargetLoader.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"

#include <cstring>
#include <utility>

namespace richengine::model {
namespace {

void requireGemmaLayout(const Gemma4MoeLayout &layout) {
  const auto zero = [](auto... dimensions) { return ((dimensions == 0) || ...); };
  if (zero(layout.layers, layout.hiddenSize, layout.vocabularySize,
           layout.attentionQueryHeads, layout.attentionKvHeads,
           layout.attentionHeadDimension, layout.globalKvHeads,
           layout.globalHeadDimension, layout.rotaryPairs,
           layout.globalRotaryPairs, layout.slidingWindowTokens,
           layout.experts, layout.expertsPerToken,
           layout.expertIntermediateSize, layout.packedExpertWidth,
           layout.sharedIntermediateSize) ||
      !(layout.rotaryTheta > 0.0F) || !(layout.globalRotaryTheta > 0.0F) ||
      !(layout.logitSoftcap > 0.0F)) {
    throw WeightStoreError("Gemma 4 target layout contains a zero dimension");
  }
  if (layout.layers > 64 ||
      layout.expertsPerToken > layout.experts ||
      layout.attentionQueryHeads % layout.attentionKvHeads ||
      layout.attentionQueryHeads % layout.globalKvHeads ||
      // The rotated pairs of each layer kind cover their head dimension at
      // most once.
      2 * layout.rotaryPairs > layout.attentionHeadDimension ||
      2 * layout.globalRotaryPairs > layout.globalHeadDimension ||
      // 704 pads to 768 for the expert tiles' 128-column granularity; the
      // pad must hold at least one tile and stay a multiple of it.
      layout.packedExpertWidth < layout.expertIntermediateSize ||
      layout.packedExpertWidth % 128 ||
      layout.expertIntermediateSize % 64 ||
      // The shared expert's padding likewise: its dense tiles are 256 rows.
      layout.packedSharedWidth < layout.sharedIntermediateSize ||
      layout.packedSharedWidth % 256 ||
      layout.sharedIntermediateSize % 64 ||
      !layout.kvLayout().valid() ||
      std::ranges::any_of(layout.hiddenCaptureLayers,
                          [&](uint32_t layer) { return layer >= layout.layers; })) {
    throw WeightStoreError("Gemma 4 target layout is inconsistent");
  }
  const uint32_t hidden = layout.hiddenSize;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    validateQ4Layout(layout.packedWidthAt(layer), hidden);
    validateQ4Layout(hidden, layout.attentionWidthAt(layer));
  }
  validateQ4Layout(layout.packedExpertWidth, hidden);
  validateQ4Layout(hidden, layout.packedExpertWidth);
  validateQ4Layout(layout.packedSharedWidth, hidden);
  validateQ4Layout(hidden, layout.packedSharedWidth);
  validateQ4Layout(layout.vocabularySize, hidden);
}

// One layer file's sections, in the order the packed writer stores them.
Gemma4MoeLayerWeights readLayer(WeightFile &file, const Gemma4MoeLayout &layout,
                                uint32_t index) {
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t headDim = layout.headDimensionAt(index);
  const uint32_t packedWidth = layout.packedWidthAt(index);
  const uint32_t attentionWidth = layout.attentionWidthAt(index);
  const uint32_t width = layout.packedExpertWidth;

  Gemma4MoeLayerWeights layer;
  layer.inputNorm = readNorm(file, hidden, false, "input-norm");
  AttentionMixerWeights mixer;
  mixer.inputProjection = readAffineProjection(file, packedWidth, hidden,
                                               "attention-input");
  mixer.queryNorm = readNorm(file, headDim, false, "query-norm");
  mixer.keyNorm = readNorm(file, headDim, false, "key-norm");
  mixer.outputProjection =
      readAffineProjection(file, hidden, attentionWidth, "attention-output");
  layer.mixer = std::move(mixer);
  layer.postAttentionNorm = readNorm(file, hidden, false, "post-attention-norm");
  layer.preFfnNorm = readNorm(file, hidden, false, "pre-ffn-norm");
  layer.preFfnNormRouted = readNorm(file, hidden, false, "pre-ffn-norm-routed");
  layer.routerScale = file.section(
      checkedMultiply<WeightStoreError>(hidden, kBFloat16Bytes,
                                        "router scale bytes"),
      "router-scale");
  layer.routerWeights = file.section(
      checkedMultiply<WeightStoreError>(
          checkedMultiply<WeightStoreError>(layout.experts, hidden,
                                            "router elements"),
          kBFloat16Bytes, "router bytes"),
      "router-weights");
  layer.perExpertScale = file.section(
      checkedMultiply<WeightStoreError>(layout.experts, uint64_t{4},
                                        "per-expert scale bytes"),
      "per-expert-scale");
  layer.expertGate =
      readAffineExpertProjection(file, layout.experts, width, hidden,
                                 "experts-gate");
  layer.expertUp =
      readAffineExpertProjection(file, layout.experts, width, hidden, "experts-up");
  layer.expertDown =
      readAffineExpertProjection(file, layout.experts, hidden, width, "experts-down");
  // The shared expert's slabs are padded to packedSharedWidth rows/inputs;
  // the padding holds zeros, so gelu(gate)*up and the down sum see 2112.
  layer.sharedGate = readAffineProjection(file, layout.packedSharedWidth,
                                          hidden, "shared-expert-gate");
  layer.sharedUp = readAffineProjection(file, layout.packedSharedWidth,
                                        hidden, "shared-expert-up");
  layer.sharedDown = readAffineProjection(file, hidden,
                                          layout.packedSharedWidth,
                                          "shared-expert-down");
  layer.postFfnNormShared =
      readNorm(file, hidden, false, "post-ffn-norm-shared");
  layer.postFfnNormRouted =
      readNorm(file, hidden, false, "post-ffn-norm-routed");
  layer.postFfnNorm = readNorm(file, hidden, false, "post-ffn-norm");
  // layer_scalar: one fp32 read on the host; it rides the file so every
  // image's bytes stay the source of truth.
  const metal::MetalBuffer scalar = file.section(4, "layer-scalar");
  std::memcpy(&layer.layerScalar, scalar.contents(), 4);
  return layer;
}

} // namespace

Gemma4MoeWeights
loadGemma4MoeWeights(metal::MetalBackend &backend, Gemma4MoeLayout layout,
                     const TargetFiles<Gemma4MoeLayout> &files) {
  requireGemmaLayout(layout);
  const auto *packed =
      std::get_if<PackedTargetFiles<Gemma4MoeLayout>>(&files);
  if (!packed) {
    // The affine and GGUF planners have no Gemma 4 tensor map yet; the
    // packed format is the only path this target loads through.
    throw WeightStoreError("Gemma 4 targets load from packed files only");
  }
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Gemma4MoeWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = packed->layer(layerIndex);
    result.layers.push_back(readLayer(file, layout, layerIndex));
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = packed->head();
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    result.logitsProjection = readAffineProjection(
        file, layout.vocabularySize, layout.hiddenSize, "logits");
    // bf16 logits would round near-ties together: their spacing is 0.125 at
    // logits of 16 to 32.
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = packed->embedding();
    result.tokenEmbedding = readAffineEmbedding(
        file, layout.vocabularySize, layout.hiddenSize, "embedding",
        packed->tiledEmbedding);
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace richengine::model

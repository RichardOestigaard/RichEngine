#include "model/DiffusionGemma.hpp"
#include "Checked.hpp"
#include "model/TargetWeights.hpp"
#include "model/TargetLoader.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <cstring>
#include <utility>

namespace richengine::model {
namespace {

void requireDiffusionLayout(const TargetLayout &layout) {
  // The trunk's own consistency rules are Gemma 4's; here only the
  // diffusion fields and the canvas's fit into the paged KV geometry need
  // checking.
  if (!layout.kvLayout().valid() || !layout.diffusion.valid() ||
      layout.diffusion.canvasLength > layout.maximumContextTokens ||
      layout.diffusion.padToken >= layout.vocabularySize) {
    throw WeightStoreError("DiffusionGemma layout is inconsistent");
  }
  validateQ4Layout(layout.packedSharedWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.packedSharedWidth);
}

// self_conditioning.bin (GEMM0002, layer 0, type 3): pre_norm bf16, then the
// GeGLU triple at the packed shared width — gate, up [packedShared, hidden]
// and down [hidden, packedShared].
DiffusionSelfConditioningWeights
readSelfConditioning(WeightFile &file, const TargetLayout &layout) {
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t width = layout.packedSharedWidth;
  DiffusionSelfConditioningWeights result;
  result.preNorm = readNorm(file, hidden, false, "self-conditioning-pre-norm");
  result.gate = readAffineProjection(file, width, hidden,
                                     "self-conditioning-gate");
  result.up = readAffineProjection(file, width, hidden,
                                   "self-conditioning-up");
  result.down = readAffineProjection(file, hidden, width,
                                     "self-conditioning-down");
  return result;
}

} // namespace

DiffusionGemmaWeights
loadDiffusionGemmaWeights(metal::MetalBackend &backend,
                          const TargetLayout &layout,
                          const TargetFiles &files) {
  requireDiffusionLayout(layout);
  const auto *packed = std::get_if<PackedTargetFiles>(&files);
  if (!packed) {
    // As the Gemma 4 trunk: the packed format is the only path this target
    // loads through.
    throw WeightStoreError(
        "DiffusionGemma targets load from packed files only");
  }
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  DiffusionGemmaWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    WeightFile file = packed->layer(layerIndex);
    // The packed layer files are the Gemma 4 trunk's: the same section
    // order readLayer() consumes, with the DECODER's layer_scalar in the
    // trailing fp32.
    const uint32_t hidden = layout.hiddenSize;
    const uint32_t headDim = layout.headDimensionAt(layerIndex);
    const uint32_t packedWidth = layout.packedWidthAt(layerIndex);
    const uint32_t attentionWidth = layout.attentionWidthAt(layerIndex);
    const uint32_t width = layout.packedExpertWidth;

    TargetLayerWeights layer;
    layer.inputNorm = readNorm(file, hidden, false, "input-norm");
    AttentionMixerWeights mixer;
    mixer.inputProjection = readAffineProjection(file, packedWidth, hidden,
                                                 "attention-input");
    mixer.queryNorm = readNorm(file, headDim, false, "query-norm");
    mixer.keyNorm = readNorm(file, headDim, false, "key-norm");
    mixer.outputProjection = readAffineProjection(file, hidden,
                                                  attentionWidth,
                                                  "attention-output");
    layer.mixer = std::move(mixer);
    layer.postAttentionNorm =
        readNorm(file, hidden, false, "post-attention-norm");
    auto &ffn = layer.ffn.emplace<Gemma4FfnWeights>();
    ffn.preFfnNorm = readNorm(file, hidden, false, "pre-ffn-norm");
    ffn.preFfnNormRouted =
        readNorm(file, hidden, false, "pre-ffn-norm-routed");
    ffn.routerScale = file.section(
        checkedMultiply<WeightStoreError>(hidden, kBFloat16Bytes,
                                          "router scale bytes"),
        "router-scale");
    ffn.routerWeights = file.section(
        checkedMultiply<WeightStoreError>(
            checkedMultiply<WeightStoreError>(layout.experts, hidden,
                                              "router elements"),
            kBFloat16Bytes, "router bytes"),
        "router-weights");
    ffn.perExpertScale = file.section(
        checkedMultiply<WeightStoreError>(layout.experts, uint64_t{4},
                                          "per-expert scale bytes"),
        "per-expert-scale");
    ffn.expertGate = readAffineExpertProjection(file, layout.experts, width,
                                                hidden, "experts-gate");
    ffn.expertUp = readAffineExpertProjection(file, layout.experts, width,
                                              hidden, "experts-up");
    ffn.expertDown = readAffineExpertProjection(file, layout.experts, hidden,
                                                width, "experts-down");
    ffn.sharedGate = readAffineProjection(file, layout.packedSharedWidth,
                                          hidden, "shared-expert-gate");
    ffn.sharedUp = readAffineProjection(file, layout.packedSharedWidth,
                                        hidden, "shared-expert-up");
    ffn.sharedDown = readAffineProjection(file, hidden,
                                          layout.packedSharedWidth,
                                          "shared-expert-down");
    ffn.postFfnNormShared =
        readNorm(file, hidden, false, "post-ffn-norm-shared");
    ffn.postFfnNormRouted =
        readNorm(file, hidden, false, "post-ffn-norm-routed");
    ffn.postFfnNorm = readNorm(file, hidden, false, "post-ffn-norm");
    const metal::MetalBuffer scalar = file.section(4, "layer-scalar");
    std::memcpy(&ffn.layerScalar, scalar.contents(), 4);
    result.layers.push_back(std::move(layer));
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = packed->head();
    result.finalNorm = readNorm(file, layout.hiddenSize, false, "final-norm");
    result.logitsProjection = readAffineProjection(
        file, layout.vocabularySize, layout.hiddenSize, "logits");
    // The canvas head writes 256 bf16 rows: fp32 destinations exist only
    // for decode-phase projections, and the canvas kernels transform on
    // load anyway.
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
  {
    WeightFile file = packed->images.load(
        packedImage(packed->directory / "self_conditioning.bin",
                    "target/self_conditioning.bin",
                    layout.headMagic, 0, 3));
    result.selfConditioning = readSelfConditioning(file, layout);
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = packed->images.load(
        packedImage(packed->directory / "encoder_scalars.bin",
                    "target/encoder_scalars.bin",
                    layout.headMagic, 0, 4));
    const metal::MetalBuffer scalars =
        file.section(uint64_t{layout.layers} * sizeof(float),
                     "encoder-layer-scalars");
    result.encoderLayerScalars.resize(layout.layers);
    std::memcpy(result.encoderLayerScalars.data(), scalars.contents(),
                layout.layers * sizeof(float));
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace richengine::model

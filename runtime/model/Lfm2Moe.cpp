#include "model/Lfm2Moe.hpp"
#include "model/TargetLoader.hpp"
#include "Checked.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace richengine::model {
namespace {

constexpr uint64_t kBfloat16 = kBFloat16Bytes;

// LFM2-MoE's attention mixer, identical to LFM2's: a fused QKV projection
// (query rows alone), the per-head q/k RMS norms of head dimension 64, and
// the output projection.
template <class Format>
MixerWeights readLfmAttention(WeightFile &file, const Format &format,
                                const MixerGeometry &geometry) {
  AttentionMixerWeights attention;
  attention.inputProjection =
      format.fused(file, geometry.packedFullWidth, geometry.hiddenSize,
                   "attention-input", {"attn-q", "attn-k", "attn-v"});
  attention.queryNorm = format.norm(file, geometry.attentionHeadDimension, "query-norm");
  attention.keyNorm = format.norm(file, geometry.attentionHeadDimension, "key-norm");
  attention.outputProjection = format.projection(
      file, geometry.hiddenSize, geometry.attentionWidth, "attention-output");
  return attention;
}

// The conv mixer, identical to LFM2's: the [B|C|x] in_proj, the causal
// depthwise weights and the out_proj. Every source stores the kernel
// channel-major ([dim, taps]): the GGUF's squeezed [dim, 1, taps] HF
// tensor, packed and MLX images alike.
template <class Format>
MixerWeights readLfmConv(WeightFile &file, const Format &format,
                             const MixerGeometry &geometry) {
  LfmConvWeights conv;
  conv.inputProjection = format.projection(file, geometry.packedGdnWidth,
                                           geometry.hiddenSize, "conv-input");
  conv.convolutionWeights = file.section(
      checkedMultiply<WeightStoreError>(
          checkedMultiply<WeightStoreError>(geometry.convolutionDimension,
                                            geometry.convolutionTaps,
                                            "convolution elements"),
          kBfloat16, "convolution bytes"),
      "conv-weights");
  conv.convolutionTapsMajor = false;
  conv.outputProjection = format.projection(
      file, geometry.hiddenSize, geometry.convolutionDimension, "conv-output");
  return conv;
}

void requireLfm2MoeLayout(const Lfm2MoeLayout &layout) {
  if (layout.layers != 24 || layout.hiddenSize != 2048 ||
      layout.vocabularySize != 128000 || layout.packedGdnWidth != 6144 ||
      layout.convolutionDimension != 2048 || layout.gdnKeyHeads ||
      layout.gdnValueHeads || layout.gdnHeadDimension ||
      layout.attentionWidth != 2048 || layout.attentionQueryHeads != 32 ||
      layout.attentionKvHeads != 8 || layout.attentionHeadDimension != 64 ||
      layout.rotaryPairs != 32 || !(layout.rotaryTheta > 0.0F) ||
      !layout.maximumContextTokens || !layout.intermediateSize ||
      layout.experts != 32 || !layout.expertsPerToken ||
      layout.expertsPerToken > 4 ||
      layout.expertIntermediateSize != 1792 ||
      layout.attentionLayerCount() != 6 ||
      Lfm2MoeLayout::denseLayers >= layout.layers ||
      layout.packedFullWidth !=
          layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.packedGdnWidth != 3 * layout.convolutionDimension ||
      std::ranges::any_of(layout.hiddenCaptureLayers,
                          [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("LFM2-MoE target layout is inconsistent");
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.hiddenSize, layout.convolutionDimension);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize);
  validateQ4Layout(kQ4StorageN, layout.hiddenSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

// The norm_eps 1e-5 of every layer, head and final norm.
ops::NormWeights e5(ops::NormWeights weights) {
  weights.rmsEpsilon = 1e-5F;
  return weights;
}

// The dense FFN of a leading layer.
template <class Format>
Lfm2DenseFfn readDenseFfn(WeightFile &file, const Format &format,
                          const Lfm2MoeLayout &layout) {
  return {format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate"),
          format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up"),
          format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down")};
}

// The sigmoid-gated MoE block of a GGUF image: the F32 router, the F32
// per-expert selection bias (exp_probs_b) and the quantized expert slabs.
// There is no shared expert.
void readMoeFfn(WeightFile &file, Lfm2MoeLayerWeights &layer,
                const Lfm2MoeLayout &, const BlockTargetFormat &) {
  ops::BlockMoeWeights ffn;
  ffn.router = readQuantizedSegment(file, "router");
  ffn.expertBias = readQuantizedSegment(file, "expert-bias");
  ffn.gate.routed = readQuantizedSegment(file, "experts-gate");
  ffn.up.routed = readQuantizedSegment(file, "experts-up");
  ffn.down.routed = readQuantizedSegment(file, "experts-down");
  layer.ffn = std::move(ffn);
}

// Packed and MLX images keep a Q8 router padded to whole 256-row tiles (the
// score kernels' StorageN), an F32 bias of `experts` values and Q4 expert
// slabs.
void readMoeFfn(WeightFile &file, Lfm2MoeLayerWeights &layer,
                const Lfm2MoeLayout &layout, const AffineTargetFormat &) {
  ops::AffineMoeWeights ffn;
  ffn.router = readAffineQ8Projection(file, kQ4StorageN, layout.hiddenSize, "router");
  ffn.expertBias = file.section(uint64_t{layout.experts} * sizeof(float), "expert-bias");
  ffn.expertGate = readAffineExpertProjection(file, layout.experts,
                                              layout.expertIntermediateSize,
                                              layout.hiddenSize, "experts-gate");
  ffn.expertUp = readAffineExpertProjection(file, layout.experts,
                                            layout.expertIntermediateSize,
                                            layout.hiddenSize, "experts-up");
  ffn.expertDown = readAffineExpertProjection(file, layout.experts,
                                              layout.hiddenSize,
                                              layout.expertIntermediateSize, "experts-down");
  layer.ffn = std::move(ffn);
}

} // namespace

// The LFM2-MoE mixers' reader (found by readTargetMixer's ADL), identical to
// LFM2's.
template <class Format>
MixerWeights readTargetMixer(const Lfm2MoeLayout &, WeightFile &file,
                                 const Format &format,
                                 const MixerGeometry &geometry,
                                 bool fullAttention) {
  if (fullAttention) return readLfmAttention(file, format, geometry);
  return readLfmConv(file, format, geometry);
}

Lfm2MoeWeights loadLfm2MoeWeights(metal::MetalBackend &backend,
                                  Lfm2MoeLayout layout,
                                  const TargetFiles<Lfm2MoeLayout> &files) {
  requireLfm2MoeLayout(layout);
  // Layers are read in order; the leading denseLayers carry a dense FFN.
  uint32_t layerIndex = 0;
  const auto readFfn = [&](WeightFile &file, Lfm2MoeLayerWeights &layer, const auto &format) {
    if (layout.isMoeLayer(layerIndex++))
      readMoeFfn(file, layer, layout, format);
    else
      layer.ffn = readDenseFfn(file, format, layout);
  };
  const auto load = [&](auto &&source, const auto &format) {
    layerIndex = 0;
    Lfm2MoeWeights weights = readTargetModelWeights<Lfm2MoeWeights>(
        backend, layout, std::forward<decltype(source)>(source), format, readFfn);
    if (layerIndex != layout.layers)
      throw WeightStoreError("LFM2-MoE layer count mismatch");
    // The norm_eps 1e-5 of every layer, head and final norm.
    for (auto &layer : weights.layers) {
      layer.inputNorm = e5(layer.inputNorm);
      layer.postAttentionNorm = e5(layer.postAttentionNorm);
      if (auto *attention = std::get_if<AttentionMixerWeights>(&layer.mixer)) {
        attention->queryNorm = e5(attention->queryNorm);
        attention->keyNorm = e5(attention->keyNorm);
      }
    }
    weights.finalNorm = e5(weights.finalNorm);
    return weights;
  };
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    return load(gguf->get(), BlockTargetFormat{});
  if (const auto *mlx = std::get_if<std::reference_wrapper<AffineTargetLoader>>(&files))
    return load(mlx->get(), AffineTargetFormat{});
  const auto &packed = std::get<PackedTargetFiles<Lfm2MoeLayout>>(files);
  return load(packed, AffineTargetFormat{packed.tiledEmbedding});
}

} // namespace richengine::model

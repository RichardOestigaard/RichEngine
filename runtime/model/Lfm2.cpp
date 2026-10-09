#include "model/Lfm2.hpp"
#include "model/TargetLoader.hpp"
#include "Checked.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace richengine::model {
namespace {

constexpr uint64_t kBfloat16 = kBFloat16Bytes;

// LFM2's attention mixer: a fused QKV projection (query rows alone), the
// per-head q/k RMS norms of head dimension 64, and the output projection.
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

// The conv mixer: the [B|C|x] in_proj, the causal depthwise weights and the
// out_proj. Every source stores the kernel channel-major ([dim, taps]): the
// GGUF's squeezed [dim, 1, taps] HF tensor, packed and MLX images alike.
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

void requireLfm2Layout(const Lfm2Layout &layout) {
  if (layout.layers != 30 || layout.hiddenSize != 2048 ||
      layout.vocabularySize != 128000 || layout.packedGdnWidth != 6144 ||
      layout.convolutionDimension != 2048 || layout.gdnKeyHeads ||
      layout.gdnValueHeads || layout.gdnHeadDimension ||
      layout.attentionWidth != 2048 || layout.attentionQueryHeads != 32 ||
      layout.attentionKvHeads != 8 || layout.attentionHeadDimension != 64 ||
      layout.rotaryPairs != 32 || !(layout.rotaryTheta > 0.0F) ||
      !layout.maximumContextTokens || !layout.intermediateSize ||
      layout.attentionLayerCount() != 8 ||
      layout.packedFullWidth !=
          layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.packedGdnWidth != 3 * layout.convolutionDimension ||
      std::ranges::any_of(layout.hiddenCaptureLayers,
                          [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("LFM2 target layout is inconsistent");
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.hiddenSize, layout.convolutionDimension);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

// The LFM2 norms' epsilon (norm_eps 1e-5): every norm the loader reads gets
// it so normKernel selects the _e5 variants.
ops::NormWeights e5(ops::NormWeights weights) {
  weights.rmsEpsilon = 1e-5F;
  return weights;
}

} // namespace

// The LFM2 mixers' reader (found by readTargetMixer's ADL).
template <class Format>
MixerWeights readTargetMixer(const Lfm2Layout &, WeightFile &file,
                                 const Format &format,
                                 const MixerGeometry &geometry,
                                 bool fullAttention) {
  if (fullAttention) return readLfmAttention(file, format, geometry);
  return readLfmConv(file, format, geometry);
}

Lfm2Weights loadLfm2Weights(metal::MetalBackend &backend, Lfm2Layout layout,
                            const TargetFiles<Lfm2Layout> &files) {
  requireLfm2Layout(layout);
  const auto readFfn = [&](WeightFile &file, DenseLayerWeights &layer, const auto &format) {
    layer.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  const auto load = [&](auto &&source, const auto &format) {
    Lfm2Weights weights = readTargetModelWeights<Lfm2Weights>(
        backend, layout, std::forward<decltype(source)>(source), format, readFfn);
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
  const auto &packed = std::get<PackedTargetFiles<Lfm2Layout>>(files);
  return load(packed, AffineTargetFormat{packed.tiledEmbedding});
}

} // namespace richengine::model

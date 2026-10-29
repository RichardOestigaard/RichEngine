#include "model/Lfm3.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "Checked.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace richengine::model {
namespace {

constexpr uint64_t kBfloat16 = kBFloat16Bytes;

// LFM3's attention mixer: a fused QKV projection (query rows alone), the
// per-head q/k RMS norms of head dimension 128, the output projection and
// the head-wise gate projection (hidden -> one scalar per query head).
template <class Format>
MixerWeights readLfm3Attention(WeightFile &file, const Format &format,
                               const MixerGeometry &geometry) {
  AttentionMixerWeights attention;
  attention.inputProjection =
      format.fused(file, geometry.packedFullWidth, geometry.hiddenSize,
                   "attention-input", {"attn-q", "attn-k", "attn-v"});
  attention.queryNorm = format.norm(file, geometry.attentionHeadDimension, "query-norm");
  attention.keyNorm = format.norm(file, geometry.attentionHeadDimension, "key-norm");
  attention.outputProjection = format.projection(
      file, geometry.hiddenSize, geometry.attentionWidth, "attention-output");
  // gate_proj is far too narrow for Q4's 256-row storage tiles: either
  // source's image pads it to one 256-row tile, which the projection reads
  // as a block segment like every other.
  attention.gateProjection =
      format.projection(file, geometry.attentionGateRows, geometry.hiddenSize,
                        "attention-gate");
  return attention;
}

// The conv mixer: the [B|C|x] in_proj, the causal depthwise weights and the
// out_proj. Every source stores the kernel channel-major ([dim, taps]).
template <class Format>
MixerWeights readLfm3Conv(WeightFile &file, const Format &format,
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

MixerWeights readLfm3Mixer(WeightFile &file, const TargetLayout &,
                           const auto &format, const MixerGeometry &geometry,
                           bool fullAttention) {
  if (fullAttention) return readLfm3Attention(file, format, geometry);
  return readLfm3Conv(file, format, geometry);
}

void requireLfm3Layout(const TargetLayout &layout) {
  if (layout.encoder)
    throw WeightStoreError(
        "lfm3 encoders are not supported by this build");
  const uint64_t attention = layout.attentionMask;
  const uint64_t globals = layout.globalAttentionMask;
  if (layout.layers != 24 || layout.hiddenSize != 2048 ||
      layout.vocabularySize != 128000 || layout.packedGdnWidth != 6144 ||
      layout.convolutionDimension != 2048 || layout.convolutionTaps != 7 ||
      layout.gdnKeyHeads || layout.gdnValueHeads || layout.gdnHeadDimension ||
      layout.attentionWidth != 2048 || layout.attentionQueryHeads != 16 ||
      layout.attentionKvHeads != 4 || layout.attentionHeadDimension != 128 ||
      layout.rotaryPairs != 64 || !(layout.rotaryTheta > 0.0F) ||
      !layout.maximumContextTokens || !layout.intermediateSize ||
      !layout.sandwichNorm || !layout.attentionHeadGate ||
      !layout.altCarriesValues || !layout.altRotaryRealPairs ||
      layout.altRotaryRealPairs > layout.altRotaryPairs ||
      !layout.slidingWindowTokens ||
      !globals || (globals & ~attention) != 0 ||
      layout.packedFullWidth !=
          layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.packedGdnWidth != 3 * layout.convolutionDimension ||
      std::ranges::any_of(layout.hiddenCaptureLayers,
                          [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("LFM3 target layout is inconsistent");
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  // gate_proj (16 x hidden) rides as a float part, never a Q4 projection.
  validateQ4Layout(layout.hiddenSize, layout.convolutionDimension);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

// The LFM3 norms' epsilon (norm_eps 1e-5): every norm the loader reads gets
// it so normKernel selects the _e5 variants.
ops::NormWeights e5(ops::NormWeights weights) {
  weights.rmsEpsilon = 1e-5F;
  return weights;
}

} // namespace

TargetModelWeights loadLfm3Weights(metal::MetalBackend &backend,
                                   const TargetLayout &layout,
                                   const TargetFiles &files) {
  requireLfm3Layout(layout);
  const auto readMixer = [](WeightFile &file, const TargetLayout &l,
                            const auto &format, const MixerGeometry &g,
                            bool full) {
    return readLfm3Mixer(file, l, format, g, full);
  };
  const auto readFfn = [&](WeightFile &file, TargetLayerWeights &layer, const auto &format) {
    auto &ffn = layer.ffn.emplace<DenseFfnWeights>();
    ffn.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    ffn.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    ffn.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
    // The sandwich norms trail the FFN sections in the layer image.
    layer.postMixerNorm = format.norm(file, layout.hiddenSize, "post-mixer-norm");
    layer.postFfnNorm = format.norm(file, layout.hiddenSize, "post-ffn-norm");
  };
  TargetModelWeights weights =
      loadTargetWeights(backend, layout, files, readMixer, readFfn);
  // The norm_eps 1e-5 of every layer, head and final norm.
  for (auto &layer : weights.layers) {
    layer.inputNorm = e5(layer.inputNorm);
    layer.postAttentionNorm = e5(layer.postAttentionNorm);
    layer.postMixerNorm = e5(layer.postMixerNorm);
    layer.postFfnNorm = e5(layer.postFfnNorm);
    if (auto *attention = std::get_if<AttentionMixerWeights>(&layer.mixer)) {
      attention->queryNorm = e5(attention->queryNorm);
      attention->keyNorm = e5(attention->keyNorm);
    }
  }
  weights.finalNorm = e5(weights.finalNorm);
  return weights;
}

} // namespace richengine::model

#include "model/Granite.hpp"
#include "model/TargetLoader.hpp"
#include "Checked.hpp"

#include <algorithm>
#include <cmath>

namespace richengine::model {
namespace {

// The granite attention mixer is the dense one: fused QKV (query rows
// alone), no per-head norms, output projection.
template <class Format>
MixerWeights readGraniteAttention(WeightFile &file, const Format &format,
                                      const QwenMixerGeometry &geometry) {
  AttentionMixerWeights attention;
  attention.inputProjection =
      format.fused(file, geometry.packedFullWidth, geometry.hiddenSize,
                   "attention-input", {"attn-q", "attn-k", "attn-v"});
  attention.outputProjection = format.projection(
      file, geometry.hiddenSize, geometry.attentionWidth, "attention-output");
  return attention;
}

void requireGraniteLayout(const GraniteLayout &layout) {
  const bool eightB = layout.hiddenSize == 4096;
  if (layout.layers != 40 || layout.vocabularySize != 100352 ||
      layout.packedGdnWidth || layout.convolutionDimension ||
      layout.gdnKeyHeads || layout.gdnValueHeads || layout.gdnHeadDimension ||
      layout.attentionWidth != layout.hiddenSize ||
      layout.attentionKvHeads != 8 ||
      layout.attentionHeadDimension != (eightB ? 128u : 64u) ||
      layout.attentionQueryHeads !=
          layout.attentionWidth / layout.attentionHeadDimension ||
      layout.rotaryPairs != layout.attentionHeadDimension / 2 ||
      !(layout.rotaryTheta > 0.0F) ||
      layout.attentionScale <= 0.0F ||
      std::abs(layout.attentionScale * layout.attentionHeadDimension - 1.0F) >
          1e-6F ||
      layout.maximumContextTokens != 131072 || !layout.intermediateSize ||
      layout.packedFullWidth !=
          layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      std::ranges::any_of(layout.hiddenCaptureLayers,
                          [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("granite target layout is inconsistent");
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

} // namespace

// The granite mixer's reader (found by readTargetMixer's ADL).
template <class Format>
MixerWeights readTargetMixer(const GraniteLayout &, WeightFile &file,
                                 const Format &format,
                                 const QwenMixerGeometry &geometry, bool) {
  return readGraniteAttention(file, format, geometry);
}

GraniteWeights loadGraniteWeights(metal::MetalBackend &backend,
                                  GraniteLayout layout,
                                  const TargetFiles<GraniteLayout> &files) {
  requireGraniteLayout(layout);
  const auto readFfn = [&](WeightFile &file, DenseLayerWeights &layer, const auto &format) {
    layer.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    return readTargetModelWeights<GraniteWeights>(backend, layout, gguf->get(),
                                                 BlockTargetFormat{}, readFfn);
  throw WeightStoreError("granite targets load from GGUF only");
}

} // namespace richengine::model

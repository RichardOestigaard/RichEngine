#include "model/Dense.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "Checked.hpp"

#include <algorithm>

namespace richengine::model {
namespace {

// The dense attention mixer: a fused QKV projection (query rows alone), no
// per-head norms, and the output projection.
template <class Format>
MixerWeights readDenseAttention(WeightFile &file, const Format &format,
                                    const MixerGeometry &geometry) {
  AttentionMixerWeights attention;
  attention.inputProjection =
      format.fused(file, geometry.packedFullWidth, geometry.hiddenSize,
                   "attention-input", {"attn-q", "attn-k", "attn-v"});
  attention.outputProjection = format.projection(
      file, geometry.hiddenSize, geometry.attentionWidth, "attention-output");
  return attention;
}

// The dense mixer's reader, passed to loadTargetWeights as the mixer
// reader; every layer is full attention.
MixerWeights readTargetMixer(WeightFile &file, const TargetLayout &,
                             const auto &format,
                             const MixerGeometry &geometry, bool) {
  return readDenseAttention(file, format, geometry);
}

// Every dimension set, the query-only packed width consistent and every
// projection on the Q4 storage tiles; all layers are full attention and the
// recurrent state is empty.
void requireDenseLayout(const TargetLayout &layout) {
  if (layout.layers != 42 || layout.hiddenSize != 2048 ||
      layout.vocabularySize != 130560 ||
      layout.packedGdnWidth || layout.convolutionDimension ||
      layout.gdnKeyHeads || layout.gdnValueHeads || layout.gdnHeadDimension ||
      layout.attentionWidth != 2048 || layout.attentionQueryHeads != 16 ||
      layout.attentionKvHeads != 2 || layout.attentionHeadDimension != 128 ||
      layout.rotaryPairs != 64 || !(layout.rotaryTheta > 0.0F) ||
      !layout.maximumContextTokens || !layout.intermediateSize ||
      layout.packedFullWidth !=
          layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      std::ranges::any_of(layout.hiddenCaptureLayers,
                          [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("dense target layout is inconsistent");
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

// The arch registration's entry; the factory's Dense path calls
// loadDenseWeights directly so its preferBlockWeights flag reaches an MLX
// source.

} // namespace

TargetModelWeights loadDenseWeights(metal::MetalBackend &backend,
                                    const TargetLayout &layout,
                                    const TargetFiles &files,
                                    bool preferBlockWeights) {
  requireDenseLayout(layout);
  const auto readMixer = [](WeightFile &file, const TargetLayout &l,
                            const auto &format, const MixerGeometry &g,
                            bool full) {
    return readTargetMixer(file, l, format, g, full);
  };
  const auto readFfn = [&](WeightFile &file, TargetLayerWeights &layer, const auto &format) {
    auto &ffn = layer.ffn.emplace<DenseFfnWeights>();
    ffn.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    ffn.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    ffn.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  // An MLX source's block-layout images (preferBlockWeights) read through
  // AffineBlockTargetFormat; every other source takes its ordinary format.
  if (preferBlockWeights)
    if (const auto *mlx = std::get_if<std::reference_wrapper<AffineTargetLoader>>(&files))
      return readTargetModelWeights(backend, layout, mlx->get(),
                                    AffineBlockTargetFormat{}, readMixer, readFfn);
  return loadTargetWeights(backend, layout, files, readMixer, readFfn);
}

} // namespace richengine::model

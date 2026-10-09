#include "model/Dense.hpp"
#include "model/TargetLoader.hpp"
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

// Every dimension set, the query-only packed width consistent and every
// projection on the Q4 storage tiles; all layers are full attention and the
// recurrent state is empty.
void requireDenseLayout(const DenseLayout &layout) {
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

} // namespace

// The dense mixer's reader (found by readTargetMixer's ADL).
template <class Format>
MixerWeights readTargetMixer(const DenseLayout &, WeightFile &file,
                                 const Format &format,
                                 const MixerGeometry &geometry, bool) {
  return readDenseAttention(file, format, geometry);
}

DenseWeights loadDenseWeights(metal::MetalBackend &backend, DenseLayout layout,
                              const TargetFiles<DenseLayout> &files) {
  requireDenseLayout(layout);
  const auto readFfn = [&](WeightFile &file, DenseLayerWeights &layer, const auto &format) {
    layer.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    return readTargetModelWeights<DenseWeights>(backend, layout, gguf->get(),
                                               BlockTargetFormat{}, readFfn);
  if (const auto *mlx = std::get_if<std::reference_wrapper<AffineTargetLoader>>(&files))
    return readTargetModelWeights<DenseWeights>(backend, layout, mlx->get(),
                                               AffineTargetFormat{}, readFfn);
  const auto &packed = std::get<PackedTargetFiles<DenseLayout>>(files);
  return readTargetModelWeights<DenseWeights>(backend, layout, packed,
                                             AffineTargetFormat{packed.tiledEmbedding}, readFfn);
}

} // namespace richengine::model

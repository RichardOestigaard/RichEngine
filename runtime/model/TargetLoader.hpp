#pragma once

#include "model/AffineTarget.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/TargetModel.hpp"
#include "model/TargetFiles.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>
#include <variant>

namespace richengine::model {

// How a target's files store its tensors; loadTargetWeights pairs each source's
// files with their format. Affine files, packed or written from MLX, hold
// every projection, a fused one too, as one affine Q4 tensor, and bf16 norms.
// tiledEmbedding marks the packed format's 256-row embedding planes; the
// safetensors images' copies are the checkpoint's flat quantized rows.
struct AffineTargetFormat final {
  const bool tiledEmbedding;
  AffineTargetFormat(bool tiledEmbedding = false)
      : tiledEmbedding(tiledEmbedding) {}
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Grouped;

  [[nodiscard]] ops::NormWeights norm(WeightFile &file, uint32_t width, std::string_view label) const {
    return readNorm(file, width, false, label);
  }
  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readAffineProjection(file, outputSize, inputSize, label);
  }
  // The tensor `label`; block images keep the projection as `tensors`.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view label,
                                      std::initializer_list<std::string_view>) const {
    return projection(file, outputSize, inputSize, label);
  }
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readAffineEmbedding(file, outputSize, inputSize, "embedding",
                               tiledEmbedding);
  }
};

// GGUF images hold each GGUF tensor as one block-quantized segment,
// a fused projection as its tensors in output column order, and the GGUF's
// F32 norms. The GGUF keeps the GDN output projection's input columns in
// llama.cpp's tiled value-head order, so the GDN writes its output in it; a
// rotated Prism ML GGUF keeps them grouped, and rotateInputs (Qwen3_8.cpp)
// switches its GDN to that order.
struct BlockTargetFormat final {
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Tiled;

  [[nodiscard]] ops::NormWeights norm(WeightFile &file, uint32_t width, std::string_view label) const {
    return readNorm(file, width, true, label);
  }
  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readBlockProjection(file, outputSize, inputSize, label);
  }
  // The tensors, which may leave padding columns past the last one
  // (LinearGguf.cpp requireSegments); affine files keep one tensor.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view,
                                      std::initializer_list<std::string_view> tensors) const;
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readBlockEmbedding(file, outputSize, inputSize, "embedding");
  }
};

// Reads the mixer sections that follow a layer's input norm, in file order
// (instantiated for both formats).
template <class Format>
[[nodiscard]] MixerWeights readMixer(WeightFile &file, const Format &format,
                                             const MixerGeometry &geometry,
                                             bool fullAttention);

// The family's mixer reader; the default reads the hybrid GDN/full-attention
// mixer, Dense.hpp and Lfm2.hpp overload it for their mixers.
template <class Layout, class Format>
[[nodiscard]] MixerWeights readTargetMixer(const Layout &, WeightFile &file,
                                               const Format &format,
                                               const MixerGeometry &geometry,
                                               bool fullAttention) {
  return readMixer(file, format, geometry, fullAttention);
}

// Loads the packed files of a target directory: one per hybrid layer,
// head.bin and embedding.bin.
template <class Layout> struct PackedTargetFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const Layout &layout;
  // Whether the package's embedding.bin stores the 256-row tiled planes of
  // install/pack.py's packed formats; the splash-packed-q4* formats store
  // the checkpoint's flat quantized rows (ModelDescriptor::packedTiledEmbedding).
  bool tiledEmbedding = false;
  [[nodiscard]] WeightFile layer(uint32_t index) const {
    const std::string filename = "layer-" + std::to_string(index) + ".bin";
    return images.load(packedImage(directory / filename, "target/" + filename, Layout::layerMagic, index,
                                   layout.isFullAttentionLayer(index) ? 1U : 0U));
  }
  [[nodiscard]] WeightFile head() const {
    return images.load(packedImage(directory / "head.bin", "target/head.bin", Layout::headMagic, layout.layers, 2));
  }
  [[nodiscard]] WeightFile embedding() const {
    return images.load(packedImage(directory / "embedding.bin", "target/embedding.bin", kEmbeddingMagic,
                                   layout.vocabularySize, layout.hiddenSize));
  }
};

// Reads a target through the files of its images, packaged or written from
// an upstream source, in their format: per layer the input norm, mixer,
// post-attention norm and the architecture's FFN through readFfn, then the
// head and the token embedding. Weights is the architecture's weight struct.
template <class Weights, class Layout, class Files, class Format, class ReadFfn>
[[nodiscard]] Weights
readTargetModelWeights(metal::MetalBackend &backend, const Layout &layout, Files &&files,
                      const Format &format, ReadFfn readFfn) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    WeightFile file = files.layer(layerIndex);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
    layer.mixer = readTargetMixer(layout, file, format, layout.mixerGeometry(), fullAttention);
    layer.postAttentionNorm = format.norm(file, layout.hiddenSize, "post-attention-norm");
    readFfn(file, layer, format);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file = files.head();
    result.finalNorm = format.norm(file, layout.hiddenSize, "final-norm");
    result.logitsProjection =
        format.projection(file, layout.vocabularySize, layout.hiddenSize, "logits");
    // bf16 logits would round near-ties together: their spacing is 0.125 at
    // logits of 16 to 32.
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = files.embedding();
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

// Throws unless every dimension of a family's layout is set, the dimensions
// agree with each other and every projection fits the Q4 storage tiles.
template <class Layout> void requireTargetLayout(const Layout &layout) {
  const auto zero = [](auto... dimensions) { return ((dimensions == 0) || ...); };
  uint32_t ffnWidth = 0;
  bool ffnZero = false;
  bool routingInconsistent = false;
  if constexpr (Layout::ffnKind == FfnKind::Dense) {
    ffnWidth = layout.intermediateSize;
    ffnZero = zero(ffnWidth);
  } else {
    ffnWidth = layout.expertIntermediateSize;
    ffnZero = zero(layout.experts, layout.expertsPerToken, ffnWidth);
    routingInconsistent = layout.expertsPerToken > layout.experts;
  }
  if (ffnZero || !(layout.rotaryTheta > 0.0F) ||
      zero(layout.maximumContextTokens, layout.layers, layout.hiddenSize, layout.vocabularySize,
           layout.packedGdnWidth, layout.packedFullWidth, layout.convolutionDimension, layout.gdnKeyHeads,
           layout.gdnValueHeads, layout.gdnHeadDimension, layout.attentionWidth, layout.attentionQueryHeads,
           layout.attentionKvHeads, layout.attentionHeadDimension, layout.rotaryPairs,
           layout.fullAttentionPeriod))
    throw WeightStoreError("target layout contains a zero dimension");
  if (routingInconsistent || layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension != (2 * layout.gdnKeyHeads + layout.gdnValueHeads) * layout.gdnHeadDimension ||
      layout.attentionWidth != layout.attentionQueryHeads * layout.attentionHeadDimension ||
      // The GDN value rows are sized with attentionWidth throughout.
      layout.gdnValueHeads * layout.gdnHeadDimension != layout.attentionWidth ||
      layout.packedFullWidth !=
          Layout::attentionQueryStride * layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      std::ranges::any_of(layout.hiddenCaptureLayers, [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("target layout is inconsistent");
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(ffnWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, ffnWidth);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

// Checks the layout and loads a target from its files. The architecture
// reads its FFN through readFfn, called with the file, the layer and the
// format.
template <class Weights, class Layout, class ReadFfn>
[[nodiscard]] Weights
loadTargetWeights(metal::MetalBackend &backend, const Layout &layout, const TargetFiles<Layout> &files,
               ReadFfn readFfn) {
  requireTargetLayout(layout);
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    return readTargetModelWeights<Weights>(backend, layout, gguf->get(), BlockTargetFormat{}, readFfn);
  if (const auto *mlx = std::get_if<std::reference_wrapper<AffineTargetLoader>>(&files))
    return readTargetModelWeights<Weights>(backend, layout, mlx->get(), AffineTargetFormat{}, readFfn);
  const auto &packed = std::get<PackedTargetFiles<Layout>>(files);
  return readTargetModelWeights<Weights>(backend, layout, packed,
                                        AffineTargetFormat{packed.tiledEmbedding}, readFfn);
}

} // namespace richengine::model

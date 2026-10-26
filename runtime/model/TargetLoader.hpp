#pragma once

#include "model/AffineTarget.hpp"
#include "model/GgufTarget.hpp"
#include "model/OqTarget.hpp"
#include "model/TargetLayout.hpp"
#include "model/TargetModel.hpp"
#include "model/TargetFiles.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"

#include <cstdlib>
#include <type_traits>
#include "ops/GDN.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <span>
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
    // RICHENGINE_AF4P: read the affine planes as AF4P block segments, which
    // route the staged gguf_decode tiles (gguf_decode_af4p_*).
    static const bool af4p = std::getenv("RICHENGINE_AF4P") != nullptr;
    return af4p ? readAf4pProjection(file, outputSize, inputSize, label)
                : readAffineProjection(file, outputSize, inputSize, label);
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

// The dense target's block-layout MLX images (AffineTargetLoader under
// preferBlockWeights): every projection section is one AF4 segment, a fused
// one included, where a GGUF image splits it into its tensors; norms stay
// bf16 and the token table its flat affine rows, as AffineTargetFormat.
struct AffineBlockTargetFormat final {
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Grouped;

  [[nodiscard]] ops::NormWeights norm(WeightFile &file, uint32_t width, std::string_view label) const {
    return readNorm(file, width, false, label);
  }
  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readBlockProjection(file, outputSize, inputSize, label);
  }
  // A fused projection is one segment too: the parts' rows are stacked in
  // the tensor, not separate segments as in a GGUF image.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view label,
                                      std::initializer_list<std::string_view>) const {
    return projection(file, outputSize, inputSize, label);
  }
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readAffineEmbedding(file, outputSize, inputSize, "embedding", false);
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

// The oQ (oMLX mixed-precision) images of an MLX safetensors source
// (model/OqTarget.cpp): the same Block32 layout a GGUF image keeps — each
// projection a run of descriptor+plane segments, the fused ones read as a
// segment sequence — with bf16 norms and the GDN output projection's
// grouped column order, as the affine source keeps them.
struct OqTargetFormat final {
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Grouped;

  [[nodiscard]] ops::NormWeights norm(WeightFile &file, uint32_t width, std::string_view label) const {
    return readNorm(file, width, false, label);
  }
  // A projection reads its descriptor then its payload: affine tiles for a
  // ggml::kAffineQ4 section (the fast Affine64 kernels), a block segment for
  // the rest.
  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readOqProjection(file, outputSize, inputSize, label);
  }
  // The fused projection's segments in file order until they cover it —
  // or, for a ggml::kAffineQ4 section, its affine tiles; the tensor names
  // are only labels — mixed specs may merge or split them.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view label,
                                      std::initializer_list<std::string_view>) const {
    return readOqFused(file, outputSize, inputSize, label);
  }
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readOqEmbedding(file, outputSize, inputSize, "embedding");
  }
};

// Reads the mixer sections that follow a layer's input norm, in file order
// (instantiated for both formats).
template <class Format>
[[nodiscard]] MixerWeights readMixer(WeightFile &file, const Format &format,
                                             const MixerGeometry &geometry,
                                             bool fullAttention);

// The family's mixer reader, passed to loadTargetWeights and
// readTargetModelWeights as the readMixer callable (file, layout, format,
// geometry, fullAttention); this default reads the hybrid GDN/full-attention
// mixer — the families with another mixer (Dense.cpp, the LFM2s) pass their
// own.
template <class Format>
[[nodiscard]] MixerWeights readTargetMixer(const TargetLayout &, WeightFile &file,
                                               const Format &format,
                                               const MixerGeometry &geometry,
                                               bool fullAttention) {
  return readMixer(file, format, geometry, fullAttention);
}

// Loads the packed files of a target directory: one per hybrid layer,
// head.bin and embedding.bin.
struct PackedTargetFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const TargetLayout &layout;
  // Whether the package's embedding.bin stores the 256-row tiled planes of
  // install/pack.py's packed formats; the splash-packed-q4* formats store
  // the checkpoint's flat quantized rows (ModelDescriptor::packedTiledEmbedding).
  bool tiledEmbedding = false;
  [[nodiscard]] WeightFile layer(uint32_t index) const {
    const std::string filename = "layer-" + std::to_string(index) + ".bin";
    return images.load(packedImage(directory / filename, "target/" + filename, layout.layerMagic, index,
                                   layout.isFullAttentionLayer(index) ? 1U : 0U));
  }
  [[nodiscard]] WeightFile head() const {
    return images.load(packedImage(directory / "head.bin", "target/head.bin", layout.headMagic, layout.layers, 2));
  }
  [[nodiscard]] WeightFile embedding() const {
    return images.load(packedImage(directory / "embedding.bin", "target/embedding.bin", kEmbeddingMagic,
                                   layout.vocabularySize, layout.hiddenSize));
  }
};

// Reads a target through the files of its images, packaged or written from
// an upstream source, in their format: per layer the input norm, the
// architecture's mixer through readMixer, post-attention norm and its FFN
// through readFfn, then the head and the token embedding.
template <class Files, class Format, class ReadMixer, class ReadFfn>
[[nodiscard]] TargetModelWeights
readTargetModelWeights(metal::MetalBackend &backend, const TargetLayout &layout, Files &&files,
                      const Format &format, ReadMixer readMixer, ReadFfn readFfn) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  TargetModelWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    WeightFile file = files.layer(layerIndex);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm = format.norm(file, layout.hiddenSize, "input-norm");
    layer.mixer = readMixer(file, layout, format, layout.mixerGeometry(), fullAttention);
    layer.postAttentionNorm = format.norm(file, layout.hiddenSize, "post-attention-norm");
    readFfn(file, layer, format);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file = files.head();
    result.finalNorm = format.norm(file, layout.hiddenSize, "final-norm");
    result.logitsProjection =
        format.projection(file, paddedVocabularySize(layout.vocabularySize), layout.hiddenSize, "logits");
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
inline void requireTargetLayout(const TargetLayout &layout) {
  const auto zero = [](auto... dimensions) { return ((dimensions == 0) || ...); };
  uint32_t ffnWidth = 0;
  bool ffnZero = false;
  bool routingInconsistent = false;
  if (layout.ffnKind == FfnKind::Dense) {
    ffnWidth = layout.intermediateSize;
    ffnZero = zero(ffnWidth);
  } else {
    ffnWidth = layout.expertIntermediateSize;
    ffnZero = zero(layout.experts, layout.expertsPerToken, ffnWidth);
    routingInconsistent = layout.expertsPerToken > layout.experts;
  }
  // The GDN fields bind only when the target carries recurrent state
  // (gdnHeadDimension != 0); a conv-only target (convolutionTaps, no GDN
  // heads) needs just its FIFO geometry, a pure attention target neither.
  const bool gdn = layout.gdnHeadDimension != 0;
  const bool conv = layout.convolutionTaps != 0;
  // The full-attention schedule: a mask, a period, or Gemma's global
  // period — every target declares exactly one.
  const bool scheduled = layout.attentionMask || layout.fullAttentionPeriod ||
                         layout.globalLayerPeriod;
  if (ffnZero || !(layout.rotaryTheta > 0.0F) || !scheduled ||
      zero(layout.maximumContextTokens, layout.layers, layout.hiddenSize, layout.vocabularySize,
           layout.packedFullWidth, layout.attentionWidth, layout.attentionQueryHeads,
           layout.attentionKvHeads, layout.attentionHeadDimension, layout.rotaryPairs) ||
      (gdn && zero(layout.packedGdnWidth, layout.convolutionDimension,
                   layout.gdnKeyHeads, layout.gdnValueHeads)) ||
      (conv && zero(layout.packedGdnWidth, layout.convolutionDimension)))
    throw WeightStoreError("target layout contains a zero dimension");
  // The GDN consistency rows apply only to a recurrent target; for one the
  // value rows are sized with attentionWidth throughout.
  const bool gdnConsistent =
      !gdn ||
      (layout.gdnValueHeads % layout.gdnKeyHeads == 0 &&
       layout.convolutionDimension ==
           (2 * layout.gdnKeyHeads + layout.gdnValueHeads) * layout.gdnHeadDimension &&
       layout.gdnValueHeads * layout.gdnHeadDimension == layout.attentionWidth);
  if (routingInconsistent || !gdnConsistent ||
      layout.attentionWidth != layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.packedFullWidth !=
          layout.attentionQueryStride * layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      std::ranges::any_of(std::span(layout.hiddenCaptureLayers).first(layout.captureLayerCount),
                          [&](uint32_t layer) { return layer >= layout.layers; }) ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid())
    throw WeightStoreError("target layout is inconsistent");
  if (gdn || conv)
    validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(ffnWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, ffnWidth);
  validateQ4Layout(paddedVocabularySize(layout.vocabularySize), layout.hiddenSize);
}

// Checks the layout and loads a target from its files. The architecture
// reads its mixer through readMixer, called with the file, the layout, the
// format, the mixer geometry and whether the layer is full attention, and
// its FFN through readFfn, called with the file, the layer and the format.
template <class ReadMixer, class ReadFfn>
[[nodiscard]] TargetModelWeights
loadTargetWeights(metal::MetalBackend &backend, const TargetLayout &layout, const TargetFiles &files,
               ReadMixer readMixer, ReadFfn readFfn) {
  requireTargetLayout(layout);
  if (const auto *gguf = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files))
    return readTargetModelWeights(backend, layout, gguf->get(), BlockTargetFormat{}, readMixer, readFfn);
  if (const auto *mlx = std::get_if<std::reference_wrapper<AffineTargetLoader>>(&files))
    return readTargetModelWeights(backend, layout, mlx->get(), AffineTargetFormat{}, readMixer, readFfn);
  if (const auto *oq = std::get_if<std::reference_wrapper<OqTargetLoader>>(&files))
    return readTargetModelWeights(backend, layout, oq->get(), OqTargetFormat{}, readMixer, readFfn);
  const auto &packed = std::get<PackedTargetFiles>(files);
  return readTargetModelWeights(backend, layout, packed,
                                        AffineTargetFormat{packed.tiledEmbedding}, readMixer, readFfn);
}

} // namespace richengine::model

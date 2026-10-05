#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenTargetLoader.hpp"

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace splash::model {

void requireCompatibleModelPackage(const ModelPackage &package) {
  const DFlashDraftLayout &draftLayout =
      std::visit([](const auto &weights) -> const DFlashDraftLayout & {
        return weights.layout;
      }, package.draft);
  if (!package.descriptor.valid() ||
      package.descriptor.draft != draftLayout ||
      !std::visit(
          [&](const auto &target) {
            return package.descriptor.target == TargetLayout{target.layout} &&
                   target.layout.vocabularySize ==
                       draftLayout.vocabularySize;
          },
          package.target)) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

namespace {

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles<Qwen3_8Layout> &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Ornith9BLayout &layout,
                         const QwenTargetFiles<Ornith9BLayout> &files) {
  return loadOrnith9BWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles<Qwen3_6MoeLayout> &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const DenseLayout &layout,
                         const QwenTargetFiles<DenseLayout> &files) {
  return loadDenseWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Lfm2Layout &layout,
                         const QwenTargetFiles<Lfm2Layout> &files) {
  return loadLfm2Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Lfm2MoeLayout &layout,
                         const QwenTargetFiles<Lfm2MoeLayout> &files) {
  return loadLfm2MoeWeights(backend, layout, files);
}

template <class Image> uint64_t imageBytes(const std::vector<Image> &images) {
  uint64_t total = 0;
  for (const Image &image : images) total += image.bytes;
  return total;
}

} // namespace

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor) {
  ModelPackage result;
  result.descriptor = descriptor;
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  result.images = std::make_shared<WeightImages>(backend, result.descriptor.sourceIdentity);
  WeightImages &images = *result.images;
  // Every source's metadata is checked before the first image is written:
  // the vision tower's and the draft's here, the target's by its loader.
  const auto vision = planVisionLoader(root, result.descriptor);
  const bool plainDraft = result.descriptor.draft.kind == DraftKind::Plain;
  const bool dsparkDraft = result.descriptor.draft.kind == DraftKind::DSpark;
  std::optional<DraftCheckpointLoader> draft;
  std::optional<PlainDraftCheckpointLoader> plainDraftLoader;
  std::optional<DSparkCheckpointLoader> dsparkDraftLoader;
  if (result.descriptor.draftFromCheckpoint()) {
    if (plainDraft)
      plainDraftLoader.emplace(images, root / "draft", result.descriptor.draft);
    else if (dsparkDraft)
      dsparkDraftLoader.emplace(images, root / "draft", result.descriptor.draft);
    else
      draft.emplace(images, root / "draft", result.descriptor.draft);
  }
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        using Layout = std::remove_cvref_t<decltype(layout)>;
        const std::filesystem::path directory = root / "target";
        switch (result.descriptor.targetSource) {
        case TargetSource::Packed:
          return readTarget(backend, layout, PackedTargetFiles<Layout>{images, directory, layout});
        case TargetSource::Mlx: {
          AffineTargetLoader loader(images, directory, layout);
          return readTarget(backend, layout, std::ref(loader));
        }
        case TargetSource::Gguf: {
          GgufTargetLoader loader(backend, images, findTargetGguf(directory), ggufTargetGeometry(layout));
          return readTarget(backend, layout, std::ref(loader));
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
  if (plainDraft) {
    result.draft = loadPlainDraftWeights(
        backend,
        plainDraftLoader
            ? PlainDraftFiles(std::ref(*plainDraftLoader))
            : PlainDraftFiles(PackedPlainDraftFiles{images, root / "draft", result.descriptor.draft}),
        result.descriptor.draft);
  } else if (dsparkDraft) {
    result.draft = loadDSparkDraftWeights(
        backend,
        dsparkDraftLoader
            ? DSparkDraftFiles(std::ref(*dsparkDraftLoader))
            : DSparkDraftFiles(PackedDSparkDraftFiles{images, root / "draft", result.descriptor.draft}),
        result.descriptor.draft);
  } else {
    result.draft = loadDFlashDraftWeights(
        backend,
        draft ? DraftFiles(std::ref(*draft))
              : DraftFiles(PackedDraftFiles{images, root / "draft", result.descriptor.draft}),
        result.descriptor.draft);
  }
  result.vision = loadVisionWeights(backend, images, root, result.descriptor, vision.get());

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  const std::span<const WeightFileRecord> draftFiles = std::visit(
      [](const auto &weights) { return std::span<const WeightFileRecord>(weights.files); },
      result.draft);
  records.insert(records.end(), draftFiles.begin(), draftFiles.end());
  records.insert(records.end(), result.vision.files.begin(),
                 result.vision.files.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModelPackage(result);
  return result;
}

std::unique_ptr<VisionLoader> planVisionLoader(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  if (descriptor.visionSource != VisionSource::Mlx && descriptor.visionSource != VisionSource::Gguf)
    return nullptr;
  return std::make_unique<VisionLoader>(root / "vision", descriptor.visionSource, descriptor.vision);
}

QwenVisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                    const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                    const VisionLoader *loader) {
  if (loader) return loadQwenVisionWeights(backend, images, *loader);
  if (descriptor.visionSource == VisionSource::Packed)
    return loadQwenVisionWeights(backend, images, root / "vision", descriptor.vision);
  return {};
}

uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  uint64_t bytes = 0;
  if (descriptor.targetSource == TargetSource::Gguf) {
    WeightSource source(findTargetGguf(root / "target"));
    const GgufFile file(source);
    bytes = std::visit(
        [&](const auto &layout) { return imageBytes(gguf::planImages(file, ggufTargetGeometry(layout))); },
        descriptor.target);
  } else if (descriptor.targetSource == TargetSource::Mlx) {
    bytes = std::visit([](const auto &layout) { return imageBytes(affineTargetImages(layout)); }, descriptor.target);
  }
  if (descriptor.draftFromCheckpoint())
    bytes += descriptor.draft.kind == DraftKind::Plain
                 ? imageBytes(plainDraftCheckpointImages(descriptor.draft))
                 : descriptor.draft.kind == DraftKind::DSpark
                       ? imageBytes(dsparkDraftCheckpointImages(descriptor.draft))
                       : imageBytes(draftCheckpointImages(descriptor.draft));
  if (descriptor.visionSource == VisionSource::Mlx || descriptor.visionSource == VisionSource::Gguf)
    bytes += visionImageBytes(descriptor.vision);
  for (std::string_view directory : {"target", "draft", "vision"}) {
    if (directory == "vision" && descriptor.visionSource != VisionSource::Packed) continue;
    if (directory == "draft" && descriptor.draftFromCheckpoint()) continue;
    if (directory == "target" && descriptor.targetSource != TargetSource::Packed) continue;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root / directory)) {
      if (!entry.is_regular_file()) continue;
      const uint64_t size = entry.file_size();
      if (size > std::numeric_limits<uint64_t>::max() - bytes) throw std::overflow_error("model weight size overflows");
      bytes += size;
    }
  }
  if (!bytes) throw std::invalid_argument("model package contains no regular files");
  return bytes;
}

} // namespace splash::model

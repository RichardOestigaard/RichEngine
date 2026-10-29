#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/AffineToBlock.hpp"
#include "model/Dense.hpp"
#include "model/NullDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/ModelRegistry.hpp"
#include "model/OqTarget.hpp"
#include "model/TargetLoader.hpp"

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace richengine::model {

void requireCompatibleModelPackage(const ModelPackage &package) {
  const DFlashDraftLayout &draftLayout =
      std::visit([](const auto &weights) -> const DFlashDraftLayout & {
        return weights.layout;
      }, package.draft);
  const TargetLayout &targetLayout = std::visit(
      [](const TargetModelWeights &weights) -> const TargetLayout & {
        return weights.layout;
      },
      package.target);
  if (!package.descriptor.valid() ||
      package.descriptor.draft != draftLayout ||
      package.descriptor.target != targetLayout ||
      targetLayout.vocabularySize != draftLayout.vocabularySize) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

namespace {

// The registry lookup the per-layout overload set was: the family's
// registered load*Weights entry, keyed on TargetLayout::arch.
TargetWeights readTarget(metal::MetalBackend &backend,
                         const TargetLayout &layout,
                         const TargetFiles &files) {
  const auto *arch = findTargetArch(layout.arch);
  if (!arch)
    throw std::invalid_argument("unknown target architecture " +
                                std::string(layout.arch));
  return arch->loadTarget(backend, layout, files);
}

template <class Image> uint64_t imageBytes(const std::vector<Image> &images) {
  uint64_t total = 0;
  for (const Image &image : images) total += image.bytes;
  return total;
}

} // namespace

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor,
                              bool preferBlockWeights) {
  ModelPackage result;
  result.descriptor = descriptor;
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  result.images = std::make_shared<WeightImages>(backend, result.descriptor.sourceIdentity);
  WeightImages &images = *result.images;
  // Every source's metadata is checked before the first image is written:
  // the vision tower's and the draft's here, the target's by its loader.
  const auto vision = planVisionLoader(root, result.descriptor);
  const bool nullDraft = result.descriptor.draft.kind == DraftKind::Null;
  const bool dflashV1Draft = result.descriptor.draft.kind == DraftKind::DFlashV1;
  const bool dsparkDraft = result.descriptor.draft.kind == DraftKind::DSpark;
  std::optional<DFlash2CheckpointLoader> draft;
  std::optional<DFlashV1DraftCheckpointLoader> dflashV1DraftLoader;
  std::optional<DSparkCheckpointLoader> dsparkDraftLoader;
  if (result.descriptor.draftFromCheckpoint() && !nullDraft) {
    if (dflashV1Draft)
      dflashV1DraftLoader.emplace(images, root / "draft", result.descriptor.draft);
    else if (dsparkDraft)
      dsparkDraftLoader.emplace(images, root / "draft", result.descriptor.draft);
    else
      draft.emplace(images, root / "draft", result.descriptor.draft);
  }
  // The kernel layout the target's weights loaded in, for /status.
  bool targetBlock = false;
  const TargetLayout &layout = result.descriptor.target;
  const std::filesystem::path directory = root / "target";
  switch (result.descriptor.targetSource) {
  case TargetSource::Packed:
    result.target = readTarget(backend, layout,
                               PackedTargetFiles{images, directory, layout,
                                                 result.descriptor.packedTiledEmbedding});
    break;
  case TargetSource::Mlx:
    // A packed-only layout declares it (TargetLayout::packedOnly); the
    // affine and GGUF planners have no tensor map for it.
    if (layout.packedOnly)
      throw std::invalid_argument(
          "this target loads from packed files only");
    // An oQ (oMLX mixed-precision) checkpoint plans into the block
    // weight layout instead of the affine tiles; the probe reads only
    // its config.json.
    if (oqCheckpoint(directory)) {
      OqTargetLoader loader(images, directory, layout);
      targetBlock = true;
      result.target = readTarget(backend, layout, std::ref(loader));
    } else if (layout.mlxBlockWeights) {
      // Dense's block path bypasses the registered loader so
      // preferBlockWeights can reach loadDenseWeights: it plans and reads
      // the images' projections in the block layout. Sections without a
      // block form load affine instead.
      if (preferBlockWeights) {
        try {
          AffineTargetLoader loader(images, directory, layout, true);
          result.target =
              loadDenseWeights(backend, layout, std::ref(loader), true);
          targetBlock = true;
        } catch (const WeightStoreError &) {
        }
      }
      if (!targetBlock) {
        AffineTargetLoader loader(images, directory, layout);
        result.target =
            loadDenseWeights(backend, layout, std::ref(loader), false);
      }
    } else {
      AffineTargetLoader loader(images, directory, layout);
      result.target = readTarget(backend, layout, std::ref(loader));
    }
    break;
  case TargetSource::Gguf:
    if (layout.packedOnly)
      throw std::invalid_argument(
          "this target loads from packed files only");
    {
      GgufTargetLoader loader(backend, images, findTargetGguf(directory),
                              ggufTargetGeometry(layout));
      targetBlock = true;
      result.target = readTarget(backend, layout, std::ref(loader));
    }
    break;
  default:
    throw std::invalid_argument("unknown target source");
  }
  result.weightLayout =
      targetBlock ? ops::WeightLayout::Block32 : ops::WeightLayout::Affine64;
  if (nullDraft) {
    result.draft = NullDraftWeights{result.descriptor.draft, {}, 0};
  } else if (dflashV1Draft) {
    result.draft = loadDFlashV1DraftWeights(
        backend,
        dflashV1DraftLoader
            ? DFlashV1DraftFiles(std::ref(*dflashV1DraftLoader))
            : DFlashV1DraftFiles(PackedDFlashV1DraftFiles{images, root / "draft", result.descriptor.draft}),
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
  result.vision = loadVisionWeights(backend, images, root, result.descriptor, vision);
  // The audio encoder binds its own sections of the same shared mmproj.
  result.audio = loadAudioWeights(backend, images, root, result.descriptor);
  // As does the speech decoder (its a.gen.* sections), decoded host-side.
  result.speech = loadSpeechWeights(backend, images, root, result.descriptor);

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  const std::span<const WeightFileRecord> draftFiles = std::visit(
      [](const auto &weights) { return std::span<const WeightFileRecord>(weights.files); },
      result.draft);
  records.insert(records.end(), draftFiles.begin(), draftFiles.end());
  const std::span<const WeightFileRecord> visionFiles = result.visionFiles();
  records.insert(records.end(), visionFiles.begin(), visionFiles.end());
  const std::span<const WeightFileRecord> audioFiles = result.audioFiles();
  records.insert(records.end(), audioFiles.begin(), audioFiles.end());
  const std::span<const WeightFileRecord> speechFiles = result.speechFiles();
  records.insert(records.end(), speechFiles.begin(), speechFiles.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModelPackage(result);
  return result;
}

VisionLoaderPlan planVisionLoader(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  if (descriptor.visionSource != VisionSource::Mlx && descriptor.visionSource != VisionSource::Gguf)
    return std::monostate{};
  if (descriptor.visionTower == VisionTower::Lfm3Vl)
    return std::make_unique<LfmVisionLoader>(root / "vision", descriptor.lfmVision);
  return std::make_unique<VisionLoader>(root / "vision", descriptor.visionSource, descriptor.vision);
}

VisionWeights loadVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                                const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                const VisionLoaderPlan &loader) {
  if (const auto *qwen = std::get_if<std::unique_ptr<VisionLoader>>(&loader))
    return loadQwenVisionWeights(backend, images, **qwen);
  if (const auto *lfm = std::get_if<std::unique_ptr<LfmVisionLoader>>(&loader))
    return loadLfmVisionWeights(backend, images, **lfm);
  if (descriptor.visionSource == VisionSource::Packed)
    return loadQwenVisionWeights(backend, images, root / "vision", descriptor.vision);
  return QwenVisionWeights{};
}

AudioWeights loadAudioWeights(metal::MetalBackend &backend, WeightImages &images,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor) {
  if (descriptor.audioSource != AudioSource::Gguf)
    return std::monostate{};
  // The encoder and the vision tower share the omni mmproj file; each
  // loader binds only the tensors it names.
  LfmAudioLoader loader(root / "vision", descriptor.lfmAudio);
  return loadLfmAudioWeights(backend, images, loader);
}

std::optional<LfmSpeechModel>
loadSpeechWeights(metal::MetalBackend &backend, WeightImages &images,
                  const std::filesystem::path &root,
                  const ModelDescriptor &descriptor) {
  if (descriptor.speechSource != SpeechSource::Gguf)
    return std::nullopt;
  // The decoder and the towers share the omni mmproj file; the loader
  // binds only the a.gen.* tensors it names and validates their metadata.
  LfmGenLoader loader(root / "vision");
  return loadLfmSpeechWeights(backend, images, loader);
}

uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                          bool preferBlockWeights) {
  uint64_t bytes = 0;
  const TargetLayout &layout = descriptor.target;
  if (descriptor.targetSource == TargetSource::Gguf) {
    WeightSource source(findTargetGguf(root / "target"));
    const GgufFile file(source);
    bytes = imageBytes(gguf::planImages(file, ggufTargetGeometry(layout)));
  } else if (descriptor.targetSource == TargetSource::Mlx) {
    const std::filesystem::path directory = root / "target";
    // A packed-only layout never reaches this branch's planners (the
    // loader throws first); it keeps the affine estimate below.
    if (layout.packedOnly) {
      bytes = imageBytes(affineTargetImages(layout));
    } else if (oqCheckpoint(directory)) {
      bytes = oqTargetImageBytes(directory, layout);
    } else {
      if (layout.mlxBlockWeights && preferBlockWeights) {
        // Sections without a block form budget for the affine load the
        // fallback below performs.
        try {
          bytes = imageBytes(
              affine::blockImages(affineTargetImages(layout)));
        } catch (const WeightStoreError &) {
          bytes = imageBytes(affineTargetImages(layout));
        }
      } else {
        bytes = imageBytes(affineTargetImages(layout));
      }
    }
  }
  if (descriptor.draftFromCheckpoint() &&
      descriptor.draft.kind != DraftKind::Null) {
    auto draftImagesFor = [&descriptor] {
      return descriptor.draft.kind == DraftKind::DFlashV1
                 ? dflashV1DraftCheckpointImages(descriptor.draft)
                 : descriptor.draft.kind == DraftKind::DSpark
                       ? dsparkDraftCheckpointImages(descriptor.draft)
                       : draftCheckpointImages(descriptor.draft);
    };
    if (preferBlockWeights) {
      // Sections without a block form budget for the affine load the
      // loader's fallback performs.
      try {
        bytes += imageBytes(affine::blockImages(draftImagesFor()));
      } catch (const WeightStoreError &) {
        bytes += imageBytes(draftImagesFor());
      }
    } else {
      bytes += imageBytes(draftImagesFor());
    }
  }
  if (descriptor.visionSource == VisionSource::Mlx || descriptor.visionSource == VisionSource::Gguf)
    bytes += descriptor.visionTower == VisionTower::Lfm3Vl
                 ? lfmVisionImageBytes(descriptor.lfmVision)
                 : visionImageBytes(descriptor.vision);
  if (descriptor.audioSource == AudioSource::Gguf)
    bytes += lfmAudioImageBytes(descriptor.lfmAudio);
  if (descriptor.speechSource == SpeechSource::Gguf)
    bytes += lfmGenImageBytes();
  for (std::string_view directory : {"target", "draft", "vision"}) {
    if (directory == "vision" && descriptor.visionSource != VisionSource::Packed) continue;
    if (directory == "draft" &&
        (descriptor.draftFromCheckpoint() ||
         descriptor.draft.kind == DraftKind::Null))
      continue;
    if (!std::filesystem::exists(root / directory)) continue;
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

} // namespace richengine::model

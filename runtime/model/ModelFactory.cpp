#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/AffineToBlock.hpp"
#include "model/Granite.hpp"
#include "model/NullDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufTarget.hpp"
#include "model/OqTarget.hpp"
#include "model/TargetLoader.hpp"

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace richengine::model {

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
                         const TargetFiles<Qwen3_8Layout> &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Ornith9BLayout &layout,
                         const TargetFiles<Ornith9BLayout> &files) {
  return loadOrnith9BWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const TargetFiles<Qwen3_6MoeLayout> &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const MiniCpmV47MoeLayout &layout,
                         const TargetFiles<MiniCpmV47MoeLayout> &files) {
  return loadMiniCpmV47MoeWeights(backend, layout, files);
}

// Packed and GGUF files fix the layout they store, so the flag only reaches
// the MLX case below, which calls loadDenseWeights itself.
TargetWeights readTarget(metal::MetalBackend &backend, const DenseLayout &layout,
                         const TargetFiles<DenseLayout> &files) {
  return loadDenseWeights(backend, layout, files, false);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Lfm2Layout &layout,
                         const TargetFiles<Lfm2Layout> &files) {
  return loadLfm2Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Lfm2MoeLayout &layout,
                         const TargetFiles<Lfm2MoeLayout> &files) {
  return loadLfm2MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Lfm2Moe24BLayout &layout,
                         const TargetFiles<Lfm2Moe24BLayout> &files) {
  return loadLfm2Moe24BWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const GraniteLayout &layout,
                         const TargetFiles<GraniteLayout> &files) {
  return loadGraniteWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Gemma4MoeLayout &layout,
                         const TargetFiles<Gemma4MoeLayout> &files) {
  return loadGemma4MoeWeights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend,
                         const DiffusionGemmaLayout &layout,
                         const TargetFiles<DiffusionGemmaLayout> &files) {
  return loadDiffusionGemmaWeights(backend, layout, files);
}

// Whether the layout declares packed-only loading
// (Gemma4MoeLayout::packedOnly): true only where the member exists and is
// true, false for a layout without it.
template <class Layout>
concept PackedOnlyLayout = requires { requires Layout::packedOnly; };

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
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        using Layout = std::remove_cvref_t<decltype(layout)>;
        const std::filesystem::path directory = root / "target";
        switch (result.descriptor.targetSource) {
        case TargetSource::Packed:
          return readTarget(backend, layout,
                            PackedTargetFiles<Layout>{images, directory, layout,
                                                      result.descriptor.packedTiledEmbedding});
        case TargetSource::Mlx: {
          // A packed-only layout declares it (Gemma4MoeLayout::packedOnly);
          // the affine and GGUF planners have no tensor map for it.
          if constexpr (PackedOnlyLayout<Layout>)
            throw std::invalid_argument(
                "this target loads from packed files only");
          // An oQ (oMLX mixed-precision) checkpoint plans into the block
          // weight layout instead of the affine tiles; the probe reads only
          // its config.json.
          else if (oqCheckpoint(directory)) {
            OqTargetLoader loader(images, directory, layout);
            targetBlock = true;
            return readTarget(backend, layout, std::ref(loader));
          }
          else if constexpr (std::is_same_v<Layout, DenseLayout>) {
            // preferBlockWeights plans and reads the images' projections in
            // the block layout; every other layout stays affine this
            // milestone. Sections without a block form load affine instead.
            if (preferBlockWeights) {
              try {
                AffineTargetLoader loader(images, directory, layout, true);
                targetBlock = true;
                return loadDenseWeights(backend, layout, std::ref(loader), true);
              } catch (const WeightStoreError &) {
              }
            }
            AffineTargetLoader loader(images, directory, layout);
            return loadDenseWeights(backend, layout, std::ref(loader), false);
          } else {
            AffineTargetLoader loader(images, directory, layout);
            return readTarget(backend, layout, std::ref(loader));
          }
        }
        case TargetSource::Gguf: {
          if constexpr (PackedOnlyLayout<Layout>)
            throw std::invalid_argument(
                "this target loads from packed files only");
          else {
            GgufTargetLoader loader(backend, images, findTargetGguf(directory), ggufTargetGeometry(layout));
            targetBlock = true;
            return readTarget(backend, layout, std::ref(loader));
          }
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
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

uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                          bool preferBlockWeights) {
  uint64_t bytes = 0;
  if (descriptor.targetSource == TargetSource::Gguf) {
    WeightSource source(findTargetGguf(root / "target"));
    const GgufFile file(source);
    bytes = std::visit(
        [&](const auto &layout) { return imageBytes(gguf::planImages(file, ggufTargetGeometry(layout))); },
        descriptor.target);
  } else if (descriptor.targetSource == TargetSource::Mlx) {
    const std::filesystem::path directory = root / "target";
    bytes = std::visit(
        [&](const auto &layout) -> uint64_t {
          using Layout = std::remove_cvref_t<decltype(layout)>;
          // Packed-only layouts never take this branch; the probe and the oQ
          // planner are not instantiated for them.
          if constexpr (!PackedOnlyLayout<Layout>) {
            if (oqCheckpoint(directory)) return oqTargetImageBytes(directory, layout);
            if constexpr (std::is_same_v<Layout, DenseLayout>) {
              if (preferBlockWeights) {
                // Sections without a block form budget for the affine load the
                // fallback below performs.
                try {
                  return imageBytes(affine::blockImages(affineTargetImages(layout)));
                } catch (const WeightStoreError &) {
                }
              }
            }
          }
          return imageBytes(affineTargetImages(layout));
        },
        descriptor.target);
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
    bytes += visionImageBytes(descriptor.vision);
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

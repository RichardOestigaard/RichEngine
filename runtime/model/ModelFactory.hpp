#pragma once

#include "ops/Vision.hpp"
#include "DFlashDraft.hpp"
#include "DFlashTwoDraft.hpp"
#include "DSparkDraft.hpp"
#include "DFlashV1Draft.hpp"
#include "ModelDescriptor.hpp"
#include "TargetWeights.hpp"
#include "QwenVision.hpp"
#include "LfmVision.hpp"
#include "LfmAudio.hpp"
#include "model/NullDraft.hpp"
#include "VisionLoader.hpp"
#include "LfmVisionLoader.hpp"
#include "LfmAudioLoader.hpp"
#include "LfmGenLoader.hpp"
#include "LfmGen.hpp"
#include "ops/PageStorage.hpp"
#include "ops/ExecutionPlans.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace richengine::ops {
class FfnSplit;
} // namespace richengine::ops

namespace richengine::model {

class CompositeStateStorage;

using DraftWeights =
    std::variant<NullDraftWeights, DFlashDraftWeights, DFlashV1DraftWeights,
                 DSparkDraftWeights>;

// The vision tower's weights, whichever tower the model serves; the default
// alternative is the empty Qwen weights a text-only model carries.
using VisionWeights = std::variant<QwenVisionWeights, LfmVisionWeights>;

// The lfm3a audio encoder's weights; monostate for every model without one.
using AudioWeights = std::variant<std::monostate, LfmAudioWeights>;

struct ModelPackage final {
  ModelDescriptor descriptor;
  // The memory of every image the weights below are views of.
  std::shared_ptr<WeightImages> images;
  TargetWeights target;
  DraftWeights draft;
  VisionWeights vision;
  AudioWeights audio;
  // The lfm3a_gen speech decoder's host-side weights; absent for a model
  // without speech output (its audio/lfm-gen.bin image is not loaded).
  std::optional<LfmSpeechModel> speech;
  std::string manifestFingerprintSha256;
  // The kernel layout the target's weights actually loaded in (a requested
  // block layout falls back to affine on sections without a block form).
  ops::WeightLayout weightLayout = ops::WeightLayout::Affine64;

  [[nodiscard]] const std::string &name() const noexcept {
    return descriptor.name;
  }
  [[nodiscard]] kv::Layout targetKvLayout(kv::Format format) const noexcept {
    auto layout = descriptor.targetKvLayout;
    layout.format = format;
    return layout;
  }
  [[nodiscard]] CompositeStateLayout stateLayout() const noexcept {
    return descriptor.stateLayout;
  }
  [[nodiscard]] uint32_t maximumContextTokens() const noexcept {
    return descriptor.capabilities.maximumContextTokens;
  }
  [[nodiscard]] uint64_t targetActualAllocatedBytes() const noexcept {
    return std::visit([](const TargetModelWeightsBase &weights) {
      return weights.actualAllocatedBytes;
    }, target);
  }
  [[nodiscard]] uint64_t draftActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) {
      return weights.actualAllocatedBytes;
    }, draft);
  }
  [[nodiscard]] const std::string &targetManifestFingerprint() const noexcept {
    return std::visit([](const TargetModelWeightsBase &weights) -> const std::string & {
      return weights.manifestFingerprintSha256;
    }, target);
  }
  [[nodiscard]] std::span<const WeightFileRecord> targetFiles() const noexcept {
    return std::visit([](const TargetModelWeightsBase &weights) ->
                          std::span<const WeightFileRecord> {
      return weights.files;
    }, target);
  }
  // The vision tower's file records and loaded bytes, whichever tower it is;
  // both are empty for a text-only model.
  [[nodiscard]] std::span<const WeightFileRecord> visionFiles() const noexcept {
    return std::visit([](const auto &weights) -> std::span<const WeightFileRecord> {
      return weights.files;
    }, vision);
  }
  [[nodiscard]] uint64_t visionActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) { return weights.actualAllocatedBytes; },
                      vision);
  }
  // The audio encoder's file records and loaded bytes; both are empty for a
  // model without the lfm3a tower.
  [[nodiscard]] std::span<const WeightFileRecord> audioFiles() const noexcept {
    return std::visit([](const auto &weights) -> std::span<const WeightFileRecord> {
      if constexpr (requires { weights.files; })
        return weights.files;
      return {};
    }, audio);
  }
  [[nodiscard]] uint64_t audioActualAllocatedBytes() const noexcept {
    return std::visit([](const auto &weights) -> uint64_t {
      if constexpr (requires { weights.actualAllocatedBytes; })
        return weights.actualAllocatedBytes;
      return 0;
    }, audio);
  }
  // The speech decoder's file records and loaded bytes; both are empty for
  // a model without the lfm3a_gen tower.
  [[nodiscard]] std::span<const WeightFileRecord> speechFiles() const noexcept {
    return speech ? std::span<const WeightFileRecord>(speech->files)
                  : std::span<const WeightFileRecord>{};
  }
  [[nodiscard]] uint64_t speechActualAllocatedBytes() const noexcept {
    return speech ? speech->actualAllocatedBytes : 0;
  }
};

// Model execution resources, which the engine assembles. What a request's
// start allocates is admitted through the state storage.
struct RuntimeContext final {
  metal::MetalBackend &backend;
  const ModelPackage &package;
  kv::PageStorage &kvPages;
  CompositeStateStorage &stateStorage;
  const ops::ExecutionPlans &operators;
  // The prefill FFN's Neural Engine split (engine::startAneFfn), if any.
  ops::FfnSplit *aneFfn = nullptr;
};

// Validates only the interface between independently defined target and draft
// architectures. Each architecture validates its own tensor and state layout.
void requireCompatibleModelPackage(const ModelPackage &package);

// The bytes of every image the model's weights load into. preferBlockWeights
// sizes a dense MLX target's images in the block layout (loadModelPackage).
[[nodiscard]] uint64_t modelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                        bool preferBlockWeights = false);

// The vision role's upstream source, planned: a Qwen loader for the
// qwen3vl_merger tower, an LFM loader for lfm3vl, monostate for a packed
// vision file or a model without vision.
using VisionLoaderPlan =
    std::variant<std::monostate, std::unique_ptr<VisionLoader>,
                 std::unique_ptr<LfmVisionLoader>>;
[[nodiscard]] VisionLoaderPlan
planVisionLoader(const std::filesystem::path &root, const ModelDescriptor &descriptor);
// The vision role: written by the plan's loader when there is one, else the
// packed file; empty weights for a model without vision.
[[nodiscard]] VisionWeights
loadVisionWeights(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &root,
                  const ModelDescriptor &descriptor, const VisionLoaderPlan &loader);

// The audio encoder's upstream source, planned: an LFM audio loader for a
// descriptor whose audioSource is Gguf, monostate otherwise.
[[nodiscard]] AudioWeights
loadAudioWeights(metal::MetalBackend &backend, WeightImages &images,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor);

// The speech decoder's upstream source, planned: an LFM gen loader bound
// to the same shared mmproj for a descriptor whose speechSource is Gguf,
// empty otherwise.
[[nodiscard]] std::optional<LfmSpeechModel>
loadSpeechWeights(metal::MetalBackend &backend, WeightImages &images,
                  const std::filesystem::path &root,
                  const ModelDescriptor &descriptor);

// Production loading is selected by the validated package descriptor. There
// is one shared engine and DFlash controller; only model execution differs.
// preferBlockWeights loads a dense target from an MLX source with its
// projections in the GGUF block layout (GGUF_FMT_AF4 planes, the block
// kernels); every other source and architecture is unaffected.
[[nodiscard]] ModelPackage
loadModelPackage(metal::MetalBackend &backend,
                 const std::filesystem::path &root,
                 const ModelDescriptor &descriptor,
                 bool preferBlockWeights = false);

// The FFN layers of a dense target, which the prefill FFN's Neural Engine
// split (ops::AneFfn) may take; none for another target.
[[nodiscard]] std::vector<ops::SwiGluProjections> aneFfnLayers(const ModelPackage &model);
// Runs `use` on a prefill arena allocated for the call: on the dense FFN's
// buffers of a full chunk and the hidden rows layers alternate between.
void withPrefillArena(
    metal::MetalBackend &backend, const ModelPackage &model, const ops::ExecutionPlans &operators, kv::Format format,
    const std::function<void(const ops::PrefillFfnBuffers &, const std::array<metal::MetalBuffer, 2> &)> &use);
[[nodiscard]] ModelMemoryPlan
plannedRuntimeMemory(const DeviceCapabilities &device,
                     const ModelPackage &package,
                     const ops::ExecutionPlans &operators,
                     kv::Format format);
[[nodiscard]] std::unique_ptr<RuntimeModel>
createRuntime(RuntimeContext context);

} // namespace richengine::model

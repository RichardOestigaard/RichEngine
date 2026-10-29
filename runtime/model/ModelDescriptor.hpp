#pragma once

#include "model/TargetLayout.hpp"
#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "ModelTuning.hpp"
#include "ops/Vision.hpp"
#include "ops/LfmVision.hpp"
#include "ops/LfmAudioEnc.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace richengine::model {

// Where a model's weights come from: files already in the packed layout, or
// an MLX or GGUF checkpoint prepared into images when it loads. The vision
// tower is None for a model installed with --language-only.
enum class TargetSource : uint8_t { Packed, Mlx, Gguf };
enum class VisionSource : uint8_t { Packed, Mlx, Gguf, None };
// The lfm3a audio encoder loads only from the shared GGUF mmproj; None for
// a model without it (every family but LFM3-Omni).
enum class AudioSource : uint8_t { Gguf, None };
// The lfm3a_gen speech decoder also loads only from the shared GGUF
// mmproj (its a.gen.* sections); None for a model without speech output.
enum class SpeechSource : uint8_t { Gguf, None };
// Which tower a vision-capable model serves: the Qwen3.5 tower (its
// vision_config's projector_type "qwen3vl_merger", or absent on records
// that predate the field) or LFM3-VL ("lfm3vl", GGUF mmproj only).
enum class VisionTower : uint8_t { QwenVl, Lfm3Vl };

// Package metadata validated before weight buffers are loaded. The engine
// consumes capabilities; model loading consumes the concrete layouts.
struct ModelDescriptor final {
  std::string name;
  // The registered family's name (ModelRegistry's registrations replaced
  // kModelFamilies; "" when the package manifest predates family names):
  // model-scoped measured tuning rows match on it.
  std::string family;
  TargetLayout target;
  DFlashDraftLayout draft;
  ops::VisionLayout vision;
  // The LFM3-VL tower's geometry when visionTower is Lfm3Vl; vision stays
  // the Qwen tower's (zeroed but for its projection width otherwise).
  ops::LfmVisionLayout lfmVision;
  // The lfm3a audio encoder's geometry when the record's config carries an
  // audio_config; zeroed otherwise.
  ops::LfmAudioLayout lfmAudio;
  VisionTower visionTower = VisionTower::QwenVl;
  ModelCapabilities capabilities;
  kv::Layout targetKvLayout;
  CompositeStateLayout stateLayout;
  // Runtime policy the family maker chose; makeModelDescriptor derives the
  // tree-verify default from the draft kind, a maker may override it.
  ModelTuning tuning;
  // Container selection belongs to loading; runtime dispatch follows each weight.
  TargetSource targetSource = TargetSource::Packed;
  VisionSource visionSource = VisionSource::Packed;
  AudioSource audioSource = AudioSource::None;
  SpeechSource speechSource = SpeechSource::None;
  // Only the packed formats install/pack.py writes (richengine-packed-q4-
  // gemma4, -diffusiongemma) store the token table as 256-row tiles; the
  // splash-packed-q4* formats keep the checkpoint's flat quantized rows.
  bool packedTiledEmbedding = false;
  // The SHA-256 of the record that names the digest of every source file,
  // an assembly's model.json or a package's manifest.json, which the
  // installer verifies at every start (inspectModelPackage): what every
  // image is written from (WeightFileRecord).
  std::string sourceIdentity;

  // A source model's draft is a DFlash2 checkpoint; a packed package carries
  // its draft packed.
  [[nodiscard]] bool draftFromCheckpoint() const noexcept {
    return targetSource != TargetSource::Packed;
  }

  // A model installed with --language-only has no vision tower: it loads no
  // vision weights and serves no image requests.
  [[nodiscard]] bool hasVision() const noexcept {
    return visionSource != VisionSource::None;
  }
  // An omni record whose config declares the audio tower loads its encoder
  // weights and serves audio requests.
  [[nodiscard]] bool hasAudio() const noexcept {
    return audioSource != AudioSource::None;
  }
  // An omni record whose config declares the speech tower loads the
  // speech decoder's weights and emits audio while it generates.
  [[nodiscard]] bool hasSpeech() const noexcept {
    return speechSource != SpeechSource::None;
  }
  [[nodiscard]] bool valid() const noexcept;
};

// Derives the capabilities and cache layouts the engine consumes from the
// concrete target and draft layouts.
[[nodiscard]] ModelDescriptor makeModelDescriptor(std::string name,
                                                  TargetLayout target,
                                                  DFlashDraftLayout draft,
                                                  ops::VisionLayout vision);
[[nodiscard]] ModelDescriptor
inspectModelPackage(const std::filesystem::path &root);

} // namespace richengine::model

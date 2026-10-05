#pragma once

#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

struct DFlashDraftLayout;

namespace affine {
struct PlannedCheckpoint;
}

// A DFlash2 checkpoint as its repository releases it, config.json and BF16
// safetensors -> the packed draft files of a Splash package (DFlashDraft.cpp):
// every projection quantized to affine Q4 as those drafts were, every other
// tensor copied as stored. The checkpoint is planned once; each file is
// written into memory when it is opened.
class DraftCheckpointLoader final {
public:
  DraftCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout);
  ~DraftCheckpointLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, then model.bin
};

// Every planned file of a layout, its sections at their offsets: the layers,
// then model.bin.
[[nodiscard]] std::vector<affine::Image> draftCheckpointImages(const DFlashDraftLayout &layout);

// A plain transformer DFlash checkpoint ("DFlashDraftModel"), config.json
// and BF16 safetensors -> the packed files PlainDraft.cpp reads: every
// projection quantized to affine Q4, norms copied as stored. No
// convolutions, selector projection or codebooks exist to write.
class PlainDraftCheckpointLoader final {
public:
  PlainDraftCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                             const DFlashDraftLayout &layout);
  ~PlainDraftCheckpointLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, then model.bin
};

[[nodiscard]] std::vector<affine::Image> plainDraftCheckpointImages(const DFlashDraftLayout &layout);

// A DSpark checkpoint ("Qwen3DSparkModel"/"Lfm2DSparkDraftModel"),
// config.json and BF16 safetensors -> the packed files DSparkDraft.cpp
// reads: the plain draft's quantized layer sections, then model.bin with the
// fc projection, the norms, the Markov head's two vocabulary tables copied
// as stored, and the confidence head (loaded, unscored).
class DSparkCheckpointLoader final {
public:
  DSparkCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                         const DFlashDraftLayout &layout);
  ~DSparkCheckpointLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, then model.bin
};

[[nodiscard]] std::vector<affine::Image> dsparkDraftCheckpointImages(const DFlashDraftLayout &layout);

} // namespace splash::model

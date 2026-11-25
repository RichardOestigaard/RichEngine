#pragma once

#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace richengine::model {

struct DFlashDraftLayout;

namespace affine {
struct PlannedCheckpoint;
}

// A DFlash2 checkpoint as its repository releases it, config.json and BF16
// safetensors -> the packed draft files of a RichEngine package (DFlashDraft.cpp):
// every projection quantized to affine Q4 as those drafts were, every other
// tensor copied as stored. The checkpoint is planned once; each file is
// written into memory when it is opened.
class DFlash2CheckpointLoader final {
public:
  DFlash2CheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                        const DFlashDraftLayout &layout);
  ~DFlash2CheckpointLoader();
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
// and BF16 safetensors -> the packed files DFlashV1Draft.cpp reads: every
// projection quantized to affine Q4, norms copied as stored. No
// convolutions, selector projection or codebooks exist to write.
class DFlashV1DraftCheckpointLoader final {
public:
  DFlashV1DraftCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                             const DFlashDraftLayout &layout);
  ~DFlashV1DraftCheckpointLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, then model.bin
};

[[nodiscard]] std::vector<affine::Image> dflashV1DraftCheckpointImages(const DFlashDraftLayout &layout);

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
  [[nodiscard]] WeightImages &images() noexcept { return images_; }

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, then model.bin
};

[[nodiscard]] std::vector<affine::Image> dsparkDraftCheckpointImages(const DFlashDraftLayout &layout);

// A DSpline checkpoint ("DFlashDraftModel" with a selected_prefix_bridge),
// config.json and BF16 safetensors -> the packed files DSplineDraft.cpp
// reads: the plain draft's quantized layer sections, then model.bin with
// the fc projection, the norms, the code bridge's bf16 tensors and the
// format-version scalars.
class DSplineCheckpointLoader final {
public:
  DSplineCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                          const DFlashDraftLayout &layout);
  ~DSplineCheckpointLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile model();

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, then model.bin
};

[[nodiscard]] std::vector<affine::Image> dsplineDraftCheckpointImages(const DFlashDraftLayout &layout);

// A NemotronHMtpModel checkpoint (dev/convert/gguf_mtp_to_safetensors.py),
// config.json and BF16 safetensors -> the single mtp.bin of a nemotron MTP
// package (NemotronMtpDraft.cpp): every projection an F32 float segment
// (hidden 2688 sits off every quantized tile), every norm and the MoE
// tensors bf16 copies the draft's own kernels read. One file: mtp() loads
// it.
class NemotronMtpCheckpointLoader final {
public:
  NemotronMtpCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                          const DFlashDraftLayout &layout);
  ~NemotronMtpCheckpointLoader();
  [[nodiscard]] WeightFile mtp();

private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // mtp.bin alone
};

[[nodiscard]] std::vector<affine::Image> nemotronMtpDraftCheckpointImages(const DFlashDraftLayout &layout);

} // namespace richengine::model

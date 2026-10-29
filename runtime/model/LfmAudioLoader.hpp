#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/LfmAudioEnc.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace richengine::model {

// Plans the conversion of an LFM3-Omni mmproj's audio encoder sections
// (a.conv1d.*, a.pre_encode, a.blk.*, a.post_ln, mm.a.mlp.* of
// vision/mmproj.gguf) into the packed BF16 lfm3a weight image. The same
// file carries the vision sections LfmVisionLoader binds (v.*, mm.*) and
// the speech decoder's (a.gen.*), which belongs to the speech agent; this
// loader binds only the encoder tensors it names.
class LfmAudioLoader final {
public:
  LfmAudioLoader(const std::filesystem::path &directory,
                 const ops::LfmAudioLayout &layout);
  ~LfmAudioLoader();
  LfmAudioLoader(const LfmAudioLoader &) = delete;
  LfmAudioLoader &operator=(const LfmAudioLoader &) = delete;

  [[nodiscard]] const ops::LfmAudioLayout &layout() const noexcept;
  // Per-block kind as the mmproj's tensors declare it: true at the conv
  // blocks' indices. The weight reader follows the same flags over the
  // image's sections.
  [[nodiscard]] const std::vector<bool> &convBlocks() const noexcept;
  // The image of the packed encoder the loader's source writes.
  [[nodiscard]] ImagePlan image() const;

private:
  struct Planned;
  std::shared_ptr<Planned> planned_;
};

// The byte size of the image layout's plan, for load budgeting.
[[nodiscard]] uint64_t lfmAudioImageBytes(const ops::LfmAudioLayout &layout);

// The structural checks every entry point performs on a declared layout:
// the fields the kernels' tile assumptions rely on.
void requireLfmAudioLayout(const ops::LfmAudioLayout &layout);

} // namespace richengine::model

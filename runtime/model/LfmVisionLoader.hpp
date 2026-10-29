#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/LfmVision.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace richengine::model {

// Plans the conversion of an lfm3vl mmproj (vision/mmproj.gguf) into the
// packed BF16 LFM3-VL weight image. The same file carries the model's audio
// sections (a.*, mm.a.*, a.gen.*); this loader binds only the v.* and mm.*
// tensors it names and leaves the rest for the audio loaders.
class LfmVisionLoader final {
public:
  LfmVisionLoader(const std::filesystem::path &directory,
                  const ops::LfmVisionLayout &layout);
  ~LfmVisionLoader();
  LfmVisionLoader(const LfmVisionLoader &) = delete;
  LfmVisionLoader &operator=(const LfmVisionLoader &) = delete;

  [[nodiscard]] const ops::LfmVisionLayout &layout() const noexcept;
  // Per-block kind as the mmproj's tensors declare it: true at the conv
  // blocks' indices. The weight reader follows the same flags over the
  // image's sections.
  [[nodiscard]] const std::vector<bool> &convBlocks() const noexcept;
  // The image of the packed tower the loader's source writes.
  [[nodiscard]] ImagePlan image() const;

private:
  struct Planned;
  std::shared_ptr<Planned> planned_;
};

// The byte size of the image layout's plan, for load budgeting.
[[nodiscard]] uint64_t lfmVisionImageBytes(const ops::LfmVisionLayout &layout);

// The structural checks every entry point performs on a declared layout:
// the fields the kernels' tile assumptions rely on.
void requireLfmVisionLayout(const ops::LfmVisionLayout &layout);

} // namespace richengine::model

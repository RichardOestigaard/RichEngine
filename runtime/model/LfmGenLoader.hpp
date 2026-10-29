#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace richengine::model {

// Plans the conversion of the speech-generation sections (a.gen.code.* +
// a.gen.wav.*) of an lfm3a mmproj into the packed BF16 image
// audio/lfm-gen.bin. The same file carries the vision (v.*, mm.*) and
// audio-input (a.*, mm.a.*) sections their loaders own.
class LfmGenLoader final {
public:
  explicit LfmGenLoader(const std::filesystem::path &directory);
  ~LfmGenLoader();
  LfmGenLoader(const LfmGenLoader &) = delete;
  LfmGenLoader &operator=(const LfmGenLoader &) = delete;

  [[nodiscard]] ImagePlan image() const;
  // The image's per-section offsets in bf16 elements, in the order
  // ops::decodeLfmSpeechWeights consumes them (its take() sequence).
  [[nodiscard]] std::vector<uint64_t> sectionOffsets() const;
  // Per-wav-block kind as the mmproj's tensors declare it: true at the
  // conv blocks' indices (0,1,3,5,7); the weight reader follows it over
  // the image's sections.
  [[nodiscard]] const std::vector<bool> &wavConvBlocks() const noexcept;

private:
  struct Planned;
  std::shared_ptr<Planned> planned_;
};

[[nodiscard]] uint64_t lfmGenImageBytes();

} // namespace richengine::model

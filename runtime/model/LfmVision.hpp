#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/LfmVision.hpp"

#include <cstdint>
#include <vector>

namespace richengine::model {

class LfmVisionLoader;

// The packed LFM3-VL tower's weights bound for the ops encoder, mirroring
// QwenVisionWeights for the Qwen tower.
struct LfmVisionWeights final {
  ops::LfmVisionWeights tensors;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

// The image the loader's mmproj writes, read into encoder weights.
[[nodiscard]] LfmVisionWeights
loadLfmVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                     const LfmVisionLoader &source);

} // namespace richengine::model

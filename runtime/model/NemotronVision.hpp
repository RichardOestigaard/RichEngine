#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/NemotronVision.hpp"

#include <cstdint>
#include <vector>

namespace richengine::model {

class NemotronVisionLoader;

// The packed Nemotron Omni tower's weights bound for the ops encoder,
// mirroring LfmVisionWeights for the LFM tower.
struct NemotronVisionWeights final {
  ops::NemotronVisionWeights tensors;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

// The image the loader's mmproj writes, read into encoder weights.
[[nodiscard]] NemotronVisionWeights
loadNemotronVisionWeights(metal::MetalBackend &backend, WeightImages &images,
                          const NemotronVisionLoader &source);

} // namespace richengine::model

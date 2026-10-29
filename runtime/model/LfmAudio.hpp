#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/LfmAudioEnc.hpp"

#include <cstdint>
#include <vector>

namespace richengine::model {

class LfmAudioLoader;

// The packed lfm3a audio encoder's weights bound for the ops encoder,
// mirroring LfmVisionWeights for the vision tower.
struct LfmAudioWeights final {
  ops::LfmAudioWeights tensors;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

// The image the loader's mmproj writes, read into encoder weights.
[[nodiscard]] LfmAudioWeights
loadLfmAudioWeights(metal::MetalBackend &backend, WeightImages &images,
                    const LfmAudioLoader &source);

} // namespace richengine::model

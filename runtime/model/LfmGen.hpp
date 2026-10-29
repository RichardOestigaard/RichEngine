#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/LfmSpeechGen.hpp"

#include <cstdint>
#include <vector>

namespace richengine::model {

class LfmGenLoader;

// The lfm3a_gen speech decoder's weights: the a.gen.* sections decoded
// host-side to fp32, shared by every request's ops::LfmSpeechGen, plus
// the image's load accounting as the vision and audio towers carry it.
struct LfmSpeechModel final {
  ops::LfmSpeechWeights tensors;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

// The image the loader's mmproj writes, decoded into host weights. The
// packed bf16 image stays registered like every other weight image; the
// decoded copy is what the decode loop reads.
[[nodiscard]] LfmSpeechModel
loadLfmSpeechWeights(metal::MetalBackend &backend, WeightImages &images,
                     const LfmGenLoader &source);

} // namespace richengine::model

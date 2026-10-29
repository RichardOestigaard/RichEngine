#include "model/LfmGen.hpp"
#include "model/LfmGenLoader.hpp"
#include "model/WeightLayout.hpp"

#include <span>
#include <stdexcept>

namespace richengine::model {

LfmSpeechModel loadLfmSpeechWeights(metal::MetalBackend &backend,
                                    WeightImages &images,
                                    const LfmGenLoader &source) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  LfmSpeechModel result;

  const ImagePlan plan = source.image();
  WeightFile file = images.load(plan);
  // The whole image's bytes, which the host decode indexes by section
  // offset: the just-loaded component is the contents list's tail.
  std::span<const uint8_t> bytes;
  for (const WeightImages::Contents &contents : images.contents()) {
    if (contents.component == plan.component)
      bytes = contents.bytes;
  }
  if (bytes.size() != plan.bytes)
    throw WeightStoreError("lfm-gen image is not host visible");
  const std::span<const uint16_t> bf16(
      reinterpret_cast<const uint16_t *>(bytes.data()), bytes.size() / 2);
  const std::vector<uint64_t> offsets = source.sectionOffsets();
  try {
    result.tensors = ops::decodeLfmSpeechWeights(bf16, offsets);
  } catch (const std::invalid_argument &error) {
    throw WeightStoreError(std::string("lfm-gen image decode failed: ") +
                           error.what());
  }
  // The weight reader takes no section views; consuming the body as one
  // section keeps the whole-file accounting finish() checks.
  static_cast<void>(file.section(plan.bytes - kWeightFileAlignment,
                                 "lfm-gen-body"));
  file.finish();
  result.files.push_back(file.record());

  result.actualAllocatedBytes =
      metal::allocationDelta(allocationBaseline,
                             backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace richengine::model

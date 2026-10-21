#include "model/families/FamilyMakers.hpp"

#include "model/MiniCpmV47.hpp"

namespace richengine::model {

// The vision tower is a two-stage SigLIP merger this runtime's vision
// pipeline does not serve: text and draft only.
ModelDescriptor minicpmV47Descriptor(std::string name) {
  // The MiniCPM-V SigLIP tower is not served; only the descriptor's
  // vision/draft hidden-size contract needs the output width.
  ops::VisionLayout vision;
  vision.outputHiddenSize = MiniCpmV47MoeLayout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), MiniCpmV47MoeLayout{}, minicpmV47DraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

} // namespace richengine::model

#include "model/OqTarget.hpp"

#include "model/AffineTarget.hpp"
#include "model/SafetensorsCheckpoint.hpp"

namespace richengine::model {

OqTargetLoader::OqTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                               const TargetLayout &layout)
    : images_(images), directory_(directory), planned_(oqPlannedCheckpoint(directory, layout)) {}

OqTargetLoader::~OqTargetLoader() = default;
ImagePlan OqTargetLoader::imagePlan(size_t index, std::string_view directory) {
  const affine::Image &image = planned_->images.at(index);
  if (packed_ < 0)
    packed_ = affine::packedCacheUsable(directory_, planned_->images) ? 1 : 0;
  if (packed_)
    return packedImage(directory_ / image.name,
                       std::string(directory) + "/" + image.name, image.magic,
                       image.layer, image.type);
  ImagePlan result = affine::imagePlan(planned_, index, directory);
  const std::filesystem::path cached = directory_ / image.name;
  const std::vector<affine::Image> *plan_images = &planned_->images;
  result.write = [write = std::move(result.write), cached, plan_images,
                  directory = directory_](std::span<uint8_t> destination,
                                          const metal::MetalBuffer &buffer) {
    write(destination, buffer);
    affine::cachePackedImage(cached, destination);
    affine::refreshPackedStamp(directory, *plan_images);
  };
  return result;
}
WeightFile OqTargetLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 2) throw WeightStoreError("target layer is out of range");
  return images_.load(imagePlan(index, "target"));
}
WeightFile OqTargetLoader::head() {
  return images_.load(imagePlan(planned_->images.size() - 2, "target"));
}
WeightFile OqTargetLoader::embedding() {
  return images_.load(imagePlan(planned_->images.size() - 1, "target"));
}

bool oqCheckpoint(const std::filesystem::path &directory) {
  // An unreadable or missing configuration is not an oQ checkpoint; the
  // loader that runs instead reports the real error.
  try {
    return SafetensorsCheckpoint::hasMixedQuantization(directory);
  } catch (const WeightStoreError &) {
    return false;
  }
}

} // namespace richengine::model

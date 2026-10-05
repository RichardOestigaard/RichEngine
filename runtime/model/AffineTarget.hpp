#pragma once

#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace richengine::model {

struct Qwen3_8Layout;
struct Ornith9BLayout;
struct Qwen3_6MoeLayout;
struct DenseLayout;
struct GraniteLayout;
struct Lfm2Layout;
struct Lfm2MoeLayout;

namespace affine {
struct PlannedCheckpoint;
}

// Native MLX affine source -> the existing packed target ABI; neither this
// adapter nor the block-quantized one changes inference kernels. The
// checkpoint is planned once; each image is written into memory when it is
// opened.
class AffineTargetLoader final {
public:
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_8Layout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Ornith9BLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const DenseLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2Layout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2MoeLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const GraniteLayout &layout);
  ~AffineTargetLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
private:
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, head, embedding
};

// Every planned image of a layout, its sections at their offsets: the layers,
// the head, the embedding.
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_8Layout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Ornith9BLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_6MoeLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const DenseLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Lfm2Layout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Lfm2MoeLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const GraniteLayout &layout);

} // namespace richengine::model

#pragma once

// The oQ (oMLX mixed-precision) safetensors source -> the Block32 weight
// layout: the same images as the affine MLX source, every projection
// section planned against the checkpoint's per-module quantization
// (model/AffineTarget.cpp's oqPlannedCheckpoint) and written as Oq segments
// (model/OqWrite.cpp), which OqTargetFormat (model/TargetLoader.hpp) reads.
// oqCheckpoint is the config.json probe the MLX target branch runs first;
// a uniform Q4 group-64 affine checkpoint keeps the legacy path.

#include "model/AffinePlan.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>

namespace richengine::model {

struct Qwen3_8Layout;
struct Ornith9BLayout;
struct Qwen3_6MoeLayout;
struct MiniCpmV47MoeLayout;
struct DenseLayout;
struct GraniteLayout;
struct Lfm2Layout;
struct Lfm2MoeLayout;
struct Lfm2Moe24BLayout;

class OqTargetLoader final {
public:
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_8Layout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Ornith9BLayout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const MiniCpmV47MoeLayout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const DenseLayout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2Layout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2MoeLayout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2Moe24BLayout &layout);
  OqTargetLoader(WeightImages &images, const std::filesystem::path &directory, const GraniteLayout &layout);
  ~OqTargetLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
private:
  // The image of index in the plan: verbatim from the package's packed
  // cache when its stamp covers the sources and this plan, else written by
  // the oQ writer and cached for the next load.
  [[nodiscard]] ImagePlan imagePlan(size_t index, std::string_view directory);

  WeightImages &images_;
  std::filesystem::path directory_;                  // <package>/target
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, head, embedding
  // -1 unchecked, 1 read the cache, 0 write it.
  int8_t packed_ = -1;
};

// Whether directory's config.json declares a quantization the affine path
// cannot plan: a per-module entry, or a global one other than Q4 group-64
// affine (SafetensorsCheckpoint::hasMixedQuantization).
[[nodiscard]] bool oqCheckpoint(const std::filesystem::path &directory);

} // namespace richengine::model

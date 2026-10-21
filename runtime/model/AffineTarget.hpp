#pragma once

#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace richengine::model {

struct Qwen3_8Layout;
struct Ornith9BLayout;
struct Qwen3_6MoeLayout;
struct MiniCpmV47MoeLayout;
struct DenseLayout;
struct Gemma4MoeLayout;
struct GraniteLayout;
struct Lfm2Layout;
struct Lfm2MoeLayout;
struct Lfm2Moe24BLayout;

namespace affine {
struct PlannedCheckpoint;
}

// A module's spec as the checkpoint quantizes it, nullopt when the module
// is unquantized (a plain float weight the predicate skipped). An empty
// resolver plans today's affine images; the oQ one plans Oq sections.
using QuantResolver =
    std::function<std::optional<SafetensorsCheckpoint::QuantSpec>(std::string_view module)>;

// Native MLX affine source -> the existing packed target ABI; neither this
// adapter nor the block-quantized one changes inference kernels. The
// checkpoint is planned once; each image is written into memory when it is
// opened.
class AffineTargetLoader final {
public:
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_8Layout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Ornith9BLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const MiniCpmV47MoeLayout &layout);
  // preferBlockWeights plans the images' projections as AF4 block sections
  // (model/AffineToBlock.hpp) instead of affine tiles: the reader must be
  // AffineBlockTargetFormat (model/TargetLoader.hpp), which loadDenseWeights
  // picks under the same flag.
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const DenseLayout &layout,
                     bool preferBlockWeights = false);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2Layout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2MoeLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Lfm2Moe24BLayout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const GraniteLayout &layout);
  ~AffineTargetLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
private:
  // The image of index in the plan: verbatim from the package's packed
  // cache when its stamp covers the sources and this plan, else written by
  // the affine writer and cached for the next load.
  [[nodiscard]] ImagePlan imagePlan(size_t index, std::string_view directory);

  WeightImages &images_;
  std::filesystem::path directory_;                  // <package>/target
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // layers, head, embedding
  // -1 unchecked, 1 read the cache, 0 write it.
  int8_t packed_ = -1;
};

// Every planned image of a layout, its sections at their offsets: the layers,
// the head, the embedding.
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_8Layout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Ornith9BLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_6MoeLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const MiniCpmV47MoeLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const DenseLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Lfm2Layout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Lfm2MoeLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Lfm2Moe24BLayout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const GraniteLayout &layout);

// Gemma 4 has no MLX image plan: the packed files only. The overload keeps
// the target variant's generic dispatch compilable.
[[nodiscard]] inline std::vector<affine::Image>
affineTargetImages(const Gemma4MoeLayout &) {
  throw std::invalid_argument("Gemma 4 targets load from packed files only");
}

// The oQ (oMLX mixed-precision) plan of a checkpoint: the layout's images
// with every projection an Oq section (model/OqWrite.cpp) planned against
// the checkpoint's per-module specs — modules it left unquantized emit F32
// float segments. OqTargetLoader loads through it; modelWeightBytes budgets
// with oqTargetImageBytes.
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const Qwen3_8Layout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const Ornith9BLayout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const MiniCpmV47MoeLayout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const DenseLayout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const Lfm2Layout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const Lfm2MoeLayout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const Lfm2Moe24BLayout &layout);
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const GraniteLayout &layout);
[[nodiscard]] inline std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &, const Gemma4MoeLayout &) {
  throw std::invalid_argument("Gemma 4 targets load from packed files only");
}

// The total bytes of a layout's oQ images, planned from the checkpoint at
// directory (its config.json and index only — nothing is written).
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Qwen3_8Layout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Ornith9BLayout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const MiniCpmV47MoeLayout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const DenseLayout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Lfm2Layout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Lfm2MoeLayout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Lfm2Moe24BLayout &layout);
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const GraniteLayout &layout);
[[nodiscard]] inline uint64_t oqTargetImageBytes(const std::filesystem::path &, const Gemma4MoeLayout &) {
  throw std::invalid_argument("Gemma 4 targets load from packed files only");
}

} // namespace richengine::model

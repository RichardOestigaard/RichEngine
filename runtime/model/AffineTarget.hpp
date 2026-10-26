#pragma once

#include "model/AffinePreparation.hpp"
#include "model/TargetLayout.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string_view>
#include <stdexcept>
#include <utility>
#include <vector>

namespace richengine::model {

// A module's spec as the checkpoint quantizes it, nullopt when the module
// is unquantized (a plain float weight the predicate skipped). An empty
// resolver plans today's affine images; the oQ one plans Oq sections.
using QuantResolver =
    std::function<std::optional<SafetensorsCheckpoint::QuantSpec>(std::string_view module)>;

namespace affine {
struct PlannedCheckpoint;

// The section planners the families' registered AffineArchHooks share with
// the default GDN schema (AffineTarget.cpp): projection's stacked parts,
// the token table's Oq section, MiniCPM-V's fused expert window and the
// MoE FFN block the LFM2-MoE 8B and 24B images share.
void projection(Image &image,
                std::initializer_list<std::pair<std::string, uint32_t>> parts,
                uint32_t rows, uint32_t columns, uint32_t bits = 4,
                uint32_t experts = 1, const QuantResolver &resolve = {},
                bool forceFloat = false, bool preferAffine = false);
void embeddingSection(Image &image, const std::string &prefix, uint32_t rows,
                      uint32_t columns, const QuantResolver &resolve);
void fusedWindow(Image &image, const std::string &name, uint32_t rows,
                 uint32_t columns, uint32_t bits, uint32_t experts,
                 uint32_t stride, uint32_t offset,
                 const QuantResolver &resolve);
void lfm2MoeFfn(Image &result, const TargetLayout &layout,
                const std::string &ffn, const std::string &gate,
                const std::string &up, const std::string &down,
                const QuantResolver &resolve);
} // namespace affine

// A target arch's affine source behavior (ModelRegistry's
// TargetArchRegistration::affineHooks): the image plan its checkpoint's
// tensor names follow and the config.json fields it validates. An arch
// without hooks plans the GDN-hybrid schema and validates nothing.
struct AffineArchHooks final {
  // The layout's images against the opened checkpoint. `source` is null
  // under affineTargetImages' byte budgeting (no checkpoint is opened), so
  // a hook that derives tensor names from it plans its default schema then.
  std::vector<affine::Image> (*planImages)(const SafetensorsCheckpoint *source,
                                           const TargetLayout &layout,
                                           const QuantResolver &resolve) = nullptr;
  void (*validateConfiguration)(const SafetensorsCheckpoint &source,
                                const TargetLayout &layout) = nullptr;
};

// Native MLX affine source -> the existing packed target ABI; neither this
// adapter nor the block-quantized one changes inference kernels. The
// checkpoint is planned once; each image is written into memory when it is
// opened.
class AffineTargetLoader final {
public:
  // preferBlockWeights plans the images' projections as AF4 block sections
  // (model/AffineToBlock.hpp) instead of affine tiles: the reader must be
  // AffineBlockTargetFormat (model/TargetLoader.hpp), which loadDenseWeights
  // picks under the same flag.
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                     const TargetLayout &layout, bool preferBlockWeights = false);
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
// the head, the embedding. A packed-only layout (Gemma 4, DiffusionGemma)
// has no MLX image plan, and Granite's sources are packed or GGUF: both
// throw.
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const TargetLayout &layout);

// The oQ (oMLX mixed-precision) plan of a checkpoint: the layout's images
// with every projection an Oq section (model/OqWrite.cpp) planned against
// the checkpoint's per-module specs — modules it left unquantized emit F32
// float segments. OqTargetLoader loads through it; modelWeightBytes budgets
// with oqTargetImageBytes. Throws as affineTargetImages does.
[[nodiscard]] std::shared_ptr<affine::PlannedCheckpoint>
oqPlannedCheckpoint(const std::filesystem::path &directory, const TargetLayout &layout);

// The total bytes of a layout's oQ images, planned from the checkpoint at
// directory (its config.json and index only — nothing is written).
[[nodiscard]] uint64_t oqTargetImageBytes(const std::filesystem::path &directory,
                                          const TargetLayout &layout);

} // namespace richengine::model

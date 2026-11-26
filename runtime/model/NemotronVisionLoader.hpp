#pragma once

#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/NemotronVision.hpp"

#include <filesystem>
#include <memory>

namespace richengine::model {

// Plans the conversion of a nemotron_omni mmproj (vision/mmproj.gguf) into
// the packed BF16 Nemotron vision weight image.
class NemotronVisionLoader final {
public:
  NemotronVisionLoader(const std::filesystem::path &directory,
                       const ops::NemotronVisionLayout &layout);
  ~NemotronVisionLoader();
  NemotronVisionLoader(const NemotronVisionLoader &) = delete;
  NemotronVisionLoader &operator=(const NemotronVisionLoader &) = delete;

  [[nodiscard]] const ops::NemotronVisionLayout &layout() const noexcept;
  // The image of the packed tower the loader's source writes.
  [[nodiscard]] ImagePlan image() const;

private:
  struct Planned;
  std::shared_ptr<Planned> planned_;
};

// The byte size of the image layout's plan, for load budgeting.
[[nodiscard]] uint64_t nemotronVisionImageBytes(const ops::NemotronVisionLayout &layout);

// The structural checks every entry point performs on a declared layout:
// the fields the kernels' tile assumptions rely on.
void requireNemotronVisionLayout(const ops::NemotronVisionLayout &layout);

} // namespace richengine::model

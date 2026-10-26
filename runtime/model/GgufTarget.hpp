#pragma once

// Source adapter for a target GGUF: its images are written into memory and
// read like packaged weights.

#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/TargetLayout.hpp"
#include "model/WeightImages.hpp"

namespace richengine::model {

// The single .gguf in a target directory (shards are not supported).
[[nodiscard]] std::filesystem::path findTargetGguf(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // Plans every image from the GGUF's metadata once.
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &path,
                   const gguf::TargetGeometry &geometry);
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

private:
  // The GGUF and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &path) : source(path) {}
    WeightSource source;
    std::vector<gguf::Image> images; // layers, head, embedding
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  std::optional<GgufRotation> rotation_;
};

// The GGUF geometry of a target: the dense ("llama"), Granite ("granite"),
// LFM2 ("lfm2") and LFM2-MoE ("lfm2moe") targets' metadata schema, or the
// hybrid qwen35 one. A packed-only layout — Gemma 4, DiffusionGemma — has
// no GGUF image plan and throws: the packed files only.
[[nodiscard]] gguf::TargetGeometry ggufTargetGeometry(const TargetLayout &layout);

} // namespace richengine::model

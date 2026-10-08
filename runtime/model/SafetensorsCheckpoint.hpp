#pragma once

#include "model/WeightSource.hpp"

#include <filesystem>
#include <memory>
#include <string_view>

namespace richengine::model {

// A checkpoint's quantization configuration and safetensors index. Opening
// parses only metadata; the images read tensor data in bounded slices,
// without loading the MLX runtime or allocating tensors. The rest of its
// config.json is the model's, which inspectModelPackage checks.
class SafetensorsCheckpoint final {
public:
  explicit SafetensorsCheckpoint(const std::filesystem::path &directory);
  ~SafetensorsCheckpoint();
  [[nodiscard]] const SourceTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  void requireQuantization(std::string_view projection, uint32_t bits) const;
  void requireConfigNumber(std::string_view key, double expected) const;
  // Passes when `key` is absent or equals `expected`; for fields such as
  // `attention_bias` whose configs omit the default value.
  void requireConfigNumberOrAbsent(std::string_view key, double expected) const;
  // The source may declare a smaller value than `maximum` (e.g. a context
  // budget narrower than the layout's); larger values are rejected.
  void requireConfigNumberAtMost(std::string_view key, double maximum) const;
  // `legacyKey`, when given, names the field in configurations that predate `key`.
  void requireConfigString(std::string_view key, std::string_view expected,
                           std::string_view legacyKey = {}) const;
  void requireLayerTypes(uint32_t layers, uint32_t fullAttentionPeriod) const;
  // The layer_types array of a mask-typed hybrid (LFM2): `attention` at the
  // mask's bits, `other` at the rest.
  void requireLayerTypeMask(uint32_t layers, uint64_t attentionMask,
                          std::string_view attention, std::string_view other) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace richengine::model

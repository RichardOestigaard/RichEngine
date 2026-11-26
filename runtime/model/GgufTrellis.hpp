#pragma once

// NariQ trellis metadata (nariq/FORMAT.md). This implementation accepts the
// identity/block-Hadamard subset used by the published Qwen3.8 artifacts.

#include "model/GgufFile.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace richengine::model {

struct TrellisDescriptor final {
  int32_t rowsKind = 0;
  int32_t rowsBlock = 0;
  int32_t rowsShards = 0;
  int32_t colsKind = 0;
  int32_t colsBlock = 0;
  int32_t colsShards = 0;
  int32_t parts = 0;
  int32_t part0 = 0;
};

struct TrellisTensorInfo final {
  const GgufTensor *weight = nullptr;
  const GgufTensor *s16 = nullptr;
  const GgufTensor *pre = nullptr;
  const GgufTensor *post = nullptr;
  const GgufTensor *pin = nullptr;
  const GgufTensor *pout = nullptr;
  TrellisDescriptor descriptor;

  [[nodiscard]] uint32_t bits() const noexcept;
  [[nodiscard]] uint32_t inputSize() const noexcept;
  [[nodiscard]] uint32_t outputSize() const noexcept;
};

[[nodiscard]] bool ggufHasTrellis(const GgufFile &file);
[[nodiscard]] uint32_t ggufTrellisBandCount(const GgufFile &file, std::string_view key);
[[nodiscard]] TrellisDescriptor readTrellisDescriptor(const GgufFile &file, const GgufTensor &weight);
[[nodiscard]] TrellisTensorInfo readTrellisTensor(const GgufFile &file, std::string_view name);

} // namespace richengine::model

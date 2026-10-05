#pragma once

#include "metal/MetalBackend.hpp"

#include <memory>
#include <string>

namespace richengine::test {

// The memory statistics the stand-in backend below reports.
[[nodiscard]] inline metal::MetalMemoryStats &metalStatistics() noexcept {
  static metal::MetalMemoryStats statistics;
  return statistics;
}

} // namespace richengine::test

// The memory governor and the control pass read nothing from the backend but
// its memory statistics, so this stand-in lets a test set them without a GPU.
// It defines the backend's members: one translation unit of a test that links
// no Metal backend includes it.
namespace richengine::metal {

struct MetalBackend::Impl {};
MetalBackend::MetalBackend(std::string, double)
    : impl_(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() = default;
MetalMemoryStats MetalBackend::memoryStats() const noexcept {
  return test::metalStatistics();
}
MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
  return test::metalStatistics();
}

} // namespace richengine::metal

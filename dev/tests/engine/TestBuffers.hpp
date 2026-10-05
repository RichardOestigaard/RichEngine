#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>

namespace richengine::test {

// A shared, unlabeled buffer of `bytes` bytes, the kind tests allocate.
inline metal::MetalBuffer sharedBuffer(metal::MetalBackend &backend, uint64_t bytes) {
  return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, {});
}

} // namespace richengine::test

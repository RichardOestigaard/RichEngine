#pragma once

#include "qwen4/metal/MetalBackend.hpp"
#include "qwen4/model/GgufImage.hpp"
#include "qwen4/model/WeightSource.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace splash::model {

// The Metal staging of the rows of one repack step, whatever the tensor,
// layer or expert count.
inline constexpr uint64_t kGgufRepackStagingBytes = 32 << 20;

// Writes every byte of a planned image into its buffer: the header and
// descriptors, the copied rows and the planes the GPU repacks into it, each
// read from the source its rows name (a GGUF's, or an MLX checkpoint's
// tensors).
void writeGgufImage(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Image &plan);

// Rows [first, first + count) of a repack (its sources' rows in order), whole
// and back to back into `to`, count * ggufRowBytes(format, columns) bytes:
// the native rows the GPU repacks into those rows' planes. A streamed
// expert's rows are read with it (model/ExpertStore.hpp), from any thread.
void readRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count, uint8_t *to);

// Where rows [first, first + count) of a repack are in a file, when one of
// its sources stores them as they are and in order, so that they are read
// with the file's bytes alone: the file and the offset of the first row from
// its start. None when they are built (a quantized safetensors tensor's,
// NVFP4 rows, a quantized float tensor) or permuted, or span two sources.
struct StoredRows final {
  const WeightSource *file = nullptr;
  uint64_t offset = 0;
};
[[nodiscard]] std::optional<StoredRows> storedRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count);

// A plane of a repack's rows as its file stores them: where its row 0 is,
// from the file's start, and its bytes a row; none (rowBytes 0) for a
// format without that plane.
struct PlaneRows final {
  const WeightSource *file = nullptr;
  uint64_t offset = 0;
  uint64_t rowBytes = 0;
};
// The planes of a repack whose one source is an MLX affine or mxfp4 tensor
// of all its rows in order, from which the GPU builds the native blocks it
// repacks (moe_expert_load): an affine tensor's codes, scales and biases, an
// mxfp4 tensor's codes and scales. None for any other repack, whose native
// rows readRepackRows builds.
[[nodiscard]] std::optional<std::array<PlaneRows, 3>> storedPlanes(const gguf::Repack &repack);

} // namespace splash::model

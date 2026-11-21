#pragma once

#include <cstddef>
#include <cstdint>

namespace richengine::model::statecodec {

// A cached state's int8 coding of a GDN recurrent layer (StateLayout's
// codecBlock): `rows` rows of `columns` float values, each value an int8
// code, then one bf16 scale per row and block — the block's largest
// magnitude over 127. A block that holds a non-finite value records a NaN
// scale, which dequantize restores as NaN values rather than finite
// garbage. The host codes on shared buffers; splash's GPU kernels make the
// same layout.

inline constexpr uint32_t kBlock = 32;
// The int8 scale of 1/127 a code multiplies.
inline constexpr float kCodeScale = 1.0f / 127.0f;

// codes rows*columns int8 then rows*(columns/kBlock) bf16 scales, in that
// order; `columns` is a whole number of blocks.
void quantize(const float *source, std::byte *destination, uint64_t rows,
              uint32_t columns);
void dequantize(const std::byte *source, float *destination, uint64_t rows,
                uint32_t columns);

} // namespace richengine::model::statecodec

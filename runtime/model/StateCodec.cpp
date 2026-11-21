#include "model/StateCodec.hpp"

#include <cmath>
#include <cstring>

namespace richengine::model::statecodec {
namespace {

// bf16 keeps the upper half of the f32 bits, rounded to nearest even.
uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

float fromBf16(uint16_t bits) {
  const uint32_t expanded = uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

constexpr uint16_t kBf16NaN = 0x7FC0;

} // namespace

void quantize(const float *source, std::byte *destination, uint64_t rows,
              uint32_t columns) {
  const uint32_t blocks = columns / kBlock;
  int8_t *codes = reinterpret_cast<int8_t *>(destination);
  uint16_t *scales = reinterpret_cast<uint16_t *>(codes + rows * columns);
  for (uint64_t row = 0; row < rows; ++row) {
    const float *values = source + row * columns;
    for (uint32_t block = 0; block < blocks; ++block) {
      const float *head = values + block * kBlock;
      float magnitude = 0.0f;
      bool finite = true;
      for (uint32_t index = 0; index < kBlock; ++index) {
        const float absolute = std::fabs(head[index]);
        finite &= std::isfinite(absolute);
        if (absolute > magnitude)
          magnitude = absolute;
      }
      uint16_t *scale = scales + row * blocks + block;
      int8_t *out = codes + row * columns + block * kBlock;
      // A block that is not finite writes a NaN scale; dequantize restores
      // the block's values as NaN rather than finite garbage.
      if (!finite || !std::isfinite(magnitude)) {
        *scale = kBf16NaN;
        std::memset(out, 0, kBlock);
        continue;
      }
      const float unit = fromBf16(toBf16(magnitude * kCodeScale));
      *scale = toBf16(magnitude * kCodeScale);
      if (unit == 0.0f) {
        std::memset(out, 0, kBlock);
        continue;
      }
      for (uint32_t index = 0; index < kBlock; ++index)
        out[index] = static_cast<int8_t>(
            std::lrint(head[index] / unit));
    }
  }
}

void dequantize(const std::byte *source, float *destination, uint64_t rows,
                uint32_t columns) {
  const uint32_t blocks = columns / kBlock;
  const int8_t *codes = reinterpret_cast<const int8_t *>(source);
  const uint16_t *scales =
      reinterpret_cast<const uint16_t *>(codes + rows * columns);
  for (uint64_t row = 0; row < rows; ++row) {
    float *values = destination + row * columns;
    for (uint32_t block = 0; block < blocks; ++block) {
      const float unit = fromBf16(scales[row * blocks + block]);
      const int8_t *in = codes + row * columns + block * kBlock;
      float *out = values + block * kBlock;
      if (std::isnan(unit)) {
        for (uint32_t index = 0; index < kBlock; ++index)
          out[index] = std::nanf("");
        continue;
      }
      for (uint32_t index = 0; index < kBlock; ++index)
        out[index] = float(in[index]) * unit;
    }
  }
}

} // namespace richengine::model::statecodec

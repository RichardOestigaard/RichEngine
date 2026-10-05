#pragma once

// CPU oracle for the FP8 E4M3 KV page tier: same Page32 geometry and
// per-(token, head) fp32 scales as the INT8 format, one e4m3 byte per
// element. The encoder matches the Metal pack instruction (round to nearest
// even, saturate to 448), so Metal quantization is checked bit for bit.

#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace splash::kv {

// E4M3: sign, four exponent bits (bias 7), three mantissa bits; the largest
// finite magnitude is 448 (exponent field 15, mantissa 6; mantissa 7 is NaN).
inline constexpr float kFp8E4m3Maximum = 448.0F;

inline uint8_t floatToFp8E4m3(float value) {
  const uint8_t sign = std::signbit(value) ? uint8_t{0x80} : uint8_t{0};
  const float a = std::fabs(value);
  // Saturating pack: anything rounding to 480 or more clamps to 448.
  if (!std::isfinite(a))
    return static_cast<uint8_t>(sign | 0x7F);
  if (a >= 512.0F)
    return static_cast<uint8_t>(sign | 0x7E);
  if (a < 0x1p-6F) {
    // Subnormal codes m = round(a / 2^-9); m == 8 is the smallest normal.
    const long code = std::lrintf(a * 512.0F);
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(std::min<long>(code, 8)));
  }
  // Normal: mantissa field is rne(a / 2^e * 8) - 8 at exponent field e + 7,
  // with a carry past 16 advancing the exponent.
  int exponent = std::ilogb(a);
  long mantissa = std::lrintf(std::ldexp(a, 3 - exponent));
  if (mantissa == 16) {
    ++exponent;
    mantissa = 8;
  }
  if (exponent > 8 || (exponent == 8 && mantissa - 8 > 6))
    return static_cast<uint8_t>(sign | 0x7E);
  return static_cast<uint8_t>(
      sign | static_cast<uint8_t>(((exponent + 7) << 3) | (mantissa - 8)));
}

inline float fp8E4m3ToFloat(uint8_t code) {
  const float sign = (code & 0x80) ? -1.0F : 1.0F;
  const int exponent = (code >> 3) & 0xF;
  const int mantissa = code & 0x7;
  if (exponent == 0)
    return sign * std::ldexp(float(mantissa), -9);
  if (exponent == 15 && mantissa == 7)
    return std::numeric_limits<float>::quiet_NaN();
  return sign * std::ldexp(1.0F + float(mantissa) / 8.0F, exponent - 7);
}

// The fp16 value the production tile stages each bf16 query element as for
// its half x fp8 matmul; the CPU reference scores the same operands.
inline float queryStage(float value) {
  return static_cast<float>(static_cast<_Float16>(value));
}

// This oracle targets the Qwen3.8 kernel specialization.
inline constexpr Layout kFp8OracleLayout{16, 4, 256, Format::Float8E4M3};
inline constexpr uint32_t kFp8KvHeads = kFp8OracleLayout.kvHeads;
inline constexpr uint32_t kFp8HeadDimension = kFp8OracleLayout.headDimension;
inline constexpr uint64_t kFp8ElementsPerLayerPage =
    kFp8OracleLayout.elementsPerLayerPage();
inline constexpr uint64_t kFp8ScalesPerTensorLayerPage =
    kFp8OracleLayout.scalesPerTensorLayerPage();
inline constexpr uint64_t kFp8DataBytesPerLayerPage =
    kFp8OracleLayout.dataBytesPerLayerPage();
inline constexpr uint64_t kFp8ScaleBytesPerLayerPage =
    kFp8OracleLayout.scaleBytesPerLayerPage();
inline constexpr uint64_t kFp8BytesPerLayerPage =
    kFp8OracleLayout.bytesPerLayerPage();

struct Fp8LayerPage final {
  std::array<uint8_t, kFp8ElementsPerLayerPage> keys{};
  std::array<float, kFp8ScalesPerTensorLayerPage> keyScales{};
  std::array<uint8_t, kFp8ElementsPerLayerPage> values{};
  std::array<float, kFp8ScalesPerTensorLayerPage> valueScales{};
};

static_assert(sizeof(Fp8LayerPage) == kFp8BytesPerLayerPage);
static_assert(std::is_standard_layout_v<Fp8LayerPage>);

struct Fp8KVMetalPageParams final {
  uint32_t sourcePage = 0;
  uint32_t destinationPage = 0;
  uint32_t validTokens = kPageTokens;
  uint32_t reserved = 0;
};

static_assert(sizeof(Fp8KVMetalPageParams) == 16);
static_assert(std::is_standard_layout_v<Fp8KVMetalPageParams>);

constexpr uint64_t fp8LogicalIndex(uint32_t token, uint32_t head,
                                   uint32_t dimension) {
  return (uint64_t{token} * kFp8KvHeads + head) * kFp8HeadDimension + dimension;
}

namespace fp8_reference_detail {

constexpr uint64_t logicalElements(uint32_t validTokens) {
  return uint64_t{validTokens} * kFp8KvHeads * kFp8HeadDimension;
}

inline void validateInput(std::span<const float> values,
                          uint32_t validTokens, const char *name) {
  if (validTokens > kPageTokens)
    throw std::invalid_argument("FP8 KV valid token count exceeds one page");
  if (values.size() != logicalElements(validTokens))
    throw std::invalid_argument(std::string(name) +
                                " has the wrong logical page size");
  if (std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument(std::string(name) +
                                " contains a non-finite value");
}

inline void validateOutput(std::span<float> values, const char *name) {
  if (values.size() != kFp8ElementsPerLayerPage)
    throw std::invalid_argument(std::string(name) +
                                " has the wrong logical page size");
}

inline float storedScale(float maximum) {
  return maximum == 0.0F ? 0.0F : maximum / kFp8E4m3Maximum;
}

inline uint8_t quantize(float value, float maximum) {
  if (maximum == 0.0F)
    return 0;
  return floatToFp8E4m3(value * kFp8E4m3Maximum / maximum);
}

inline void checkElement(uint32_t head, uint32_t token, uint32_t dimension) {
  if (head >= kFp8KvHeads || token >= kPageTokens || dimension >= kFp8HeadDimension)
    throw std::out_of_range("FP8 KV element is outside the page");
}

} // namespace fp8_reference_detail

inline float dequantizeFp8Key(const Fp8LayerPage &source, uint32_t head,
                              uint32_t token, uint32_t dimension) {
  fp8_reference_detail::checkElement(head, token, dimension);
  return fp8E4m3ToFloat(
             source.keys[splash_kv_key_element(head, token, dimension)]) *
         source.keyScales[splash_kv_scale_element(head, token)];
}

inline float dequantizeFp8Value(const Fp8LayerPage &source, uint32_t head,
                                uint32_t token, uint32_t dimension) {
  fp8_reference_detail::checkElement(head, token, dimension);
  return fp8E4m3ToFloat(
             source.values[splash_kv_value_element(head, token, dimension)]) *
         source.valueScales[splash_kv_scale_element(head, token)];
}

inline void quantizeFp8LayerPage(std::span<const float> logicalKeys,
                                 std::span<const float> logicalValues,
                                 uint32_t validTokens,
                                 Fp8LayerPage &destination) {
  fp8_reference_detail::validateInput(logicalKeys, validTokens, "logical keys");
  fp8_reference_detail::validateInput(logicalValues, validTokens,
                                      "logical values");
  destination = {};
  for (uint32_t head = 0; head < kFp8KvHeads; ++head) {
    for (uint32_t token = 0; token < validTokens; ++token) {
      float keyMaximum = 0.0F;
      float valueMaximum = 0.0F;
      for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension) {
        keyMaximum = std::max(
            keyMaximum,
            std::abs(logicalKeys[fp8LogicalIndex(token, head, dimension)]));
        valueMaximum = std::max(
            valueMaximum,
            std::abs(logicalValues[fp8LogicalIndex(token, head, dimension)]));
      }
      destination.keyScales[splash_kv_scale_element(head, token)] =
          fp8_reference_detail::storedScale(keyMaximum);
      destination.valueScales[splash_kv_scale_element(head, token)] =
          fp8_reference_detail::storedScale(valueMaximum);
      for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension) {
        destination.keys[splash_kv_key_element(head, token, dimension)] =
            fp8_reference_detail::quantize(
                logicalKeys[fp8LogicalIndex(token, head, dimension)],
                keyMaximum);
        destination.values[splash_kv_value_element(head, token, dimension)] =
            fp8_reference_detail::quantize(
                logicalValues[fp8LogicalIndex(token, head, dimension)],
                valueMaximum);
      }
    }
  }
}

inline void dequantizeFp8LayerPage(const Fp8LayerPage &source,
                                   uint32_t validTokens,
                                   std::span<float> logicalKeys,
                                   std::span<float> logicalValues) {
  if (validTokens > kPageTokens)
    throw std::invalid_argument("FP8 KV valid token count exceeds one page");
  fp8_reference_detail::validateOutput(logicalKeys, "logical keys");
  fp8_reference_detail::validateOutput(logicalValues, "logical values");
  std::fill(logicalKeys.begin(), logicalKeys.end(), 0.0F);
  std::fill(logicalValues.begin(), logicalValues.end(), 0.0F);
  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kFp8KvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension) {
        const uint64_t output = fp8LogicalIndex(token, head, dimension);
        logicalKeys[output] = dequantizeFp8Key(source, head, token, dimension);
        logicalValues[output] =
            dequantizeFp8Value(source, head, token, dimension);
      }
    }
  }
}

} // namespace splash::kv

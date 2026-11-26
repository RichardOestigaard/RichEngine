#pragma once

// Host-side NariQ trellis helpers. Preparation uses the permutation helpers;
// engine tests use the decoder as the reference for the Metal kernels.

#include "model/GgufTrellis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <cstring>

namespace richengine::model::trellis {

inline int value(uint32_t state) {
  const uint32_t x = state * 0x83DCD12Du;
  return static_cast<int>(((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)) - 126;
}

inline int codedValue(const uint32_t *tile, uint32_t bits, uint32_t p) {
  const uint32_t total = 256 * bits;
  const uint32_t start = ((p + 1) * bits - 16 + total) % total;
  const uint32_t word = start >> 5, offset = start & 31, words = 8 * bits;
  const uint64_t pair = (uint64_t{tile[word]} << 32) | tile[(word + 1) % words];
  return value(static_cast<uint32_t>(pair >> (48 - offset)) & 0xffffu);
}

inline uint32_t position(uint32_t row, uint32_t column) {
  return 8 * (4 * (row % 8) + (column % 8) / 2) + 4 * (row / 8) + 2 * (column / 8) + column % 2;
}

inline void decodeRows(const uint32_t *codes, uint32_t bits, uint32_t columns, uint32_t firstRow,
                       uint32_t rows, const uint16_t *scales, float *out) {
  const uint32_t columnTiles = columns / 16, words = 8 * bits;
  for (uint32_t tileRow = firstRow / 16; tileRow < (firstRow + rows) / 16; ++tileRow) {
    for (uint32_t columnTile = 0; columnTile < columnTiles; ++columnTile) {
      const uint32_t *tile = codes + (size_t{tileRow} * columnTiles + columnTile) * words;
      float scale = 1.0f;
      if (scales) {
        __fp16 half;
        std::memcpy(&half, scales + columnTile, sizeof half);
        scale = static_cast<float>(half);
      }
      for (uint32_t r = 0; r < 16; ++r) {
        for (uint32_t c = 0; c < 16; ++c) {
          const uint32_t row = tileRow * 16 + r - firstRow;
          const uint32_t column = columnTile * 16 + c;
          out[size_t{row} * columns + column] = float(codedValue(tile, bits, position(r, c))) * scale;
        }
      }
    }
  }
}

inline void fwht(float *values, uint32_t block, float scale) {
  for (uint32_t h = 1; h < block; h *= 2) {
    for (uint32_t i = 0; i < block; i += 2 * h) {
      for (uint32_t j = i; j < i + h; ++j) {
        const float a = values[j], b = values[j + h];
        values[j] = a + b;
        values[j + h] = a - b;
      }
    }
  }
  for (uint32_t i = 0; i < block; ++i) values[i] *= scale;
}

inline void blockTransform(float *values, uint32_t length, uint32_t block) {
  const float scale = static_cast<float>(std::pow(static_cast<double>(block), -0.5));
  for (uint32_t first = 0; first < length; first += block) fwht(values + first, block, scale);
}

inline void applySide(float *values, uint32_t length, int32_t kind, int32_t block) {
  if (kind == 1) blockTransform(values, length, static_cast<uint32_t>(block));
}

inline std::vector<float> linearReference(const uint32_t *codes, uint32_t bits, uint32_t columns, uint32_t rows,
                                          const uint16_t *scales, const float *pre, const float *post,
                                          const int32_t *pin, const int32_t *pout, const TrellisDescriptor &descriptor,
                                          const float *input) {
  std::vector<float> weights(size_t{rows} * columns);
  decodeRows(codes, bits, columns, 0, rows, scales, weights.data());
  std::vector<float> x(columns);
  for (uint32_t j = 0; j < columns; ++j) x[j] = input[pin ? pin[j] : j] * pre[j];
  applySide(x.data(), columns, descriptor.colsKind, descriptor.colsBlock);
  std::vector<float> coded(rows, 0.0f);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t j = 0; j < columns; ++j) coded[row] += weights[size_t{row} * columns + j] * x[j];
  applySide(coded.data(), rows, descriptor.rowsKind, descriptor.rowsBlock);
  std::vector<float> output(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t source = pout ? static_cast<uint32_t>(pout[row]) : row;
    output[row] = coded[source] * post[source];
  }
  return output;
}

inline bool permutation(std::span<const int32_t> values) {
  std::vector<char> seen(values.size());
  for (int32_t value : values) {
    if (value < 0 || static_cast<size_t>(value) >= values.size() || seen[static_cast<size_t>(value)]) return false;
    seen[static_cast<size_t>(value)] = 1;
  }
  return true;
}

inline std::vector<int32_t> inversePermutation(std::span<const int32_t> values) {
  if (!permutation(values)) throw GgufError("trellis gather is not a permutation");
  std::vector<int32_t> inverse(values.size());
  for (uint32_t i = 0; i < values.size(); ++i) inverse[static_cast<uint32_t>(values[i])] = static_cast<int32_t>(i);
  return inverse;
}

inline std::vector<int32_t> composeOutputPermutation(std::span<const int32_t> pout, uint32_t rows, uint32_t from,
                                                     uint32_t headRows, uint32_t keyHeads, uint32_t valueHeadsPerKey) {
  if (!pout.empty() && (pout.size() != rows || !permutation(pout))) throw GgufError("invalid trellis output gather");
  std::vector<int32_t> composed(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    uint32_t logical = row;
    if (row >= from) {
      const uint32_t local = row - from, head = local / headRows, within = local % headRows;
      logical = from + (head % valueHeadsPerKey) * keyHeads * headRows + (head / valueHeadsPerKey) * headRows + within;
    }
    composed[row] = pout.empty() ? static_cast<int32_t>(logical) : pout[logical];
  }
  return composed;
}

} // namespace richengine::model::trellis

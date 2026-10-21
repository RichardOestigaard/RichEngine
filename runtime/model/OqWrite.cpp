#include "model/OqWrite.hpp"

#include "Checked.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace richengine::model::affine::oq {
namespace {

// The block format a spec repacks to; every (bits, group size) oMLX emits
// has one (metal/abi/QuantFormat.h).
uint32_t formatOf(uint32_t bits, uint32_t groupSize) {
  if (groupSize == 64) {
    if (bits == 4) return GGUF_FMT_AF4;
    if (bits == 5) return GGUF_FMT_AF5;
    if (bits == 6) return GGUF_FMT_AF6;
    if (bits == 8) return GGUF_FMT_AF8;
  }
  if (groupSize == 128) {
    if (bits == 4) return GGUF_FMT_AF4G128;
    if (bits == 5) return GGUF_FMT_AF5G128;
    if (bits == 6) return GGUF_FMT_AF6G128;
    if (bits == 8) return GGUF_FMT_AF8G128;
  }
  throw WeightStoreError("unsupported oQ format: " + std::to_string(bits) + " bits, " +
                         std::to_string(groupSize) + " group size");
}

// Whether the section is the token table's: one part of gathered native
// rows rather than multiplied planes.
bool native(const Section &section) {
  return section.parts.size() == 1 && section.parts.front().nativeRows;
}

// The flat rows a part covers in the section's row space: an expert slab's
// [E, N, K] tensor is E * N rows of one segment.
uint64_t partRows(const Section &section, const ProjectionPart &part) {
  return section.experts > 1 ? uint64_t(section.experts) * part.rows : part.rows;
}
uint64_t totalRows(const Section &section) {
  return section.experts > 1 ? uint64_t(section.experts) * section.rows : section.rows;
}

// One segment of a section: the rows it covers, the parts they read, its
// format (bits 0 for an F32 float segment) and where its descriptor and
// planes sit within the section.
struct Segment {
  uint32_t firstPart = 0, partCount = 0;
  uint64_t rowStart = 0; // the flat row its coverage begins at
  uint64_t rows = 0;
  uint32_t bits = 0, groupSize = 0;
  uint64_t offset = 0, plane0 = 0, meta = 0;
  uint64_t plane0Bytes = 0, metaBytes = 0;
  uint64_t end = 0;
};

// The segments tiling the section's rows in part order. The last segment of
// either kind covers the zero rows the parts leave (the reader wants the
// projection's declared output size exactly).
std::vector<Segment> segments(const Section &section) {
  if (section.parts.empty()) throw WeightStoreError("oQ section has no parts");
  if (section.experts > 1 && section.parts.size() != 1)
    throw WeightStoreError("oQ expert sections hold a single part");
  std::vector<Segment> result;
  const uint64_t total = totalRows(section);
  uint64_t pos = 0;
  for (size_t i = 0; i < section.parts.size();) {
    const ProjectionPart &part = section.parts[i];
    if (part.unquantized || part.dequantize) {
      // A float segment's rows need no tile alignment. A section's only
      // part — a router, a scalar gate — emits its own rows alone; inside a
      // fused projection the last part's segment covers the padding, as the
      // reader expects the declared output size exactly.
      Segment segment;
      segment.firstPart = static_cast<uint32_t>(i);
      segment.partCount = 1;
      segment.rowStart = pos;
      segment.rows = i + 1 == section.parts.size() && section.parts.size() > 1
                         ? total - pos
                         : partRows(section, part);
      result.push_back(segment);
      pos += segment.rows;
      ++i;
      continue;
    }
    // The maximal run of consecutive quantized parts.
    size_t j = i;
    uint64_t covered = 0;
    uint32_t bits = 0, groupSize = std::numeric_limits<uint32_t>::max();
    bool aligned = true;
    while (j < section.parts.size() && !section.parts[j].unquantized && !section.parts[j].dequantize) {
      const ProjectionPart &next = section.parts[j];
      static_cast<void>(formatOf(next.bits, next.groupSize)); // throws on an unsupported spec
      covered += partRows(section, next);
      aligned = aligned && next.rows % QUANT_TILE_ROWS == 0;
      bits = std::max(bits, next.bits);
      groupSize = std::min(groupSize, next.groupSize);
      ++j;
    }
    const bool runLast = j == section.parts.size();
    if (aligned) {
      for (; i < j; ++i) {
        Segment segment;
        segment.firstPart = static_cast<uint32_t>(i);
        segment.partCount = 1;
        segment.rowStart = pos;
        segment.rows = partRows(section, section.parts[i]);
        segment.bits = section.parts[i].bits;
        segment.groupSize = section.parts[i].groupSize;
        if (runLast && i + 1 == j) segment.rows = total - pos;
        result.push_back(segment);
        pos += segment.rows;
      }
    } else {
      Segment segment;
      segment.firstPart = static_cast<uint32_t>(i);
      segment.partCount = static_cast<uint32_t>(j - i);
      segment.rowStart = pos;
      segment.rows = runLast ? total - pos : covered;
      segment.bits = bits;
      segment.groupSize = groupSize;
      result.push_back(segment);
      pos += segment.rows;
      i = j;
    }
  }
  // A single-part section may leave padded rows unwritten — its reader
  // takes one segment. Fused sections must cover theirs exactly.
  if (pos != total && (pos > total || section.parts.size() != 1))
    throw WeightStoreError("oQ section parts do not cover its rows");
  uint32_t quantizedSegments = 0;
  for (const Segment &segment : result) {
    if (segment.bits) ++quantizedSegments;
    if (segment.bits &&
        (segment.rows % QUANT_TILE_ROWS || segment.rows > std::numeric_limits<uint32_t>::max() ||
         section.columns % kGgufBlockColumns || section.columns % segment.groupSize))
      throw WeightStoreError("oQ projection parts do not tile the section");
    if (segment.rows > std::numeric_limits<uint32_t>::max())
      throw WeightStoreError("oQ segment is too large for its descriptor");
  }
  // The fused decode dispatch takes at most three quantized segments
  // (ops/LinearGguf.cpp's GgufDecodeFusedParams).
  if (quantizedSegments > 3)
    throw WeightStoreError("oQ projection has too many formats for the fused decode");
  return result;
}

// The segment's offsets within the section: the descriptor's 16 KiB block,
// then plane0, then meta, each aligned as the reader's sections align.
void layOut(Segment &segment, const Section &section, uint64_t &cursor) {
  segment.offset = cursor;
  segment.plane0 = cursor + kWeightFileAlignment;
  if (segment.bits) {
    const GgufPlaneBytes planes =
        ggufPlaneBytes(kQuantFormats[formatOf(segment.bits, segment.groupSize)], segment.rows, section.columns);
    segment.plane0Bytes = planes.plane0;
    segment.metaBytes = planes.meta;
    segment.meta = segment.plane0 + alignWeightOffset(segment.plane0Bytes);
    segment.end = segment.meta + alignWeightOffset(segment.metaBytes);
  } else {
    segment.plane0Bytes = checkedMultiply<WeightStoreError>(segment.rows, uint64_t(section.columns) * sizeof(float),
                                                          "oQ float segment");
    segment.end = segment.plane0 + alignWeightOffset(segment.plane0Bytes);
  }
  cursor = segment.end;
}

// The BF16 nearest a finite value, ties to even.
uint16_t nearestBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// An IEEE binary16 value as float.
float halfValue(uint16_t bits) {
  const uint32_t mantissa = bits & 0x3FF;
  const uint32_t exponent = (bits >> 10) & 0x1F;
  float value;
  if (exponent == 0x1F)
    value = mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
  else if (!exponent)
    value = std::ldexp(static_cast<float>(mantissa), -24);
  else
    value = std::ldexp(static_cast<float>(mantissa + 1024), static_cast<int>(exponent) - 25);
  return bits & 0x8000 ? -value : value;
}

// The bf16 meta value `bytes` holds: a BF16 source passes through, an F16
// one narrows.
uint16_t metaValue(const uint8_t *bytes, bool f16) {
  uint16_t value;
  std::memcpy(&value, bytes, sizeof value);
  return f16 ? nearestBfloat16(halfValue(value)) : value;
}

// Element e's code in a row's contiguous little-endian bitstream, element e
// at bits [e * bits, e * bits + bits).
uint32_t sourceCode(const uint8_t *row, uint32_t e, uint32_t bits) {
  const uint32_t bit = e * bits;
  uint32_t word = row[bit >> 3];
  if ((bit & 7) + bits > 8) word |= uint32_t(row[(bit >> 3) + 1]) << 8;
  return (word >> (bit & 7)) & ((1u << bits) - 1);
}

// code at slot's bits of a group zeroed before it is filled.
void putCode(uint8_t *group, uint32_t slot, uint32_t code, uint32_t bits) {
  const uint32_t bit = slot * bits;
  const uint32_t word = code << (bit & 7);
  group[bit >> 3] |= static_cast<uint8_t>(word);
  if ((bit & 7) + bits > 8) group[(bit >> 3) + 1] |= static_cast<uint8_t>(word >> 8);
}

// The flat row a part starts at in the section's row space.
uint64_t partStart(const Section &section, uint32_t part) {
  uint64_t start = 0;
  for (uint32_t i = 0; i < part; ++i) start += partRows(section, section.parts[i]);
  return start;
}

// One row of a segment's planes: the codes of every 32-group at slot order
// (AF4's pair words for the legacy format, the slot-order bitstream for the
// oQ ones), each target meta unit's scale and bias from the source unit
// covering it.
void repackRow(const Section &section, const ProjectionPart &part, const Segment &segment, uint32_t format,
               uint64_t row, const uint8_t *codes, const uint8_t *scales, const uint8_t *biases, bool f16,
               uint8_t *plane0, uint8_t *meta) {
  const uint32_t columns = section.columns;
  const uint32_t groups = columns / 32, units = columns / segment.groupSize;
  const QuantFormat &layout = kQuantFormats[format];
  for (uint32_t group = 0; group < groups; ++group) {
    uint8_t slots[32];
    for (uint32_t e = 0; e < 32; ++e) slots[quant_slot(e)] = static_cast<uint8_t>(
        sourceCode(codes, 32 * group + e, part.bits));
    uint8_t *const tile = plane0 + quant_tile_index(static_cast<uint32_t>(row), group, groups) * layout.plane0_bytes;
    if (format == GGUF_FMT_AF4) {
      // The legacy packing: word c packs pairs of slots 8c..8c+7.
      for (uint32_t c = 0; c < 4; ++c) {
        uint32_t word = 0;
        for (uint32_t i = 0; i < 8; ++i) word |= uint32_t(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1));
        std::memcpy(tile + 4 * c, &word, 4);
      }
    } else {
      std::memset(tile, 0, layout.plane0_bytes);
      for (uint32_t slot = 0; slot < 32; ++slot) putCode(tile, slot, slots[slot], segment.bits);
    }
  }
  for (uint32_t unit = 0; unit < units; ++unit) {
    // A merged segment's target group size is the run's minimum: every
    // source unit covers a whole number of target units and replicates.
    const uint64_t source = uint64_t(unit) * segment.groupSize / part.groupSize;
    const uint16_t scale = metaValue(scales + 2 * source, f16);
    const uint16_t bias = metaValue(biases + 2 * source, f16);
    uint8_t *const tile = meta + quant_tile_index(static_cast<uint32_t>(row), unit, units) * layout.meta_bytes;
    std::memcpy(tile, &scale, 2);
    std::memcpy(tile + 2, &bias, 2);
  }
}

// Zeroes one row's plane0 and meta tiles — the rows the parts do not cover.
void zeroRow(const Section &section, const Segment &segment, uint32_t format, uint64_t row, uint8_t *plane0,
             uint8_t *meta) {
  const uint32_t groups = section.columns / 32, units = section.columns / segment.groupSize;
  const QuantFormat &layout = kQuantFormats[format];
  for (uint32_t group = 0; group < groups; ++group)
    std::memset(plane0 + quant_tile_index(static_cast<uint32_t>(row), group, groups) * layout.plane0_bytes, 0,
                layout.plane0_bytes);
  for (uint32_t unit = 0; unit < units; ++unit)
    std::memset(meta + quant_tile_index(static_cast<uint32_t>(row), unit, units) * layout.meta_bytes, 0,
                layout.meta_bytes);
}

// The step's rows of the segment's parts, staged and repacked.
void repackStep(const Section &section, const Segment &segment, uint64_t firstRow, uint32_t rows,
                std::vector<uint8_t> &staging, uint8_t *plane0, uint8_t *meta) {
  const uint32_t columns = section.columns;
  const uint32_t format = formatOf(segment.bits, segment.groupSize);
  for (uint32_t p = segment.firstPart; p < segment.firstPart + segment.partCount; ++p) {
    const ProjectionPart &part = section.parts[p];
    const uint64_t flat = partStart(section, p) - segment.rowStart;
    const uint64_t rows64 = partRows(section, part);
    const uint64_t begin = std::max(firstRow, flat), end = std::min(firstRow + rows, flat + rows64);
    if (begin >= end) continue;
    const uint64_t local = begin - flat, count = end - begin;
    const uint64_t codeBytes = uint64_t(columns) * part.bits / 8;
    const uint64_t units = columns / part.groupSize, metaBytes = units * kBFloat16Bytes;
    if (staging.size() < count * (codeBytes + 2 * metaBytes)) staging.resize(count * (codeBytes + 2 * metaBytes));
    uint8_t *const codes = staging.data();
    uint8_t *const scales = codes + count * codeBytes;
    uint8_t *const biases = scales + count * metaBytes;
    // A strided part's flat rows step through per-expert windows of its
    // source block; contiguous parts read straight.
    for (uint64_t done = 0; done < count;) {
      const uint64_t flatRow = local + done;
      const uint64_t chunk = part.expertStride ? std::min(count - done, part.rows - flatRow % part.rows)
                                              : count - done;
      const uint64_t source = part.sourceRow(flatRow);
      part.fields[0].tensor->read(source * codeBytes, {codes + done * codeBytes, chunk * codeBytes});
      part.fields[1].tensor->read(source * metaBytes, {scales + done * metaBytes, chunk * metaBytes});
      part.fields[2].tensor->read(source * metaBytes, {biases + done * metaBytes, chunk * metaBytes});
      done += chunk;
    }
    const bool f16 = part.fields[1].tensor->dtype == "F16";
    for (uint64_t i = 0; i < count; ++i)
      repackRow(section, part, segment, format, begin + i, codes + i * codeBytes, scales + i * metaBytes,
                biases + i * metaBytes, f16, plane0, meta);
  }
  // The zero rows the parts leave.
  uint64_t covered = 0;
  for (uint32_t p = segment.firstPart; p < segment.firstPart + segment.partCount; ++p)
    covered += partRows(section, section.parts[p]);
  for (uint64_t row = std::max(firstRow, covered); row < firstRow + rows; ++row)
    zeroRow(section, segment, format, row, plane0, meta);
}

// The widest source rows of the segment's unquantized part: BF16 and F16
// widen to F32, F32 copies; the rows past the part are zero.
void floatStep(const Section &section, const Segment &segment, uint64_t firstRow, uint32_t rows,
               std::vector<uint8_t> &staging, uint8_t *to) {
  const ProjectionPart &part = section.parts[segment.firstPart];
  const uint32_t columns = section.columns;
  const uint32_t elementBytes = part.fields[0].tensor->dtype == "F32" ? 4 : 2;
  const uint64_t sourceRowBytes = uint64_t(columns) * elementBytes;
  const uint64_t available = partRows(section, part) - std::min(firstRow, partRows(section, part));
  const uint64_t count = std::min<uint64_t>(rows, available);
  if (staging.size() < count * sourceRowBytes) staging.resize(count * sourceRowBytes);
  for (uint64_t done = 0; done < count;) {
    const uint64_t chunk = part.expertStride
                               ? std::min(count - done, part.rows - (firstRow + done) % part.rows)
                               : count - done;
    part.fields[0].tensor->read(part.sourceRow(firstRow + done) * sourceRowBytes,
                                {staging.data() + done * sourceRowBytes, chunk * sourceRowBytes});
    done += chunk;
  }
  const bool bf16 = part.fields[0].tensor->dtype == "BF16";
  for (uint64_t row = 0; row < rows; ++row) {
    uint8_t *const out = to + row * uint64_t(columns) * sizeof(float);
    if (row >= count) {
      std::memset(out, 0, uint64_t(columns) * sizeof(float));
      continue;
    }
    const uint8_t *const source = staging.data() + row * sourceRowBytes;
    for (uint32_t e = 0; e < columns; ++e) {
      float value;
      if (elementBytes == 4) {
        std::memcpy(&value, source + 4 * e, 4);
      } else {
        uint16_t bits;
        std::memcpy(&bits, source + 2 * e, 2);
        value = bf16 ? std::bit_cast<float>(uint32_t(bits) << 16) : halfValue(bits);
      }
      std::memcpy(out + 4 * e, &value, 4);
    }
  }
}

// The float rows of a part the source quantizes but the reader wants F32
// (the block MoE's router and scalar gate): each value dequantizes to
// f32(scale * code + bias), the kernel's own arithmetic; the rows past the
// part are zero.
void dequantFloatStep(const Section &section, const Segment &segment, uint64_t firstRow, uint32_t rows,
                      std::vector<uint8_t> &staging, uint8_t *to) {
  const ProjectionPart &part = section.parts[segment.firstPart];
  const uint32_t columns = section.columns;
  const uint64_t codeBytes = uint64_t(columns) * part.bits / 8;
  const uint64_t units = columns / part.groupSize, metaBytes = units * kBFloat16Bytes;
  const uint64_t available = partRows(section, part) - std::min(firstRow, partRows(section, part));
  const uint64_t count = std::min<uint64_t>(rows, available);
  if (staging.size() < count * (codeBytes + 2 * metaBytes))
    staging.resize(count * (codeBytes + 2 * metaBytes));
  uint8_t *const codes = staging.data();
  uint8_t *const scales = codes + count * codeBytes;
  uint8_t *const biases = scales + count * metaBytes;
  for (uint64_t done = 0; done < count;) {
    const uint64_t chunk = part.expertStride
                               ? std::min(count - done, part.rows - (firstRow + done) % part.rows)
                               : count - done;
    const uint64_t source = part.sourceRow(firstRow + done);
    part.fields[0].tensor->read(source * codeBytes, {codes + done * codeBytes, chunk * codeBytes});
    part.fields[1].tensor->read(source * metaBytes, {scales + done * metaBytes, chunk * metaBytes});
    part.fields[2].tensor->read(source * metaBytes, {biases + done * metaBytes, chunk * metaBytes});
    done += chunk;
  }
  const bool f16 = part.fields[1].tensor->dtype == "F16";
  for (uint64_t row = 0; row < rows; ++row) {
    float *const out = reinterpret_cast<float *>(to + row * uint64_t(columns) * sizeof(float));
    if (row >= count) {
      std::memset(out, 0, uint64_t(columns) * sizeof(float));
      continue;
    }
    const uint8_t *const rowCodes = codes + row * codeBytes;
    const uint8_t *const rowScales = scales + row * metaBytes;
    const uint8_t *const rowBiases = biases + row * metaBytes;
    for (uint32_t e = 0; e < columns; ++e) {
      const uint32_t unit = e / part.groupSize;
      const float scale = std::bit_cast<float>(uint32_t(metaValue(rowScales + 2 * unit, f16)) << 16);
      const float bias = std::bit_cast<float>(uint32_t(metaValue(rowBiases + 2 * unit, f16)) << 16);
      out[e] = scale * float(sourceCode(rowCodes, e, part.bits)) + bias;
    }
  }
}

// The section's descriptor tasks and row tasks over its segments.
void addSegmentTasks(const Section &section, uint8_t *at,
                     std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks) {
  std::vector<Segment> list = segments(section);
  uint64_t cursor = 0;
  for (Segment &segment : list) layOut(segment, section, cursor);
  for (const Segment &segment : list) {
    const uint32_t format = segment.bits ? formatOf(segment.bits, segment.groupSize) : GGUF_FMT_COUNT;
    tasks.push_back([&section, segment, format, at](std::vector<uint8_t> &) {
      GgufTensorDescriptor descriptor{};
      descriptor.outputSize = static_cast<uint32_t>(segment.rows);
      descriptor.inputSize = section.columns;
      if (segment.bits) {
        const QuantFormat &layout = kQuantFormats[format];
        descriptor.type = layout.ggml_type;
        descriptor.p0 = layout.plane0_bytes;
        descriptor.p1 = layout.plane1_bytes;
        descriptor.metaBytes = layout.meta_bytes;
        descriptor.metaGroups = layout.meta_groups;
        descriptor.plane0Bytes = segment.plane0Bytes;
        descriptor.metaTotalBytes = segment.metaBytes;
      } else {
        descriptor.type = ggml::kF32;
        descriptor.plane0Bytes = segment.plane0Bytes;
      }
      uint8_t *const base = at + segment.offset;
      std::memcpy(base, &descriptor, sizeof descriptor);
      // The padding the row tasks do not write; the section is one extent.
      std::memset(base + sizeof descriptor, 0, segment.plane0 - (segment.offset + sizeof descriptor));
      if (segment.bits) {
        std::memset(at + segment.plane0 + segment.plane0Bytes, 0,
                    segment.meta - (segment.plane0 + segment.plane0Bytes));
        std::memset(at + segment.meta + segment.metaBytes, 0, segment.end - (segment.meta + segment.metaBytes));
      } else {
        std::memset(at + segment.plane0 + segment.plane0Bytes, 0,
                    segment.end - (segment.plane0 + segment.plane0Bytes));
      }
    });
    if (!segment.bits) {
      // Rows of F32 values within the staging bound.
      const ProjectionPart &part = section.parts[segment.firstPart];
      const uint64_t codeBytes = uint64_t(section.columns) * part.bits / 8;
      const uint64_t metaBytes = uint64_t(section.columns) / part.groupSize * kBFloat16Bytes;
      const uint64_t rowBytes = part.dequantize
                                    ? codeBytes + 2 * metaBytes
                                    : uint64_t(section.columns) *
                                          (part.fields[0].tensor->dtype == "F32" ? 4 : 2);
      const uint64_t stepRows = std::max<uint64_t>(1, kLoadStepBytes / rowBytes);
      for (uint64_t first = 0; first < segment.rows; first += stepRows) {
        const uint32_t rows = static_cast<uint32_t>(std::min(stepRows, segment.rows - first));
        tasks.push_back([&section, segment, first, rows, at](std::vector<uint8_t> &staging) {
          uint8_t *const to = at + segment.plane0 + first * section.columns * 4;
          if (section.parts[segment.firstPart].dequantize)
            dequantFloatStep(section, segment, first, rows, staging, to);
          else
            floatStep(section, segment, first, rows, staging, to);
        });
      }
      continue;
    }
    // Steps of rows within the staging bound: the widest part's codes and
    // meta rows.
    uint64_t rowBytes = 0;
    for (uint32_t p = segment.firstPart; p < segment.firstPart + segment.partCount; ++p) {
      const ProjectionPart &part = section.parts[p];
      rowBytes = std::max<uint64_t>(rowBytes, uint64_t(section.columns) * part.bits / 8 +
                                                2 * uint64_t(section.columns) / part.groupSize * kBFloat16Bytes);
    }
    const uint64_t stepRows = std::max<uint64_t>(1, kLoadStepBytes / rowBytes);
    for (uint64_t first = 0; first < segment.rows; first += stepRows) {
      const uint32_t rows = static_cast<uint32_t>(std::min(stepRows, segment.rows - first));
      tasks.push_back([&section, segment, first, rows, at](std::vector<uint8_t> &staging) {
        repackStep(section, segment, first, rows, staging, at + segment.plane0, at + segment.meta);
      });
    }
  }
}

// The token table's section: the descriptor in its 16 KiB block, then the
// native rows — per group of `groupSize` elements the element-order code
// bytes and the {bf16 scale, bf16 bias} meta the gather decodes — or the
// widened F32 rows of an unquantized table.
void addNativeTasks(const Section &section, uint8_t *at,
                    std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks) {
  const ProjectionPart &part = section.parts.front();
  const uint32_t columns = section.columns;
  const uint32_t format = part.unquantized ? GGUF_FMT_COUNT : formatOf(part.bits, part.groupSize);
  const uint64_t rowBytes =
      part.unquantized ? uint64_t(columns) * sizeof(float)
                       : ggufRowBytes(kQuantFormats[format], columns);
  const uint64_t dataBytes = checkedMultiply<WeightStoreError>(part.rows, rowBytes, "oQ embedding rows");
  tasks.push_back([at, &part, columns, format, dataBytes](std::vector<uint8_t> &) {
    GgufTensorDescriptor descriptor{};
    descriptor.type = part.unquantized ? ggml::kF32 : kQuantFormats[format].ggml_type;
    descriptor.outputSize = part.rows;
    descriptor.inputSize = columns;
    descriptor.plane0Bytes = dataBytes;
    std::memcpy(at, &descriptor, sizeof descriptor);
    std::memset(at + sizeof descriptor, 0, kWeightFileAlignment - sizeof descriptor);
  });
  uint8_t *const rows = at + kWeightFileAlignment;
  if (part.unquantized) {
    Segment segment;
    segment.rows = part.rows;
    const uint64_t sourceRowBytes =
        uint64_t(columns) * (part.fields[0].tensor->dtype == "F32" ? 4 : 2);
    const uint64_t stepRows = std::max<uint64_t>(1, kLoadStepBytes / sourceRowBytes);
    for (uint64_t first = 0; first < part.rows; first += stepRows) {
      const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(stepRows, part.rows - first));
      tasks.push_back([&section, segment, first, count, rows](std::vector<uint8_t> &staging) {
        floatStep(section, segment, first, count, staging, rows + first * section.columns * 4);
      });
    }
    return;
  }
  const uint64_t codeBytes = uint64_t(columns) * part.bits / 8;
  const uint64_t units = columns / part.groupSize, metaBytes = units * kBFloat16Bytes;
  const uint64_t sourceRowBytes = codeBytes + 2 * metaBytes;
  const uint64_t stepRows = std::max<uint64_t>(1, kLoadStepBytes / sourceRowBytes);
  for (uint64_t first = 0; first < part.rows; first += stepRows) {
    const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(stepRows, part.rows - first));
    tasks.push_back([&part, first, count, columns, codeBytes, metaBytes, rowBytes,
                    rows](std::vector<uint8_t> &staging) {
      if (staging.size() < uint64_t(count) * (codeBytes + 2 * metaBytes))
        staging.resize(uint64_t(count) * (codeBytes + 2 * metaBytes));
      uint8_t *const codes = staging.data();
      uint8_t *const scales = codes + uint64_t(count) * codeBytes;
      uint8_t *const biases = scales + uint64_t(count) * metaBytes;
      part.fields[0].tensor->read(first * codeBytes, {codes, uint64_t(count) * codeBytes});
      part.fields[1].tensor->read(first * metaBytes, {scales, uint64_t(count) * metaBytes});
      part.fields[2].tensor->read(first * metaBytes, {biases, uint64_t(count) * metaBytes});
      const bool f16 = part.fields[1].tensor->dtype == "F16";
      const uint32_t groups = columns / part.groupSize;
      for (uint64_t row = 0; row < count; ++row) {
        uint8_t *to = rows + (first + row) * rowBytes;
        for (uint32_t group = 0; group < groups; ++group) {
          // The codes are already an element-order bitstream: a group's
          // bytes move whole, then its meta follows.
          const uint64_t codeBytesPerGroup = uint64_t(part.groupSize) * part.bits / 8;
          std::memcpy(to, codes + row * codeBytes + group * codeBytesPerGroup, codeBytesPerGroup);
          const uint16_t scale = metaValue(scales + row * metaBytes + 2 * group, f16);
          const uint16_t bias = metaValue(biases + row * metaBytes + 2 * group, f16);
          std::memcpy(to + codeBytesPerGroup, &scale, 2);
          std::memcpy(to + codeBytesPerGroup + 2, &bias, 2);
          to += codeBytesPerGroup + 4;
        }
      }
    });
  }
}

} // namespace

uint64_t sectionBytes(const Section &section) {
  if (native(section)) {
    const ProjectionPart &part = section.parts.front();
    const uint64_t rowBytes = part.unquantized
                                  ? uint64_t(section.columns) * sizeof(float)
                                  : ggufRowBytes(kQuantFormats[formatOf(part.bits, part.groupSize)], section.columns);
    return kWeightFileAlignment +
           checkedMultiply<WeightStoreError>(part.rows, rowBytes, "oQ embedding rows");
  }
  uint64_t cursor = 0;
  for (Segment segment : segments(section)) layOut(segment, section, cursor);
  return cursor;
}

void addTasks(const Section &section, uint8_t *at,
              std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks) {
  if (native(section)) return addNativeTasks(section, at, tasks);
  addSegmentTasks(section, at, tasks);
}

} // namespace richengine::model::affine::oq

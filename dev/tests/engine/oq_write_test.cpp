// The Oq sections (model/OqWrite.cpp) of an oMLX mixed-precision checkpoint:
// aligned parts emit one segment each at their own format, an unaligned run
// merges at its finest spec, an unquantized part is an F32 segment and the
// token table keeps its native rows — every emitted weight dequantizing to
// the source's. Runs without Metal: the writers work on plain memory.

#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/AffinePlan.hpp"
#include "model/AffinePreparation.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/OqWrite.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace richengine::model;
using richengine::test::rejects;
using richengine::test::require;
namespace {

constexpr uint32_t kColumns = 256;

uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352d;
  value ^= value >> 15;
  value *= 0x846ca68b;
  return value ^ (value >> 16);
}
uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return uint16_t((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
float fromBf16(uint16_t bits) {
  const uint32_t wide = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &wide, 4);
  return value;
}
uint16_t f16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const uint32_t sign = (bits >> 16) & 0x8000;
  const int exponent = int((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFF;
  if (exponent <= 0) return uint16_t(sign | (exponent < -10 ? 0 : (mantissa | 0x800000) >> (14 - exponent)));
  if (exponent >= 31) return uint16_t(sign | 0x7BFF);
  return uint16_t(sign | (uint32_t(exponent) << 10) | (mantissa >> 13));
}
float halfValue(uint16_t bits) {
  const uint32_t mantissa = bits & 0x3FF;
  const uint32_t exponent = (bits >> 10) & 0x1F;
  float value;
  if (exponent == 0x1F)
    value = mantissa ? NAN : INFINITY;
  else if (!exponent)
    value = std::ldexp(float(mantissa), -24);
  else
    value = std::ldexp(float(mantissa + 1024), int(exponent) - 25);
  return bits & 0x8000 ? -value : value;
}

// A part's safetensors: the element-order code bitstream, the meta as BF16
// or F16 and the values a reader dequantizes with.
struct Quant {
  std::vector<uint8_t> codes, scales, biases;
  std::vector<uint32_t> code;
  std::vector<float> scale, bias;
  uint32_t rows = 0, cols = 0, bits = 0, gs = 0;
  uint64_t codeRow() const { return uint64_t(cols) * bits / 8; }
  uint64_t metaRow() const { return uint64_t(cols) / gs * 2; }
};

Quant makeQuant(uint32_t rows, uint32_t cols, uint32_t bits, uint32_t gs, bool f16meta, uint32_t seed) {
  Quant q;
  q.rows = rows; q.cols = cols; q.bits = bits; q.gs = gs;
  q.code.resize(uint64_t(rows) * cols);
  for (uint64_t i = 0; i < q.code.size(); ++i) q.code[i] = mix(seed + uint32_t(i)) % (1u << bits);
  q.codes.assign(uint64_t(rows) * cols * bits / 8, 0);
  for (uint64_t e = 0; e < q.code.size(); ++e) {
    const uint64_t bit = e * bits;
    const uint32_t word = q.code[e] << (bit & 7);
    q.codes[bit >> 3] |= uint8_t(word);
    if ((bit & 7) + bits > 8) q.codes[(bit >> 3) + 1] |= uint8_t(word >> 8);
  }
  const uint64_t units = cols / gs;
  q.scales.resize(rows * units * 2);
  q.biases.resize(rows * units * 2);
  q.scale.resize(rows * units);
  q.bias.resize(rows * units);
  for (uint64_t i = 0; i < q.scale.size(); ++i) {
    const float s = 0.01F + float(mix(seed ^ 0x9e3779b9u + uint32_t(i)) % 23) * 0.0005F;
    const float b = -s * float(1 + ((mix(seed + uint32_t(i)) >> 8) & 3));
    uint16_t sv, bv;
    if (f16meta) {
      const uint16_t hs = f16(s), hb = f16(b);
      std::memcpy(q.scales.data() + 2 * i, &hs, 2);
      std::memcpy(q.biases.data() + 2 * i, &hb, 2);
      sv = bf16(halfValue(hs));
      bv = bf16(halfValue(hb));
    } else {
      sv = bf16(s);
      bv = bf16(b);
      std::memcpy(q.scales.data() + 2 * i, &sv, 2);
      std::memcpy(q.biases.data() + 2 * i, &bv, 2);
    }
    q.scale[i] = fromBf16(sv);
    q.bias[i] = fromBf16(bv);
  }
  return q;
}

// scale * code + bias of element (row, col), the value any layout of the
// part must dequantize to.
float quantValue(const Quant &q, uint64_t row, uint64_t col) {
  const uint64_t units = q.cols / q.gs;
  return q.scale[row * units + col / q.gs] * float(q.code[row * q.cols + col]) + q.bias[row * units + col / q.gs];
}

// One safetensors shard; the JSON header of the tensors added to it.
struct Shard {
  std::vector<uint8_t> data;
  std::string header = "{";
  void add(std::string_view name, const char *dtype, std::vector<uint64_t> shape,
           const std::vector<uint8_t> &bytes) {
    const uint64_t begin = data.size();
    data.insert(data.end(), bytes.begin(), bytes.end());
    std::string dims;
    for (const uint64_t dim : shape) dims += std::to_string(dim) + ",";
    dims.pop_back();
    header += "\"" + std::string(name) + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + dims +
              "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(data.size()) + "]},";
  }
  void write(const std::filesystem::path &path) {
    header.back() = '}';
    std::vector<uint8_t> file(sizeof(uint64_t));
    const uint64_t length = header.size();
    std::memcpy(file.data(), &length, sizeof length);
    file.insert(file.end(), header.begin(), header.end());
    file.insert(file.end(), data.begin(), data.end());
    richengine::test::writeFile(path, file);
  }
};

struct Fixture {
  Quant five = makeQuant(256, kColumns, 5, 64, false, 11);   // global dict entry: 5/64
  Quant four = makeQuant(256, kColumns, 4, 64, false, 13);   // the global 4/64
  Quant eight = makeQuant(256, kColumns, 8, 128, false, 17); // 8/128
  Quant six = makeQuant(256, kColumns, 6, 64, true, 19);     // 6/64, F16 meta
  Quant smallA = makeQuant(64, kColumns, 4, 64, false, 23);  // an unaligned run
  Quant smallB = makeQuant(192, kColumns, 6, 64, false, 29);
  Quant tok = makeQuant(512, kColumns, 4, 64, false, 31);    // the token table
  std::vector<uint8_t> plain;                                // 128 x 256 BF16
  std::vector<float> plainValues;

  Fixture() {
    plain.resize(128 * kColumns * 2);
    for (uint64_t i = 0; i < 128 * kColumns; ++i) {
      const uint16_t bits = bf16(0.02F * float(int(mix(uint32_t(i) + 41) % 17) - 8));
      std::memcpy(plain.data() + 2 * i, &bits, 2);
      plainValues.push_back(fromBf16(bits));
    }
  }

  void write(const std::filesystem::path &root) const {
    richengine::test::writeFile(
        root / "config.json",
        R"({"quantization":{"bits":4,"group_size":64,"mode":"affine",)"
        R"("five":{"bits":5,"group_size":64},)"
        R"("eight":{"bits":8,"group_size":128},)"
        R"("six":{"bits":6,"group_size":64},)"
        R"("small_b":{"bits":6,"group_size":64}}})");
    Shard shard;
    const auto quantized = [&](const Quant &q, const std::string &name, const char *metaType) {
      shard.add(name + ".weight", "U32", {q.rows, uint64_t(q.cols) * q.bits / 32}, q.codes);
      shard.add(name + ".scales", metaType, {q.rows, uint64_t(q.cols) / q.gs}, q.scales);
      shard.add(name + ".biases", metaType, {q.rows, uint64_t(q.cols) / q.gs}, q.biases);
    };
    quantized(five, "five", "BF16");
    quantized(four, "four", "BF16");
    quantized(eight, "eight", "BF16");
    quantized(six, "six", "F16");
    quantized(smallA, "small_a", "BF16");
    quantized(smallB, "small_b", "BF16");
    quantized(tok, "tok", "BF16");
    shard.add("plain.weight", "BF16", {128, kColumns}, plain);
    shard.add("norm", "BF16", {64}, std::vector<uint8_t>(128, 0x3C));
    shard.write(root / "model.safetensors");
  }

  // The image the oQ planner emits: a norm copy, the mixed-spec section, the
  // unaligned run, the F16-meta section, the unquantized section and the
  // token table's native rows.
  affine::Image image() const {
    affine::Image result = affine::image("layer-0.bin", "MDFN0001", 0, 1);
    affine::copy(result, "norm", {64});
    const auto part = [](const Quant &q, const std::string &name) {
      affine::ProjectionPart part{q.rows, {}};
      part.bits = q.bits;
      part.groupSize = q.gs;
      part.fields.push_back({name + ".weight", {"U32"}, {q.rows, uint64_t(q.cols) * q.bits / 32}});
      part.fields.push_back({name + ".scales", {"BF16", "F16"}, {q.rows, uint64_t(q.cols) / q.gs}});
      part.fields.push_back({name + ".biases", {"BF16", "F16"}, {q.rows, uint64_t(q.cols) / q.gs}});
      return part;
    };
    const auto expect = [&](const std::string &name, uint32_t bits, uint32_t gs) {
      result.quantized.emplace_back(name, affine::QuantSpec{bits, gs, "affine"});
    };
    const auto section = [&](uint32_t rows, std::initializer_list<affine::ProjectionPart> parts) {
      affine::Section section;
      section.kind = affine::SectionKind::Oq;
      section.rows = rows;
      section.columns = kColumns;
      for (const affine::ProjectionPart &part : parts) section.parts.push_back(part);
      section.bytes = affine::oq::sectionBytes(section);
      affine::append(result, std::move(section));
    };
    expect("five", 5, 64);
    expect("four", 4, 64);
    expect("eight", 8, 128);
    section(3 * 256, {part(five, "five"), part(four, "four"), part(eight, "eight")});
    expect("small_a", 4, 64);
    expect("small_b", 6, 64);
    section(256, {part(smallA, "small_a"), part(smallB, "small_b")});
    expect("six", 6, 64);
    section(256, {part(six, "six")});
    result.quantized.emplace_back("plain", std::nullopt);
    {
      affine::Section section;
      section.kind = affine::SectionKind::Oq;
      section.rows = 256;
      section.columns = kColumns;
      affine::ProjectionPart part{128, {}};
      part.unquantized = true;
      part.fields.push_back({"plain.weight", {"BF16", "F16", "F32"}, {128, kColumns}});
      section.parts.push_back(std::move(part));
      section.bytes = affine::oq::sectionBytes(section);
      affine::append(result, std::move(section));
    }
    expect("tok", 4, 64);
    {
      affine::Section section;
      section.kind = affine::SectionKind::Oq;
      section.rows = tok.rows;
      section.columns = kColumns;
      affine::ProjectionPart part{tok.rows, {}};
      part.nativeRows = true;
      part.bits = tok.bits;
      part.groupSize = tok.gs;
      part.fields.push_back({"tok.weight", {"U32"}, {tok.rows, uint64_t(kColumns) * tok.bits / 32}});
      part.fields.push_back({"tok.scales", {"BF16", "F16"}, {tok.rows, uint64_t(kColumns) / tok.gs}});
      part.fields.push_back({"tok.biases", {"BF16", "F16"}, {tok.rows, uint64_t(kColumns) / tok.gs}});
      section.parts.push_back(std::move(part));
      section.bytes = affine::oq::sectionBytes(section);
      affine::append(result, std::move(section));
    }
    return result;
  }
};

// A segment's descriptor and plane offsets as the writer lays them out.
struct Segment {
  GgufTensorDescriptor descriptor{};
  uint64_t base = 0, plane0 = 0, meta = 0, end = 0;
  [[nodiscard]] uint32_t bits() const { return descriptor.p0 * 8 / 32; }
  [[nodiscard]] uint32_t groupSize() const { return descriptor.metaGroups * 32; }
};

std::vector<Segment> segments(const std::vector<uint8_t> &image, const affine::Section &section) {
  std::vector<Segment> result;
  for (uint64_t at = section.offset; at < section.offset + section.bytes;) {
    Segment segment;
    std::memcpy(&segment.descriptor, image.data() + at, sizeof segment.descriptor);
    segment.base = at;
    segment.plane0 = at + kWeightFileAlignment;
    if (segment.descriptor.type == ggml::kF32) {
      segment.end = segment.plane0 + alignWeightOffset(segment.descriptor.plane0Bytes);
    } else {
      segment.meta = segment.plane0 + alignWeightOffset(segment.descriptor.plane0Bytes);
      segment.end = segment.meta + alignWeightOffset(segment.descriptor.metaTotalBytes);
    }
    result.push_back(segment);
    at = segment.end;
  }
  return result;
}

// The fp32 weight the segment's planes hold at (row, col) — the slot-order
// bitstream of the oQ formats, AF4's pair words for the legacy one.
float emittedValue(const std::vector<uint8_t> &image, const Segment &segment, uint32_t row, uint32_t col) {
  const GgufTensorDescriptor &d = segment.descriptor;
  const uint32_t groups = d.inputSize / 32;
  const uint8_t *tile = image.data() + segment.plane0 + quant_tile_index(row, col / 32, groups) * d.p0;
  const uint32_t slot = quant_slot(col & 31);
  uint32_t code;
  if (d.type == GGUF_PRIVATE_TYPE_AF4) {
    uint32_t word;
    std::memcpy(&word, tile + 4 * (slot / 8), 4);
    code = (word >> (((slot & 7) & 1) * 16 + 4 * ((slot & 7) >> 1))) & 15;
  } else {
    const uint32_t bits = segment.bits(), bit = slot * bits;
    uint32_t word = tile[bit >> 3];
    if ((bit & 7) + bits > 8) word |= uint32_t(tile[(bit >> 3) + 1]) << 8;
    code = (word >> (bit & 7)) & ((1u << bits) - 1);
  }
  const uint32_t units = d.inputSize / segment.groupSize();
  const uint8_t *unit = image.data() + segment.meta + quant_tile_index(row, col / segment.groupSize(), units) * 4;
  uint16_t scale, bias;
  std::memcpy(&scale, unit, 2);
  std::memcpy(&bias, unit + 2, 2);
  return fromBf16(scale) * float(code) + fromBf16(bias);
}

void requireDescriptor(const Segment &segment, uint32_t format, uint32_t rows, const char *label) {
  const QuantFormat &layout = kQuantFormats[format];
  require(segment.descriptor.type == layout.ggml_type && segment.descriptor.outputSize == rows &&
              segment.descriptor.inputSize == kColumns && segment.descriptor.p0 == layout.plane0_bytes &&
              !segment.descriptor.p1 && !segment.descriptor.plane1Bytes &&
              segment.descriptor.metaBytes == layout.meta_bytes &&
              segment.descriptor.metaGroups == layout.meta_groups &&
              segment.descriptor.plane0Bytes == uint64_t(rows) * (kColumns / 32) * layout.plane0_bytes &&
              segment.descriptor.metaTotalBytes == uint64_t(rows) * (kColumns / layout.block_elements) * 4,
          label);
}

} // namespace

int main() {
  try {
    const richengine::test::TemporaryDirectory directory("richengine-oq-write");
    const std::filesystem::path &root = directory.path();
    const Fixture fixture;
    fixture.write(root);
    SafetensorsCheckpoint source(root);
    affine::Image image = fixture.image();
    affine::bind(image, source);
    std::vector<uint8_t> bytes(image.bytes);
    affine::writeAffineImage(bytes, image);

    require(image.sections.size() == 6 && image.sections[0].kind == affine::SectionKind::Copy,
            "the oQ image's sections changed shape");

    // The mixed-spec section: one segment per aligned part at its format.
    {
      const std::vector<Segment> segs = segments(bytes, image.sections[1]);
      require(segs.size() == 3, "aligned parts did not emit one segment each");
      requireDescriptor(segs[0], GGUF_FMT_AF5, 256, "the AF5 descriptor is inconsistent");
      requireDescriptor(segs[1], GGUF_FMT_AF4, 256, "the AF4 descriptor is inconsistent");
      requireDescriptor(segs[2], GGUF_FMT_AF8G128, 256, "the AF8G128 descriptor is inconsistent");
      const Quant *expected[3] = {&fixture.five, &fixture.four, &fixture.eight};
      for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t row = 0; row < 256; ++row)
          for (uint32_t col = 0; col < kColumns; ++col)
            require(emittedValue(bytes, segs[s], row, col) == quantValue(*expected[s], row, col),
                    "the emitted segment's weights differ from the source's");
    }

    // The unaligned run merged into one segment at its finest spec, 6/64.
    {
      const std::vector<Segment> segs = segments(bytes, image.sections[2]);
      require(segs.size() == 1, "the unaligned run did not merge");
      requireDescriptor(segs[0], GGUF_FMT_AF6, 256, "the merged descriptor is inconsistent");
      for (uint32_t row = 0; row < 256; ++row)
        for (uint32_t col = 0; col < kColumns; ++col)
          require(emittedValue(bytes, segs[0], row, col) ==
                      (row < fixture.smallA.rows ? quantValue(fixture.smallA, row, col)
                                                 : quantValue(fixture.smallB, row - fixture.smallA.rows, col)),
                  "the merged segment's weights differ from the source's");
    }

    // The F16 meta narrowed to BF16.
    {
      const std::vector<Segment> segs = segments(bytes, image.sections[3]);
      require(segs.size() == 1, "the F16-meta part did not emit one segment");
      requireDescriptor(segs[0], GGUF_FMT_AF6, 256, "the AF6 descriptor is inconsistent");
      for (uint32_t row = 0; row < 256; ++row)
        for (uint32_t col = 0; col < kColumns; ++col)
          require(emittedValue(bytes, segs[0], row, col) == quantValue(fixture.six, row, col),
                  "the F16 meta did not narrow to the BF16 the planes hold");
    }

    // A single-part section's F32 segment holds the part's rows alone: the
    // block MoE's router and scalar gate read their segments by the part's
    // count, not the section's padded one.
    {
      const std::vector<Segment> segs = segments(bytes, image.sections[4]);
      require(segs.size() == 1 && segs[0].descriptor.type == ggml::kF32 &&
                  segs[0].descriptor.outputSize == 128 && segs[0].descriptor.inputSize == kColumns &&
                  segs[0].descriptor.plane0Bytes == uint64_t(128) * kColumns * sizeof(float),
              "the float segment's descriptor is inconsistent");
      const float *rows = reinterpret_cast<const float *>(bytes.data() + segs[0].plane0);
      for (uint32_t row = 0; row < 128; ++row)
        for (uint32_t col = 0; col < kColumns; ++col) {
          const float expected = fixture.plainValues[uint64_t(row) * kColumns + col];
          require(rows[uint64_t(row) * kColumns + col] == expected, "the float segment's values differ");
        }
    }

    // The token table's native rows: one AF4 block {codes, scale, bias} per
    // group of 64.
    {
      const affine::Section &section = image.sections[5];
      GgufTensorDescriptor descriptor;
      std::memcpy(&descriptor, bytes.data() + section.offset, sizeof descriptor);
      require(descriptor.type == GGUF_PRIVATE_TYPE_AF4 && descriptor.outputSize == fixture.tok.rows &&
                  descriptor.inputSize == kColumns &&
                  descriptor.plane0Bytes == uint64_t(fixture.tok.rows) * ggufRowBytes(kQuantFormats[GGUF_FMT_AF4], kColumns),
              "the embedding descriptor is inconsistent");
      const uint8_t *rows = bytes.data() + section.offset + kWeightFileAlignment;
      for (uint64_t row = 0; row < fixture.tok.rows; ++row)
        for (uint32_t group = 0; group < kColumns / 64; ++group) {
          const uint8_t *block = rows + row * ggufRowBytes(kQuantFormats[GGUF_FMT_AF4], kColumns) + group * 36;
          require(std::memcmp(block, fixture.tok.codes.data() + row * fixture.tok.codeRow() + group * 32, 32) == 0,
                  "the native row's codes differ from the source's");
          const uint64_t unit = row * (kColumns / 64) + group;
          require(std::memcmp(block + 32, fixture.tok.scales.data() + 2 * unit, 2) == 0,
                  "the native row's scale differs from the source's");
          require(std::memcmp(block + 34, fixture.tok.biases.data() + 2 * unit, 2) == 0,
                  "the native row's bias differs from the source's");
        }
    }
    std::cout << "oq write: per-part segments, merged run, F16 meta, float and native rows PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

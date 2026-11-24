// One-off verification for the stored-planes GPU repack
// (model/AffineToBlock.cpp blockStoredPlanes, gguf_repack_stored): writes a
// Block-layout image through writeAffineImage's Metal form and compares it
// byte for byte with the CPU writer, which affine_block_test already oracles
// against the affine tiles. Covers multi-part boundaries, rows past the
// parts and multiple pipeline chunks.

#include "metal/MetalBackend.hpp"
#include "model/AffinePlan.hpp"
#include "model/AffineToBlock.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace richengine;
namespace {

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

void writeFile(const std::filesystem::path &path, const std::vector<uint8_t> &bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}
void writeFile(const std::filesystem::path &path, std::string_view text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
}

struct Part {
  uint64_t weight, scales, biases;
};

// A shard of Q4 projections of partRows[p] rows and columns columns each.
void writeFixture(const std::filesystem::path &root, const std::vector<uint32_t> &partRows,
                  uint32_t columns, std::vector<Part> &offsets) {
  writeFile(root / "config.json", R"({"quantization":{"bits":4,"group_size":64}})");
  std::vector<uint8_t> data;
  const auto tensor = [&](std::vector<uint8_t> bytes) {
    data.insert(data.end(), bytes.begin(), bytes.end());
    return data.size() - bytes.size();
  };
  offsets.clear();
  std::string headerTensors;
  char entry[512];
  for (uint32_t part = 0; part < partRows.size(); ++part) {
    const uint32_t rows = partRows[part];
    std::vector<uint8_t> weight(uint64_t(rows) * columns / 2), scales(uint64_t(rows) * (columns / 64) * 2),
        biases(scales.size());
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = uint8_t(mix(uint32_t(i) + 17 * part));
    for (size_t i = 0; i < scales.size() / 2; ++i) {
      const float scale = 0.004F + float(mix(uint32_t(i) + 31 * part) % 17) * 0.0001F;
      const uint16_t s = bf16(scale), b = bf16(-7.5F * scale);
      std::memcpy(scales.data() + 2 * i, &s, 2);
      std::memcpy(biases.data() + 2 * i, &b, 2);
    }
    const uint64_t w = tensor(std::move(weight)), s = tensor(std::move(scales)),
                   b = tensor(std::move(biases)), e = data.size();
    offsets.push_back({w, s, b});
    const std::string name = "p" + std::to_string(part);
    std::snprintf(entry, sizeof entry,
                  R"("%s.weight":{"dtype":"U32","shape":[%u,%u],"data_offsets":[%llu,%llu]},)"
                  R"("%s.scales":{"dtype":"BF16","shape":[%u,%u],"data_offsets":[%llu,%llu]},)"
                  R"("%s.biases":{"dtype":"BF16","shape":[%u,%u],"data_offsets":[%llu,%llu]},)",
                  name.c_str(), rows, columns / 8, (unsigned long long)w, (unsigned long long)s,
                  name.c_str(), rows, columns / 64, (unsigned long long)s, (unsigned long long)b,
                  name.c_str(), rows, columns / 64, (unsigned long long)b, (unsigned long long)e);
    headerTensors += entry;
  }
  // A bf16 norm copy, to keep a non-Block section in the image.
  const uint64_t norm = tensor(std::vector<uint8_t>(128, 0x3C));
  std::snprintf(entry, sizeof entry, R"("norm":{"dtype":"BF16","shape":[64],"data_offsets":[%llu,%llu]})",
                (unsigned long long)norm, (unsigned long long)(norm + 128));
  headerTensors += entry;
  const std::string header = "{" + headerTensors + "}";
  std::vector<uint8_t> file(sizeof(uint64_t));
  const uint64_t headerBytes = header.size();
  std::memcpy(file.data(), &headerBytes, sizeof headerBytes);
  file.insert(file.end(), header.begin(), header.end());
  file.insert(file.end(), data.begin(), data.end());
  writeFile(root / "model.safetensors", file);
}

// A block image of one norm copy plus Block sections of the parts; rows
// [padFrom, rows) of each section have no part (zero rows).
model::affine::Image image(std::string name, uint32_t rows, uint32_t columns, uint32_t parts, uint32_t partRows,
                           uint32_t offset) {
  model::affine::Image result = model::affine::image(std::move(name), "MDFN0001", 0, 1);
  model::affine::copy(result, "norm", {64});
  model::affine::Section section;
  section.kind = model::affine::SectionKind::Projection;
  section.rows = rows;
  section.columns = columns;
  section.bits = 4;
  section.experts = 1;
  section.bytes = uint64_t(rows) * columns * 4 / 8 + uint64_t(rows) * columns / 16;
  for (uint32_t part = 0; part < parts; ++part) {
    const std::string p = "p" + std::to_string(offset + part);
    result.quantized.emplace_back(p, 4);
    model::affine::ProjectionPart piece{partRows, {}};
    piece.fields.push_back({p + ".weight", {"U32"}, {partRows, columns / 8}});
    piece.fields.push_back({p + ".scales", {"BF16"}, {partRows, columns / 64}});
    piece.fields.push_back({p + ".biases", {"BF16"}, {partRows, columns / 64}});
    section.parts.push_back(std::move(piece));
  }
  model::affine::append(result, std::move(section));
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: affine-block-gpu-check METALLIB DIRECTORY");
    const std::filesystem::path root(argv[2]);
    std::filesystem::create_directories(root);
    // Two chunks' worth of rows: step rows bound staging at
    // kGgufRepackStagingBytes over 9 * columns / 16 bytes a row, so
    // 2 * 8192 rows of 8192 columns need three chunks.
    constexpr uint32_t kColumns = 8192, kPartRows = 8192;
    std::vector<Part> offsets;
    writeFixture(root, {kPartRows, kPartRows, kPartRows / 2, kPartRows}, kColumns, offsets);
    auto planned = std::make_shared<model::affine::PlannedCheckpoint>(root);
    // A two-part section of 16384 rows (parts p0, p1) and one whose last
    // 4096 of 8192 rows are past its part p2 (zero rows); p3 unused.
    planned->images = {image("layer-0.bin", 2 * kPartRows, kColumns, 2, kPartRows, 0),
                       image("layer-1.bin", kPartRows, kColumns, 1, kPartRows / 2, 2)};
    for (auto &img : planned->images) model::affine::bind(img, planned->source);

    metal::MetalBackend backend(argv[1]);
    for (size_t i = 0; i < planned->images.size(); ++i) {
      const model::affine::Image blockImage = model::affine::blockImages(planned->images[i]);
      std::vector<uint8_t> cpu(blockImage.bytes);
      model::affine::writeAffineImage(cpu, blockImage);
      for (const auto &section : blockImage.sections)
        if (section.kind == model::affine::SectionKind::Block &&
            !model::affine::blockStoredPlanes(section))
          throw std::runtime_error("the fixture section did not take the stored-planes path");
      metal::MetalBuffer buffer =
          backend.allocateBuffer(blockImage.bytes, metal::BufferStorage::Shared, "check");
      model::affine::writeAffineImage(backend, buffer, blockImage);
      const std::span<const uint8_t> gpu{static_cast<const uint8_t *>(buffer.contents()),
                                         buffer.sizeBytes()};
      if (gpu.size() != cpu.size() || std::memcmp(gpu.data(), cpu.data(), cpu.size())) {
        size_t at = 0;
        while (at < cpu.size() && gpu[at] == cpu[at]) ++at;
        throw std::runtime_error("GPU stored-planes image differs from the CPU image at byte " +
                                 std::to_string(at));
      }
      std::cout << blockImage.name << ": " << blockImage.bytes << " bytes identical\n";
    }
    std::cout << "affine block GPU stored-planes: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

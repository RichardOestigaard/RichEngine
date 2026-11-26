#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "TestGguf.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufTrellis.hpp"
#include "model/TrellisCodec.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <cstring>
#include <cmath>

namespace {
namespace gguf = richengine::test::gguf;
using richengine::test::require;
using richengine::model::ggml::kF32;

constexpr uint32_t kQ4_K = 12;
static_assert(std::string_view(richengine::model::ggmlTypeTraits(kQ4_K)->name) == "Q4_K");
constexpr uint64_t kQ4KBlockBytes = 144;

// A file of one tensor, "weight", with one Q4_K block of data.
gguf::Bytes model(const std::vector<uint64_t> &dims, uint64_t displacement = 0, uint32_t type = kQ4_K) {
  return gguf::file({}, {{"weight", dims, type, gguf::Bytes(kQ4KBlockBytes), displacement}});
}

// A file whose only key, "unused", is an array: its element type and count,
// then the encoded elements.
gguf::Bytes array(const gguf::Bytes &value) { return gguf::file({{"unused", gguf::kArray, value}}, {}); }

// An array that declares count uint64 elements and holds none.
gguf::Bytes emptyUint64Array(uint64_t count) {
  gguf::Bytes value;
  gguf::append(value, gguf::kUint64);
  gguf::append(value, count);
  return array(value);
}
} // namespace

int main() {
  try {
    const richengine::test::TemporaryDirectory directory("richengine-gguf-file");
    const auto path = directory.path() / "test.gguf";
    // Parsing data must fail with the expected error.
    const auto rejects = [&](const char *name, const gguf::Bytes &data, std::string_view expected) {
      richengine::test::writeFile(path, data);
      richengine::test::rejects([&] {
        richengine::model::WeightSource source(path);
        richengine::model::GgufFile file(source);
      }, expected, std::string("accepted invalid GGUF: ") + name);
    };
    richengine::test::writeFile(path, gguf::file({gguf::stringKey("general.architecture", "fixture"),
                                              gguf::uint32Key("fixture.block_count", 2)},
                                             {{"weight", {256, 1}, kQ4_K, gguf::Bytes(kQ4KBlockBytes)}}));
    richengine::model::WeightSource source(path);
    richengine::model::GgufFile valid(source);
    require(valid.architecture() == "fixture" && valid.unsignedValue("fixture.block_count") == 2,
            "metadata values changed");
    const auto &weight = valid.require("weight");
    require(weight.bytes == kQ4KBlockBytes && weight.elements() == 256 && weight.rows() == 1,
            "valid Q4_K shape/size changed");
    require(source.dataOffset() + weight.offset + weight.bytes == source.bytes(),
            "valid data must end exactly at EOF");

    constexpr std::string_view overflow = "GGUF size overflows", pastEnd = "runs past the end of the file";
    rejects("shape overflow", model({256, uint64_t{1} << 60}), overflow);
    rejects("row product overflow", model({256, uint64_t{1} << 63, 2}), overflow);
    rejects("byte size overflow", model({uint64_t{1} << 62}, 0, kF32), overflow);
    rejects("offset addition overflow", model({256, 1}, std::numeric_limits<uint64_t>::max() - 31), pastEnd);
    rejects("offset past EOF", model({256, 1}, 160), pastEnd);
    rejects("zero dimension", model({256, 0}), "zero tensor dimension");
    rejects("unaligned block", model({255, 1}), "not block aligned");
    rejects("unaligned offset", model({256, 1}, 1), "tensor data is misaligned");
    auto truncated = model({256, 1});
    truncated.pop_back();
    rejects("truncated tensor", truncated, pastEnd);
    const gguf::Tensor block{"weight", {256, 1}, kQ4_K, gguf::Bytes(kQ4KBlockBytes)};
    rejects("duplicate name", gguf::file({}, {block, block}), "duplicate GGUF tensor: weight");
    // Whatever the types of the two values, and for an array too.
    rejects("duplicate key",
            gguf::file({gguf::uint32Key("fixture.block_count", 2), gguf::stringKey("fixture.block_count", "2")}, {}),
            "duplicate GGUF metadata key: fixture.block_count");
    rejects("duplicate array key",
            gguf::file({gguf::int32ArrayKey("fixture.widths", {1}), gguf::int32ArrayKey("fixture.widths", {2})}, {}),
            "duplicate GGUF metadata key: fixture.widths");
    rejects("overflowing array size", emptyUint64Array(uint64_t{1} << 62), overflow);
    rejects("truncated array", emptyUint64Array(100), "GGUF header is truncated");
    gguf::Bytes nested;
    for (int i = 0; i < 18; ++i) {
      gguf::append(nested, gguf::kArray);
      gguf::append<uint64_t>(nested, 1);
    }
    rejects("excessive nesting", array(nested), "nesting is too deep");

    // TRL type sizing and the exact trellis metadata subset.
    {
      using richengine::model::ggmlTrellisBits;
      require(ggmlTrellisBits(45) == 2 && ggmlTrellisBits(49) == 6 && !ggmlTrellisBits(44 - 1),
              "trellis bit lookup changed");
      constexpr uint32_t trl2 = 45, f16 = 1, i32 = 26;
      gguf::Bytes weight(16 * 32 / 8 * 2), scales(2 * 2), pre(32 * 4), post(16 * 4), pin(32 * 4), pout(16 * 4);
      for (uint32_t i = 0; i < weight.size(); ++i) weight[i] = uint8_t(17 * i + 3);
      for (uint32_t i = 0; i < 32; ++i) {
        const int32_t v = static_cast<int32_t>(i);
        std::memcpy(pin.data() + 4 * i, &v, 4);
      }
      for (uint32_t i = 0; i < 16; ++i) {
        const int32_t v = static_cast<int32_t>(15 - i);
        std::memcpy(pout.data() + 4 * i, &v, 4);
      }
      const gguf::Bytes data = gguf::file(
          {gguf::uint32Key("trellis.version", 1), gguf::int32ArrayKey("trellis.desc.weight", {0, 0, 1, 0, 0, 1, 1, 16})},
          {{"weight", {32, 16}, trl2, weight},
           {"weight.trellis_s16", {2}, f16, scales},
           {"weight.trellis_pre", {32}, kF32, pre},
           {"weight.trellis_post", {16}, kF32, post},
           {"weight.trellis_pin", {32}, i32, pin},
           {"weight.trellis_pout", {16}, i32, pout}});
      richengine::test::writeFile(path, data);
      richengine::model::WeightSource trellisSource(path);
      richengine::model::GgufFile trellisFile(trellisSource);
      const auto info = richengine::model::readTrellisTensor(trellisFile, "weight");
      require(info.bits() == 2 && info.inputSize() == 32 && info.outputSize() == 16 && info.pin && info.pout,
              "trellis tensor metadata changed");
      require(trellisFile.require("weight").bytes == weight.size(), "TRL2 tensor sizing changed");
    }

    // Host reference invariants used later by the Metal kernel tests.
    {
      namespace trl = richengine::model::trellis;
      constexpr uint32_t bits = 3, columns = 32, rows = 16;
      std::vector<uint32_t> codes(2 * 8 * bits);
      for (uint32_t i = 0; i < codes.size(); ++i) codes[i] = 0x9e3779b9u * (i + 1);
      uint16_t scaleBits[2];
      for (uint32_t i = 0; i < 2; ++i) {
        const __fp16 one = static_cast<__fp16>(1.0f);
        std::memcpy(scaleBits + i, &one, 2);
      }
      std::vector<float> decoded(rows * columns);
      trl::decodeRows(codes.data(), bits, columns, 0, rows, scaleBits, decoded.data());
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < columns; ++c) {
          const uint32_t tile = (r / 16) * (columns / 16) + c / 16;
          const int expected = trl::codedValue(codes.data() + tile * 8 * bits, bits, trl::position(r % 16, c % 16));
          require(decoded[r * columns + c] == float(expected), "trellis tile decode mismatch");
        }

      std::vector<float> wave(32), original(32);
      for (uint32_t i = 0; i < wave.size(); ++i) wave[i] = original[i] = std::sin(float(i) * 0.37f);
      trl::blockTransform(wave.data(), 32, 32);
      trl::blockTransform(wave.data(), 32, 32);
      for (uint32_t i = 0; i < wave.size(); ++i)
        require(std::fabs(wave[i] - original[i]) < 1e-5f, "normalized block Hadamard is not involutive");

      std::vector<int32_t> pout = {2, 0, 3, 1};
      const auto inverse = trl::inversePermutation(pout);
      require((inverse == std::vector<int32_t>{1, 3, 0, 2}), "inverse permutation mismatch");
      const auto composed = trl::composeOutputPermutation({}, 8, 4, 1, 2, 2);
      require((composed == std::vector<int32_t>{0, 1, 2, 3, 4, 6, 5, 7}), "grouped output permutation mismatch");
    }
    std::cout << "GGUF parser bounds tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

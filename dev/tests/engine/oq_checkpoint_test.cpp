// The quantization metadata of an oMLX checkpoint (SafetensorsCheckpoint):
// a module's spec resolves its per-module entry, else the global one, and a
// module keeping no scales is unquantized whatever the dict's global
// entry says. hasMixedQuantization names the configurations the affine path
// cannot plan; oqCheckpoint probes config.json without the shards.

#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/OqTarget.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace richengine::model;
using richengine::test::rejects;
using richengine::test::require;
namespace {

// One shard holding each module's weight/scales/biases and the plain
// module's weight alone, 16 bytes per tensor.
void writeShard(const std::filesystem::path &path) {
  const char *names[] = {"a.weight", "a.scales", "a.biases", "b.weight", "b.scales", "b.biases",
                         "plain.weight", "model.embed_tokens.weight", "model.embed_tokens.scales",
                         "model.embed_tokens.biases"};
  std::string header = "{";
  uint64_t offset = 0;
  for (const char *name : names) {
    header += "\"" + std::string(name) +
              "\":{\"dtype\":\"U32\",\"shape\":[2,2],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + 16) + "]},";
    offset += 16;
  }
  header.back() = '}';
  std::vector<uint8_t> file(sizeof(uint64_t));
  const uint64_t length = header.size();
  std::memcpy(file.data(), &length, sizeof length);
  file.insert(file.end(), header.begin(), header.end());
  for (uint64_t i = 0; i < offset; ++i) file.push_back(uint8_t(i));
  richengine::test::writeFile(path, file);
}

void config(const std::filesystem::path &root, std::string_view json) {
  richengine::test::writeFile(root / "config.json", std::string(json));
}

using QuantSpec = SafetensorsCheckpoint::QuantSpec;

} // namespace

int main() {
  try {
    const richengine::test::TemporaryDirectory directory("richengine-oq-checkpoint");
    const std::filesystem::path &root = directory.path();
    writeShard(root / "model.safetensors");

    // The uniform global spec the affine path plans.
    config(root, R"({"quantization":{"bits":4,"group_size":64,"mode":"affine"}})");
    {
      SafetensorsCheckpoint source(root);
      require(!source.hasMixedQuantization() && !SafetensorsCheckpoint::hasMixedQuantization(root) &&
                  !oqCheckpoint(root),
              "a uniform Q4 group-64 affine config probed as mixed");
      require(source.quantization("a") == QuantSpec{4, 64, "affine"}, "the global spec did not resolve");
      require(!source.quantization("plain"), "a module without scales resolved as quantized");
    }

    // Per-module entries mixed with the global's.
    config(root, R"({"quantization":{"bits":4,"group_size":64,"mode":"affine",)"
              R"("a":{"bits":5},"model.embed_tokens":{"bits":6,"group_size":64}}})");
    {
      SafetensorsCheckpoint source(root);
      require(source.hasMixedQuantization() && SafetensorsCheckpoint::hasMixedQuantization(root) &&
                  oqCheckpoint(root),
              "per-module entries did not probe as mixed");
      require(source.quantization("a") == QuantSpec{5, 64, "affine"},
              "a per-module entry did not resolve or inherit");
      require(source.quantization("b") == QuantSpec{4, 64, "affine"},
              "a module without an entry did not take the global's");
      // The "language_model." prefix fallback find() has.
      require(source.quantization("language_model.model.embed_tokens") == QuantSpec{6, 64, "affine"},
              "the prefixed module's entry did not resolve");
      require(!source.quantization("language_model.plain") && !source.quantization("plain"),
              "a plain module resolved as quantized");
    }

    // A differing global alone is mixed too.
    config(root, R"({"quantization":{"bits":8,"group_size":128}})");
    {
      SafetensorsCheckpoint source(root);
      require(source.hasMixedQuantization() && oqCheckpoint(root), "a global 8/128 spec did not probe as mixed");
      require(source.quantization("b") == QuantSpec{8, 128, "affine"}, "the 8/128 global did not resolve");
    }

    // No quantization at all: nothing is declared quantized.
    config(root, R"({})");
    {
      SafetensorsCheckpoint source(root);
      require(!source.hasMixedQuantization() && !oqCheckpoint(root), "a config without quantization probed as mixed");
      require(!source.quantization("a"), "a checkpoint without a dict resolved a spec");
    }

    // A per-module spec outside the oQ set still resolves; the oQ resolver
    // (AffineTarget.cpp) is the one that rejects it.
    config(root, R"({"quantization":{"a":{"bits":2,"group_size":32}}})");
    {
      SafetensorsCheckpoint source(root);
      require(source.hasMixedQuantization() && source.quantization("a") == QuantSpec{2, 32, "affine"},
              "the declared spec did not resolve verbatim");
    }

    // The probe reports no checkpoint for a directory without a readable
    // config.json (the loader that runs instead reports it).
    const std::filesystem::path empty = root / "empty";
    std::filesystem::create_directory(empty);
    require(!oqCheckpoint(empty), "an unreadable config probed as an oQ checkpoint");
    rejects([&] { static_cast<void>(SafetensorsCheckpoint::hasMixedQuantization(empty)); },
            "cannot read", "a missing config was accepted");

    std::cout << "oq checkpoint: per-module specs, prefix fallback, unquantized modules, probes PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

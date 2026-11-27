// Phase-zero mixed-row GEMM ceiling probe for affine Q4 projections.
//
//   mixed-gemm-bench <metallib> [rows] [decode-rows] [rounds]
//
// rows is a comma-separated prefill-row list (default 128,512); decode-rows a
// comma-separated verify-row list (default 16,32). For every Ornith-1.5-9B
// projection shape the probe times three productions plans on DRAM-cold weight
// copies: decode rows alone, prefill rows alone, and one row space holding
// prefill+decode rows. The last case is not a correctness implementation: it
// is the representative unified-GEMM ceiling, since every projection row of
// either phase performs the same W x activation operation.
#include "tests/engine/AffineQ4Fixture.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace richengine;
using namespace richengine::ops;
using richengine::metal::BufferStorage;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;

namespace {

constexpr uint64_t kRingBytes = 384ull << 20;
constexpr uint32_t kMinimumCopies = 3, kMaximumCopies = 32;
constexpr double kWarmupMilliseconds = 100.0;

struct Shape final {
  const char *label;
  uint32_t output;
  uint32_t input;
  LinearEpilogue epilogue;
};

struct Case final {
  std::string label;
  uint32_t rows = 0;
  LinearPlan plan;
  std::vector<double> milliseconds;
};

std::vector<uint32_t> rowList(const char *text) {
  std::vector<uint32_t> result;
  std::stringstream stream(text);
  for (std::string part; std::getline(stream, part, ',');) {
    const uint32_t rows = uint32_t(std::stoul(part));
    if (!rows || rows > RICHENGINE_PREFILL_TOKEN_BUDGET)
      throw std::invalid_argument("row counts are 1..2048");
    result.push_back(rows);
  }
  if (result.empty()) throw std::invalid_argument("empty row list");
  return result;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *label) {
  if (!bytes) return {};
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

uint64_t affineBytes(LinearMatrix matrix) {
  const uint64_t parameters = uint64_t{matrix.outputSize} * (matrix.inputSize / 64);
  return parameters * 36;
}

void profileShape(MetalBackend &backend, const Shape &shape,
                  std::span<const uint32_t> prefillRows,
                  std::span<const uint32_t> decodeRows, uint32_t rounds) {
  const LinearMatrix matrix{shape.output, shape.input};
  const uint64_t bytes = affineBytes(matrix);
  const uint32_t copies = uint32_t(std::clamp<uint64_t>(
      (kRingBytes + bytes - 1) / bytes, kMinimumCopies, kMaximumCopies));
  std::vector<Projection> ring;
  ring.reserve(copies);
  for (uint32_t copy = 0; copy < copies; ++copy)
    ring.push_back(test::deterministicQ4Projection(backend, matrix, 917 + copy));

  const Linear linear(backend.capabilities());
  std::vector<Case> cases;
  for (const uint32_t decode : decodeRows) {
    if (decode % 8 || decode > RICHENGINE_TARGET_VERIFY_ROWS * RICHENGINE_MAXIMUM_BATCH_WIDTH)
      throw std::invalid_argument("decode rows are a multiple of 8 up to the batch tile");
    const LinearWorkload workload{matrix, decode, LinearPhase::Decode, shape.epilogue};
    cases.push_back({"D" + std::to_string(decode), decode,
                     linear.plan(workload, ring.front()), {}});
  }
  for (const uint32_t prefill : prefillRows) {
    const LinearWorkload workload{matrix, prefill, LinearPhase::Prefill, shape.epilogue};
    cases.push_back({"P" + std::to_string(prefill), prefill,
                     linear.plan(workload, ring.front()), {}});
    for (const uint32_t decode : decodeRows) {
      const uint32_t mixed = prefill + decode;
      const LinearWorkload mixedWorkload{matrix, mixed, LinearPhase::Prefill, shape.epilogue};
      cases.push_back({"M" + std::to_string(prefill) + "+D" + std::to_string(decode),
                       mixed, linear.plan(mixedWorkload, ring.front()), {}});
    }
  }

  LinearScratchSize scratchSize;
  uint32_t storageRows = 0;
  uint64_t sumsBytes = 0;
  for (const Case &c : cases) {
    scratchSize.include(c.plan.scratchSize());
    storageRows = std::max(storageRows, c.plan.storageRows());
    sumsBytes = std::max(sumsBytes, c.plan.sumsBytes());
  }
  MetalBuffer input = zeros(backend, uint64_t{storageRows} * matrix.inputSize * 2,
                            "mixed-gemm input");
  MetalBuffer output = zeros(backend, uint64_t{storageRows} * matrix.outputSize * 2,
                             "mixed-gemm output");
  MetalBuffer residual = zeros(backend, uint64_t{storageRows} * matrix.outputSize * 2,
                               "mixed-gemm residual");
  MetalBuffer sums = zeros(backend, sumsBytes, "mixed-gemm sums");
  const LinearScratch scratch{zeros(backend, scratchSize.input, "mixed-gemm table"),
                              zeros(backend, scratchSize.sums, "mixed-gemm table sums"),
                              zeros(backend, scratchSize.partials, "mixed-gemm partials"),
                              zeros(backend, scratchSize.counters, "mixed-gemm counters")};
  // Fill input and residual with deterministic finite bf16 values. Weight
  // values are irrelevant to timing, but NaNs would make clock throttling
  // measurements hard to reproduce.
  {
    std::mt19937 random(shape.output * 31 + shape.input);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    auto *x = static_cast<uint16_t *>(input.contents());
    for (uint64_t index = 0; index < uint64_t{storageRows} * matrix.inputSize; ++index)
      x[index] = ops::tuning::floatToBf16(unit(random));
    auto *r = static_cast<uint16_t *>(residual.contents());
    for (uint64_t index = 0; index < uint64_t{storageRows} * matrix.outputSize; ++index)
      r[index] = ops::tuning::floatToBf16(unit(random));
  }

  uint32_t next = 0;
  const auto time = [&](Case &c) {
    CommandGraph graph;
    LinearBuffers buffers{.input = input, .output = output, .sums = sums,
                          .residual = shape.epilogue == LinearEpilogue::Residual ? residual : MetalBuffer{},
                          .scratch = scratch};
    for (uint32_t copy = 0; copy < copies; ++copy, ++next)
      static_cast<void>(linear.add(graph, buffers, ring[next % copies], c.plan));
    return backend.submitCommand(graph.dispatches()).gpuSeconds * 1e3 / copies;
  };
  for (Case &c : cases)
    for (double spent = 0.0; spent < kWarmupMilliseconds;) spent += time(c) * copies;
  for (uint32_t round = 0; round < rounds; ++round)
    for (size_t index = 0; index < cases.size(); ++index) {
      Case &c = cases[round % 2 ? cases.size() - 1 - index : index];
      c.milliseconds.push_back(time(c));
    }

  std::printf("%s %ux%u epilogue=%s copies=%u weightMiB=%.1f\n", shape.label,
              matrix.outputSize, matrix.inputSize,
              shape.epilogue == LinearEpilogue::Residual ? "residual" : "none",
              copies, bytes / double(1 << 20));
  for (const uint32_t decode : decodeRows) {
    for (const uint32_t prefill : prefillRows) {
      const auto find = [&](const std::string &label) -> const Case & {
        for (const Case &c : cases)
          if (c.label == label) return c;
        throw std::logic_error("missing mixed-gemm case");
      };
      const Case &d = find("D" + std::to_string(decode));
      const Case &p = find("P" + std::to_string(prefill));
      const Case &m = find("M" + std::to_string(prefill) + "+D" + std::to_string(decode));
      const double decodeMs = median(d.milliseconds);
      const double prefillMs = median(p.milliseconds);
      const double mixedMs = median(m.milliseconds);
      const double separate = decodeMs + prefillMs;
      std::printf("  P%u+D%u: decode %.4f ms, prefill %.4f ms, separate %.4f ms, "
                  "mixed-rows %.4f ms, speedup %.2fx, decode marginal %.4f ms, "
                  "prefill marginal %.4f ms\n",
                  prefill, decode, decodeMs, prefillMs, separate, mixedMs,
                  separate / mixedMs, mixedMs - prefillMs, mixedMs - decodeMs);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2 || argc > 5) {
      std::cerr << "usage: mixed-gemm-bench <metallib> [rows] [decode-rows] [rounds]\n";
      return 2;
    }
    try {
      const std::vector<uint32_t> prefillRows = rowList(argc > 2 ? argv[2] : "128,512");
      const std::vector<uint32_t> decodeRows = rowList(argc > 3 ? argv[3] : "16,32");
      const uint32_t rounds = argc > 4 ? uint32_t(std::stoul(argv[4])) : 7;
      if (!rounds) throw std::invalid_argument("rounds must be positive");
      MetalBackend backend(argv[1]);
      const DeviceCapabilities &device = backend.capabilities();
      std::printf("%s (GPU family %u, %u cores): affine Q4 mixed-row ceiling probe\n",
                  device.deviceName.c_str(), device.appleGpuFamily, device.gpuCoreCount);
      // Ornith-1.5-9B text layers: 24 GDN mixers and eight full-attention
      // mixers, each followed by the dense SwiGLU pair. Gate and up are one
      // weight stream each; this probe times one of them as a plain matrix.
      const Shape shapes[] = {
          {"gdn_input", 13'312, 4'096, LinearEpilogue::None},
          {"attention_input", 10'240, 4'096, LinearEpilogue::None},
          {"mixer_output", 4'096, 4'096, LinearEpilogue::Residual},
          {"ffn_up_one_side", 12'288, 4'096, LinearEpilogue::None},
          {"ffn_down", 4'096, 12'288, LinearEpilogue::Residual},
      };
      for (const Shape &shape : shapes)
        profileShape(backend, shape, prefillRows, decodeRows, rounds);
    } catch (const std::exception &error) {
      std::cerr << "mixed-gemm-bench: " << error.what() << '\n';
      return 1;
    }
    return 0;
  }
}

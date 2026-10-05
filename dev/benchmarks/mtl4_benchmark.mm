// A/B benchmark of the Metal 3 and Metal 4 submission paths.
//
//   mtl4-benchmark METALLIB [dispatches] [rounds]
//
// Submits a decode-shaped command graph (many small compute dispatches with
// buffer and byte bindings) through MetalBackend::submitCommandAsync and
// reports per-submission encode+commit wall time, end-to-end wall time and
// GPU time. Run once normally and once with SPLASH_MTL4=1 to compare the
// encoders. Two graphs are measured: "direct" dispatches encode every
// dispatch, while "icb" dispatches all carry bakeable so the baked-span
// replay path covers them after the first submission.
//
// The kernels come from dev/tests/engine/metal_backend_test.metal:
// test_add_u32 adds to a device buffer; test_copy_u32 copies between two.

#include "metal/MetalBackend.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace splash;
using Clock = std::chrono::steady_clock;

namespace {

double milliseconds(Clock::duration value) {
  return std::chrono::duration<double, std::milli>(value).count();
}

struct RunStats final {
  double submitMs = 0.0;  // submitCommandAsync: prepare + encode + commit.
  double waitMs = 0.0;    // ticket wait, GPU execution mostly.
  double gpuMs = 0.0;     // driver-reported GPU seconds.
};

metal::ComputeDispatch addDispatch(metal::MetalBuffer values, uint32_t count,
                                   const uint32_t *increment, bool bakeable) {
  metal::ComputeDispatch dispatch;
  dispatch.pipelineName = "test_add_u32";
  dispatch.buffers.push_back({0, values});
  dispatch.bytes.push_back({1, &count, sizeof(count)});
  dispatch.bytes.push_back({2, increment, sizeof(*increment)});
  dispatch.threadgroups = {1, 1, 1};
  dispatch.threadsPerThreadgroup = {64, 1, 1};
  dispatch.bakeable = bakeable;
  return dispatch;
}

metal::ComputeDispatch copyDispatch(metal::MetalBuffer source,
                                    metal::MetalBuffer destination,
                                    uint32_t count, bool bakeable) {
  metal::ComputeDispatch dispatch;
  dispatch.pipelineName = "test_copy_u32";
  dispatch.buffers.push_back({0, source});
  dispatch.buffers.push_back({1, destination});
  dispatch.bytes.push_back({2, &count, sizeof(count)});
  dispatch.threadgroups = {1, 1, 1};
  dispatch.threadsPerThreadgroup = {64, 1, 1};
  dispatch.bakeable = bakeable;
  return dispatch;
}

RunStats runGraph(metal::MetalBackend &backend,
                  std::span<const metal::ComputeDispatch> graph, int rounds,
                  uint64_t &checksumOut) {
  RunStats stats;
  for (int round = 0; round < rounds; ++round) {
    const auto submitStart = Clock::now();
    auto ticket = backend.submitCommandAsync(graph);
    const auto waitStart = Clock::now();
    const metal::CommandTiming timing = ticket.wait();
    const auto end = Clock::now();
    stats.submitMs += milliseconds(waitStart - submitStart);
    stats.waitMs += milliseconds(end - waitStart);
    stats.gpuMs += timing.gpuSeconds * 1e3;
    if (round == rounds - 1) {
      // A sanity checksum: identical values must result on both paths.
      const auto *words = static_cast<const uint32_t *>(
          graph.front().buffers.front().buffer.contents());
      checksumOut = words ? words[0] : 0;
    }
  }
  stats.submitMs /= rounds;
  stats.waitMs /= rounds;
  stats.gpuMs /= rounds;
  return stats;
}

void report(const char *name, const RunStats &stats, size_t dispatches,
            uint64_t checksum) {
  const double totalMs = stats.submitMs + stats.waitMs;
  std::printf("%-8s %4zu dispatches: encode+commit %8.1f us, wait %8.1f us, "
              "gpu %8.1f us, submission %8.1f us (%6.0f/s) checksum %llu\n",
              name, dispatches, stats.submitMs * 1e3, stats.waitMs * 1e3,
              stats.gpuMs * 1e3, totalMs * 1e3,
              totalMs > 0.0 ? 1e3 / totalMs : 0.0,
              static_cast<unsigned long long>(checksum));
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s METALLIB [dispatches] [rounds]\n", argv[0]);
    return 2;
  }
  const size_t dispatches = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 200;
  const int rounds = argc > 3 ? std::atoi(argv[3]) : 40;

  try {
    metal::MetalBackend backend(argv[1]);
    const uint32_t count = 64;
    const uint32_t increment = 1;
    metal::MetalBuffer values =
        backend.allocateBuffer(count * sizeof(uint32_t),
                               metal::BufferStorage::Shared, "mtl4-values");
    metal::MetalBuffer source =
        backend.allocateBuffer(count * sizeof(uint32_t),
                               metal::BufferStorage::Shared, "mtl4-source");
    metal::MetalBuffer scratch =
        backend.allocateBuffer(count * sizeof(uint32_t),
                               metal::BufferStorage::Shared, "mtl4-scratch");

    std::vector<metal::ComputeDispatch> direct, icb;
    direct.reserve(dispatches);
    icb.reserve(dispatches);
    for (size_t i = 0; i < dispatches; ++i) {
      const bool add = (i & 1) == 0;
      direct.push_back(add ? addDispatch(values, count, &increment, false)
                           : copyDispatch(source, scratch, count, false));
      icb.push_back(add ? addDispatch(values, count, &increment, true)
                        : copyDispatch(source, scratch, count, true));
    }
    // Bake before measuring so both runs cost one cached span replay.
    backend.preparePipelines(icb);
    [[maybe_unused]] const metal::CommandTiming warmup =
        backend.submitCommand(icb);

    uint64_t checksum = 0;
    const RunStats directStats =
        runGraph(backend, direct, rounds, checksum);
    report("direct", directStats, direct.size(), checksum);
    const RunStats icbStats = runGraph(backend, icb, rounds, checksum);
    report("icb", icbStats, icb.size(), checksum);

    std::printf("path: %s (SPLASH_MTL4 %s)\n",
                std::getenv("SPLASH_MTL4") ? "mtl4" : "mtl3",
                std::getenv("SPLASH_MTL4") ? "set" : "unset");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "mtl4-benchmark: %s\n", error.what());
    return 1;
  }
  return 0;
}

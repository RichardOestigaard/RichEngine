// A/B benchmark of the Metal 3 and Metal 4 submission paths.
//
//   mtl4-benchmark METALLIB [dispatches] [rounds] [--emit|--compare]
//
// Submits a decode-shaped command graph (many small compute dispatches with
// buffer and byte bindings) through MetalBackend::submitCommandAsync and
// reports per-submission encode+commit wall time, end-to-end wall time and
// GPU time. Three graphs are measured: "direct" dispatches encode every
// dispatch, "icb" dispatches all carry bakeable so the baked-span replay
// path covers them after the first submission, and "gpuspan" dispatches
// carry gpuSpan so the GPU-encoded span path covers them when
// RICHENGINE_GPU_SPANS is set.
//
// --compare (the default without it is a single-path run on the current
// environment) re-executes the binary twice, once with RICHENGINE_MTL4
// removed and once with it set, parses each run's --emit stats and prints
// the two paths side by side with the Metal 4 delta.
//
// The kernels come from dev/tests/engine/metal_backend_test.metal:
// test_add_u32 adds to a device buffer; test_copy_u32 copies between two.

#include "metal/MetalBackend.hpp"

#import <Foundation/Foundation.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace richengine;
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
                                   const uint32_t *increment, int span) {
  metal::ComputeDispatch dispatch;
  dispatch.pipelineName = "test_add_u32";
  dispatch.buffers.push_back({0, values});
  dispatch.bytes.push_back({1, &count, sizeof(count)});
  dispatch.bytes.push_back({2, increment, sizeof(*increment)});
  dispatch.threadgroups = {1, 1, 1};
  dispatch.threadsPerThreadgroup = {64, 1, 1};
  dispatch.bakeable = span == 1;
  dispatch.gpuSpan = span == 2;
  return dispatch;
}

metal::ComputeDispatch copyDispatch(metal::MetalBuffer source,
                                    metal::MetalBuffer destination,
                                    uint32_t count, int span) {
  metal::ComputeDispatch dispatch;
  dispatch.pipelineName = "test_copy_u32";
  dispatch.buffers.push_back({0, source});
  dispatch.buffers.push_back({1, destination});
  dispatch.bytes.push_back({2, &count, sizeof(count)});
  dispatch.threadgroups = {1, 1, 1};
  dispatch.threadsPerThreadgroup = {64, 1, 1};
  dispatch.bakeable = span == 1;
  dispatch.gpuSpan = span == 2;
  return dispatch;
}

struct GraphResult final {
  RunStats stats;
  uint64_t checksum = 0;
};

GraphResult runGraph(metal::MetalBackend &backend,
                     std::span<const metal::ComputeDispatch> graph,
                     int rounds) {
  GraphResult result;
  for (int round = 0; round < rounds; ++round) {
    const auto submitStart = Clock::now();
    auto ticket = backend.submitCommandAsync(graph);
    const auto waitStart = Clock::now();
    const metal::CommandTiming timing = ticket.wait();
    const auto end = Clock::now();
    result.stats.submitMs += milliseconds(waitStart - submitStart);
    result.stats.waitMs += milliseconds(end - waitStart);
    result.stats.gpuMs += timing.gpuSeconds * 1e3;
    if (round == rounds - 1) {
      // A sanity checksum: identical values must result on both paths.
      const auto *words = static_cast<const uint32_t *>(
          graph.front().buffers.front().buffer.contents());
      result.checksum = words ? words[0] : 0;
    }
  }
  result.stats.submitMs /= rounds;
  result.stats.waitMs /= rounds;
  result.stats.gpuMs /= rounds;
  return result;
}

struct NamedResult {
  const char *name;
  GraphResult result;
};

// One full measurement pass over the three graphs; emit prints the
// machine-readable STATS lines --compare parses.
std::vector<NamedResult> measure(const char *metallib, size_t dispatches,
                                 int rounds) {
  metal::MetalBackend backend(metallib);
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

  std::vector<metal::ComputeDispatch> direct, icb, gpuspan;
  direct.reserve(dispatches);
  icb.reserve(dispatches);
  gpuspan.reserve(dispatches);
  for (size_t i = 0; i < dispatches; ++i) {
    const bool add = (i & 1) == 0;
    direct.push_back(add ? addDispatch(values, count, &increment, 0)
                         : copyDispatch(source, scratch, count, 0));
    icb.push_back(add ? addDispatch(values, count, &increment, 1)
                      : copyDispatch(source, scratch, count, 1));
    gpuspan.push_back(add ? addDispatch(values, count, &increment, 2)
                          : copyDispatch(source, scratch, count, 2));
  }
  // Bake before measuring so both runs cost one cached span replay. The
  // gpu-span ICBs write on their first replay through gpu_encode.
  backend.preparePipelines(icb);
  backend.preparePipelines(gpuspan);
  [[maybe_unused]] const metal::CommandTiming warmup =
      backend.submitCommand(icb);
  static_cast<void>(backend.submitCommand(gpuspan));

  return {{"direct", runGraph(backend, direct, rounds)},
          {"icb", runGraph(backend, icb, rounds)},
          {"gpuspan", runGraph(backend, gpuspan, rounds)}};
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

// --emit writes one parseable line per graph for the --compare parent.
void emit(const NamedResult &row, size_t dispatches) {
  const RunStats &s = row.result.stats;
  std::printf("STATS %s %zu %.3f %.3f %.3f %llu\n", row.name, dispatches,
              s.submitMs * 1e3, s.waitMs * 1e3, s.gpuMs * 1e3,
              static_cast<unsigned long long>(row.result.checksum));
}

// Runs this binary once under the given RICHENGINE_MTL4 setting and parses
// its STATS lines back.
std::vector<NamedResult> runChild(const char *self, const char *metallib,
                                  size_t dispatches, int rounds,
                                  bool mtl4) {
  int out[2];
  if (pipe(out) != 0) throw std::runtime_error("pipe failed");
  const pid_t child = fork();
  if (child == 0) {
    close(out[0]);
    dup2(out[1], STDOUT_FILENO);
    close(out[1]);
    std::string dispatchCount = std::to_string(dispatches);
    std::string roundCount = std::to_string(rounds);
    setenv("RICHENGINE_GPU_SPANS", "1", 1);
    unsetenv("RICHENGINE_MTL4");
    if (mtl4) {
      unsetenv("RICHENGINE_MTL4_OFF");
    } else {
      setenv("RICHENGINE_MTL4_OFF", "1", 1);
    }
    execlp(self, self, metallib, dispatchCount.c_str(), roundCount.c_str(),
           "--emit", nullptr);
    _exit(127);
  }
  close(out[1]);
  std::string text;
  char buffer[4096];
  for (;;) {
    const ssize_t got = read(out[0], buffer, sizeof(buffer));
    if (got <= 0) break;
    text.append(buffer, static_cast<size_t>(got));
  }
  close(out[0]);
  int status = 0;
  waitpid(child, &status, 0);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error(std::string("child run failed: ") +
                             text.substr(0, 2000));

  std::vector<NamedResult> rows;
  size_t at = 0;
  while (at < text.size()) {
    const size_t eol = text.find('\n', at);
    const std::string line =
        text.substr(at, eol == std::string::npos ? eol : eol - at);
    at = eol == std::string::npos ? text.size() : eol + 1;
    if (line.compare(0, 6, "STATS ") != 0) continue;
    char name[32] = {};
    size_t n = 0;
    double submit = 0, wait = 0, gpu = 0;
    unsigned long long checksum = 0;
    if (std::sscanf(line.c_str(), "STATS %31s %zu %lf %lf %lf %llu", name, &n,
                    &submit, &wait, &gpu, &checksum) == 6) {
      RunStats stats;
      stats.submitMs = submit / 1e3;
      stats.waitMs = wait / 1e3;
      stats.gpuMs = gpu / 1e3;
      rows.push_back({strdup(name), {stats, checksum}});
    }
  }
  return rows;
}

int compare(const char *self, const char *metallib, size_t dispatches,
            int rounds) {
  try {
    const std::vector<NamedResult> mtl3 =
        runChild(self, metallib, dispatches, rounds, false);
    const std::vector<NamedResult> mtl4 =
        runChild(self, metallib, dispatches, rounds, true);
    if (mtl3.size() != mtl4.size() || mtl3.empty())
      throw std::runtime_error("child runs produced different graph sets");
    std::printf("%-8s %14s %14s %10s %14s %14s\n", "graph", "mtl3 submit us",
                "mtl4 submit us", "speedup", "mtl3 total us",
                "mtl4 total us");
    bool checksumsAgree = true;
    for (size_t i = 0; i < mtl3.size(); ++i) {
      if (std::strcmp(mtl3[i].name, mtl4[i].name) != 0)
        throw std::runtime_error("graph order mismatch between runs");
      checksumsAgree &=
          mtl3[i].result.checksum == mtl4[i].result.checksum;
      const double submit3 = mtl3[i].result.stats.submitMs;
      const double submit4 = mtl4[i].result.stats.submitMs;
      const double total3 = submit3 + mtl3[i].result.stats.waitMs;
      const double total4 = submit4 + mtl4[i].result.stats.waitMs;
      std::printf("%-8s %14.1f %14.1f %9.2fx %14.1f %14.1f\n", mtl3[i].name,
                  submit3 * 1e3, submit4 * 1e3,
                  submit4 > 0.0 ? submit3 / submit4 : 0.0, total3 * 1e3,
                  total4 * 1e3);
    }
    std::printf("checksums: %s\n", checksumsAgree ? "match" : "MISMATCH");
    return checksumsAgree ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "mtl4-benchmark --compare: %s\n", error.what());
    return 1;
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s METALLIB [dispatches] [rounds] [--emit|--compare]\n",
                 argv[0]);
    return 2;
  }
  const size_t dispatches = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 200;
  const int rounds = argc > 3 ? std::atoi(argv[3]) : 40;
  bool emit = false;
  bool compare = false;
  for (int i = 4; i < argc; ++i) {
    emit |= std::strcmp(argv[i], "--emit") == 0;
    compare |= std::strcmp(argv[i], "--compare") == 0;
  }
  if (compare) return ::compare(argv[0], argv[1], dispatches, rounds);

  try {
    for (const NamedResult &row : measure(argv[1], dispatches, rounds)) {
      if (emit) {
        ::emit(row, dispatches);
      } else {
        report(row.name, row.result.stats, dispatches, row.result.checksum);
      }
    }
    if (!emit)
      std::printf("path: %s (RICHENGINE_MTL4_OFF %s, RICHENGINE_GPU_SPANS %s)\n",
                  std::getenv("RICHENGINE_MTL4_OFF") ? "mtl3" : "mtl4",
                  std::getenv("RICHENGINE_MTL4_OFF") ? "set" : "unset",
                  std::getenv("RICHENGINE_GPU_SPANS") ? "set" : "unset");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "mtl4-benchmark: %s\n", error.what());
    return 1;
  }
  return 0;
}

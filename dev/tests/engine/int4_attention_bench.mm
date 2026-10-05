// Times the packed-INT4 verify attention path of two metallibs A/B: each
// backend builds the same logical case (same codes, scales and queries) but
// fills its pages in that library's value layout — dim-pair-major for the
// unpack+int8 kernels, token-major for the native int4b ones — then runs the
// store + split + reduce verify graph and reports median fused GPU ms plus
// per-pipeline times. usage: int4_attention_bench NEWLIB OLDLIB
// [--history N] [--lanes N] [--repeat N]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/PagedAttention.hpp"
#include "tuning/HostKvExtents.hpp"
#include "DispatchReplay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

using namespace richengine;
using richengine::ops::tuning::HostKvExtents;

namespace {

constexpr uint32_t kLayer = 0;
constexpr uint32_t kKvHeads = 4, kQueryHeads = 24;   // 27b verify shape
constexpr uint32_t kLanes = 4;
constexpr uint32_t kRows = RICHENGINE_TARGET_VERIFY_ROWS;

metal::MetalBuffer alloc(metal::MetalBackend &backend, uint64_t bytes) {
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

uint16_t toBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return uint16_t(bits >> 16);
}

int code(uint32_t token, uint32_t head, uint32_t d, uint32_t lane, bool value) {
  return int((value ? token * 53 + head * 79 + d * 29 + token * d * 5
                    : token * 37 + head * 101 + d * 17 + token * d * 3) +
             lane * 7) % 15 - 7;
}

// One int4 verify case per backend: lanes x kKvHeads, history tokens of
// packed pages in `oldLayout` order, then the verify graph timed `repeat`
// times (median fused GPU ms and median per-pipeline ms).
struct Result {
  double fused = 0;
  std::vector<std::pair<std::string, double>> pipelines;
};

Result run(metal::MetalBackend &backend, uint32_t history, uint32_t lanes,
           uint32_t repeat, bool oldLayout) {
  const kv::Layout layout{1, kKvHeads, 256, kv::Format::Int4};
  const kv::Layout poolLayout{2, kKvHeads, 256, kv::Format::Int4};
  const uint32_t tokens = history + kRows;
  const uint32_t pages = (tokens + 31) / 32;
  const auto geometry = HostKvExtents::aligned(poolLayout, pages * lanes + 4);
  HostKvExtents pool(backend, poolLayout, geometry.extentPages, geometry.extents);
  const auto layer = pool.layer(kLayer);
  std::vector<uint32_t> pageIds(pages * lanes);
  for (uint32_t i = 0; i < pageIds.size(); ++i) pageIds[i] = (2 * i + 1) % pool.pageCount();

  std::array<metal::MetalBuffer, kLanes> tables;
  std::array<kv::ChunkedPrefillParams, kLanes> stores{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    tables[lane] = alloc(backend, pages * sizeof(RichKvPage));
    pool.writeTable(std::span(pageIds).subspan(lane * pages, pages),
                    tables[lane].contents());
    stores[lane] = ops::PagedAttention::verifyParams(history, pages);
    for (uint32_t token = 0; token < history; ++token) {
      const uint32_t id = pageIds[lane * pages + token / 32];
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        const uint64_t slot = richengine_kv_scale_element(head, token % 32);
        pool.slab<float>(kLayer, RICHENGINE_KV_KEY_SCALES, id)[slot] = 0.096f;
        pool.slab<float>(kLayer, RICHENGINE_KV_VALUE_SCALES, id)[slot] = 0.112f;
        for (uint32_t d = 0; d < 256; d += 2) {
          const uint8_t keyByte = uint8_t((code(token, head, d, lane, false) & 0xF) |
                                          (code(token, head, d + 1, lane, false) & 0xF) << 4);
          const uint8_t valueByte = uint8_t((code(token, head, d, lane, true) & 0xF) |
                                            (code(token, head, d + 1, lane, true) & 0xF) << 4);
          const uint64_t keyIndex = richengine_kv_key_element(head, token % 32, d) / 2;
          // The unpack kernels' value order is dim-pair-major; the int4b
          // operand order is token-major like the keys.
          const uint64_t valueIndex =
              oldLayout ? uint64_t(head) * 4096 + (d / 2) * 32 + (token % 32)
                        : richengine_kv_key_element(head, token % 32, d) / 2;
          pool.slab<uint8_t>(kLayer, RICHENGINE_KV_KEYS, id)[keyIndex] = keyByte;
          pool.slab<uint8_t>(kLayer, RICHENGINE_KV_VALUES, id)[valueIndex] = valueByte;
        }
      }
    }
  }

  const auto plan = ops::PagedAttention::verifyPlan(
      lanes, kQueryHeads, layout,
      std::vector<uint32_t>(lanes, history));
  auto queries = alloc(backend, uint64_t(lanes) * kKvHeads * RICHENGINE_VERIFY_CHUNK_STRIDE *
                                    (kQueryHeads / kKvHeads) * 256 * 2);
  auto keys = alloc(backend, uint64_t(lanes) * kKvHeads * RICHENGINE_VERIFY_CHUNK_STRIDE * 256 * 2);
  auto values = alloc(backend, keys.sizeBytes());
  auto partials = alloc(backend, plan.workspace.partialsBytes);
  auto statistics = alloc(backend, plan.workspace.statisticsBytes);
  auto output = alloc(backend, queries.sizeBytes());
  auto *q = static_cast<uint16_t *>(queries.contents());
  auto *ck = static_cast<uint16_t *>(keys.contents());
  auto *cv = static_cast<uint16_t *>(values.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t head = 0; head < kQueryHeads; ++head)
        for (uint32_t d = 0; d < 256; ++d) {
          const uint64_t qi =
              ((uint64_t{lane} * kKvHeads + head / 6) * RICHENGINE_VERIFY_CHUNK_STRIDE + row) * 6 +
              head % 6;
          q[qi * 256 + d] =
              toBf16(float(int((row * 43 + head * 67 + d * 11) % 1019) - 509) / 1018.0f);
        }
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t head = 0; head < kKvHeads; ++head)
        for (uint32_t d = 0; d < 256; ++d) {
          const uint64_t base = (uint64_t{lane} * kKvHeads + head) *
                                RICHENGINE_VERIFY_CHUNK_STRIDE * 256;
          ck[base + row * 256 + d] = toBf16(code(history + row, head, d, lane, false) * 0.096f);
          cv[base + d * RICHENGINE_VERIFY_CHUNK_STRIDE + row] =
              toBf16(code(history + row, head, d, lane, true) * 0.112f);
        }

  ops::PagedVerifyBuffers buffers{keys, values, queries, partials,
                                  statistics, output, std::span(tables)};
  metal::CommandGraph graph;
  ops::PagedAttention::addVerify(graph, layer, buffers,
                                 std::span(stores).first(lanes), plan);
  // Warmup, then medians.
  std::vector<double> fused;
  for (uint32_t i = 0; i < repeat + 2; ++i) {
    const double seconds = backend.submitCommand(graph.dispatches()).gpuSeconds;
    if (i >= 2) fused.push_back(seconds * 1000.0);
  }
  Result result;
  std::sort(fused.begin(), fused.end());
  result.fused = fused[fused.size() / 2];
  for (const auto &[name, seconds] :
       benchmark::replayDispatches(backend, graph.dispatches()))
    result.pipelines.emplace_back(name, seconds * 1000.0);
  return result;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc < 3) {
        std::cerr << "usage: int4_attention_bench NEWLIB OLDLIB [--history N] "
                     "[--lanes N] [--repeat N]\n";
        return 64;
      }
      uint32_t history = 32768, lanes = kLanes, repeat = 7;
      for (int i = 3; i + 1 < argc; i += 2) {
        const std::string_view option(argv[i]);
        const uint32_t value = uint32_t(std::stoul(argv[i + 1]));
        if (option == "--history") history = value;
        else if (option == "--lanes") lanes = value;
        else if (option == "--repeat") repeat = value;
        else throw std::invalid_argument("unknown option " + std::string(option));
      }
      metal::MetalBackend next(argv[1]), old(argv[2]);
      std::cerr << "int4 verify bench: history=" << history << " lanes=" << lanes
                << " repeat=" << repeat << "\n";
      const Result a = run(next, history, lanes, repeat, false);
      const Result b = run(old, history, lanes, repeat, true);
      std::cout << "variant fused_ms";
      for (const auto &[name, ms] : a.pipelines) std::cout << ' ' << name;
      std::cout << '\n';
      std::cout << "int4b " << a.fused;
      for (const auto &[name, ms] : a.pipelines) std::cout << ' ' << ms;
      std::cout << "\nunpack " << b.fused;
      std::map<std::string, double> bmap(b.pipelines.begin(), b.pipelines.end());
      for (const auto &[name, ms] : a.pipelines)
        std::cout << ' ' << (bmap.count(name) ? bmap[name] : 0.0);
      std::cout << '\n';
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "int4_attention_bench: " << error.what() << '\n';
      return 1;
    }
  }
}

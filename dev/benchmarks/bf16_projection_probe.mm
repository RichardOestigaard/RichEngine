// Isolated GGUF BF16 parity and timing. No model or runtime allocations.
#include "metal/abi/Gguf.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t bufferLimit = 128 * MiB;
constexpr uint16_t canary = 0xffff;

struct Variant {
  const char *name;
  unsigned threads;
  bool accelerator;
  unsigned step;
  unsigned maxRows;
};
constexpr Variant simd{"gguf_float_wbf16_bf16", 512, false, 8, 32};
constexpr Variant na{"gguf_float_na_wbf16_bf16", 128, true, 32, 32};
constexpr std::array candidates{
    simd, na,
    Variant{"gguf_float_wbf16_bf16_s4", 128, false, 8, 32},
    Variant{"gguf_float_wbf16_bf16_s8", 256, false, 8, 32},
    Variant{"gguf_float_wbf16_bf16_m8", 512, false, 8, 8},
    Variant{"gguf_float_na_wbf16_bf16_k64", 128, true, 64, 32},
    Variant{"gguf_float_na_wbf16_bf16_k128", 128, true, 128, 32},
};

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                        id<MTLLibrary> library,
                                        const Variant &v) {
  id<MTLFunction> function = [library newFunctionWithName:@(v.name)];
  if (!function) throw std::runtime_error(v.name);
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < v.threads)
    throw std::runtime_error(error ? error.localizedDescription.UTF8String : v.name);
  return pipeline;
}

void fill(id<MTLBuffer> buffer, uint32_t seed) {
  std::mt19937 rng(seed);
  auto *words = static_cast<uint16_t *>(buffer.contents);
  for (size_t i = 0; i < buffer.length / sizeof(*words); ++i) {
    const uint32_t bits = rng();
    words[i] = uint16_t((bits & 0x807f) | ((121 + ((bits >> 16) % 7)) << 7));
  }
}

double run(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
           const Variant &v, id<MTLBuffer> input, id<MTLBuffer> weights,
           id<MTLBuffer> output, const GgufFloatParams &p) {
  std::memset(output.contents, 0xff, output.length);
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    if (!cmd || !encoder) throw std::runtime_error("Metal encoder unavailable");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weights offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&p length:sizeof(p) atIndex:3];
    const unsigned columns = v.accelerator ? 32 : 8;
    [encoder dispatchThreadgroups:MTLSizeMake((p.output_size + columns - 1) / columns, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(v.threads, 1, 1)];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "Metal command failed");
    const double elapsed = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6;
    if (!(elapsed > 0)) throw std::runtime_error("invalid GPU timestamp");
    return elapsed;
  }
}

void checkPadding(id<MTLBuffer> buffer, const GgufFloatParams &p) {
  const auto *words = static_cast<const uint16_t *>(buffer.contents);
  for (unsigned row = 0; row <= p.rows; ++row)
    for (unsigned col = 0; col < p.out_stride; ++col) {
      const bool live = row < p.rows && col >= p.out_offset && col < p.out_offset + p.output_size;
      if ((words[row * p.out_stride + col] == canary) == live)
        throw std::runtime_error("unexpected output or padding write");
    }
}

struct Work {
  unsigned n, k, rows;
};

void probe(id<MTLDevice> device, id<MTLCommandQueue> queue,
           NSArray<id<MTLComputePipelineState>> *pipelines,
           NSArray<id<MTLComputePipelineState>> *original,
           const std::vector<Work> &work, uint64_t cap, bool timing) {
  std::array<bool, candidates.size()> exact;
  exact.fill(true);
  for (unsigned phase = 0; phase < (timing ? 2u : 1u); ++phase) {
    for (const auto [n, k, rows] : work) {
      @autoreleasepool {
        const GgufFloatParams p{rows, k, n, n + 16, 8};
        const size_t inputBytes = size_t{rows} * k * 2, weightBytes = size_t{n} * k * 2;
        const size_t outputBytes = size_t{rows + 1} * p.out_stride * 2;
        const uint64_t pending = inputBytes + weightBytes + 2 * outputBytes;
        const uint64_t allocated = device.currentAllocatedSize;
        if (pending > bufferLimit || allocated > cap || pending > cap - allocated)
          throw std::runtime_error("probe memory cap exceeded before allocation");
        id<MTLBuffer> input = [device newBufferWithLength:inputBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights = [device newBufferWithLength:weightBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> a = [device newBufferWithLength:outputBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> b = [device newBufferWithLength:outputBytes options:MTLResourceStorageModeShared];
        if (!input || !weights || !a || !b || device.currentAllocatedSize > cap)
          throw std::runtime_error("probe allocation failed or exceeded cap");
        fill(input, 7821 + rows);
        fill(weights, 149 + n + k);
        for (unsigned i = 0; i < candidates.size(); ++i) {
          const Variant &v = candidates[i];
          if (rows > v.maxRows || (v.accelerator && rows < 16) || k % v.step) continue;
          const Variant &ref = v.accelerator ? na : simd;
          id<MTLComputePipelineState> base = original[v.accelerator ? 1 : 0];
          if (phase == 0) {
            run(queue, base, ref, input, weights, a, p);
            run(queue, pipelines[i], v, input, weights, b, p);
            checkPadding(a, p);
            checkPadding(b, p);
            const auto *x = static_cast<const uint16_t *>(a.contents);
            const auto *y = static_cast<const uint16_t *>(b.contents);
            size_t differences = 0;
            for (size_t j = 0; j < outputBytes / 2; ++j) differences += x[j] != y[j];
            exact[i] &= differences == 0;
            std::printf("parity %s rows=%u n=%u k=%u diff=%zu values=%u\n",
                        v.name, rows, n, k, differences, rows * n);
          } else if (exact[i]) {
            run(queue, base, ref, input, weights, a, p);
            run(queue, pipelines[i], v, input, weights, b, p);
            std::array<double, 5> ratios;
            for (unsigned j = 0; j < ratios.size(); ++j) {
              double ta, tb;
              if (j % 2 == 0) {
                ta = run(queue, base, ref, input, weights, a, p);
                tb = run(queue, pipelines[i], v, input, weights, b, p);
              } else {
                tb = run(queue, pipelines[i], v, input, weights, b, p);
                ta = run(queue, base, ref, input, weights, a, p);
              }
              ratios[j] = tb / ta;
            }
            std::sort(ratios.begin(), ratios.end());
            std::printf("timing %s rows=%u n=%u k=%u ratio=%.5f\n",
                        v.name, rows, n, k, ratios[ratios.size() / 2]);
          }
        }
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  @autoreleasepool {
    try {
      if (argc != 6 || std::string_view(argv[4]) != "--memory-cap-mib")
        throw std::runtime_error("usage: probe ORIGINAL CURRENT smoke|full --memory-cap-mib 256");
      unsigned mib = 0;
      const std::string_view limit(argv[5]), mode(argv[3]);
      const auto parsed = std::from_chars(limit.data(), limit.data() + limit.size(), mib);
      if (parsed.ec != std::errc{} || parsed.ptr != limit.data() + limit.size() ||
          mib < 128 || mib > 256 || (mode != "smoke" && mode != "full"))
        throw std::runtime_error("invalid probe mode or memory cap");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (!queue) throw std::runtime_error("Metal queue unavailable");
      NSError *error = nil;
      id<MTLLibrary> old = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
      if (!old) throw std::runtime_error(error.localizedDescription.UTF8String);
      id<MTLLibrary> current = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error];
      if (!current) throw std::runtime_error(error.localizedDescription.UTF8String);
      NSArray *original = @[makePipeline(device, old, simd), makePipeline(device, old, na)];
      NSMutableArray *pipelines = [NSMutableArray array];
      for (const Variant &v : candidates) [pipelines addObject:makePipeline(device, current, v)];
      std::vector<Work> shapes;
      for (unsigned rows : {8u, 16u, 24u, 32u}) {
        if (mode == "smoke") {
          shapes.push_back({64, 128, rows});
          continue;
        }
        for (const auto [n, k] : std::array<std::array<unsigned, 2>, 6>{{
                 {256, 2688}, {4096, 2688}, {2688, 4096},
                 {6144, 2688}, {2688, 6144}, {2688, 16128},
             }})
          shapes.push_back({n, k, rows});
      }
      probe(device, queue, pipelines, original, shapes, uint64_t{mib} * MiB, mode == "full");
      return 0;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "probe: %s\n", error.what());
      return 1;
    }
  }
}

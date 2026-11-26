// Times the gguf_prefill_tile shape variants of gguf_prefill_tile_bench.metal on one GGUF projection shape:
//   gguf-prefill-tile-bench <metallib> <fmt> <N> <K> <rows[,rows...]> [rounds] [check]
// Every variant is one kernel of the metallib with its own tile geometry (rows per threadgroup, column tile,
// threads); the harness sizes each grid to match. The weights stream DRAM-cold through a ring of plane copies, as
// gguf_projection_benchmark.mm's prefill cases do. With `check`, each variant's output is compared against the
// base variant's on the first copy before timing (count of differing outputs and maximum absolute difference
// over the live rows).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../tests/engine/GgufFormatReference.hpp"
#include "metal/abi/Gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace gguf_reference;

namespace {

constexpr uint64_t kRingBytes = 384ull << 20;
constexpr uint32_t kMinimumCopies = 3, kMaximumCopies = 64;
constexpr double kWarmupSeconds = 0.1;

std::vector<std::string> split(const std::string &text, char separator) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  for (std::string part; std::getline(stream, part, separator);) parts.push_back(part);
  return parts;
}

// One variant kernel and the dispatch geometry its tile implies.
struct Variant {
  const char *suffix;
  uint32_t rowsPerTG, tileN, threads;
};
const Variant kVariants[] = {
    {"base", 128, 64, 128}, {"sg8", 256, 64, 256}, {"sg2", 64, 64, 64},
    {"n32", 128, 32, 128},  {"n32ks128", 128, 32, 128}, {"r64sg2", 128, 64, 64},
    {"es2", 128, 64, 128},
    {"ks32", 128, 64, 128}, {"n32sg8", 256, 32, 256}, {"r16sg8", 128, 64, 256}, {"r16sg4", 64, 64, 128},
    {"n32ks32", 128, 32, 128}, {"n32sg2", 64, 32, 64}, {"n16", 128, 16, 128},
    {"nodeq", 128, 64, 128}, {"nomatmul", 128, 64, 128},
};
constexpr size_t kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 6 || argc > 8) {
      std::cerr << "usage: gguf-prefill-tile-bench <metallib> <fmt> <N> <K> <rows[,rows...]> [rounds] [check]\n";
      return 2;
    }
    const std::string formatName = argv[2];
    const uint32_t N = uint32_t(std::stoul(argv[3])), K = uint32_t(std::stoul(argv[4]));
    const uint32_t rounds = argc > 6 ? uint32_t(std::stoul(argv[6])) : 12;
    const bool check = argc > 7 && std::string(argv[7]) == "check";
    std::vector<uint32_t> rowCases;
    for (const std::string &rows : split(argv[5], ',')) rowCases.push_back(uint32_t(std::stoul(rows)));
    const Fmt fmt = fmtNamed(formatName);
    if (fmt == FMT_COUNT || !N || N % 32 || !K || K % 128 || rowCases.empty()) {
      std::cerr << "bad arguments: known format, N a multiple of 32, K of 128, one or more row counts\n";
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!library) {
      std::cerr << "gguf-prefill-tile-bench: " << error.localizedDescription.UTF8String << '\n';
      return 1;
    }
    id<MTLComputePipelineState> pipelines[kVariantCount];
    for (size_t v = 0; v < kVariantCount; ++v) {
      const std::string name = "pfb_" + formatName + "_" + kVariants[v].suffix;
      id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
      if (!function) {
        std::cerr << "gguf-prefill-tile-bench: missing kernel " << name << '\n';
        return 1;
      }
      pipelines[v] = [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipelines[v]) {
        std::cerr << "gguf-prefill-tile-bench: " << error.localizedDescription.UTF8String << '\n';
        return 1;
      }
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    std::mt19937 rng(9);
    const Packed planes = repack(fmt, makeNative(fmt, N, K, rng), N, K, nullptr);
    const uint64_t planeBytes = planes.w0.size() + planes.w1.size() + planes.meta.size();
    const uint32_t copies =
        uint32_t(std::clamp<uint64_t>((kRingBytes + planeBytes - 1) / planeBytes, kMinimumCopies, kMaximumCopies));
    struct Copy {
      id<MTLBuffer> w0, w1, meta;
    };
    std::vector<Copy> ring(copies);
    const auto upload = [&](const std::vector<uint8_t> &bytes) {
      return [device newBufferWithBytes:bytes.data() length:bytes.size() options:MTLResourceStorageModeShared];
    };
    for (Copy &c : ring) {
      c.w0 = upload(planes.w0);
      c.meta = upload(planes.meta);
      c.w1 = planes.w1.empty() ? nil : upload(planes.w1);
    }
    const uint32_t maxRows = *std::max_element(rowCases.begin(), rowCases.end());
    const uint32_t storageRows = (maxRows + 255) / 256 * 256;
    id<MTLBuffer> input = [device newBufferWithLength:uint64_t{storageRows} * K * sizeof(__bf16)
                                              options:MTLResourceStorageModeShared];
    {
      std::uniform_real_distribution<float> unit(-1.f, 1.f);
      auto *x = static_cast<__bf16 *>(input.contents);
      for (uint64_t i = 0; i < uint64_t{storageRows} * K; ++i) x[i] = __bf16(unit(rng));
    }
    id<MTLBuffer> outputs[kVariantCount];
    for (size_t v = 0; v < kVariantCount; ++v)
      outputs[v] = [device newBufferWithLength:uint64_t{storageRows} * N * sizeof(__bf16)
                                       options:MTLResourceStorageModeShared];

    const auto encode = [&](id<MTLComputeCommandEncoder> enc, size_t v, uint32_t rows, uint32_t copy) {
      const Variant &variant = kVariants[v];
      const Copy &c = ring[copy % copies];
      GgufPrefillParams params{K, rows, N, 0};
      [enc setComputePipelineState:pipelines[v]];
      [enc setBuffer:input offset:0 atIndex:0];
      [enc setBuffer:c.w0 offset:0 atIndex:1];
      [enc setBuffer:(c.w1 ? c.w1 : c.meta) offset:0 atIndex:2];
      [enc setBuffer:c.meta offset:0 atIndex:3];
      [enc setBuffer:outputs[v] offset:0 atIndex:4];
      [enc setBytes:&params length:sizeof(params) atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake((rows + variant.rowsPerTG - 1) / variant.rowsPerTG, N / variant.tileN, 1)
          threadsPerThreadgroup:MTLSizeMake(variant.threads, 1, 1)];
    };
    const auto submit = [&](size_t v, uint32_t rows) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
      for (uint32_t c = 0; c < copies; ++c) encode(enc, v, rows, c);
      [enc endEncoding];
      [command commit];
      [command waitUntilCompleted];
      return (command.GPUEndTime - command.GPUStartTime) * 1e3 / copies;
    };

    for (const uint32_t rows : rowCases) {
      if (check) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        for (size_t v = 0; v < kVariantCount; ++v) encode(enc, v, rows, 0);
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        const auto *base = static_cast<const __bf16 *>(outputs[0].contents);
        std::printf("rows %u check:", rows);
        for (size_t v = 1; v < kVariantCount; ++v) {
          const auto *got = static_cast<const __bf16 *>(outputs[v].contents);
          uint64_t different = 0;
          double worst = 0;
          for (uint64_t i = 0; i < uint64_t{rows} * N; ++i) {
            const float d = std::fabs(float(got[i]) - float(base[i]));
            if (d) ++different;
            worst = std::max(worst, double(d));
          }
          std::printf(" %s diff=%llu max=%.3g", kVariants[v].suffix, (unsigned long long)different, worst);
        }
        std::printf("\n");
      }
      std::vector<double> ms[kVariantCount];
      for (size_t v = 0; v < kVariantCount; ++v)
        for (double spent = 0; spent < kWarmupSeconds;) spent += submit(v, rows) * copies / 1e3;
      for (uint32_t r = 0; r < rounds; ++r)
        for (size_t i = 0; i < kVariantCount; ++i) {
          const size_t v = r % 2 ? kVariantCount - 1 - i : i;
          ms[v].push_back(submit(v, rows));
        }
      const double flops = 2.0 * N * K * rows;
      std::printf("rows %-5u (%.1f GFLOP):", rows, flops / 1e9);
      for (size_t v = 0; v < kVariantCount; ++v) {
        const double t = median(ms[v]);
        std::printf(" %s %.4f (%.1f TF)", kVariants[v].suffix, t, flops / t / 1e9);
      }
      std::printf("\n");
    }
    std::printf("%s %ux%u, %u copies of %.1f MB, median ms of %u rounds\n", formatName.c_str(), N, K, copies,
                planeBytes / 1e6, rounds);
    return 0;
  }
}

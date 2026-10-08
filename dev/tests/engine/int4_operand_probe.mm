// Host driver for int4_operand_probe.metal: fills a bf16 A tile and packed
// INT4 B bytes, runs each probe kernel at the strides given on the command
// line, and compares against a CPU fp32 dot under the nibble convention
// (element e -> nibble e, even element in the low nibble, two's complement).
// usage: int4_operand_probe METALLIB <qk|pv> bs0 bs1
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ops/KernelNames.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static float bf16(uint16_t v) {
  uint32_t bits = uint32_t(v) << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
static uint16_t to_bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return uint16_t(bits >> 16);
}
static int8_t nibble(uint8_t byte, int parity) {
  int code = (byte >> (4 * parity)) & 0xF;
  return int8_t(code - (code & 8 ? 16 : 0));
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      fprintf(stderr, "usage: int4_operand_probe METALLIB <qk|pv> bs0 bs1\n");
      return 64;
    }
    const bool qk = std::string(argv[2]) == "qk";
    const bool nn = std::string(argv[2]) == "pvnn";
    const int M = 48, N = qk ? 32 : 256, K = qk ? 256 : 32;
    const int bs0 = atoi(argv[3]), bs1 = atoi(argv[4]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithFile:@(argv[1]) error:&err];
    if (!lib) { fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String); return 70; }
    id<MTLFunction> fn =
        [lib newFunctionWithName:qk ? @(richengine::ops::kProbeInt4Qk.data()) : @(richengine::ops::kProbeInt4PvNn.data())];
    if (!fn) { fprintf(stderr, "missing kernel\n"); return 71; }
    id<MTLComputePipelineState> pipe =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 72; }

    const uint64_t aBytes = uint64_t(K) * M * 2;
    // B covers the largest byte index the strides can reach, plus margin.
    const uint64_t bBytes = uint64_t(std::abs(bs0)) * K + uint64_t(std::abs(bs1)) * N + 4096;
    id<MTLBuffer> aBuf = [dev newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuf = [dev newBufferWithLength:bBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> cBuf = [dev newBufferWithLength:uint64_t(M) * N * 4 options:MTLResourceStorageModeShared];
    auto *a = (uint16_t *)aBuf.contents;
    auto *b = (uint8_t *)bBuf.contents;
    auto *c = (float *)cBuf.contents;

    std::mt19937 rng(11);
    auto code = [&](int e) { return int((e * 73 + (e >> 3) * 5) % 15) - 7; };
    // Fill every byte of B so any nibble decode is deterministic.
    for (uint64_t i = 0; i < bBytes; ++i) {
      const int e = int(2 * i);
      b[i] = uint8_t((code(e) & 0xF) | ((code(e + 1) & 0xF) << 4));
    }
    for (int i = 0; i < K * M; ++i)
      a[i] = to_bf16(float(int(rng() % 101) - 50) / 101.0f);
    memset(c, 0, size_t(M) * N * 4);

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:aBuf offset:0 atIndex:0];
    [enc setBuffer:bBuf offset:0 atIndex:1];
    [enc setBuffer:cBuf offset:0 atIndex:2];
    [enc setBytes:&bs0 length:4 atIndex:3];
    [enc setBytes:&bs1 length:4 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      fprintf(stderr, "command failed: %s\n",
              cmd.error.localizedDescription.UTF8String);
      return 73;
    }

    // CPU reference: B[k][n] = nibble at element index k*bs0 + n*bs1.
    int bad = 0;
    float worst = 0;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        float expected = 0;
        for (int k = 0; k < K; ++k) {
          const int e = nn ? n * bs0 + k * bs1 : k * bs0 + n * bs1;
          expected += bf16(a[k + m * K]) * nibble(b[e / 2], e & 1);
        }
        const float diff = std::fabs(c[m * N + n] - expected);
        if (diff > worst) worst = diff;
        if (diff > 0.01f && bad++ < 4)
          printf("m=%d n=%d got=%f want=%f\n", m, n, c[m * N + n], expected);
      }
    printf("%s bs={%d,%d}: %s worst=%g (%d bad)\n", qk ? "qk" : "pv", bs0, bs1,
           bad ? "FAIL" : "ok", worst, bad);
    return bad ? 1 : 0;
  }
}

// The Gemma GeGLU activation's saturation: geglu_multiply and the softcap
// kernels must stay finite when their tanh arguments leave the exp-based
// tanh's fp32 domain (|x| ≳ 44 overflows exp(2x); the inf/inf quotient
// NaNs). The trunk's real gate rows reach ±30, past the old kernel's NaN
// knee — this test runs geglu_multiply at the trunk's exact dispatch over
// gate/up values that straddle it and requires a finite, bounded product.
//
// Standalone like canvas-kernels: the kernels live in the production
// metallib, so the test links no engine objects and takes $(LIB) as argv[1].
//
// usage: gelu-saturation METALLIB
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ops/KernelNames.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      fprintf(stderr, "usage: gelu-saturation METALLIB\n");
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSData *fileData =
        [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
    if (!fileData) {
      fprintf(stderr, "unable to read metallib\n");
      return 1;
    }
    dispatch_data_t data = dispatch_data_create(
        fileData.bytes, fileData.length, nullptr, ^{ (void)fileData; });
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (!library) {
      fprintf(stderr, "library: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLFunction> function = [library newFunctionWithName:@(richengine::ops::kGegluMultiply.data())];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
      fprintf(stderr, "pipeline: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }

    // The shared expert's exact prefill shape: 256 rows of 2304, gate/up
    // uniform over ±28 — past the ±11 gate where the unclamped tanh NaNs.
    const uint32_t rows = 256, width = 2304;
    const uint32_t count = rows * width;
    std::mt19937 random(7);
    std::uniform_real_distribution<float> values(-28.0f, 28.0f);
    id<MTLBuffer> gate = [device newBufferWithLength:uint64_t{count} * 2
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> up = [device newBufferWithLength:uint64_t{count} * 2
                                           options:MTLResourceStorageModeShared];
    uint16_t *gateBits = static_cast<uint16_t *>(gate.contents);
    uint16_t *upBits = static_cast<uint16_t *>(up.contents);
    for (uint32_t i = 0; i < count; ++i) {
      const float g = values(random), u = values(random);
      uint32_t gb, ub;
      std::memcpy(&gb, &g, 4);
      std::memcpy(&ub, &u, 4);
      gateBits[i] = static_cast<uint16_t>((gb + 0x7FFF + ((gb >> 16) & 1)) >> 16);
      upBits[i] = static_cast<uint16_t>((ub + 0x7FFF + ((ub >> 16) & 1)) >> 16);
    }
    id<MTLBuffer> output = [device newBufferWithLength:uint64_t{count} * 2
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:gate offset:0 atIndex:0];
    [encoder setBuffer:up offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    uint32_t countValue = count;
    [encoder setBytes:&countValue length:4 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];

    uint16_t *outBits = static_cast<uint16_t *>(output.contents);
    uint64_t nonFinite = 0;
    float maximum = 0;
    for (uint32_t i = 0; i < count; ++i) {
      float f;
      uint32_t bits = uint32_t(outBits[i]) << 16;
      std::memcpy(&f, &bits, 4);
      if (!std::isfinite(f))
        ++nonFinite;
      else
        maximum = std::max(maximum, std::abs(f));
    }
    if (nonFinite) {
      fprintf(stderr, "FAIL: geglu_multiply emitted %llu non-finite values "
                      "of %u (max finite %.3f)\n",
              (unsigned long long)nonFinite, count, maximum);
      return 1;
    }
    // gelu_tanh(gate) * up saturates near |gate| * |up| = 784 here.
    if (maximum < 100.0f) {
      fprintf(stderr, "FAIL: geglu_multiply's maximum %.3f looks wrong\n",
              maximum);
      return 1;
    }
    printf("PASS gelu saturation: geglu_multiply finite over %u elements, "
           "max %.1f\n", count, maximum);
    return 0;
  }
}

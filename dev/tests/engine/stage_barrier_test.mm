// stage_barrier.h's self-test host: runs the mutual-barrier dispatch and
// the producer/consumer pair (one command buffer, two encoders), then a
// bounded-failure case whose ninth worker never arrives — the poll must
// report it and release the waiters, never hang.
// usage: stage-barrier-test METALLIB
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>

#include "ops/KernelNames.hpp"

namespace {

constexpr uint32_t kWorkers = 8;
constexpr uint32_t kThreads = 64;
constexpr uint32_t kBarrierWords = 34;  // flags[32] + released + error
constexpr uint32_t kEpochMutual = 1;
constexpr uint32_t kEpochPair = 2;
constexpr uint32_t kEpochMissing = 3;

struct Params {
  uint32_t epoch;
  uint32_t workers;
};

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    ++failures;
    fprintf(stderr, "FAIL: %s\n", what);
  }
}

id<MTLBuffer> buffer(id<MTLDevice> dev, NSUInteger bytes) {
  id<MTLBuffer> result =
      [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  std::memset(result.contents, 0, bytes);
  return result;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                     const char *name, NSError **error) {
  return [dev newComputePipelineStateWithFunction:
                [lib newFunctionWithName:[NSString stringWithUTF8String:name]]
                    error:error];
}

void bind(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
          std::initializer_list<id<MTLBuffer>> buffers,
          const Params &params, NSUInteger index) {
  [enc setComputePipelineState:pso];
  uint32_t at = 0;
  for (id<MTLBuffer> b : buffers)
    [enc setBuffer:b offset:0 atIndex:at++];
  [enc setBytes:&params length:sizeof(params) atIndex:index];
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      fprintf(stderr, "usage: stage-barrier-test METALLIB\n");
      return 64;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib =
        [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    if (!lib) {
      fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String);
      return 70;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    using namespace richengine::ops;
    id<MTLComputePipelineState> sync =
        pipeline(dev, lib, std::string(kTestStageBarrierSync).c_str(), &err);
    id<MTLComputePipelineState> produce = pipeline(
        dev, lib, std::string(kTestStageBarrierProduce).c_str(), &err);
    id<MTLComputePipelineState> consume = pipeline(
        dev, lib, std::string(kTestStageBarrierConsume).c_str(), &err);
    if (!sync || !produce || !consume) {
      fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String);
      return 70;
    }

    const NSUInteger dataBytes =
        (kWorkers + 1) * kThreads * sizeof(uint32_t);

    // Mutual barrier: eight worker threadgroups in one dispatch.
    {
      id<MTLBuffer> barrier = buffer(dev, kBarrierWords * sizeof(uint32_t));
      id<MTLBuffer> data = buffer(dev, dataBytes);
      id<MTLBuffer> bad = buffer(dev, sizeof(uint32_t));
      id<MTLBuffer> ok = buffer(dev, kWorkers * sizeof(uint32_t));
      const Params params{kEpochMutual, kWorkers};
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      bind(enc, sync, {barrier, data, bad, ok}, params, 4);
      [enc dispatchThreadgroups:MTLSizeMake(kWorkers, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      check(cb.status == MTLCommandBufferStatusCompleted,
            "mutual barrier command did not complete");
      const uint32_t *okWords = (const uint32_t *)ok.contents;
      bool allOk = true;
      for (uint32_t w = 0; w < kWorkers; ++w)
        allOk &= okWords[w] == 1;
      check(allOk, "mutual barrier released every worker");
      check(*(const uint32_t *)bad.contents == 0,
            "mutual barrier exposed stale or torn worker data");
      const uint32_t *state = (const uint32_t *)barrier.contents;
      check(state[32] == kEpochMutual && state[33] == 0,
            "mutual barrier left a wrong release or error");
    }

    // Producer/consumer pair: two dispatches of one command buffer, the
    // consumer's single simdgroup polling the producers' flags.
    for (uint32_t missing = 0; missing <= 1; ++missing) {
      id<MTLBuffer> barrier = buffer(dev, kBarrierWords * sizeof(uint32_t));
      id<MTLBuffer> data = buffer(dev, dataBytes);
      id<MTLBuffer> bad = buffer(dev, sizeof(uint32_t));
      id<MTLBuffer> ok = buffer(dev, sizeof(uint32_t));
      id<MTLBuffer> sink = buffer(dev, kWorkers * sizeof(uint32_t));
      const uint32_t workers = kWorkers + missing;
      const Params params{kEpochPair + missing, workers};
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      bind(enc, produce, {barrier, data, sink}, params, 3);
      [enc dispatchThreadgroups:MTLSizeMake(kWorkers, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
      [enc endEncoding];
      enc = [cb computeCommandEncoder];
      bind(enc, consume, {barrier, data, bad, ok}, params, 4);
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      check(cb.status == MTLCommandBufferStatusCompleted,
            missing ? "missing-worker command hung"
                    : "producer/consumer command did not complete");
      const uint32_t *state = (const uint32_t *)barrier.contents;
      if (!missing) {
        check(*(const uint32_t *)ok.contents == 1,
              "producer/consumer pair released cleanly");
        check(state[33] == 0, "producer/consumer pair raised an error");
      } else {
        check(*(const uint32_t *)ok.contents == 0,
              "missing worker was not reported to the consumer");
        check(state[33] == kWorkers + 1,
              "bounded spin did not name the missing worker");
      }
    }

    if (failures) {
      fprintf(stderr, "stage_barrier_test: FAIL (%d)\n", failures);
      return 1;
    }
    printf("stage_barrier_test: PASS\n");
    return 0;
  }
}

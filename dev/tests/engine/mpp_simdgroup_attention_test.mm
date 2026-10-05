// Numerical check and micro-benchmark for the per-simdgroup MPP TensorOps
// attention tile in mpp_simdgroup_attention_test.metal: synthetic INT8 paged
// tiles verified against a CPU fp32 reference, then timed against the
// production-structure threadgroup variant.
// usage: mpp-simdgroup-attention METALLIB [groups] [pages] [reps]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

static const uint32_t M = 48, D = 256, N = 32, QH = 6;

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

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: probe METALLIB [groups] [pages] [reps]\n"); return 64; }
    const uint32_t groups = argc > 2 ? atoi(argv[2]) : 8;
    const uint32_t pages = argc > 3 ? atoi(argv[3]) : 64;
    const uint32_t reps = argc > 4 ? atoi(argv[4]) : 100;
    const uint32_t committed = pages * N - 8;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    if (!lib) { fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String); return 70; }

    const uint64_t qBytes = uint64_t(groups) * M * D * 2;
    const uint64_t kBytes = uint64_t(groups) * pages * N * D;
    const uint64_t vBytes = kBytes;
    const uint64_t sBytes = uint64_t(groups) * pages * N * 4;
    const uint64_t oBytes = uint64_t(groups) * M * D * 4;
    const uint64_t stBytes = uint64_t(groups) * M * 2 * 4;
    id<MTLBuffer> qBuf = [dev newBufferWithLength:qBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuf = [dev newBufferWithLength:kBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> vBuf = [dev newBufferWithLength:vBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> ksBuf = [dev newBufferWithLength:sBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> vsBuf = [dev newBufferWithLength:sBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> oBuf = [dev newBufferWithLength:oBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> oBuf2 = [dev newBufferWithLength:oBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> stBuf = [dev newBufferWithLength:stBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> stBuf2 = [dev newBufferWithLength:stBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> flBuf = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];

    std::mt19937 rng(7);
    uint16_t *q = (uint16_t *)qBuf.contents;
    int8_t *k = (int8_t *)kBuf.contents;
    int8_t *v = (int8_t *)vBuf.contents;
    float *ks = (float *)ksBuf.contents;
    float *vs = (float *)vsBuf.contents;
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::uniform_int_distribution<int> qi(-8, 8);
    for (uint64_t i = 0; i < qBytes / 2; ++i) q[i] = to_bf16(uf(rng) * 2.f);
    for (uint64_t i = 0; i < kBytes; ++i) k[i] = int8_t(qi(rng));
    for (uint64_t i = 0; i < vBytes; ++i) v[i] = int8_t(qi(rng));
    for (uint64_t i = 0; i < sBytes / 4; ++i) ks[i] = 0.05f + 0.05f * uf(rng);
    for (uint64_t i = 0; i < sBytes / 4; ++i) vs[i] = 0.05f + 0.05f * uf(rng);

    id<MTLCommandQueue> queue = [dev newCommandQueue];

    auto run = [&](NSString *name, id<MTLBuffer> outB, id<MTLBuffer> stB, bool timed) -> double {
      id<MTLComputePipelineState> pso =
          [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
      if (!pso) { fprintf(stderr, "pso %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(70); }
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:qBuf offset:0 atIndex:0];
      [enc setBuffer:kBuf offset:0 atIndex:1];
      [enc setBuffer:vBuf offset:0 atIndex:2];
      [enc setBuffer:ksBuf offset:0 atIndex:3];
      [enc setBuffer:vsBuf offset:0 atIndex:4];
      [enc setBuffer:outB offset:0 atIndex:5];
      [enc setBuffer:stB offset:0 atIndex:6];
      [enc setBuffer:flBuf offset:0 atIndex:7];
      uint32_t p = pages, c = committed;
      [enc setBytes:&p length:4 atIndex:8];
      [enc setBytes:&c length:4 atIndex:9];
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "%s: %s\n", name.UTF8String, cb.error.localizedDescription.UTF8String);
        exit(70);
      }
      if (!timed) return 0;
      // timed loop
      std::vector<double> ts;
      for (uint32_t r = 0; r < reps; ++r) {
        id<MTLCommandBuffer> t = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [t computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:qBuf offset:0 atIndex:0];
        [e setBuffer:kBuf offset:0 atIndex:1];
        [e setBuffer:vBuf offset:0 atIndex:2];
        [e setBuffer:ksBuf offset:0 atIndex:3];
        [e setBuffer:vsBuf offset:0 atIndex:4];
        [e setBuffer:outB offset:0 atIndex:5];
        [e setBuffer:stB offset:0 atIndex:6];
        [e setBuffer:flBuf offset:0 atIndex:7];
        [e setBytes:&p length:4 atIndex:8];
        [e setBytes:&c length:4 atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [t commit];
        [t waitUntilCompleted];
        ts.push_back(t.GPUEndTime - t.GPUStartTime);
      }
      std::sort(ts.begin(), ts.end());
      return ts[ts.size() / 2];
    };

    run(@"mpp_attention_threadgroup", oBuf2, stBuf2, false);
    run(@"mpp_attention_simdgroup", oBuf, stBuf, false);
    uint32_t *fl = (uint32_t *)flBuf.contents;
    printf("flags: left_compat=%u iter_compat=%u scores_cap=%u running_cap=%u rowred_cap=%u tg_running_cap=%u\n",
           fl[0], fl[1], fl[2], fl[3], fl[4], fl[8]);

    const float *o1 = (const float *)oBuf.contents;
    const float *o2 = (const float *)oBuf2.contents;
    const float *stA = (const float *)stBuf.contents;
    const float *stB = (const float *)stBuf2.contents;
    double maxDiff = 0, statDiff = 0;
    for (uint64_t i = 0; i < uint64_t(groups) * M * D; ++i)
      maxDiff = std::max(maxDiff, double(std::fabs(o1[i] - o2[i])));
    for (uint64_t i = 0; i < uint64_t(groups) * M * 2; ++i)
      statDiff = std::max(statDiff, double(std::fabs(stA[i] - stB[i])));
    printf("kernels: |sg-tg|max=%.5g stats |diff|max=%.6g\n", maxDiff, statDiff);

    // CPU reference, group g
    auto refGroup = [&](uint32_t g, std::vector<float> &ref) {
      const uint16_t *qg = q + uint64_t(g) * M * D;
      const int8_t *kg = k + uint64_t(g) * pages * N * D;
      const int8_t *vg = v + uint64_t(g) * pages * D * N;
      const float *ksg = ks + uint64_t(g) * pages * N;
      const float *vsg = vs + uint64_t(g) * pages * N;
    for (uint32_t row = 0; row < M; ++row) {
      const uint32_t qrow = row / QH;
      const uint32_t limit = std::min(committed + 8, committed + std::min(qrow, 7u) + 1);
      (void)0;
      std::vector<float> acc(D, 0.f), prob(pages * N);
      float rmax = -INFINITY, rsum = 0;
      for (uint32_t pg = 0; pg < pages; ++pg) {
        float s[N];
        float lmax = -INFINITY;
        for (uint32_t t = 0; t < N; ++t) {
          float acc_s = 0;
          for (uint32_t d = 0; d < D; ++d)
            acc_s += bf16(qg[row * D + d]) * float(kg[(pg * N + t) * D + d]);
          s[t] = acc_s * ksg[pg * N + t] * 0.0625f;
          if (pg * N + t >= limit) s[t] = -INFINITY;
          lmax = std::max(lmax, s[t]);
        }
        const float nmax = std::max(rmax, lmax);
        const float scale = (nmax == -INFINITY || nmax == rmax) ? 1.f : expf(rmax - nmax);
        float lsum = 0;
        for (uint32_t t = 0; t < N; ++t) {
          const float e = s[t] == -INFINITY ? 0.f : expf(s[t] - nmax);
          prob[pg * N + t] = e * vsg[pg * N + t];
          lsum += e;
        }
        rsum = rsum * scale + lsum;
        for (uint32_t d = 0; d < D; ++d) acc[d] *= scale;
        for (uint32_t d = 0; d < D; ++d) {
          float a = 0;
          for (uint32_t t = 0; t < N; ++t) a += prob[pg * N + t] * float(vg[(pg * D + d) * N + t]);
          acc[d] += a;
        }
        rmax = nmax;
      }
      for (uint32_t d = 0; d < D; ++d) ref[row * D + d] = rsum > 0 ? acc[d] / rsum : 0;
    }
    };
    // CPU reference for every group; both kernels must land near it.
    std::vector<float> ref(M * D);
    double refDiff1 = 0, refDiff2 = 0;
    for (uint32_t g = 0; g < groups; ++g) {
      refGroup(g, ref);
      const float *o1g = o1 + uint64_t(g) * M * D;
      const float *o2g = o2 + uint64_t(g) * M * D;
      for (uint64_t i = 0; i < uint64_t(M) * D; ++i) {
        refDiff1 = std::max(refDiff1, double(std::fabs(o1g[i] - ref[i])));
        refDiff2 = std::max(refDiff2, double(std::fabs(o2g[i] - ref[i])));
      }
    }
    printf("verify: |sg-ref|max=%.5g |tg-ref|max=%.5g over %u groups\n",
           refDiff1, refDiff2, groups);

    double tsg = run(@"mpp_attention_simdgroup", oBuf, stBuf, true);
    double ttg = run(@"mpp_attention_threadgroup", oBuf2, stBuf2, true);
    printf("groups=%u pages=%u reps=%u\n", groups, pages, reps);
    printf("simdgroup-chained: %.3f ms\n", tsg * 1e3);
    printf("threadgroup-staged: %.3f ms\n", ttg * 1e3);
    printf("ratio sg/tg: %.3f\n", tsg / ttg);
    // The threadgroup baseline is a structural timing reference; the
    // prototype must match the fp32 reference and chain its CTs.
    if (refDiff1 > 0.02 || !fl[0]) {
      fprintf(stderr, "FAIL: prototype diverged or left-input chaining unavailable\n");
      return 1;
    }
    printf("PASS\n");
  }
  return 0;
}

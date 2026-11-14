// Host for acc16_probe.metal: runs each value-format kernel once and
// compares the bfloat-CT and float-CT PV accumulators bitwise.
// usage: acc16-probe METALLIB
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr uint32_t M = 32, D = 256, N = 32, PAGES = 8;
static constexpr uint64_t ELEMENTS = uint64_t(M) * D;
static uint16_t to_bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7fff + lsb;
  return uint16_t(bits >> 16);
}
static float from_bf16(uint16_t v) {
  uint32_t bits = uint32_t(v) << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// CPU mirror of the kernel's prob hash and value fills (fp64 accumulate).
static double cpuProb(uint32_t page, uint32_t token, uint32_t row) {
  uint32_t h = page * 0x9e3779b1u + token * 0x85ebca6bu + row * 0xc2b2ae35u;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  return double(h & 0xffffffu) / 16777216.0;
}
static double cpuI8(uint64_t idx) {
  return double(int(((uint32_t(idx) * 2654435761u) >> 28) & 0xf) - 8);
}
static double cpuBF(uint64_t idx) {
  const uint32_t h = uint32_t(idx) * 2654435761u;
  return double(from_bf16(to_bf16(float(int((h >> 30) & 3) - 2) * 0.03125f)));
}
// int4b: nibble e of the page's byte slab, sign-extended from 4 bits.
static double cpuI4Nibble(const uint8_t *page, uint64_t e) {
  int nib = (page[e / 2] >> (4 * (e & 1))) & 0xf;
  if (nib >= 8) nib -= 16;
  return nib;
}
static double cpuRef(int kind, uint32_t row, uint32_t dim,
                     const uint8_t *values) {
  double acc = 0.0;
  for (uint32_t page = 0; page < PAGES; ++page) {
    const double s = page == 0 ? 1.0 : 0.9999;
    acc *= s;
    double c = 0.0;
    for (uint32_t t = 0; t < N; ++t) {
      double v;
      if (kind == 0)      // i8: dim-major [D][N] slab per page
        v = cpuI8(uint64_t(page) * D * N + dim * N + t);
      else if (kind == 1) // bf16: same layout
        v = cpuBF(uint64_t(page) * D * N + dim * N + t);
      else                // i4: element (dim + t*D) nibbles of the page slab
        v = cpuI4Nibble(values + uint64_t(page) * (D * N / 2),
                        uint64_t(dim) + uint64_t(t) * D);
      c += cpuProb(page, t, row) * v;
    }
    acc += c;
  }
  return acc;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2) {
      fprintf(stderr, "usage: acc16-probe METALLIB\n");
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

    struct Case {
      const char *name;
      const char *kernel;
      uint64_t valueBytes;
      void (*fill)(void *, uint64_t);
    };
    auto fillI8 = [](void *p, uint64_t n) {
      int8_t *v = (int8_t *)p;
      for (uint64_t i = 0; i < n; ++i)
        v[i] = int8_t(int(((uint32_t(i) * 2654435761u) >> 28) & 0xf) - 8);
    };
    auto fillBF = [](void *p, uint64_t n) {
      uint16_t *v = (uint16_t *)p;
      for (uint64_t i = 0; i < n / 2; ++i) {
        const uint32_t h = uint32_t(i) * 2654435761u;
        float f = float(int((h >> 30) & 3) - 2) * 0.03125f;
        v[i] = to_bf16(f);
      }
    };
    auto fillI4 = [](void *p, uint64_t n) {
      uint8_t *v = (uint8_t *)p;
      for (uint64_t i = 0; i < n; ++i)
        v[i] = uint8_t(((uint32_t(i) * 40503u) >> 8) & 0x77);  // nibbles in [-4, 3]
    };
    Case cases[] = {
        {"i8", "acc16_probe_i8", uint64_t(PAGES) * D * N, fillI8},
        {"bf16", "acc16_probe_bf16", uint64_t(PAGES) * D * N * 2, fillBF},
        {"i4", "acc16_probe_i4", uint64_t(PAGES) * D * N / 2, fillI4},
    };

    for (const Case &c : cases) {
      id<MTLBuffer> values =
          [dev newBufferWithLength:c.valueBytes
                           options:MTLResourceStorageModeShared];
      c.fill(values.contents, c.valueBytes);
      id<MTLBuffer> outBF =
          [dev newBufferWithLength:ELEMENTS * 4
                           options:MTLResourceStorageModeShared];
      id<MTLBuffer> outFP =
          [dev newBufferWithLength:ELEMENTS * 4
                           options:MTLResourceStorageModeShared];
      NSString *kernelName = [NSString stringWithUTF8String:c.kernel];
      id<MTLComputePipelineState> pso =
          [dev newComputePipelineStateWithFunction:
                    [lib newFunctionWithName:kernelName]
                                        error:&err];
      if (!pso) {
        fprintf(stderr, "%s pso: %s\n", c.name,
                err.localizedDescription.UTF8String);
        return 70;
      }
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:values offset:0 atIndex:0];
      [enc setBuffer:outBF offset:0 atIndex:1];
      [enc setBuffer:outFP offset:0 atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "%s: %s\n", c.name,
                cb.error.localizedDescription.UTF8String);
        return 70;
      }
      const float *bf = (const float *)outBF.contents;
      const float *fp = (const float *)outFP.contents;
      uint64_t exact = 0, bf16ofFP = 0, diff = 0, nonfinite = 0;
      float minBF = 1e30f, maxBF = -1e30f, minFP = 1e30f, maxFP = -1e30f;
      double maxAbs = 0, maxRel = 0;
      for (uint64_t i = 0; i < ELEMENTS; ++i) {
        const float b = bf[i], f = fp[i];
        minBF = std::min(minBF, b); maxBF = std::max(maxBF, b);
        minFP = std::min(minFP, f); maxFP = std::max(maxFP, f);
        if (!std::isfinite(b) || !std::isfinite(f)) ++nonfinite;
        if (b == f) { ++exact; continue; }
        // Does the bf16 element equal fp32 rounded to bf16 (and back)?
        if (from_bf16(to_bf16(f)) == b) ++bf16ofFP;
        ++diff;
        const double d = std::fabs(double(b) - double(f));
        maxAbs = std::max(maxAbs, d);
        maxRel = std::max(maxRel, d / std::max(1e-30, std::fabs(double(f))));
      }
      int kind = c.name[0] == 'i' && c.name[1] == '8' ? 0
               : c.name[0] == 'b' ? 1 : 2;
      const uint8_t *vbytes = (const uint8_t *)values.contents;
      double errBF = 0, errFP = 0, maxRelBF = 0;
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t dim = 0; dim < D; ++dim) {
          const double ref = cpuRef(kind, row, dim, vbytes);
          const float b = bf[row * D + dim], f = fp[row * D + dim];
          errBF = std::max(errBF, std::fabs(double(b) - ref));
          errFP = std::max(errFP, std::fabs(double(f) - ref));
          maxRelBF = std::max(maxRelBF,
                              std::fabs(double(b) - ref) /
                                  std::max(1e-30, std::fabs(ref)));
        }
      printf("%s: exact=%llu/%llu bf16(fp32)=%llu diff=%llu nonfinite=%llu "
             "bf16range=[%.4g,%.4g] fprange=[%.4g,%.4g] "
             "maxAbs=%.4g maxRel=%.4g | vsRef64: bf16=%.4g fp32=%.4g "
             "maxRelBF=%.4g\n",
             c.name, (unsigned long long)exact, (unsigned long long)ELEMENTS,
             (unsigned long long)bf16ofFP, (unsigned long long)diff,
             (unsigned long long)nonfinite, minBF, maxBF, minFP, maxFP,
             maxAbs, maxRel, errBF, errFP, maxRelBF);
    }
  }
  return 0;
}

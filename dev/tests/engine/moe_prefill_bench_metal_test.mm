// MoE expert-pass microbenchmark at the DiffusionGemma canvas shape:
// hidden 2816, expert intermediate 768, 128 experts top-8 + shared, 256
// canvas rows. On the live model each layer routes to a union of ~34-75
// experts (~85-119 m32 tiles), and the production split passes reach only
// ~50 GB/s — the trunk is ~115 ms of the 139 ms canvas step and these
// passes dominate it. This bench times the production kernels against
// bench variants (bench_moe_* kernels, moe_prefill_bench.metal): narrower
// column tiles, four-simdgroup tiles, 16-row grouped tiles and a K-split
// publishing fp32 partials.
//
// usage: moe-prefill-bench PROD_METALLIB BENCH_METALLIB [reps]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ops/KernelNames.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

namespace {

constexpr uint32_t kRows = 256;      // canvas rows
constexpr uint32_t kHidden = 2816;   // expert K (gate/up input)
constexpr uint32_t kInter = 768;     // expert N (gate/up output)
constexpr uint32_t kExperts = 128;
constexpr uint32_t kTopK = 8;
constexpr uint32_t kRoutesPerRow = kTopK + 1; // + shared slot
constexpr uint32_t kTileRows = 32;

struct MoeTileDescriptorH {
  uint32_t expert, rows;
};
struct MoeExpertParamsH {
  uint32_t input_size, output_size, experts, reserved0;
  uint64_t expert_stride_bytes_0, expert_stride_bytes_1;
};
struct Q4ParamsH {
  uint32_t output_size, input_size;
};

float bf16(uint16_t v) {
  uint32_t bits = uint32_t(v) << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
uint16_t to_bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, 4);
  return uint16_t(bits >> 16);
}
float gelu_tanh(float x) {
  const float inner = 0.79788456080286536f * (x + 0.044715f * x * x * x);
  const float t = inner > 10.0f ? 1.0f : inner < -10.0f ? -1.0f : tanhf(inner);
  return 0.5f * x * (1.0f + t);
}

// Packed Q4 slab layout ([weights][scales][biases], 256-column tiles):
// parameter index of output n, quant group g.
uint64_t paramIndex(uint32_t n, uint32_t g, uint32_t inputSize) {
  return (uint64_t(n / 256) * (inputSize / 64) + g) * 256 + n % 256;
}
uint64_t slabBytes(uint32_t outputSize, uint32_t inputSize) {
  const uint64_t elements = uint64_t(outputSize) * inputSize;
  return elements / 2 + 2 * (elements / 32);
}
void fillSlab(uint8_t *slab, uint32_t outputSize, uint32_t inputSize,
              std::mt19937 &rng) {
  const uint64_t elements = uint64_t(outputSize) * inputSize;
  std::uniform_int_distribution<int> byte(0, 255);
  for (uint64_t i = 0; i < elements / 2; ++i)
    slab[i] = uint8_t(byte(rng));
  uint16_t *scales = reinterpret_cast<uint16_t *>(slab + elements / 2);
  uint16_t *biases =
      reinterpret_cast<uint16_t *>(slab + elements / 2 + elements / 32);
  std::uniform_real_distribution<float> sc(0.005f, 0.02f), bi(-0.02f, 0.02f);
  for (uint64_t i = 0; i < elements / 64; ++i) {
    scales[i] = to_bf16(sc(rng));
    biases[i] = to_bf16(bi(rng));
  }
}
// Dequantize one weight of the packed slab (CPU parity reference).
float dequantAt(const uint8_t *slab, uint32_t outputSize, uint32_t inputSize,
                uint32_t n, uint32_t k) {
  const uint64_t elements = uint64_t(outputSize) * inputSize;
  const uint64_t p = paramIndex(n, k / 64, inputSize);
  const uint64_t nibble = p * 64 + k % 64;
  const uint8_t packed = slab[nibble / 2];
  const uint32_t code = (nibble & 1) ? packed >> 4 : packed & 15;
  const uint16_t *scales =
      reinterpret_cast<const uint16_t *>(slab + elements / 2);
  const uint16_t *biases =
      reinterpret_cast<const uint16_t *>(slab + elements / 2 + elements / 32);
  return float(code) * bf16(scales[p]) + bf16(biases[p]);
}

// Synthesize a grouped tile layout: rows x topK routes hashed over `unionN`
// experts, sorted by expert, chunked into tileRows-row tiles.
struct Layout {
  std::vector<MoeTileDescriptorH> tiles;
  std::vector<uint32_t> routes; // tiles * tileRows, ~0u padding
  uint32_t tileRows;
};
Layout makeLayout(uint32_t unionN, uint32_t tileRows) {
  struct Route {
    uint32_t expert, route;
  };
  std::vector<Route> all;
  for (uint32_t row = 0; row < kRows; ++row)
    for (uint32_t slot = 0; slot < kTopK; ++slot) {
      const uint32_t h = (row * 131 + slot * 17 + row * slot * 7 + 11);
      all.push_back({h % unionN, row * kRoutesPerRow + slot});
    }
  std::stable_sort(all.begin(), all.end(), [](const Route &a, const Route &b) {
    return a.expert < b.expert;
  });
  Layout layout;
  layout.tileRows = tileRows;
  size_t i = 0;
  while (i < all.size()) {
    const uint32_t expert = all[i].expert;
    size_t j = i;
    while (j < all.size() && all[j].expert == expert)
      ++j;
    for (size_t first = i; first < j; first += tileRows) {
      const uint32_t live = uint32_t(std::min<size_t>(tileRows, j - first));
      layout.tiles.push_back({expert, live});
      for (uint32_t r = 0; r < tileRows; ++r)
        layout.routes.push_back(first + r < j ? all[first + r].route : ~0u);
    }
    i = j;
  }
  return layout;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr, "usage: moe-prefill-bench PROD_LIB BENCH_LIB [reps]\n");
      return 64;
    }
    const uint32_t reps = argc > 3 ? atoi(argv[3]) : 20;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> prod =
        [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])]
                         error:&err];
    if (!prod) {
      fprintf(stderr, "prod library: %s\n",
              err.localizedDescription.UTF8String);
      return 70;
    }
    id<MTLLibrary> bench =
        [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])]
                         error:&err];
    if (!bench) {
      fprintf(stderr, "bench library: %s\n",
              err.localizedDescription.UTF8String);
      return 70;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    auto shared = [&](uint64_t bytes) {
      return [dev newBufferWithLength:std::max<uint64_t>(bytes, 16)
                              options:MTLResourceStorageModeShared];
    };
    auto pso = [&](id<MTLLibrary> lib, const char *name) {
      id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
      if (!fn) {
        fprintf(stderr, "missing kernel %s\n", name);
        exit(70);
      }
      id<MTLComputePipelineState> p =
          [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!p) {
        fprintf(stderr, "pso %s: %s\n", name,
                err.localizedDescription.UTF8String);
        exit(70);
      }
      return p;
    };

    // ---- Weights: per-expert packed slabs at each projection's stride ---
    std::mt19937 rng(7);
    const uint64_t upStride = slabBytes(kInter, kHidden);
    const uint64_t downStride = slabBytes(kHidden, kInter);
    id<MTLBuffer> gatePacked = shared(upStride * kExperts);
    id<MTLBuffer> upPacked = shared(upStride * kExperts);
    id<MTLBuffer> downPacked = shared(downStride * kExperts);
    id<MTLBuffer> sharedSlab = shared(upStride);
    {
      auto fillExperts = [&](id<MTLBuffer> buf, uint32_t out, uint32_t in) {
        uint8_t *base = (uint8_t *)buf.contents;
        const uint64_t stride = slabBytes(out, in);
        for (uint32_t e = 0; e < kExperts; ++e)
          fillSlab(base + e * stride, out, in, rng);
      };
      fillExperts(gatePacked, kInter, kHidden);
      fillExperts(upPacked, kInter, kHidden);
      fillExperts(downPacked, kHidden, kInter);
      fillSlab((uint8_t *)sharedSlab.contents, kInter, kHidden, rng);
    }

    // ---- Layouts (union 50 ≈ the live model's ~34-75) -------------------
    const Layout lay32 = makeLayout(50, 32);
    const Layout lay16 = makeLayout(50, 16);
    const Layout lay128 = makeLayout(128, 32);
    const uint32_t maxTiles = uint32_t(std::max(
        {lay32.tiles.size(), lay16.tiles.size(), lay128.tiles.size()}));
    const uint32_t groupedRows = maxTiles * kTileRows;
    fprintf(stderr,
            "tiles: union50 m32=%zu  union50 m16=%zu  union128 m32=%zu\n",
            lay32.tiles.size(), lay16.tiles.size(), lay128.tiles.size());

    id<MTLBuffer> input = shared(uint64_t(kRows) * kHidden * 2);
    {
      std::uniform_real_distribution<float> d(-1.f, 1.f);
      uint16_t *v = (uint16_t *)input.contents;
      for (uint64_t i = 0; i < uint64_t(kRows) * kHidden; ++i)
        v[i] = to_bf16(d(rng));
    }
    id<MTLBuffer> interm = shared(uint64_t(groupedRows) * kInter * 2);
    {
      std::uniform_real_distribution<float> d(-1.f, 1.f);
      uint16_t *v = (uint16_t *)interm.contents;
      for (uint64_t i = 0; i < uint64_t(groupedRows) * kInter; ++i)
        v[i] = to_bf16(d(rng));
    }
    id<MTLBuffer> gateOut = shared(uint64_t(groupedRows) * kInter * 2);
    id<MTLBuffer> upOut = shared(uint64_t(groupedRows) * kInter * 2);
    id<MTLBuffer> downOut = shared(uint64_t(groupedRows) * kHidden * 2);

    auto uploadLayout = [&](const Layout &lay, id<MTLBuffer> *tiles,
                            id<MTLBuffer> *routes) {
      *tiles = shared(lay.tiles.size() * sizeof(MoeTileDescriptorH));
      memcpy((*tiles).contents, lay.tiles.data(),
             lay.tiles.size() * sizeof(MoeTileDescriptorH));
      *routes = shared(lay.routes.size() * sizeof(uint32_t));
      memcpy((*routes).contents, lay.routes.data(),
             lay.routes.size() * sizeof(uint32_t));
    };
    id<MTLBuffer> tiles32, routes32, tiles16, routes16, tiles128, routes128;
    uploadLayout(lay32, &tiles32, &routes32);
    uploadLayout(lay16, &tiles16, &routes16);
    uploadLayout(lay128, &tiles128, &routes128);
    id<MTLBuffer> tileCount = shared(4);
    id<MTLBuffer> partials =
        shared(uint64_t(maxTiles) * 3 * 4 * kTileRows * 256 * 4);
    id<MTLBuffer> counters = shared(uint64_t(maxTiles) * 3 * 4);
    id<MTLBuffer> denseSums = shared(uint64_t(kRows) * (kHidden / 64) * 4);
    {
      const uint16_t *v = (const uint16_t *)input.contents;
      float *s = (float *)denseSums.contents;
      for (uint32_t r = 0; r < kRows; ++r)
        for (uint32_t g = 0; g < kHidden / 64; ++g) {
          float sum = 0;
          for (uint32_t k = 0; k < 64; ++k)
            sum += bf16(v[uint64_t(r) * kHidden + g * 64 + k]);
          s[r * (kHidden / 64) + g] = sum;
        }
    }
    id<MTLBuffer> denseOut = shared(uint64_t(kRows) * kInter * 2);
    (void)denseOut;

    MoeExpertParamsH upParams{kHidden, kInter, kExperts, kRoutesPerRow,
                              upStride, 0};
    MoeExpertParamsH downParams{kInter, kHidden, kExperts, kRoutesPerRow,
                                downStride, 0};

    auto runOnce = [&](id<MTLLibrary> lib, const char *name,
                       void (^bind)(id<MTLComputeCommandEncoder>)) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:pso(lib, name)];
      bind(e);
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "%s: %s\n", name,
                cb.error.localizedDescription.UTF8String);
        exit(70);
      }
    };
    auto time = [&](id<MTLLibrary> lib, const char *name,
                    void (^bind)(id<MTLComputeCommandEncoder>)) {
      std::vector<double> ts;
      for (uint32_t rep = 0; rep < reps + 2; ++rep) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso(lib, name)];
        bind(e);
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
          fprintf(stderr, "%s: %s\n", name,
                  cb.error.localizedDescription.UTF8String);
          exit(70);
        }
        if (rep >= 2)
          ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
      }
      std::sort(ts.begin(), ts.end());
      return ts[ts.size() / 2] * 1e6;
    };

    // Indirect gate/up signature (up_gelu: gate at 6, output at 7,
    // params at 8; production gate drops the gate arg: output at 6,
    // params at 7).
    auto bindUp = ^(id<MTLBuffer> routes, id<MTLBuffer> tiles,
                    uint32_t tilesN, uint32_t colTiles, uint32_t splits,
                    uint32_t tpg) {
      return ^(id<MTLComputeCommandEncoder> x) {
        [x setBuffer:input offset:0 atIndex:0];
        [x setBuffer:routes offset:0 atIndex:1];
        [x setBuffer:tiles offset:0 atIndex:2];
        [x setBuffer:tileCount offset:0 atIndex:3];
        [x setBuffer:upPacked offset:0 atIndex:4];
        [x setBuffer:sharedSlab offset:0 atIndex:5];
        [x setBuffer:gateOut offset:0 atIndex:6];
        [x setBuffer:upOut offset:0 atIndex:7];
        [x setBytes:&upParams length:sizeof(upParams) atIndex:8];
        if (splits) {
          [x setBuffer:partials offset:0 atIndex:9];
          [x setBuffer:counters offset:0 atIndex:10];
        }
        [x dispatchThreadgroups:MTLSizeMake(colTiles, tilesN,
                                          splits ? splits : 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
      };
    };
    auto bindGate = ^(id<MTLBuffer> routes, id<MTLBuffer> tiles,
                      uint32_t tilesN, uint32_t colTiles, uint32_t tpg,
                      bool benchAbi) {
      return ^(id<MTLComputeCommandEncoder> x) {
        [x setBuffer:input offset:0 atIndex:0];
        [x setBuffer:routes offset:0 atIndex:1];
        [x setBuffer:tiles offset:0 atIndex:2];
        [x setBuffer:tileCount offset:0 atIndex:3];
        [x setBuffer:gatePacked offset:0 atIndex:4];
        [x setBuffer:sharedSlab offset:0 atIndex:5];
        if (benchAbi) {
          [x setBuffer:gateOut offset:0 atIndex:6];
          [x setBuffer:gateOut offset:0 atIndex:7];
          [x setBytes:&upParams length:sizeof(upParams) atIndex:8];
        } else {
          [x setBuffer:gateOut offset:0 atIndex:6];
          [x setBytes:&upParams length:sizeof(upParams) atIndex:7];
        }
        [x dispatchThreadgroups:MTLSizeMake(colTiles, tilesN, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
      };
    };
    auto bindDown = ^(id<MTLBuffer> tiles, uint32_t tilesN,
                      uint32_t colTiles) {
      return ^(id<MTLComputeCommandEncoder> x) {
        [x setBuffer:interm offset:0 atIndex:0];
        [x setBuffer:tiles offset:0 atIndex:1];
        [x setBuffer:tileCount offset:0 atIndex:2];
        [x setBuffer:downPacked offset:0 atIndex:3];
        [x setBuffer:sharedSlab offset:0 atIndex:4];
        [x setBuffer:downOut offset:0 atIndex:5];
        [x setBytes:&downParams length:sizeof(downParams) atIndex:6];
        [x dispatchThreadgroups:MTLSizeMake(colTiles, tilesN, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      };
    };

    const uint32_t tilesN = uint32_t(lay32.tiles.size());
    const uint32_t tiles16N = uint32_t(lay16.tiles.size());
    const uint32_t tiles128N = uint32_t(lay128.tiles.size());
    *(uint32_t *)tileCount.contents = tilesN;

    // ---- Parity --------------------------------------------------------
    // CPU reference for tile 0 row 0 of the m32 layout.
    {
      const MoeTileDescriptorH &t = lay32.tiles[0];
      const uint8_t *gSlab =
          (const uint8_t *)gatePacked.contents + t.expert * upStride;
      const uint8_t *uSlab =
          (const uint8_t *)upPacked.contents + t.expert * upStride;
      const uint16_t *x = (const uint16_t *)input.contents;
      const uint32_t inRow = lay32.routes[0] / kRoutesPerRow;
      std::vector<float> gateRef(kInter), upRef(kInter);
      for (uint32_t n = 0; n < kInter; ++n) {
        float gs = 0, us = 0;
        for (uint32_t k = 0; k < kHidden; ++k) {
          const float xv = bf16(x[uint64_t(inRow) * kHidden + k]);
          gs += xv * dequantAt(gSlab, kInter, kHidden, n, k);
          us += xv * dequantAt(uSlab, kInter, kHidden, n, k);
        }
        gateRef[n] = bf16(to_bf16(gs));
        upRef[n] = bf16(to_bf16(gelu_tanh(bf16(to_bf16(gs))) *
                                bf16(to_bf16(us))));
      }

      runOnce(prod, richengine::ops::kPrefillMoeExpertQ4N256IndirectM32.data(),
              bindGate(routes32, tiles32, tilesN, 3, 256, false));
      const uint16_t *g = (const uint16_t *)gateOut.contents;
      double worst = 0;
      for (uint32_t n = 0; n < kInter; ++n)
        worst = std::max(
            worst, std::fabs(double(bf16(g[n])) - double(gateRef[n])));
      double refMax = 0;
      for (uint32_t n = 0; n < kInter; ++n)
        refMax = std::max(refMax, std::fabs(double(gateRef[n])));
      printf("parity gate tile0 row0: max|gpu-ref|=%.4g (ref max %.4g)\n",
             worst, refMax);
      if (!(worst < 0.005 + 0.02 * refMax)) {
        fprintf(stderr, "gate parity failed\n");
        return 1;
      }
      runOnce(prod, richengine::ops::kPrefillMoeExpertQ4N256UpGeluIndirectM32.data(),
              bindUp(routes32, tiles32, tilesN, 3, 0, 256));
      const uint16_t *u = (const uint16_t *)upOut.contents;
      worst = 0;
      for (uint32_t n = 0; n < kInter; ++n)
        worst =
            std::max(worst, std::fabs(double(bf16(u[n])) - double(upRef[n])));
      refMax = 0;
      for (uint32_t n = 0; n < kInter; ++n)
        refMax = std::max(refMax, std::fabs(double(upRef[n])));
      printf("parity up_gelu tile0 row0: max|gpu-ref|=%.4g (ref max %.4g)\n",
             worst, refMax);
      if (!(worst < 0.005 + 0.02 * refMax)) {
        fprintf(stderr, "up parity failed\n");
        return 1;
      }
      // Variants reproduce the production up output (bitwise for the
      // same-order tiles; small fp drift for the split).
      struct Variant {
        const char *name;
        uint32_t colTiles, splits, tpg;
        double tol;
      };
      const Variant vs[] = {
          {richengine::ops::kBenchMoeN128UpGeluIndirectM32.data(), 6, 0, 256, 0},
          {richengine::ops::kBenchMoeN64UpGeluIndirectM32.data(), 12, 0, 256, 0},
          {richengine::ops::kBenchMoeN256Sg4UpGeluIndirectM32.data(), 3, 0, 128, 0},
          {richengine::ops::kBenchMoeN128Sg4UpGeluIndirectM32.data(), 6, 0, 128, 0},
          {richengine::ops::kBenchMoeUpGeluKsplit2M32.data(), 3, 2, 256, 0.02},
          {richengine::ops::kBenchMoeUpGeluKsplit4M32.data(), 3, 4, 256, 0.02},
          {richengine::ops::kBenchMoeN256UpGeluPipeM32.data(), 3, 0, 256, 0},
      };
      std::vector<uint16_t> ref(kInter);
      memcpy(ref.data(), upOut.contents, kInter * 2);
      for (const Variant &v : vs) {
        memset(counters.contents, 0, counters.length);
        memset(upOut.contents, 0, uint64_t(groupedRows) * kInter * 2);
        runOnce(bench, v.name,
                bindUp(routes32, tiles32, tilesN, v.colTiles, v.splits,
                       v.tpg));
        double d = 0, refMag = 0;
        for (uint32_t n = 0; n < kInter; ++n) {
          d = std::max(d, std::fabs(double(bf16(u[n])) - double(bf16(ref[n]))));
          refMag = std::max(refMag, std::fabs(double(bf16(ref[n]))));
        }
        printf("parity %-44s tile0 row0: %.4g\n", v.name, d);
        if (d > v.tol + 0.01 * refMag) {
          fprintf(stderr, "%s parity failed\n", v.name);
          return 1;
        }
      }
    }

    // ---- Timed matrix ---------------------------------------------------
    printf("\n%-46s %8s %10s\n", "kernel", "us", "GB/s(w)");
    auto report = [&](const char *name, double us, double gb) {
      printf("%-46s %8.0f %10.0f\n", name, us,
             us > 0 ? gb / (us * 1e-6) : 0);
    };

    struct Row {
      const char *label;
      id<MTLLibrary> lib;
      const char *name;
      id<MTLBuffer> routes, tiles;
      uint32_t tilesN, colTiles, splits, tpg;
    };
    const Row rows[] = {
        {"up_gelu n256 (prod)", prod,
         richengine::ops::kPrefillMoeExpertQ4N256UpGeluIndirectM32.data(), routes32,
         tiles32, tilesN, 3, 0, 256},
        {"up_gelu n128", bench, richengine::ops::kBenchMoeN128UpGeluIndirectM32.data(),
         routes32, tiles32, tilesN, 6, 0, 256},
        {"up_gelu n64", bench, richengine::ops::kBenchMoeN64UpGeluIndirectM32.data(),
         routes32, tiles32, tilesN, 12, 0, 256},
        {"up_gelu n256 sg4", bench,
         richengine::ops::kBenchMoeN256Sg4UpGeluIndirectM32.data(), routes32, tiles32,
         tilesN, 3, 0, 128},
        {"up_gelu n128 sg4", bench,
         richengine::ops::kBenchMoeN128Sg4UpGeluIndirectM32.data(), routes32, tiles32,
         tilesN, 6, 0, 128},
        {"up_gelu m16 tiles", bench,
         richengine::ops::kBenchMoeN256UpGeluIndirectM16.data(), routes16, tiles16,
         tiles16N, 3, 0, 256},
        {"up_gelu ksplit2", bench, richengine::ops::kBenchMoeUpGeluKsplit2M32.data(),
         routes32, tiles32, tilesN, 3, 2, 256},
        {"up_gelu ksplit4", bench, richengine::ops::kBenchMoeUpGeluKsplit4M32.data(),
         routes32, tiles32, tilesN, 3, 4, 256},
        {"up_gelu n256 pipelined", bench,
         richengine::ops::kBenchMoeN256UpGeluPipeM32.data(), routes32, tiles32, tilesN, 3,
         0, 256},
    };
    for (const Row &r : rows) {
      *(uint32_t *)tileCount.contents = r.tilesN;
      memset(counters.contents, 0, counters.length);
      const double us = time(r.lib, r.name,
                             bindUp(r.routes, r.tiles, r.tilesN, r.colTiles,
                                    r.splits, r.tpg));
      report(r.label, us, double(upStride) * r.tilesN / 1e9);
    }
    for (const auto &r : (Row[]){
             {"gate n256 (prod)", prod,
              richengine::ops::kPrefillMoeExpertQ4N256IndirectM32.data(), routes32, tiles32,
              tilesN, 3, 0, 256},
             {"gate n128", bench, richengine::ops::kBenchMoeN128GateIndirectM32.data(),
              routes32, tiles32, tilesN, 6, 0, 256},
             {"gate n64", bench, richengine::ops::kBenchMoeN64GateIndirectM32.data(),
              routes32, tiles32, tilesN, 12, 0, 256},
         }) {
      *(uint32_t *)tileCount.contents = r.tilesN;
      const double us =
          time(r.lib, r.name,
               bindGate(r.routes, r.tiles, r.tilesN, r.colTiles, r.tpg,
                        r.lib == bench));
      report(r.label, us, double(upStride) * r.tilesN / 1e9);
    }

    // Down pass: dense grouped input, N = hidden.
    *(uint32_t *)tileCount.contents = tilesN;
    for (const auto &r : (const char *[]){
             richengine::ops::kPrefillMoeExpertQ4N256M32.data(), richengine::ops::kBenchMoeN128M32.data(),
             richengine::ops::kBenchMoeN64M32.data()}) {
      const bool isBench = r[0] == 'b';
      const uint32_t colTiles = strstr(r, "n64")   ? 44
                                : strstr(r, "n128") ? 22
                                                    : 11;
      const double us = time(isBench ? bench : prod, r,
                             bindDown(tiles32, tilesN, colTiles));
      report(r, us, double(downStride) * tilesN / 1e9);
    }

    // Union sweep on the production up kernel: saturation vs tile count.
    for (const auto &[label, tilesBuf, routesBuf, n] :
         std::vector<std::tuple<const char *, id<MTLBuffer>, id<MTLBuffer>,
                                uint32_t>>{
             {"up_gelu union128 m32", tiles128, routes128, tiles128N}}) {
      *(uint32_t *)tileCount.contents = n;
      const double us =
          time(prod, richengine::ops::kPrefillMoeExpertQ4N256UpGeluIndirectM32.data(),
               bindUp(routesBuf, tilesBuf, n, 3, 0, 256));
      report(label, us, double(upStride) * n / 1e9);
    }

    // Dense comparison: the shared prefill tile over the same projection
    // (M=256, N=768, K=2816 — 24 threadgroups, no grouping), then the
    // trunk's real dense projections. The dense kernel re-reads the slab
    // once per 32-row tile (8x at M=256); GB/s counts the real traffic.
    printf("\n");
    {
      struct DenseShape {
        const char *label;
        uint32_t n, k;
      };
      const DenseShape shapes[] = {
          {"dense N768 K2816 (per-expert N)", kInter, kHidden},
          {"dense N2304 K2816 (shared gate/up)", 2304, 2816},
          {"dense N2816 K2304 (shared down)", 2816, 2304},
          {"dense N8192 K2816 (local qkv)", 8192, 2816},
          {"dense N10240 K2816 (global qkv)", 10240, 2816},
          {"dense N2816 K8192 (global o proj)", 2816, 8192},
      };
      // Tile variants over the same weights/input/sums: prod n256 and n128
      // (staged sums, 8 simdgroups), prod n128_sg4, then the bench's
      // pipelined, sg4 and 64-row forms.
      struct DenseVariant {
        const char *label;
        id<MTLLibrary> lib;
        const char *kernel;
        uint32_t tileM, tileN, threads;
      };
      const DenseVariant variants[] = {
          {" n256 (prod)", prod, richengine::ops::kPrefillLinearQ4N256.data(), 32, 256, 256},
          {" n128 (prod)", prod, richengine::ops::kPrefillLinearQ4N128.data(), 32, 128, 256},
          {" n128_sg4 (prod)", prod, richengine::ops::kPrefillLinearQ4N128Sg4.data(), 32, 128,
           128},
          {" n256 pipelined", bench, richengine::ops::kBenchDenseQ4N256Pipe.data(), 32, 256,
           256},
          {" n256 sg4", bench, richengine::ops::kBenchDenseQ4N256Sg4.data(), 32, 256, 128},
          {" m64 n256 sg4", bench, richengine::ops::kBenchDenseQ4N256M64Sg4.data(), 64, 256,
           128},
      };
      id<MTLBuffer> bigInput = shared(uint64_t(kRows) * 8192 * 2);
      id<MTLBuffer> bigSums = shared(uint64_t(kRows) * (8192 / 64) * 4);
      id<MTLBuffer> bigOut = shared(uint64_t(kRows) * 10240 * 2);
      id<MTLBuffer> bigW = shared(slabBytes(10240, 8192));
      {
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        uint16_t *v = (uint16_t *)bigInput.contents;
        for (uint64_t i = 0; i < uint64_t(kRows) * 8192; ++i)
          v[i] = to_bf16(d(rng));
      }
      // Parity: variants must match prod n256 bitwise on the first outputs
      // (same per-element accumulation order; only staging/issue differ).
      {
        const DenseShape &s = shapes[0];
        fillSlab((uint8_t *)bigW.contents, s.n, s.k, rng);
        const uint64_t elements = uint64_t(s.n) * s.k;
        const uint16_t *v = (const uint16_t *)bigInput.contents;
        float *sums = (float *)bigSums.contents;
        for (uint32_t r = 0; r < kRows; ++r)
          for (uint32_t g = 0; g < s.k / 64; ++g) {
            float sum = 0;
            for (uint32_t k = 0; k < 64; ++k)
              sum += bf16(v[uint64_t(r) * s.k + g * 64 + k]);
            sums[r * (s.k / 64) + g] = sum;
          }
        Q4ParamsH params{s.n, s.k};
        auto bindDense = ^(const DenseVariant &v2) {
          return ^(id<MTLComputeCommandEncoder> x) {
            [x setBuffer:bigInput offset:0 atIndex:0];
            [x setBuffer:bigW offset:0 atIndex:1];
            [x setBuffer:bigW offset:elements / 2 atIndex:2];
            [x setBuffer:bigW offset:elements / 2 + elements / 32 atIndex:3];
            [x setBuffer:bigOut offset:0 atIndex:4];
            [x setBuffer:bigSums offset:0 atIndex:5];
            [x setBytes:&params length:sizeof(params) atIndex:6];
            [x dispatchThreadgroups:MTLSizeMake(kRows / v2.tileM,
                                              s.n / v2.tileN, 1)
                threadsPerThreadgroup:MTLSizeMake(v2.threads, 1, 1)];
          };
        };
        runOnce(prod, richengine::ops::kPrefillLinearQ4N256.data(), bindDense(variants[0]));
        std::vector<uint16_t> ref(kRows * s.n);
        memcpy(ref.data(), bigOut.contents, ref.size() * 2);
        for (const DenseVariant &v2 : variants) {
          memset(bigOut.contents, 0, bigOut.length);
          runOnce(v2.lib, v2.kernel, bindDense(v2));
          const uint16_t *o = (const uint16_t *)bigOut.contents;
          uint32_t diffs = 0;
          for (uint64_t i = 0; i < uint64_t(kRows) * s.n; ++i)
            diffs += o[i] != ref[i];
          printf("parity dense%s: %u/%llu differ\n", v2.label, diffs,
                 (unsigned long long)(uint64_t(kRows) * s.n));
          if (diffs) {
            fprintf(stderr, "dense variant %s mismatch\n", v2.kernel);
            return 1;
          }
        }
      }
      for (const DenseShape &s : shapes) {
        fillSlab((uint8_t *)bigW.contents, s.n, s.k, rng);
        const uint64_t elements = uint64_t(s.n) * s.k;
        const uint16_t *v = (const uint16_t *)bigInput.contents;
        float *sums = (float *)bigSums.contents;
        for (uint32_t r = 0; r < kRows; ++r)
          for (uint32_t g = 0; g < s.k / 64; ++g) {
            float sum = 0;
            for (uint32_t k = 0; k < 64; ++k)
              sum += bf16(v[uint64_t(r) * s.k + g * 64 + k]);
            sums[r * (s.k / 64) + g] = sum;
          }
        Q4ParamsH params{s.n, s.k};
        for (const DenseVariant &v2 : variants) {
          const double us = time(
              v2.lib, v2.kernel,
              ^(id<MTLComputeCommandEncoder> x) {
                [x setBuffer:bigInput offset:0 atIndex:0];
                [x setBuffer:bigW offset:0 atIndex:1];
                [x setBuffer:bigW offset:elements / 2 atIndex:2];
                [x setBuffer:bigW
                    offset:elements / 2 + elements / 32
                    atIndex:3];
                [x setBuffer:bigOut offset:0 atIndex:4];
                [x setBuffer:bigSums offset:0 atIndex:5];
                [x setBytes:&params length:sizeof(params) atIndex:6];
                [x dispatchThreadgroups:MTLSizeMake(kRows / v2.tileM,
                                                  s.n / v2.tileN, 1)
                    threadsPerThreadgroup:MTLSizeMake(v2.threads, 1, 1)];
              });
          // The slab is re-read once per row tile.
          char label[96];
          snprintf(label, sizeof(label), "%s%s", s.label, v2.label);
          report(label, us,
                 double(elements) / 2 * (kRows / v2.tileM) / 1e9);
        }
        printf("\n");
      }
    }

    // Chain: gate -> up_gelu -> down, the layer's MoE as encoded.
    {
      *(uint32_t *)tileCount.contents = tilesN;
      std::vector<double> ts;
      for (uint32_t rep = 0; rep < reps + 2; ++rep) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso(
            prod, richengine::ops::kPrefillMoeExpertQ4N256IndirectM32.data())];
        bindGate(routes32, tiles32, tilesN, 3, 256, false)(e);
        [e setComputePipelineState:pso(
            prod, richengine::ops::kPrefillMoeExpertQ4N256UpGeluIndirectM32.data())];
        bindUp(routes32, tiles32, tilesN, 3, 0, 256)(e);
        [e setComputePipelineState:pso(prod,
                                       richengine::ops::kPrefillMoeExpertQ4N256M32.data())];
        bindDown(tiles32, tilesN, 11)(e);
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (rep >= 2)
          ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
      }
      std::sort(ts.begin(), ts.end());
      const double us = ts[ts.size() / 2] * 1e6;
      report("MoE chain gate+up+down (prod)", us,
             (2 * double(upStride) + downStride) * tilesN / 1e9);
      printf("\nMoE chain x30 layers: %.1f ms  (measured trunk ~115 ms/step;"
             " remainder is attention, dense FFN, norms)\n",
             us * 30 / 1000);
    }
    return 0;
  }
}

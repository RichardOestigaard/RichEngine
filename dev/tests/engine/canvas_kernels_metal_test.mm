// A/B parity + micro-benchmark for the DiffusionGemma canvas logit tail:
//   canvas_row_stats        vs  canvas_row_stats_fused   (single-pass online
//   canvas_soft_embed_topk  vs  canvas_soft_embed_histogram (256-bin
//                                                        threshold bracket)
// on a real-size 256 x 262144 fp32 logits buffer over three synthetic
// distributions per band of rows — wide gaussian, near-delta, near-uniform —
// each checked against a CPU fp64 reference implementing the same selection
// rule, then every kernel timed with MTLCommandBuffer GPU timestamps.
//
// usage: canvas-kernels METALLIB [reps]
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
#include <vector>

static const uint32_t kRows = 256, kVocab = 262144, kHidden = 2816,
                      kQGroups = 44, kTopK = 64;
static const float kCap = 30.0f, kInvT = 1.3f;
static const float kEmbedScale = 53.05432f; // sqrt(2816)

struct RowStatsParams {
  uint32_t vocabulary, seed;
};
struct RowStatsFusedParams {
  uint32_t vocabulary, seed;
  float cap, scale;
};
struct SoftEmbedParams {
  uint32_t vocabulary, topK;
};
struct SoftEmbedHistParams {
  uint32_t vocabulary, topK;
  float cap, scale;
};

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

static float xform(float l, float cap, float scale) {
  if (cap > 0.0f)
    l = cap * std::tanh(l / cap);
  if (scale != 0.0f)
    l *= scale;
  return l;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2) {
      fprintf(stderr, "usage: canvas-kernels METALLIB [reps]\n");
      return 64;
    }
    const uint32_t reps = argc > 2 ? atoi(argv[2]) : 20;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib =
        [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])]
                         error:&err];
    if (!lib) {
      fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String);
      return 70;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    const uint64_t logitCount = uint64_t(kRows) * kVocab;

    auto shared = [&](uint64_t bytes) {
      return [dev newBufferWithLength:bytes
                              options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> logitsRaw = shared(logitCount * 2);
    id<MTLBuffer> logitsOld = shared(logitCount * 2);
    id<MTLBuffer> weights = shared(uint64_t(kVocab) * kHidden / 2);
    id<MTLBuffer> scales = shared(uint64_t(kVocab) * kQGroups * 2);
    id<MTLBuffer> biases = shared(uint64_t(kVocab) * kQGroups * 2);
    id<MTLBuffer> outOld = shared(uint64_t(kRows) * kHidden * 2);
    id<MTLBuffer> outNew = shared(uint64_t(kRows) * kHidden * 2);
    id<MTLBuffer> sampledOld = shared(kRows * 4), sampledNew = shared(kRows * 4);
    id<MTLBuffer> argmaxOld = shared(kRows * 4), argmaxNew = shared(kRows * 4);
    id<MTLBuffer> entropyOld = shared(kRows * 4), entropyNew = shared(kRows * 4);

    // Canvas logits are bf16 (the head writes bf16; the kernels
    // transform on load).
    uint16_t *lr = (uint16_t *)logitsRaw.contents;
    auto lrVal = [&](uint32_t row, uint32_t v) {
      return bf16(lr[uint64_t(row) * kVocab + v]);
    };
    // Row bands: wide gaussian / near-delta / near-uniform, fixed seed.
    {
      std::mt19937 rng(11);
      std::normal_distribution<float> g(0.f, 4.f), gn(0.f, 0.5f),
          gu(0.f, 0.01f);
      for (uint32_t row = 0; row < kRows; ++row) {
        uint16_t *r = lr + uint64_t(row) * kVocab;
        for (uint32_t v = 0; v < kVocab; ++v)
          r[v] = to_bf16(row < 86 ? g(rng) : row < 171 ? gn(rng) : gu(rng));
        if (row >= 86 && row < 171)
          r[(row * 7919) % kVocab] =
              to_bf16(bf16(r[(row * 7919) % kVocab]) + 25.f); // near-delta spike
      }
      std::mt19937 wrng(13);
      std::uniform_int_distribution<int> bytes(0, 255);
      uint8_t *w = (uint8_t *)weights.contents;
      for (uint64_t i = 0; i < uint64_t(kVocab) * kHidden / 2; ++i)
        w[i] = uint8_t(bytes(wrng));
      uint16_t *s = (uint16_t *)scales.contents,
               *b = (uint16_t *)biases.contents;
      std::uniform_real_distribution<float> sc(0.01f, 0.06f),
          bi(-0.05f, 0.05f);
      for (uint64_t i = 0; i < uint64_t(kVocab) * kQGroups; ++i) {
        s[i] = to_bf16(sc(wrng));
        b[i] = to_bf16(bi(wrng));
      }
    }
    memcpy(logitsOld.contents, lr, logitCount * 2);

    auto pso = [&](NSString *name) {
      id<MTLComputePipelineState> p = [dev
          newComputePipelineStateWithFunction:[lib newFunctionWithName:name]
                                        error:&err];
      if (!p) {
        fprintf(stderr, "pso %s: %s\n", name.UTF8String,
                err.localizedDescription.UTF8String);
        exit(70);
      }
      return p;
    };
    auto encode = [&](id<MTLComputeCommandEncoder> e, NSString *name,
                      void (^bind)(id<MTLComputeCommandEncoder>)) {
      [e setComputePipelineState:pso(name)];
      bind(e);
    };

    // ---- Old path: softcap + scale in place on logitsOld, then the
    // two-pass stats and the bisection soft-embed.
    const uint32_t count = uint32_t(logitCount);
    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      encode(e, @(richengine::ops::kCanvasLogitSoftcap.data()), ^(id<MTLComputeCommandEncoder> x) {
        [x setBuffer:logitsOld offset:0 atIndex:0];
        [x setBytes:&kCap length:4 atIndex:1];
        [x setBytes:&count length:4 atIndex:2];
        [x dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      });
      encode(e, @(richengine::ops::kCanvasLogitsScale.data()), ^(id<MTLComputeCommandEncoder> x) {
        [x setBuffer:logitsOld offset:0 atIndex:0];
        [x setBytes:&kInvT length:4 atIndex:1];
        [x setBytes:&count length:4 atIndex:2];
        [x dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      });
      encode(e, @(richengine::ops::kCanvasRowStats.data()), ^(id<MTLComputeCommandEncoder> x) {
        RowStatsParams p{kVocab, 77};
        [x setBuffer:logitsOld offset:0 atIndex:0];
        [x setBuffer:sampledOld offset:0 atIndex:1];
        [x setBuffer:argmaxOld offset:0 atIndex:2];
        [x setBuffer:entropyOld offset:0 atIndex:3];
        [x setBytes:&p length:sizeof(p) atIndex:4];
        [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      });
      encode(e, @(richengine::ops::kCanvasSoftEmbedTopk.data()), ^(id<MTLComputeCommandEncoder> x) {
        SoftEmbedParams p{kVocab, kTopK};
        [x setBuffer:logitsOld offset:0 atIndex:0];
        [x setBuffer:weights offset:0 atIndex:1];
        [x setBuffer:scales offset:0 atIndex:2];
        [x setBuffer:biases offset:0 atIndex:3];
        [x setBuffer:outOld offset:0 atIndex:4];
        [x setBytes:&p length:sizeof(p) atIndex:5];
        [x setBytes:&kEmbedScale length:4 atIndex:6];
        [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      });
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "old path: %s\n",
                cb.error.localizedDescription.UTF8String);
        return 70;
      }
    }
    // ---- New path: fused kernels on the raw logits, transform in-register.
    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      encode(e, @(richengine::ops::kCanvasRowStatsFused.data()), ^(id<MTLComputeCommandEncoder> x) {
        RowStatsFusedParams p{kVocab, 77, kCap, kInvT};
        [x setBuffer:logitsRaw offset:0 atIndex:0];
        [x setBuffer:sampledNew offset:0 atIndex:1];
        [x setBuffer:argmaxNew offset:0 atIndex:2];
        [x setBuffer:entropyNew offset:0 atIndex:3];
        [x setBytes:&p length:sizeof(p) atIndex:4];
        [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      });
      encode(e, @(richengine::ops::kCanvasSoftEmbedHistogram.data()),
             ^(id<MTLComputeCommandEncoder> x) {
               SoftEmbedHistParams p{kVocab, kTopK, kCap, kInvT};
               [x setBuffer:logitsRaw offset:0 atIndex:0];
               [x setBuffer:weights offset:0 atIndex:1];
               [x setBuffer:scales offset:0 atIndex:2];
               [x setBuffer:biases offset:0 atIndex:3];
               [x setBuffer:outNew offset:0 atIndex:4];
               [x setBytes:&p length:sizeof(p) atIndex:5];
               [x setBytes:&kEmbedScale length:4 atIndex:6];
               [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
             });
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "new path: %s\n",
                cb.error.localizedDescription.UTF8String);
        return 70;
      }
    }

    // ---- CPU fp64 reference over the transformed raw logits. The tables
    // use the packed format's 256-row tile order: each plane holds
    // [vocab/256][groups][256] units of its group size.
    auto dequant = [&](uint64_t v, uint32_t d) -> double {
      const uint64_t g =
          (v / 256 * kQGroups + d / 64) * 256 + v % 256;
      const uint8_t packed =
          ((const uint8_t *)weights.contents)[g * 32 + (d % 64) / 2];
      const double code = double((packed >> ((d & 1) * 4)) & 15);
      return code * bf16(((const uint16_t *)scales.contents)[g]) +
             bf16(((const uint16_t *)biases.contents)[g]);
    };
    const uint32_t *smO = (const uint32_t *)sampledOld.contents;
    const uint32_t *smN = (const uint32_t *)sampledNew.contents;
    const uint32_t *amO = (const uint32_t *)argmaxOld.contents;
    const uint32_t *amN = (const uint32_t *)argmaxNew.contents;
    const float *enO = (const float *)entropyOld.contents;
    const float *enN = (const float *)entropyNew.contents;
    const uint16_t *oO = (const uint16_t *)outOld.contents;
    const uint16_t *oN = (const uint16_t *)outNew.contents;

    double maxEntO = 0, maxEntN = 0, maxProbDiff = 0, embO = 0, embN = 0,
           embON = 0;
    uint32_t argmaxMismatch = 0, sampleMismatch = 0, rowsKept0 = 0;
    uint32_t worstRow = 0, worstKB = 0, worstKH = 0;
    double worstMB = 0, worstMH = 0;
    std::vector<float> row(kVocab), rowOld(kVocab);
    std::vector<double> prob(kVocab), probOld(kVocab);
    std::vector<double> refBisect(kHidden), refHist(kHidden);
    for (uint32_t r = 0; r < kRows; ++r) {
      auto src = [&](uint32_t v) { return lrVal(r, v); };
      double mx = -INFINITY, mn = INFINITY;
      double mxO = -INFINITY;
      for (uint32_t v = 0; v < kVocab; ++v) {
        row[v] = xform(src(v), kCap, kInvT);
        // The unfused path stores bf16 between its two passes:
        // bf16(scale * bf16(cap*tanh(l/cap))).
        rowOld[v] = bf16(to_bf16(bf16(to_bf16(kCap * std::tanh(src(v) / kCap))) * kInvT));
        mx = std::max(mx, double(row[v]));
        mxO = std::max(mxO, double(rowOld[v]));
        mn = std::min(mn, double(row[v]));
      }
      double z = 0, w = 0, zO = 0, wO = 0;
      uint32_t am = 0, amOref = 0;
      double bv = -INFINITY, bvO = -INFINITY;
      for (uint32_t v = 0; v < kVocab; ++v) {
        const double e = std::exp(row[v] - mx);
        prob[v] = e;
        z += e;
        w += e * row[v];
        const double eO = std::exp(rowOld[v] - mxO);
        probOld[v] = eO;
        zO += eO;
        wO += eO * rowOld[v];
        if (row[v] > bv) {
          bv = row[v];
          am = v;
        }
        if (rowOld[v] > bvO) {
          bvO = rowOld[v];
          amOref = v;
        }
      }
      for (uint32_t v = 0; v < kVocab; ++v) {
        prob[v] /= z;
        probOld[v] /= zO;
      }
      const double ent = std::log(z) + mx - w / z;
      const double entO = std::log(zO) + mxO - wO / zO;
      if (amOref != amO[r] || am != amN[r])
        ++argmaxMismatch;
      maxEntO = std::max(maxEntO, std::fabs(entO - enO[r]));
      maxEntN = std::max(maxEntN, std::fabs(ent - enN[r]));
      // The draws may land on different indices within an ulp of the
      // threshold: compare the drawn probabilities instead.
      const double pO = probOld[smO[r]];
      const double pN = prob[smN[r]];
      if (smO[r] != smN[r])
        ++sampleMismatch;
      maxProbDiff = std::max(maxProbDiff, std::fabs(pO - pN));

      // Soft-embed references, each reproducing its kernel's selection:
      // old — the exact 32-iteration bisection on [0,1] counting p >= mid,
      // kept = p >= hi; new — 256 bins over [mn, mx], kept = l >= l_tau.
      double lo = 0, hi = 1;
      for (uint32_t it = 0; it < 32; ++it) {
        const double mid = (lo + hi) * 0.5;
        uint64_t c = 0;
        for (uint32_t v = 0; v < kVocab; ++v)
          c += prob[v] >= mid;
        if (c > kTopK)
          lo = mid;
        else
          hi = mid;
      }
      std::fill(refBisect.begin(), refBisect.end(), 0.0);
      double massB = 0;
      uint32_t keptB = 0;
      for (uint32_t v = 0; v < kVocab; ++v) {
        if (prob[v] >= hi) {
          massB += prob[v];
          ++keptB;
          for (uint32_t d = 0; d < kHidden; ++d)
            refBisect[d] += prob[v] * dequant(v, d);
        }
      }
      // New-rule reference: emulate the kernel's fp32 binning and tau so
      // kept sets match bitwise; the mass stays fp64 for the embed check.
      const float mnF = float(mn), mxF = float(mx);
      const float rangeF = mxF - mnF;
      const float invwF = rangeF > 0 ? 256.0f / rangeF : 0.0f;
      uint32_t hist[256] = {};
      for (uint32_t v = 0; v < kVocab; ++v) {
        const uint32_t b =
            std::min(uint32_t((row[v] - mnF) * invwF), 255u);
        ++hist[b];
      }
      uint32_t cum = 0, bracket = 256;
      for (int b = 255; b >= 0; --b) {
        const uint32_t next = cum + hist[b];
        if (next > kTopK && bracket == 256)
          bracket = b + 1;
        cum = next;
      }
      const float tauL = bracket == 256
                             ? INFINITY
                             : mnF + float(bracket) * (rangeF / 256.0f);
      std::fill(refHist.begin(), refHist.end(), 0.0);
      double massH = 0;
      uint32_t kept = 0;
      for (uint32_t v = 0; v < kVocab; ++v) {
        if (row[v] >= tauL && kept < 1024) {
          massH += prob[v];
          ++kept;
          for (uint32_t d = 0; d < kHidden; ++d)
            refHist[d] += prob[v] * dequant(v, d);
        }
      }
      if (!kept)
        ++rowsKept0;
      const double normB = massB > 0 ? kEmbedScale / massB : 0;
      const double normH = massH > 0 ? kEmbedScale / massH : 0;
      // Per-row L2 relative error: a boundary flip in the kept set changes
      // every dim by ~one term; the L2 ratio catches a systematically
      // different selection while tolerating a few edge flips.
      double dO = 0, dN = 0, rB2 = 0, rH2 = 0;
      for (uint32_t d = 0; d < kHidden; ++d) {
        const double rB = refBisect[d] * normB, rH = refHist[d] * normH;
        const double gO = bf16(oO[r * kHidden + d]),
                     gN = bf16(oN[r * kHidden + d]);
        dO += (gO - rB) * (gO - rB);
        dN += (gN - rH) * (gN - rH);
        rB2 += rB * rB;
        rH2 += rH * rH;
        embON = std::max(embON, std::fabs(gO - gN));
      }
      const double eO = dO / (1.0 + rB2), eN = dN / (1.0 + rH2);
      if (eO > embO)
        embO = eO;
      if (eN > embN) {
        embN = eN;
        worstRow = r;
        worstKB = keptB;
        worstKH = kept;
        worstMB = massB;
        worstMH = massH;
      }
    }
    printf("  worst new row %u: keptB=%u massB=%.4g  keptH=%u massH=%.4g\n",
           worstRow, worstKB, worstMB, worstKH, worstMH);
    printf("parity (transformed logits, cap=%.1f invT=%.2f):\n", kCap, kInvT);
    printf("  argmax mismatches vs ref:   %u\n", argmaxMismatch);
    printf("  entropy |old-ref|max=%.3g  |new-ref|max=%.3g\n", maxEntO,
           maxEntN);
    printf("  sampled-index diffs old/new: %u rows, max |p_old-p_new|=%.3g\n",
           sampleMismatch, maxProbDiff);
    printf("  soft-embed L2err old-vs-refBisect=%.4g  new-vs-refHist=%.4g  "
           "|old-new|absmax=%.4g  (rows keeping 0: %u)\n",
           embO, embN, embON, rowsKept0);

    // ---- Timing: median GPU seconds over `reps` dispatches each.
    auto time = [&](NSString *name,
                    void (^bind)(id<MTLComputeCommandEncoder>)) {
      // warmup + correctness-neutral repeat
      std::vector<double> ts;
      for (uint32_t rep = 0; rep < reps + 2; ++rep) {
        id<MTLCommandBuffer> t = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [t computeCommandEncoder];
        encode(e, name, bind);
        [e endEncoding];
        [t commit];
        [t waitUntilCompleted];
        if (rep >= 2)
          ts.push_back(t.GPUEndTime - t.GPUStartTime);
      }
      std::sort(ts.begin(), ts.end());
      return ts[ts.size() / 2] * 1e6;
    };
    auto bindStatsOld = ^(id<MTLComputeCommandEncoder> x) {
      RowStatsParams p{kVocab, 77};
      [x setBuffer:logitsOld offset:0 atIndex:0];
      [x setBuffer:sampledOld offset:0 atIndex:1];
      [x setBuffer:argmaxOld offset:0 atIndex:2];
      [x setBuffer:entropyOld offset:0 atIndex:3];
      [x setBytes:&p length:sizeof(p) atIndex:4];
      [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto bindStatsNew = ^(id<MTLComputeCommandEncoder> x) {
      RowStatsFusedParams p{kVocab, 77, kCap, kInvT};
      [x setBuffer:logitsRaw offset:0 atIndex:0];
      [x setBuffer:sampledNew offset:0 atIndex:1];
      [x setBuffer:argmaxNew offset:0 atIndex:2];
      [x setBuffer:entropyNew offset:0 atIndex:3];
      [x setBytes:&p length:sizeof(p) atIndex:4];
      [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto bindEmbedOld = ^(id<MTLComputeCommandEncoder> x) {
      SoftEmbedParams p{kVocab, kTopK};
      [x setBuffer:logitsOld offset:0 atIndex:0];
      [x setBuffer:weights offset:0 atIndex:1];
      [x setBuffer:scales offset:0 atIndex:2];
      [x setBuffer:biases offset:0 atIndex:3];
      [x setBuffer:outOld offset:0 atIndex:4];
      [x setBytes:&p length:sizeof(p) atIndex:5];
      [x setBytes:&kEmbedScale length:4 atIndex:6];
      [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto bindEmbedNew = ^(id<MTLComputeCommandEncoder> x) {
      SoftEmbedHistParams p{kVocab, kTopK, kCap, kInvT};
      [x setBuffer:logitsRaw offset:0 atIndex:0];
      [x setBuffer:weights offset:0 atIndex:1];
      [x setBuffer:scales offset:0 atIndex:2];
      [x setBuffer:biases offset:0 atIndex:3];
      [x setBuffer:outNew offset:0 atIndex:4];
      [x setBytes:&p length:sizeof(p) atIndex:5];
      [x setBytes:&kEmbedScale length:4 atIndex:6];
      [x dispatchThreadgroups:MTLSizeMake(kRows, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto bindSoftcap = ^(id<MTLComputeCommandEncoder> x) {
      [x setBuffer:logitsOld offset:0 atIndex:0];
      [x setBytes:&kCap length:4 atIndex:1];
      [x setBytes:&count length:4 atIndex:2];
      [x dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto bindScale = ^(id<MTLComputeCommandEncoder> x) {
      [x setBuffer:logitsOld offset:0 atIndex:0];
      [x setBytes:&kInvT length:4 atIndex:1];
      [x setBytes:&count length:4 atIndex:2];
      [x dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    printf("timing (median of %u, us):\n", reps);
    printf("  canvas_row_stats:            %9.1f\n",
           time(@(richengine::ops::kCanvasRowStats.data()), bindStatsOld));
    printf("  canvas_row_stats_fused:      %9.1f\n",
           time(@(richengine::ops::kCanvasRowStatsFused.data()), bindStatsNew));
    printf("  canvas_soft_embed_topk:      %9.1f\n",
           time(@(richengine::ops::kCanvasSoftEmbedTopk.data()), bindEmbedOld));
    printf("  canvas_soft_embed_histogram: %9.1f\n",
           time(@(richengine::ops::kCanvasSoftEmbedHistogram.data()), bindEmbedNew));
    printf("  canvas_logit_softcap:        %9.1f\n",
           time(@(richengine::ops::kCanvasLogitSoftcap.data()), bindSoftcap));
    printf("  canvas_logits_scale:         %9.1f\n",
           time(@(richengine::ops::kCanvasLogitsScale.data()), bindScale));

    // With bf16 logits the unfused path re-quantizes between its passes,
    // so its entropy and draws legitimately differ from the fused path's —
    // they are checked against their own (double-quantized) references.
    const bool ok = argmaxMismatch == 0 && maxEntO < 0.3 &&
                    maxEntN < 1e-3 && embO < 0.05 && embN < 0.05;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
  }
}

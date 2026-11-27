// A/B of the register-dequantized NAXR-style Q4 small-M GEMM candidate
// (q4_naxr_ab_naxr.metal, after locals/nax_kernel.py's locals_qmm_naxr)
// against the production direct-packed matmul2d tiles
// (q4_naxr_ab_base.metal, the q4_mpp_tiles.h instantiations of
// decode/linear_q4.metal) on RichEngine's packed AF4P planes:
//   q4-naxr-ab <base metallib> <naxr metallib> [rounds]
// Synthetic weights: random codes, scales and biases packed in the planes'
// [tile][group][column] order with linear nibbles, validated against a CPU
// reference dequant matmul before any timing. Every (shape, rows) case times
// the candidates in palindromic ABBA order (A B C D D C B A per round), one
// dispatch per command buffer, median GPU ms per candidate.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/abi/Linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTileColumns = 256;

struct Shape final {
  const char *label;
  uint32_t N, K, groups; // groups: the production policy's reference count
};

// The decode shapes of q4_decode_profile.mm's sweep (27B first, then the
// dense 2B-ish widths), all multiples of the 256-column storage tile.
constexpr Shape kShapes[] = {
    {"qkv14336x5120", 14'336, 5'120, 60},
    {"down5120x17408", 5'120, 17'408, 80},
    {"gateup17408x5120", 17'408, 5'120, 36},
    {"mixer5120x6144", 5'120, 6'144, 40},
    {"denseffn6144x2048", 6'144, 2'048, 36},
    {"denseqkv2560x2048", 2'560, 2'048, 12},
};

struct Candidate final {
  const char *name;
  bool naxr;
  uint32_t tileColumns;
  uint32_t threads;
  uint32_t maxRows; // rows the instantiation covers
  bool ceiling = false; // structural ceiling: wrong outputs, not validated
};

constexpr Candidate kCandidates[] = {
    {"base_n128", false, 128, 256, 0},
    {"base_n256", false, 256, 256, 0},
    {"base_n256sg16", false, 256, 512, 64},
    {"naxr_s4", true, 128, 128, 0},
    {"naxr_s8", true, 256, 256, 0},
    {"naxr_ceiling_s8", true, 256, 256, 0, true},
};

std::string kernelName(const Candidate &c, uint32_t rows) {
  if (c.ceiling)
    return "bench_naxr_ceiling_s8_m" + std::to_string(rows);
  std::string name = c.naxr ? "bench_naxr_s" : "bench_base_n";
  if (c.naxr) {
    name += c.tileColumns == 128 ? "4" : "8";
  } else {
    name += std::to_string(c.tileColumns);
  }
  name += "_m" + std::to_string(rows);
  if (!c.naxr && c.threads == 512) name += "_sg16";
  return name;
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// One plane set in the AF4P order the decode kernels index: codes
// [tile][group][column][32 linear-nibble bytes], scales and biases
// [tile][group][column] bf16.
struct Planes final {
  std::vector<uint8_t> codes;
  std::vector<__bf16> scales;
  std::vector<__bf16> biases;
};

Planes makePlanes(uint32_t N, uint32_t K, std::mt19937 &rng) {
  const uint32_t G = K / 64, tiles = N / kTileColumns;
  Planes p;
  p.codes.resize(size_t(tiles) * G * kTileColumns * 32);
  p.scales.resize(size_t(tiles) * G * kTileColumns);
  p.biases.resize(size_t(tiles) * G * kTileColumns);
  std::uniform_int_distribution<int> nibble(0, 15);
  std::uniform_real_distribution<float> scaleMag(0.005f, 0.03f);
  std::uniform_real_distribution<float> biasMag(-0.05f, 0.05f);
  // Draw every column's codes, scales and biases, then place them.
  std::vector<float> scales(N, 0.f), biases(N, 0.f); // per (n, g): index g*N+n
  scales.resize(size_t(N) * G);
  biases.resize(size_t(N) * G);
  std::vector<uint8_t> code(size_t(N) * K);
  for (auto &c : code) c = uint8_t(nibble(rng));
  for (auto &s : scales) s = scaleMag(rng);
  for (auto &b : biases) b = biasMag(rng);
  for (uint32_t n = 0; n < N; ++n) {
    const uint32_t tile = n / kTileColumns, tin = n % kTileColumns;
    for (uint32_t g = 0; g < G; ++g) {
      const size_t planeIndex = (size_t(tile) * G + g) * kTileColumns + tin;
      p.scales[planeIndex] = __bf16(scales[size_t(g) * N + n]);
      p.biases[planeIndex] = __bf16(biases[size_t(g) * N + n]);
      uint8_t *bytes = p.codes.data() + planeIndex * 32;
      for (uint32_t k = 0; k < 64; ++k) {
        const uint8_t c = code[size_t(n) * K + g * 64 + k];
        if (k & 1)
          bytes[k / 2] |= uint8_t(c << 4);
        else
          bytes[k / 2] |= c;
      }
    }
  }
  return p;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: q4-naxr-ab <base metallib> <naxr metallib> [rounds]\n";
      return 2;
    }
    const uint32_t rounds = argc > 3 ? uint32_t(std::stoul(argv[3])) : 20;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    std::printf("device=\"%s\"\n", device.name.UTF8String);
    NSError *error = nil;
    id<MTLLibrary> baseLibrary =
        [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    if (!baseLibrary) {
      std::cerr << "base metallib: " << error.localizedDescription.UTF8String << '\n';
      return 1;
    }
    id<MTLLibrary> naxrLibrary =
        [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[2])] error:&error];
    if (!naxrLibrary) {
      std::cerr << "naxr metallib: " << error.localizedDescription.UTF8String << '\n';
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::mt19937 rng(7);

    constexpr uint32_t kRowCases[] = {16, 32, 64};
    bool validationFailed = false;
    for (const Shape &shape : kShapes) {
      const uint32_t N = shape.N, K = shape.K, G = K / 64;
      Planes planes = makePlanes(N, K, rng);
      id<MTLBuffer> weights =
          [device newBufferWithBytes:planes.codes.data()
                              length:planes.codes.size()
                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> scales =
          [device newBufferWithBytes:planes.scales.data()
                              length:planes.scales.size() * sizeof(__bf16)
                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> biases =
          [device newBufferWithBytes:planes.biases.data()
                              length:planes.biases.size() * sizeof(__bf16)
                             options:MTLResourceStorageModeShared];
      constexpr uint32_t kMaxRows = 64;
      id<MTLBuffer> input = [device newBufferWithLength:uint64_t{kMaxRows} * K * sizeof(__bf16)
                                                options:MTLResourceStorageModeShared];
      {
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        auto *x = static_cast<__bf16 *>(input.contents);
        for (uint64_t i = 0; i < uint64_t{kMaxRows} * K; ++i) x[i] = __bf16(unit(rng));
      }
      // CPU reference for all 64 rows: y[m][n] = sum_k x[m][k] * (s c + b),
      // codes read back out of the packed planes exactly as packed.
      std::vector<double> reference(size_t(kMaxRows) * N, 0.0);
      {
        const auto *x = static_cast<const __bf16 *>(input.contents);
        for (uint32_t n = 0; n < N; ++n) {
          const uint32_t tile = n / kTileColumns, tin = n % kTileColumns;
          for (uint32_t g = 0; g < G; ++g) {
            const size_t planeIndex = (size_t(tile) * G + g) * kTileColumns + tin;
            const float s = float(planes.scales[planeIndex]);
            const float b = float(planes.biases[planeIndex]);
            const uint8_t *bytes = planes.codes.data() + planeIndex * 32;
            for (uint32_t kk = 0; kk < 64; ++kk) {
              const uint32_t k = g * 64 + kk;
              const uint8_t byte = bytes[kk / 2];
              const uint32_t c = (kk & 1) ? (byte >> 4) : (byte & 0xF);
              const float w = s * float(c) + b;
              for (uint32_t m = 0; m < kMaxRows; ++m)
                reference[size_t(m) * N + n] += double(float(x[size_t(m) * K + k])) * w;
            }
          }
        }
      }

      for (const uint32_t rows : kRowCases) {
        // The candidates of this row count: the sg16 wide tile only at 64.
        std::vector<Candidate> active;
        for (const Candidate &c : kCandidates) {
          if (c.maxRows && c.maxRows != rows) continue;
          if (!c.maxRows || rows == 64) active.push_back(c);
        }
        std::vector<std::string> names;
        std::vector<id<MTLComputePipelineState>> pipes;
        bool missing = false;
        for (const Candidate &c : active) {
          const std::string name = kernelName(c, rows);
          names.push_back(name);
          id<MTLFunction> function =
              [(c.naxr ? naxrLibrary : baseLibrary) newFunctionWithName:@(name.c_str())];
          if (!function) {
            std::printf("%s rows%u: missing kernel %s\n", shape.label, rows, name.c_str());
            missing = true;
            break;
          }
          id<MTLComputePipelineState> pipe =
              [device newComputePipelineStateWithFunction:function error:&error];
          if (!pipe) {
            std::printf("%s rows%u: %s\n", shape.label, rows,
                        error.localizedDescription.UTF8String);
            missing = true;
            break;
          }
          pipes.push_back(pipe);
        }
        if (missing) {
          validationFailed = true;
          continue;
        }
        const size_t count = active.size();
        std::vector<id<MTLBuffer>> outputs(count);
        for (size_t v = 0; v < count; ++v)
          outputs[v] = [device newBufferWithLength:uint64_t{rows} * N * sizeof(__bf16)
                                           options:MTLResourceStorageModeShared];

        Q4PersistentParams params{N, K, shape.groups};

        const auto encode = [&](id<MTLComputeCommandEncoder> enc, size_t v) {
          [enc setComputePipelineState:pipes[v]];
          [enc setBuffer:input offset:0 atIndex:0];
          [enc setBuffer:weights offset:0 atIndex:1];
          [enc setBuffer:scales offset:0 atIndex:2];
          [enc setBuffer:biases offset:0 atIndex:3];
          [enc setBuffer:outputs[v] offset:0 atIndex:4];
          [enc setBytes:&params length:sizeof(params) atIndex:5];
          const uint32_t tiles = N / active[v].tileColumns;
          const uint32_t groups = std::min(shape.groups, tiles);
          [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(active[v].threads, 1, 1)];
        };
        const auto submit = [&](size_t v) {
          id<MTLCommandBuffer> command = [queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
          encode(enc, v);
          [enc endEncoding];
          [command commit];
          [command waitUntilCompleted];
          if (command.error) {
            std::printf("%s rows%u %s: %s\n", shape.label, rows, names[v].c_str(),
                        command.error.localizedDescription.UTF8String);
            validationFailed = true;
            return -1.0;
          }
          return (command.GPUEndTime - command.GPUStartTime) * 1e3;
        };

        // Correctness first: one dispatch per candidate against the reference.
        std::printf("%s rows%u check:", shape.label, rows);
        for (size_t v = 0; v < count; ++v) {
          std::memset(outputs[v].contents, 0, uint64_t{rows} * N * sizeof(__bf16));
          if (submit(v) < 0) continue;
          if (active[v].ceiling) {
            std::printf(" %s n/a(ceiling)", names[v].c_str());
            continue;
          }
          const auto *got = static_cast<const __bf16 *>(outputs[v].contents);
          double worst = 0, refMax = 0;
          uint64_t mismatch = 0;
          for (uint64_t i = 0; i < uint64_t{rows} * N; ++i) {
            refMax = std::max(refMax, std::fabs(reference[i]));
            const double d = std::fabs(double(float(got[i])) - reference[i]);
            worst = std::max(worst, d);
            // Counts the elements wrong beyond the output's magnitude scale:
            // bf16 weight rounding admits ~2^-8 of the running sum's scale.
            if (d > 0.01 * refMax) ++mismatch;
          }
          const bool pass = worst <= 0.03 * (1.0 + refMax);
          validationFailed |= !pass;
          std::printf(" %s max=%.3g bad=%llu/%llu %s", names[v].c_str(), worst,
                      (unsigned long long)mismatch, (unsigned long long)(uint64_t{rows} * N),
                      pass ? "PASS" : "FAIL");
        }
        std::printf("\n");

        // Warmup, then palindromic ABBA rounds: A B C ... C B A per round.
        for (size_t v = 0; v < count; ++v) submit(v);
        std::vector<std::vector<double>> samples(count);
        for (uint32_t r = 0; r < rounds; ++r) {
          for (size_t i = 0; i < count; ++i) samples[i].push_back(submit(i));
          for (size_t i = count; i-- > 0;) samples[i].push_back(submit(i));
        }
        std::printf("%s rows%u ms:", shape.label, rows);
        for (size_t v = 0; v < count; ++v) {
          const double t = median(samples[v]);
          std::printf(" %s %.4f", names[v].c_str(), t);
        }
        std::printf("\n");
        std::printf("%s rows%u GB/s(weights+x):", shape.label, rows);
        for (size_t v = 0; v < count; ++v) {
          const double t = median(samples[v]);
          const double wbytes = double(N) * K / 2 + 2.0 * double(N) * K / 64 * sizeof(__bf16);
          const double xbytes = double(rows) * K * sizeof(__bf16) * (double(N) / active[v].tileColumns);
          std::printf(" %s %.0f", names[v].c_str(), (wbytes + xbytes) / (t * 1e-3) / 1e9);
        }
        std::printf("\n");
      }
    }
    std::printf("%s\n", validationFailed ? "VALIDATION FAILED" : "validation pass");
    return validationFailed ? 1 : 0;
  }
}

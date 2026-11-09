// i8_prefill_ab: GEMM-level A/B of the RICHENGINE_PREFILL_FAST_INT8 path
// against the bf16 prefill tile on identical operands.
//
// For each shape it fills a bf16 activation tile with a normal-plus-outlier
// distribution, random affine-Q4 weights, then runs:
//   1. prefill_linear_q4_sums32 + prefill_linear_q4_n256   (bf16 reference)
//   2. prefill_linear_i8_quant + prefill_linear_i8_n256   (int8 candidate)
// and reports per-output-element diffs: bf16-bit mismatches, max/rms delta,
// plus each path's rms error against an fp64 CPU reference on a sampled
// tile. Optional --residual runs the residual epilogue variants instead.
// Timings report median GPU ms per path.
//
// usage: i8-prefill-ab <metallib> [--residual] [--terms N]

#include "../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using richengine::metal::BufferStorage;
using richengine::metal::ComputeDispatch;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;

constexpr uint32_t kQuantGroup = 64;
constexpr uint32_t kTileRows = 32;
constexpr uint32_t kStorageN = 256;

MetalBuffer shared(MetalBackend &backend, uint64_t bytes,
                   const std::string &label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Decode the packed Q4 weight element (input element e of output column j)
// for a CPU reference. Planes: [tile][group][column] tiles; each column's 64
// weights pack two nibbles per byte, low nibble even elements.
uint32_t weightCode(const uint8_t *plane, uint32_t groups, uint32_t j,
                    uint32_t e) {
  const uint32_t tile = j / kStorageN, column = j % kStorageN,
                 group = e / kQuantGroup;
  const uint8_t *wb = plane +
      ((uint64_t(tile) * groups + group) * kStorageN + column) * 32;
  const uint8_t byte = wb[e % kQuantGroup / 2];
  return (e % 2) ? (byte >> 4) : (byte & 0xF);
}

struct Stats {
  uint64_t bitDiffs = 0;
  double maxAbs = 0;
  double rms = 0;
  double refRmsA = 0; // bf16 path vs fp64
  double refRmsB = 0; // i8 path vs fp64
};

Stats compare(MetalBackend &backend, const MetalBuffer &a,
              const MetalBuffer &b, uint32_t rows, uint32_t n,
              const std::vector<double> &fp64ref) {
  const auto *x = static_cast<const __bf16 *>(a.contents());
  const auto *y = static_cast<const __bf16 *>(b.contents());
  Stats s;
  double se = 0, sa = 0, sb = 0;
  uint64_t sampled = 0;
  for (uint64_t i = 0; i < uint64_t(rows) * n; ++i) {
    const float xv = float(x[i]), yv = float(y[i]);
    if (std::memcmp(&x[i], &y[i], sizeof(__bf16))) {
      ++s.bitDiffs;
      s.maxAbs = std::max(s.maxAbs, double(std::fabs(xv - yv)));
    }
    const double d = double(xv) - double(yv);
    se += d * d;
    if (!fp64ref.empty()) {
      sa += (double(xv) - fp64ref[i]) * (double(xv) - fp64ref[i]);
      sb += (double(yv) - fp64ref[i]) * (double(yv) - fp64ref[i]);
      ++sampled;
    }
  }
  s.rms = std::sqrt(se / (uint64_t(rows) * n));
  if (sampled) {
    s.refRmsA = std::sqrt(sa / sampled);
    s.refRmsB = std::sqrt(sb / sampled);
  }
  static_cast<void>(backend);
  return s;
}

void run(MetalBackend &backend, uint32_t rows, uint32_t k, uint32_t n,
         std::mt19937 &random, bool residual, uint32_t terms) {
  const uint32_t rowTiles = rows / kTileRows;
  const uint64_t inputElements = uint64_t(rows) * k;
  const uint64_t outputElements = uint64_t(rows) * n;
  const uint64_t weightElements = uint64_t(k) * n;
  const uint64_t parameterElements = weightElements / kQuantGroup;
  const uint64_t sumElements = uint64_t(rows) * (k / kQuantGroup);

  MetalBuffer input = shared(backend, inputElements * 2, "input");
  MetalBuffer weights = shared(backend, weightElements / 2, "weights");
  MetalBuffer scales = shared(backend, parameterElements * 2, "scales");
  MetalBuffer biases = shared(backend, parameterElements * 2, "biases");
  MetalBuffer resid = shared(backend, outputElements * 2, "residual");
  MetalBuffer sums = shared(backend, sumElements * 4, "sums");
  MetalBuffer outBf = shared(backend, outputElements * 2, "out-bf16");
  MetalBuffer outI8 = shared(backend, outputElements * 2, "out-i8");
  MetalBuffer codesHi = shared(backend, inputElements, "codes-hi");
  MetalBuffer codesLo = shared(backend, inputElements, "codes-lo");
  MetalBuffer codes3 = shared(backend, inputElements, "codes-3");
  MetalBuffer params = shared(backend, sumElements * 16, "i8-params");

  // Realistic activations: unit-normal bulk with a few heavy outlier
  // channels and a row-wise mean offset, then bf16.
  {
    std::normal_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> mag(20.0f, 80.0f);
    std::uniform_real_distribution<float> small(-0.02f, 0.02f);
    auto *ip = static_cast<__bf16 *>(input.contents());
    std::vector<float> outlierScale(k, 1.0f);
    uint32_t outliers = 0;
    for (uint32_t c = 0; c < k; ++c) {
      if (random() % 401 == 0) {
        outlierScale[c] = mag(random);
        ++outliers;
      }
    }
    for (uint64_t i = 0; i < inputElements; ++i)
      ip[i] = __bf16(unit(random) * outlierScale[i % k]);
    auto *wp = static_cast<uint8_t *>(weights.contents());
    for (uint64_t i = 0; i < weightElements / 2; ++i)
      wp[i] = static_cast<uint8_t>(random());
    auto *sp = static_cast<__bf16 *>(scales.contents());
    auto *bp = static_cast<__bf16 *>(biases.contents());
    for (uint64_t i = 0; i < parameterElements; ++i) {
      sp[i] = __bf16(small(random));
      bp[i] = __bf16(small(random));
    }
    auto *rp = static_cast<__bf16 *>(resid.contents());
    for (uint64_t i = 0; i < outputElements; ++i) rp[i] = __bf16(unit(random));
    std::cout << "shape M=" << rows << " K=" << k << " N=" << n
              << " outlier_channels=" << outliers << "\n";
  }

  const Q4Params qp{n, k};
  const auto submit = [&](const char *name,
                          std::vector<std::pair<uint32_t, MetalBuffer>> buffers,
                          const void *bytes, uint32_t byteIndex,
                          uint32_t byteSize,
                          richengine::metal::DispatchSize groups) {
    ComputeDispatch d;
    d.pipelineName = name;
    for (auto &[index, buffer] : buffers)
      d.buffers.push_back({index, std::move(buffer)});
    if (bytes) d.bytes.push_back({byteIndex, bytes, byteSize});
    d.threadgroups = groups;
    d.threadsPerThreadgroup = {256, 1, 1};
    return backend.submit(d);
  };

  // Input group sums feed the bf16 epilogue's bias term.
  submit("prefill_linear_q4_sums32", {{0, input}, {1, sums}}, &k, 2,
         sizeof(k), {rowTiles, 1, 1});

  const std::string bf16Kernel =
      residual ? "prefill_linear_q4_n256_residual" : "prefill_linear_q4_n256";
  const std::string i8Kernel =
      residual ? "prefill_linear_i8_n256_residual" : "prefill_linear_i8_n256";

  auto runBf16 = [&] {
    if (residual)
      return submit(bf16Kernel.c_str(),
                    {{0, input}, {1, weights}, {2, scales}, {3, biases},
                     {4, resid}, {5, outBf}, {6, sums}},
                    &qp, 7, sizeof(qp), {rowTiles, n / 256, 1});
    return submit(bf16Kernel.c_str(),
                  {{0, input}, {1, weights}, {2, scales}, {3, biases},
                   {4, outBf}, {5, sums}},
                  &qp, 6, sizeof(qp), {rowTiles, n / 256, 1});
  };
  auto runQuant = [&] {
    return submit("prefill_linear_i8_quant",
                  {{0, input}, {1, codesHi}, {2, codesLo}, {3, params}},
                  &k, 4, sizeof(k), {rowTiles, 1, 1});
  };
  auto runI8 = [&] {
    // Buffer 10 is the input-sum feed the patched epilogue reads; the
    // unmodified kernel ignores it.
    return submit(i8Kernel.c_str(),
                  {{0, codesHi}, {1, codesLo}, {2, params}, {3, weights},
                   {4, scales}, {5, biases}, {6, resid},
                   {7, outI8}, {8, sums}, {10, sums}},
                  &qp, 9, sizeof(qp), {rowTiles, n / 256, 1});
  };

  runBf16();
  runQuant();
  runI8();
  std::vector<double> a, b, c;
  for (uint32_t r = 0; r < 7; ++r) {
    a.push_back(runBf16().gpuSeconds * 1000);
    c.push_back(runQuant().gpuSeconds * 1000);
    b.push_back(runI8().gpuSeconds * 1000);
  }
  const double bf16ms = median(a), i8ms = median(b), quantms = median(c);

  // fp64 CPU reference on a sample: rows 0..3, all columns of those rows.
  std::vector<double> ref(uint64_t(rows) * n, 0.0);
  {
    const auto *ip = static_cast<const __bf16 *>(input.contents());
    const auto *sp = static_cast<const __bf16 *>(scales.contents());
    const auto *bp = static_cast<const __bf16 *>(biases.contents());
    const auto *wp = static_cast<const uint8_t *>(weights.contents());
    const uint32_t groups = k / kQuantGroup;
    const uint32_t sampleRows = std::min(rows, 4u);
    for (uint32_t r = 0; r < sampleRows; ++r) {
      for (uint32_t j = 0; j < n; ++j) {
        double acc = 0;
        for (uint32_t g = 0; g < groups; ++g) {
          const uint32_t tile = j / kStorageN, column = j % kStorageN;
          const uint64_t parameter =
              (uint64_t(tile) * groups + g) * kStorageN + column;
          const double sw = float(sp[parameter]), bw = float(bp[parameter]);
          double dotc = 0, sumx = 0;
          for (uint32_t e = g * kQuantGroup; e < (g + 1) * kQuantGroup; ++e) {
            const uint32_t code = weightCode(wp, groups, j, e);
            dotc += double(float(ip[uint64_t(r) * k + e])) * code;
            sumx += float(ip[uint64_t(r) * k + e]);
          }
          acc += sw * dotc + bw * sumx;
        }
        if (residual) {
          const auto *rp = static_cast<const __bf16 *>(resid.contents());
          acc += float(rp[uint64_t(r) * n + j]);
        }
        ref[uint64_t(r) * n + j] = acc;
      }
    }
    // Only the sampled region carries an fp64 reference.
    for (uint64_t i = uint64_t(sampleRows) * n; i < ref.size(); ++i)
      ref[i] = std::numeric_limits<double>::quiet_NaN();
  }
  // Isolate the quant kernel: reconstruct x_hat from codes+params and
  // report its error against the bf16 input, plus the GEMM's implied
  // output had it multiplied x_hat exactly (sw * sum(x_hat*c) + bw*sum(x)).
  {
    const auto *ip = static_cast<const __bf16 *>(input.contents());
    const auto *q1 = static_cast<const uint8_t *>(codesHi.contents());
    const auto *q2 = static_cast<const uint8_t *>(codesLo.contents());
    const auto *pa = static_cast<const float *>(params.contents());
    const auto *sp = static_cast<const __bf16 *>(scales.contents());
    const auto *bp = static_cast<const __bf16 *>(biases.contents());
    const auto *wp = static_cast<const uint8_t *>(weights.contents());
    const uint32_t groups = k / kQuantGroup;
    double qe = 0, qmax = 0, ge = 0, gmax = 0;
    const uint32_t checkRows = std::min(rows, 4u);
    for (uint32_t r = 0; r < checkRows; ++r) {
      for (uint32_t g = 0; g < groups; ++g) {
        const float s1 = pa[(r * groups + g) * 4 + 0];
        const float s2 = pa[(r * groups + g) * 4 + 1];
        const float off = pa[(r * groups + g) * 4 + 2];
        for (uint32_t e = 0; e < kQuantGroup; ++e) {
          const uint32_t at = r * k + g * kQuantGroup + e;
          const double xh = double(q1[at]) * s1 + double(q2[at]) * s2 + off;
          const double d = xh - float(ip[at]);
          qe += d * d;
          qmax = std::max(qmax, std::fabs(d));
        }
      }
      // Implied output for a few columns: sw * sum(x_hat*c) + bw*sum(x).
      const auto *xi = static_cast<const __bf16 *>(outI8.contents());
      for (uint32_t j = 0; j < std::min(n, 64u); ++j) {
        float acc = 0;
        for (uint32_t g = 0; g < groups; ++g) {
          const uint32_t tile = j / kStorageN, column = j % kStorageN;
          const uint64_t p =
              (uint64_t(tile) * groups + g) * kStorageN + column;
          const float s1 = pa[(r * groups + g) * 4 + 0];
          const float s2 = pa[(r * groups + g) * 4 + 1];
          const float off = pa[(r * groups + g) * 4 + 2];
          // fp32 simulation of the epilogue: integer dot sums first.
          uint64_t i1 = 0, i2 = 0, cs = 0;
          for (uint32_t e = 0; e < kQuantGroup; ++e) {
            const uint32_t at = r * k + g * kQuantGroup + e;
            const uint32_t code = weightCode(wp, groups, j, g * kQuantGroup + e);
            i1 += uint64_t(q1[at]) * code;
            i2 += uint64_t(q2[at]) * code;
            cs += code;
          }
          acc += float(sp[p]) * (s1 * float(i1) + s2 * float(i2)) +
                 off * float(sp[p]) * float(cs) +
                 float(sums.contents()
                           ? static_cast<const float *>(sums.contents())
                                 [r * groups + g]
                           : 0.0f) * float(bp[p]);
        }
        const double d = acc - float(xi[uint64_t(r) * n + j]);
        ge += d * d;
        gmax = std::max(gmax, std::fabs(d));
      }
    }
    std::cout << "  quant x_hat rms=" << std::sqrt(qe / (checkRows * k))
              << " max=" << qmax << "  implied-y rms="
              << std::sqrt(ge / (checkRows * std::min(n, 64u)))
              << " max=" << gmax << "\n";
  }

  Stats s = compare(backend, outBf, outI8, 4, n, ref);
  // Full-tile diff without the reference.
  const auto *xb = static_cast<const __bf16 *>(outBf.contents());
  const auto *xi = static_cast<const __bf16 *>(outI8.contents());
  Stats full;
  double se = 0;
  for (uint64_t i = 0; i < outputElements; ++i) {
    if (std::memcmp(&xb[i], &xi[i], sizeof(__bf16))) {
      ++full.bitDiffs;
      full.maxAbs = std::max(full.maxAbs,
                             double(std::fabs(float(xb[i]) - float(xi[i]))));
    }
    const double d = double(float(xb[i])) - double(float(xi[i]));
    se += d * d;
  }
  full.rms = std::sqrt(se / outputElements);
  std::cout << "  bf16_ms=" << bf16ms << " i8_ms=" << i8ms
            << " quant_ms=" << quantms << " terms=" << terms << "\n"
            << "  diff bits=" << full.bitDiffs << "/" << outputElements
            << " max_abs=" << full.maxAbs << " rms=" << full.rms << "\n"
            << "  vs fp64 (4 rows): bf16_rms=" << s.refRmsA
            << " i8_rms=" << s.refRmsB << "\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2) {
      std::cerr << "usage: i8-prefill-ab <metallib> [--residual] [--terms N]\n";
      return 2;
    }
    bool residual = false;
    uint32_t terms = 2;
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--residual") residual = true;
      else if (arg == "--terms" && i + 1 < argc) terms = std::stoul(argv[++i]);
    }
    try {
      MetalBackend backend(argv[1]);
      std::mt19937 random(0x1ab5eed);
      run(backend, 512, 4096, 12288, random, residual, terms);
      run(backend, 512, 12288, 4096, random, residual, terms);
      run(backend, 128, 4096, 4096, random, residual, terms);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}

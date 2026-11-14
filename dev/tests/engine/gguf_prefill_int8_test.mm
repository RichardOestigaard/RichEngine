// gguf_prefill_int8_test — RICHENGINE_PREFILL_FAST_INT8's GGUF half
// (kernels/prefill/gguf_prefill_int8.metal, ops/LinearGguf.cpp's
// addGgufPrefill): the AF formats' prefill tiles quantize each chunk row's
// 64-input groups to int8 codes and run int8 matmul2d fragments. The env
// flag must be set before the first modelPolicy() read, so this binary sets
// it at start and every GGUF prefill plan of more than 64 rows takes the
// int8 tiles. Each run's outputs are compared with an fp64 simulation of the
// same math — the activation quantization rule of
// kernels/common/int8_activations.h and each weight's affine decode — not
// with the bf16 tile's, which the int8 path deliberately does not match.
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace richengine;
using namespace richengine::ops;
using namespace gguf_reference;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;
using richengine::ops::tuning::bf16ToFloat;
using richengine::ops::tuning::floatToBf16;

namespace {

std::mt19937 rng(7);
int failures = 0;

void fail(const std::string &what) {
  std::cout << "  FAIL " << what << '\n';
  ++failures;
}
void section(const std::string &what) {
  std::cout << what << (failures ? " FAIL (" + std::to_string(failures) + " failures)" : " ok") << '\n';
}

// ---------------------------------------------------------------- weights
struct Tensor {
  Fmt format;
  uint32_t N, K;
  std::vector<uint8_t> native;
  QuantizedSegment segment;
};

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = test::sharedBuffer(backend, bytes.size());
  if (!bytes.empty()) std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}
MetalBuffer uploadRows(MetalBackend &backend, const std::vector<uint16_t> &rows) {
  MetalBuffer buffer = test::sharedBuffer(backend, rows.size() * 2);
  if (!rows.empty()) std::memcpy(buffer.contents(), rows.data(), rows.size() * 2);
  return buffer;
}

Tensor tensor(MetalBackend &backend, Fmt f, uint32_t N, uint32_t K) {
  Tensor t{f, N, K, makeNative(f, N, K, rng), {}};
  const Packed planes = repack(f, t.native, N, K, nullptr);
  t.segment = QuantizedSegment::planes(f, N, K, upload(backend, planes.w0),
                                       kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1) : MetalBuffer{},
                                       upload(backend, planes.meta));
  return t;
}
Projection projection(const Tensor &w) {
  BlockWeights blocks;
  blocks.segments.push_back(w.segment);
  return Projection(w.N, w.K, std::move(blocks));
}

// ---------------------------------------------------------------- fp64 reference
// The weight row n's code and affine pair of 64-input group g, decoded from
// the native block the planes were repacked from (the same bytes the kernel's
// planes carry, decoded independently): s * code + z per element.
struct WeightGroup {
  double s = 0, z = 0;
  int code[64];
};
WeightGroup weightGroup(const Tensor &w, uint32_t n, uint32_t g) {
  const QuantFormat &fi = kQuantFormats[w.format];
  const uint32_t be = fi.block_elements, bb = fi.block_bytes;
  const uint8_t *row = w.native.data() + size_t{n} * rowBytes(w.format, w.K);
  const uint32_t unit = 2 * g / fi.meta_groups;   // the meta unit's 64-input groups
  const uint8_t *blk = row + size_t{unit} * bb;
  WeightGroup out{};
  uint16_t s16, m16;
  std::memcpy(&s16, blk + bb - 4, 2);
  std::memcpy(&m16, blk + bb - 2, 2);
  out.s = b2f(s16);
  out.z = b2f(m16);
  const uint32_t base = (64 * g) % be;   // element of the block the group starts at
  if (w.format == AF4) {
    // {qs[32], bf16 scale, bf16 bias}: element e's code is nibble e & 1 of
    // byte 16 (e / 32) + (e % 32) / 2 — groupPack's AF4 arm.
    for (uint32_t i = 0; i < 64; ++i) {
      const uint32_t e = base + i, k = e % 32;
      out.code[i] = (blk[16 * (e / 32) + k / 2] >> (4 * (k & 1))) & 15;
    }
    return out;
  }
  const uint32_t bits = afBits(w.format);
  for (uint32_t i = 0; i < 64; ++i) {
    const uint32_t o = (base + i) * bits;
    out.code[i] = int((uint32_t(blk[o / 8]) | (uint32_t(blk[o / 8 + 1]) << 8)) >> (o % 8)) &
                  ((1u << bits) - 1);
  }
  return out;
}

// One chunk row's 64-input groups quantized as gguf_prefill_int8_quantize
// does: scale m / 127 of the group's largest magnitude, codes
// clamp(rint(x * 127 / m), -127, 127), ties to even. The kernel's quotients
// are fp32 — precise::divide for 127 / m and an fp32 multiply before rint —
// so the reference computes them in float too, or a product landing on a
// rounding boundary codes one off from the kernel's.
struct ActGroup {
  double scale = 0, sum = 0;
  int code[64];
};
ActGroup actGroup(const float *x) {
  ActGroup out{};
  float largest = 0;
  for (uint32_t i = 0; i < 64; ++i) largest = std::max(largest, std::fabs(x[i]));
  if (largest > 0x1p-100f) {
    const float scale = largest / 127.0f;      // precise::divide's correctly rounded quotient
    const float inverse = 127.0f / largest;
    out.scale = scale;
    for (uint32_t i = 0; i < 64; ++i) {
      const float q = std::nearbyint(x[i] * inverse);   // fp32 product, rint: ties to even
      out.code[i] = int(std::clamp(q, -127.0f, 127.0f));
      out.sum += out.code[i];
    }
  }
  return out;
}

std::vector<float> activations(uint32_t rows, uint32_t width) {
  std::uniform_real_distribution<float> unit(-1.f, 1.f);
  std::vector<float> x(uint64_t{rows} * width);
  for (float &v : x) v = bf16ToFloat(floatToBf16(unit(rng)));
  return x;
}
std::vector<uint16_t> bf16Rows(const std::vector<float> &x, uint32_t width, uint32_t active,
                               uint32_t storage) {
  std::vector<uint16_t> rows(uint64_t{storage} * width, 0x7FC0);   // bf16 NaN past the rows
  for (uint64_t i = 0; i < uint64_t{active} * width; ++i) rows[i] = floatToBf16(x[i]);
  return rows;
}

// ---------------------------------------------------------------- running a plan
struct Run {
  std::vector<uint16_t> output;
  std::vector<std::string> kernels;
};

Run run(MetalBackend &backend, const Linear &linear, const LinearPlan &plan, const Projection &p,
        const std::vector<uint16_t> &x, const std::vector<uint16_t> &aux) {
  const LinearWorkload w = plan.workload();
  const uint32_t rows = w.rows, storage = plan.storageRows();
  const auto [n, k] = w.matrix;
  MetalBuffer input = uploadRows(backend, x), auxBuffer = uploadRows(backend, aux);
  MetalBuffer output = test::sharedBuffer(backend, uint64_t{storage} * n * 2);
  std::memset(output.contents(), 0xFF, uint64_t{storage} * n * 2);
  // The int8 operand buffers the arena binds in production: the chunk's
  // rows' codes and per-(group, row) float2 (scale, scale * code sum).
  MetalBuffer i8codes = test::sharedBuffer(backend, uint64_t{rows} * k);
  MetalBuffer i8params =
      test::sharedBuffer(backend, uint64_t{k / GGUF_PREFILL_INT8_GROUP} * storage * 8);
  LinearScratch scratch{.i8codes = i8codes, .i8params = i8params};
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool gated = w.epilogue == LinearEpilogue::UpWithGate;
  LinearBuffers buffers{.input = input, .output = output,
                        .residual = residual ? auxBuffer : MetalBuffer{},
                        .gateScratch = gated ? auxBuffer : MetalBuffer{},
                        .scratch = scratch};
  CommandGraph graph;
  static_cast<void>(linear.add(graph, buffers, p, plan));
  std::vector<std::string> names;
  for (const auto &dispatch : graph.dispatches()) names.push_back(dispatch.pipelineName);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const auto *out = static_cast<const uint16_t *>(output.contents());
  return {std::vector<uint16_t>(out, out + uint64_t{storage} * n), names};
}

// The fp64 value the int8 tile's epilogue leaves, before bf16 rounding.
double silu(double g) { return g / (1 + std::exp(-g)); }

// ---------------------------------------------------------------- checks
void check(MetalBackend &backend, const Linear &linear, const Tensor &w, uint32_t rows,
           LinearEpilogue epilogue) {
  const QuantFormat &fi = kQuantFormats[w.format];
  const uint32_t n = w.N, k = w.K, groups = k / GGUF_PREFILL_INT8_GROUP;
  const Projection p = projection(w);
  const LinearPlan plan = linear.prefillPlan(p, rows, epilogue);
  if (plan.configuration().tile != LinearTile::GgufPrefill) {
    fail(std::string(fmtName(w.format)) + " " + std::to_string(rows) +
         " rows: not the GGUF prefill tile");
    return;
  }
  const uint32_t storage = plan.storageRows();
  const std::vector<float> x = activations(rows, k);
  const std::vector<float> auxf = activations(rows, n);
  const std::vector<uint16_t> xi = bf16Rows(x, k, rows, storage);
  const std::vector<uint16_t> ai = bf16Rows(auxf, n, rows, storage);
  const std::string what = std::string(fmtName(w.format)) + " " + std::to_string(n) + "x" +
                           std::to_string(k) + " " + std::to_string(rows) + " rows " +
                           (epilogue == LinearEpilogue::None    ? "plain"
                            : epilogue == LinearEpilogue::Residual ? "residual"
                                                                   : "up-with-gate");
  const Run out = run(backend, linear, plan, p, xi, ai);
  // The int8 quantize and tile must be what ran: every AF segment's kernels.
  const std::string stem = std::string("gguf_prefill_int8_") + fi.name + "_" +
                           (epilogue == LinearEpilogue::None    ? "a"
                            : epilogue == LinearEpilogue::Residual ? "r"
                                                                   : "g");
  const bool quantized = std::any_of(out.kernels.begin(), out.kernels.end(),
                                     [](const std::string &s) { return s == "gguf_prefill_int8_quantize"; });
  const bool tiled = std::any_of(out.kernels.begin(), out.kernels.end(),
                                 [&](const std::string &s) { return s == stem; });
  if (!quantized || !tiled)
    fail(what + ": the int8 kernels did not run (quantize " + std::to_string(quantized) + ", " + stem + " " +
         std::to_string(tiled) + ")");
  // The fp64 simulation: quantize each row's groups once, then per column
  // the groups' sa * (s_w qa.qw + z_w sum qa).
  std::vector<ActGroup> acts(uint64_t{rows} * groups);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t g = 0; g < groups; ++g) acts[uint64_t{r} * groups + g] = actGroup(&x[uint64_t{r} * k + 64 * g]);
  std::vector<WeightGroup> wgs(uint64_t{n} * groups);
  for (uint32_t c = 0; c < n; ++c)
    for (uint32_t g = 0; g < groups; ++g) wgs[uint64_t{c} * groups + g] = weightGroup(w, c, g);
  uint64_t wrong = 0;
  double worst = 0;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < n; ++c) {
      double y = 0;
      for (uint32_t g = 0; g < groups; ++g) {
        const ActGroup &a = acts[uint64_t{r} * groups + g];
        const WeightGroup &wg = wgs[uint64_t{c} * groups + g];
        double dot = 0;
        for (uint32_t i = 0; i < 64; ++i) dot += double(a.code[i]) * wg.code[i];
        y += a.scale * (wg.s * dot + wg.z * a.sum);
      }
      // The epilogue, with the kernel's roundings: the plain value is fp32
      // accumulated then bf16-rounded; residual adds the bf16 aux in fp32;
      // up-with-gate rounds to bf16, multiplies silu(aux), rounds again.
      // floatToBf16 returns the bf16 bits: bf16ToFloat recovers the value.
      float expected;
      if (epilogue == LinearEpilogue::Residual)
        expected = bf16ToFloat(
            floatToBf16(float(float(y) + bf16ToFloat(ai[uint64_t{r} * n + c]))));
      else if (epilogue == LinearEpilogue::UpWithGate) {
        const float v = bf16ToFloat(floatToBf16(float(y)));
        expected = bf16ToFloat(
            floatToBf16(v * float(silu(double(bf16ToFloat(ai[uint64_t{r} * n + c]))))));
      } else
        expected = bf16ToFloat(floatToBf16(float(y)));
      const float got = bf16ToFloat(out.output[uint64_t{r} * n + c]);
      // bf16's quantum at the value plus the fp32 accumulation's slack: a
      // wrong slot order moves whole product terms, far past this.
      const double ulp = std::ldexp(1.0, std::max(-20, int(std::floor(std::log2(std::fabs(double(expected))))) - 7));
      const double tol = 4 * ulp + 1e-3 * std::fabs(double(expected)) + 1e-3;
      const double d = std::fabs(double(got) - double(expected));
      if (!(d <= tol) || !std::isfinite(got)) {
        if (wrong < 4)
          std::cout << "    " << what << " r" << r << " c" << c << ": got " << got << " want ~" << expected
                    << " (|d| " << d << ", tol " << tol << ")\n";
        ++wrong;
      }
      worst = std::max(worst, d / (tol + 1e-30));
    }
  if (wrong) fail(what + ": " + std::to_string(wrong) + " outputs off (worst " + std::to_string(worst) + ")");
}

} // namespace

int main(int argc, char **argv) {
  // Before anything reads modelPolicy(): this binary exists to run the
  // opt-in path.
  setenv("RICHENGINE_PREFILL_FAST_INT8", "1", 1);
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf-prefill-int8 <production-and-test.metallib>\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      const Linear linear(backend.capabilities());
      int before = failures;
      for (const Fmt f :
           {AF4, AF5, AF6, AF8, AF4G128, AF5G128, AF6G128, AF8G128, AF2, AF3, AF2G128, AF3G128}) {
        // 320 columns (five 64-column tiles, an odd tile count for the
        // kernel's tile walk), 512 inputs; and 256 x 384 at 80 rows for a
        // second meta-unit count and a partial row tile.
        for (const auto &[n, k, rows] : {std::tuple<uint32_t, uint32_t, uint32_t>{320, 512, 144},
                                         {256, 384, 80}}) {
          Tensor w = tensor(backend, f, n, k);
          for (const LinearEpilogue e :
               {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
            check(backend, linear, w, rows, e);
        }
      }
      section("int8 prefill: all twelve AF formats, plain/residual/up-with-gate, two shapes each, "
              "fp64-simulated outputs");
      std::cout << "gguf-prefill-int8: " << (failures - before ? "FAIL" : "PASS") << '\n';
      return failures - before ? 1 : 0;
    } catch (const std::exception &e) {
      std::cerr << "gguf-prefill-int8: FAIL: " << e.what() << '\n';
      return 1;
    }
  }
}

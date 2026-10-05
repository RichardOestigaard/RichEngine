// Chunked-parallel GDN scan A/B: parity vs the serial prefill_gdn_scan and
// GPU time per pass at 2048 tokens, for both compiled head layouts and each
// SPLASH_GDN_CHUNKED factor (32/64/128). Usage:
//   gdn_chunked_bench <metallib> [tokens] [reps]
// Parity gate: max |out - ref| on bf16 recurrent rows and the fp32 state
// (bf16-level tolerance — the chunked scan reassociates the recurrence).

#include "../../runtime/metal/CommandGraph.hpp"
#include "../../runtime/metal/MetalBackend.hpp"
#include "../../runtime/ops/GDN.hpp"
#include "metal/abi/GDN.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::GDN;
using splash::ops::GdnHeadOrder;
using splash::ops::GdnPrefillBuffers;
using splash::ops::GdnShape;
using splash::ops::NormWeights;

constexpr uint32_t kHeadDim = 128;


class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }
  float unit() { return float(next() & 0xFFFFFF) / 8388608.0F - 1.0F; }
  float gauss() {
    float sum = 0;
    for (int i = 0; i < 4; ++i) sum += unit();
    return sum * 0.8660254F;
  }

private:
  uint64_t state_;
};

uint16_t toBf16(float v) {
  const uint32_t bits = std::bit_cast<uint32_t>(v);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
float fromBf16(uint16_t v) {
  return std::bit_cast<float>(uint32_t{v} << 16);
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer b = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(b.contents(), 0, b.sizeBytes());
  return b;
}
template <class T> T *data(const MetalBuffer &b) {
  return static_cast<T *>(b.contents());
}

struct Inputs {
  MetalBuffer packed, convIn, recurrentIn, convWeights, decayWeights, timeBias;
  NormWeights mixerNorm;
};

GdnPrefillBuffers outputs(MetalBackend &backend, const GdnShape &shape,
                          uint32_t tokens, const Inputs &in,
                          MetalBuffer scratch) {
  const uint64_t convDim = shape.convolutionDimension;
  const uint64_t keyW = uint64_t{shape.keyHeads} * kHeadDim;
  const uint64_t valW = uint64_t{shape.valueHeads} * kHeadDim;
  const uint64_t stateBytes = uint64_t{shape.valueHeads} * kHeadDim * kHeadDim * 4;
  return {in.packed,
          in.convWeights,
          in.convIn,
          shared(backend, 3 * convDim * 2, "conv out"),
          shared(backend, uint64_t{tokens} * keyW * 2, "queries"),
          shared(backend, uint64_t{tokens} * keyW * 2, "keys"),
          shared(backend, uint64_t{tokens} * valW * 2, "values"),
          in.decayWeights,
          in.timeBias,
          shared(backend, uint64_t{tokens} * shape.valueHeads * 4, "decay"),
          shared(backend, uint64_t{tokens} * shape.valueHeads * 2, "beta"),
          in.recurrentIn,
          shared(backend, stateBytes, "state out"),
          shared(backend, uint64_t{tokens} * valW * 2, "rows"),
          in.mixerNorm,
          shared(backend, uint64_t{tokens} * valW * 2, "hidden"),
          std::move(scratch)};
}

Inputs makeInputs(MetalBackend &backend, const GdnShape &shape,
                  uint32_t tokens, uint64_t seed) {
  const uint64_t convDim = shape.convolutionDimension;
  const uint64_t packedW = shape.packedWidth;
  const uint64_t stateEl = uint64_t{shape.valueHeads} * kHeadDim * kHeadDim;
  Inputs in{shared(backend, uint64_t{tokens} * packedW * 2, "packed"),
            shared(backend, 3 * convDim * 2, "conv in"),
            shared(backend, stateEl * 4, "state in"),
            shared(backend, convDim * 4 * 2, "conv weights"),
            shared(backend, shape.valueHeads * 4, "a scale"),
            shared(backend, shape.valueHeads * 2, "dt bias"), {}};
  Random random(seed ^ (uint64_t{shape.valueHeads} << 32) ^ tokens);
  for (uint64_t i = 0; i < uint64_t{tokens} * packedW; ++i)
    data<uint16_t>(in.packed)[i] = toBf16(random.gauss());
  for (uint64_t i = 0; i < convDim * 4; ++i)
    data<uint16_t>(in.convWeights)[i] = toBf16(0.3F * random.gauss());
  for (uint64_t i = 0; i < 3 * convDim; ++i)
    data<uint16_t>(in.convIn)[i] = toBf16(random.gauss());
  auto *aScale = data<float>(in.decayWeights);
  auto *dtBias = data<uint16_t>(in.timeBias);
  for (uint32_t h = 0; h < shape.valueHeads; ++h) {
    const float unit = 0.5F * (random.unit() + 1.0F);
    aScale[h] = h % 11 == 7 ? -105.0F : -(0.05F + 8.0F * unit * unit);
    dtBias[h] = toBf16(random.gauss());
  }
  for (uint64_t i = 0; i < stateEl; ++i)
    data<float>(in.recurrentIn)[i] = 0.05F * random.gauss();
  MetalBuffer norm = shared(backend, kHeadDim * 2, "mixer norm");
  for (uint32_t i = 0; i < kHeadDim; ++i)
    data<uint16_t>(norm)[i] = toBf16(1.0F + 0.2F * random.gauss());
  in.mixerNorm = {std::move(norm), false, 1e-6F};
  return in;
}

double run(MetalBackend &backend, GdnPrefillBuffers &buffers,
           const GdnShape &shape, uint32_t tokens, uint32_t reps) {
  double best = 1e9;
  for (uint32_t i = 0; i < reps; ++i) {
    CommandGraph graph;
    GDN::addPrefill(graph, buffers, shape, tokens, GdnHeadOrder::Grouped);
    const auto timing = backend.submitCommand(graph.dispatches());
    best = std::min(best, timing.gpuSeconds);
  }
  return best;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2) {
      std::cerr << "usage: gdn_chunked_bench <metallib> [tokens] [reps]\n";
      return 2;
    }
    const uint32_t tokens = argc > 2 ? std::atoi(argv[2]) : 2048;
    const uint32_t reps = argc > 3 ? std::atoi(argv[3]) : 5;
    MetalBackend backend(argv[1]);
    for (const GdnShape &shape :
         {GdnShape{16, 48, 128, 10240, 16640},
          GdnShape{16, 32, 128, 8192, 12544}}) {
      Inputs in = makeInputs(backend, shape, tokens, 0x9E3779B97F4A7C15ULL);
      const uint64_t valW = uint64_t{shape.valueHeads} * kHeadDim;
      const uint64_t stateEl = uint64_t{shape.valueHeads} * kHeadDim * kHeadDim;

      GdnPrefillBuffers serial = outputs(backend, shape, tokens, in, {});
      const double serialMs = 1e3 * run(backend, serial, shape, tokens, reps);
      std::vector<uint16_t> refRows(tokens * valW);
      std::vector<float> refState(stateEl);
      std::memcpy(refRows.data(), serial.recurrentRows.contents(),
                  refRows.size() * 2);
      std::memcpy(refState.data(), serial.recurrentOut.contents(),
                  refState.size() * 4);
      printf("vh%-3u tokens=%-5u serial      %7.2f ms\n", shape.valueHeads,
             tokens, serialMs);

      for (uint32_t factor : {32u, 64u, 128u}) {
        const uint64_t floats =
            GDN::chunkScratchFloats(shape, tokens, factor);
        MetalBuffer scratch =
            shared(backend, floats * 4, "chunk scratch");
        GdnPrefillBuffers chunked =
            outputs(backend, shape, tokens, in, scratch);
        setenv("SPLASH_GDN_CHUNKED", std::to_string(factor).c_str(), 1);
        const double ms = 1e3 * run(backend, chunked, shape, tokens, reps);
        // Parity: bf16 rows and fp32 state against the serial pass.
        double rowErr = 0, stateErr = 0;
        auto *gotRows = data<uint16_t>(chunked.recurrentRows);
        auto *gotState = data<float>(chunked.recurrentOut);
        for (uint64_t i = 0; i < uint64_t{tokens} * valW; ++i)
          rowErr = std::max(
              rowErr, std::fabs(double(fromBf16(gotRows[i])) -
                                double(fromBf16(refRows[i]))));
        for (uint64_t i = 0; i < stateEl; ++i)
          stateErr = std::max(stateErr,
                              std::fabs(double(gotState[i] - refState[i])));
        printf("vh%-3u tokens=%-5u chunked C=%-3u %7.2f ms  speedup %4.2fx  "
               "rowErr %.4f stateErr %.5f\n",
               shape.valueHeads, tokens, factor, ms, serialMs / ms, rowErr,
               stateErr);
      }
      unsetenv("SPLASH_GDN_CHUNKED");
    }
  }
  return 0;
}

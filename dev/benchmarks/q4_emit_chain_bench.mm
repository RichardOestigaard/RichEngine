// Times the dense-FFN decode chain both ways at a production shape:
// prepare + gate_up + prepare + residual vs gate_up_t64 + residual_psum.
// Usage: q4-emit-chain <metallib> [rows]
#include "../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using richengine::metal::BufferStorage;
using richengine::metal::ComputeDispatch;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;

namespace {
struct Aff {
  MetalBuffer w, s, b;
};
Aff weights(MetalBackend &backend, uint64_t n, uint64_t k, const char *label) {
  const uint64_t params = n * k / 64;
  Aff a{backend.allocateBuffer(params * 32, BufferStorage::Shared, label),
        backend.allocateBuffer(params * 2, BufferStorage::Shared, label),
        backend.allocateBuffer(params * 2, BufferStorage::Shared, label)};
  std::memset(a.w.contents(), 0x5a, params * 32);
  std::memset(a.s.contents(), 0x3c, params * 2);
  std::memset(a.b.contents(), 0x3c, params * 2);
  return a;
}
double run(MetalBackend &backend, const std::vector<ComputeDispatch> &chain, uint32_t repeats) {
  std::vector<double> samples;
  for (uint32_t i = 0; i < repeats; ++i) {
    double seconds = 0;
    for (const auto &d : chain) seconds += backend.submit(d).gpuSeconds;
    samples.push_back(seconds);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) { std::cerr << "usage: q4-emit-chain <metallib> [rows]\n"; return 1; }
    MetalBackend backend(argv[1]);
    const uint32_t rows = argc > 2 ? uint32_t(std::atoi(argv[2])) : 32;
    const uint64_t hidden = 5120, inter = 17408, out = 5120; // Qwen3.8-27B FFN
    const uint32_t lanes = rows / 8;

    const Aff gate = weights(backend, inter, hidden, "gate");
    const Aff up = weights(backend, inter, hidden, "up");
    const Aff down = weights(backend, out, inter, "down");
    MetalBuffer input = backend.allocateBuffer(uint64_t{rows} * hidden * 2, BufferStorage::Shared, "input");
    MetalBuffer table = backend.allocateBuffer(uint64_t{rows} * hidden * 2, BufferStorage::Private, "table");
    MetalBuffer tsums = backend.allocateBuffer(uint64_t{rows} * hidden / 16, BufferStorage::Private, "sums");
    MetalBuffer mid = backend.allocateBuffer(uint64_t{rows} * inter * 2, BufferStorage::Private, "mid");
    MetalBuffer midT = backend.allocateBuffer(uint64_t{rows} * inter * 2, BufferStorage::Private, "midT");
    MetalBuffer dsums = backend.allocateBuffer(uint64_t{rows} * inter / 16, BufferStorage::Private, "dsums");
    MetalBuffer psums = backend.allocateBuffer(uint64_t{rows} * inter / 8, BufferStorage::Private, "psums");
    MetalBuffer mid2 = backend.allocateBuffer(uint64_t{rows} * inter * 2, BufferStorage::Private, "mid2");
    MetalBuffer output = backend.allocateBuffer(uint64_t{rows} * out * 2, BufferStorage::Shared, "out");
    MetalBuffer residual = backend.allocateBuffer(uint64_t{rows} * out * 2, BufferStorage::Shared, "res");
    MetalBuffer partials = backend.allocateBuffer(64 * 1024 * 1024, BufferStorage::Private, "partials");
    MetalBuffer counters = backend.allocateBuffer(1 * 1024 * 1024, BufferStorage::Shared, "counters");
    std::memset(counters.contents(), 0, 1 * 1024 * 1024);

    const uint32_t gu = 0;
    Q4Params gp{uint32_t(inter), uint32_t(hidden)}, dp{uint32_t(out), uint32_t(inter)};
    uint32_t k32 = uint32_t(hidden), d32 = uint32_t(inter);

    // Unfused: prepare -> gate_up -> prepare -> residual.
    ComputeDispatch prepIn;
    prepIn.pipelineName = "decode_linear_q4_prepare";
    prepIn.buffers = {{0, input}, {1, table}, {2, tsums}};
    prepIn.bytes = {{3, &k32, sizeof(k32)}};
    prepIn.threadgroups = {hidden / 32, lanes, 1};
    prepIn.threadsPerThreadgroup = {128, 1, 1};

    ComputeDispatch gateUp;
    gateUp.pipelineName = "decode_linear_q4_sg_gate_up";
    gateUp.buffers = {{0, table}, {1, gate.w}, {2, gate.s}, {3, gate.b},
                      {4, mid},   {5, tsums},  {6, partials}, {7, counters},
                      {8, up.w},  {9, up.s},   {10, up.b}};
    gateUp.bytes = {{11, &gp, sizeof(gp)}};
    gateUp.threadgroups = {inter / 32, 1, lanes};
    gateUp.threadsPerThreadgroup = {128, 1, 1};

    ComputeDispatch prepDown;
    prepDown.pipelineName = "decode_linear_q4_prepare";
    prepDown.buffers = {{0, mid}, {1, mid2}, {2, dsums}};
    prepDown.bytes = {{3, &d32, sizeof(d32)}};
    prepDown.threadgroups = {inter / 32, lanes, 1};
    prepDown.threadsPerThreadgroup = {128, 1, 1};

    ComputeDispatch res;
    res.pipelineName = "decode_linear_q4_sg_residual";
    res.buffers = {{0, mid2},     {1, down.w}, {2, down.s}, {3, down.b}, {4, output},
                   {5, dsums},    {6, partials}, {7, counters}, {8, residual}};
    res.bytes = {{9, &dp, sizeof(dp)}};
    res.threadgroups = {out / 64, 1, lanes};
    res.threadsPerThreadgroup = {128, 1, 1};

    // Fused: prepare -> gate_up_t64 -> residual_psum.
    ComputeDispatch gateUpT = gateUp;
    gateUpT.pipelineName = "decode_linear_q4_sg_gate_up_t64";
    gateUpT.buffers = {{0, table}, {1, gate.w}, {2, gate.s}, {3, gate.b},
                       {4, midT},  {5, tsums},  {6, partials}, {7, counters},
                       {8, up.w},  {9, up.s},   {10, up.b},    {11, psums}};
    gateUpT.bytes = {{12, &gp, sizeof(gp)}};

    ComputeDispatch resP = res;
    resP.pipelineName = "decode_linear_q4_sg_residual_psum";
    resP.buffers = {{0, midT},     {1, down.w}, {2, down.s}, {3, down.b}, {4, output},
                    {5, psums},    {6, partials}, {7, counters}, {8, residual}};
    resP.bytes = {{9, &dp, sizeof(dp)}};

    (void)gu;
    const uint32_t repeats = 50;
    const std::vector<ComputeDispatch> plain{prepIn, gateUp, prepDown, res};
    const std::vector<ComputeDispatch> fused{prepIn, gateUpT, resP};
    std::vector<double> pSamples, fSamples;
    for (uint32_t i = 0; i < repeats; ++i) {
      double s = 0;
      for (const auto &d : plain) s += backend.submit(d).gpuSeconds;
      pSamples.push_back(s);
      s = 0;
      for (const auto &d : fused) s += backend.submit(d).gpuSeconds;
      fSamples.push_back(s);
    }
    std::sort(pSamples.begin(), pSamples.end());
    std::sort(fSamples.begin(), fSamples.end());
    std::cout << "rows=" << rows << " plain=" << pSamples[pSamples.size() / 2] * 1e3
              << " ms fused=" << fSamples[fSamples.size() / 2] * 1e3 << " ms\n";
    // Per-kernel isolation.
    for (const auto &[label, chain] :
         {std::pair{"prepDown", std::vector<ComputeDispatch>{prepDown}},
          {"gateUp", std::vector<ComputeDispatch>{gateUp}},
          {"gateUpT64", std::vector<ComputeDispatch>{gateUpT}},
          {"residual", std::vector<ComputeDispatch>{res}},
          {"residualPsum", std::vector<ComputeDispatch>{resP}}}) {
      std::vector<double> s;
      for (uint32_t i = 0; i < repeats; ++i) s.push_back(backend.submit(chain[0]).gpuSeconds);
      std::sort(s.begin(), s.end());
      std::cout << "  " << label << " " << s[s.size() / 2] * 1e3 << " ms\n";
    }
  }
}

// A/B for an ANE MoE expert split (docs/ANE_EXPERIMENTS.md): one 35B-family
// routed expert (hidden 2048, intermediate 512, affine Q4) run the GPU's own
// prefill SwiGLU, then packed and staged for a whole-expert W8A8 evaluation
// on the Neural Engine through the same machinery the dense FFN split uses
// (ane_ffn_rotate/pack/weights/join, ane::Program, ane::Handoff). Per row
// bucket the bench reports the GPU-only ms, the ANE eval ms, the GPU-side
// overhead the ANE path adds (rotate+pack, join, weight staging), and the
// int8 error against the GPU's output. A chained pass then measures the
// eval-rate ceiling and projects a 2048-row MoE layer under correlated
// routing (the gguf-moe-benchmark MOE_BENCH_SKEW=1 mix): which experts the
// ANE can take while its chain stays under the GPU's remaining work.
//   ane-moe-benchmark <metallib> [rounds]
#include "../tests/engine/AffineQ4Fixture.hpp"
#include "../tests/engine/TestBuffers.hpp"
#include "AwakeClock.hpp"
#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/AneFfn.h"
#include "ops/KernelNames.hpp"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using richengine::AwakeClock;
using richengine::metal::BufferStorage;
using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;
using namespace richengine;
using namespace richengine::ops;
using richengine::ops::tuning::bf16ToFloat;
using richengine::ops::tuning::floatToBf16;
using Element = ane::Surface::Element;

// The 35B-family routed expert shape.
constexpr uint32_t kHidden = 2048, kIntermediate = 512, kExperts = 8;
constexpr uint32_t kInputSegment = 1024, kInputSegments = kHidden / kInputSegment;
constexpr uint32_t kRowsMax = 2048;
constexpr uint32_t kOutput = kHidden + 1; // partial + per-token scale row
constexpr std::array<uint32_t, 8> kBuckets{64, 128, 256, 512, 768, 1024, 1536, 2048};
constexpr auto kHandoffBound = std::chrono::seconds(2);
constexpr auto kProgramLimit = std::chrono::seconds(120);

std::mt19937 rng(20261026);

MetalBuffer buffer(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}
MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer b = buffer(backend, bytes, label);
  std::memset(b.contents(), 0, bytes);
  return b;
}

// The split's rotation signs (ops/AneFfn.cpp): both the GPU kernels and the
// MIL's rotation conv read them, so they must be generated identically.
std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> rotationSigns() {
  std::mt19937 generator(20260930);
  std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs{};
  for (float &sign : signs) sign = (generator() & 1) ? -1.0f : 1.0f;
  return signs;
}

// The MIL's [channels, 512, 1, 1] grouped-conv rotation of the intermediate
// rows (ops/AneFfn.cpp rotationBlob).
std::vector<uint8_t> rotationBlob(uint32_t channels, const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> &signs) {
  constexpr uint32_t block = ANE_FFN_INTERMEDIATE_BLOCK;
  std::vector<_Float16> values(uint64_t{channels} * block);
  const float norm = 1.0f / std::sqrt(float(block));
  for (uint32_t channel = 0; channel < channels; ++channel)
    for (uint32_t input = 0; input < block; ++input) {
      const uint32_t output = channel % block;
      const float hadamard = (std::popcount(output & input) & 1) ? -1.0f : 1.0f;
      values[uint64_t{channel} * block + input] = _Float16(signs[output] * hadamard * norm);
    }
  return ane::constantBlob(values);
}

std::string dimensions(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char *type, uint64_t rows, uint64_t width) {
  return std::string("tensor<") + type + ", " + dimensions(rows, width) + ">";
}
std::string fp16(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}
std::string functionName(uint32_t rows) { return "ffn" + std::to_string(rows); }

// An input's declared [channels, width] MIL shape; width 0 reads the chunk's
// rows.
struct Input {
  std::string name;
  const ane::Surface *surface;
  uint32_t rows = 0, width = 0;
};

// One whole expert's SwiGLU over `rows` rows: the AneFfn MIL body with
// channels = the expert's full intermediate (no GPU channel share), two
// hidden input segments and one down segment.
std::string function(const std::vector<Input> &inputs, const ane::Surface &output, uint32_t rows) {
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  for (const Input &input : inputs) {
    const uint32_t width = input.width ? input.width : rows;
    parameters += (parameters.empty() ? "" : ", ") + input.surface->bufferType(input.rows, width) + " " + input.name;
    line(tensor(input.surface->element == Element::Int8 ? "int8" : "fp16", input.rows, width) + " " + input.name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
  }
  for (uint32_t k = 0; k < kInputSegments; ++k)
    f16("x" + std::to_string(k) + "_d", kInputSegment, rows,
        "dequantize(input = x" + std::to_string(k) + "_t, scale = " + unit + ")");
  for (const char *p : {"g", "u"}) {
    for (uint32_t k = 0; k < kInputSegments; ++k) {
      const std::string w = std::string("w") + p + std::to_string(k), m = std::string(p) + "m" + std::to_string(k);
      f16(w + "_d", kIntermediate, kInputSegment, "dequantize(input = " + w + "_t, scale = " + unit + ")");
      line(tensor("fp16", kIntermediate, rows) + " " + m +
           " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + w + "_d, y = x" +
           std::to_string(k) + "_d)");
    }
    f16(std::string(p) + "m_sum1", kIntermediate, rows,
        "add(x = " + std::string(p) + "m0, y = " + std::string(p) + "m1)");
    f16(std::string(p) + "s", kIntermediate, rows,
        "mul(x = " + std::string(p) + "m_sum1, y = s" + std::string(p) + "_t)");
  }
  const std::string c = std::to_string(kIntermediate), r = std::to_string(rows),
                  block = std::to_string(ANE_FFN_INTERMEDIATE_BLOCK);
  f16("gt", kIntermediate, rows, "mul(x = gs, y = tx_t)");
  f16("gh", kIntermediate, rows, "mul(x = gt, y = " + fp16(0.5) + ")");
  f16("th", kIntermediate, rows, "tanh(x = gh)");
  f16("tp", kIntermediate, rows, "add(x = th, y = " + fp16(1.0) + ")");
  f16("silu", kIntermediate, rows, "mul(x = gh, y = tp)");
  f16("h", kIntermediate, rows, "mul(x = silu, y = us)");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c +
       ", 1, " + r + "]))");
  line("tensor<fp16, [" + c + ", " + block + ", 1, 1]> rotation = const()[name = string(\"rotation\"), "
       "val = tensor<fp16, [" + c + ", " + block + ", 1, 1]>(BLOBFILE(path = "
       "string(\"@model_path/weights.bin\"), offset = uint64(" + std::to_string(ane::kConstantOffset) + ")))]");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), "
       "groups = int32(1), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), "
       "strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)");
  f16("hr", kIntermediate, rows, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dimensions(kIntermediate, rows) +
      "))");
  f16("habs", kIntermediate, rows, "abs(x = hr)");
  f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("floor", 1, rows, "maximum(x = peak, y = " + fp16(0x1p-9) + ")");
  f16("inverse", 1, rows, "real_div(x = " + fp16(ANE_FFN_INT8_PEAK) + ", y = floor)");
  f16("hs", kIntermediate, rows, "mul(x = hr, y = inverse)");
  line(tensor("int8", kIntermediate, rows) +
       " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", kIntermediate, rows, "dequantize(input = hq, scale = " + unit + ")");
  f16("hscale", 1, rows, "mul(x = floor, y = " + fp16(double(ANE_FFN_INT8_UNIT) / ANE_FFN_INT8_PEAK) + ")");
  f16("hd0", kIntermediate, rows,
      "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, 0, 0]), size = tensor<int32, [4]>(" +
      dimensions(kIntermediate, rows) + "))");
  f16("wd0_d", kHidden, kIntermediate, "dequantize(input = wd0_t, scale = " + unit + ")");
  line(tensor("fp16", kHidden, rows) +
       " dm0 = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wd0_d, y = hd0)");
  f16("ds", kHidden, rows, "mul(x = dm0, y = sd_t)");
  f16("yt", kOutput, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))");
  line(output.bufferType(kOutput, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + output.strides(kOutput) + "))");
  return "    func " + functionName(rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

// The surfaces of one expert evaluation: activations (shared, one set since
// joins serialize evals), and one of the two alternating weight sets.
struct Activation {
  std::array<ane::Surface, kInputSegments> x;
  ane::Surface tx;
};
struct Weights {
  std::array<ane::Surface, kInputSegments> gate, up;
  ane::Surface gateScale, upScale, down, downScale;
};

int run(MetalBackend &backend, uint32_t rounds) {
  if (const std::optional<std::string> why = ane::unavailable()) {
    std::cerr << "ane-moe-benchmark: Neural Engine interface unavailable: " << *why << '\n';
    return 1;
  }
  const Linear linear(backend.capabilities());
  const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs = rotationSigns();
  const MetalBuffer signsBuffer = [&] {
    MetalBuffer b = buffer(backend, signs.size() * sizeof(float), "ane moe signs");
    std::memcpy(b.contents(), signs.data(), signs.size() * sizeof(float));
    return b;
  }();
  const MetalBuffer status = zeros(backend, 4, "ane moe status");

  // Eight synthetic experts of affine Q4 projections (the packed layout the
  // real 35B MLX target uses).
  std::vector<Projection> gate, up, down;
  gate.reserve(kExperts);
  up.reserve(kExperts);
  down.reserve(kExperts);
  for (uint32_t e = 0; e < kExperts; ++e) {
    gate.push_back(test::deterministicQ4Projection(backend, {kIntermediate, kHidden}, 11 + 3 * e));
    up.push_back(test::deterministicQ4Projection(backend, {kIntermediate, kHidden}, 12 + 3 * e));
    down.push_back(test::deterministicQ4Projection(backend, {kHidden, kIntermediate}, 13 + 3 * e));
  }

  // The activation set, two alternating weight sets and the partial output.
  const auto surface = [&](uint32_t rows, uint32_t width, Element element) {
    return ane::Surface::create(backend, rows, width, element);
  };
  Activation activation{std::array{surface(kInputSegment, kRowsMax, Element::Int8),
                                   surface(kInputSegment, kRowsMax, Element::Int8)},
                        surface(1, kRowsMax, Element::Float16)};
  std::array<Weights, 2> sets;
  for (Weights &set : sets) {
    for (uint32_t k = 0; k < kInputSegments; ++k) {
      set.gate[k] = surface(kIntermediate, kInputSegment, Element::Int8);
      set.up[k] = surface(kIntermediate, kInputSegment, Element::Int8);
    }
    set.gateScale = surface(kIntermediate, 1, Element::Float16);
    set.upScale = surface(kIntermediate, 1, Element::Float16);
    set.down = surface(kHidden, kIntermediate, Element::Int8);
    set.downScale = surface(kHidden, 1, Element::Float16);
  }
  const ane::Surface partial = surface(kOutput, kRowsMax, Element::Float16);
  const MetalBuffer rotated = buffer(backend, uint64_t{kRowsMax} * kHidden * 2, "ane moe rotated");
  // Per-expert weight row scales: gate's, up's and down's, computed once.
  const MetalBuffer rowScales = buffer(backend, uint64_t{kExperts} * (2 * kIntermediate + kHidden) * 2,
                                       "ane moe row scales");
  const auto scalesOf = [&](uint32_t expert, uint32_t matrix) {
    const uint64_t offset = (uint64_t{expert} * (2 * kIntermediate + kHidden) + matrix * kIntermediate) * 2;
    return backend.view(rowScales, offset,
                        uint64_t{matrix == 2 ? kHidden : kIntermediate} * sizeof(_Float16));
  };

  // The program's inputs in declaration order; the weights entries name set
  // 0's surfaces as the shape donors the MIL types them with.
  const std::vector<Input> inputs = [&] {
    std::vector<Input> result;
    for (uint32_t k = 0; k < kInputSegments; ++k)
      result.push_back({"x" + std::to_string(k), &activation.x[k], kInputSegment, 0});
    result.push_back({"tx", &activation.tx, 1, 0});
    for (uint32_t k = 0; k < kInputSegments; ++k) {
      result.push_back({"wg" + std::to_string(k), &sets[0].gate[k], kIntermediate, kInputSegment});
      result.push_back({"wu" + std::to_string(k), &sets[0].up[k], kIntermediate, kInputSegment});
    }
    result.push_back({"sg", &sets[0].gateScale, kIntermediate, 1});
    result.push_back({"su", &sets[0].upScale, kIntermediate, 1});
    result.push_back({"wd0", &sets[0].down, kHidden, kIntermediate});
    result.push_back({"sd", &sets[0].downScale, kHidden, 1});
    return result;
  }();
  // Inputs whose surface alternates between the weight sets, bound per set.
  const auto surfaceFor = [&](const Input &input, uint32_t set) -> const ane::Surface & {
    if (input.name == "sg") return sets[set].gateScale;
    if (input.name == "su") return sets[set].upScale;
    if (input.name == "sd") return sets[set].downScale;
    if (input.name == "wd0") return sets[set].down;
    if (input.name[0] == 'w' && input.name[1] == 'g')
      return sets[set].gate[input.name.back() - '0'];
    if (input.name[0] == 'w' && input.name[1] == 'u')
      return sets[set].up[input.name.back() - '0'];
    return *input.surface;
  };

  // MIL: one function per bucket.
  std::string mil = "program(1.3)\n{\n";
  for (const uint32_t rows : kBuckets) mil += function(inputs, partial, rows);
  mil += "}\n";
  ane::Program program(mil, rotationBlob(kIntermediate, signs), ane::Program::Limits{kProgramLimit, {}});

  // A binding per (bucket, weight set).
  std::vector<std::vector<ane::Program::Binding>> bindings;
  for (const uint32_t rows : kBuckets) {
    const uint32_t procedure = program.procedure(functionName(rows));
    std::vector<ane::Program::Binding> pair;
    for (uint32_t set = 0; set < 2; ++set) {
      std::vector<ane::Surface> surfaces;
      for (const std::string &name : program.inputs(procedure)) {
        const auto input = std::ranges::find(inputs, name, &Input::name);
        if (input == inputs.end()) throw std::logic_error("ANE MoE program has an unknown input " + name);
        surfaces.push_back(surfaceFor(*input, set));
      }
      pair.push_back(program.bind(procedure, surfaces, partial));
    }
    bindings.push_back(std::move(pair));
  }
  const auto bucketIndex = [&](uint32_t rows) {
    const auto found = std::ranges::find_if(kBuckets, [&](uint32_t bucket) { return bucket >= rows; });
    if (found == kBuckets.end()) throw std::invalid_argument("ANE MoE has no function of these rows");
    return uint32_t(found - kBuckets.begin());
  };

  ane::Handoff handoff(backend, kHandoffBound);

  // Weight row scales once (matrix 0 gate, 1 up, 2 down), then the per-eval
  // staging helpers, mirroring AneFfn::addWeights with the ANE taking all
  // channels.
  {
    CommandGraph graph;
    for (uint32_t e = 0; e < kExperts; ++e) {
      // One scale over the whole input row of gate's and up's output rows,
      // over the whole intermediate row of down's.
      for (const auto &[projection, matrix] : {std::pair{&gate[e], 0u}, {&up[e], 1u}}) {
        const AffineWeights &w = projection->affine();
        graph.add("ane_ffn_row_scale_inputs",
                  {w.weights, w.scales, w.biases, scalesOf(e, matrix), signsBuffer},
                  AneFfnWeightParams{uint32_t(kHidden / 64), 0, 0, kHidden, 0, 0, 0},
                  {kIntermediate / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
      }
      const AffineWeights &w = down[e].affine();
      graph.add("ane_ffn_row_scale_intermediate",
                {w.weights, w.scales, w.biases, scalesOf(e, 2), signsBuffer},
                AneFfnWeightParams{uint32_t(kIntermediate / 64), 0, 0, kIntermediate, 0, 0, 0},
                {kHidden / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
    }
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  }
  const auto addStage = [&](CommandGraph &graph, uint32_t expert, uint32_t set) {
    const Weights &target = sets[set];
    const auto scaleStride = [](const ane::Surface &s) { return s.strideBytes / uint32_t{sizeof(_Float16)}; };
    for (uint32_t k = 0; k < kInputSegments; ++k) {
      for (const auto &[planes, wset, scale, matrix] :
           {std::tuple{&gate[expert], &target.gate[k], &target.gateScale, 0u},
            {&up[expert], &target.up[k], &target.upScale, 1u}}) {
        const AffineWeights &w = planes->affine();
        graph.add("ane_ffn_weights_inputs",
                  {w.weights, w.scales, w.biases, scalesOf(expert, matrix), wset->buffer, scale->buffer,
                   signsBuffer},
                  AneFfnWeightParams{uint32_t(kHidden / 64), 0, k * kInputSegment, kInputSegment,
                                     wset->strideBytes, scaleStride(*scale), 0},
                  {kIntermediate / ANE_FFN_WEIGHT_ROWS, kInputSegment / ANE_FFN_INPUT_BLOCK, 1},
                  {ANE_FFN_WEIGHT_THREADS, 1, 1});
      }
    }
    const AffineWeights &w = down[expert].affine();
    graph.add("ane_ffn_weights_intermediate",
              {w.weights, w.scales, w.biases, scalesOf(expert, 2), target.down.buffer,
               target.downScale.buffer, signsBuffer},
              AneFfnWeightParams{uint32_t(kIntermediate / 64), 0, 0, kIntermediate, target.down.strideBytes,
                                 scaleStride(target.downScale), 0},
              {kHidden / ANE_FFN_WEIGHT_ROWS, kIntermediate / ANE_FFN_INTERMEDIATE_BLOCK, 1},
              {ANE_FFN_WEIGHT_THREADS, 1, 1});
  };

  // The chunk's buffers: normalized rows and the GPU path's FFN scratch.
  const auto bf16Rows = [&](uint32_t width) { return buffer(backend, uint64_t{kRowsMax} * width * 2, "rows"); };
  const auto sumRows = [&](uint32_t width) { return buffer(backend, uint64_t{kRowsMax} * (width / 64) * 4, "sums"); };
  PrefillFfnBuffers ffn{bf16Rows(kHidden), sumRows(kHidden), bf16Rows(kIntermediate), bf16Rows(kIntermediate),
                        sumRows(kIntermediate), {}};
  const MetalBuffer residual = zeros(backend, uint64_t{kRowsMax} * kHidden * 2, "residual");
  const MetalBuffer gpuOut = buffer(backend, uint64_t{kRowsMax} * kHidden * 2, "gpu out"),
                    aneOut = buffer(backend, uint64_t{kRowsMax} * kHidden * 2, "ane out");
  const auto fillRows = [&](uint32_t rows) {
    // The rows the split's own verify writes: uniform on +-2.
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    auto *values = static_cast<uint16_t *>(ffn.normalized.contents());
    for (uint64_t i = 0; i < uint64_t{rows} * kHidden; ++i)
      values[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(uniform(rng)) >> 16);
  };
  const auto submitWait = [&](CommandGraph &graph) {
    const auto start = AwakeClock::now();
    static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    return std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count();
  };
  const auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };

  // Per-eval encode: pack the expert's rows, hand off the evaluation, stage
  // the next expert's weights under it as the split stages a layer's under
  // its GPU part, wait and join (weight sets alternate).
  const auto addEval = [&](CommandGraph &graph, uint32_t bucket, uint32_t set, uint32_t rows,
                         bool stageNext, uint32_t nextExpert) {
    const uint32_t tiles = (rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
    graph.add(std::string(kAneFfnRotate), {ffn.normalized, signsBuffer, rotated, activation.tx.buffer},
              AneFfnRotateParams{kHidden}, {rows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
    for (uint32_t k = 0; k < kInputSegments; ++k)
      graph.add(std::string(kAneFfnPack), {rotated, activation.x[k].buffer},
                AneFfnPackParams{kHidden, k * kInputSegment, activation.x[k].strideBytes},
                {tiles, kInputSegment / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
    const uint64_t ready = handoff.next();
    graph.signal(handoff.event(), ready);
    if (stageNext) addStage(graph, nextExpert, set ^ 1);
    const uint64_t done = handoff.next();
    graph.wait(handoff.event(), done);
    graph.add(std::string(kAneFfnJoin), {aneOut, partial.buffer, activation.tx.buffer, status},
              AneFfnJoinParams{kHidden, partial.strideBytes / uint32_t{sizeof(_Float16)}, rows},
              {tiles, kHidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
    handoff.queue(ready, done,
                  [&, bucket, set](const metal::SharedEvent &event, uint64_t wait, uint64_t signal,
                                   ane::Handoff::Report report) {
                    program.enqueue(bindings[bucket][set], event, wait, signal, std::move(report));
                  });
  };

  std::printf("%s, GPU family %u, %u cores: ANE MoE expert A/B (%u rounds)\n",
              backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
              backend.capabilities().gpuCoreCount, rounds);
  std::printf("expert: hidden %u, intermediate %u, affine Q4; ANE takes the whole expert\n",
            kHidden, kIntermediate);
  std::printf("%8s %10s %10s %10s %10s %8s\n", "rows", "gpu_ms", "ane_ms", "side_ms", "stage_ms", "err_%");

  struct Row {
    uint32_t rows;
    double gpuMs = 0, aneMs = 0, overheadMs = 0, stageMs = 0, error = 0;
  };
  std::vector<Row> measured;
  for (const uint32_t rows : kBuckets) {
    fillRows(rows);
    // A: the GPU's own SwiGLU of the expert's projections.
    std::vector<double> gpu;
    std::vector<uint16_t> reference;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      CommandGraph graph;
      linear.addPrefillSums(graph, ffn.normalized, ffn.sums, gate[0], rows);
      linear.addPrefillSwiGlu(graph, {&gate[0], &up[0], &down[0]}, ffn, residual, gpuOut, rows);
      const double ms = submitWait(graph);
      if (i) gpu.push_back(ms);
      if (i == rounds) {
        const uint16_t *bits = static_cast<const uint16_t *>(gpuOut.contents());
        reference.assign(bits, bits + uint64_t{rows} * kHidden);
      }
    }
    // B: stage the expert, pack the rows, evaluate, join.
    std::vector<double> ane;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      std::memset(aneOut.contents(), 0, uint64_t{rows} * kHidden * 2);
      CommandGraph graph;
      addStage(graph, 0, 0);
      const uint32_t bucket = bucketIndex(rows);
      addEval(graph, bucket, 0, rows, false, 0);
      static_cast<void>(submitWait(graph));
      const std::optional<AwakeClock::duration> ran = handoff.finish();
      if (!ran || std::exchange(*static_cast<uint32_t *>(status.contents()), 0u)) {
        std::printf("  ANE eval at %u rows failed (%s)\n", rows, handoff.reason().c_str());
        return 1;
      }
      if (i) {
        ane.push_back(std::chrono::duration<double, std::milli>(*ran).count());
      }
    }
    // RMS error of the ANE path's output against the GPU's.
    double difference = 0.0, magnitude = 0.0;
    const uint16_t *got = static_cast<const uint16_t *>(aneOut.contents());
    for (uint64_t i = 0; i < uint64_t{rows} * kHidden; ++i) {
      const double delta = double(bf16ToFloat(got[i])) - bf16ToFloat(reference[i]);
      difference += delta * delta;
      magnitude += double(bf16ToFloat(reference[i])) * bf16ToFloat(reference[i]);
    }
    // GPU-side overhead alone: pack + join of the bucket's rows.
    std::vector<double> overhead;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      CommandGraph graph;
      const uint32_t tiles = (rows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
      graph.add(std::string(kAneFfnRotate), {ffn.normalized, signsBuffer, rotated, activation.tx.buffer},
                AneFfnRotateParams{kHidden}, {rows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
      for (uint32_t k = 0; k < kInputSegments; ++k)
        graph.add(std::string(kAneFfnPack), {rotated, activation.x[k].buffer},
                  AneFfnPackParams{kHidden, k * kInputSegment, activation.x[k].strideBytes},
                  {tiles, kInputSegment / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
      graph.add(std::string(kAneFfnJoin), {aneOut, partial.buffer, activation.tx.buffer, status},
                AneFfnJoinParams{kHidden, partial.strideBytes / uint32_t{sizeof(_Float16)}, rows},
                {tiles, kHidden / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
      const double ms = submitWait(graph);
      if (i) overhead.push_back(ms);
    }
    // Weight staging alone.
    std::vector<double> stage;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      CommandGraph graph;
      addStage(graph, 0, 0);
      const double ms = submitWait(graph);
      if (i) stage.push_back(ms);
    }
    Row row{rows, median(gpu), median(ane), median(overhead), median(stage),
            std::sqrt(difference / magnitude) * 100.0};
    measured.push_back(row);
    std::printf("%8u %10.3f %10.3f %10.3f %10.3f %8.2f\n", rows, row.gpuMs, row.aneMs, row.overheadMs,
                row.stageMs, row.error);
  }

  // The eval-rate ceiling: K expert evals chained in one command, each
  // packing its rows, evaluating, staging the next expert's weights under
  // the evaluation and joining — the per-layer loop shape.
  std::printf("\nchained evals (one command, weight sets alternating):\n");
  std::printf("%8s %8s %10s %10s %10s\n", "K", "rows", "wall_ms", "ane_ms", "evals/s");
  for (const uint32_t k : {8u, 16u, 32u, 64u}) {
    for (const uint32_t rows : {64u, 128u, 256u}) {
      fillRows(rows);
      CommandGraph graph;
      addStage(graph, 0, 0);
      const uint32_t bucket = bucketIndex(rows);
      for (uint32_t e = 0; e < k; ++e)
        addEval(graph, bucket, e & 1, rows, e + 1 < k, (e + 1) % kExperts);
      const double wall = submitWait(graph);
      const std::optional<AwakeClock::duration> ran = handoff.finish();
      if (!ran || std::exchange(*static_cast<uint32_t *>(status.contents()), 0u)) {
        std::printf("  chained evals failed (%s)\n", handoff.reason().c_str());
        return 1;
      }
      const double aneMs = std::chrono::duration<double, std::milli>(*ran).count();
      std::printf("%8u %8u %10.3f %10.3f %10.0f\n", k, rows, wall, aneMs, k / (aneMs / 1000.0));
    }
  }

  // Layer projection: a 2048-row chunk's correlated routing (70% of each
  // row's routes to the hot 64, 30% uniform — gguf-moe-benchmark's skew),
  // the GPU-alone expert pass measured beside this bench, and the measured
  // per-eval costs above.
  std::printf("\nlayer projection (2048 rows, 256 experts, top 8, 70/30 hot-64 skew):\n");
  {
    std::vector<uint32_t> counts(256, 0);
    std::mt19937 local(9);
    std::vector<uint32_t> order(256);
    std::iota(order.begin(), order.end(), 0u);
    for (uint32_t r = 0; r < 2048; ++r) {
      if (r % 8 == 0) std::shuffle(order.begin(), order.end(), local);
      std::uniform_real_distribution<float> pick(0.0f, 1.0f);
      std::vector<bool> chosen(256, false);
      uint32_t left = 8;
      while (left--) {
        const uint32_t e = pick(local) < 0.7f ? order[local() % 64] : order[local() % 256];
        if (!chosen[e]) chosen[e] = true, ++counts[e];
        else ++left;
      }
    }
    std::vector<uint32_t> sorted(counts);
    std::sort(sorted.begin(), sorted.end(), std::greater<uint32_t>());
    const uint32_t routed = std::accumulate(counts.begin(), counts.end(), 0u);
    // An expert's rows evaluate on the next function's rows: the cost is the
    // bucket's eval, not the rows'.
    const auto evalMs = [&](uint32_t rows) { return measured[bucketIndex(rows)].aneMs; };
    // The GPU's expert-pass share of a layer (gguf-moe-benchmark, skewed):
    // ~5.6 ms of a ~6.7 ms layer at this row count on this Mac.
    const double gpuLayer = 6.745, gpuExperts = 5.639, gpuFixed = gpuLayer - gpuExperts;
    std::printf("%8s %10s %10s %10s %10s %10s\n", "ane_K", "rows_cov", "ane_busy", "gpu_left", "layer_ms",
                "speedup");
    for (const uint32_t k : {8u, 16u, 24u, 32u, 48u, 64u}) {
      uint32_t covered = 0;
      double aneBusy = 0;
      for (uint32_t i = 0; i < k && i < sorted.size(); ++i) {
        covered += sorted[i];
        aneBusy += evalMs(sorted[i]);
      }
      const double share = double(covered) / routed;
      // GPU keeps the uncovered routed rows at roughly the same per-row
      // cost; the ANE path's own GPU work (pack/join per eval, staging) adds.
      const double gpuLeft = gpuExperts * (1.0 - share);
      const double overhead = k * (measured[0].overheadMs + measured[0].stageMs);
      const double layer = gpuFixed + std::max(gpuLeft + overhead, aneBusy);
      std::printf("%8u %9.0f%% %10.3f %10.3f %10.3f %9.2fx\n", k, 100.0 * share, aneBusy, gpuLeft, layer,
                  gpuLayer / layer);
    }
  }
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: ane-moe-benchmark <metallib> [rounds]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      return run(backend, argc > 2 ? std::stoul(argv[2]) : 20);
    } catch (const std::exception &error) {
      std::cerr << "ane-moe-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}

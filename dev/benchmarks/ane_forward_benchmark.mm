// Two A/Bs around the ANE per-evaluation floor (docs/ANE_EXPERIMENTS.md).
//
// Floor anatomy: what the ~0.2 ms the MoE probe measured per evaluation is
// made of — programs of rising input count and compute, chained K times in
// one command with only the event signals between them. The deltas price
// dispatch, per-input surface handling and compute separately.
//
// d1 chained full-forward: the d1-3B decision model (LFM2 family: 30 layers,
// hidden 2048, FFN intermediate 10752, 22 conv + 8 attention layers, one
// forward per judgment, no decode) as a chain of ANE evaluations — its real
// FFN as one eval per layer, its conv and attention projections as matmul
// evals, against the GPU's real kernels at the same shapes, then the
// projected forward time per prompt size.
//   ane-forward-benchmark <metallib> [rounds]
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
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string>
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
using Element = ane::Surface::Element;

constexpr uint32_t kRowsMax = 2048;
constexpr auto kHandoffBound = std::chrono::seconds(2);
constexpr auto kProgramLimit = std::chrono::seconds(120);

MetalBuffer buffer(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}
MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer b = buffer(backend, bytes, label);
  std::memset(b.contents(), 0, bytes);
  return b;
}

std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> rotationSigns() {
  std::mt19937 generator(20260930);
  std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs{};
  for (float &sign : signs) sign = (generator() & 1) ? -1.0f : 1.0f;
  return signs;
}
std::vector<_Float16> rotationValues(uint32_t channels,
                                     const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> &signs) {
  constexpr uint32_t block = ANE_FFN_INTERMEDIATE_BLOCK;
  std::vector<_Float16> values(uint64_t{channels} * block);
  const float norm = 1.0f / std::sqrt(float(block));
  for (uint32_t channel = 0; channel < channels; ++channel)
    for (uint32_t input = 0; input < block; ++input) {
      const uint32_t output = channel % block;
      const float hadamard = (std::popcount(output & input) & 1) ? -1.0f : 1.0f;
      values[uint64_t{channel} * block + input] = _Float16(signs[output] * hadamard * norm);
    }
  return values;
}
std::vector<uint8_t> rotationBlob(uint32_t channels, const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> &signs) {
  return ane::constantBlob(rotationValues(channels, signs));
}

// Several fp16 constant tensors in one weights blob: a 64-byte header, then
// per tensor a 64-byte record {sentinel, type fp16, size, absolute data
// offset} followed by its data, as maderix's QKV blob lays them out and the
// dense split's single-record blob does. Returns the blob and each record's
// offset for its BLOBFILE.
std::pair<std::vector<uint8_t>, std::vector<uint64_t>> constantBlobs(
    std::initializer_list<std::span<const _Float16>> tensors) {
  uint64_t size = 64;
  for (const auto &tensor : tensors) size += 64 + tensor.size_bytes();
  std::vector<uint8_t> blob(size);
  const auto put = [&](uint64_t offset, auto value) { std::memcpy(blob.data() + offset, &value, sizeof value); };
  put(0, uint32_t{1}); // blobs
  put(4, uint32_t{2}); // storage version
  std::vector<uint64_t> offsets;
  uint64_t at = 64;
  for (const auto &tensor : tensors) {
    offsets.push_back(at);
    put(at, uint32_t{0xdeadbeef});
    put(at + 4, uint32_t{1}); // fp16
    put(at + 8, uint64_t{tensor.size_bytes()});
    put(at + 16, at + 64);
    std::memcpy(blob.data() + at + 64, tensor.data(), tensor.size_bytes());
    at += 64 + tensor.size_bytes();
  }
  return {blob, offsets};
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

// An input's declared [channels, width] MIL shape; width 0 reads the eval's
// rows.
struct Input {
  std::string name;
  const ane::Surface *surface;
  uint32_t rows = 0, width = 0;
};

const char *milType(const Input &input) {
  return ane::Surface::milType(input.surface->element);
}

// A SwiGLU FFN of `shape` — the dense split's MIL body generalized: ane
// channels of gate/up over inputSegment-wide hidden segments, then down over
// the channel segments, all W8A8 with the rotation conv and per-token
// rescale. One function of `rows` rows.
struct Shape {
  uint32_t hidden, inputSegment, channels;
  std::vector<uint32_t> down;
};

// `tap`/`tapHeight` concat an intermediate tensor into the output after
// hscale: the output surface gains `tapHeight` rows holding it — a forward
// tap with no extra evaluation.
std::string function(const Shape &shape, const std::vector<Input> &inputs, const ane::Surface &output,
                     uint32_t rows, const char *tap = nullptr, uint32_t tapHeight = 0) {
  const uint32_t inputSegments = shape.hidden / shape.inputSegment;
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  for (const Input &input : inputs) {
    const uint32_t width = input.width ? input.width : rows;
    parameters += (parameters.empty() ? "" : ", ") + input.surface->bufferType(input.rows, width) + " " + input.name;
    line(tensor(milType(input), input.rows, width) + " " + input.name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
  }
  const auto matmul = [&](const std::string &name, const std::string &weights, const std::string &values,
                          uint64_t height, uint64_t width) {
    f16(weights + "_d", height, width, "dequantize(input = " + weights + "_t, scale = " + unit + ")");
    line(tensor("fp16", height, rows) + " " + name +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + weights + "_d, y = " + values +
         ")");
  };
  const auto sum = [&](const std::string &prefix, size_t terms, uint64_t height) {
    std::string total = prefix + "0";
    for (size_t term = 1; term < terms; ++term) {
      const std::string next = prefix + "_sum" + std::to_string(term);
      f16(next, height, rows, "add(x = " + total + ", y = " + prefix + std::to_string(term) + ")");
      total = next;
    }
    return total;
  };
  for (uint32_t k = 0; k < inputSegments; ++k) {
    const std::string x = "x" + std::to_string(k);
    f16(x + "_d", shape.inputSegment, rows, "dequantize(input = " + x + "_t, scale = " + unit + ")");
  }
  for (const char *p : {"g", "u"}) {
    for (uint32_t k = 0; k < inputSegments; ++k)
      matmul(std::string(p) + "m" + std::to_string(k), "w" + std::string(p) + std::to_string(k),
             "x" + std::to_string(k) + "_d", shape.channels, shape.inputSegment);
    f16(std::string(p) + "s", shape.channels, rows,
        "mul(x = " + sum(std::string(p) + "m", inputSegments, shape.channels) + ", y = s" + std::string(p) + "_t)");
  }
  const std::string c = std::to_string(shape.channels), r = std::to_string(rows),
                  block = std::to_string(ANE_FFN_INTERMEDIATE_BLOCK);
  f16("gt", shape.channels, rows, "mul(x = gs, y = tx_t)");
  f16("gh", shape.channels, rows, "mul(x = gt, y = " + fp16(0.5) + ")");
  f16("th", shape.channels, rows, "tanh(x = gh)");
  f16("tp", shape.channels, rows, "add(x = th, y = " + fp16(1.0) + ")");
  f16("silu", shape.channels, rows, "mul(x = gh, y = tp)");
  f16("h", shape.channels, rows, "mul(x = silu, y = us)");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c +
       ", 1, " + r + "]))");
  line("tensor<fp16, [" + c + ", " + block +
       ", 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" + c + ", " + block +
       ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" +
       std::to_string(ane::kConstantOffset) + ")))]");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
       std::to_string(shape.channels / ANE_FFN_INTERMEDIATE_BLOCK) +
       "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, "
       "1]), weight = rotation, x = h4)");
  f16("hr", shape.channels, rows, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dimensions(shape.channels, rows) +
      "))");
  f16("habs", shape.channels, rows, "abs(x = hr)");
  f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("floor", 1, rows, "maximum(x = peak, y = " + fp16(0x1p-9) + ")");
  f16("inverse", 1, rows, "real_div(x = " + fp16(ANE_FFN_INT8_PEAK) + ", y = floor)");
  f16("hs", shape.channels, rows, "mul(x = hr, y = inverse)");
  line(tensor("int8", shape.channels, rows) +
       " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", shape.channels, rows, "dequantize(input = hq, scale = " + unit + ")");
  f16("hscale", 1, rows, "mul(x = floor, y = " + fp16(double(ANE_FFN_INT8_UNIT) / ANE_FFN_INT8_PEAK) + ")");
  uint32_t begin = 0;
  for (size_t i = 0; i < shape.down.size(); ++i) {
    const uint32_t width = shape.down[i];
    const std::string index = std::to_string(i), slice = "hd" + index;
    f16(slice, width, rows,
        "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
            ", 0]), size = tensor<int32, [4]>(" + dimensions(width, rows) + "))");
    matmul("dm" + index, "wd" + index, slice, shape.hidden, width);
    begin += width;
  }
  f16("ds", shape.hidden, rows, "mul(x = " + sum("dm", shape.down.size(), shape.hidden) + ", y = sd_t)");
  const uint32_t outRows = shape.hidden + 1 + tapHeight;
  f16("yt", outRows, rows,
      "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale" +
          (tap ? std::string(", ") + tap : std::string()) + "))");
  line(output.bufferType(outRows, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + output.strides(outRows) + "))");
  return "    func " + functionName(rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

// `function` with the normalization moved into MIL: the x inputs arrive
// fp16 and unnormalized; RMSNorm (square, reduce_sum, pow -0.5), the
// Hadamard rotation conv, the per-token peak/scale and the int8
// quantize/dequantize all run on the ANE before the same matmul path —
// what the GPU's rmsnorm + ane_ffn_rotate/pack kernels + tx surface would
// otherwise do. tx is computed in MIL instead of arriving as an input.
// Weights blob record 0 is the h rotation at [channels, block], record 1
// the input rotation at [inputSegment, block], offset `rotXOffset`.
std::string functionNorm(const Shape &shape, const std::vector<Input> &inputs, const ane::Surface &output,
                         uint32_t rows, uint64_t rotXOffset) {
  const uint32_t inputSegments = shape.hidden / shape.inputSegment;
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  const std::string block = std::to_string(ANE_FFN_INTERMEDIATE_BLOCK),
                  seg = std::to_string(shape.inputSegment), r = std::to_string(rows);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  const auto f16r = [&](const std::string &name, uint64_t height, const std::string &expression) {
    f16(name, height, rows, expression);
  };
  for (const Input &input : inputs) {
    const uint32_t width = input.width ? input.width : rows;
    parameters += (parameters.empty() ? "" : ", ") + input.surface->bufferType(input.rows, width) + " " + input.name;
    line(tensor(milType(input), input.rows, width) + " " + input.name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
  }
  // RMSNorm: mean of x^2 over all hidden channels -> rsqrt broadcast.
  for (uint32_t k = 0; k < inputSegments; ++k)
    f16r("sq" + std::to_string(k), shape.inputSegment,
         "mul(x = x" + std::to_string(k) + "_t, y = x" + std::to_string(k) + "_t)");
  std::string total = "sq0";
  for (uint32_t k = 1; k < inputSegments; ++k) {
    f16r("sqs" + std::to_string(k), shape.inputSegment,
         "add(x = " + total + ", y = sq" + std::to_string(k) + ")");
    total = "sqs" + std::to_string(k);
  }
  f16("ms", 1, rows, "reduce_sum(x = " + total + ", axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("ms1", 1, rows, "mul(x = ms, y = " + fp16(1.0 / shape.hidden) + ")");
  f16("mse", 1, rows, "add(x = ms1, y = " + fp16(1e-5) + ")");
  f16("rs", 1, rows, "pow(x = mse, y = " + fp16(-0.5) + ")");
  // Normalize, rotate, per-token peak over every segment, int8 quantize.
  for (uint32_t k = 0; k < inputSegments; ++k) {
    const std::string kn = "xn" + std::to_string(k);
    f16r(kn, shape.inputSegment, "mul(x = x" + std::to_string(k) + "_t, y = rs)");
    line("tensor<fp16, [1, " + seg + ", 1, " + r + "]> " + kn + "4 = reshape(x = " + kn +
         ", shape = tensor<int32, [4]>([1, " + seg + ", 1, " + r + "]))");
  }
  line("tensor<fp16, [" + seg + ", " + block + ", 1, 1]> rotationx = const()[name = string(\"rotationx\"), val = "
       "tensor<fp16, [" + seg + ", " + block + ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), "
       "offset = uint64(" + std::to_string(rotXOffset) + ")))]");
  for (uint32_t k = 0; k < inputSegments; ++k) {
    const std::string kn = "xn" + std::to_string(k), kr = "xr" + std::to_string(k);
    line("tensor<fp16, [1, " + seg + ", 1, " + r + "]> " + kr + "4 = conv(dilations = tensor<int32, [2]>([1, 1]), "
         "groups = int32(" + std::to_string(shape.inputSegment / ANE_FFN_INTERMEDIATE_BLOCK) +
         "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), "
         "strides = tensor<int32, [2]>([1, 1]), weight = rotationx, x = " + kn + "4)");
    f16r(kr, shape.inputSegment, "reshape(x = " + kr + "4, shape = tensor<int32, [4]>(" +
         dimensions(shape.inputSegment, rows) + "))");
    f16r("xa" + std::to_string(k), shape.inputSegment, "abs(x = " + kr + ")");
    f16("xp" + std::to_string(k), 1, rows,
        "reduce_max(x = xa" + std::to_string(k) + ", axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  }
  std::string xpeak = "xp0";
  for (uint32_t k = 1; k < inputSegments; ++k) {
    const std::string next = k + 1 == inputSegments ? "xpeak" : "xpm" + std::to_string(k);
    f16(next, 1, rows, "maximum(x = " + xpeak + ", y = xp" + std::to_string(k) + ")");
    xpeak = next;
  }
  if (inputSegments == 1) f16("xpeak", 1, rows, "mul(x = xp0, y = fp16(1))");
  f16("xfloor", 1, rows, "maximum(x = xpeak, y = " + fp16(0x1p-9) + ")");
  f16("xinv", 1, rows, "real_div(x = " + fp16(ANE_FFN_INT8_PEAK) + ", y = xfloor)");
  for (uint32_t k = 0; k < inputSegments; ++k) {
    const std::string kn = std::to_string(k);
    f16r("xs" + kn, shape.inputSegment, "mul(x = xr" + kn + ", y = xinv)");
    line(tensor("int8", shape.inputSegment, rows) +
         " xq" + kn + " = quantize(input = xs" + kn + ", scale = fp16(1), output_dtype = string(\"int8\"))");
    f16r("xd" + kn, shape.inputSegment, "dequantize(input = xq" + kn + ", scale = " + unit + ")");
  }
  f16("txmil", 1, rows, "mul(x = xfloor, y = " + fp16(double(ANE_FFN_INT8_UNIT) / ANE_FFN_INT8_PEAK) + ")");
  const auto matmul = [&](const std::string &name, const std::string &weights, const std::string &values,
                          uint64_t height, uint64_t width) {
    f16(weights + "_d", height, width, "dequantize(input = " + weights + "_t, scale = " + unit + ")");
    line(tensor("fp16", height, rows) + " " + name +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + weights + "_d, y = " + values +
         ")");
  };
  const auto sum = [&](const std::string &prefix, size_t terms, uint64_t height) {
    std::string sum = prefix + "0";
    for (size_t term = 1; term < terms; ++term) {
      const std::string next = prefix + "_sum" + std::to_string(term);
      f16(next, height, rows, "add(x = " + sum + ", y = " + prefix + std::to_string(term) + ")");
      sum = next;
    }
    return sum;
  };
  for (const char *p : {"g", "u"}) {
    for (uint32_t k = 0; k < inputSegments; ++k)
      matmul(std::string(p) + "m" + std::to_string(k), "w" + std::string(p) + std::to_string(k),
             "xd" + std::to_string(k), shape.channels, shape.inputSegment);
    f16(std::string(p) + "s", shape.channels, rows,
        "mul(x = " + sum(std::string(p) + "m", inputSegments, shape.channels) + ", y = s" + std::string(p) + "_t)");
  }
  const std::string c = std::to_string(shape.channels);
  f16("gt", shape.channels, rows, "mul(x = gs, y = txmil)");
  f16("gh", shape.channels, rows, "mul(x = gt, y = " + fp16(0.5) + ")");
  f16("th", shape.channels, rows, "tanh(x = gh)");
  f16("tp", shape.channels, rows, "add(x = th, y = " + fp16(1.0) + ")");
  f16("silu", shape.channels, rows, "mul(x = gh, y = tp)");
  f16("h", shape.channels, rows, "mul(x = silu, y = us)");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + c +
       ", 1, " + r + "]))");
  line("tensor<fp16, [" + c + ", " + block +
       ", 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" + c + ", " + block +
       ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(" +
       std::to_string(ane::kConstantOffset) + ")))]");
  line("tensor<fp16, [1, " + c + ", 1, " + r + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" +
       std::to_string(shape.channels / ANE_FFN_INTERMEDIATE_BLOCK) +
       "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, "
       "1]), weight = rotation, x = h4)");
  f16("hr", shape.channels, rows, "reshape(x = hr4, shape = tensor<int32, [4]>(" + dimensions(shape.channels, rows) +
      "))");
  f16("habs", shape.channels, rows, "abs(x = hr)");
  f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
  f16("floor", 1, rows, "maximum(x = peak, y = " + fp16(0x1p-9) + ")");
  f16("inverse", 1, rows, "real_div(x = " + fp16(ANE_FFN_INT8_PEAK) + ", y = floor)");
  f16("hs", shape.channels, rows, "mul(x = hr, y = inverse)");
  line(tensor("int8", shape.channels, rows) +
       " hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))");
  f16("hd", shape.channels, rows, "dequantize(input = hq, scale = " + unit + ")");
  f16("hscale", 1, rows, "mul(x = floor, y = " + fp16(double(ANE_FFN_INT8_UNIT) / ANE_FFN_INT8_PEAK) + ")");
  uint32_t begin = 0;
  for (size_t i = 0; i < shape.down.size(); ++i) {
    const uint32_t width = shape.down[i];
    const std::string index = std::to_string(i), slice = "hd" + index;
    f16(slice, width, rows,
        "slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + std::to_string(begin) +
            ", 0]), size = tensor<int32, [4]>(" + dimensions(width, rows) + "))");
    matmul("dm" + index, "wd" + index, slice, shape.hidden, width);
    begin += width;
  }
  f16("ds", shape.hidden, rows, "mul(x = " + sum("dm", shape.down.size(), shape.hidden) + ", y = sd_t)");
  f16("yt", shape.hidden + 1, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))");
  line(output.bufferType(shape.hidden + 1, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + output.strides(shape.hidden + 1) + "))");
  return "    func " + functionName(rows) + "n<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

// A bare matmul y[N,R] = w[N,I] x x[I,R] in fp16 out, for the floor programs
// and the conv/attention projection proxies.
std::string matmulFunction(const char *name, const Input &x, bool xInt8, const Input &w, const Input *scale,
                           const ane::Surface &output, uint32_t rows) {
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters = x.surface->bufferType(x.rows, rows) + " " + x.name + ", " +
                           w.surface->bufferType(w.rows, w.width) + " " + w.name, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  line(tensor(milType(x), x.rows, rows) + " " + x.name + "_t = tensor_buffer_to_tensor<ios17>(input = " + x.name +
       ")");
  line(tensor(milType(w), w.rows, w.width) + " " + w.name + "_t = tensor_buffer_to_tensor<ios17>(input = " + w.name +
       ")");
  if (scale) {
    parameters += ", " + scale->surface->bufferType(scale->rows, 1) + " " + scale->name;
    line(tensor(milType(*scale), scale->rows, 1) + " " + scale->name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + scale->name + ")");
  }
  if (xInt8)
    line(tensor("fp16", x.rows, rows) + " " + x.name + "_d = dequantize(input = " + x.name + "_t, scale = " + unit +
         ")");
  else
    line(tensor("fp16", x.rows, rows) + " " + x.name + "_d = mul(x = " + x.name + "_t, y = fp16(1))");
  line(tensor("fp16", w.rows, w.width) + " " + w.name + "_d = dequantize(input = " + w.name + "_t, scale = " + unit +
       ")");
  line(tensor("fp16", w.rows, rows) + " mm = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " +
       w.name + "_d, y = " + x.name + "_d)");
  line(tensor("fp16", w.rows, rows) +
       std::string(" yt = ") + (scale ? "mul(x = mm, y = " + scale->name + "_t)" : "mul(x = mm, y = fp16(1))"));
  line(output.bufferType(w.rows, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + output.strides(w.rows) + "))");
  return std::string("    func ") + name + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

// The floor probe: program `p` chained `k` times at `rows`, only the event
// signals between evals. Returns ms/eval.
struct Probe {
  ane::Program *program;
  ane::Program::Binding binding;
};
// A command must hold a dispatch; `dispatch` runs one trivial kernel once.
double chainN(MetalBackend &backend, ane::Handoff &handoff, std::span<const Probe> probes, uint32_t k,
              const std::function<void(CommandGraph &)> &dispatch) {
  CommandGraph graph;
  if (dispatch) dispatch(graph);
  for (uint32_t i = 0; i < k; ++i) {
    for (const Probe &probe : probes) {
      const uint64_t ready = handoff.next(), done = handoff.next();
      graph.signal(handoff.event(), ready);
      graph.wait(handoff.event(), done);
      handoff.queue(ready, done,
                    [&probe](const metal::SharedEvent &event, uint64_t wait, uint64_t signal,
                             ane::Handoff::Report report) {
                      probe.program->enqueue(probe.binding, event, wait, signal, std::move(report));
                    });
    }
  }
  static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
  const std::optional<AwakeClock::duration> ran = handoff.finish();
  if (!ran) throw std::runtime_error("ANE chain failed (" + handoff.reason() + ")");
  return std::chrono::duration<double, std::milli>(*ran).count() / k;
}
double chain(MetalBackend &backend, ane::Handoff &handoff, const Probe &probe, uint32_t k,
             const std::function<void(CommandGraph &)> &dispatch) {
  return chainN(backend, handoff, std::span<const Probe>(&probe, 1), k, dispatch);
}

int run(MetalBackend &backend, uint32_t rounds) {
  if (const std::optional<std::string> why = ane::unavailable()) {
    std::cerr << "ane-forward-benchmark: Neural Engine interface unavailable: " << *why << '\n';
    return 1;
  }
  const Linear linear(backend.capabilities());
  const std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs = rotationSigns();
  ane::Handoff handoff(backend, kHandoffBound);
  const auto surface = [&](uint32_t rows, uint32_t width, Element element) {
    return ane::Surface::create(backend, rows, width, element);
  };
  const auto fill = [](ane::Surface &s, uint8_t value) {
    std::memset(s.buffer.contents(), value, s.buffer.sizeBytes());
  };
  const auto fillScales = [](ane::Surface &s) {
    auto *halves = static_cast<_Float16 *>(s.buffer.contents());
    for (uint64_t i = 0; i < s.buffer.sizeBytes() / 2; ++i) halves[i] = _Float16(0.01);
  };

  // Shared surfaces: activations, token scales, per-shape weight sets, a
  // partial output and one output per bare-matmul width.
  std::array<ane::Surface, 2> x{surface(1024, kRowsMax, Element::Int8), surface(1024, kRowsMax, Element::Int8)};
  ane::Surface tx = surface(1, kRowsMax, Element::Float16);
  fill(x[0], 7);
  fill(x[1], 5);
  fillScales(tx);
  const ane::Surface partial = surface(2049, kRowsMax, Element::Float16);

  struct FfnSet {
    Shape shape;
    std::array<ane::Surface, 2> gate, up;
    ane::Surface gateScale, upScale;
    std::vector<ane::Surface> down;
    ane::Surface downScale;
    std::vector<Input> inputs;
  };
  const auto ffnSet = [&](uint32_t channels, std::vector<uint32_t> down) {
    FfnSet set{Shape{2048, 1024, channels, std::move(down)},
               {surface(channels, 1024, Element::Int8), surface(channels, 1024, Element::Int8)},
               {surface(channels, 1024, Element::Int8), surface(channels, 1024, Element::Int8)},
               surface(channels, 1, Element::Float16), surface(channels, 1, Element::Float16),
               {}, surface(2048, 1, Element::Float16), {}};
    for (const uint32_t width : set.shape.down) set.down.push_back(surface(2048, width, Element::Int8));
    for (ane::Surface *s : {&set.gate[0], &set.gate[1], &set.up[0], &set.up[1]}) fill(*s, 3);
    for (ane::Surface &d : set.down) fill(d, 3);
    fillScales(set.gateScale);
    fillScales(set.upScale);
    fillScales(set.downScale);
    return set;
  };
  // Built after the set's surfaces are in their final storage — the inputs
  // hold their addresses.
  const auto ffnInputs = [&](FfnSet &set) {
    const uint32_t channels = set.shape.channels;
    set.inputs = {{"x0", &x[0], 1024, 0}, {"x1", &x[1], 1024, 0}, {"tx", &tx, 1, 0},
                  {"wg0", &set.gate[0], channels, 1024}, {"wu0", &set.up[0], channels, 1024},
                  {"wg1", &set.gate[1], channels, 1024}, {"wu1", &set.up[1], channels, 1024},
                  {"sg", &set.gateScale, channels, 1}, {"su", &set.upScale, channels, 1}};
    for (uint32_t i = 0; i < set.down.size(); ++i)
      set.inputs.push_back({"wd" + std::to_string(i), &set.down[i], 2048, set.shape.down[i]});
    set.inputs.push_back({"sd", &set.downScale, 2048, 1});
  };
  FfnSet expert = ffnSet(512, {512});
  FfnSet d1 = ffnSet(10752, {2560, 2560, 2560, 2560, 512});
  ffnInputs(expert);
  ffnInputs(d1);
  // fp16 inputs for the in-MIL norm variants (the int8 split's raw input).
  ane::Surface xf0 = surface(1024, kRowsMax, Element::Float16), xf1 = surface(1024, kRowsMax, Element::Float16);
  fillScales(xf0);
  fillScales(xf1);
  // The norm variant's inputs: same weights and scales, fp16 x, no tx.
  const auto normInputs = [&](const FfnSet &set) {
    const uint32_t channels = set.shape.channels;
    std::vector<Input> ins = {{"x0", &xf0, 1024, 0}, {"x1", &xf1, 1024, 0},
                              {"wg0", &set.gate[0], channels, 1024}, {"wu0", &set.up[0], channels, 1024},
                              {"wg1", &set.gate[1], channels, 1024}, {"wu1", &set.up[1], channels, 1024},
                              {"sg", &set.gateScale, channels, 1}, {"su", &set.upScale, channels, 1}};
    for (uint32_t i = 0; i < set.down.size(); ++i)
      ins.push_back({"wd" + std::to_string(i), &set.down[i], 2048, set.shape.down[i]});
    ins.push_back({"sd", &set.downScale, 2048, 1});
    return ins;
  };
  const std::vector<Input> expertNormInputs = normInputs(expert), d1NormInputs = normInputs(d1);

  // Bare-matmul proxy weights and outputs for the conv/attention shapes and
  // the smallest floor programs.
  const auto weightSurface = [&](uint32_t n, uint32_t inputs) {
    ane::Surface s = surface(n, inputs, Element::Int8);
    fill(s, 3);
    return s;
  };
  ane::Surface w512 = weightSurface(512, 2048), w2048 = weightSurface(2048, 2048),
               w3072 = weightSurface(3072, 2048), w6144 = weightSurface(6144, 2048);
  ane::Surface xwide = surface(2048, kRowsMax, Element::Int8);
  fill(xwide, 7);
  ane::Surface xfp = surface(2048, kRowsMax, Element::Float16);
  fillScales(xfp);
  const auto outOf = [&](uint32_t n) { return surface(n, kRowsMax, Element::Float16); };
  ane::Surface out512 = outOf(512), out2048 = outOf(2048), out3072 = outOf(3072), out6144 = outOf(6144);

  const std::vector<uint8_t> emptyBlob = ane::constantBlob(std::span<const _Float16>{});
  // A command needs a dispatch; one residency_kick thread satisfies it.
  const MetalBuffer kickTarget = zeros(backend, 4, "kick");
  const auto kick = [&](CommandGraph &graph) {
    graph.add("residency_kick", {kickTarget}, {1, 1, 1}, {1, 1, 1});
  };
  // ANE_DUMP_MIL=<dir> writes each generated program for inspection.
  const char *dumpDir = std::getenv("ANE_DUMP_MIL");
  auto makeProgram = [&](const std::string &name, const std::string &mil, const std::vector<uint8_t> &blob) {
    if (dumpDir) {
      const std::string path = std::string(dumpDir) + "/" + name + ".mil";
      FILE *file = std::fopen(path.c_str(), "w");
      if (file) {
        std::fwrite(mil.data(), 1, mil.size(), file);
        std::fclose(file);
      }
    }
    return std::make_unique<ane::Program>(mil, blob, ane::Program::Limits{kProgramLimit, {}});
  };

  // ---- Floor anatomy -----------------------------------------------------
  std::printf("%s, GPU family %u, %u cores: ANE floor + d1 forward A/B (%u rounds)\n",
              backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
              backend.capabilities().gpuCoreCount, rounds);

  // Tap outputs: the FFN output with `h` concat'd behind it, and a 512-wide
  // fp16 input for the second-eval copy at the expert's tap size.
  const ane::Surface tapOutExpert = surface(2048 + 1 + 512, kRowsMax, Element::Float16),
                     tapOutD1 = surface(2048 + 1 + 10752, kRowsMax, Element::Float16);
  ane::Surface xfp512 = surface(512, kRowsMax, Element::Float16);
  fillScales(xfp512);

  struct Floor {
    std::string name, fn;
    std::unique_ptr<ane::Program> program;
    std::vector<Input> inputs;
    const ane::Surface *output;
    std::optional<ane::Program::Binding> binding = std::nullopt;
    double ms = 0;
  };
  std::vector<Floor> floors;
  {
    // copy: x fp16 through, nothing else.
    std::string mil = "program(1.3)\n{\n    func copy<ios18>(" +
                      xfp.bufferType(2048, 256) + " x) {\n" +
                      "        " + tensor("fp16", 2048, 256) +
                      " x_t = tensor_buffer_to_tensor<ios17>(input = x);\n        " + tensor("fp16", 2048, 256) +
                      " y_t = mul(x = x_t, y = fp16(1));\n        " + out2048.bufferType(2048, 256) +
                      " y = tensor_to_tensor_buffer<ios17>(input = y_t, interleave_factors = "
                      "tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" + out2048.strides(2048) +
                      "));\n    } -> (y);\n}\n";
    floors.push_back({"copy (fp16 pass-through)", "copy", makeProgram("copy", mil, emptyBlob),
                      {{"x", &xfp, 2048, 0}}, &out2048});
  }
  {
    // mm_i8: one int8 matmul 512x2048, 2 inputs.
    const Input xi{"x", &xwide, 2048, 0}, wi{"w", &w512, 512, 2048};
    std::string mil = "program(1.3)\n{\n" +
                      matmulFunction("mm", xi, true, wi, nullptr, out512, 256) + "}\n";
    floors.push_back({"mm_i8 (x int8, w int8, 2 in)", "mm", makeProgram("mm_i8", mil, emptyBlob),
                      {xi, wi}, &out512});
  }
  {
    // mm_fp: same matmul, x already fp16.
    const Input xi{"x", &xfp, 2048, 0}, wi{"w", &w512, 512, 2048};
    std::string mil = "program(1.3)\n{\n" +
                      matmulFunction("mm", xi, false, wi, nullptr, out512, 256) + "}\n";
    floors.push_back({"mm_fp (x fp16, w int8, 2 in)", "mm", makeProgram("mm_fp", mil, emptyBlob),
                      {xi, wi}, &out512});
  }
  {
    // gate: the gate half of an expert — x0,x1,tx,wg0,wg1,sg (6 inputs).
    const std::vector<Input> ins{{"x0", &x[0], 1024, 0}, {"x1", &x[1], 1024, 0}, {"tx", &tx, 1, 0},
                                 {"wg0", &expert.gate[0], 512, 1024}, {"wg1", &expert.gate[1], 512, 1024},
                                 {"sg", &expert.gateScale, 512, 1}};
    const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
    std::string parameters, body;
    const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
    for (const Input &input : ins) {
      const uint32_t width = input.width ? input.width : 256;
      parameters += (parameters.empty() ? "" : ", ") + input.surface->bufferType(input.rows, width) + " " + input.name;
      line(tensor(milType(input), input.rows, width) + " " + input.name +
           "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
    }
    for (uint32_t k = 0; k < 2; ++k) {
      line(tensor("fp16", 1024, 256) + " x" + std::to_string(k) + "_d = dequantize(input = x" +
           std::to_string(k) + "_t, scale = " + unit + ")");
      line(tensor("fp16", 512, 1024) + " wg" + std::to_string(k) + "_d = dequantize(input = wg" +
           std::to_string(k) + "_t, scale = " + unit + ")");
      line(tensor("fp16", 512, 256) + " gm" + std::to_string(k) +
           " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = wg" + std::to_string(k) +
           "_d, y = x" + std::to_string(k) + "_d)");
    }
    line(tensor("fp16", 512, 256) + " gm_sum1 = add(x = gm0, y = gm1)");
    line(tensor("fp16", 512, 256) + " gs = mul(x = gm_sum1, y = sg_t)");
    line(tensor("fp16", 512, 256) + " gt = mul(x = gs, y = tx_t)");
    line(out512.bufferType(512, 256) +
         " y = tensor_to_tensor_buffer<ios17>(input = gt, interleave_factors = "
         "tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" + out512.strides(512) + "))");
    floors.push_back({"gate (2 matmul + 2 scale, 6 in)", "gate",
                      makeProgram("gate", "program(1.3)\n{\n    func gate<ios18>(" + parameters + ") {\n" +
                                                         body + "    } -> (y);\n}\n",
                                                     emptyBlob),
                      ins, &out512});
  }
  {
    // The expert's whole W8A8 SwiGLU (11 inputs) and d1's FFN (15 inputs).
    const Shape eshape{2048, 1024, 512, {512}};
    floors.push_back({"expert SwiGLU (512 ch, 11 in)", "ffn256",
                      makeProgram("expert", "program(1.3)\n{\n" + function(eshape, expert.inputs, partial, 256) +
                                                         "}\n",
                                                     rotationBlob(512, signs)),
                      expert.inputs, &partial});
    floors.push_back({"d1 FFN (10752 ch, 15 in)", "ffn256",
                      makeProgram("d1ffn", "program(1.3)\n{\n" + function(d1.shape, d1.inputs, partial, 256) +
                                                         "}\n",
                                                     rotationBlob(10752, signs)),
                      d1.inputs, &partial});
    // The in-MIL norm variants: fp16 unnormalized x in, RMSNorm + rotation +
    // quantize inside the program, tx computed in MIL (14 inputs).
    const Shape eshape2{2048, 1024, 512, {512}};
    const auto expertNormBlob =
        constantBlobs({rotationValues(eshape2.channels, signs), rotationValues(eshape2.inputSegment, signs)});
    floors.push_back({"expert SwiGLU+MILnorm (fp16 x)", "ffn256n",
                      makeProgram("expertnorm",
                                  "program(1.3)\n{\n" +
                                      functionNorm(eshape2, expertNormInputs, partial, 256,
                                                   expertNormBlob.second[1]) +
                                      "}\n",
                                  expertNormBlob.first),
                      expertNormInputs, &partial});
    const auto d1NormBlob =
        constantBlobs({rotationValues(d1.shape.channels, signs), rotationValues(d1.shape.inputSegment, signs)});
    floors.push_back({"d1 FFN+MILnorm (fp16 x)", "ffn256n",
                      makeProgram("d1norm",
                                  "program(1.3)\n{\n" +
                                      functionNorm(d1.shape, d1NormInputs, partial, 256, d1NormBlob.second[1]) +
                                      "}\n",
                                  d1NormBlob.first),
                      d1NormInputs, &partial});
    // Forward taps: h concat'd into the FFN output vs a second eval that
    // copies the same bytes. copy512 moves exactly the expert's tap.
    const Shape eshape3{2048, 1024, 512, {512}};
    floors.push_back({"expert SwiGLU + tap h", "ffn256",
                      makeProgram("experttap",
                                  "program(1.3)\n{\n" +
                                      function(eshape3, expert.inputs, tapOutExpert, 256, "h", 512) + "}\n",
                                  rotationBlob(512, signs)),
                      expert.inputs, &tapOutExpert});
    floors.push_back({"d1 FFN + tap h", "ffn256",
                      makeProgram("d1tap",
                                  "program(1.3)\n{\n" +
                                      function(d1.shape, d1.inputs, tapOutD1, 256, "h", 10752) + "}\n",
                                  rotationBlob(10752, signs)),
                      d1.inputs, &tapOutD1});
    std::string copyMil = "program(1.3)\n{\n    func copy<ios18>(" +
                          xfp512.bufferType(512, 256) + " x) {\n" +
                          "        " + tensor("fp16", 512, 256) +
                          " x_t = tensor_buffer_to_tensor<ios17>(input = x);\n        " + tensor("fp16", 512, 256) +
                          " y_t = mul(x = x_t, y = fp16(1));\n        " + out512.bufferType(512, 256) +
                          " y = tensor_to_tensor_buffer<ios17>(input = y_t, interleave_factors = "
                          "tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" + out512.strides(512) +
                          "));\n    } -> (y);\n}\n";
    floors.push_back({"copy512 (fp16 pass-through)", "copy", makeProgram("copy512", copyMil, emptyBlob),
                      {{"x", &xfp512, 512, 0}}, &out512});
  }

  std::printf("\nfloor anatomy (K=32 chained, rows 256, medians of %u rounds):\n", rounds);
  std::printf("%-34s %10s %10s\n", "program", "us/eval", "evals/s");
  for (Floor &floor : floors) {
    const uint32_t procedure = floor.program->procedure(floor.fn);
    std::vector<ane::Surface> surfaces;
    for (const std::string &name : floor.program->inputs(procedure)) {
      const auto input = std::ranges::find(floor.inputs, name, &Input::name);
      if (input == floor.inputs.end()) throw std::logic_error("unbound input " + name);
      surfaces.push_back(*input->surface);
    }
    const ane::Program::Binding binding = floor.program->bind(procedure, surfaces, *floor.output);
    std::vector<double> evals;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double per = chain(backend, handoff, {floor.program.get(), binding}, 32, kick);
      if (i) evals.push_back(per);
    }
    std::sort(evals.begin(), evals.end());
    floor.ms = evals[evals.size() / 2];
    std::printf("%-34s %10.1f %10.0f\n", floor.name.c_str(), floor.ms * 1000.0, 1000.0 / floor.ms);
    floor.binding = std::move(binding);
  }

  // ---- Dispatch variants ------------------------------------------------
  // The floor outside chaining: synchronous evaluations with no shared
  // events on each entry point the interface has — the client's async path,
  // its real-time variant, and the model's own synchronous evaluate — plus
  // the driver's hardware report when a perfStatsMask populates it.
  std::printf("\ndispatch variants (K=16, rows 256, medians of %u rounds):\n", rounds);
  const ane::Program::EvalInterface iface = floors.front().program->evalInterface();
  std::printf("interface:%s\n", iface.detail.c_str());
  uint64_t statsMask = 0;
  if (iface.perfStats && iface.perfStatsMask) {
    for (const uint64_t candidate : {1ULL, 0x3fULL, 0xffULL, 0xffffULL, 0xffffffffULL, ~0ULL}) {
      try {
        if (floors.front().program
                ->evaluate(*floors.front().binding, ane::Program::EvalPath::Client, candidate)
                .hardwareNs) {
          statsMask = candidate;
          break;
        }
      } catch (const std::exception &error) {
        std::printf("  mask 0x%llx rejected: %s\n", static_cast<unsigned long long>(candidate), error.what());
      }
    }
  }
  {
    char maskText[32];
    if (statsMask)
      std::snprintf(maskText, sizeof maskText, "0x%llx", static_cast<unsigned long long>(statsMask));
    else
      std::snprintf(maskText, sizeof maskText, "%s", iface.perfStatsMask ? "no mask populated stats" : "n/a");
    std::printf("perfStatsMask: %s\n", maskText);
  }
  std::vector<std::pair<const char *, ane::Program::EvalPath>> paths{{"unchained", ane::Program::EvalPath::Client}};
  if (iface.clientRealtime) paths.push_back({"client-rt", ane::Program::EvalPath::ClientRealtime});
  if (iface.modelEval) paths.push_back({"model", ane::Program::EvalPath::Model});
  std::printf("%-34s %-10s %9s %9s %9s %6s\n", "program", "path", "wall_us", "call_us", "hw_us", "hw/ane");
  for (Floor &floor : floors) {
    for (const auto &[pathName, path] : paths) {
      std::vector<double> walls, calls, hws;
      bool reported = true;
      try {
        for (uint32_t i = 0; i < rounds + 1; ++i) {
          const ane::Program::DirectTiming t = floor.program->evaluate(*floor.binding, path, statsMask);
          if (i) {
            walls.push_back(t.wallMs);
            calls.push_back(t.callMs);
            if (t.hardwareNs) hws.push_back(*t.hardwareNs / 1e3);
            reported = t.reported;
          }
        }
      } catch (const std::exception &error) {
        std::printf("%-34s %-10s  %s\n", floor.name.c_str(), pathName, error.what());
        continue;
      }
      const auto median = [](std::vector<double> &v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      const double wall = median(walls) * 1000.0, call = median(calls) * 1000.0;
      if (hws.empty())
        std::printf("%-34s %-10s %9.1f %9.1f %9s %6s\n", floor.name.c_str(), pathName, wall, call, "-",
                    reported ? "" : "unrep");
      else
        std::printf("%-34s %-10s %9.1f %9.1f %9.1f %5.0f%%\n", floor.name.c_str(), pathName, wall, call,
                    median(hws), 100.0 * median(hws) / wall);
    }
  }

  // ---- Forward taps ------------------------------------------------------
  // A tap concat'd into the FFN's own output vs a second eval moving the
  // same bytes: the pair chains [FFN; copy] per iteration, the tap variant
  // carries h in its output rows.
  std::printf("\nforward taps (K=32 chained pairs, rows 256, medians of %u rounds):\n", rounds);
  const auto floorOf = [&](const std::string &name) -> Floor & {
    const auto it = std::ranges::find_if(floors, [&](const Floor &f) { return f.name == name; });
    if (it == floors.end()) throw std::logic_error("missing floor " + name);
    return *it;
  };
  Floor &copy512 = floorOf("copy512 (fp16 pass-through)");
  for (const auto &[baseName, tapName] :
       {std::pair{"expert SwiGLU (512 ch, 11 in)", "expert SwiGLU + tap h"},
        std::pair{"d1 FFN (10752 ch, 15 in)", "d1 FFN + tap h"}}) {
    Floor &base = floorOf(baseName), &tap = floorOf(tapName);
    const Probe pair[]{{base.program.get(), *base.binding}, {copy512.program.get(), *copy512.binding}};
    std::vector<double> pairs;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double per = chainN(backend, handoff, pair, 32, kick);
      if (i) pairs.push_back(per);
    }
    std::sort(pairs.begin(), pairs.end());
    const double pairMs = pairs[pairs.size() / 2];
    std::printf("%-30s base %7.1f  +tap %7.1f (cost %+6.1f)  pair %7.1f (2nd eval %+6.1f)  tap saves %5.1f us\n",
                baseName, base.ms * 1000.0, tap.ms * 1000.0, (tap.ms - base.ms) * 1000.0, pairMs * 1000.0,
                (pairMs - base.ms) * 1000.0, (pairMs - tap.ms) * 1000.0);
  }

  // ---- d1 forward model --------------------------------------------------
  // The d1-3B layer program at each prompt size: FFN eval (the dominant
  // term, measured), conv/attention projections as bare matmul evals, GPU
  // baselines through the real kernels.
  std::printf("\nd1-3B forward model (LFM2: 30 layers — 30 FFN, 22 conv, 8 attention):\n");
  const std::vector<uint32_t> sizes{64, 128, 256, 512, 1024, 2048};

  // Programs for the d1 FFN at every size and the proxy matmuls.
  std::string d1mil = "program(1.3)\n{\n";
  for (const uint32_t rows : sizes) d1mil += function(d1.shape, d1.inputs, partial, rows);
  d1mil += "}\n";
  std::unique_ptr<ane::Program> d1ProgramPtr = makeProgram("d1program", d1mil, rotationBlob(10752, signs));
  ane::Program &d1Program = *d1ProgramPtr;
  const auto d1Bind = [&](uint32_t rows) {
    const uint32_t procedure = d1Program.procedure(functionName(rows));
    std::vector<ane::Surface> surfaces;
    for (const std::string &name : d1Program.inputs(procedure)) {
      const auto input = std::ranges::find(d1.inputs, name, &Input::name);
      surfaces.push_back(*input->surface);
    }
    return d1Program.bind(procedure, surfaces, partial);
  };
  // Eval a binding K times and return median ms/eval over `rounds`.
  const auto evalMs = [&](ane::Program &program, const ane::Program::Binding &binding) {
    std::vector<double> evals;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double per = chain(backend, handoff, {&program, binding}, 16, kick);
      if (i) evals.push_back(per);
    }
    std::sort(evals.begin(), evals.end());
    return evals[evals.size() / 2];
  };

  // Proxy programs for conv (6144), QKV (3072) and O (2048): one program per
  // weight, one function per rows.
  const std::vector<std::pair<ane::Surface *, uint32_t>> proxyWeights{{&w6144, 6144}, {&w3072, 3072}, {&w2048, 2048}};
  std::vector<ane::Surface *> proxyOuts{&out6144, &out3072, &out2048};
  std::vector<std::unique_ptr<ane::Program>> proxyPrograms;
  proxyPrograms.reserve(3);
  for (size_t i = 0; i < proxyWeights.size(); ++i) {
    auto &[w, n] = proxyWeights[i];
    std::string mil = "program(1.3)\n{\n";
    for (const uint32_t rows : sizes)
      mil += matmulFunction(("mm" + std::to_string(rows)).c_str(), Input{"x", &xfp, 2048, 0}, false,
                            Input{"w", w, n, 2048}, nullptr, *proxyOuts[i], rows);
    mil += "}\n";
    proxyPrograms.push_back(makeProgram("proxy" + std::to_string(proxyWeights[i].second), mil, emptyBlob));
  }
  const auto proxyBind = [&](ane::Program &program, ane::Surface &w, ane::Surface &out, uint32_t rows) {
    const uint32_t procedure = program.procedure("mm" + std::to_string(rows));
    std::vector<ane::Surface> surfaces;
    for (const std::string &name : program.inputs(procedure))
      surfaces.push_back(name == "x" ? xfp : w);
    return program.bind(procedure, surfaces, out);
  };

  // GPU baselines: the real prefill SwiGLU at the d1 FFN shape, and bare
  // projections at the conv/attention shapes.
  const auto bf16Rows = [&](uint32_t width) { return buffer(backend, uint64_t{kRowsMax} * width * 2, "rows"); };
  const auto sumRows = [&](uint32_t width) { return buffer(backend, uint64_t{kRowsMax} * (width / 64) * 4, "sums"); };
  PrefillFfnBuffers ffn{bf16Rows(2048), sumRows(2048), bf16Rows(10752), bf16Rows(10752), sumRows(10752), {}};
  const MetalBuffer residual = zeros(backend, uint64_t{kRowsMax} * 2048 * 2, "residual"),
                    gpuOut = buffer(backend, uint64_t{kRowsMax} * 2048 * 2, "out"),
                    projOut = buffer(backend, uint64_t{kRowsMax} * 6144 * 2, "proj out");
  {
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    std::mt19937 random(20261026);
    auto *values = static_cast<uint16_t *>(ffn.normalized.contents());
    for (uint64_t i = 0; i < uint64_t{kRowsMax} * 2048; ++i)
      values[i] = static_cast<uint16_t>(std::bit_cast<uint32_t>(uniform(random)) >> 16);
  }
  Projection gate = test::deterministicQ4Projection(backend, {10752, 2048}, 11),
             up = test::deterministicQ4Projection(backend, {10752, 2048}, 12),
             down = test::deterministicQ4Projection(backend, {2048, 10752}, 13),
             conv = test::deterministicQ4Projection(backend, {6144, 2048}, 14),
             qkv = test::deterministicQ4Projection(backend, {3072, 2048}, 15),
             proj = test::deterministicQ4Projection(backend, {2048, 2048}, 16);
  const auto gpuMs = [&](CommandGraph &graph) {
    const auto start = AwakeClock::now();
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    return std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count();
  };
  const auto gpuMedian = [&](const auto &encode) {
    std::vector<double> times;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      CommandGraph graph;
      encode(graph);
      const double ms = gpuMs(graph);
      if (i) times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
  };

  std::printf("%8s %9s %9s %9s %10s %10s %9s %9s %10s %9s\n", "rows", "ffn_ane", "conv_ane", "attn_ane",
              "fwd_ane", "ffn_gpu", "conv_gpu", "attn_gpu", "fwd_gpu", "ane/gpu");
  for (const uint32_t rows : sizes) {
    const double ffnAne = evalMs(d1Program, d1Bind(rows));
    const double convAne = evalMs(*proxyPrograms[0], proxyBind(*proxyPrograms[0], w6144, *proxyOuts[0], rows));
    const double qkvAne = evalMs(*proxyPrograms[1], proxyBind(*proxyPrograms[1], w3072, *proxyOuts[1], rows));
    const double oAne = evalMs(*proxyPrograms[2], proxyBind(*proxyPrograms[2], w2048, *proxyOuts[2], rows));
    const double attnAne = qkvAne + oAne; // plus T^2 attention, in the model below

    const double ffnGpu = gpuMedian([&](CommandGraph &graph) {
      linear.addPrefillSums(graph, ffn.normalized, ffn.sums, gate, rows);
      linear.addPrefillSwiGlu(graph, {&gate, &up, &down}, ffn, residual, gpuOut, rows);
    });
    const double convGpu = gpuMedian([&](CommandGraph &graph) {
      linear.addPrefillSums(graph, ffn.normalized, ffn.sums, conv, rows);
      linear.addPrefill(graph, ffn.normalized, conv, projOut, ffn.sums, rows);
    });
    const double qkvGpu = gpuMedian([&](CommandGraph &graph) {
      linear.addPrefillSums(graph, ffn.normalized, ffn.sums, qkv, rows);
      linear.addPrefill(graph, ffn.normalized, qkv, projOut, ffn.sums, rows);
    });
    const double oGpu = gpuMedian([&](CommandGraph &graph) {
      linear.addPrefillSums(graph, ffn.normalized, ffn.sums, proj, rows);
      linear.addPrefill(graph, ffn.normalized, proj, projOut, ffn.sums, rows);
    });
    // The T^2 attention term (32 heads x 64, two matmuls): ~16 GFLOP at
    // 2048 rows each way — take the GPU matmul time at the same shape as
    // the per-TFLOP rate both sides share.
    const double t2 = double(rows) * rows * 64.0 * 32 * 2 * 2 / 1e9; // GFLOP
    const double attnAneT2 = t2 * (qkvAne + oAne) / ((3072.0 + 2048.0) * 2048.0 * rows / 1e9);
    const double attnGpuT2 = t2 * (qkvGpu + oGpu) / ((3072.0 + 2048.0) * 2048.0 * rows / 1e9);

    const double fwdAne = 30.0 * ffnAne + 22.0 * convAne + 8.0 * (attnAne + attnAneT2);
    const double fwdGpu = 30.0 * ffnGpu + 22.0 * convGpu + 8.0 * (qkvGpu + oGpu + attnGpuT2);
    std::printf("%8u %9.3f %9.3f %9.3f %10.1f %10.3f %9.3f %9.3f %10.1f %8.2fx\n", rows, ffnAne, convAne,
                attnAne, fwdAne, ffnGpu, convGpu, qkvGpu + oGpu, fwdGpu, fwdAne / fwdGpu);
  }
  std::printf("\nffn_ane/conv_ane/attn_ane are ms per eval (16-chain median); fwd_* project the 30-layer\n");
  std::printf("forward: 30 FFN + 22 conv-projection + 8 attention (QKV+O+T^2 estimate). Embedding/head\n");
  std::printf("excluded — identical both sides. conv covers the in_proj matmul only (taps-3 is trivial).\n");
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: ane-forward-benchmark <metallib> [rounds]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      return run(backend, argc > 2 ? std::stoul(argv[2]) : 10);
    } catch (const std::exception &error) {
      std::cerr << "ane-forward-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}

// trellis <richengine.metallib>: Metal trellis kernels against the host
// reference decoder for TRL2-TRL6, transforms, gathers and embedding rows.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Trellis.h"
#include "model/GgufTrellis.hpp"
#include "model/TrellisCodec.hpp"
#include "model/WeightSource.hpp"
#include "ops/Linear.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

using richengine::metal::CommandGraph;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;
using richengine::model::TrellisDescriptor;
using richengine::test::sharedBuffer;
namespace trl = richengine::model::trellis;

namespace {

float bf16ToFloat(uint16_t bits) {
  const uint32_t word = uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}
uint16_t floatToBf16(float value) {
  uint32_t word;
  std::memcpy(&word, &value, 4);
  return uint16_t((word + 0x7fff + ((word >> 16) & 1)) >> 16);
}
uint16_t floatToHalf(float value) {
  const __fp16 half = static_cast<__fp16>(value);
  uint16_t bits;
  std::memcpy(&bits, &half, 2);
  return bits;
}

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = sharedBuffer(backend, values.size() * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}

bool closeEnough(float got, float expected) {
  return std::fabs(got - expected) <= 0.005f + 0.005f * std::fabs(expected);
}

struct Fixture {
  const uint32_t columns, rows, inputBlock, outputBlock;
  std::vector<uint32_t> codes;
  std::vector<uint16_t> scales, inputBits;
  std::vector<float> pre, post, input;
  std::vector<int32_t> pin, pout, scatter;
  TrellisDescriptor descriptor;

  Fixture(uint32_t bits, uint32_t columnCount = 64, uint32_t rowCount = 64,
          uint32_t inputTransform = 32, uint32_t outputTransform = 16)
      : columns(columnCount), rows(rowCount), inputBlock(inputTransform), outputBlock(outputTransform),
        descriptor{1, static_cast<int32_t>(outputTransform), 1, 1, static_cast<int32_t>(inputTransform), 1, 1,
                   static_cast<int32_t>(rows)} {
    std::mt19937 rng(100 + bits + columns * 3 + rows);
    const uint32_t words = rows / 16 * columns / 16 * 8 * bits;
    codes.resize(words);
    for (uint32_t &word : codes) word = rng();
    scales.resize(columns / 16);
    for (uint16_t &scale : scales) scale = floatToHalf(0.75f);
    pre.resize(columns);
    post.resize(rows);
    input.resize(columns);
    inputBits.resize(columns);
    pin.resize(columns);
    pout.resize(rows);
    scatter.resize(rows);
    for (uint32_t i = 0; i < columns; ++i) {
      pre[i] = 0.5f + float(i % 5) * 0.1f;
      pin[i] = static_cast<int32_t>((i * 7 + 3) % columns);
      input[i] = (float(int32_t(rng() % 17)) - 8.0f) / 32.0f;
      inputBits[i] = floatToBf16(input[i]);
      input[i] = bf16ToFloat(inputBits[i]);
    }
    for (uint32_t i = 0; i < rows; ++i) {
      post[i] = 0.6f + float(i % 7) * 0.07f;
      pout[i] = static_cast<int32_t>((i * 5 + 1) % rows);
      scatter[i] = static_cast<int32_t>(rows - 1 - i);
    }
    // The tests need permutations, not merely congruence classes.
    for (uint32_t i = columns; i-- > 1;) std::swap(pin[i], pin[rng() % (i + 1)]);
    for (uint32_t i = rows; i-- > 1;) std::swap(pout[i], pout[rng() % (i + 1)]);
  }
};

int compareRows(const char *what, const uint16_t *got, const std::vector<std::vector<float>> &expected) {
  uint32_t wrong = 0, total = 0;
  for (uint32_t row = 0; row < expected.size(); ++row)
    for (uint32_t column = 0; column < expected[row].size(); ++column, ++total)
      if (!closeEnough(bf16ToFloat(got[row * expected[row].size() + column]), expected[row][column])) {
        if (wrong < 4)
          std::cout << "  [" << row << "," << column << "] got " << bf16ToFloat(got[row * expected[row].size() + column])
                    << " expected " << expected[row][column] << '\n';
        ++wrong;
      }
  std::cout << what << ": " << wrong << " of " << total << " wrong " << (wrong ? "FAIL" : "ok") << '\n';
  return wrong != 0;
}

int testProduct(MetalBackend &backend, uint32_t bits, uint32_t activationRows, int matrix,
                uint32_t columns = 64, uint32_t rows = 64, uint32_t inputBlock = 32, uint32_t outputBlock = 16,
                bool mv = false, bool anchors = false) {
  Fixture f(bits, columns, rows, inputBlock, outputBlock);
  std::vector<std::vector<float>> inputs(activationRows, f.input), expected(activationRows);
  for (uint32_t row = 0; row < activationRows; ++row)
    expected[row] = trl::linearReference(f.codes.data(), bits, f.columns, f.rows, f.scales.data(), f.pre.data(),
                                         f.post.data(), f.pin.data(), f.pout.data(), f.descriptor, inputs[row].data());
  std::vector<uint16_t> inputBits(f.columns * activationRows);
  for (uint32_t row = 0; row < activationRows; ++row)
    for (uint32_t j = 0; j < f.columns; ++j) inputBits[row * f.columns + j] = floatToBf16(inputs[row][j]);

  MetalBuffer input = upload(backend, inputBits), pre = upload(backend, f.pre), pin = upload(backend, f.pin);
  MetalBuffer x = sharedBuffer(backend, f.columns * ((activationRows + 15) / 16 * 16) * 4);
  MetalBuffer y = sharedBuffer(backend, f.rows * activationRows * 4);
  MetalBuffer output = sharedBuffer(backend, f.rows * activationRows * 2);
  MetalBuffer codes = upload(backend, f.codes), scales = upload(backend, f.scales), post = upload(backend, f.post);
  MetalBuffer pout = upload(backend, f.pout), scatter = upload(backend, f.scatter);
  TrellisParams params{f.columns, f.rows, activationRows, f.rows, 0, 32, 16, 3};
  CommandGraph graph;
  graph.add("trellis_input", {input, pre, pin, x}, params, {f.columns, activationRows, 1}, {256, 1, 1});
  const float inputScale = 1.0f / std::sqrt(float(f.inputBlock));
  graph.add("trellis_fwht", {x}, TrellisTransformParams{f.columns, f.inputBlock, inputScale, 0}, {f.columns / f.inputBlock, activationRows, 1}, {256, 1, 1});
  if (mv) {
    const uint32_t mvColumns = activationRows == 1 ? 1 : activationRows == 2 ? 2 : 4;
    graph.add("trellis_product_mv_" + std::to_string(bits) + "_" + std::to_string(mvColumns), {codes, scales, x, y}, params,
              {f.rows / 16, 1, 1}, {32, 4, 1});
  } else if (matrix == 2) {
    graph.add("trellis_zero", {y}, params, {(activationRows * f.rows + 255) / 256, 1, 1}, {256, 1, 1});
    graph.add((anchors ? "trellis_product_mm8_anchors_" : "trellis_product_mm8_") + std::to_string(bits), {codes, scales, x, y}, params, {(f.rows + 31) / 32, (activationRows + 7) / 8, 4}, {128, 1, 1});
  } else if (matrix == 3) {
    graph.add("trellis_product_mm16x32_" + std::to_string(bits) + "_16", {codes, scales, x, y}, params, {(f.rows + 31) / 32, (activationRows + 15) / 16, 1}, {128, 1, 1});
  } else if (matrix == 4) {
    graph.add("trellis_product_mm16x32_" + std::to_string(bits) + "_32", {codes, scales, x, y}, params, {(f.rows + 31) / 32, (activationRows + 31) / 32, 1}, {128, 1, 1});

  } else if (matrix) {
    graph.add("trellis_product_mm_" + std::to_string(bits), {codes, scales, x, y}, params, {f.rows / 16, 1, 1}, {128, 1, 1});
  } else {
    const uint32_t columns = activationRows == 1 ? 1 : 4;
    graph.add("trellis_product_" + std::to_string(bits) + "_" + std::to_string(columns), {codes, scales, x, y}, params,
              {f.rows / 4, (activationRows + columns - 1) / columns, 1}, {128, 1, 1});
  }
  const float outputScale = 1.0f / std::sqrt(float(f.outputBlock));
  graph.add("trellis_fwht", {y}, TrellisTransformParams{f.rows, f.outputBlock, outputScale, 0}, {f.rows / f.outputBlock, activationRows, 1}, {256, 1, 1});
  graph.add("trellis_output_bf16", {y, post, pout, scatter, output, output}, params, {f.rows, activationRows, 1}, {256, 1, 1});
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  return compareRows(mv ? "trellis_product_mv" : matrix == 2 ? "trellis_product_mm8" : (matrix == 3 || matrix == 4) ? "trellis_product_mm16x32" : matrix ? "trellis_product_mm" : "trellis_product",
                     static_cast<const uint16_t *>(output.contents()), expected);
}


int testLinearIntegration(MetalBackend &backend, uint32_t bits) {
  Fixture f(bits);
  constexpr uint32_t activationRows = 4;
  std::vector<std::vector<float>> inputs(activationRows, f.input), expected(activationRows);
  for (uint32_t row = 0; row < activationRows; ++row)
    expected[row] = trl::linearReference(f.codes.data(), bits, f.columns, f.rows, f.scales.data(), f.pre.data(),
                                         f.post.data(), f.pin.data(), f.pout.data(), f.descriptor, inputs[row].data());

  richengine::ops::TrellisWeight weight;
  weight.bits = bits;
  weight.inputBlock = f.inputBlock;
  weight.outputBlock = f.outputBlock;
  weight.hasPin = weight.hasPout = true;
  weight.codes = upload(backend, f.codes);
  weight.scales = upload(backend, f.scales);
  weight.pre = upload(backend, f.pre);
  weight.post = upload(backend, f.post);
  weight.pin = upload(backend, f.pin);
  weight.pout = upload(backend, f.pout);
  weight.scatter = upload(backend, f.scatter);
  weight.hasScatter = true;
  richengine::ops::Projection projection(
      f.rows, f.columns,
      richengine::ops::BlockWeights{{richengine::ops::QuantizedSegment::trellis(std::move(weight), f.rows, f.columns)}});
  richengine::ops::Linear linear(backend.capabilities());
  const richengine::ops::LinearPlan plan =
      linear.prefillPlan(projection, activationRows, richengine::ops::LinearEpilogue::None);
  const uint32_t storageRows = plan.storageRows();
  std::vector<uint16_t> inputBits(storageRows * f.columns);
  for (uint32_t row = 0; row < activationRows; ++row)
    for (uint32_t j = 0; j < f.columns; ++j) inputBits[row * f.columns + j] = floatToBf16(inputs[row][j]);
  MetalBuffer input = upload(backend, inputBits);
  MetalBuffer output = sharedBuffer(backend, uint64_t{storageRows} * f.rows * 2);
  const richengine::ops::LinearScratchSize sizes = plan.scratchSize();
  richengine::ops::LinearScratch scratch;
  scratch.input = sharedBuffer(backend, std::max<uint64_t>(sizes.input, uint64_t{storageRows} * f.columns * 4));
  scratch.sums = sharedBuffer(backend, std::max<uint64_t>(sizes.sums, 1));
  scratch.partials = sharedBuffer(backend, std::max<uint64_t>(sizes.partials, uint64_t{storageRows} * f.rows * 4));
  scratch.counters = sharedBuffer(backend, std::max<uint64_t>(sizes.counters, 1));

  CommandGraph graph;
  linear.add(graph, {.input = input, .output = output, .scratch = scratch}, projection, plan);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  std::vector<std::vector<float>> padded(storageRows, std::vector<float>(f.rows));
  for (uint32_t row = 0; row < activationRows; ++row)
    for (uint32_t column = 0; column < f.rows; ++column)
      padded[row][static_cast<uint32_t>(f.scatter[column])] = expected[row][column];
  return compareRows("trellis Linear integration", static_cast<const uint16_t *>(output.contents()), padded);
}

template <class T>
std::vector<T> readTensorData(const richengine::model::WeightSource &source, const richengine::model::GgufTensor &tensor) {
  std::vector<T> values(tensor.bytes / sizeof(T));
  if (tensor.bytes % sizeof(T)) throw std::runtime_error("unaligned tensor payload: " + tensor.name);
  source.readData(tensor.offset, std::span<uint8_t>(reinterpret_cast<uint8_t *>(values.data()), tensor.bytes));
  return values;
}

float realReferenceRow(const std::vector<uint32_t> &codes, uint32_t bits, uint32_t columns, uint32_t rows,
                       const std::vector<uint16_t> &scales, const std::vector<float> &post,
                       const std::vector<int32_t> &pout, const richengine::model::TrellisDescriptor &descriptor,
                       const std::vector<float> &transformedInput, uint32_t outputRow) {
  const uint32_t source = pout.empty() ? outputRow : static_cast<uint32_t>(pout[outputRow]);
  const uint32_t block = descriptor.rowsKind == 1 ? static_cast<uint32_t>(descriptor.rowsBlock) : 1;
  const uint32_t first = block > 1 ? source / block * block : source / 16 * 16;
  const uint32_t count = block > 1 ? block : 16;
  if (first + count > rows) throw std::runtime_error("real reference row crosses the tensor");
  std::vector<float> weights(size_t{count} * columns), coded(count, 0.0f);
  trl::decodeRows(codes.data(), bits, columns, first, count, scales.data(), weights.data());
  for (uint32_t row = 0; row < count; ++row) {
    float sum = 0.0f;
    for (uint32_t column = 0; column < columns; ++column)
      sum += weights[size_t{row} * columns + column] * transformedInput[column];
    coded[row] = sum;
  }
  if (descriptor.rowsKind == 1) trl::blockTransform(coded.data(), count, block);
  return coded[source - first] * post[source];
}

int testRealTensor(MetalBackend &backend, const std::filesystem::path &path, const std::string &name) {
  richengine::model::WeightSource source(path);
  const richengine::model::GgufFile file(source);
  const richengine::model::TrellisTensorInfo info = richengine::model::readTrellisTensor(file, name);
  const uint32_t columns = info.inputSize(), rows = info.outputSize(), bits = info.bits();
  std::vector<uint32_t> codes = readTensorData<uint32_t>(source, *info.weight);
  std::vector<uint16_t> scales = readTensorData<uint16_t>(source, *info.s16);
  std::vector<float> pre = readTensorData<float>(source, *info.pre);
  std::vector<float> post = readTensorData<float>(source, *info.post);
  std::vector<int32_t> pin, pout;
  if (info.pin) pin = readTensorData<int32_t>(source, *info.pin);
  if (info.pout) pout = readTensorData<int32_t>(source, *info.pout);

  std::mt19937 rng(1701);
  std::vector<float> input(columns);
  for (float &value : input) value = bf16ToFloat(floatToBf16((float(int32_t(rng() % 1024)) - 512.0f) / 256.0f));
  std::vector<float> transformed(columns);
  for (uint32_t column = 0; column < columns; ++column)
    transformed[column] = input[pin.empty() ? column : static_cast<uint32_t>(pin[column])] * pre[column];
  trl::applySide(transformed.data(), columns, info.descriptor.colsKind, info.descriptor.colsBlock);

  MetalBuffer x = sharedBuffer(backend, columns * sizeof(float));
  MetalBuffer y = sharedBuffer(backend, rows * sizeof(float));
  MetalBuffer output = sharedBuffer(backend, rows * sizeof(uint16_t));
  CommandGraph graph;
  const uint32_t flags = (info.pin ? 1u : 0u) | (info.pout ? 2u : 0u);
  const TrellisParams params{columns, rows, 1, rows, 0,
                             info.descriptor.colsKind == 1 ? uint32_t(info.descriptor.colsBlock) : 1,
                             info.descriptor.rowsKind == 1 ? uint32_t(info.descriptor.rowsBlock) : 1, flags};
  std::vector<uint16_t> inputBits(columns);
  for (uint32_t column = 0; column < columns; ++column) inputBits[column] = floatToBf16(input[column]);
  MetalBuffer inputBuffer = upload(backend, inputBits);
  MetalBuffer preBuffer = upload(backend, pre), postBuffer = upload(backend, post);
  MetalBuffer pinBuffer = info.pin ? upload(backend, pin) : preBuffer;
  MetalBuffer poutBuffer = info.pout ? upload(backend, pout) : postBuffer;
  graph.add("trellis_input", {inputBuffer, preBuffer, pinBuffer, x}, params, {columns, 1, 1}, {256, 1, 1});
  if (params.input_block > 1)
    graph.add("trellis_fwht", {x}, TrellisTransformParams{columns, params.input_block, 1.0f / std::sqrt(float(params.input_block)), 0},
              {columns / params.input_block, 1, 1}, {256, 1, 1});
  graph.add("trellis_product_" + std::to_string(bits) + "_1",
            {upload(backend, codes), upload(backend, scales), x, y}, params, {rows / 4, 1, 1}, {128, 1, 1});
  if (params.output_block > 1)
    graph.add("trellis_fwht", {y}, TrellisTransformParams{rows, params.output_block, 1.0f / std::sqrt(float(params.output_block)), 0},
              {rows / params.output_block, 1, 1}, {256, 1, 1});
  graph.add("trellis_output_bf16", {y, postBuffer, poutBuffer, postBuffer, output, output}, params,
            {rows, 1, 1}, {256, 1, 1});
  static_cast<void>(backend.submitCommand(graph.dispatches()));

  std::vector<uint32_t> sample{0, 1, rows / 2, rows - 1};
  if (params.output_block > 1) {
    sample.push_back(params.output_block - 1);
    sample.push_back(params.output_block);
  }
  uint32_t wrong = 0;
  const uint16_t *got = static_cast<const uint16_t *>(output.contents());
  for (uint32_t row : sample) {
    if (row >= rows) continue;
    const float expected = realReferenceRow(codes, bits, columns, rows, scales, post, pout, info.descriptor, transformed, row);
    if (!closeEnough(bf16ToFloat(got[row]), expected)) {
      if (wrong < 3)
        std::cout << "  row " << row << " got " << bf16ToFloat(got[row]) << " expected " << expected << '\n';
      ++wrong;
    }
  }
  std::cout << name << ": " << wrong << " of " << sample.size() << " sampled rows wrong " << (wrong ? "FAIL" : "ok") << '\n';
  return wrong != 0;
}

int testTransform(MetalBackend &backend, uint32_t block) {
  std::mt19937 rng(900 + block);
  std::vector<float> values(block), expected(block);
  for (float &value : values) value = float(int32_t(rng() % 1024) - 512) / 128.0f;
  expected = values;
  trl::blockTransform(expected.data(), block, block);
  MetalBuffer buffer = upload(backend, values);
  CommandGraph graph;
  graph.add("trellis_fwht", {buffer}, TrellisTransformParams{block, block, 1.0f / std::sqrt(float(block)), 0},
            {1, 1, 1}, {256, 1, 1});
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  uint32_t wrong = 0;
  const float *got = static_cast<const float *>(buffer.contents());
  for (uint32_t i = 0; i < block; ++i) wrong += !closeEnough(got[i], expected[i]);
  std::cout << "trellis_fwht_" << block << ": " << wrong << " of " << block << " wrong " << (wrong ? "FAIL" : "ok") << '\n';
  return wrong != 0;
}

int testEmbedding(MetalBackend &backend, uint32_t bits) {
  Fixture f(bits);
  const std::vector<uint32_t> tokens{3, 0, 15, 3};
  std::vector<std::vector<float>> expected(tokens.size(), std::vector<float>(f.columns));
  for (uint32_t rowIndex = 0; rowIndex < tokens.size(); ++rowIndex) {
    const uint32_t token = tokens[rowIndex];
    std::vector<float> decoded(f.rows * f.columns);
    trl::decodeRows(f.codes.data(), bits, f.columns, 0, f.rows, f.scales.data(), decoded.data());
    std::copy(decoded.begin() + token * f.columns, decoded.begin() + (token + 1) * f.columns, expected[rowIndex].begin());
    trl::blockTransform(expected[rowIndex].data(), f.columns, f.inputBlock);
    for (uint32_t j = 0; j < f.columns; ++j) expected[rowIndex][j] *= f.pre[j] * f.post[token];
  }
  MetalBuffer tokenBuffer = upload(backend, tokens), codes = upload(backend, f.codes), scales = upload(backend, f.scales);
  MetalBuffer pre = upload(backend, f.pre), post = upload(backend, f.post);
  MetalBuffer output = sharedBuffer(backend, tokens.size() * f.columns * 2);
  CommandGraph graph;
  graph.add("trellis_embedding_" + std::to_string(bits), {tokenBuffer, codes, scales, pre, post, output},
            TrellisEmbeddingParams{uint32_t(tokens.size()), f.columns, f.inputBlock, 1.0f / std::sqrt(float(f.inputBlock))},
            {uint32_t(tokens.size()), 1, 1}, {256, 1, 1});
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  return compareRows("trellis_embedding", static_cast<const uint16_t *>(output.contents()), expected);
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2 && argc != 3) {
      std::cerr << "usage: trellis <richengine.metallib> [REAL-NARIQ.gguf]\n";
      return 2;
    }
    int failures = 0;
    try {
      MetalBackend backend(argv[1]);
      for (uint32_t block : {128u, 1024u, 2048u, 4096u}) failures += testTransform(backend, block);
      if (argc == 3)
        for (const char *name : {"blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.3.attn_q.weight", "output.band03.weight"})
          failures += testRealTensor(backend, argv[2], name);
      for (uint32_t bits = 2; bits <= 6; ++bits) {
        std::cout << "TRL" << bits << '\n';
        failures += testProduct(backend, bits, 1, false);
        failures += testProduct(backend, bits, 1, false, 256, 256, 32, 16, true);
        failures += testProduct(backend, bits, 1, false, 320, 512, 64, 32, true);
        failures += testProduct(backend, bits, 2, false, 256, 256, 32, 16, true);
        failures += testProduct(backend, bits, 3, false, 256, 256, 32, 16, true);
        failures += testProduct(backend, bits, 4, false, 256, 256, 32, 16, true);
        failures += testProduct(backend, bits, 5, 2, 256, 256, 32, 16);
        failures += testProduct(backend, bits, 8, 2, 320, 512, 64, 32);
        failures += testProduct(backend, bits, 7, 2, 128, 160, 32, 16);
        for (uint32_t k : {16u, 32u, 48u, 80u, 96u, 112u, 272u, 288u, 304u}) {
          failures += testProduct(backend, bits, 8, 2, k, 160, 16, 16);
          if (bits == 3) failures += testProduct(backend, bits, 8, 2, k, 160, 16, 16, false, true);
          failures += testProduct(backend, bits, 16, 3, k, 160, 16, 16);
        }
        if (bits == 3) failures += testProduct(backend, bits, 8, 2, 320, 512, 64, 32, false, true);
        failures += testProduct(backend, bits, 9, 3, 256, 256, 32, 16);
        failures += testProduct(backend, bits, 16, 3, 320, 512, 64, 32);
        failures += testProduct(backend, bits, 32, 3, 128, 160, 32, 16);
        failures += testProduct(backend, bits, 17, 4, 256, 256, 32, 16);
        failures += testProduct(backend, bits, 32, 4, 320, 512, 32, 16);
        failures += testProduct(backend, bits, 64, 4, 128, 160, 32, 16);
        failures += testProduct(backend, bits, 4, false);
        failures += testProduct(backend, bits, 5, true);
        failures += testProduct(backend, bits, 8, true);
        failures += testProduct(backend, bits, 16, true);
        if (bits == 3) failures += testProduct(backend, bits, 16, true, 256, 64, 128, 16);
        failures += testLinearIntegration(backend, bits);
        failures += testEmbedding(backend, bits);
      }
    } catch (const std::exception &error) {
      std::cerr << "trellis: FAIL: " << error.what() << '\n';
      return 1;
    }
    std::cout << (failures ? "trellis tests FAILED" : "trellis tests passed") << '\n';
    return failures ? 1 : 0;
  }
}

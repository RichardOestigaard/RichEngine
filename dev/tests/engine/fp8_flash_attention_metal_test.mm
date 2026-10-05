#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "TestChecks.hpp"
#include "ops/PagedAttention.hpp"
#include "tuning/HostKvExtents.hpp"
#include "Fp8PageFormatReference.hpp"
#include "Q8PageFormatReference.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace richengine::kv;
using richengine::ops::tuning::HostKvExtents;

namespace {

constexpr uint32_t kStride = 32;
constexpr uint32_t kQueryStride = kStride;
constexpr uint32_t kRows = kVerifyRows;
static_assert(kStride == RICHENGINE_VERIFY_CHUNK_STRIDE);

// The three production GQA geometries. The group size selects the kernel
// specialization; the suffix names its pipelines.
struct Shape {
  uint32_t kvHeads;
  uint32_t queryHeadsPerKvHead;
  const char *suffix;
  uint32_t queryHeads() const { return kvHeads * queryHeadsPerKvHead; }
  uint32_t fusedRows() const { return kRows * queryHeadsPerKvHead; }
  Layout layout() const { return {1, kvHeads, kFp8HeadDimension, Format::Float8E4M3}; }
};
constexpr std::array<Shape, 3> kShapes{
    {{4, 6, ""}, {4, 4, "_kv4_g4"}, {2, 8, "_kv2_g8"}}};

using richengine::test::require;

id<MTLBuffer> makeBuffer(id<MTLDevice> device, uint64_t bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:std::max<uint64_t>(bytes, 1)
                          options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal buffer allocation failed");
  std::memset(result.contents, 0, result.length);
  return result;
}

id<MTLComputePipelineState>
makePipeline(id<MTLDevice> device, id<MTLLibrary> library,
             const std::string &name) {
  id<MTLFunction> function =
      [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!function)
    throw std::runtime_error("missing Metal kernel: " + name);
  NSError *error = nil;
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!result)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return result;
}

void finish(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::string message = "Metal command failed";
    if (command.error) {
      message += ": ";
      message += command.error.localizedDescription.UTF8String;
    }
    throw std::runtime_error(message);
  }
}

float keyPattern(uint32_t token, uint32_t head, uint32_t dimension) {
  int32_t centered =
      int32_t((uint64_t{token} * 37 + head * 101 + dimension * 17 +
               uint64_t{token} * dimension * 3) %
              2003) -
      1001;
  return float(centered) / 2002.0f;
}

float valuePattern(uint32_t token, uint32_t head, uint32_t dimension) {
  int32_t centered =
      int32_t((uint64_t{token} * 53 + head * 79 + dimension * 29 +
               uint64_t{token} * dimension * 5) %
              2011) -
      1005;
  return float(centered) / 1005.0f;
}

float queryPattern(uint32_t row, uint32_t head, uint32_t dimension) {
  int32_t centered = int32_t((uint64_t{row} * 43 + head * 67 + dimension * 11 +
                              uint64_t{head} * dimension * 7) %
                             1019) -
                     509;
  return float(centered) / 2036.0f;
}

// The attention layer under test is the second of a pool's two, so its region
// starts past the first one's in every extent.
constexpr uint32_t kLayer = 1;

// One lane's rows after its history. All eight rows run the verify
// entries; fewer run the prefill entries, which share the page loop and the
// reduction, as one query tile of the parameters held here.
struct Case {
  Shape shape;
  PrefillAttentionParams params;
  std::vector<uint32_t> pageTable;
  id<MTLBuffer> pageTableBuffer;
  std::vector<id<MTLBuffer>> extents;
  std::unique_ptr<HostKvExtents> pool;
  id<MTLBuffer> queries;

  uint64_t queryIndex(uint32_t head, uint32_t row, uint32_t dimension) const {
    const uint32_t kvHead = head / shape.queryHeadsPerKvHead;
    const uint32_t localHead = head % shape.queryHeadsPerKvHead;
    return ((uint64_t{kvHead} * kQueryStride + row) * shape.queryHeadsPerKvHead +
            localHead) *
               kFp8HeadDimension +
           dimension;
  }
  template <typename T> T *slab(uint32_t tensor, uint32_t logicalPage) const {
    return pool->slab<T>(kLayer, tensor, pageTable[logicalPage]);
  }
};

// The case's pages, mixed over three or more extents of its pool.
Case makeCase(id<MTLDevice> device, Shape shape, uint32_t committed,
              uint32_t activeRows, uint32_t splits) {
  Case result;
  result.shape = shape;
  Layout layout = shape.layout();
  layout.attentionLayers = kLayer + 1;
  uint32_t pages = (committed + activeRows + kPageTokens - 1) / kPageTokens;
  const auto spread = HostKvExtents::spread(pages + 2);
  std::vector<HostKvExtents::Extent> extents;
  for (uint32_t extent = 0; extent < spread.extents; ++extent) {
    result.extents.push_back(makeBuffer(
        device, HostKvExtents::extentBytes(layout, spread.extentPages)));
    extents.push_back({static_cast<std::byte *>(result.extents.back().contents),
                       result.extents.back().gpuAddress});
  }
  result.pool = std::make_unique<HostKvExtents>(layout, spread.extentPages,
                                                std::move(extents));
  result.pageTable = HostKvExtents::mixedPages(spread, pages, committed + activeRows);
  result.pageTableBuffer = makeBuffer(device, pages * sizeof(RichKvPage));
  result.pool->writeTable(result.pageTable, result.pageTableBuffer.contents);
  result.params = {committed, activeRows, kStride, pages,
                   result.pool->layer(kLayer), splits, 0};
  uint64_t queryElements =
      uint64_t{shape.queryHeads()} * kQueryStride * kFp8HeadDimension;
  result.queries = makeBuffer(device, queryElements * sizeof(BFloat16Bits));
  return result;
}

// The page format is head-major inside a page, so a kv2_g8 page is the first
// two heads of the four-head oracle page: quantize through the shared oracle
// and store the layout's own head count.
void fill(Case &data) {
  const Shape shape = data.shape;
  const Layout layout = shape.layout();
  std::vector<float> pageKeys;
  std::vector<float> pageValues;
  auto quantized = std::make_unique<Fp8LayerPage>();
  uint32_t visibleTokens = data.params.committed_tokens + data.params.rows;
  uint32_t visiblePages = (visibleTokens + kPageTokens - 1) / kPageTokens;
  for (uint32_t logicalPage = 0; logicalPage < visiblePages; ++logicalPage) {
    uint32_t valid =
        std::min(kPageTokens, visibleTokens - logicalPage * kPageTokens);
    pageKeys.assign(uint64_t{valid} * kFp8KvHeads * kFp8HeadDimension, 0.0f);
    pageValues.assign(uint64_t{valid} * kFp8KvHeads * kFp8HeadDimension, 0.0f);
    for (uint32_t token = 0; token < valid; ++token) {
      uint32_t global = logicalPage * kPageTokens + token;
      for (uint32_t head = 0; head < shape.kvHeads; ++head) {
        for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension) {
          uint64_t index = fp8LogicalIndex(token, head, dimension);
          pageKeys[index] = bfloat16ToFloat(
              floatToBFloat16(keyPattern(global, head, dimension)));
          pageValues[index] = bfloat16ToFloat(
              floatToBFloat16(valuePattern(global, head, dimension)));
        }
      }
    }
    quantizeFp8LayerPage(pageKeys, pageValues, valid, *quantized);
    const uint64_t elements = layout.elementsPerLayerPage();
    const uint64_t scales = layout.scalesPerTensorLayerPage();
    std::copy_n(quantized->keys.begin(), elements,
                data.slab<uint8_t>(RICHENGINE_KV_KEYS, logicalPage));
    std::copy_n(quantized->keyScales.begin(), scales,
                data.slab<float>(RICHENGINE_KV_KEY_SCALES, logicalPage));
    std::copy_n(quantized->values.begin(), elements,
                data.slab<uint8_t>(RICHENGINE_KV_VALUES, logicalPage));
    std::copy_n(quantized->valueScales.begin(), scales,
                data.slab<float>(RICHENGINE_KV_VALUE_SCALES, logicalPage));
  }

  auto *queries = static_cast<BFloat16Bits *>(data.queries.contents);
  for (uint32_t head = 0; head < shape.queryHeads(); ++head)
    for (uint32_t row = 0; row < data.params.rows; ++row)
      for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension)
        queries[data.queryIndex(head, row, dimension)] =
            floatToBFloat16(queryPattern(row, head, dimension));
}

float loadKey(const Case &data, uint32_t token, uint32_t head,
              uint32_t dimension) {
  uint32_t logicalPage = token / kPageTokens;
  uint32_t pageToken = token % kPageTokens;
  float scale = data.slab<const float>(RICHENGINE_KV_KEY_SCALES, logicalPage)[
      richengine_kv_scale_element(head, pageToken)];
  return fp8E4m3ToFloat(data.slab<const uint8_t>(RICHENGINE_KV_KEYS, logicalPage)[
             richengine_kv_key_element(head, pageToken, dimension)]) *
         scale;
}

float loadValue(const Case &data, uint32_t token, uint32_t head,
                uint32_t dimension) {
  uint32_t logicalPage = token / kPageTokens;
  uint32_t pageToken = token % kPageTokens;
  float scale = data.slab<const float>(RICHENGINE_KV_VALUE_SCALES, logicalPage)[
      richengine_kv_scale_element(head, pageToken)];
  return fp8E4m3ToFloat(data.slab<const uint8_t>(RICHENGINE_KV_VALUES, logicalPage)[
             richengine_kv_value_element(head, pageToken, dimension)]) *
         scale;
}

// Double-precision softmax attention over stored FP8 pages or the
// unquantized BF16 pattern, with the fp16 query rounding the production tile
// applies and BF16 rounding only at the final output.
std::vector<BFloat16Bits> cpuReference(const Case &data, bool quantized) {
  const Shape shape = data.shape;
  std::vector<BFloat16Bits> output(
      uint64_t{shape.queryHeads()} * kQueryStride * kFp8HeadDimension,
      BFloat16Bits{0});
  const auto *queries =
      static_cast<const BFloat16Bits *>(data.queries.contents);
  for (uint32_t queryHead = 0; queryHead < shape.queryHeads(); ++queryHead) {
    uint32_t kvHead = queryHead / shape.queryHeadsPerKvHead;
    for (uint32_t row = 0; row < data.params.rows; ++row) {
      uint32_t visible = data.params.committed_tokens + row + 1;
      std::vector<double> weights(visible);
      double maximum = -std::numeric_limits<double>::infinity();
      for (uint32_t token = 0; token < visible; ++token) {
        double score = 0.0;
        for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension) {
          // The tile feeds fp16-staged queries; the reference rounds the same
          // way so the comparison isolates the fp8 read path.
          const float query =
              quantized
                  ? queryStage(bfloat16ToFloat(
                        queries[data.queryIndex(queryHead, row, dimension)]))
                  : bfloat16ToFloat(
                        queries[data.queryIndex(queryHead, row, dimension)]);
          const float key =
              quantized
                  ? loadKey(data, token, kvHead, dimension)
                  : bfloat16ToFloat(floatToBFloat16(
                        keyPattern(token, kvHead, dimension)));
          score += double(query) * key;
        }
        weights[token] = score * 0.0625;
        maximum = std::max(maximum, weights[token]);
      }
      double denominator = 0.0;
      for (double &weight : weights) {
        weight = std::exp(weight - maximum);
        denominator += weight;
      }
      for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension) {
        double value = 0.0;
        for (uint32_t token = 0; token < visible; ++token) {
          const float storedValue =
              quantized
                  ? loadValue(data, token, kvHead, dimension)
                  : bfloat16ToFloat(floatToBFloat16(
                        valuePattern(token, kvHead, dimension)));
          value += weights[token] * storedValue;
        }
        output[data.queryIndex(queryHead, row, dimension)] =
            floatToBFloat16(float(value / denominator));
      }
    }
  }
  return output;
}

// The verify and prefill split of one geometry, each with the shared
// reduction.
struct Pipelines {
  std::string splitName;
  id<MTLComputePipelineState> split;
  id<MTLComputePipelineState> reduce;
  std::string prefillSplitName;
  id<MTLComputePipelineState> prefillSplit;
  id<MTLComputePipelineState> prefillReduce;
};

Pipelines makePipelines(id<MTLDevice> device, id<MTLLibrary> library,
                      Shape shape) {
  Pipelines result;
  result.splitName = std::string("verify_attention_fp8_split") + shape.suffix;
  result.split = makePipeline(device, library, result.splitName);
  result.reduce = makePipeline(
      device, library, std::string("verify_attention_reduce") + shape.suffix);
  result.prefillSplitName = std::string("prefill_attention_fp8_split") + shape.suffix;
  result.prefillSplit = makePipeline(device, library, result.prefillSplitName);
  result.prefillReduce = makePipeline(
      device, library, std::string("prefill_attention_reduce") + shape.suffix);
  std::cout << "pipeline=" << result.splitName
            << " threadgroup_bytes=" << result.split.staticThreadgroupMemoryLength
            << '\n';
  return result;
}

struct Dispatch {
  std::vector<uint8_t> output;
};

std::vector<uint8_t> copyOf(id<MTLBuffer> buffer) {
  const auto *bytes = static_cast<const uint8_t *>(buffer.contents);
  return {bytes, bytes + buffer.length};
}

// Every lane of a verify step of `width` lanes attends the case's rows.
Dispatch dispatch(id<MTLDevice> device, id<MTLCommandQueue> queue,
                  id<MTLComputePipelineState> split,
                  id<MTLComputePipelineState> reduce, const Case &data,
                  uint32_t width) {
  const Shape shape = data.shape;
  const uint32_t splits = data.params.split_count;
  require(data.params.rows == kRows, "a verify lane attends all its rows");
  const uint64_t laneBytes = data.queries.length;
  id<MTLBuffer> queries = makeBuffer(device, width * laneBytes);
  id<MTLBuffer> output = makeBuffer(device, width * laneBytes);
  const uint64_t slots = uint64_t{width} * shape.kvHeads * splits;
  id<MTLBuffer> partials = makeBuffer(
      device, slots * shape.fusedRows() * kFp8HeadDimension * sizeof(float));
  id<MTLBuffer> statistics =
      makeBuffer(device, slots * shape.fusedRows() * 2 * sizeof(float));
  for (uint32_t lane = 0; lane < width; ++lane) {
    std::memcpy(static_cast<uint8_t *>(queries.contents) + lane * laneBytes,
                data.queries.contents, laneBytes);
  }
  std::array<VerifyAttentionParams, 4> params{};
  params.fill({data.params.committed_tokens, data.params.page_table_entries,
               data.params.kv, splits, splits, 8, 8, 0});
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:split];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:partials offset:0 atIndex:1];
  [encoder setBuffer:statistics offset:0 atIndex:2];
  for (uint32_t index = 3; index < 7; ++index)
    [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:index];
  [encoder setBytes:params.data()
              length:sizeof(VerifyAttentionParams) * params.size()
             atIndex:7];
  for (id<MTLBuffer> extent : data.extents)
    [encoder useResource:extent usage:MTLResourceUsageRead];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, splits, width)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:partials offset:0 atIndex:0];
  [encoder setBuffer:statistics offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:params.data()
              length:sizeof(VerifyAttentionParams) * params.size()
             atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, shape.fusedRows(), width)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  finish(command);
  return {copyOf(output)};
}

Dispatch dispatchPrefill(id<MTLDevice> device, id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> split,
                         id<MTLComputePipelineState> reduce, const Case &data) {
  const Shape shape = data.shape;
  const uint32_t splits = data.params.split_count;
  id<MTLBuffer> output = makeBuffer(device, data.queries.length);
  const uint64_t slots = uint64_t{shape.kvHeads} * splits;
  id<MTLBuffer> partials = makeBuffer(
      device, slots * shape.fusedRows() * kFp8HeadDimension * sizeof(float));
  id<MTLBuffer> statistics =
      makeBuffer(device, slots * shape.fusedRows() * 2 * sizeof(float));
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:split];
  [encoder setBuffer:data.queries offset:0 atIndex:0];
  [encoder setBuffer:partials offset:0 atIndex:1];
  [encoder setBuffer:statistics offset:0 atIndex:2];
  [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:3];
  [encoder setBytes:&data.params length:sizeof(data.params) atIndex:4];
  for (id<MTLBuffer> extent : data.extents)
    [encoder useResource:extent usage:MTLResourceUsageRead];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, 1, splits)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:partials offset:0 atIndex:0];
  [encoder setBuffer:statistics offset:0 atIndex:1];
  [encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:&data.params length:sizeof(data.params) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, shape.fusedRows(), 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  finish(command);
  return {copyOf(output)};
}

// The fp8 gate compares against the FP8-quantized reference tightly and
// against the unquantized BF16 pattern with a looser bound: e4m3's three
// mantissa bits cost far more element error than int8's seven.
void checkOutput(const Case &data, uint32_t width,
                 const std::vector<BFloat16Bits> &expectedFp8,
                 const std::vector<BFloat16Bits> &expectedBf16,
                 const std::vector<uint8_t> &outputBytes, bool qualityGate,
                 const std::string &label) {
  const Shape shape = data.shape;
  const uint32_t activeRows = data.params.rows;
  const auto *actual =
      reinterpret_cast<const BFloat16Bits *>(outputBytes.data());
  const uint64_t laneElements = expectedFp8.size();
  require(outputBytes.size() == width * laneElements * sizeof(BFloat16Bits),
          "verify output size departed from its lanes");
  double dot = 0.0;
  double actualSquared = 0.0;
  double expectedSquared = 0.0;
  float maximumAbsolute = 0.0f;
  double qualityDot = 0.0;
  double qualityActualSquared = 0.0;
  double qualityBf16Squared = 0.0;
  float qualityMaximumAbsolute = 0.0f;
  for (uint32_t lane = 0; lane < width; ++lane) {
    for (uint32_t head = 0; head < shape.queryHeads(); ++head) {
      for (uint32_t row = 0; row < activeRows; ++row) {
        for (uint32_t dimension = 0; dimension < kFp8HeadDimension; ++dimension) {
          const uint64_t local = data.queryIndex(head, row, dimension);
          const uint64_t index = uint64_t{lane} * laneElements + local;
          const float observed = bfloat16ToFloat(actual[index]);
          const float fp8Reference = bfloat16ToFloat(expectedFp8[local]);
          const float bf16Reference = bfloat16ToFloat(expectedBf16[local]);
          maximumAbsolute =
              std::max(maximumAbsolute, std::abs(observed - fp8Reference));
          dot += double(observed) * fp8Reference;
          actualSquared += double(observed) * observed;
          expectedSquared += double(fp8Reference) * fp8Reference;
          qualityMaximumAbsolute = std::max(
              qualityMaximumAbsolute, std::abs(observed - bf16Reference));
          qualityDot += double(observed) * bf16Reference;
          qualityActualSquared += double(observed) * observed;
          qualityBf16Squared += double(bf16Reference) * bf16Reference;
        }
      }
    }
  }
  double cosine = dot / std::sqrt(actualSquared * expectedSquared);
  const double qualityCosine =
      qualityDot / std::sqrt(qualityActualSquared * qualityBf16Squared);
  std::cout << label << " lanes=" << width
            << " active_rows=" << activeRows
            << " committed=" << data.params.committed_tokens
            << " splits=" << data.params.split_count << " fp8_cosine=" << cosine
            << " bf16_cosine=" << qualityCosine
            << " bf16_max_absolute=" << qualityMaximumAbsolute << '\n';
  require(cosine > 0.999,
          "production fp8 attention differs from its FP8 reference");
  require(maximumAbsolute < 0.02f,
          "production fp8 attention exceeds its FP8 error bound");
  require(!qualityGate ||
              (qualityCosine > 0.995 && qualityMaximumAbsolute < 0.05f),
          "FP8 production attention failed its BF16 quality gate");
}

void runCase(id<MTLDevice> device, id<MTLCommandQueue> queue,
             const Pipelines &pipelines, Shape shape,
             uint32_t committed, uint32_t activeRows, uint32_t width,
             bool qualityGate = true, uint32_t splits = kVerifySplits) {
  require(width >= 1 && width <= 4 &&
              (activeRows == kRows ||
               (width == 1 && splits <= RICHENGINE_PREFILL_ATTENTION_MAXIMUM_SPLITS)),
          "invalid attention case");
  Case data = makeCase(device, shape, committed, activeRows, splits);
  fill(data);
  const std::vector<BFloat16Bits> expectedFp8 = cpuReference(data, true);
  const std::vector<BFloat16Bits> expectedBf16 = cpuReference(data, false);
  const bool verify = activeRows == kRows;
  const std::string &name = verify ? pipelines.splitName : pipelines.prefillSplitName;
  const auto run = [&](const Case &c) {
    return verify ? dispatch(device, queue, pipelines.split, pipelines.reduce, c, width)
                  : dispatchPrefill(device, queue, pipelines.prefillSplit,
                                    pipelines.prefillReduce, c);
  };
  const Dispatch first = run(data);
  checkOutput(data, width, expectedFp8, expectedBf16, first.output, qualityGate, name);
  const Dispatch repeat = run(data);
  checkOutput(data, width, expectedFp8, expectedBf16, repeat.output, qualityGate,
              name + "_repeat");
  require(first.output == repeat.output,
          "attention repeat is not bit-identical to the first submission");
}

// Read-path bandwidth: one verify step per pipeline over the same pages.
// fp8 moves the same bytes as q8 (one byte per element) but skips the int8
// unpack to bf16-like operands; the measurement isolates the split kernel.
void benchmark(id<MTLDevice> device, id<MTLCommandQueue> queue,
               id<MTLLibrary> library, Shape shape, uint32_t committed) {
  constexpr uint32_t kIterations = 20;
  Case data = makeCase(device, shape, committed, kRows, kVerifySplits);
  fill(data);
  const uint32_t splits = data.params.split_count;
  const uint32_t width = 4;
  const uint64_t laneBytes = data.queries.length;
  id<MTLBuffer> queries = makeBuffer(device, width * laneBytes);
  id<MTLBuffer> output = makeBuffer(device, width * laneBytes);
  const uint64_t slots = uint64_t{width} * shape.kvHeads * splits;
  id<MTLBuffer> partials = makeBuffer(
      device, slots * shape.fusedRows() * kFp8HeadDimension * sizeof(float));
  id<MTLBuffer> statistics =
      makeBuffer(device, slots * shape.fusedRows() * 2 * sizeof(float));
  std::array<VerifyAttentionParams, 4> params{};
  params.fill({data.params.committed_tokens, data.params.page_table_entries,
               data.params.kv, splits, splits, 8, 8, 0});
  const uint64_t kvBytes =
      uint64_t{committed + kRows} * shape.kvHeads * kFp8HeadDimension * 2;
  for (const char *base :
       {"verify_attention_q8_split", "verify_attention_fp8_split"}) {
    const std::string name = std::string(base) + shape.suffix;
    id<MTLComputePipelineState> split = makePipeline(device, library, name);
    // Warm up once so the measurement excludes pipeline setup.
    for (uint32_t iteration = 0; iteration <= kIterations; ++iteration) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:split];
      [encoder setBuffer:queries offset:0 atIndex:0];
      [encoder setBuffer:partials offset:0 atIndex:1];
      [encoder setBuffer:statistics offset:0 atIndex:2];
      for (uint32_t index = 3; index < 7; ++index)
        [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:index];
      [encoder setBytes:params.data()
                  length:sizeof(VerifyAttentionParams) * params.size()
                 atIndex:7];
      for (id<MTLBuffer> extent : data.extents)
        [encoder useResource:extent usage:MTLResourceUsageRead];
      [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, splits, width)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      if (iteration == 0) {
        finish(command);
        continue;
      }
      const auto start = std::chrono::steady_clock::now();
      finish(command);
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count();
      static_cast<void>(output);
      if (iteration == kIterations / 2)
        std::cout << "benchmark " << name << " committed=" << committed
                  << " width=" << width << " splits=" << splits
                  << " ms=" << seconds * 1000.0
                  << " kv_gbps=" << kvBytes * width / seconds * 1e-9 << '\n';
    }
  }
}

void run(const char *libraryPath) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device)
    throw std::runtime_error("Metal device unavailable");
  NSError *error = nil;
  NSURL *url =
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryPath]];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (!library)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLCommandQueue> queue = [device newCommandQueue];
  for (const Shape shape : kShapes) {
    const Pipelines pipelines = makePipelines(device, library, shape);
    for (uint32_t width = 1; width <= 4; ++width)
      runCase(device, queue, pipelines, shape, 127, 8, width);
    runCase(device, queue, pipelines, shape, 0, 8, 4);
    for (uint32_t activeRows = 1; activeRows < kRows; ++activeRows)
      runCase(device, queue, pipelines, shape, 127, activeRows, 1);
    runCase(device, queue, pipelines, shape, 129, 8, 4);
    runCase(device, queue, pipelines, shape, 421, 3, 1);
    runCase(device, queue, pipelines, shape, 1'100, 8, 2, false);
    runCase(device, queue, pipelines, shape, 4'093, 8, 3, false, 65);
    runCase(device, queue, pipelines, shape, 8'192, 8, 2, false, 65);
    benchmark(device, queue, library, shape, 32'768);
  }
  std::cout << "fp8_flash_attention_metal_test: ok\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::runtime_error(
            "usage: fp8_flash_attention_metal_test <metallib>");
      run(argv[1]);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "fp8_flash_attention_metal_test: " << error.what() << '\n';
      return 1;
    }
  }
}

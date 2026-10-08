#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// hd512 (Gemma 4 global layer) paged-attention kernels: the single-pass
// *_hd512 splits against their two-pass *_hd512_m2 variants. The check
// dispatches both over the same random KV pages and queries and requires
// the reduced bf16 outputs to match bit for bit — every fused row runs the
// identical page and rescale sequence in both, only the accumulator's
// footprint differs. A scalar CPU oracle validates one configuration
// outright. The timing loop reports µs per split dispatch for both
// pipelines at real canvas shapes (2 KV heads x hd512 x 256 rows) over
// prefix lengths 0 / 1024 / 8192.
//
// usage: hd512-attention METALLIB [--quick]

#include "TestChecks.hpp"
#include "Q8PageFormatReference.hpp"
#include "ops/KernelNames.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace richengine::kv;
using richengine::test::require;

constexpr uint32_t kLayer = 1; // second of the pool's two layers
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kGroup = 8;
constexpr uint32_t kDim = 512;
constexpr uint32_t kPageTokens = RICHENGINE_TARGET_KV_BLOCK_TOKENS;
constexpr uint32_t kTileRows = RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;
constexpr uint32_t kFused = kTileRows * kGroup; // 64
constexpr uint32_t kVerifyRows = RICHENGINE_TARGET_VERIFY_ROWS;
constexpr uint32_t kVerifyStride = RICHENGINE_VERIFY_CHUNK_STRIDE;

id<MTLBuffer> makeBuffer(id<MTLDevice> device, uint64_t bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:std::max<uint64_t>(bytes, 1)
                          options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal buffer allocation failed");
  std::memset(result.contents, 0, result.length);
  return result;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
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

// The ABI placement helpers (metal/abi/KvExtent.h) are inline, so the test
// pool needs no engine objects: extents are MTLBuffers, pages the entries
// of a table it writes itself.
struct Case {
  Format format;
  uint32_t committed = 0;   // prefix tokens
  uint32_t rows = 0;        // query rows (canvas: 256)
  uint32_t stride = 0;      // chunk_stride (multiple of 32)
  uint32_t splits = 1;
  uint32_t window = 0;      // canvas swa window, 0 = none
  Layout layout{};
  uint32_t extentPages = 0;
  uint32_t dataBytes = 0;   // one tensor of one layer's page
  uint32_t scaleBytes = 0;
  std::vector<id<MTLBuffer>> extents;
  std::vector<uint32_t> pageIds;
  id<MTLBuffer> pageTable;
  id<MTLBuffer> queries;    // [kv head][stride][group][dim] bf16
  PrefillAttentionParams params{};

  uint64_t queryIndex(uint32_t kvHead, uint32_t row, uint32_t groupRow,
                      uint32_t dim) const {
    return ((uint64_t{kvHead} * stride + row) * kGroup + groupRow) * kDim + dim;
  }
  uint32_t page(uint32_t token) const { return pageIds[token / kPageTokens]; }
  std::byte *extent(uint32_t pageId) const {
    return static_cast<std::byte *>(
        extents[pageId / extentPages].contents);
  }
  template <typename T> T *slab(uint32_t tensor, uint32_t token) const {
    const uint32_t id = page(token);
    return reinterpret_cast<T *>(
        extent(id) + richengine_kv_offset(extentPages, dataBytes, scaleBytes,
                                          kLayer, tensor, id % extentPages));
  }
};

// Deterministic pseudo-random K/V of the layout's format over every visible
// page, plus the query tile. INT8 stores signed bytes, INT4 packed nibbles
// (token-major), BF16 stored values with no scales. Three extents of a
// non-power-of-two page count hold the pages, so a table crosses extents.
Case makeCase(id<MTLDevice> device, Format format, uint32_t committed,
              uint32_t rows, uint32_t splits, uint32_t seed) {
  Case data;
  data.format = format;
  data.committed = committed;
  data.rows = rows;
  data.stride = (rows + kPageTokens - 1) / kPageTokens * kPageTokens;
  data.splits = splits;
  data.layout = {uint32_t{kLayer + 1}, kKvHeads, kDim, format};
  data.dataBytes = uint32_t(data.layout.dataBytesPerLayerPage());
  data.scaleBytes = uint32_t(data.layout.scaleBytesPerLayerPage());
  const uint32_t visible = committed + rows;
  const uint32_t pages = (visible + kPageTokens - 1) / kPageTokens;
  data.extentPages =
      std::clamp((pages + 2) / 3, 3U, uint32_t{RICHENGINE_KV_PAGE_INDEX_MASK});
  if (std::has_single_bit(data.extentPages))
    ++data.extentPages;
  const uint32_t extentCount =
      std::max(3U, (pages + data.extentPages - 1) / data.extentPages);
  const uint64_t extentBytes =
      uint64_t{data.extentPages} * data.layout.bytesPerModelPage();
  for (uint32_t extent = 0; extent < extentCount; ++extent)
    data.extents.push_back(makeBuffer(device, extentBytes));
  data.pageIds.resize(pages);
  {
    // Shuffled page ids spread over the extents.
    std::vector<uint32_t> ids(extentCount * data.extentPages);
    for (uint32_t i = 0; i < ids.size(); ++i) ids[i] = i;
    std::mt19937 shuffler(seed);
    std::shuffle(ids.begin(), ids.end(), shuffler);
    std::copy_n(ids.begin(), pages, data.pageIds.begin());
  }
  data.pageTable = makeBuffer(device, pages * sizeof(RichKvPage));
  {
    auto *entries = static_cast<RichKvPage *>(data.pageTable.contents);
    for (uint32_t i = 0; i < pages; ++i) {
      const uint32_t id = data.pageIds[i];
      entries[i] = richengine_kv_page_entry(
          data.extents[id / data.extentPages].gpuAddress,
          id % data.extentPages);
    }
  }
  data.params = {committed, rows, data.stride, pages,
                 richengine_kv_layer(data.extentPages, data.dataBytes,
                                     data.scaleBytes, kLayer),
                 splits, 0.0f};

  std::mt19937 random(seed);
  const auto code = [&] {
    if (format == Format::Int4)
      return int(random() % 15) - 7;
    return int(random() % 255) - 127;
  };
  const float keyScale = format == Format::Int4 ? 0.096f : 0.006f;
  const float valueScale = format == Format::Int4 ? 0.112f : 0.007f;
  for (uint32_t token = 0; token < visible; ++token) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
      const uint64_t scaleIndex =
          richengine_kv_scale_element(head, token % kPageTokens);
      if (format == Format::Int8 || format == Format::Int4) {
        data.slab<float>(RICHENGINE_KV_KEY_SCALES, token)[scaleIndex] = keyScale;
        data.slab<float>(RICHENGINE_KV_VALUE_SCALES, token)[scaleIndex] =
            valueScale;
      }
      for (uint32_t dim = 0; dim < kDim; dim += 2) {
        const int key0 = code(), key1 = code();
        const int val0 = code(), val1 = code();
        const uint64_t keyElement = richengine_kv_key_element_dim(
            head, token % kPageTokens, dim, kDim);
        if (format == Format::Int4) {
          // Keys and values both pack token-major dim pairs.
          data.slab<uint8_t>(RICHENGINE_KV_KEYS, token)[keyElement / 2] =
              uint8_t((key0 & 0xF) | (key1 & 0xF) << 4);
          data.slab<uint8_t>(RICHENGINE_KV_VALUES, token)[keyElement / 2] =
              uint8_t((val0 & 0xF) | (val1 & 0xF) << 4);
          continue;
        }
        const uint64_t valueElement = richengine_kv_value_element_dim(
            head, token % kPageTokens, dim, kDim);
        if (format == Format::Int8) {
          data.slab<int8_t>(RICHENGINE_KV_KEYS, token)[keyElement] = int8_t(key0);
          data.slab<int8_t>(RICHENGINE_KV_KEYS, token)[keyElement + 1] =
              int8_t(key1);
          data.slab<int8_t>(RICHENGINE_KV_VALUES, token)[valueElement] =
              int8_t(val0);
          data.slab<int8_t>(RICHENGINE_KV_VALUES, token)[valueElement + kPageTokens] =
              int8_t(val1);
        } else {
          data.slab<uint16_t>(RICHENGINE_KV_KEYS, token)[keyElement] =
              floatToBFloat16(key0 * 0.006f);
          data.slab<uint16_t>(RICHENGINE_KV_KEYS, token)[keyElement + 1] =
              floatToBFloat16(key1 * 0.006f);
          data.slab<uint16_t>(RICHENGINE_KV_VALUES, token)[valueElement] =
              floatToBFloat16(val0 * 0.007f);
          data.slab<uint16_t>(RICHENGINE_KV_VALUES, token)[valueElement + kPageTokens] =
              floatToBFloat16(val1 * 0.007f);
        }
      }
    }
  }

  const uint64_t queryElements =
      uint64_t{kKvHeads} * data.stride * kGroup * kDim;
  data.queries = makeBuffer(device, queryElements * sizeof(BFloat16Bits));
  auto *queries = static_cast<BFloat16Bits *>(data.queries.contents);
  for (uint64_t index = 0; index < queryElements; ++index)
    queries[index] =
        floatToBFloat16(float(int(random() % 1019) - 509) / 1018.0f);
  return data;
}

// The stored key/value element dequantized, for the CPU oracle.
float storedKey(const Case &data, uint32_t token, uint32_t head, uint32_t dim) {
  const uint64_t element = richengine_kv_key_element_dim(
      head, token % kPageTokens, dim, kDim);
  const float scale =
      data.format == Format::BFloat16
          ? 1.0f
          : data.slab<const float>(RICHENGINE_KV_KEY_SCALES, token)[
                richengine_kv_scale_element(head, token % kPageTokens)];
  if (data.format == Format::BFloat16)
    return bfloat16ToFloat(
        data.slab<const uint16_t>(RICHENGINE_KV_KEYS, token)[element]);
  if (data.format == Format::Int4) {
    const uint8_t byte =
        data.slab<const uint8_t>(RICHENGINE_KV_KEYS, token)[element / 2];
    const int nibble = int(byte >> ((dim & 1) * 4)) & 0xF;
    return float(nibble - (nibble & 0x8 ? 16 : 0)) * scale;
  }
  return float(data.slab<const int8_t>(RICHENGINE_KV_KEYS, token)[element]) *
         scale;
}

float storedValue(const Case &data, uint32_t token, uint32_t head,
                  uint32_t dim) {
  const float scale =
      data.format == Format::BFloat16
          ? 1.0f
          : data.slab<const float>(RICHENGINE_KV_VALUE_SCALES, token)[
                richengine_kv_scale_element(head, token % kPageTokens)];
  if (data.format == Format::Int4) {
    const uint64_t byte = richengine_kv_key_element_dim(
                              head, token % kPageTokens, dim, kDim) /
                          2;
    const uint8_t packed =
        data.slab<const uint8_t>(RICHENGINE_KV_VALUES, token)[byte];
    const int nibble = int(packed >> ((dim & 1) * 4)) & 0xF;
    return float(nibble - (nibble & 0x8 ? 16 : 0)) * scale;
  }
  const uint64_t element = richengine_kv_value_element_dim(
      head, token % kPageTokens, dim, kDim);
  if (data.format == Format::BFloat16)
    return bfloat16ToFloat(
        data.slab<const uint16_t>(RICHENGINE_KV_VALUES, token)[element]);
  return float(data.slab<const int8_t>(RICHENGINE_KV_VALUES, token)[element]) *
         scale;
}

// One fused row's scalar softmax reference. causal=true attends
// [0, committed + queryRow]; canvas attends the whole visible range with the
// optional prefix window.
std::vector<float> reference(const Case &data, uint32_t kvHead,
                             uint32_t fusedRow, bool causal) {
  const uint32_t queryRow = fusedRow / kGroup;
  const uint32_t groupRow = fusedRow % kGroup;
  const uint32_t tileStart = queryRow / kTileRows * kTileRows;
  const uint32_t active = std::min(kTileRows, data.rows - tileStart);
  const uint32_t committed = data.committed + tileStart;
  const uint32_t visible = data.committed + data.rows;
  const uint32_t tileRow = queryRow - tileStart;
  const uint32_t causalEnd =
      causal ? committed + std::min(tileRow, active - 1) + 1 : visible;
  const uint32_t begin =
      !causal && data.window && data.committed > data.window
          ? data.committed - data.window
          : (causal && data.window && causalEnd > data.window
                 ? causalEnd - data.window
                 : 0);
  const auto *queries =
      static_cast<const BFloat16Bits *>(data.queries.contents);
  std::vector<double> weights(visible, 0.0);
  double maximum = -std::numeric_limits<double>::infinity();
  for (uint32_t token = begin; token < causalEnd; ++token) {
    double score = 0.0;
    for (uint32_t dim = 0; dim < kDim; ++dim)
      score += double(bfloat16ToFloat(
                   queries[data.queryIndex(kvHead, queryRow, groupRow, dim)])) *
               storedKey(data, token, kvHead, dim);
    weights[token] = score * 0.0625; // 1/sqrt(512) default scale
    maximum = std::max(maximum, weights[token]);
  }
  double denominator = 0.0;
  for (double &weight : weights) {
    weight = weight == 0.0 ? 0.0 : std::exp(weight - maximum);
    denominator += weight;
  }
  std::vector<float> output(kDim);
  for (uint32_t dim = 0; dim < kDim; ++dim) {
    double value = 0.0;
    for (uint32_t token = begin; token < causalEnd; ++token)
      value += weights[token] * storedValue(data, token, kvHead, dim);
    output[dim] = float(value / denominator);
  }
  return output;
}

struct Scratch {
  id<MTLBuffer> partials;
  id<MTLBuffer> statistics;
  id<MTLBuffer> output;
};

Scratch makeScratch(id<MTLDevice> device, uint64_t slots, uint64_t outElements) {
  return {makeBuffer(device, slots * kFused * kDim * sizeof(float)),
          makeBuffer(device, slots * kFused * 2 * sizeof(float)),
          makeBuffer(device, outElements * sizeof(BFloat16Bits))};
}

// Encode one prefill/canvas split (and, when reduce is set, the shared
// gemma_hd512 reduce) into `encoder`.
void encodePrefillSplit(id<MTLComputeCommandEncoder> encoder,
                        id<MTLComputePipelineState> split, const Case &data,
                        const Scratch &scratch, bool windowedSlot,
                        bool canvas) {
  [encoder setComputePipelineState:split];
  [encoder setBuffer:data.queries offset:0 atIndex:0];
  [encoder setBuffer:scratch.partials offset:0 atIndex:1];
  [encoder setBuffer:scratch.statistics offset:0 atIndex:2];
  [encoder setBuffer:data.pageTable offset:0 atIndex:3];
  [encoder setBytes:&data.params length:sizeof(data.params) atIndex:4];
  if (windowedSlot || canvas)
    [encoder setBytes:&data.window length:sizeof(data.window) atIndex:5];
  for (id<MTLBuffer> extent : data.extents)
    [encoder useResource:extent usage:MTLResourceUsageRead];
  const uint32_t tiles = (data.rows + kTileRows - 1) / kTileRows;
  [encoder dispatchThreadgroups:MTLSizeMake(kKvHeads, tiles, data.splits)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void encodePrefillReduce(id<MTLComputeCommandEncoder> encoder,
                         id<MTLComputePipelineState> reduce, const Case &data,
                         const Scratch &scratch) {
  const uint32_t tiles = (data.rows + kTileRows - 1) / kTileRows;
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:scratch.partials offset:0 atIndex:0];
  [encoder setBuffer:scratch.statistics offset:0 atIndex:1];
  [encoder setBuffer:scratch.output offset:0 atIndex:2];
  [encoder setBytes:&data.params length:sizeof(data.params) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(kKvHeads, kFused, tiles)
          threadsPerThreadgroup:MTLSizeMake(kDim, 1, 1)];
}

std::vector<uint16_t> runPrefill(id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> split,
                                 id<MTLComputePipelineState> reduce,
                                 const Case &data, bool windowedSlot,
                                 bool canvas) {
  const uint32_t tiles = (data.rows + kTileRows - 1) / kTileRows;
  const uint64_t slots = uint64_t{tiles} * kKvHeads * data.splits;
  auto scratch =
      makeScratch(queue.device, slots, uint64_t{kKvHeads} * data.stride *
                                           kGroup * kDim);
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  encodePrefillSplit(encoder, split, data, scratch, windowedSlot, canvas);
  encodePrefillReduce(encoder, reduce, data, scratch);
  [encoder endEncoding];
  finish(command);
  const auto *begin = static_cast<const uint16_t *>(scratch.output.contents);
  return {begin, begin + scratch.output.length / 2};
}

// Verify: `lanes` identical lanes of kVerifyRows rows, one page table each.
std::vector<uint16_t> runVerify(id<MTLCommandQueue> queue,
                                id<MTLComputePipelineState> split,
                                id<MTLComputePipelineState> reduce,
                                const Case &data, uint32_t lanes,
                                bool windowed) {
  const uint64_t slots = uint64_t{lanes} * kKvHeads * data.splits;
  const uint64_t laneElements =
      uint64_t{kKvHeads} * kVerifyStride * kGroup * kDim;
  id<MTLBuffer> queries = makeBuffer(queue.device, lanes * laneElements * 2);
  // Re-tile the case's first 8 rows into verify staging (stride 32).
  const auto *source = static_cast<const uint16_t *>(data.queries.contents);
  auto *staged = static_cast<uint16_t *>(queries.contents);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t head = 0; head < kKvHeads; ++head)
      for (uint32_t row = 0; row < kVerifyRows; ++row)
        for (uint32_t g = 0; g < kGroup; ++g)
          for (uint32_t d = 0; d < kDim; ++d)
            staged[((uint64_t{lane} * kKvHeads + head) * kVerifyStride * kGroup +
                    uint64_t{row} * kGroup + g) *
                       kDim +
                   d] = source[data.queryIndex(head, row, g, d)];
  auto scratch = makeScratch(queue.device, slots, lanes * laneElements);
  std::array<VerifyAttentionParams, 4> params{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    params[lane] = {data.committed, data.params.page_table_entries,
                    data.params.kv, data.splits, data.splits,
                    kVerifyRows, kVerifyRows, 0.0f};
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:split];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:scratch.partials offset:0 atIndex:1];
  [encoder setBuffer:scratch.statistics offset:0 atIndex:2];
  for (uint32_t index = 0; index < 4; ++index)
    [encoder setBuffer:data.pageTable offset:0 atIndex:3 + index];
  [encoder setBytes:params.data()
              length:sizeof(VerifyAttentionParams) * params.size()
             atIndex:7];
  if (windowed)
    [encoder setBytes:&data.window length:sizeof(data.window) atIndex:8];
  for (id<MTLBuffer> extent : data.extents)
    [encoder useResource:extent usage:MTLResourceUsageRead];
  [encoder dispatchThreadgroups:MTLSizeMake(kKvHeads, data.splits, lanes)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:scratch.partials offset:0 atIndex:0];
  [encoder setBuffer:scratch.statistics offset:0 atIndex:1];
  [encoder setBuffer:scratch.output offset:0 atIndex:2];
  [encoder setBytes:params.data()
              length:sizeof(VerifyAttentionParams) * params.size()
             atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(kKvHeads, kFused, lanes)
          threadsPerThreadgroup:MTLSizeMake(kDim, 1, 1)];
  [encoder endEncoding];
  finish(command);
  const auto *begin = static_cast<const uint16_t *>(scratch.output.contents);
  return {begin, begin + scratch.output.length / 2};
}

struct Comparison {
  uint64_t mismatched = 0;
  float maximumError = 0;
  double cosine = 0;
};

Comparison compare(const std::vector<uint16_t> &a,
                   const std::vector<uint16_t> &b) {
  require(a.size() == b.size(), "output sizes differ");
  Comparison result;
  double dot = 0, asq = 0, bsq = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      ++result.mismatched;
      result.maximumError = std::max(
          result.maximumError,
          std::abs(bfloat16ToFloat(a[i]) - bfloat16ToFloat(b[i])));
    }
    const double av = bfloat16ToFloat(a[i]), bv = bfloat16ToFloat(b[i]);
    dot += av * bv;
    asq += av * av;
    bsq += bv * bv;
  }
  result.cosine = dot / std::sqrt(asq * bsq);
  return result;
}

// µs per split-dispatch iteration, `iterations` encoded back to back in one
// command buffer after a warm-up.
double timeSplit(id<MTLCommandQueue> queue,
                 id<MTLComputePipelineState> split, const Case &data,
                 bool windowedSlot, bool canvas, uint32_t iterations) {
  const uint32_t tiles = (data.rows + kTileRows - 1) / kTileRows;
  const uint64_t slots = uint64_t{tiles} * kKvHeads * data.splits;
  auto scratch =
      makeScratch(queue.device, slots, uint64_t{kKvHeads} * data.stride *
                                           kGroup * kDim);
  const auto encodeAll = [&](id<MTLComputeCommandEncoder> encoder) {
    for (uint32_t i = 0; i < iterations; ++i)
      encodePrefillSplit(encoder, split, data, scratch, windowedSlot, canvas);
  };
  {
    id<MTLCommandBuffer> warmup = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [warmup computeCommandEncoder];
    encodePrefillSplit(encoder, split, data, scratch, windowedSlot, canvas);
    [encoder endEncoding];
    finish(warmup);
  }
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  encodeAll(encoder);
  [encoder endEncoding];
  const auto start = std::chrono::steady_clock::now();
  finish(command);
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(end - start).count() /
         iterations;
}

std::string stem(Format format) {
  return format == Format::Int8     ? "q8"
         : format == Format::Int4   ? "int4"
                                    : "bf16";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc >= 2, "usage: hd512-attention METALLIB [--quick]");
    const bool quick = argc > 2 && std::string_view(argv[2]) == "--quick";
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "no Metal device");
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                     error:&error];
    if (!library)
      throw std::runtime_error(error.localizedDescription.UTF8String);
    id<MTLCommandQueue> queue = [device newCommandQueue];

    const std::array<Format, 3> formats{Format::Int8, Format::Int4,
                                        Format::BFloat16};
    id<MTLComputePipelineState> prefillReduce =
        makePipeline(device, library, std::string(richengine::ops::kPrefillAttentionReduceGemmaHd512));
    id<MTLComputePipelineState> canvasReduce = makePipeline(
        device, library, std::string(richengine::ops::kPrefillAttentionReduceCanvasGemmaHd512));
    id<MTLComputePipelineState> verifyReduce =
        makePipeline(device, library, std::string(richengine::ops::kVerifyAttentionReduceGemmaHd512));

    // ---- Correctness: every format, prefill + canvas + verify, old vs m2.
    for (Format format : formats) {
      const std::string name = stem(format);
      const struct {
        id<MTLComputePipelineState> prefill, prefillM2, canvas, canvasM2,
            verify, verifyM2;
      } pipes{
          makePipeline(device, library,
                       std::string(richengine::ops::kPrefillAttention) + "_" + name + "_split_hd512"),
          makePipeline(device, library,
                       std::string(richengine::ops::kPrefillAttention) + "_" + name + "_split_hd512_m2"),
          makePipeline(device, library,
                       std::string(richengine::ops::kPrefillAttention) + "_" + name + "_split_canvas_hd512"),
          makePipeline(device, library,
                       std::string(richengine::ops::kPrefillAttention) + "_" + name + "_split_canvas_hd512_m2"),
          makePipeline(device, library,
                       std::string(richengine::ops::kVerifyAttention) + "_" + name + "_split_gemma_hd512"),
          makePipeline(device, library,
                       std::string(richengine::ops::kVerifyAttention) + "_" + name + "_split_gemma_hd512_m2"),
      };
      for (const uint32_t committed : {0U, 1024U}) {
        const uint32_t rows = 64; // eight tiles
        const uint32_t splits = committed == 0 ? 1 : 3;
        auto data = makeCase(device, format, committed, rows, splits,
                             0x5150 + committed + uint32_t(format));

        const auto prefillOld =
            runPrefill(queue, pipes.prefill, prefillReduce, data, false, false);
        const auto prefillNew = runPrefill(queue, pipes.prefillM2,
                                           prefillReduce, data, false, false);
        auto cmp = compare(prefillOld, prefillNew);
        require(cmp.mismatched == 0,
                "hd512_m2 prefill output differs from hd512");
        std::cout << "parity " << name << " prefill committed=" << committed
                  << ": bitwise identical\n";

        const auto canvasOld =
            runPrefill(queue, pipes.canvas, canvasReduce, data, false, true);
        const auto canvasNew = runPrefill(queue, pipes.canvasM2, canvasReduce,
                                          data, false, true);
        cmp = compare(canvasOld, canvasNew);
        require(cmp.mismatched == 0,
                "hd512_m2 canvas output differs from hd512");
        std::cout << "parity " << name << " canvas committed=" << committed
                  << ": bitwise identical\n";

        const auto verifyOld =
            runVerify(queue, pipes.verify, verifyReduce, data, 2, false);
        const auto verifyNew =
            runVerify(queue, pipes.verifyM2, verifyReduce, data, 2, false);
        cmp = compare(verifyOld, verifyNew);
        require(cmp.mismatched == 0,
                "hd512_m2 verify output differs from hd512");
        std::cout << "parity " << name << " verify committed=" << committed
                  << ": bitwise identical\n";

        // Scalar oracle on the m2 canvas path (all rows visible) for a few
        // fused rows: proves the harness and the kernel read the pages right.
        if (format == Format::Int8) {
          double dot = 0, asq = 0, bsq = 0;
          float maxError = 0;
          for (const uint32_t fusedRow : {0U, 33U, 63U}) {
            for (const uint32_t kvHead : {0U, 1U}) {
              const auto expected = reference(data, kvHead, fusedRow, false);
              const uint32_t tile = (fusedRow / kGroup) / kTileRows;
              const uint32_t row = (fusedRow / kGroup) % kTileRows;
              const uint32_t g = fusedRow % kGroup;
              for (uint32_t d = 0; d < kDim; ++d) {
                const float actual = bfloat16ToFloat(
                    canvasNew[data.queryIndex(kvHead, tile * kTileRows + row, g,
                                              d)]);
                maxError = std::max(maxError,
                                    std::abs(actual - expected[d]));
                dot += double(actual) * expected[d];
                asq += double(actual) * actual;
                bsq += double(expected[d]) * expected[d];
              }
            }
          }
          const double cosine = dot / std::sqrt(asq * bsq);
          require(maxError < 0.02f && cosine > 0.9995,
                  "hd512_m2 canvas fails the scalar oracle");
          std::cout << "oracle " << name << " canvas: maxErr=" << maxError
                    << " cosine=" << cosine << " PASS\n";
        }
      }
    }

    // SWA parity spot-check: windowed canvas and verify, int8.
    {
      auto data = makeCase(device, Format::Int8, 2048, 64, 4, 0x777);
      data.window = 1024;
      const auto oldPipe = makePipeline(
          device, library, std::string(richengine::ops::kPrefillAttentionQ8SplitCanvasSwaHd512));
      const auto newPipe = makePipeline(
          device, library, std::string(richengine::ops::kPrefillAttentionQ8SplitCanvasSwaHd512M2));
      const auto a = runPrefill(queue, oldPipe, canvasReduce, data, true, true);
      const auto b = runPrefill(queue, newPipe, canvasReduce, data, true, true);
      require(compare(a, b).mismatched == 0,
              "canvas swa hd512_m2 output differs");
      const auto oldVerify = makePipeline(
          device, library, std::string(richengine::ops::kVerifyAttentionQ8SplitSwaHd512));
      const auto newVerify = makePipeline(
          device, library, std::string(richengine::ops::kVerifyAttentionQ8SplitSwaHd512M2));
      const auto va = runVerify(queue, oldVerify, verifyReduce, data, 2, true);
      const auto vb = runVerify(queue, newVerify, verifyReduce, data, 2, true);
      require(compare(va, vb).mismatched == 0,
              "verify swa hd512_m2 output differs");
      std::cout << "parity q8 swa canvas+verify: bitwise identical\n";
    }

    // ---- Timing: canvas-shape 256 rows (32 tiles) at prefixes 0/1024/8192.
    // splits follow the production plan's shape for big tiles: few splits.
    std::cout << "\n-- hd512 canvas split timings (256 rows, 2 KV heads) --\n";
    for (Format format : formats) {
      const std::string name = stem(format);
      const auto oldPipe = makePipeline(
          device, library, std::string(richengine::ops::kPrefillAttention) + "_" + name + "_split_canvas_hd512");
      const auto newPipe = makePipeline(
          device, library,
          std::string(richengine::ops::kPrefillAttention) + "_" + name + "_split_canvas_hd512_m2");
      for (const uint32_t committed : {0U, 1024U, 8192U}) {
        const uint32_t splits = committed >= 8192 ? 4 : 1;
        auto data = makeCase(device, format, committed, 256, splits,
                             0x999 + committed);
        const uint32_t iterations = committed >= 8192 ? 4 : (quick ? 8 : 16);
        const double oldUs =
            timeSplit(queue, oldPipe, data, false, true, iterations);
        const double newUs =
            timeSplit(queue, newPipe, data, false, true, iterations);
        std::cout << name << " prefix=" << committed << " splits=" << splits
                  << " old=" << oldUs << "us m2=" << newUs << "us ("
                  << (newUs < oldUs ? "m2 wins " : "old wins ")
                  << (100.0 * (oldUs - newUs) / oldUs) << "%)\n"
                  << std::flush;
      }
    }

    std::cout << "hd512 attention m2: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hd512 attention m2: FAIL: " << error.what() << '\n';
    return 1;
  }
}

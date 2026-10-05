// Sweeps the GGUF staged decode kernels (gguf_decode_*_m* of
// shared/gguf_linear.metal) over the production projection shapes and K-split
// counts. Every split streams the same weight bytes, so effective GB/s is the
// tuning signal — the image planes are byte-identical per dispatch.
#include "../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/QuantFormat.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using richengine::metal::BufferBinding;
using richengine::metal::BufferStorage;
using richengine::metal::ComputeDispatch;
using richengine::metal::MetalBackend;
using richengine::metal::MetalBuffer;

struct Shape final {
  const char *label;
  uint32_t outputSize;
  uint32_t inputSize;
  uint32_t formatId;
};

// A fused projection's segments (gguf_decode_fused_m*: one grid over the
// column tiles of all segments, one split count for all).
struct FusedShape final {
  const char *label;
  uint32_t cols[3];
  uint32_t inputSize;
  uint32_t formatId;
};

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

struct Image final {
  MetalBuffer plane0;
  MetalBuffer plane1;
  MetalBuffer meta;
  uint64_t bytes = 0;
};

// Sizes the planes the way the image planner does: [N/T][K/32][T][plane bytes]
// and meta [N/T][K/(32*meta_groups)][T][meta_bytes].
Image allocateImage(MetalBackend &backend, const Shape &shape,
                    const std::string &label) {
  const QuantFormat &f = kQuantFormats[shape.formatId];
  const uint64_t tiles = shape.outputSize / QUANT_TILE_ROWS;
  const uint64_t groups = shape.inputSize / 32;
  Image image{
      backend.allocateBuffer(tiles * groups * QUANT_TILE_ROWS * f.plane0_bytes,
                             BufferStorage::Shared, label + " plane0"),
      MetalBuffer{},
      backend.allocateBuffer(tiles * (groups / f.meta_groups) * QUANT_TILE_ROWS *
                                 f.meta_bytes,
                             BufferStorage::Shared, label + " meta"),
  };
  if (f.plane1_bytes)
    image.plane1 = backend.allocateBuffer(
        tiles * groups * QUANT_TILE_ROWS * f.plane1_bytes,
        BufferStorage::Shared, label + " plane1");
  std::memset(image.plane0.contents(), 0x5a, image.plane0.sizeBytes());
  std::memset(image.meta.contents(), 0x3c, image.meta.sizeBytes());
  if (image.plane1)
    std::memset(image.plane1.contents(), 0x5a, image.plane1.sizeBytes());
  image.bytes = image.plane0.sizeBytes() + image.meta.sizeBytes() +
                (image.plane1 ? image.plane1.sizeBytes() : 0);
  return image;
}

void run(const std::string &metallibPath) {
  MetalBackend backend(metallibPath);
  const auto &capabilities = backend.capabilities();
  std::cout << "device=\"" << capabilities.deviceName
            << "\" apple_gpu_family=" << capabilities.appleGpuFamily << '\n';
  const Shape shapes[] = {
      // MiniCPM5-2B dense target (2048-wide): fused-QKV segments run plain,
      // attention out and FFN down run residual, the head is Q6_K.
      {"dense_qkv_q", 2'048, 2'048, GGUF_FMT_Q4K},
      {"dense_qkv_kv", 512, 2'048, GGUF_FMT_Q4K},
      {"dense_attn_out", 2'048, 2'048, GGUF_FMT_Q4K},
      {"dense_attn_out_q6k", 2'048, 2'048, GGUF_FMT_Q6K},
      {"dense_qkv_q6k", 2'048, 2'048, GGUF_FMT_Q6K},
      {"dense_kv_q6k", 512, 2'048, GGUF_FMT_Q6K},
      {"dense_up_q6k", 3'072, 2'048, GGUF_FMT_Q6K},
      {"dense_down_q6k", 2'048, 6'144, GGUF_FMT_Q6K},
      {"dense_up", 3'072, 2'048, GGUF_FMT_Q4K},
      {"dense_down", 2'048, 6'144, GGUF_FMT_Q4K},
      {"dense_head", 130'560, 2'048, GGUF_FMT_Q6K},
      // LFM2.5-2.6B (2048-wide).
      {"lfm_qkv_q", 2'048, 2'048, GGUF_FMT_Q4K},
      {"lfm_qkv_kv", 1'024, 2'048, GGUF_FMT_Q4K},
      {"lfm_down", 2'048, 10'752, GGUF_FMT_Q4K},
      {"lfm_head", 128'000, 2'048, GGUF_FMT_Q6K},
      // Granite-4.2-3B (2560-wide): gate and up share one packed projection,
      // K is Q4_K while V is Q6_K, down and the head are Q6_K.
      {"g3_q", 2'560, 2'560, GGUF_FMT_Q4K},
      {"g3_out", 2'560, 2'560, GGUF_FMT_Q4K},
      {"g3_k", 512, 2'560, GGUF_FMT_Q4K},
      {"g3_v", 512, 2'560, GGUF_FMT_Q6K},
      {"g3_up", 8'192, 2'560, GGUF_FMT_Q4K},
      {"g3_down", 2'560, 8'192, GGUF_FMT_Q6K},
      {"g3_head", 100'352, 2'560, GGUF_FMT_Q6K},
      // Granite-4.2-8B (4096-wide).
      {"g8_q", 4'096, 4'096, GGUF_FMT_Q4K},
      {"g8_out", 4'096, 4'096, GGUF_FMT_Q4K},
      {"g8_k", 1'024, 4'096, GGUF_FMT_Q4K},
      {"g8_v", 1'024, 4'096, GGUF_FMT_Q6K},
      {"g8_up", 12'800, 4'096, GGUF_FMT_Q4K},
      {"g8_down", 4'096, 12'800, GGUF_FMT_Q6K},
      {"g8_head", 100'352, 4'096, GGUF_FMT_Q6K},
  };
  const FusedShape fused[] = {
      // MiniCPM5-2B: q + kv + kv (16 query heads, 2 KV heads x 128).
      {"dense_qkv", {2'048, 256, 256}, 2'048, GGUF_FMT_Q4K},
      // LFM2.5-2.6B: q + kv + kv (2 KV heads x 128 on the 2048 model).
      {"lfm_qkv", {2'048, 512, 512}, 2'048, GGUF_FMT_Q4K},
      // Granite-4.2-3B: q + k + v (8 KV heads of 64) and gate + up; its V is
      // Q6_K while the sweep's segments share one format, so the QKV numbers
      // read slightly slow against the mixed projection.
      {"g3_qkv", {2'560, 512, 512}, 2'560, GGUF_FMT_Q4K},
      {"g3_gateup", {8'192, 8'192}, 2'560, GGUF_FMT_Q4K},
      // Granite-4.2-8B: q + k + v (8 KV heads of 128) and gate + up.
      {"g8_qkv", {4'096, 1'024, 1'024}, 4'096, GGUF_FMT_Q4K},
      {"g8_gateup", {12'800, 12'800}, 4'096, GGUF_FMT_Q4K},
  };
  const struct {
    const char *kernel;
    uint32_t rows;
  } tiles[] = {
      {"gguf_decode_%s_m8_a", 8},
      {"gguf_decode_%s_m16_a", 16},
      {"gguf_decode_%s_m32_a", 32},
  };
  for (const Shape &shape : shapes) {
    const QuantFormat &f = kQuantFormats[shape.formatId];
    if (shape.outputSize % QUANT_TILE_ROWS) {
      std::cout << shape.label << " skipped (output not a tile multiple)\n";
      continue;
    }
    const Image image =
        allocateImage(backend, shape, shape.label);
    for (const auto &tile : tiles) {
      char name[64];
      std::snprintf(name, sizeof(name), tile.kernel, f.name);
      const uint32_t columnTiles = shape.outputSize / GGUF_TILE_COLUMNS;
      MetalBuffer input = backend.allocateBuffer(
          uint64_t{tile.rows} * shape.inputSize * sizeof(__bf16),
          BufferStorage::Shared, "input");
      MetalBuffer output = backend.allocateBuffer(
          uint64_t{tile.rows} * shape.outputSize * sizeof(__bf16),
          BufferStorage::Shared, "output");
      MetalBuffer partials = backend.allocateBuffer(
          uint64_t{8} * tile.rows * shape.outputSize * sizeof(float),
          BufferStorage::Shared, "partials");
      MetalBuffer counters = backend.allocateBuffer(
          uint64_t{columnTiles} * sizeof(uint32_t), BufferStorage::Shared,
          "counters");
      for (uint32_t splits = 1; splits <= 8; ++splits) {
        if (shape.inputSize / GGUF_STAGED_STEP / splits == 0) continue;
        // Whole splits per group partition; uneven splits are allowed but the
        // policy only takes divisors.
        if ((shape.inputSize / GGUF_STAGED_STEP) % splits) continue;
        ComputeDispatch dispatch;
        dispatch.pipelineName = name;
        const MetalBuffer &plane1 = image.plane1 ? image.plane1 : image.meta;
        dispatch.buffers = {
            {0, input},  {1, image.plane0}, {2, plane1},
            {3, image.meta}, {4, output}, {5, partials}, {6, counters},
            {7, output}};
        GgufDecodeParams params{shape.inputSize, splits, shape.outputSize, 0};
        dispatch.bytes = {{8, &params, sizeof(params)}};
        dispatch.threadgroups = {columnTiles, splits, 1};
        dispatch.threadsPerThreadgroup = {GGUF_STAGED_THREADS, 1, 1};
        for (uint32_t warmup = 0; warmup < 2; ++warmup)
          static_cast<void>(backend.submit(dispatch));
        std::vector<double> samples;
        for (uint32_t repeat = 0; repeat < 9; ++repeat)
          samples.push_back(backend.submit(dispatch).gpuSeconds);
        const double seconds = median(samples);
        std::cout << shape.label << ' ' << name << " splits=" << splits
                  << " ms=" << seconds * 1e3
                  << " GB/s=" << double(image.bytes) / seconds / 1e9 << '\n';
      }
    }
  }
  // The fused three-segment projections: one image per segment, one split
  // count for the whole dispatch, grid over every segment's column tiles.
  for (const FusedShape &fusedShape : fused) {
    uint32_t totalCols = 0;
    Image images[3]{};
    uint64_t fusedBytes = 0;
    for (uint32_t s = 0; s < 3; ++s) {
      if (!fusedShape.cols[s]) continue;
      images[s] = allocateImage(backend,
                                {fusedShape.label, fusedShape.cols[s],
                                 fusedShape.inputSize, fusedShape.formatId},
                                std::string(fusedShape.label) + ".seg" +
                                    std::to_string(s));
      fusedBytes += images[s].bytes;
      totalCols += fusedShape.cols[s];
    }
    // A two-segment shape still binds three image sets; the empty segment's
    // zero columns keep its weight planes unread.
    for (uint32_t s = 0; s < 3; ++s)
      if (!images[s].plane0) images[s] = images[0];
    for (const auto &tile : tiles) {
      char name[64];
      std::snprintf(name, sizeof(name), "gguf_decode_fused_m%u%s", tile.rows,
                    capabilities.appleGpuFamily >= 10 ? "_n" : "");
      MetalBuffer input = backend.allocateBuffer(
          uint64_t{tile.rows} * fusedShape.inputSize * sizeof(__bf16),
          BufferStorage::Shared, "input");
      MetalBuffer output = backend.allocateBuffer(
          uint64_t{tile.rows} * totalCols * sizeof(__bf16),
          BufferStorage::Shared, "output");
      MetalBuffer partials = backend.allocateBuffer(
          uint64_t{8} * tile.rows * totalCols * sizeof(float),
          BufferStorage::Shared, "partials");
      MetalBuffer counters = backend.allocateBuffer(
          uint64_t{totalCols / GGUF_TILE_COLUMNS} * sizeof(uint32_t),
          BufferStorage::Shared, "counters");
      for (uint32_t splits = 1; splits <= 8; ++splits) {
        if ((fusedShape.inputSize / GGUF_STAGED_STEP) % splits) continue;
        ComputeDispatch dispatch;
        dispatch.pipelineName = name;
        dispatch.buffers = {
            {0, input},
            {1, images[0].plane0}, {2, images[0].plane1 ? images[0].plane1 : images[0].meta}, {3, images[0].meta},
            {4, images[1].plane0}, {5, images[1].plane1 ? images[1].plane1 : images[1].meta}, {6, images[1].meta},
            {7, images[2].plane0}, {8, images[2].plane1 ? images[2].plane1 : images[2].meta}, {9, images[2].meta},
            {10, output}, {11, partials}, {12, counters}};
        GgufDecodeFusedParams params{fusedShape.inputSize, splits, totalCols,
                                     {fusedShape.cols[0], fusedShape.cols[1], fusedShape.cols[2]},
                                     {fusedShape.formatId, fusedShape.formatId, fusedShape.formatId},
                                     {0, fusedShape.cols[0], fusedShape.cols[0] + fusedShape.cols[1]}};
        dispatch.bytes = {{13, &params, sizeof(params)}};
        dispatch.threadgroups = {totalCols / GGUF_TILE_COLUMNS, splits, 1};
        dispatch.threadsPerThreadgroup = {GGUF_STAGED_THREADS, 1, 1};
        for (uint32_t warmup = 0; warmup < 2; ++warmup)
          static_cast<void>(backend.submit(dispatch));
        std::vector<double> samples;
        for (uint32_t repeat = 0; repeat < 9; ++repeat)
          samples.push_back(backend.submit(dispatch).gpuSeconds);
        const double seconds = median(samples);
        std::cout << fusedShape.label << ' ' << name << " splits=" << splits
                  << " ms=" << seconds * 1e3
                  << " GB/s=" << double(fusedBytes) / seconds / 1e9 << '\n';
      }
    }
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf_decode_sweep <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}

#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct VisionGemmParams {
  uint32_t output_size;
  uint32_t input_size;
};

static_assert(sizeof(VisionGemmParams) == 8,
              "Vision GEMM parameters are 8 bytes on both sides");

struct VisionNormParams {
  uint32_t width;
};

static_assert(sizeof(VisionNormParams) == 4,
              "Vision LayerNorm parameters are 4 bytes on both sides");

struct VisionGridParams {
  uint32_t grid_height;
  uint32_t grid_width;
};

static_assert(sizeof(VisionGridParams) == 8,
              "Vision grid parameters are 8 bytes on both sides");

// The LFM3-VL patchify's grid plus the temporal frame count its pixel
// payload carries (1 still image, 2 video pair).
struct VisionPatchifyParams {
  uint32_t grid_height;
  uint32_t grid_width;
  uint32_t frames;
};

static_assert(sizeof(VisionPatchifyParams) == 12,
              "Vision patchify parameters are 12 bytes on both sides");

struct VisionQkvParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  // LFM3-VL only: nonzero when the q/k/v inputs are one fused
  // [Q | K | V]-per-token buffer produced by lfm_vision_gemm_qkv instead
  // of three separate projection outputs.
  uint32_t fused = 0;
};

static_assert(sizeof(VisionQkvParams) == 12,
              "Vision QKV parameters are 12 bytes on both sides");

struct VisionAttentionParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  float scale;
};

static_assert(sizeof(VisionAttentionParams) == 12,
              "Vision attention parameters are 12 bytes on both sides");

#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct RopeTableParams {
  uint32_t target_rows;
  uint32_t draft_rows;
  // Rotary pairs each target row's table holds, and the positions a row
  // carries: 3 for the Qwen3.5 M-RoPE (dim % 3 chooses the axis), 1 for the
  // dense and LFM2 targets' single position axis.
  uint32_t target_dims;
  uint32_t target_axes;
  // Rotary pairs each draft row's table holds: the draft's head dimension
  // over two. Hardcoded 64 broke the 64-dim DSpark drafts' context commits.
  uint32_t draft_dims;
  // What the draft table's cosines and sines are scaled by: YaRN's
  // attention factor (model::yarnAttentionFactor), 1 without YaRN.
  float draft_scale;
};

static_assert(sizeof(RopeTableParams) == 24,
              "RoPE table parameters are 24 bytes on both sides");

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
};

static_assert(sizeof(RopeTableParams) == 16,
              "RoPE table parameters are 16 bytes on both sides");

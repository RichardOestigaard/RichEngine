#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The LFM2 short convolution's packed input width and row state.
// `packed` rows hold [B | C | x] chunks of `dimension` each (the in_proj
// output); `state` holds the previous taps-1 tokens' B*x elements per layer
// and lane. `taps_major` records the conv weight's layout: [tap][channel]
// when set, [channel][tap] otherwise — the order every current source
// stores (a GGUF's squeezed [dim, 1, taps] HF tensor, packed and MLX).
struct RichLfmConvParams {
  uint32_t rows;            // rows this sequence or step computes
  uint32_t dimension;       // conv channels (the hidden size)
  uint32_t taps;            // kernel taps (3)
  uint32_t taps_major;      // weights are [tap][channel] when set
  uint64_t state_layer_bytes; // one layer's conv state bytes per lane
  uint32_t layer;           // the first recurrent layer's state slot
  uint32_t lanes;           // verify lanes in the dispatch
  uint32_t layers;          // conv layers batched in one commit dispatch
  uint64_t mixed_layer_stride; // bytes between layers' `mixed` blocks
};

typedef struct RichLfmConvParams RichLfmConvParams;

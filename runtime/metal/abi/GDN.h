#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The packed width, the value heads and the packed, mixed and gate row
// strides of the GDN kernels are constants of their compiled variant and the
// grids give the tasks; the state cells' strides come from the host.

// The prefill prepare and scan kernels.
struct GDNPrefillParams {
  uint32_t tokens;
};

static_assert(sizeof(GDNPrefillParams) == 4,
              "GDN prefill parameters are 4 bytes on both sides");

// The chunkwise-parallel (WY/UT) scan kernels in prefill/gdn_chunked.metal,
// dispatched when a GdnPrefillBuffers.chunkScratch is bound. The chunk
// factor C is compiled into the kernel name (gdn_chunked_*_cC).
struct GDNChunkedParams {
  uint32_t tokens;
  uint32_t key_heads;
  uint32_t value_heads;
  uint32_t chunks; // ceil(tokens / C)
};

static_assert(sizeof(GDNChunkedParams) == 16,
              "GDN chunked parameters are 16 bytes on both sides");

// tiled_heads (0 or 1) selects the value-head order of the GDN output, the
// out_proj input columns: 0 keeps a key head's value heads adjacent (head h
// at h); 1 is llama.cpp's tiled GGUF order, head h at
// (h % heads per key) * key heads + h / heads per key.
struct GDNGatePrefillParams {
  uint32_t tiled_heads;
};

static_assert(sizeof(GDNGatePrefillParams) == 4,
              "GDN prefill gate parameters are 4 bytes on both sides");

struct GDNDecodeBatchParams {
  uint32_t tiled_heads; // As in GDNGatePrefillParams.
  uint32_t layer;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  // Adaptive proposal budgets (RICHENGINE_ADAPTIVE_PROPOSALS): each lane's live
  // verify rows bound the serial scan. Zero or RICHENGINE_TARGET_VERIFY_ROWS
  // scans all eight rows; the commit replays only retained rows regardless.
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(GDNDecodeBatchParams) == 48,
              "GDN decode parameters are 48 bytes on both sides");

struct GDNBatchCommitParams {
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
};

static_assert(sizeof(GDNBatchCommitParams) == 24,
              "GDN commit parameters are 24 bytes on both sides");

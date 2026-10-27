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

// flags: GDN_TILED_HEADS selects the value-head order of the GDN output, the
// out_proj input columns: without it a key head's value heads stay adjacent
// (head h at h); with it, llama.cpp's tiled GGUF order puts head h at
// (h % heads per key) * key heads + h / heads per key. GDN_SIGMOID_GATE gates
// the normalized output by sigmoid(z) (qwen4exp) instead of silu(z).
#define GDN_TILED_HEADS 1u
#define GDN_SIGMOID_GATE 2u
struct GDNGatePrefillParams {
  uint32_t flags;
};

static_assert(sizeof(GDNGatePrefillParams) == 4,
              "GDN prefill gate parameters are 4 bytes on both sides");

struct GDNDecodeBatchParams {
  uint32_t flags; // As in GDNGatePrefillParams.
  uint32_t layer;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  // Adaptive proposal budgets (RICHENGINE_ADAPTIVE_PROPOSALS): each lane's live
  // verify rows bound the serial scan. Zero or RICHENGINE_TARGET_VERIFY_ROWS
  // scans all eight rows; the commit replays only retained rows regardless.
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The packed lane stride of the verify tensors: RICHENGINE_TARGET_VERIFY_ROWS
  // unless the draft's trained block is shorter (draft.verifyRows()).
  uint32_t verify_stride;
  // The committed-state fold: per lane, the byte offset of the previous
  // step's retained rows in the tape buffer (their k/v columns, decay and
  // beta as the decode tensors pack them) and how many the lane kept. The
  // scan advances the state over them without emitting; a lane with zero
  // retains none. GDN_TAPE_FOLD keeps the standalone commit when off.
  uint64_t pending_tape[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t pending_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The lane's current-step tape region: the fused kernel's prologue leaves
  // the step's rows (conv inputs, kv, decay, beta) there for the next step's
  // pending prefix. Parity alternates with the state slots.
  uint64_t step_tape[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(GDNDecodeBatchParams) == 136,
              "GDN decode parameters are 136 bytes on both sides");

struct GDNBatchCommitParams {
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  // The packed lane stride of the commit's packed tensors: a chain batch's
  // verify stride, RICHENGINE_TREE_VERIFY_NODES for a tree lane.
  uint32_t verify_stride;
};

static_assert(sizeof(GDNBatchCommitParams) == 32,
              "GDN commit parameters are 32 bytes on both sides");

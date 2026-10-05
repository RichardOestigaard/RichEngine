#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/KvExtent.h"

// stride: rows of one KV head's (and one query group's) chunk staging;
// equals the store's chunk_stride.
struct FullPrefillParams {
  uint32_t tokens;
  uint32_t stride;
};

static_assert(sizeof(FullPrefillParams) == 8,
              "Full attention prefill parameters are 8 bytes on both sides");

// Each lane holds `rows` of its verify rows in a chunk staging of
// RICHENGINE_VERIFY_CHUNK_STRIDE rows: RICHENGINE_TARGET_VERIFY_ROWS in chain mode,
// RICHENGINE_TREE_VERIFY_NODES in tree mode.
struct FullDecodeBatchParams {
  uint32_t lanes;
  uint32_t rows;
};

static_assert(sizeof(FullDecodeBatchParams) == 8,
              "Full attention verify parameters are 8 bytes on both sides");

// One full-attention layer's paged KV. Every current row is written directly
// into its final page slot before attention. Prefill and verify both read
// all visible history from the same paged representation. Decode accepts rows
// only by advancing committed_tokens; the next command overwrites rejected
// slots. Page tables hold one RichKvPage per logical page, and kv places the
// layer in the pool's extents; the host sets it for each layer it encodes.
struct RichChunkedPrefillParams {
  uint32_t committed_tokens;
  uint32_t chunk_tokens;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  RichKvLayer kv;
};

static_assert(sizeof(RichChunkedPrefillParams) == 24,
              "Chunked store parameters are 24 bytes on both sides");

// Prefill divides each query tile's visible Page32 history into balanced
// splits. The same count and partition rule are used by split and reduce.
struct RichPrefillAttentionParams {
  uint32_t committed_tokens;
  uint32_t rows;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  RichKvLayer kv;
  uint32_t split_count;
  // The q*k softmax scale: zero selects the head dimension's 1/sqrt(d).
  float score_scale;
};

static_assert(sizeof(RichPrefillAttentionParams) == 32,
              "Prefill attention parameters are 32 bytes on both sides");

// A verify lane attends all its `active_rows` live rows
// (RICHENGINE_TARGET_VERIFY_ROWS chain, up to RICHENGINE_TREE_VERIFY_NODES - 1
// tree), which its queries and output hold in a chunk staging of
// RICHENGINE_VERIFY_CHUNK_STRIDE rows per KV head. row_capacity is the tile's
// compile-time row count (8 or RICHENGINE_TREE_VERIFY_NODES) and strides its
// partial slots.
struct RichVerifyAttentionParams {
  uint32_t committed_tokens;
  uint32_t page_table_entries;
  RichKvLayer kv;
  // Filled from the plan: this lane's history-scaled split count and the
  // plan-wide slot stride that every lane's partials use.
  uint32_t split_count;
  uint32_t slot_splits;
  uint32_t active_rows;
  uint32_t row_capacity;
  // The q*k softmax scale: zero selects the head dimension's 1/sqrt(d).
  float score_scale;
};

static_assert(sizeof(RichVerifyAttentionParams) == 36,
              "Verify attention parameters are 36 bytes on both sides");

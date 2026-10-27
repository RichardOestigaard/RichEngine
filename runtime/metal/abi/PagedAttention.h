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
  // The lane stride of the rope cos/sin tables: the decode positions always
  // run the full verify capacity, even on a shorter trained-block stride.
  uint32_t rope_stride;
};

static_assert(sizeof(FullDecodeBatchParams) == 12,
              "Full attention verify parameters are 12 bytes on both sides");

// Sparse attention (kernels/shared/sparse_attention.metal,
// ops::PagedAttention::addBlockLists): a query row attends only to the blocks
// of RICHENGINE_ATTENTION_BLOCK_TOKENS tokens its block mask names, within its
// causal history; the mask is per row a bitmap over the history's blocks,
// block b at bit b % 32 of word b / 32. A tile's rows' masks become its block
// list: in ascending order each block of the tile's history that some row
// attends, as block << RICHENGINE_ATTENTION_BLOCK_ROW_BITS | a bit per row of
// the tile that attends it, in lists of block_stride entries with their counts
// apart. A sparse split reads list block_list + its tile (verify: its lane's
// eight-row tiles); without a stride a tile attends to every block of its
// history.
#define RICHENGINE_ATTENTION_BLOCK_TOKENS 4u
#define RICHENGINE_ATTENTION_BLOCK_ROW_BITS 8u
// A sparse split takes at least RICHENGINE_SPARSE_SPLIT_BLOCKS entries of its
// tile's list, and a list spreads over at least RICHENGINE_SPARSE_MINIMUM_SPLITS
// splits while it gives each a partition unit, so that a lane's split count
// follows its own list, not its history.
#define RICHENGINE_SPARSE_SPLIT_BLOCKS 48u
#define RICHENGINE_SPARSE_MINIMUM_SPLITS 32u

static_assert(RICHENGINE_MAXIMUM_PHYSICAL_KV_TOKENS / RICHENGINE_ATTENTION_BLOCK_TOKENS <
                  1u << (32 - RICHENGINE_ATTENTION_BLOCK_ROW_BITS),
              "A list entry's bits above its rows' hold any block");

// The block lists of up to RICHENGINE_MAXIMUM_BATCH_WIDTH sequences of rows (a
// verify step's lanes, or a prefill chunk's sequences), in tiles of
// `tile_rows` rows: sequence s's `rows` rows, whose block masks are rows
// `mask_row` on of the masks, `mask_words` words each, the first at position
// `first_position`, its last tile holding the rest; its tile t's entries go
// to list `list` + t, `stride` entries each. A threadgroup of
// RICHENGINE_ATTENTION_BLOCK_LIST_THREADS threads lists one tile.
#define RICHENGINE_ATTENTION_BLOCK_LIST_THREADS 256u
struct RichAttentionBlockListSequence {
  uint32_t mask_row;
  uint32_t rows;
  uint32_t first_position;
  uint32_t list;
};

struct RichAttentionBlockListParams {
  uint32_t tile_rows;
  uint32_t mask_words;
  uint32_t stride;
  uint32_t sequences;
  RichAttentionBlockListSequence sequence[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(RichAttentionBlockListParams) == 16 + 16 * RICHENGINE_MAXIMUM_BATCH_WIDTH,
              "Block list parameters are the same size on both sides");

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
  // A sparse tile's block list: tile t reads list block_list + t, of
  // block_stride entries (zero: every block of the tile's history).
  uint32_t block_stride;
  uint32_t block_list;
};

static_assert(sizeof(RichPrefillAttentionParams) == 40,
              "Prefill attention parameters are 40 bytes on both sides");

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
  // The lane's first block list, which its sparse split and reduce read:
  // eight-row tile t of the lane's rows reads list block_list + t, of
  // block_stride entries (zero: every block of the tile's history).
  uint32_t block_stride;
  uint32_t block_list;
};

static_assert(sizeof(RichVerifyAttentionParams) == 44,
              "Verify attention parameters are 44 bytes on both sides");

#pragma once

// Parameters of the Maxwell MBSA indexer (kernels/shared/mbsa_indexer.metal,
// ops/Mbsa.cpp): per-token index keys and per-KV-group routes for the
// maxwell_moe target's multi-block sparse attention.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/PagedAttention.h"

// The Maxwell indexer (HF modeling_maxwell_moe.py): MBSA_INDEX_HEADS index
// query heads — one per KV group, hence per route — and one shared index key
// head of MBSA_INDEX_DIMENSION values. Every token keeps its normalized,
// rotated index key (bf16 [token][dimension]) in its page's index plane
// (kv::Layout::indexPerToken): the plane of a Page32 page holds
// RICHENGINE_TARGET_KV_BLOCK_TOKENS keys, a token's at slot
// position % RICHENGINE_TARGET_KV_BLOCK_TOKENS. A query row at position p
// routes, per group g, to blocks of MBSA_ROUTE_BLOCK tokens: the
// MBSA_PREFIX_BLOCKS first blocks that precede its own (the prefix, blocks
// b < 8 with b <= block(p)), the up-to-MBSA_LOCAL_BLOCKS blocks ending at
// block(p) that are not prefix blocks (the local band), and the
// MBSA_REMOTE_BLOCKS highest-scoring eligible blocks (the remote band).
// Eligible blocks sit between the prefix and the local band: b >=
// MBSA_PREFIX_BLOCKS and b + 1 <= max(block(p) - MBSA_LOCAL_BLOCKS + 1, 0),
// where block(p) = p / MBSA_ROUTE_BLOCK; a block's score is the largest
// q_g . k_t * index_softmax_scale (128^-0.5) of its tokens, every one of
// which precedes p by construction. The route is at most
// MBSA_PREFIX_BLOCKS + MBSA_REMOTE_BLOCKS + MBSA_LOCAL_BLOCKS blocks.
#define MBSA_INDEX_DIMENSION 128u
#define MBSA_INDEX_HEADS 4u
// Rotary pairs of the indexer's RoPE rows: the model's rotary_pairs 64 of
// head dimension 128 (half-rotary, pairs i and i + MBSA_ROPE_PAIRS), the
// width of the cos/sin table rows the project binds.
#define MBSA_ROPE_PAIRS 64u
#define MBSA_ROUTE_BLOCK 128u
#define MBSA_PREFIX_BLOCKS 8u
#define MBSA_LOCAL_BLOCKS 8u
#define MBSA_REMOTE_BLOCKS 16u
// A route tile's rows: the sparse attention's block-list row bits, one per
// tile row.
#define MBSA_ROUTE_ROWS RICHENGINE_ATTENTION_BLOCK_ROW_BITS
// The route kernel's threads, a threadgroup per (lane, group, tile).
#define MBSA_ROUTE_THREADS 256u
// The largest 128-token block index a row can sit in, hence the most route
// candidates a tile's emit loop scans (the block index, plus its own).
#define MBSA_ROUTE_MAX_BLOCKS \
  (RICHENGINE_MAXIMUM_PHYSICAL_KV_TOKENS / MBSA_ROUTE_BLOCK + 1u)
// 4-token sparse list blocks per route block, the entries a route block
// emits.
#define MBSA_ROUTE_SUBBLOCKS (MBSA_ROUTE_BLOCK / RICHENGINE_ATTENTION_BLOCK_TOKENS)

// The index projections' input width — the target's hidden size, whose
// fp32 normed rows mbsa_index_project stages per threadgroup.
#define MBSA_INDEX_INPUT_WIDTH 2048u

// mbsa_index_project's parameters: a dispatch serves `lanes` lanes (a
// verify step's, or a prefill sequence as one), each with `rows` rows whose
// first sits at logical position first_position[l]. Each (row, head)
// threadgroup dots the row's normed input — the layer's fp32 normed rows,
// input_width values — against its slice of the layer's fp32
// linear_q_proj/linear_k_proj weight planes (the projections folded into
// the kernel: the two gguf_float dispatches and their fp32 scratch the
// split path otherwise pays), normalizes and rotates; the queries go to
// the dispatch's scratch, the keys through lane l's index table to the
// index plane, index_offset bytes into the plane region each entry names
// (the layer's index region, PageStorage::indexOffset).
struct MbsaIndexParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t first_position[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t index_offset;
  uint32_t input_width;
};

static_assert(sizeof(MbsaIndexParams) == 16 + 4 * RICHENGINE_MAXIMUM_BATCH_WIDTH,
              "MBSA index parameters are the same size on both sides");

// mbsa_route's parameters: the same lanes and positions, plus the lanes'
// live row counts (a verify lane's active rows, or the chunk's rows — the
// tiles' coverage; `rows` stays the lanes' query row stride), `row_tiles`
// MBSA_ROUTE_ROWS-row tiles per lane, and the block-list geometry the
// _k4q8d128 sparse splits read: the (lane, tile, group) list at index
// (list + lane * row_tiles + tile) * MBSA_INDEX_HEADS + group of `lists`,
// `list_stride` entries each, the count at the same index of `counts`
// (ops::PagedAttention::addVerify gives lane l's attention parameters the
// list base list + lane * row_tiles, and the kernel expands each
// block_list + tile group index into the groups' consecutive lists).
struct MbsaRouteParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t first_position[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t active_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t row_tiles;
  uint32_t list;
  uint32_t list_stride;
  uint32_t index_offset;
};

static_assert(sizeof(MbsaRouteParams) == 24 + 8 * RICHENGINE_MAXIMUM_BATCH_WIDTH,
              "MBSA route parameters are the same size on both sides");

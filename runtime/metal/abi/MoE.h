#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct MoeRouteParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t experts;
  uint32_t top_k;
};

static_assert(sizeof(MoeRouteParams) == 16,
              "MoE routing parameters are 16 bytes on both sides");

// Every row carries top_k routed experts followed by the shared expert, whose
// id is `experts` and whose routing weight is the sigmoid of its scalar gate.
// Routes are sorted by expert: tile t covers grouped rows [t * tile_rows,
// (t + 1) * tile_rows) of one expert; padding rows carry the route ~0u.
//
// Written by grouping kernels; the host uses this layout to size storage.
struct MoeTileDescriptor {
  uint32_t expert;
  uint32_t rows;
};

static_assert(sizeof(MoeTileDescriptor) == 8,
              "MoE tile descriptors are 8 bytes on both sides");

struct MoeGroupParams {
  uint32_t rows;
  uint32_t top_k;
  uint32_t tile_rows;
  uint32_t experts;
  // Whether every row's routes end in the shared expert (id `experts`),
  // whose tiles follow the routed ones. LFM2-MoE has none.
  uint32_t shared;
  // RICHENGINE_MOE_STATS record generation; 0 disables stats. The kernel
  // writes the dispatch's record into ring slot (stats - 1) %
  // kMoeStatsLogSlots of the tile-count scratch.
  uint32_t stats;
};

static_assert(sizeof(MoeGroupParams) == 24,
              "MoE grouping parameters are 24 bytes on both sides");

// RICHENGINE_MOE_STATS (ops/MoE.cpp): the grouping kernel's per-dispatch
// record, appended to a ring in the tile-count scratch at
// kMoeStatsLogOffset. tile_count[1] holds the same dispatch's distinct
// routed-expert count. The host reads the ring after the command and
// prints one "moe-stats" line per record.
enum {
  kMoeStatsLogOffset = 16,
  kMoeStatsLogSlots = 256,
};
struct MoeStatsRecord {
  uint32_t generation;
  uint32_t rows;
  uint32_t routed_experts;
  uint32_t tiles;
};

static_assert(sizeof(MoeStatsRecord) == 16,
              "MoE stats records are 16 bytes on both sides");

// The fused select-and-group dispatch (moe_route_group_sigmoid) takes the
// routing and grouping parameters together.
struct MoeRouteGroupParams {
  MoeRouteParams route;
  MoeGroupParams group;
};

static_assert(sizeof(MoeRouteGroupParams) == 40,
              "MoE route-group parameters are 40 bytes on both sides");

// RICHENGINE_MOE_UNION (ops/MoE.cpp): the decode union cap keeps the
// `budget` routed experts with the largest summed routing weight across the
// dispatch's rows and dead-marks the rest (selected ~0u, weight 0), which
// moe_group_routes skips and moe_combine multiplies by zero.
struct MoeCapParams {
  uint32_t rows;
  uint32_t routes_per_row;
  uint32_t experts;
  uint32_t budget;
};

static_assert(sizeof(MoeCapParams) == 16,
              "MoE union-cap parameters are 16 bytes on both sides");

struct MoeGatherParams {
  uint32_t tile_rows;
  uint32_t input_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeGatherParams) == 12,
              "MoE gather parameters are 12 bytes on both sides");

struct MoeExpertParams {
  uint32_t input_size;
  uint32_t output_size;
  uint32_t experts;
  uint32_t reserved0;
  uint64_t expert_stride_bytes_0;
  uint64_t expert_stride_bytes_1;
};

static_assert(sizeof(MoeExpertParams) == 32,
              "MoE expert parameters are 32 bytes on both sides");

// A GGUF expert pass (ops/MoE.cpp): every routed expert of the projection
// is one image segment of experts * output_size rows, expert e's planes
// starting at tile e * output_size / QUANT_TILE_ROWS; the shared expert (id `experts`)
// has a segment of its own. Formats are GGUF_FMT_* (metal/abi/QuantFormat.h).
struct MoeGgufExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
  // K partitions of the packed tile (ops/MoE.cpp): grid.y is
  // tiles * splits, partition p covering K groups [p*per, (p+1)*per) — the
  // dense decode tile's contiguous split. 1 when unpacked or when the
  // segments do not all take the packed path.
  uint32_t splits;
};

static_assert(sizeof(MoeGgufExpertParams) == 24,
              "MoE GGUF expert parameters are 24 bytes on both sides");

struct MoeCombineParams {
  uint32_t rows;
  uint32_t hidden_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeCombineParams) == 12,
              "MoE combine parameters are 12 bytes on both sides");

#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The expert slots of the routing kernels (ops::MoE): a row's router
// scores take this many floats, and the select and grouping kernels run
// one thread per slot. A MoE block of at most MOE_EXPERT_SLOTS experts
// routes on their instances of that many slots, one of more on their wide
// instances (_e512), so a block routes at most MOE_WIDE_EXPERT_SLOTS
// experts.
#define MOE_EXPERT_SLOTS 256u
#define MOE_WIDE_EXPERT_SLOTS 512u

struct MoeRouteParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t experts;
  uint32_t top_k;
  // The affine shared-expert scalar gate: 0 when the shape has none, 1 when
  // the scores tile folds it in as a 257th column parked at
  // scores + rows * 256 for the select to read, 2 when the select re-dots
  // the input against the gate planes (RICHENGINE_MOE_FOLDED_GATE_OFF).
  uint32_t shared;
  // The routed scaling factor applied after norm_topk_prob (NVIDIA
  // Nemotron's expert_weights_scale 2.5); 1.0 leaves the weights as the
  // winners' normalized sigmoid scores.
  float weight_scale = 1.0f;
};

static_assert(sizeof(MoeRouteParams) == 24,
              "MoE routing parameters are 24 bytes on both sides");

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

// A verify step's grouping (moe_group_routes_verified) routes only each
// lane's verify rows (the draft selector's, VerifyRowsParams in abi/Sampling.h):
// row r is row r % lane_rows of lane r / lane_rows, and the routed routes of
// a row past its lane's verify rows stay out of every tile with the route row
// ~0u, which the combine skips. The shared expert's tiles still hold every
// row in order. The other groupings route every row and ignore lane_rows.
// The staged GGUF expert passes (kernels/shared/moe_gguf.metal) run an
// expert's only tile of at most this many live rows on the column tile, which
// reads those rows alone, and every other tile on the staged tile's matmuls
// (moe_column_tile): a route's output depends on its expert's rows in the
// step.
#define MOE_GGUF_COLUMN_ROWS 2u

// Whether the staged expert passes run tile t of `tiles`, the grouping's
// MoeTileDescriptors (on the GPU in device memory), on the column tile: at
// most MOE_GGUF_COLUMN_ROWS live rows, its expert's only tile. The kernels,
// the MoE test's expected tile kinds and the MoE benchmark's share of column
// tiles all ask this.
template <typename Tiles>
inline bool moe_column_tile(Tiles tiles, uint32_t t) {
  return tiles[t].rows <= MOE_GGUF_COLUMN_ROWS && (t == 0 || tiles[t - 1].expert != tiles[t].expert);
}

// The column tiles of the gate/up and down passes, whose grids the grouping
// writes (MoeTileCount).
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
  // A verify step's packed lane stride (the draft's verify stride, at most
  // RICHENGINE_TARGET_VERIFY_ROWS): lane l's block is rows
  // [l * lane_rows, (l + 1) * lane_rows). Zero on a non-verify dispatch,
  // which never runs the verified grouping.
  uint32_t lane_rows;
  // RICHENGINE_MOE_INDIRECT_DISPATCH (ops/MoE.cpp): nonzero gate/up and
  // down column counts make the grouping kernel also write the expert
  // passes' indirect dispatch grids after the live tile count — three
  // words {columns or tiles, tiles or columns, 1} per pass at
  // kMoeGateUpGridOffset and kMoeDownGridOffset bytes into the tile-count
  // scratch, the tile-major flag selecting which axis the live count
  // takes. Zero leaves those words untouched.
  uint32_t gate_up_columns = 0;
  uint32_t down_columns = 0;
  uint32_t tile_major = 0;
};

static_assert(sizeof(MoeGroupParams) == 40,
              "MoE grouping parameters are 40 bytes on both sides");

// Byte offsets in the tile-count scratch (ops/MoE.cpp): word zero holds
// the live tile count; the grouping writes the expert passes' indirect
// dispatch grids (kMoeDispatchGridBytes each) at kMoeGateUpGridOffset and
// kMoeDownGridOffset. kMoeIndirectTileCountBytes is the scratch size that
// layout needs. RICHENGINE_MOE_STATS's per-dispatch record ring follows
// at kMoeStatsLogOffset, slot (stats - 1) % kMoeStatsLogSlots; the host
// reads the ring after the command and prints one "moe-stats" line per
// record.
enum {
  kMoeGateUpGridOffset = 4,
  kMoeDownGridOffset = 16,
  kMoeDispatchGridBytes = 12,
  kMoeIndirectTileCountBytes = 28,
  kMoeStatsLogOffset = 32,
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

static_assert(sizeof(MoeRouteGroupParams) == 64,
              "MoE route-group parameters are 64 bytes on both sides");

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

// The dense shared-expert dispatches (moe_shared_*_q4, ops/MoE.cpp): one
// row tile per lane block streams the shared slab once instead of once per
// tile_rows rows. The grouping keeps each row's shared route aimed at the
// grouped row past the routed tiles (tile_count * tile_rows + row), where
// these kernels write and moe_combine reads.
struct MoeSharedParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t tile_rows;
};

static_assert(sizeof(MoeSharedParams) == 16,
              "MoE shared-expert parameters are 16 bytes on both sides");

// The dense GGUF shared-expert passes (moe_shared_gguf_*, kernels/shared/
// moe_gguf.metal): the same dispatch pattern as MoeSharedParams'
// moe_shared_*_q4 for a quantized shared slab whose intermediate width can
// differ from the routed experts' (Nemotron's is twice), so it cannot ride
// the routed passes' MoeGgufExpertParams. With the grouping's shared == 2
// every row's shared route aims at grouped row tile_count * tile_rows + row:
// the up pass reads the row's input at row `row` (input_region 0) and stores
// relu(up)^2 at tile_count * tile_rows + row of the intermediate's private
// region, which begins tile_count * tile_rows * input_region elements in —
// so the down pass, whose input_region is the routed experts' intermediate
// width, reads the same rows. Its output_region is likewise the routed
// region's row width in its output (expertOutput's hiddenSize).
struct MoeGgufSharedParams {
  uint32_t rows;
  uint32_t input_size;  // the pass's K
  uint32_t output_size; // the shared expert's own N
  uint32_t tile_rows;
  uint32_t format;
  // Element width of the routed region the input buffer leads with; 0 reads
  // the pass input ungrouped at row `row`.
  uint32_t input_region;
  // Element width of the routed region the output buffer leads with (the
  // shared rows follow at tile_count * tile_rows * output_region + row *
  // output_size).
  uint32_t output_region;
  // Nonzero applies relu(v)^2 to the bf16-rounded sum (the up pass); zero
  // stores it plain (the down pass).
  uint32_t relu2;
  // Nonzero reads a zero-padded copy of a ragged final row block and keeps
  // it on the full staged tile, preserving prefill's composition-invariant
  // summation shape. Zero keeps the sparse column tile.
  uint32_t force_staged;
};

static_assert(sizeof(MoeGgufSharedParams) == 36,
              "MoE GGUF shared-expert parameters are 36 bytes on both sides");

struct MoePadRowsParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t tile_rows;
  // Element width of the routed region preceding the dense shared rows;
  // zero means the source rows start at the input buffer's first row.
  uint32_t input_region;
};

static_assert(sizeof(MoePadRowsParams) == 16,
              "MoE row-padding parameters are 16 bytes on both sides");

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
  // Nonzero when the host swapped the dispatch grid's axes
  // (RICHENGINE_MOE_TILE_MAJOR): group arrives as (tile, column block)
  // instead of (column block, tile). Decode expert tiles only.
  uint32_t tile_major;
  uint64_t expert_stride_bytes_0;
  uint64_t expert_stride_bytes_1;
};

static_assert(sizeof(MoeExpertParams) == 40,
              "MoE expert parameters are 40 bytes on both sides");

// A GGUF expert pass (ops/MoE.cpp): every routed expert of the projection
// is one image segment of experts * output_size rows, expert e's rows
// starting at row e * output_size of the planes' tile-major image — not
// necessarily a QUANT_TILE_ROWS boundary (Nemotron's 1856), which the
// kernels' segment row0 carries to their column addressing
// (kernels/common/moe_expert_slab.h); the shared expert (id `experts`)
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
  // Nonzero when the host swapped the dispatch grid's axes
  // (RICHENGINE_MOE_TILE_MAJOR): the row index (tile * splits + split)
  // arrives on group.x and the column block on group.y.
  uint32_t tile_major;
  // Nonzero forces the staged matmul where a sparse tile would otherwise
  // take the column-dot path. Prefill uses this for composition-invariant
  // summation order; decode keeps the faster column tiles.
  uint32_t force_staged;
};

static_assert(sizeof(MoeGgufExpertParams) == 32,
              "MoE GGUF expert parameters are 32 bytes on both sides");

// The streamed GGUF expert pass (moe_expert_gguf_streamed_*): the routed
// experts' weights are not resident — the tile's expert resolves through
// the arena's slot table (expert -> global slot, ~0u while unstaged) and
// chunk address table (chunk -> the nine region GPU addresses, the staged
// convention of runtime/qwen4/metal/abi/MoE.h: projection z's plane0,
// plane1 and meta at z * 3 + p, a format without plane1 holding its meta
// region's address in plane1's entry). A global slot below floor_slots
// lies in chunk 0 at its own index; past it, chunk
// 1 + (slot - floor_slots) / chunk_slots at the remainder. The shared
// expert (id `experts`) keeps its resident segment, bound in the sw*
// slots. projection is the pass's 0 gate, 1 up, 2 down.
#define MOE_GGUF_STREAM_REGIONS 9u

struct MoeGgufStreamParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
  uint32_t floor_slots;
  uint32_t chunk_slots;
  uint32_t projection;
  uint32_t reserved;
  // Matches MoeGgufExpertParams: nonzero keeps sparse prefill tiles on the
  // staged matmul instead of the row-count-dependent column-dot path.
  uint32_t force_staged;
};

static_assert(sizeof(MoeGgufStreamParams) == 40,
              "MoE GGUF streamed expert parameters are 40 bytes on both sides");

// The fused gate+up pass (moe_expert_gguf_*_gu): one dispatch computes both
// projections of a tile and stores silu(gate) * up — the split passes'
// EpUpWithGate association without the intermediate store. Each segment
// selects its own routed/shared format pair.
struct MoeGgufFusedParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t gate_routed_format;
  uint32_t gate_shared_format;
  uint32_t up_routed_format;
  uint32_t up_shared_format;
  uint32_t reserved;
  // Nonzero forces the staged matmul where a sparse tile would otherwise
  // take the column-dot path, matching prefill's composition-invariant mode.
  uint32_t force_staged;
  // Source row divisor for the indirect column tiles
  // (moe_expert_gguf_*_gu_i): grouped route values are row * routes_per_row +
  // slot. Unused by the gathered kernels.
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeGgufFusedParams) == 40,
              "MoE GGUF fused parameters are 40 bytes on both sides");

struct MoeCombineParams {
  uint32_t rows;
  uint32_t hidden_size;
  uint32_t routes_per_row;
};

static_assert(sizeof(MoeCombineParams) == 12,
              "MoE combine parameters are 12 bytes on both sides");

// moe_gather_int8 (kernels/shared/moe_int8.metal, ops/MoE.cpp): the grouped
// rows' int8 codes and scales from the chunk's — rows of input_size codes
// and scales [input_size / 64][rows] — so every expert tile is a dense
// matrix of its live rows. Tile t covers grouped rows
// [t * tile_rows, t * tile_rows + rows) of one expert, and `routes` is the
// grouped rows' bound (tiles * tile_rows), the grouped scales' row stride.
struct MoeGatherInt8Params {
  uint32_t input_size;
  uint32_t routes_per_row;
  uint32_t rows;      // the chunk's, its scales' row stride
  uint32_t routes;    // the grouped rows', their scales' row stride
  uint32_t tile_rows;
  uint32_t reserved;
};

static_assert(sizeof(MoeGatherInt8Params) == 24,
              "MoE int8 gather parameters are 24 bytes on both sides");

// An int8 expert pass of a prefill chunk (kernels/shared/moe_int8.metal,
// ops/MoE.cpp): the grouped rows' int8 codes and scales (the dense
// prefill's rule, per row and 64 inputs, kernels/common/int8_activations.h),
// a row per route and `routes` the scales' row stride, through the AF
// planes of the tile's expert as MoeGgufExpertParams reaches them
// (moe_gguf_segment_fmt). Resident blocks only — the streamed passes'
// slot-table resolution has no int8 form.
struct MoeInt8ExpertParams {
  uint32_t input_size;
  uint32_t output_size; // per expert
  uint32_t experts;
  uint32_t routed_format;
  uint32_t shared_format;
  uint32_t tile_rows;
  uint32_t routes;
  uint32_t reserved;
};

static_assert(sizeof(MoeInt8ExpertParams) == 32,
              "MoE int8 expert parameters are 32 bytes on both sides");

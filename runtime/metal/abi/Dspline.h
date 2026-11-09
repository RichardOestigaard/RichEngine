#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The DSpline selected-prefix code bridge's elementwise phases
// (kernels/decode/dspline_bridge.metal): the trained write after every
// draft layer in the checkpoint's feature-layer mask. The five-step write
// mirrors sglang's DSpineCodeBridge fused kernels rounding point by
// rounding point: per-row RMS of the layer output with the fp32-tiny floor,
// the per-layer read projection and L2 code normalization at sqrt(code),
// the anchor row's codebook substitution, the predecessor-shifted sigmoid
// message over the packed [gate; predecessor_gate; prev] projection, and
// the RMS-scaled write-back into the residual stream. `block` is the
// lanes' packed query-row stride: rows of one lane's block are
// [lane * block, lane * block + block), and every block's row 0 is its
// anchor, which takes the anchor token's codebook row, sends no message
// and receives no write.
struct DsplineBridgeParams {
  uint32_t rows;
  uint32_t block;
  uint32_t hidden;
  uint32_t code;
  uint32_t rank;
  float eps;
};

static_assert(sizeof(DsplineBridgeParams) == 24,
              "DSpline bridge parameters are 24 bytes on both sides");

// The bridge's three bf16 projections (read [code][hidden], packed
// [3*rank][code] and feature write [hidden][rank]) share one matmul:
// out[m][n] = bf16(sum_k in[m][k] * W[n][k]) with fp32 accumulation.
// `segment` splits the output columns among the bound weight buffers —
// rank for the packed projection (three [rank][code] segments), output
// for a single weight. Column tiles are 64 wide and never straddle a
// segment boundary, so segment and output are multiples of 64.
struct DsplineMatmulParams {
  uint32_t rows;
  uint32_t output;
  uint32_t input;
  uint32_t segment;
};

static_assert(sizeof(DsplineMatmulParams) == 16,
              "DSpline matmul parameters are 16 bytes on both sides");

// The packed projection's fused variant (dspline_bridge_pack): one
// threadgroup computes a 64-column slice of all three [rank][code]
// segments — the gate, predecessor-gate and prev rows of one column tile
// share the same input rows, so a single K loop feeds three accumulators
// — then emits dspline_bridge_message's predecessor-shifted sigmoid for
// the slice from the tiles it just computed, in the same dispatch. block
// is the lanes' packed query-row stride, as in DsplineBridgeParams; the
// host only dispatches it when every 16-row tile base is a block anchor
// (16 % block == 0), so a tile never needs a predecessor row it did not
// compute.
struct DsplinePackParams {
  uint32_t rows;
  uint32_t input;
  uint32_t rank;
  uint32_t block;
};

static_assert(sizeof(DsplinePackParams) == 16,
              "DSpline pack parameters are 16 bytes on both sides");

// The feature write's fused variant (dspline_bridge_write): the same
// out[m][n] = bf16(sum_k in[m][k] * W[n][k]) projection, but instead of
// storing the delta tile its epilogue applies the RMS-scaled write-back
// in place — hidden[m][n] += rms[m] * bf16(acc[m][n]) on non-anchor rows
// (row % block != 0) — and, when `stash` is set on the last write, also
// emits the refinement's raw * final_norm rows into `rawnorm` before the
// update lands. output is the residual stream's width (hidden), input
// the message width (rank), segment the weight split as in
// DsplineMatmulParams.
struct DsplineWriteParams {
  uint32_t rows;
  uint32_t output;
  uint32_t input;
  uint32_t segment;
  uint32_t block;
  uint32_t stash;
};

static_assert(sizeof(DsplineWriteParams) == 24,
              "DSpline write parameters are 24 bytes on both sides");

// The serial last-write refinement (upstream code_bridge_second_pass,
// mode serial_table): dspline_refine_rows rescores each mask row's merged
// top-16 under every predecessor lane — the anchor for the block's first
// mask row, the previous mask row's candidates after — into an
// [positions][16][16] fp32 score table in the selector's partialValues
// tail, and dspline_refine_walk walks the chain serially into the
// draft_select_plain contract. positions is the lane's mask rows
// (query_rows - 1). The top-16 partials are global — they run over every
// lane's rows — so scores_base is the float offset past them all and
// scores_lane each lane's tail-block stride: the score table, then, for
// lanes in sampling_mask, rmsref_base's [positions][16] refined-RMS
// denominators (dspline_refine_norm, the upstream serial_rs_table
// normalization) and the shared rw_base [rows][rank] bf16 projection of
// the staged raw*norm rows onto the write weights. The walk's greedy
// branch argmaxes the raw score rows; its sampled branch divides them by
// the refined RMS and draws under the lane's joint min-p/top-k/top-p
// filter at its temperature (upstream serial_sample_chain).
struct DsplineRefineParams {
  uint32_t lanes;
  uint32_t positions;
  uint32_t query_rows;
  uint32_t rank;
  uint32_t logits_stride;
  uint32_t scores_lane;
  uint32_t scores_base;
  uint32_t rmsref_base;
  uint32_t rw_base;
  uint32_t hidden;
  uint32_t sampling_mask;
  float norm_eps;
  uint32_t top_k[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float temperature[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float top_p[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  float min_p[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DsplineRefineParams) == 112,
              "DSpline refine parameters are 112 bytes on both sides");

// The refinement's gather-column head pass (dspline_head_gather): the
// unnormalized base logits dspline_refine_rows consumes are the staged
// raw * final_norm rows against each mask row's merged top-16 head
// columns only, so the second head pass computes just those columns of
// the affine Q4 vocabulary projection instead of every output.
// positions, query_rows and logits_stride mirror DsplineRefineParams;
// input is the head's input width (the draft's hidden size, a multiple
// of the 64-input quant group).
struct DsplineGatherParams {
  uint32_t positions;
  uint32_t query_rows;
  uint32_t input;
  uint32_t logits_stride;
};

static_assert(sizeof(DsplineGatherParams) == 16,
              "DSpline gather parameters are 16 bytes on both sides");

// The load-time table kernels' payload (dspline_table_dequant): rows of
// the quantized vocabulary head to unpack into `hidden`-wide bf16 rows.
struct DsplineTableParams {
  uint32_t rows;
  uint32_t hidden;
};

static_assert(sizeof(DsplineTableParams) == 8,
              "DSpline table parameters are 8 bytes on both sides");

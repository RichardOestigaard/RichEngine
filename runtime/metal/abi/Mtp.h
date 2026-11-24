#pragma once

// The scalar payloads of the maxwell-moe MTP draft's kernels
// (kernels/decode/mtp.metal). All are packed u32 records the encode fills
// by value; nothing here exceeds 48 bytes.

#ifdef __METAL__
#include <metal_stdlib>
using namespace metal;
#else
#include <cstdint>
#endif

// mtp_gather_tokens: the flat token list of the eh rows' embedding half.
// Mode 0 (decode anchor): tokens[lane] = input[lane * stride], the packed
// draft input's anchor row. Mode 2 (prefill): tokens[row] =
// input[min(offset + row + 1, end - 1)], the rolled prompt token; the
// kv-store skips the row the clamp repeats.
struct MtpGatherTokensParams {
  uint32_t mode;
  uint32_t stride;
  uint32_t offset;
  uint32_t end;
  uint32_t rows;
};

// mtp_ehtoken / mtp_ehtoken_chain: eh[row][s] = [embedNormed[row] |
// hnorm(stream_s)] per stream, the fused eh projection's bf16 input.
// `source` picks the stream state: 0 the bf16 captured rows at
// sourceBase + row (commit and prefill rows alike), 1 the per-lane
// MtpAnchor bf16 row (decode pass 0); the chain variant reads the fp32
// streams the previous pass left.
struct MtpEhTokenParams {
  uint32_t hidden;
  uint32_t streams;
  uint32_t rows;
  uint32_t source;      // 0 captured rows, 1 anchor rows
  uint32_t sourceBase;  // mode 0's first captured row
};

// mtp_stream_init: the eh projection's bf16 output [rows][streams][hidden]
// into the fp32 streams [rows][streams][hidden].
struct MtpStreamInitParams {
  uint32_t width;
  uint32_t rows;
};

// mtp_kv_store: normed + rotated keys and raw values into the lanes'
// rings; decode mode also emits the normed + rotated query rows.
// Modes: 0 decode pass (slot = positions[lane * posStride] + pass,
// query heads run too), 1 commit rows (slot = startPosition[lane] +
// row % stride, live while row % stride < retained[lane]), 2 unused —
// prefill runs mtp_kv_store_prefill.
struct MtpKvStoreParams {
  uint32_t rows;
  uint32_t lanes;
  uint32_t heads;
  uint32_t kvHeads;
  uint32_t headDim;
  uint32_t rotaryPairs;
  float ropeTheta;
  uint32_t ring;
  uint32_t mode;
  uint32_t stride;    // mode 1: the verify row stride (16)
  uint32_t posStride; // mode 0: the draft positions lane stride
  uint32_t pass;      // mode 0: the decode pass
  float rmsEps;
  // Mode 1's per-lane first committed slot (the host span, which has no
  // device row; RICHENGINE_MAXIMUM_BATCH_WIDTH entries).
  uint32_t startPositions[4];
};

// mtp_kv_store_prefill: one prefill span's keys and values into the span's
// ring; row r commits slot base + r, dropped when base + r >= end (the
// sequence's last row, whose rolled token is not the prompt's).
struct MtpPrefillKvParams {
  uint32_t rows;
  uint32_t kvHeads;
  uint32_t headDim;
  uint32_t rotaryPairs;
  float ropeTheta;
  uint32_t ring;
  uint32_t base;      // the first row's slot position
  uint32_t end;       // the sequence's position end
  float rmsEps;
};

// mtp_attention: the decode pass's GQA flash. One threadgroup per
// (lane, query head): the query row's head attends ring positions
// [0, positions[lane * posStride] + pass] of the lane's ring.
struct MtpAttentionParams {
  uint32_t lanes;
  uint32_t heads;
  uint32_t kvHeads;
  uint32_t headDim;
  uint32_t ring;
  uint32_t posStride;
  uint32_t pass;
};

// mtp_select_top16_sharded: draft_select_top16_sharded's pass-major
// variant — proposal position p of lane b reads logits row
// (p * lanes + b) * stride — the MTP head writes contiguous pass rows.
// DraftTop16Params carries (vocabulary, stride); lanes rides along.
struct MtpTop16Params {
  uint32_t vocabulary;
  uint32_t stride;
  uint32_t lanes;
};

// mtp_argmax: the pass logits row's greedy token into tokenIn for the
// next pass's embedding gather.
struct MtpArgmaxParams {
  uint32_t lanes;
  uint32_t vocabulary;
  uint32_t stride;
};

// mtp_anchor_commit: each dispatch lane's last retained captured row into
// its anchor. mtp_anchor_prefill: one sequence's prompt tail row (bound
// already sliced) into its state lane's anchor.
struct MtpAnchorParams {
  uint32_t width;        // the streams' bf16 width (8192)
  uint32_t sourceStride; // captured rows' per-lane stride
  uint32_t lanes;
};

#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ExecutionGeometry.h"

// decode_step_pack (kernels/decode/step_pack.metal) is a committed decode
// step's last dispatch under RICHENGINE_DEVICE_VERIFY_PARAMS: it mirrors
// each lane's acceptance results into the pack block the NEXT step's
// encodes bind instead of recomputing them from a host readback of
// RetainedCount, AcceptedCount and OutputTokens. Position, cache length
// and the retention limit continue the lane's committed values; the live
// row count is the host's min(budget, maximumRetained - 1) + 1 — the
// budget and the step's own position/remaining arrive as constants
// because they are host state the kernel cannot derive.
struct DecodeStepPackParams {
  // The dispatch lanes' request ids, stamped into the pack so the next
  // step's consumers can prove the pack is their own predecessor's.
  uint64_t request_id[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t lanes;
  // The packed OutputTokens lane stride of the step being packed: the
  // chain's verify stride, the tree's node stride.
  uint32_t output_stride;
  // The next step's retention cap (kDecodeRows).
  uint32_t decode_rows;
  // The live-row clamp bound: the draft's verify stride.
  uint32_t verify_stride;
  // Each lane's logical position at this step's verify (the item's value).
  uint32_t position_base[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // maxNewTokens - generatedTokens before this step's commit.
  uint32_t remaining[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  // The next step's assumed proposal budget per lane: the same formula
  // the encode applies (adaptive budget or the flat proposal cap).
  uint32_t budget[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(DecodeStepPackParams) == 96,
              "step-pack parameters are 96 bytes on both sides");

// One lane's device mirror of the next step's host-computed inputs. A
// lane is "continuing" when its request id matches the pack's — the
// pack was written by that request's immediately preceding step.
struct DecodeStepPackLane {
  // The emitting lane's request id; 0 marks an unwritten slot.
  uint64_t request_id;
  // OutputTokens[retained - 1]: the step's last committed token, which
  // is the next step's anchor (the host's pendingToken update).
  uint32_t anchor;
  // This step's logical position plus the retained count: the next
  // step's logicalPosition — every retained token advances it by one.
  uint32_t position_base;
  // The draft ring's committed length, equal to position_base.
  uint32_t cache_length;
  // The next step's min(remaining, kDecodeRows): zero names a lane the
  // host would finish — it never reaches a next encode.
  uint32_t maximum_retained;
  // min(budget, maximum_retained - 1) + 1 clamped into
  // [1, verify_stride], matching the host's liveRows.
  uint32_t live_rows;
  // This step's committed row count and accepted proposal count.
  uint32_t retained;
  uint32_t accepted;
};

static_assert(sizeof(DecodeStepPackLane) == 40,
              "step-pack lane is 40 bytes on both sides");

// draft_predraft_pack's payload (the GPU-chained predraft, dispatched
// after decode_step_pack in the same command): the pack lanes just written
// supply each lane's next anchor and position base, so the kernel stages
// the draft input rows and rotary positions the draft graph's encodes
// would otherwise need the host's readback for.
struct DraftPredraftPackParams {
  uint32_t lanes;
  // The packed draft input's per-lane row count (draft.queryRows()).
  uint32_t query_rows;
  // The packed draft positions' per-lane row count (kDecodeRows).
  uint32_t decode_rows;
  uint32_t mask_token;
};

static_assert(sizeof(DraftPredraftPackParams) == 16,
              "draft predraft pack parameters are 16 bytes on both sides");

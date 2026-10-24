#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ExecutionGeometry.h"

struct Q4EmbeddingParams {
  uint32_t rows;
  uint32_t vocabulary_size;
};

static_assert(sizeof(Q4EmbeddingParams) == 8,
              "Q4 embedding parameters are 8 bytes on both sides");

struct VerifyInputBatchParams {
  uint32_t vocabulary;
  // The packed lane stride of the verify input: a lane's anchor and its
  // proposal rows, RICHENGINE_TARGET_VERIFY_ROWS unless the draft's trained
  // block is shorter (draft.verifyRows()).
  uint32_t verify_stride;
  // The packed lane stride of the draft input rows: the draft's trained
  // query depth, RICHENGINE_DRAFT_QUERY_ROWS unless the draft's trained
  // block is shorter (draft.queryRows()).
  uint32_t draft_stride;
  // Live verify rows per lane (anchor included): rows at or past a lane's
  // count repeat its anchor instead of a proposal, so a dead row's expert
  // routes collapse onto the anchor's rather than adding to the dispatch's
  // distinct-expert union. The payload is patchable — budgets change per
  // step inside the baked span.
  uint32_t live_rows[RICHENGINE_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(VerifyInputBatchParams) == 28,
              "Verify input parameters are 28 bytes on both sides");

// verify_input_tokens_dev's payload (RICHENGINE_DEVICE_VERIFY_PARAMS):
// the batch parameters plus the per-lane facts the kernel proves the step
// pack against before taking a device field — the request id and the
// host-computed position and live rows. device_lanes marks the lanes
// trusted to take the pack without the equality proof: the submit-ahead
// encode's lanes, zero on the readback path.
struct VerifyInputDeviceParams {
  // The u64 ids lead so the 4-byte batch fields tail without padding.
  uint64_t request_id[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  VerifyInputBatchParams verify;
  uint32_t position_base[RICHENGINE_MAXIMUM_BATCH_WIDTH];
  uint32_t device_lanes;
};

static_assert(sizeof(VerifyInputDeviceParams) == 80,
              "Verify input device parameters are 80 bytes on both sides");

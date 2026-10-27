#include "metal/abi/KernelABI.h"

// RICHENGINE_DEVICE_VERIFY_PARAMS's step bridge: a committed decode
// step's last dispatch. Each lane's thread mirrors the acceptance
// results the completion handler would read back — retained and
// accepted counts, the closing anchor token — into the pack block the
// next step's device-side consumers bind, so the next command's encode
// carries no dependency on the host seeing these values. Every field is
// the exact arithmetic Runtime's encode and finalizeDecode apply; a lane
// the host finished (retained covers its remaining budget) reports a
// zero retention limit and live-row count, and never encodes again.

kernel void decode_step_pack(
    device const uint *retained_counts [[buffer(0)]],
    device const uint *accepted_counts [[buffer(1)]],
    device const uint *output_tokens [[buffer(2)]],
    device DecodeStepPackLane *pack [[buffer(3)]],
    constant DecodeStepPackParams &params [[buffer(4)]],
    uint lane [[thread_position_in_grid]]) {
  const uint retained = retained_counts[lane];
  device DecodeStepPackLane &out = pack[lane];
  out.request_id = params.request_id[lane];
  out.retained = retained;
  out.accepted = accepted_counts[lane];
  // The retained run ends with the next anchor; a zero retained only
  // marks a lane the host rejected, so guard the index rather than
  // trust it.
  const uint last = retained ? retained - 1u : 0u;
  out.anchor = output_tokens[lane * params.output_stride + last];
  const uint base = params.position_base[lane] + retained;
  out.position_base = base;
  out.cache_length = base;
  // The next step's retention limit: the output budget left after this
  // step's commit, clamped to the decode row window.
  const uint remaining = params.remaining[lane];
  const uint next = remaining > retained ? remaining - retained : 0u;
  const uint maximum = min(next, params.decode_rows);
  out.maximum_retained = maximum;
  // The host's live-row formula: proposals clamp to one row below the
  // retention limit and live rows = proposals + 1, clamped into
  // [1, verify_stride] the way the verify-input payload applies it.
  const uint live =
      maximum ? min(params.budget[lane], maximum - 1u) + 1u : 0u;
  out.live_rows = maximum ? clamp(live, 1u, params.verify_stride) : 0u;
}

// The GPU-chained predraft's staging pass: dispatched after
// decode_step_pack in the same command, it writes the next step's draft
// inputs the host encode cannot know yet — each lane's anchor row plus
// mask padding, and its rotary positions continuing at the pack's
// position base. A lane the step ended keeps whatever the kernel writes;
// the next encode's lane set proves it away. Grid {rows, lanes}.
kernel void draft_predraft_pack(
    device const DecodeStepPackLane *pack [[buffer(0)]],
    device uint *draft_input [[buffer(1)]],
    device uint *draft_positions [[buffer(2)]],
    constant DraftPredraftPackParams &params [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
  const uint lane = index.y;
  const uint row = index.x;
  const device DecodeStepPackLane &p = pack[lane];
  if (row < params.query_rows)
    draft_input[lane * params.query_rows + row] =
        row ? params.mask_token : p.anchor;
  if (row < params.decode_rows)
    draft_positions[lane * params.decode_rows + row] = p.position_base + row;
}

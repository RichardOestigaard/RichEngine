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

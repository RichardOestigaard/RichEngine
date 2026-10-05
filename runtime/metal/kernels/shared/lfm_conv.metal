#include "metal/abi/KernelABI.h"
#include "metal/abi/LfmConv.h"

// LFM2's short convolution mixer. The packed in_proj row is [B | C | x];
// the layer computes bx = B*x, runs the causal depthwise convolution over
// the taps-1-token FIFO state and the new rows, and emits C*conv. The
// conv_state is the previous taps-1 tokens' bx rows (no input reordering,
// conv_L_cache taps), updated with the newest taps-1 rows.

// Source element p of the sequence's stream: negative p indexes the state
// FIFO, others index the row's own bx.
static inline float lfm_source(
    device const bfloat *packed, device const bfloat *state_in, int row,
    uint channel, uint tap, uint dimension, uint taps) {
  const int p = row + int(tap) - int(taps - 1);
  if (p < 0)
    return float(state_in[(p + int(taps - 1)) * dimension + channel]);
  // B*x of the row, read from the packed [B|C|x] row.
  const ulong base = ulong(p) * 3 * dimension;
  return float(packed[base + channel]) * float(packed[base + 2 * dimension + channel]);
}

// Prefill: one sequence's rows from packed into output, and the newest
// taps-1 bx rows into state_out. Tasks [rows*dimension, (rows+taps-1)*dimension)
// are the state writes.
kernel void prefill_lfm_conv(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device const bfloat *state_in [[buffer(2)]],
    device bfloat *state_out [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant SplashLfmConvParams &params [[buffer(5)]],
    uint task [[thread_position_in_grid]]) {
  const uint dimension = params.dimension;
  const uint rows = params.rows;
  const uint taps = params.taps;
  const uint state_tasks = rows + taps - 1;
  if (task >= uint64_t(state_tasks) * dimension)
    return;
  const int row = int(task / dimension);
  const uint channel = task % dimension;
  if (row >= int(rows)) {
    // State write: the newest taps-1 stream elements, bx of rows
    // [rows-(taps-1)+w]. Rows past the state's depth read the state.
    const uint w = uint(row) - rows;
    const int source_row = int(rows) - int(taps - 1) + int(w);
    const int combined = source_row + int(taps - 1); // index into [state|bx]
    float value;
    if (combined < int(taps - 1))
      value = float(state_in[combined * dimension + channel]);
    else {
      const ulong base = ulong(source_row) * 3 * dimension;
      value = float(packed[base + channel]) *
              float(packed[base + 2 * dimension + channel]);
    }
    state_out[w * dimension + channel] = bfloat(value);
    return;
  }
  float conv = 0.0f;
  for (uint tap = 0; tap < taps; ++tap) {
    const float w = float(
        weights[params.taps_major ? tap * dimension + channel
                                  : channel * taps + tap]);
    conv += w * lfm_source(packed, state_in, row, channel, tap,
                           dimension, taps);
  }
  const ulong base = ulong(row) * 3 * dimension;
  output[ulong(row) * dimension + channel] =
      bfloat(float(packed[base + dimension + channel]) * conv);
}

// Verify: all lanes' eight rows of one layer in one dispatch, from the
// layer's packed rows. Each lane's conv state is its own buffer. `mixed`
// receives every lane's bx rows for the commit that follows; output is
// C*conv for all rows — the mixer writes the out-projection's input.
kernel void verify_lfm_conv(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device const bfloat *state0 [[buffer(2)]],
    device const bfloat *state1 [[buffer(3)]],
    device const bfloat *state2 [[buffer(4)]],
    device const bfloat *state3 [[buffer(5)]],
    device bfloat *mixed [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant SplashLfmConvParams &params [[buffer(8)]],
    uint task [[thread_position_in_grid]]) {
  const uint dimension = params.dimension;
  const uint rows = params.rows;
  const uint taps = params.taps;
  if (task >= uint64_t(params.lanes) * rows * dimension)
    return;
  const uint lane = task / (rows * dimension);
  const uint row = (task % (rows * dimension)) / dimension;
  const uint channel = task % dimension;
  device const bfloat *state =
      lane == 0 ? state0 : lane == 1 ? state1 : lane == 2 ? state2 : state3;
  const ulong layer_base =
      uint64_t(params.layer) * params.state_layer_bytes / 2;
  const ulong lane_base = ulong(lane) * rows;
  const ulong base = (lane_base + row) * 3 * dimension;
  const float b = float(packed[base + channel]);
  const float c = float(packed[base + dimension + channel]);
  const float x = float(packed[base + 2 * dimension + channel]);
  const float bx = b * x;
  mixed[(ulong(lane) * rows + row) * dimension + channel] = bfloat(bx);
  float conv = 0.0f;
  for (uint tap = 0; tap < taps; ++tap) {
    const float w = float(
        weights[params.taps_major ? tap * dimension + channel
                                  : channel * taps + tap]);
    const int p = int(row) + int(tap) - int(taps - 1);
    float source;
    if (p < 0)
      source = float(state[layer_base + (p + int(taps - 1)) * dimension + channel]);
    else {
      const ulong source_base = (lane_base + uint(p)) * 3 * dimension;
      source = float(packed[source_base + channel]) *
               float(packed[source_base + 2 * dimension + channel]);
    }
    conv += w * source;
  }
  output[(ulong(lane) * rows + row) * dimension + channel] = bfloat(c * conv);
}

// Commit one layer's conv states after acceptance: every lane's new FIFO is
// the last taps-1 elements of its retained stream [state | bx]. `retained`
// holds each lane's retained rows of the step.
kernel void commit_lfm_conv(
    device const bfloat *mixed [[buffer(0)]],
    device const uint *retained [[buffer(1)]],
    device const bfloat *state0_in [[buffer(2)]],
    device const bfloat *state1_in [[buffer(3)]],
    device const bfloat *state2_in [[buffer(4)]],
    device const bfloat *state3_in [[buffer(5)]],
    device bfloat *state0_out [[buffer(6)]],
    device bfloat *state1_out [[buffer(7)]],
    device bfloat *state2_out [[buffer(8)]],
    device bfloat *state3_out [[buffer(9)]],
    constant SplashLfmConvParams &params [[buffer(10)]],
    uint task [[thread_position_in_grid]]) {
  const uint dimension = params.dimension;
  const uint taps = params.taps;
  const uint depth = taps - 1;
  if (task >= uint64_t(params.lanes) * depth * dimension)
    return;
  const uint lane = task / (depth * dimension);
  const uint w = (task % (depth * dimension)) / dimension;
  const uint channel = task % dimension;
  const uint rows = params.rows;
  device const bfloat *state_in =
      lane == 0 ? state0_in : lane == 1 ? state1_in : lane == 2 ? state2_in : state3_in;
  device bfloat *state_out =
      lane == 0 ? state0_out : lane == 1 ? state1_out : lane == 2 ? state2_out : state3_out;
  const uint kept = retained[lane];
  if (kept > rows)
    return;
  // Stream index r+w of [state0..depth-1 | bx0..rows-1]: the last depth.
  // The layer's FIFO slice inside the lane's conv-state storage.
  const ulong layer_base = uint64_t(params.layer) * params.state_layer_bytes / 2;
  const int index = int(kept) + int(w);
  float value;
  if (index < int(depth))
    value = float(state_in[layer_base + index * dimension + channel]);
  else
    value = float(mixed[(ulong(lane) * rows + uint(index) - depth) * dimension +
                        channel]);
  state_out[layer_base + w * dimension + channel] = bfloat(value);
}

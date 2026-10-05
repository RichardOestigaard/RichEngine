#include "metal/abi/KernelABI.h"
#include "metal/abi/LfmConv.h"

// LFM2's short convolution mixer. The packed in_proj row is [B | C | x];
// the layer computes bx = B*x, runs the causal depthwise convolution over
// the taps-1-token FIFO state and the new rows, and emits C*conv. The
// conv_state is the previous taps-1 tokens' bx rows (no input reordering,
// conv_L_cache taps), updated with the newest taps-1 rows.

// Element type of a dispatch: scalar bfloat, or bfloat4 when the host
// proves the channel count divides by four (every row start stays aligned).
template <uint Vec> struct LfmVec;
template <> struct LfmVec<1> {
  using T = bfloat;
  using F = float;
  static device const T *at(device const bfloat *p, ulong offset) {
    return reinterpret_cast<device const T *>(p + offset);
  }
  static device T *at(device bfloat *p, ulong offset) {
    return reinterpret_cast<device T *>(p + offset);
  }
  static F load(device const bfloat *p, ulong offset) {
    return float(*at(p, offset));
  }
  static void store(device bfloat *p, ulong offset, F value) {
    *at(p, offset) = T(value);
  }
};
template <> struct LfmVec<4> {
  using T = bfloat4;
  using F = float4;
  static F load(device const bfloat *p, ulong offset) {
    return float4(*reinterpret_cast<device const T *>(p + offset));
  }
  static void store(device bfloat *p, ulong offset, F value) {
    *reinterpret_cast<device T *>(p + offset) = T(value);
  }
};

// Source element p of the sequence's stream: negative p indexes the state
// FIFO, others index the row's own bx. Vec channels per task, so `channel`
// is the first lane's index.
template <uint Vec>
static inline typename LfmVec<Vec>::F lfm_source(
    device const bfloat *packed, device const bfloat *state_in, int row,
    uint channel, uint tap, uint dimension, uint taps) {
  const int p = row + int(tap) - int(taps - 1);
  if (p < 0)
    return LfmVec<Vec>::load(state_in,
                             ulong(p + int(taps - 1)) * dimension + channel);
  // B*x of the row, read from the packed [B|C|x] row.
  const ulong base = ulong(p) * 3 * dimension;
  return LfmVec<Vec>::load(packed, base + channel) *
         LfmVec<Vec>::load(packed, base + 2 * dimension + channel);
}

// The conv weight's Vec-lane fetch: taps_major weights are contiguous in
// the channel, the [channel][tap] order strides by `taps`.
template <uint Vec>
static inline typename LfmVec<Vec>::F lfm_weight(
    device const bfloat *weights, uint channel, uint tap, uint dimension,
    uint taps, uint taps_major) {
  if (taps_major)
    return LfmVec<Vec>::load(weights, ulong(tap) * dimension + channel);
  if constexpr (Vec == 1)
    return float(weights[ulong(channel) * taps + tap]);
  else {
    float lanes[Vec];
    for (uint v = 0; v < Vec; ++v)
      lanes[v] = float(weights[ulong(channel + v) * taps + tap]);
    if constexpr (Vec == 4)
      return float4(lanes[0], lanes[1], lanes[2], lanes[3]);
  }
}

template <uint Vec>
static inline void prefill_impl(
    device const bfloat *packed, device const bfloat *weights,
    device const bfloat *state_in, device bfloat *state_out,
    device bfloat *output, constant SplashLfmConvParams &params, uint task) {
  using V = LfmVec<Vec>;
  const uint dimension = params.dimension;
  const uint rows = params.rows;
  const uint taps = params.taps;
  const uint channels = dimension / Vec;
  const uint state_tasks = rows + taps - 1;
  if (task >= uint64_t(state_tasks) * channels)
    return;
  const int row = int(task / channels);
  const uint channel = (task % channels) * Vec;
  if (row >= int(rows)) {
    // State write: the newest taps-1 stream elements, bx of rows
    // [rows-(taps-1)+w]. Rows past the state's depth read the state.
    const uint w = uint(row) - rows;
    const int source_row = int(rows) - int(taps - 1) + int(w);
    const int combined = source_row + int(taps - 1); // index into [state|bx]
    typename V::F value;
    if (combined < int(taps - 1))
      value = V::load(state_in, ulong(combined) * dimension + channel);
    else {
      const ulong base = ulong(source_row) * 3 * dimension;
      value = V::load(packed, base + channel) *
              V::load(packed, base + 2 * dimension + channel);
    }
    V::store(state_out, ulong(w) * dimension + channel, value);
    return;
  }
  typename V::F conv = typename V::F(0.0f);
  for (uint tap = 0; tap < taps; ++tap) {
    const typename V::F w =
        lfm_weight<Vec>(weights, channel, tap, dimension, taps,
                        params.taps_major);
    conv += w * lfm_source<Vec>(packed, state_in, row, channel, tap,
                                dimension, taps);
  }
  const ulong base = ulong(row) * 3 * dimension;
  V::store(output, ulong(row) * dimension + channel,
           V::load(packed, base + dimension + channel) * conv);
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
  prefill_impl<1>(packed, weights, state_in, state_out, output, params, task);
}

kernel void prefill_lfm_conv_v4(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device const bfloat *state_in [[buffer(2)]],
    device bfloat *state_out [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant SplashLfmConvParams &params [[buffer(5)]],
    uint task [[thread_position_in_grid]]) {
  prefill_impl<4>(packed, weights, state_in, state_out, output, params, task);
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

// Vectorized verify: one threadgroup covers all `rows` of one lane's
// kLfmVerifyChannelBlock-wide channel block, so every row's bx is computed
// once, staged in threadgroup memory (and written to `mixed`, bit-identical
// to the scalar kernel), then the taps read the stage instead of
// re-deriving each source row's product. The host launches it only when
// dimension % kLfmVerifyChannelBlock == 0 and rows * kLfmVerifyVecColumns
// fits one threadgroup.
constant uint kLfmVerifyVecColumns = 32;       // bfloat4 lanes per row slice
constant uint kLfmVerifyStage = 1024;          // rows*kLfmVerifyVecColumns max

kernel void verify_lfm_conv_v4(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device const bfloat *state0 [[buffer(2)]],
    device const bfloat *state1 [[buffer(3)]],
    device const bfloat *state2 [[buffer(4)]],
    device const bfloat *state3 [[buffer(5)]],
    device bfloat *mixed [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant SplashLfmConvParams &params [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  const uint dimension = params.dimension;
  const uint rows = params.rows;
  const uint taps = params.taps;
  const uint vec_columns = dimension / 4;
  const uint blocks_per_lane = vec_columns / kLfmVerifyVecColumns;
  const uint lane = group / blocks_per_lane;
  const uint block = group % blocks_per_lane;
  const uint row = tid / kLfmVerifyVecColumns;
  const uint column = block * kLfmVerifyVecColumns + (tid % kLfmVerifyVecColumns);
  const uint channel = column * 4;
  device const bfloat *state =
      lane == 0 ? state0 : lane == 1 ? state1 : lane == 2 ? state2 : state3;
  const ulong layer_base =
      uint64_t(params.layer) * params.state_layer_bytes / 2;
  const ulong lane_base = ulong(lane) * rows;
  const ulong base = (lane_base + row) * 3 * dimension;
  const float4 b = LfmVec<4>::load(packed, base + channel);
  const float4 c = LfmVec<4>::load(packed, base + dimension + channel);
  const float4 x = LfmVec<4>::load(packed, base + 2 * dimension + channel);
  const bfloat4 bx = bfloat4(b * x);
  threadgroup bfloat4 staged[kLfmVerifyStage];
  staged[row * kLfmVerifyVecColumns + (tid % kLfmVerifyVecColumns)] = bx;
  *reinterpret_cast<device bfloat4 *>(
      mixed + (lane_base + row) * dimension + channel) = bx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float4 conv = float4(0.0f);
  for (uint tap = 0; tap < taps; ++tap) {
    const float4 w =
        lfm_weight<4>(weights, channel, tap, dimension, taps,
                      params.taps_major);
    const int p = int(row) + int(tap) - int(taps - 1);
    float4 source;
    if (p < 0)
      source = LfmVec<4>::load(
          state, layer_base + ulong(p + int(taps - 1)) * dimension + channel);
    else
      source = float4(
          staged[uint(p) * kLfmVerifyVecColumns + (tid % kLfmVerifyVecColumns)]);
    conv += w * source;
  }
  LfmVec<4>::store(output, (lane_base + row) * dimension + channel, c * conv);
}

// Commit the conv states after acceptance: every lane's new FIFO is the
// last taps-1 elements of its retained stream [state | bx]. `retained`
// holds each lane's retained rows of the step. `params.layers` batches all
// conv layers into one dispatch: task space is
// layers*lanes*depth*channels, layer l's state slot is params.layer + l and
// its `mixed` block sits l * mixed_layer_stride bytes in.
template <uint Vec>
static inline void commit_impl(
    device const bfloat *mixed, device const uint *retained,
    device const bfloat *state0_in, device const bfloat *state1_in,
    device const bfloat *state2_in, device const bfloat *state3_in,
    device bfloat *state0_out, device bfloat *state1_out,
    device bfloat *state2_out, device bfloat *state3_out,
    constant SplashLfmConvParams &params, uint task) {
  using V = LfmVec<Vec>;
  const uint dimension = params.dimension;
  const uint taps = params.taps;
  const uint depth = taps - 1;
  const uint channels = dimension / Vec;
  const uint lane_tasks = depth * channels;
  const uint layer_tasks = params.lanes * lane_tasks;
  if (task >= uint64_t(params.layers) * layer_tasks)
    return;
  const uint conv_layer = task / layer_tasks;
  const uint lane = (task % layer_tasks) / lane_tasks;
  const uint w = (task % lane_tasks) / channels;
  const uint channel = (task % channels) * Vec;
  const uint rows = params.rows;
  device const bfloat *state_in =
      lane == 0 ? state0_in : lane == 1 ? state1_in : lane == 2 ? state2_in : state3_in;
  device bfloat *state_out =
      lane == 0 ? state0_out : lane == 1 ? state1_out : lane == 2 ? state2_out : state3_out;
  const uint kept = retained[lane];
  if (kept > rows)
    return;
  // Stream index kept+w of [state0..depth-1 | bx0..rows-1]: the last depth.
  // The layer's FIFO slice inside the lane's conv-state storage.
  const ulong layer_base =
      uint64_t(params.layer + conv_layer) * params.state_layer_bytes / 2;
  const ulong mixed_base =
      uint64_t(conv_layer) * params.mixed_layer_stride / 2;
  const int index = int(kept) + int(w);
  typename V::F value;
  if (index < int(depth))
    value = V::load(state_in, layer_base + ulong(index) * dimension + channel);
  else
    value = V::load(mixed, mixed_base +
                                 (ulong(lane) * rows + uint(index) - depth) *
                                     dimension +
                                 channel);
  V::store(state_out, layer_base + ulong(w) * dimension + channel, value);
}

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
  commit_impl<1>(mixed, retained, state0_in, state1_in, state2_in, state3_in,
                 state0_out, state1_out, state2_out, state3_out, params, task);
}

kernel void commit_lfm_conv_v4(
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
  commit_impl<4>(mixed, retained, state0_in, state1_in, state2_in, state3_in,
                 state0_out, state1_out, state2_out, state3_out, params, task);
}

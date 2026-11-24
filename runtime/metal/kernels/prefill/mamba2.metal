#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/KernelABI.h"
#include "metal/abi/Mamba2.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/rms_inverse.h"

// Mamba-2 (SSD) prefill for Nemotron's mixer, the sequential form of the
// chunk scan (a chunk-parallel WY/UT variant can follow the gdn_chunked
// pattern later): prepare runs the causal conv+SiLU and the dt'/dA
// evaluation, the scan carries each head's [HeadDim][StateSize] fp32 state
// through the whole window token by token, and the gate applies the
// grouped gated RMSNorm.
//
// Prepare: one (token, group) task per threadgroup, 128 threads covering the
// group's 512 x channels plus its 128 B and 128 C channels; token 0's
// threadgroups also write the convolution carry. Scan: one threadgroup per
// (head, 16-row tile), 128 threads — a lane owns sixteen state columns of
// one row, eight lanes share a row, a simdgroup covers four rows; blocks of
// sixteen tokens of B/C (widened to fp32), x, dt' and dA stage in threadgroup
// memory while the following block prefetches into registers. Gate: one
// (token, group) task, v = y*silu(z) normalized over the group's 512 columns
// and scaled by the norm weight (NemotronHRMSNormGated, gate before norm).

constant constexpr uint Mamba2ScanBlock = 16;

// The conv channel of a group's e-th owned channel — the decode prologue's
// mamba2_group_channel, repeated: x at [group*GroupWidth), B at
// [Inner + group*State), C at [Inner + Groups*State + group*State).
template <uint InnerSize, uint Groups, uint StateSize>
inline uint mamba2_prefill_channel(uint group, uint e, uint group_width) {
  return e < group_width
             ? group * group_width + e
             : e < group_width + StateSize
                   ? InnerSize + group * StateSize + e - group_width
                   : InnerSize + Groups * StateSize + group * StateSize + e -
                         group_width - StateSize;
}

// dt' = clamp(softplus(dt + bias), dt_min, dt_max), the decode kernel's
// mamba2_step_size: the bias add and softplus round to bf16 like the
// reference, the dt_limit clamp stays fp32.
inline float mamba2_prefill_step(float dt_raw, float bias, float dt_min,
                                 float dt_max) {
  const bfloat x = bfloat(dt_raw + bias);
  const float xf = float(x);
  const float softplus =
      float(bfloat(max(xf, 0.0f) +
                   fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                       0.69314718056f));
  return clamp(softplus, dt_min, dt_max);
}

// Grid {tokens * groups}, 128 threads: the group's post-conv x|B|C rows in
// `post_conv` (for the scan and the commit), dt' in `step_sizes`, and on
// token 0 the convolution carry of the whole window.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_prepare_prefill_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *post_conv, device const bfloat *dt_bias,
    device float *step_sizes, device const bfloat *conv_bias,
    constant Mamba2PrefillParams &params,
    uint task, uint thread_index) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  constexpr uint GroupChannels = GroupWidth + 2 * StateSize;
  const uint token = task / Groups;
  const uint group = task % Groups;
  // The conv columns lead the packed row at InnerSize (after the gate).
  device const bfloat *conv_rows = packed + InnerSize;
  for (uint e = thread_index; e < GroupChannels; e += 128) {
    const uint channel =
        mamba2_prefill_channel<InnerSize, Groups, StateSize>(group, e,
                                                             GroupWidth);
    post_conv[token * ConvDim + channel] =
        gdn_conv_silu(conv_rows, conv_state_in, conv_weights, PackedWidth,
                      ConvDim, token, channel, conv_bias);
  }
  if (thread_index < HeadsPerGroup) {
    const uint head = group * HeadsPerGroup + thread_index;
    step_sizes[token * Heads + head] = mamba2_prefill_step(
        float(packed[token * PackedWidth + InnerSize + ConvDim + head]),
        float(dt_bias[head]), params.dt_min, params.dt_max);
  }
  if (token == 0) {
    for (uint row = 0; row < 3; ++row) {
      for (uint e = thread_index; e < GroupChannels; e += 128) {
        const uint channel =
            mamba2_prefill_channel<InnerSize, Groups, StateSize>(group, e,
                                                                 GroupWidth);
        conv_state_out[row * ConvDim + channel] =
            gdn_conv_carry(conv_rows, conv_state_in, PackedWidth, ConvDim,
                           params.tokens, row, channel);
      }
    }
  }
}

#define MAMBA2_PREPARE_PREFILL_ENTRY(Name, Heads, HeadDim, Groups, StateSize, \
                                     ConvDim, PackedWidth)                    \
  kernel void Name(                                                          \
      device const bfloat *packed [[buffer(0)]],                             \
      device const bfloat *conv_weights [[buffer(1)]],                       \
      device const bfloat *conv_state_in [[buffer(2)]],                      \
      device bfloat *conv_state_out [[buffer(3)]],                           \
      device bfloat *post_conv [[buffer(4)]],                                \
      device const bfloat *dt_bias [[buffer(5)]],                            \
      device float *step_sizes [[buffer(6)]],                                \
      device const bfloat *conv_bias [[buffer(7)]],                          \
      constant Mamba2PrefillParams &params [[buffer(8)]],                    \
      uint task [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                   \
    mamba2_prepare_prefill_phase<Heads, HeadDim, Groups, StateSize, ConvDim, \
                                 PackedWidth>(                               \
        packed, conv_weights, conv_state_in, conv_state_out, post_conv,      \
        dt_bias, step_sizes, conv_bias, params, task, thread_index);         \
  }

MAMBA2_PREPARE_PREFILL_ENTRY(prefill_mamba2_prepare, 64, 64, 8, 128, 6144,
                             10496)
#undef MAMBA2_PREPARE_PREFILL_ENTRY

// One threadgroup carries RICHENGINE_GDN_SCAN_STATE_ROWS rows of a head's
// 64 x 128 fp32 SSM state through the whole window, token by token:
// S = dA*S + (dt'*x_d) B'; y = C.S + D*x. A lane owns sixteen state columns
// of one row, so eight lanes share a row and each C.S dot product costs
// three shuffle steps; a simdgroup covers four rows. Blocks of
// Mamba2ScanBlock tokens of B and C (widened to fp32), x and the gates are
// staged in threadgroup memory while the following block is prefetched into
// registers, so the dependent per-token chain reads only threadgroup memory.
// The math is the decode recurrence; only the summation order of the dot
// products differs from its per-lane form.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim>
inline void mamba2_scan_prefill_phase(
    device const bfloat *post_conv, device const float *step_sizes,
    device const float *a_weights, device const float *skip_weights,
    device const float *state_in, device float *state_out,
    device bfloat *output, uint tokens, threadgroup float *bs,
    threadgroup float *cs, threadgroup float *xs, threadgroup float *gates,
    uint group, uint thread_index, uint lane, uint simd_group) {
  constexpr uint Columns = 16;
  constexpr uint LanesPerRow = StateSize / Columns;
  constexpr uint Rows = RICHENGINE_GDN_SCAN_STATE_ROWS;
  constexpr uint Threads = RICHENGINE_GDN_SCAN_THREADS;
  constexpr uint Block = Mamba2ScanBlock;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint GroupsPerHead = HeadDim / Rows;
  static_assert(Rows == (Threads / 32) * (32 / LanesPerRow),
                "the simdgroups of a threadgroup cover its state rows");
  static_assert(Block * StateSize == Threads * Columns,
                "every thread stages sixteen B and sixteen C columns");
  static_assert(Block * Rows == 32 * 8,
                "the first simdgroup stages the x block");

  const uint head = group / GroupsPerHead;
  const uint grp = head / HeadsPerGroup;
  const uint row_base = (group % GroupsPerHead) * Rows;
  const uint row = simd_group * (32 / LanesPerRow) + lane / LanesPerRow;
  const uint column = (lane % LanesPerRow) * Columns;
  const ulong state_base =
      (ulong(head) * HeadDim + row_base + row) * StateSize + column;
  const float a_scale = a_weights[head];
  const float d_skip = skip_weights[head];

  float4 state[Columns / 4];
  for (uint c = 0; c < Columns / 4; ++c)
    state[c] =
        reinterpret_cast<device const float4 *>(state_in + state_base)[c];

  bfloat4 b_fetch[2][2] = {}, c_fetch[2][2] = {}, x_fetch[2] = {};
  float dt_fetch = 0.0f, da_fetch = 0.0f;
  // Registers for tokens at or past `tokens` keep their previous values and
  // are staged as they are: the compute loop stops at the chunk's token
  // count, so those slots are never read, and zeroing them would only add
  // instructions to the per-token chain.
  auto fetch = [&](uint start) {
    for (uint n = 0; n < 2; ++n) {
      const uint element = (n * Threads + thread_index) * 8;
      const uint token = start + element / StateSize;
      if (token < tokens) {
        const ulong base = ulong(token) * ConvDim + InnerSize +
                           grp * StateSize + element % StateSize;
        for (uint h = 0; h < 2; ++h) {
          b_fetch[n][h] =
              reinterpret_cast<device const bfloat4 *>(post_conv + base)[h];
          c_fetch[n][h] = reinterpret_cast<device const bfloat4 *>(
              post_conv + base + ulong(Groups) * StateSize)[h];
        }
      }
    }
    if (thread_index < 32) {
      const uint element = thread_index * 8;
      const uint token = start + element / Rows;
      if (token < tokens) {
        const ulong base =
            ulong(token) * ConvDim + head * HeadDim + row_base + element % Rows;
        for (uint h = 0; h < 2; ++h)
          x_fetch[h] =
              reinterpret_cast<device const bfloat4 *>(post_conv + base)[h];
      }
    }
    if (thread_index < Block && start + thread_index < tokens) {
      const float dtv = step_sizes[(start + thread_index) * Heads + head];
      dt_fetch = dtv;
      da_fetch = fast::exp(dtv * a_scale);
    }
  };
  auto stage = [&]() {
    for (uint n = 0; n < 2; ++n) {
      const uint element = (n * Threads + thread_index) * 8;
      for (uint h = 0; h < 2; ++h) {
        reinterpret_cast<threadgroup float4 *>(bs + element)[h] =
            float4(b_fetch[n][h]);
        reinterpret_cast<threadgroup float4 *>(cs + element)[h] =
            float4(c_fetch[n][h]);
      }
    }
    if (thread_index < 32) {
      for (uint h = 0; h < 2; ++h) {
        reinterpret_cast<threadgroup float4 *>(xs + thread_index * 8)[h] =
            float4(x_fetch[h]);
      }
    }
    if (thread_index < Block) {
      gates[thread_index] = da_fetch;
      gates[Block + thread_index] = dt_fetch;
    }
  };

  fetch(0);
  stage();
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  for (uint start = 0; start < tokens; start += Block) {
    const bool more = start + Block < tokens;
    if (more)
      fetch(start + Block);
    const uint count = min(Block, tokens - start);
    for (uint t = 0; t < count; ++t) {
      const float dA = gates[t], dtv = gates[Block + t];
      const float x_row = xs[t * Rows + row];
      const float scale = dtv * x_row;
      const threadgroup float4 *brow =
          reinterpret_cast<const threadgroup float4 *>(bs + t * StateSize +
                                                       column);
      const threadgroup float4 *crow =
          reinterpret_cast<const threadgroup float4 *>(cs + t * StateSize +
                                                       column);
      float4 partial = 0.0f;
      for (uint c = 0; c < Columns / 4; ++c) {
        state[c] = fma(float4(scale), brow[c], state[c] * dA);
        partial = fma(state[c], crow[c], partial);
      }
      float result = (partial.x + partial.y) + (partial.z + partial.w);
      for (uint s = 1; s < LanesPerRow; s <<= 1)
        result += simd_shuffle_xor(result, s);
      result += d_skip * x_row;
      if (lane % LanesPerRow == 0) {
        output[(ulong(start + t) * InnerSize) + head * HeadDim + row_base +
               row] = bfloat(result);
      }
    }
    if (more) {
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
      stage();
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    }
  }
  for (uint c = 0; c < Columns / 4; ++c)
    reinterpret_cast<device float4 *>(state_out + state_base)[c] = state[c];
}

#define MAMBA2_SCAN_PREFILL_ENTRY(Name, Heads, HeadDim, Groups, StateSize,   \
                                  ConvDim)                                   \
  kernel void Name(                                                          \
      device const bfloat *post_conv [[buffer(0)]],                          \
      device const float *step_sizes [[buffer(1)]],                          \
      device const float *a_weights [[buffer(2)]],                           \
      device const float *skip_weights [[buffer(3)]],                        \
      device const float *state_in [[buffer(4)]],                            \
      device float *state_out [[buffer(5)]],                                 \
      device bfloat *output [[buffer(6)]],                                   \
      constant Mamba2PrefillParams &params [[buffer(7)]],                    \
      uint group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    threadgroup float bs[Mamba2ScanBlock * StateSize];                       \
    threadgroup float cs[Mamba2ScanBlock * StateSize];                       \
    threadgroup float xs[Mamba2ScanBlock * RICHENGINE_GDN_SCAN_STATE_ROWS];  \
    threadgroup float gates[2 * Mamba2ScanBlock];                            \
    mamba2_scan_prefill_phase<Heads, HeadDim, Groups, StateSize, ConvDim>(   \
        post_conv, step_sizes, a_weights, skip_weights, state_in, state_out, \
        output, params.tokens, bs, cs, xs, gates, group, thread_index, lane, \
        simd_group);                                                         \
  }

MAMBA2_SCAN_PREFILL_ENTRY(prefill_mamba2_scan, 64, 64, 8, 128, 6144)
#undef MAMBA2_SCAN_PREFILL_ENTRY

// The grouped gated RMSNorm of one (token, group) task: v = y * silu(z) over
// the group's 512 columns, normalized and scaled by the norm weight —
// NemotronHRMSNormGated's order. Each thread owns the adjacent pair
// 2*thread_index, 2*thread_index+1 — the same element order and reduction
// tree the decode gate uses, so the rows are bitwise the same. W is the norm
// weights' stored type (float: a GGUF's F32 norms, _f32).
template <uint Heads, uint HeadDim, uint Groups, uint PackedWidth, class W>
inline void mamba2_gate_prefill_phase(
    device const bfloat *recurrent, device const bfloat *packed,
    device const W *norm_weight, device bfloat *hidden, uint task,
    float rms_eps, threadgroup float *reductions, uint thread_index,
    uint lane, uint simd_group) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  const uint token = task / Groups;
  const uint group = task % Groups;
  const ulong base = ulong(token) * InnerSize + group * GroupWidth;
  const uint dim = 2 * thread_index;
  const uint column = group * GroupWidth + dim;
  const float y0 = float(recurrent[base + dim]);
  const float y1 = float(recurrent[base + dim + 1]);
  const float z0 = float(packed[token * PackedWidth + column]);
  const float z1 = float(packed[token * PackedWidth + column + 1]);
  const float v0 = y0 * richengine_silu(z0);
  const float v1 = y1 * richengine_silu(z1);
  const float inverse = rms_inverse_of_sums<8>(
      v0 * v0 + v1 * v1, GroupWidth, reductions, thread_index, lane,
      simd_group, rms_eps);
  hidden[base + dim] =
      bfloat((v0 * inverse) * float(norm_weight[column]));
  hidden[base + dim + 1] =
      bfloat((v1 * inverse) * float(norm_weight[column + 1]));
}

#define MAMBA2_GATE_PREFILL_ENTRY(Name, Heads, HeadDim, Groups, PackedWidth, \
                                  W)                                       \
  kernel void Name(                                                          \
      device const bfloat *recurrent [[buffer(0)]],                          \
      device const bfloat *packed [[buffer(1)]],                             \
      device const W *norm_weight [[buffer(2)]],                             \
      device bfloat *hidden [[buffer(3)]],                                   \
      constant Mamba2GateParams &params [[buffer(4)]],                       \
      uint task [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    threadgroup float reductions[8];                                         \
    mamba2_gate_prefill_phase<Heads, HeadDim, Groups, PackedWidth>(          \
        recurrent, packed, norm_weight, hidden, task, params.rms_eps,        \
        reductions, thread_index, lane, simd_group);                         \
  }

MAMBA2_GATE_PREFILL_ENTRY(prefill_mamba2_gate, 64, 64, 8, 10496, bfloat)
MAMBA2_GATE_PREFILL_ENTRY(prefill_mamba2_gate_f32, 64, 64, 8, 10496, float)
#undef MAMBA2_GATE_PREFILL_ENTRY

// The *_sums variant also writes the out-projection's input sums of the
// gated rows — the flat [row][64-group] layout and pairing of
// prefill_linear_q4_sums32 — so no separate sums pass runs on the mixer
// output. The group's eight 64-column spans map to its eight simdgroups.
#define MAMBA2_GATE_PREFILL_SUMS_ENTRY(Name, Heads, HeadDim, Groups,         \
                                       PackedWidth, W)                       \
  kernel void Name(                                                          \
      device const bfloat *recurrent [[buffer(0)]],                          \
      device const bfloat *packed [[buffer(1)]],                             \
      device const W *norm_weight [[buffer(2)]],                             \
      device bfloat *hidden [[buffer(3)]],                                   \
      device float *sums [[buffer(4)]],                                      \
      constant Mamba2GateParams &params [[buffer(5)]],                       \
      uint task [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    threadgroup float reductions[8];                                         \
    constexpr uint HeadsPerGroup = Heads / Groups;                           \
    constexpr uint InnerSize = Heads * HeadDim;                              \
    constexpr uint GroupWidth = HeadsPerGroup * HeadDim;                     \
    mamba2_gate_prefill_phase<Heads, HeadDim, Groups, PackedWidth>(          \
        recurrent, packed, norm_weight, hidden, task, params.rms_eps,        \
        reductions, thread_index, lane, simd_group);                         \
    const ulong base = ulong(task / Groups) * InnerSize +                    \
                       (task % Groups) * GroupWidth;                         \
    threadgroup_barrier(mem_flags::mem_device);                              \
    for (uint grp = simd_group; grp < GroupWidth / 64; grp += 8) {           \
      const ulong origin = base + grp * 64 + lane;                           \
      const float sum =                                                    \
          simd_sum(float(hidden[origin]) + float(hidden[origin + 32]));      \
      if (lane == 0)                                                       \
        sums[base / 64 + grp] = sum;                                       \
    }                                                                      \
  }

MAMBA2_GATE_PREFILL_SUMS_ENTRY(prefill_mamba2_gate_sums, 64, 64, 8, 10496,
                               bfloat)
MAMBA2_GATE_PREFILL_SUMS_ENTRY(prefill_mamba2_gate_sums_f32, 64, 64, 8, 10496,
                               float)
#undef MAMBA2_GATE_PREFILL_SUMS_ENTRY

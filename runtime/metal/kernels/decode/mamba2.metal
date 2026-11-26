#include "metal/abi/KernelABI.h"
#include "metal/abi/Mamba2.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/rms_inverse.h"

// Mamba-2 (SSD) verify decode for Nemotron's mixer: one threadgroup is one
// norm group (Heads/Groups heads = 512 dims) of one lane, so the gated group
// RMSNorm's 512-column reduction never leaves the group. 256 threads: one
// simdgroup per two verify rows in the prologue, and in the scan the group's
// HeadsPerGroup*HeadDim state rows strided over the eight simdgroups, each
// lane holding four of a row's StateSize columns.
//
// Packed row: [gate Inner | conv x|B|C ConvDim | dt Heads]. The prologue runs
// the causal four-tap conv (SiLU) over the group's channels — x for its
// heads, B and C for its group — and evaluates
// dt' = clamp(softplus(dt + dt_bias), dt_min, dt_max), dA = exp(dt' * A[h]).
// The scan applies S = dA*S + (dt'*x_d) B_n' to each head's
// [HeadDim][StateSize] fp32 matrix, emits y_d = sum_n C_n S_dn + D_h x_d to
// the ungated hidden rows, and the gate multiplies by silu(z) then
// normalizes the gated values over the group's 512 columns
// (NemotronHRMSNormGated's gate-before-norm order: the variance is taken
// over y * silu(z), then the norm weight applies).
//
// The committed-state fold mirrors GDN's: with MAMBA2_TAPE_FOLD the previous
// step's retained rows advance the incoming state as a pending prefix
// (reading their post-conv x|B|C rows and dt' off the lane's tape), the
// step's raw conv inputs, post-conv rows and dt' land on the tape for the
// next step, and state_out gets the pending-only advance; the standalone
// commit replays the retained rows when the fold is off.
//
// The tape's per-layer region lays out
//   [Rows][ConvDim] bf16 raw conv inputs,
//   [Rows][ConvDim] bf16 post-conv x|B|C,
//   [Rows][Heads]  fp32 dt'.

constant uint kMamba2DecodeSimdgroups = 8;

// One group's operands of a decode: the rows' post-conv x (512 dims), B and C
// (128 each, packed as one 256-column run — B first), the per-head dt' and
// dA, the heads' D scales, and the gate phase's reduction partials — two
// sets alternating by token parity, so a token's partial writes cannot race
// the previous token's inverse read.
template <uint HeadsPerGroup, uint HeadDim, uint StateSize,
          uint Rows = RICHENGINE_TARGET_VERIFY_ROWS>
struct Mamba2DecodeShared {
  // A group's post-conv x|B|C rows stage in threadgroup memory only when the
  // whole struct fits the 32KB budget: Bolt's sixteen-head groups would take
  // 43KB at sixteen rows, so there the scan reads the values back out of
  // `mixed` (the prologue writes every post-conv value there for the commit
  // regardless) and only the small per-head scalars stay here. The arrays
  // keep a one-element floor so the unstaged layout still types.
  // Metal lodges no static data members, so the fit arithmetic rides
  // enumerators.
  enum : uint {
    kStagedBytes = Rows * HeadsPerGroup * HeadDim * 2 +
                   Rows * 2 * StateSize * 2 + Rows * HeadsPerGroup * 4 * 2 +
                   HeadsPerGroup * 4 + 2 * kMamba2DecodeSimdgroups * 4,
    kStageOperands = kStagedBytes <= 32768 ? 1u : 0u
  };
  bfloat x[kStageOperands ? Rows * HeadsPerGroup * HeadDim : 1];
  bfloat bc[kStageOperands ? Rows * 2 * StateSize : 1];
  float step[Rows * HeadsPerGroup];
  float decay[Rows * HeadsPerGroup];
  float skip[HeadsPerGroup];
  float reductions[2 * kMamba2DecodeSimdgroups];
};

// The conv channel of a group's e-th owned channel: x covers its heads'
// [group*GroupWidth, +GroupWidth), B the group's slice at
// [Inner + group*State, +State), C at [Inner + Groups*State + group*State).
template <uint InnerSize, uint Groups, uint StateSize>
inline uint mamba2_group_channel(uint group, uint e, uint group_width) {
  return e < group_width
             ? group * group_width + e
             : e < group_width + StateSize
                   ? InnerSize + group * StateSize + e - group_width
                   : InnerSize + Groups * StateSize + group * StateSize + e -
                         group_width - StateSize;
}

// dt' = clamp(softplus(dt + bias), limits) for one (token, head), rounding
// like the bf16 reference: the bias add and the softplus land in bf16, the
// dt_limit clamp stays fp32.
inline float mamba2_step_size(float dt_raw, float bias, float dt_min,
                              float dt_max) {
  const bfloat x = bfloat(dt_raw + bias);
  const float xf = float(x);
  const float softplus =
      float(bfloat(max(xf, 0.0f) +
                   fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                       0.69314718056f));
  return clamp(softplus, dt_min, dt_max);
}

// Rows' conv+SiLU, dt'/dA evaluation and the conv carry for one group.
// Post-conv x lands in shared.x, B|C in shared.bc, all three also in `mixed`
// for the commit; dt' and dA in shared.step/decay and dt' in `step_sizes`
// for the commit and the tape. With a live tape, the pending rows' raw conv
// inputs extend the conv window, their post-conv rows advance the state in
// mamba2_decode_scan, and the step's rows write to step_tape.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_decode_prologue(
    device const bfloat *conv_rows, device const bfloat *packed,
    device const bfloat *conv_weights, device const bfloat *conv_bias,
    device const bfloat *conv_state_in,
    device bfloat *conv_state_out, device bfloat *mixed,
    device const float *a_weights, device const bfloat *dt_bias,
    device float *step_sizes, device float *decay_values,
    device const float *skip_weights,
    device const bfloat *tape_conv, uint pending, device uchar *step_tape,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize> &shared,
    float dt_min, float dt_max, uint group, uint lane, uint simd_group,
    uint tokens) {
  const uint Tokens = tokens;
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  constexpr uint GroupChannels = GroupWidth + 2 * StateSize;
  constexpr uint Rows = RICHENGINE_TARGET_VERIFY_ROWS;
  static_assert(!(RICHENGINE_TARGET_VERIFY_ROWS % kMamba2DecodeSimdgroups),
                "rows stride over the simdgroups");
  float a_w = 0.0f;
  if (lane < HeadsPerGroup)
    a_w = a_weights[group * HeadsPerGroup + lane];
  for (uint token = simd_group; token < Tokens;
       token += kMamba2DecodeSimdgroups) {
    for (uint e = lane; e < GroupChannels; e += 32) {
      const uint channel =
          mamba2_group_channel<InnerSize, Groups, StateSize>(group, e,
                                                             GroupWidth);
      const bfloat value =
          gdn_conv_silu_pending(conv_rows, conv_state_in, tape_conv, pending,
                                conv_weights, PackedWidth, ConvDim, token,
                                channel, conv_bias);
      if constexpr (Mamba2DecodeShared<HeadsPerGroup, HeadDim,
                                       StateSize>::kStageOperands) {
        if (e < GroupWidth)
          shared.x[token * GroupWidth + e] = value;
        else
          shared.bc[token * 2 * StateSize + e - GroupWidth] = value;
      }
      mixed[token * ConvDim + channel] = value;
    }
    // dt' per head of the group: lanes 0..HeadsPerGroup-1 own one each.
    if (lane < HeadsPerGroup) {
      const uint head = group * HeadsPerGroup + lane;
      const float dtv = mamba2_step_size(
          float(packed[token * PackedWidth + InnerSize + ConvDim + head]),
          float(dt_bias[head]), dt_min, dt_max);
      shared.step[token * HeadsPerGroup + lane] = dtv;
      const float decay = fast::exp(dtv * a_w);
      shared.decay[token * HeadsPerGroup + lane] = decay;
      step_sizes[token * Heads + head] = dtv;
      if (decay_values)
        decay_values[token * Heads + head] = decay;
      if (token == 0)
        shared.skip[lane] = skip_weights[head];
    }
    // The step's tape row: the raw conv inputs, the post-conv x|B|C and the
    // heads' dt' — the next step's pending prefix replays them.
    if (step_tape) {
      device bfloat *raw = reinterpret_cast<device bfloat *>(step_tape) +
                           ulong(token) * ConvDim;
      device bfloat *post = reinterpret_cast<device bfloat *>(
                                step_tape + ulong(Rows) * ConvDim * 2) +
                            ulong(token) * ConvDim;
      for (uint e = lane; e < GroupChannels; e += 32) {
        const uint channel =
            mamba2_group_channel<InnerSize, Groups, StateSize>(group, e,
                                                               GroupWidth);
        raw[channel] = conv_rows[token * PackedWidth + channel];
        // Unstaged groups read the post-conv value back out of `mixed`;
        // this thread wrote it above, so program order carries it.
        if constexpr (Mamba2DecodeShared<HeadsPerGroup, HeadDim,
                                         StateSize>::kStageOperands) {
          post[channel] = e < GroupWidth
                              ? shared.x[token * GroupWidth + e]
                              : shared
                                    .bc[token * 2 * StateSize + e - GroupWidth];
        } else {
          post[channel] = mixed[token * ConvDim + channel];
        }
      }
      if (lane < HeadsPerGroup)
        reinterpret_cast<device float *>(step_tape +
                                         2 * ulong(Rows) * ConvDim * 2)
            [token * Heads + group * HeadsPerGroup + lane] =
            shared.step[token * HeadsPerGroup + lane];
    }
  }
  // The conv carry: the last three raw inputs of the consumed prefix —
  // the pending rows when the commit is folded (the step's own rows wait for
  // the next step's tape), all rows otherwise.
  if (simd_group < 3) {
    const uint row = simd_group;
    const uint consumed = step_tape ? pending : Tokens;
    for (uint e = lane; e < GroupChannels; e += 32) {
      const uint channel =
          mamba2_group_channel<InnerSize, Groups, StateSize>(group, e,
                                                             GroupWidth);
      conv_state_out[row * ConvDim + channel] =
          gdn_conv_carry_pending(conv_rows, conv_state_in, tape_conv, pending,
                                 PackedWidth, ConvDim, consumed, row, channel);
    }
  }
}

// The SSD recurrence over the group's HeadsPerGroup*HeadDim state rows, four
// fp32 columns per lane. Per token: S = dA*S + (dt'*x_d) B_n', then the
// emission y_d = sum_n C_n S_dn + D_h x_d written to `hidden` (the gate
// phase normalizes it in place afterwards). The pending prefix (taped
// folds) advances the state without emitting, reading post-conv rows and
// dt' from the tape. RowsInFlight rows advance together to overlap their
// reductions and arithmetic; the next batch's state loads before this
// batch's stores.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint RowsInFlight>
inline void mamba2_decode_scan(
    device const float *state_in, device float *state_out,
    device const uchar *tape, device const float *a_weights,
    device const bfloat *mixed, device bfloat *hidden,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize> &shared,
    uint group, uint lane, uint simd_group, uint tokens, uint pending,
    bool tape_step) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  constexpr uint Rows = RICHENGINE_TARGET_VERIFY_ROWS;
  constexpr uint Batches = GroupWidth / kMamba2DecodeSimdgroups;
  constexpr bool kStage =
      Mamba2DecodeShared<HeadsPerGroup, HeadDim, StateSize>::kStageOperands;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the group");
  // One token's post-conv operands: threadgroup-staged when they fit, the
  // prologue's `mixed` writes otherwise (the barrier after the prologue
  // carries mem_device in that layout, and the same addresses hit the
  // threadgroup's cache lines across the scan's batches).
  const auto x_at = [&](uint token, uint row) -> bfloat {
    if constexpr (kStage)
      return shared.x[token * GroupWidth + row];
    else
      return mixed[ulong(token) * ConvDim +
                   mamba2_group_channel<InnerSize, Groups, StateSize>(
                       group, row, GroupWidth)];
  };
  const auto bc_at = [&](uint token, uint index) -> bfloat {
    if constexpr (kStage)
      return shared.bc[token * 2 * StateSize + index];
    else
      return mixed[ulong(token) * ConvDim +
                   mamba2_group_channel<InnerSize, Groups, StateSize>(
                       group, GroupWidth + index, GroupWidth)];
  };
  // Row r covers (head, dim) = (r / HeadDim, r % HeadDim); r = batch*8+sg
  // walks every (head, dim) pair of the group.
  const auto base = [&](uint batch, uint r) {
    const uint row = (batch + r) * kMamba2DecodeSimdgroups + simd_group;
    return (ulong(group) * GroupWidth + row) * StateSize + lane * 4;
  };
  // The pending region's post-conv x|B|C and dt' sections.
  device const bfloat *tape_x = reinterpret_cast<device const bfloat *>(
                                    tape + ulong(Rows) * ConvDim * 2) +
                                ulong(group) * GroupWidth;
  device const bfloat *tape_bc = reinterpret_cast<device const bfloat *>(
                                     tape + ulong(Rows) * ConvDim * 2) +
                                 ulong(InnerSize) + ulong(group) * StateSize;
  device const float *tape_dt = reinterpret_cast<device const float *>(
      tape + 2 * ulong(Rows) * ConvDim * 2);

  float state[RowsInFlight][4];
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i)
      state[r][i] = state_in[base(0, r) + i];
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    uint rows[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r)
      rows[r] = (batch + r) * kMamba2DecodeSimdgroups + simd_group;
    float a_w[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r)
      a_w[r] = a_weights[group * HeadsPerGroup + rows[r] / HeadDim];
    for (uint token = 0; token < pending; ++token) {
      for (uint r = 0; r < RowsInFlight; ++r) {
        const uint head = rows[r] / HeadDim;
        const float dtv =
            tape_dt[token * Heads + group * HeadsPerGroup + head];
        const float dA = fast::exp(dtv * a_w[r]);
        const float scale = dtv * float(tape_x[token * ConvDim + rows[r]]);
        for (uint i = 0; i < 4; ++i)
          state[r][i] =
              fma(scale, float(tape_bc[token * ConvDim + lane * 4 + i]),
                  state[r][i] * dA);
      }
    }
    float upcoming[RowsInFlight][4] = {};
    if (batch + RowsInFlight < Batches) {
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          upcoming[r][i] = state_in[base(batch + RowsInFlight, r) + i];
    }
    // The committed state is the incoming state advanced over the pending
    // rows only; the step's own rows emit but their advance waits on the
    // next step (they may be trimmed there). Unfolded, the scan still leaves
    // the provisional all-rows state — the standalone commit decides.
    float committed[RowsInFlight][4];
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        committed[r][i] = state[r][i];
    for (uint token = 0; token < tokens; ++token) {
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const uint head = rows[r] / HeadDim;
        const float dA = shared.decay[token * HeadsPerGroup + head];
        const float x = float(x_at(token, rows[r]));
        const float scale = shared.step[token * HeadsPerGroup + head] * x;
        float partial = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          state[r][i] =
              fma(scale,
                  float(bc_at(token, lane * 4 + i)),
                  state[r][i] * dA);
          partial +=
              state[r][i] *
              float(bc_at(token, StateSize + lane * 4 + i));
        }
        result[r] = simd_sum(partial) + shared.skip[head] * x;
      }
      if (lane == 0) {
        for (uint r = 0; r < RowsInFlight; ++r)
          hidden[token * InnerSize + group * GroupWidth + rows[r]] =
              bfloat(result[r]);
      }
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state_out[base(batch, r) + i] =
            tape_step ? committed[r][i] : state[r][i];
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i)
        state[r][i] = upcoming[r][i];
  }
}

// The gated group RMSNorm over one token's scan rows. `packed` and
// `lane_hidden` are already batch-offset; `table` is not, matching the fused
// phase's table write. Split out so the wide Bolt scan can run as its own
// dispatch while this phase keeps the fused kernel's exact operation order.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize,
          uint PackedWidth, class Table, class W, class TP, class SP>
inline void mamba2_decode_gate_phase(
    device const bfloat *packed, device const W *norm_weight,
    device bfloat *lane_hidden, uint stride, uint batch, float rms_eps,
    uint group, uint lane, uint simd_group,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize> &shared,
    TP *table, SP *sums) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  const uint thread_index = simd_group * 32 + lane;
  // A 512-column group takes one adjacent pair per thread. Bolt's groups
  // are two such spans; the second span's tiles sit eight heads later.
  constexpr uint GateSpans = GroupWidth / 512;
  static_assert(GroupWidth % 512 == 0,
                "Mamba2 groups fill whole 512-column spans");
  for (uint token = 0; token < stride; ++token) {
    float versus[2 * GateSpans];
    float sum = 0.0f;
    for (uint span = 0; span < GateSpans; ++span) {
      const uint dim = span * 512 + 2 * thread_index;
      const uint column = group * GroupWidth + dim;
      const float y0 = float(lane_hidden[token * InnerSize + column]);
      const float y1 = float(lane_hidden[token * InnerSize + column + 1]);
      const float z0 = float(packed[token * PackedWidth + column]);
      const float z1 = float(packed[token * PackedWidth + column + 1]);
      versus[2 * span] = y0 * richengine_silu(z0);
      versus[2 * span + 1] = y1 * richengine_silu(z1);
      sum += versus[2 * span] * versus[2 * span] +
             versus[2 * span + 1] * versus[2 * span + 1];
    }
    const float inverse = rms_inverse_of_sums<kMamba2DecodeSimdgroups>(
        sum, GroupWidth,
        shared.reductions + (token & 1) * kMamba2DecodeSimdgroups,
        thread_index, lane, simd_group, rms_eps);
    for (uint span = 0; span < GateSpans; ++span) {
      const uint dim = span * 512 + 2 * thread_index;
      const uint column = group * GroupWidth + dim;
      const bfloat gated0 =
          bfloat((versus[2 * span] * inverse) * float(norm_weight[column]));
      const bfloat gated1 =
          bfloat((versus[2 * span + 1] * inverse) *
                 float(norm_weight[column + 1]));
      lane_hidden[token * InnerSize + column] = gated0;
      lane_hidden[token * InnerSize + column + 1] = gated1;
      if (table) {
        // The thread's adjacent pair of dims is its table slot's pair: the
        // simdgroup covers this group's (g*heads+span*8+sg)-th 64-column
        // tile, lane the (2*lane, 2*lane+1) elements.
        Table::write_row(
            table + ulong(batch) * InnerSize * stride,
            sums + ulong(batch) * (stride / q4sg::kRows) *
                       Table::sums_per_tile(InnerSize),
            InnerSize,
            group * HeadsPerGroup + span * kMamba2DecodeSimdgroups +
                simd_group,
            token, lane, gated0, gated1);
      }
    }
  }
}

// Grid {groups, lanes}; each lane's rows of a layer's packed, mixed, dt' and
// hidden tensors follow the four lanes of the layer before.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth, uint RowsInFlight, class Table, class W,
          class TP, class SP>
inline void mamba2_decode_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_bias,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device bfloat *mixed, device const float *a_weights,
    device const bfloat *dt_bias, device float *step_sizes,
    device const float *skip_weights, device const W *norm_weight,
    device bfloat *hidden, constant Mamba2DecodeBatchParams &params,
    uint2 group, uint lane, uint simd_group,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize> &shared,
    TP *table, SP *sums, device uchar *tape_out) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  const uint batch = group.y;
  device const uchar *current = batch == 0
      ? current0
      : (batch == 1 ? current1 : (batch == 2 ? current2 : current3));
  device uchar *next = batch == 0
      ? next0
      : (batch == 1 ? next1 : (batch == 2 ? next2 : next3));
  const uint stride = params.verify_stride;
  packed += ulong(batch) * stride * PackedWidth;
  mixed += ulong(batch) * stride * ConvDim;
  step_sizes += ulong(batch) * stride * Heads;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device bfloat *lane_hidden = hidden + ulong(batch) * stride * InnerSize;
  // The conv columns sit InnerSize into the packed row.
  device const bfloat *conv_rows = packed + InnerSize;

  // Adaptive proposal budgets bound the conv's token sweep and the serial
  // scan; a zero or full count runs the lane's whole verify stride. Rows
  // past the count keep stale hidden values and are never committed or
  // selected.
  const uint live = params.live_rows[batch];
  const uint tokens = live && live < stride ? live : stride;
  const uint pending = params.pending_rows[batch];
  // The previous step's retained rows sit on the lane's tape; MAMBA2_TAPE_FOLD
  // gates every tape access: an unfolded dispatch leaves the slot unbound and
  // the shared argument table's stale address there must never be
  // dereferenced.
  const bool taped =
      tape_out != nullptr && (params.flags & MAMBA2_TAPE_FOLD) != 0;
  device const uchar *tape =
      taped ? tape_out + params.pending_tape[batch]
            : (device const uchar *)nullptr;
  device const bfloat *tape_conv = reinterpret_cast<device const bfloat *>(tape);

  mamba2_decode_prologue<Heads, HeadDim, Groups, StateSize, ConvDim,
                         PackedWidth>(
      conv_rows, packed, conv_weights, conv_bias, conv_state_in,
      conv_state_out, mixed,
      a_weights, dt_bias, step_sizes, (device float *)nullptr, skip_weights,
      tape_conv, pending,
      taped ? tape_out + params.step_tape[batch] : (device uchar *)nullptr,
      shared, params.dt_min, params.dt_max, group.x, lane, simd_group, tokens);
  // Unstaged groups read the prologue's `mixed` writes back in the scan, so
  // their barrier must publish device memory across the threadgroup; staged
  // groups keep the threadgroup-only barrier.
  constexpr bool kStage =
      Mamba2DecodeShared<HeadsPerGroup, HeadDim, StateSize>::kStageOperands;
  threadgroup_barrier(kStage ? mem_flags::mem_threadgroup
                             : (mem_flags::mem_threadgroup |
                                mem_flags::mem_device),
                      memory_order_acq_rel);
  mamba2_decode_scan<Heads, HeadDim, Groups, StateSize, ConvDim, RowsInFlight>(
      state_in, state_out, tape, a_weights, mixed, lane_hidden, shared,
      group.x, lane, simd_group, tokens, pending, taped);
  // The gate reads the scan's hidden writes across simdgroups.
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device,
                      memory_order_acq_rel);
  mamba2_decode_gate_phase<Heads, HeadDim, Groups, StateSize, PackedWidth,
                           Table, W>(
      packed, norm_weight, lane_hidden, stride, batch, params.rms_eps, group.x,
      lane, simd_group, shared, table, sums);
}

// W: the norm weights' stored type (float: a GGUF's F32 norms, _f32).
#define MAMBA2_DECODE_BUFFERS(W)                                                \
  device const bfloat *packed [[buffer(0)]],                                    \
      device const bfloat *conv_weights [[buffer(1)]],                          \
      device const uchar *current0 [[buffer(2)]],                               \
      device const uchar *current1 [[buffer(3)]],                               \
      device const uchar *current2 [[buffer(4)]],                               \
      device const uchar *current3 [[buffer(5)]],                               \
      device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],     \
      device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],     \
      device bfloat *mixed [[buffer(10)]],                                      \
      device const float *a_weights [[buffer(11)]],                             \
      device const bfloat *dt_bias [[buffer(12)]],                              \
      device float *step_sizes [[buffer(13)]],                                  \
      device const float *skip_weights [[buffer(14)]],                          \
      device const W *norm_weight [[buffer(15)]],                               \
      device bfloat *hidden [[buffer(16)]],                                     \
      device const bfloat *conv_bias [[buffer(21)]]
#define MAMBA2_DECODE_THREADS                                                   \
  uint2 group [[threadgroup_position_in_grid]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd_group [[simdgroup_index_in_threadgroup]]
#define MAMBA2_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim,         \
                           PackedWidth, table, sums, Layout)                   \
  threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize> shared;    \
  mamba2_decode_batch_phase<Heads, HeadDim, Groups, StateSize, ConvDim,        \
                            PackedWidth, 2, Layout>(                           \
      packed, conv_weights, conv_bias, current0, current1, current2,           \
      current3, next0, next1, next2, next3, mixed, a_weights, dt_bias,         \
      step_sizes, skip_weights, norm_weight, hidden, params, group, lane,      \
      simd_group, shared, table, sums, tape);
// Entries without a table pass null pointers, which skip the write; their
// Layout only completes the template.
#define MAMBA2_DECODE_ENTRY(Name, Heads, HeadDim, Groups, StateSize, ConvDim,  \
                            PackedWidth, W)                                    \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void Name(MAMBA2_DECODE_BUFFERS(W),                                   \
                   constant Mamba2DecodeBatchParams &params [[buffer(17)]],    \
                   device uchar *tape [[buffer(20)]], MAMBA2_DECODE_THREADS) { \
    MAMBA2_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim, PackedWidth,\
                       (device bfloat *)nullptr, (device float *)nullptr,      \
                       q4sg::Table64)                                          \
  }
// The out-projection's table (Layout: q4sg::Table64 affine, gguf_sg::Table16
// GGUF, gguf_sg::Packed the mxfp4p A-operand).
#define MAMBA2_DECODE_TABLE_ENTRY(Name, Heads, HeadDim, Groups, StateSize,     \
                                  ConvDim, PackedWidth, Layout, W)             \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void Name(MAMBA2_DECODE_BUFFERS(W),                                   \
                   device bfloat *table [[buffer(17)]],                        \
                   device float *sums [[buffer(18)]],                          \
                   constant Mamba2DecodeBatchParams &params [[buffer(19)]],    \
                   device uchar *tape [[buffer(20)]], MAMBA2_DECODE_THREADS) { \
    MAMBA2_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim, PackedWidth,\
                       table, sums, Layout)                                    \
  }
// The mxfp4p-operand variant (LinearInput::Packed): the fp16 plane and the
// exponent bytes take the table and sums slots; gguf_sg::Packed::write shares
// the table path's lane partitioning (elements 2*lane, 2*lane+1 of a
// 64-column span). F32 norms only, like Table16.
#define MAMBA2_DECODE_PACKED_ENTRY(Name, Heads, HeadDim, Groups, StateSize,    \
                                   ConvDim, PackedWidth, W)                    \
  [[max_total_threads_per_threadgroup(256)]]                                   \
  kernel void Name(MAMBA2_DECODE_BUFFERS(W),                                   \
                   device half *plane [[buffer(17)]],                          \
                   device uchar *exponents [[buffer(18)]],                     \
                   constant Mamba2DecodeBatchParams &params [[buffer(19)]],    \
                   device uchar *tape [[buffer(20)]], MAMBA2_DECODE_THREADS) { \
    MAMBA2_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim, PackedWidth,\
                       plane, exponents, gguf_sg::Packed)                      \
  }

// Nemotron 3.5's mamba2 geometry: 64 heads of 64 dims in 8 groups of 128
// states, 6144 conv channels, 10496 packed.
MAMBA2_DECODE_ENTRY(verify_mamba2_fused, 64, 64, 8, 128, 6144, 10496, bfloat)
MAMBA2_DECODE_ENTRY(verify_mamba2_fused_f32, 64, 64, 8, 128, 6144, 10496, float)
// Table64 feeds the affine models, whose norms are bf16; Table16 a GGUF's.
MAMBA2_DECODE_TABLE_ENTRY(verify_mamba2_fused_table64, 64, 64, 8, 128, 6144,
                          10496, q4sg::Table64, bfloat)
MAMBA2_DECODE_TABLE_ENTRY(verify_mamba2_fused_table16_f32, 64, 64, 8, 128,
                          6144, 10496, gguf_sg::Table16, float)
MAMBA2_DECODE_TABLE_ENTRY(verify_mamba2_fused_table16, 64, 64, 8, 128, 6144,
                          10496, gguf_sg::Table16, bfloat)
MAMBA2_DECODE_PACKED_ENTRY(verify_mamba2_fused_packed_f32, 64, 64, 8, 128,
                           6144, 10496, float)
MAMBA2_DECODE_PACKED_ENTRY(verify_mamba2_fused_packed, 64, 64, 8, 128, 6144,
                           10496, bfloat)
// Bolt's geometry (_h128): 128 heads of 64 in 8 groups, 10240 conv channels,
// 18688 packed rows.
MAMBA2_DECODE_ENTRY(verify_mamba2_fused_h128, 128, 64, 8, 128, 10240, 18688, bfloat)
MAMBA2_DECODE_ENTRY(verify_mamba2_fused_h128_f32, 128, 64, 8, 128, 10240, 18688, float)
MAMBA2_DECODE_TABLE_ENTRY(verify_mamba2_fused_h128_table64, 128, 64, 8, 128, 10240,
                          18688, q4sg::Table64, bfloat)
MAMBA2_DECODE_TABLE_ENTRY(verify_mamba2_fused_h128_table16_f32, 128, 64, 8, 128,
                          10240, 18688, gguf_sg::Table16, float)
MAMBA2_DECODE_TABLE_ENTRY(verify_mamba2_fused_h128_table16, 128, 64, 8, 128, 10240,
                          18688, gguf_sg::Table16, bfloat)
MAMBA2_DECODE_PACKED_ENTRY(verify_mamba2_fused_h128_packed_f32, 128, 64, 8, 128,
                           10240, 18688, float)
MAMBA2_DECODE_PACKED_ENTRY(verify_mamba2_fused_h128_packed, 128, 64, 8, 128, 10240,
                           18688, bfloat)
#undef MAMBA2_DECODE_PACKED_ENTRY
#undef MAMBA2_DECODE_TABLE_ENTRY
#undef MAMBA2_DECODE_ENTRY
#undef MAMBA2_DECODE_BODY
#undef MAMBA2_DECODE_THREADS
#undef MAMBA2_DECODE_BUFFERS

// ---------------------------------------------------------------------------
// Bolt's split verify decode. The fused h128 kernel leaves only 8 x lanes
// threadgroups in flight: each scans all 1024 (head, dim) rows of its norm
// group in batches of sixteen. The recurrence is sequential only across
// verify rows, so the split scan gives every sixteen-row batch its own
// threadgroup — 64 x groups x lanes threadgroups, with no threadgroup memory
// and the same simdgroup reduction shape as the fused scan. The prologue and
// gate keep the fused kernel's group grids and operation order.
// ---------------------------------------------------------------------------

template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_decode_split_prepare_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device bfloat *mixed, device const float *a_weights,
    device const bfloat *dt_bias, device float *step_sizes,
    device float *decay_values, device const float *skip_weights,
    constant Mamba2DecodeBatchParams &params, device uchar *tape_out,
    device const bfloat *conv_bias, uint2 group, uint lane, uint simd_group,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize> &shared) {
  constexpr uint InnerSize = Heads * HeadDim;
  const uint batch = group.y;
  device const uchar *current = batch == 0
      ? current0
      : (batch == 1 ? current1 : (batch == 2 ? current2 : current3));
  device uchar *next = batch == 0
      ? next0
      : (batch == 1 ? next1 : (batch == 2 ? next2 : next3));
  const uint stride = params.verify_stride;
  packed += ulong(batch) * stride * PackedWidth;
  mixed += ulong(batch) * stride * ConvDim;
  step_sizes += ulong(batch) * stride * Heads;
  decay_values += ulong(batch) * stride * Heads;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(params.layer) * params.conv_layer_bytes);
  device const bfloat *conv_rows = packed + InnerSize;
  const uint live = params.live_rows[batch];
  const uint tokens = live && live < stride ? live : stride;
  const uint pending = params.pending_rows[batch];
  const bool taped =
      tape_out != nullptr && (params.flags & MAMBA2_TAPE_FOLD) != 0;
  device const uchar *tape =
      taped ? tape_out + params.pending_tape[batch]
            : (device const uchar *)nullptr;
  mamba2_decode_prologue<Heads, HeadDim, Groups, StateSize, ConvDim,
                         PackedWidth>(
      conv_rows, packed, conv_weights, conv_bias, conv_state_in,
      conv_state_out, mixed, a_weights, dt_bias, step_sizes, decay_values,
      skip_weights, reinterpret_cast<device const bfloat *>(tape), pending,
      taped ? tape_out + params.step_tape[batch] : (device uchar *)nullptr,
      shared, params.dt_min, params.dt_max, group.x, lane, simd_group, tokens);
}

// One scan tile: position.x selects RowsInFlight * 8 adjacent (head, dim)
// rows of the norm group at position.z, position.y the batch lane. Every row
// keeps the fused scan's lane layout: lane n owns state columns 4n..4n+3 and
// simd_sum reduces those partials in the same order.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint RowsInFlight, uint RowsPerThreadgroup>
inline void mamba2_decode_split_scan_tile(
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device const bfloat *mixed,
    device const float *a_weights, device const float *step_sizes,
    device const float *decay_values, device const float *skip_weights,
    device bfloat *hidden,
    constant Mamba2DecodeBatchParams &params, device uchar *tape_out,
    uint3 position, uint lane, uint simd_group) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  constexpr uint Rows = RICHENGINE_TARGET_VERIFY_ROWS;
  static_assert(RowsPerThreadgroup ==
                    RowsInFlight * kMamba2DecodeSimdgroups,
                "a scan tile fills the threadgroup's simdgroups");
  static_assert(GroupWidth % RowsPerThreadgroup == 0,
                "scan rows tile the group");
  const uint batch = position.y;
  const uint group = position.z;
  device const uchar *current = batch == 0
      ? current0
      : (batch == 1 ? current1 : (batch == 2 ? current2 : current3));
  device uchar *next = batch == 0
      ? next0
      : (batch == 1 ? next1 : (batch == 2 ? next2 : next3));
  const uint stride = params.verify_stride;
  mixed += ulong(batch) * stride * ConvDim;
  step_sizes += ulong(batch) * stride * Heads;
  decay_values += ulong(batch) * stride * Heads;
  device bfloat *lane_hidden = hidden + ulong(batch) * stride * InnerSize;
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  const uint live = params.live_rows[batch];
  const uint tokens = live && live < stride ? live : stride;
  const uint pending = params.pending_rows[batch];
  const bool taped =
      tape_out != nullptr && (params.flags & MAMBA2_TAPE_FOLD) != 0;
  device const uchar *tape =
      taped ? tape_out + params.pending_tape[batch]
            : (device const uchar *)nullptr;
  device const bfloat *tape_x = reinterpret_cast<device const bfloat *>(
                                    tape + ulong(Rows) * ConvDim * 2) +
                                ulong(group) * GroupWidth;
  device const bfloat *tape_bc = reinterpret_cast<device const bfloat *>(
                                     tape + ulong(Rows) * ConvDim * 2) +
                                 ulong(InnerSize) + ulong(group) * StateSize;
  device const float *tape_dt = reinterpret_cast<device const float *>(
      tape + 2 * ulong(Rows) * ConvDim * 2);

  uint rows[RowsInFlight];
  float a_w[RowsInFlight];
  float skip[RowsInFlight];
  for (uint r = 0; r < RowsInFlight; ++r) {
    rows[r] = position.x * RowsPerThreadgroup +
              r * kMamba2DecodeSimdgroups + simd_group;
    const uint head = rows[r] / HeadDim;
    a_w[r] = a_weights[group * HeadsPerGroup + head];
    skip[r] = skip_weights[group * HeadsPerGroup + head];
  }
  const auto base = [&](uint r) {
    return (ulong(group) * GroupWidth + rows[r]) * StateSize + lane * 4;
  };
  const auto x_at = [&](uint token, uint row) -> bfloat {
    return mixed[ulong(token) * ConvDim + ulong(group) * GroupWidth + row];
  };
  const auto bc_at = [&](uint token, uint index) -> bfloat {
    return mixed[ulong(token) * ConvDim +
                 mamba2_group_channel<InnerSize, Groups, StateSize>(
                     group, GroupWidth + index, GroupWidth)];
  };

  float state[RowsInFlight][4];
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i)
      state[r][i] = state_in[base(r) + i];
  for (uint token = 0; token < pending; ++token) {
    for (uint r = 0; r < RowsInFlight; ++r) {
      const uint head = rows[r] / HeadDim;
      const float dtv =
          tape_dt[token * Heads + group * HeadsPerGroup + head];
      const float dA = fast::exp(dtv * a_w[r]);
      const float scale = dtv * float(tape_x[token * ConvDim + rows[r]]);
      for (uint i = 0; i < 4; ++i)
        state[r][i] =
            fma(scale, float(tape_bc[token * ConvDim + lane * 4 + i]),
                state[r][i] * dA);
    }
  }
  float committed[RowsInFlight][4];
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i)
      committed[r][i] = state[r][i];
  for (uint token = 0; token < tokens; ++token) {
    float result[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r) {
      const uint head = rows[r] / HeadDim;
      const uint full_head = group * HeadsPerGroup + head;
      const float dtv = step_sizes[token * Heads + full_head];
      const float dA = decay_values[token * Heads + full_head];
      const float x = float(x_at(token, rows[r]));
      const float scale = dtv * x;
      float partial = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        state[r][i] =
            fma(scale, float(bc_at(token, lane * 4 + i)),
                state[r][i] * dA);
        partial +=
            state[r][i] *
            float(bc_at(token, StateSize + lane * 4 + i));
      }
      result[r] = simd_sum(partial) + skip[r] * x;
    }
    if (lane == 0) {
      for (uint r = 0; r < RowsInFlight; ++r)
        lane_hidden[token * InnerSize + group * GroupWidth + rows[r]] =
            bfloat(result[r]);
    }
  }
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i)
      state_out[base(r) + i] = taped ? committed[r][i] : state[r][i];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void verify_mamba2_prepare_h128(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *conv_weights [[buffer(1)]],
    device const uchar *current0 [[buffer(2)]],
    device const uchar *current1 [[buffer(3)]],
    device const uchar *current2 [[buffer(4)]],
    device const uchar *current3 [[buffer(5)]],
    device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
    device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
    device bfloat *mixed [[buffer(10)]],
    device const float *a_weights [[buffer(11)]],
    device const bfloat *dt_bias [[buffer(12)]],
    device float *step_sizes [[buffer(13)]],
    device float *decay_values [[buffer(14)]],
    device const float *skip_weights [[buffer(15)]],
    constant Mamba2DecodeBatchParams &params [[buffer(16)]],
    device uchar *tape [[buffer(17)]],
    device const bfloat *conv_bias [[buffer(18)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup Mamba2DecodeShared<16, 64, 128> shared;
  mamba2_decode_split_prepare_phase<128, 64, 8, 128, 10240, 18688>(
      packed, conv_weights, current0, current1, current2, current3, next0,
      next1, next2, next3, mixed, a_weights, dt_bias, step_sizes,
      decay_values, skip_weights, params, tape, conv_bias, group, lane,
      simd_group, shared);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void verify_mamba2_scan_h128(
    device const uchar *current0 [[buffer(0)]],
    device const uchar *current1 [[buffer(1)]],
    device const uchar *current2 [[buffer(2)]],
    device const uchar *current3 [[buffer(3)]],
    device uchar *next0 [[buffer(4)]], device uchar *next1 [[buffer(5)]],
    device uchar *next2 [[buffer(6)]], device uchar *next3 [[buffer(7)]],
    device const bfloat *mixed [[buffer(8)]],
    device const float *a_weights [[buffer(9)]],
    device const float *step_sizes [[buffer(10)]],
    device const float *decay_values [[buffer(11)]],
    device const float *skip_weights [[buffer(12)]],
    device bfloat *hidden [[buffer(13)]],
    constant Mamba2DecodeBatchParams &params [[buffer(14)]],
    device uchar *tape [[buffer(15)]],
    uint3 position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  mamba2_decode_split_scan_tile<128, 64, 8, 128, 10240, 2, 16>(
      current0, current1, current2, current3, next0, next1, next2, next3,
      mixed, a_weights, step_sizes, decay_values, skip_weights, hidden,
      params, tape, position, lane, simd_group);
}

#define MAMBA2_SPLIT_GATE_THREADS                                                \
  uint2 group [[threadgroup_position_in_grid]],                                 \
      uint lane [[thread_index_in_simdgroup]],                                  \
      uint simd_group [[simdgroup_index_in_threadgroup]]
#define MAMBA2_SPLIT_GATE_BODY(W, table, sums, Layout)                           \
  threadgroup Mamba2DecodeShared<16, 64, 128> shared;                            \
  const uint stride = params.verify_stride;                                      \
  packed += ulong(group.y) * stride * 18688;                                     \
  device bfloat *lane_hidden =                                                   \
      hidden + ulong(group.y) * stride * 8192;                                   \
  mamba2_decode_gate_phase<128, 64, 8, 128, 18688, Layout, W>(                  \
      packed, norm_weight, lane_hidden, stride, group.y, params.rms_eps,         \
      group.x, lane, simd_group, shared, table, sums)
#define MAMBA2_SPLIT_GATE_ENTRY(Name, W)                                         \
  [[max_total_threads_per_threadgroup(256)]]                                    \
  kernel void Name(device const bfloat *packed [[buffer(0)]],                   \
                   device const W *norm_weight [[buffer(1)]],                   \
                   device bfloat *hidden [[buffer(2)]],                         \
                   constant Mamba2DecodeBatchParams &params [[buffer(3)]],      \
                   MAMBA2_SPLIT_GATE_THREADS) {                                 \
    MAMBA2_SPLIT_GATE_BODY(W, (device bfloat *)nullptr,                         \
                           (device float *)nullptr, q4sg::Table64);             \
  }
#define MAMBA2_SPLIT_GATE_TABLE_ENTRY(Name, Layout, W)                           \
  [[max_total_threads_per_threadgroup(256)]]                                    \
  kernel void Name(device const bfloat *packed [[buffer(0)]],                   \
                   device const W *norm_weight [[buffer(1)]],                   \
                   device bfloat *hidden [[buffer(2)]],                         \
                   device bfloat *table [[buffer(3)]],                          \
                   device float *sums [[buffer(4)]],                            \
                   constant Mamba2DecodeBatchParams &params [[buffer(5)]],      \
                   MAMBA2_SPLIT_GATE_THREADS) {                                 \
    MAMBA2_SPLIT_GATE_BODY(W, table, sums, Layout);                             \
  }
#define MAMBA2_SPLIT_GATE_PACKED_ENTRY(Name, W)                                  \
  [[max_total_threads_per_threadgroup(256)]]                                    \
  kernel void Name(device const bfloat *packed [[buffer(0)]],                   \
                   device const W *norm_weight [[buffer(1)]],                   \
                   device bfloat *hidden [[buffer(2)]],                         \
                   device half *plane [[buffer(3)]],                            \
                   device uchar *exponents [[buffer(4)]],                       \
                   constant Mamba2DecodeBatchParams &params [[buffer(5)]],      \
                   MAMBA2_SPLIT_GATE_THREADS) {                                 \
    MAMBA2_SPLIT_GATE_BODY(W, plane, exponents, gguf_sg::Packed);               \
  }

MAMBA2_SPLIT_GATE_ENTRY(verify_mamba2_gate_h128, bfloat)
MAMBA2_SPLIT_GATE_ENTRY(verify_mamba2_gate_h128_f32, float)
MAMBA2_SPLIT_GATE_TABLE_ENTRY(verify_mamba2_gate_h128_table64,
                              q4sg::Table64, bfloat)
MAMBA2_SPLIT_GATE_TABLE_ENTRY(verify_mamba2_gate_h128_table16_f32,
                              gguf_sg::Table16, float)
MAMBA2_SPLIT_GATE_TABLE_ENTRY(verify_mamba2_gate_h128_table16,
                              gguf_sg::Table16, bfloat)
MAMBA2_SPLIT_GATE_PACKED_ENTRY(verify_mamba2_gate_h128_packed_f32, float)
MAMBA2_SPLIT_GATE_PACKED_ENTRY(verify_mamba2_gate_h128_packed, bfloat)
#undef MAMBA2_SPLIT_GATE_PACKED_ENTRY
#undef MAMBA2_SPLIT_GATE_TABLE_ENTRY
#undef MAMBA2_SPLIT_GATE_ENTRY
#undef MAMBA2_SPLIT_GATE_BODY
#undef MAMBA2_SPLIT_GATE_THREADS

// ---------------------------------------------------------------------------
// Tree verify: one lane's RICHENGINE_TREE_VERIFY_NODES rows, the chain at
// rows 0..7 and each position's sibling leaf at row 8 + position. Mirrors
// the GDN tree kernels: the prologue takes a row's conv taps along its path
// (tree_nodes' ancestors), the scan runs the chain and extends a register
// copy per leaf so the committed chain state is untouched, and there is no
// conv_state_out — the commit always recomputes the carry over the retained
// path. The tape fold is chain-only.
// ---------------------------------------------------------------------------

// The tree prologue: simdgroup g prepares rows g, g + 8. Rows past the
// lane's live count are zeroed so a leaf's discarded advance reads no
// stale operands.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_decode_tree_prologue(
    device const bfloat *conv_rows, device const bfloat *packed,
    device const bfloat *conv_weights, device const bfloat *conv_bias,
    device const bfloat *conv_state_in, device bfloat *mixed,
    device const float *a_weights, device const bfloat *dt_bias,
    device float *step_sizes, device const float *skip_weights,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize,
                                   RICHENGINE_TREE_VERIFY_NODES> &shared,
    device const uint *nodes, uint count, float dt_min, float dt_max,
    uint group, uint lane, uint simd_group) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  constexpr uint GroupChannels = GroupWidth + 2 * StateSize;
  constexpr uint Rows = RICHENGINE_TREE_VERIFY_NODES;
  static_assert(Mamba2DecodeShared<HeadsPerGroup, HeadDim, StateSize,
                                   Rows>::kStageOperands,
                "the tree scan stages its operands in threadgroup memory");
  float a_w = 0.0f;
  if (lane < HeadsPerGroup)
    a_w = a_weights[group * HeadsPerGroup + lane];
  for (uint token = simd_group; token < Rows;
       token += kMamba2DecodeSimdgroups) {
    if (token >= count) {
      for (uint e = lane; e < GroupChannels; e += 32) {
        if (e < GroupWidth)
          shared.x[token * GroupWidth + e] = bfloat(0.0f);
        else
          shared.bc[token * 2 * StateSize + e - GroupWidth] = bfloat(0.0f);
      }
      if (lane < HeadsPerGroup) {
        shared.step[token * HeadsPerGroup + lane] = 0.0f;
        shared.decay[token * HeadsPerGroup + lane] = 0.0f;
      }
      continue;
    }
    for (uint e = lane; e < GroupChannels; e += 32) {
      const uint channel =
          mamba2_group_channel<InnerSize, Groups, StateSize>(group, e,
                                                             GroupWidth);
      const bfloat value =
          gdn_conv_silu_tree(conv_rows, conv_state_in, conv_weights,
                             PackedWidth, ConvDim, token, channel, nodes,
                             conv_bias);
      if (e < GroupWidth)
        shared.x[token * GroupWidth + e] = value;
      else
        shared.bc[token * 2 * StateSize + e - GroupWidth] = value;
      mixed[token * ConvDim + channel] = value;
    }
    // dt' per head of the group: lanes 0..HeadsPerGroup-1 own one each.
    if (lane < HeadsPerGroup) {
      const uint head = group * HeadsPerGroup + lane;
      const float dtv = mamba2_step_size(
          float(packed[token * PackedWidth + InnerSize + ConvDim + head]),
          float(dt_bias[head]), dt_min, dt_max);
      shared.step[token * HeadsPerGroup + lane] = dtv;
      shared.decay[token * HeadsPerGroup + lane] =
          fast::exp(dtv * a_w);
      step_sizes[token * Heads + head] = dtv;
      if (token == 0)
        shared.skip[lane] = skip_weights[head];
    }
  }
}

// The tree scan: the chain rows advance the lane's state as usual; after
// each of the first eight, the position's leaf extends a register copy so
// its emission is produced without disturbing the chain's state. state_out
// still takes the chain's end state; the commit replays the retained path.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint RowsInFlight>
inline void mamba2_decode_tree_scan(
    device const float *state_in, device float *state_out,
    device bfloat *hidden,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize,
                                   RICHENGINE_TREE_VERIFY_NODES> &shared,
    uint count, uint group, uint lane, uint simd_group) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  constexpr uint ChainRows = RICHENGINE_TREE_VERIFY_NODES / 2;
  constexpr uint Batches = GroupWidth / kMamba2DecodeSimdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the group");
  const auto base = [&](uint batch, uint r) {
    const uint row = (batch + r) * kMamba2DecodeSimdgroups + simd_group;
    return (ulong(group) * GroupWidth + row) * StateSize + lane * 4;
  };
  float state[RowsInFlight][4];
  for (uint r = 0; r < RowsInFlight; ++r)
    for (uint i = 0; i < 4; ++i)
      state[r][i] = state_in[base(0, r) + i];
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    uint rows[RowsInFlight];
    for (uint r = 0; r < RowsInFlight; ++r)
      rows[r] = (batch + r) * kMamba2DecodeSimdgroups + simd_group;
    // S = dA*S + (dt'*x_d) B_n' then y_d = sum_n C_n S_dn + D_h x_d — the
    // chain advance on `state`, the leaf's on the `branch` copy.
    const auto advance = [&](float (&s)[RowsInFlight][4], uint token) {
      float result[RowsInFlight];
      for (uint r = 0; r < RowsInFlight; ++r) {
        const uint head = rows[r] / HeadDim;
        const float dA = shared.decay[token * HeadsPerGroup + head];
        const float x = float(shared.x[token * GroupWidth + rows[r]]);
        const float scale = shared.step[token * HeadsPerGroup + head] * x;
        float partial = 0.0f;
        for (uint i = 0; i < 4; ++i) {
          s[r][i] =
              fma(scale,
                  float(shared.bc[token * 2 * StateSize + lane * 4 + i]),
                  s[r][i] * dA);
          partial +=
              s[r][i] *
              float(
                  shared.bc[token * 2 * StateSize + StateSize + lane * 4 + i]);
        }
        result[r] = simd_sum(partial) + shared.skip[head] * x;
      }
      if (lane == 0) {
        for (uint r = 0; r < RowsInFlight; ++r)
          hidden[token * InnerSize + group * GroupWidth + rows[r]] =
              bfloat(result[r]);
      }
    };
    for (uint token = 0; token < ChainRows; ++token) {
      advance(state, token);
      const uint leaf = ChainRows + token;
      if (leaf < count) {
        float branch[RowsInFlight][4];
        for (uint r = 0; r < RowsInFlight; ++r)
          for (uint i = 0; i < 4; ++i)
            branch[r][i] = state[r][i];
        advance(branch, leaf);
      }
    }
    float upcoming[RowsInFlight][4] = {};
    if (batch + RowsInFlight < Batches) {
      for (uint r = 0; r < RowsInFlight; ++r)
        for (uint i = 0; i < 4; ++i)
          upcoming[r][i] = state_in[base(batch + RowsInFlight, r) + i];
    }
    for (uint r = 0; r < RowsInFlight; ++r)
      for (uint i = 0; i < 4; ++i) {
        state_out[base(batch, r) + i] = state[r][i];
        state[r][i] = upcoming[r][i];
      }
  }
}

// Grid {groups, lanes}; the lane's rows of a layer's packed, mixed, dt' and
// hidden tensors are RICHENGINE_TREE_VERIFY_NODES apart.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth, uint RowsInFlight, class Table, class W,
          class TP, class SP>
inline void mamba2_decode_tree_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_bias,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device uchar *next0, device uchar *next1, device uchar *next2,
    device uchar *next3, device bfloat *mixed, device const float *a_weights,
    device const bfloat *dt_bias, device float *step_sizes,
    device const float *skip_weights, device const W *norm_weight,
    device bfloat *hidden, device const uint *tree_nodes,
    device const uint *tree_counts,
    constant Mamba2DecodeBatchParams &params, uint2 group, uint lane,
    uint simd_group,
    threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize,
                                   RICHENGINE_TREE_VERIFY_NODES> &shared,
    TP *table, SP *sums) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint GroupWidth = HeadsPerGroup * HeadDim;
  constexpr uint Rows = RICHENGINE_TREE_VERIFY_NODES;
  const uint batch = group.y;
  device const uchar *current = batch == 0
      ? current0
      : (batch == 1 ? current1 : (batch == 2 ? current2 : current3));
  device uchar *next = batch == 0
      ? next0
      : (batch == 1 ? next1 : (batch == 2 ? next2 : next3));
  packed += ulong(batch) * Rows * PackedWidth;
  mixed += ulong(batch) * Rows * ConvDim;
  step_sizes += ulong(batch) * Rows * Heads;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device const uint *nodes = tree_nodes + batch * Rows;
  const uint count = tree_counts[batch];
  device bfloat *lane_hidden = hidden + ulong(batch) * Rows * InnerSize;
  // The conv columns sit InnerSize into the packed row.
  device const bfloat *conv_rows = packed + InnerSize;

  mamba2_decode_tree_prologue<Heads, HeadDim, Groups, StateSize, ConvDim,
                              PackedWidth>(
      conv_rows, packed, conv_weights, conv_bias, conv_state_in, mixed,
      a_weights, dt_bias, step_sizes, skip_weights, shared, nodes, count,
      params.dt_min, params.dt_max, group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  mamba2_decode_tree_scan<Heads, HeadDim, Groups, StateSize, ConvDim,
                          RowsInFlight>(state_in, state_out, lane_hidden,
                                        shared, count, group.x, lane,
                                        simd_group);
  // The gate reads the scan's hidden writes across simdgroups.
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device,
                      memory_order_acq_rel);
  const uint thread_index = simd_group * 32 + lane;
  constexpr uint GateSpans = GroupWidth / 512;
  static_assert(GroupWidth % 512 == 0,
                "Mamba2 groups fill whole 512-column spans");
  for (uint token = 0; token < Rows; ++token) {
    float versus[2 * GateSpans];
    float sum = 0.0f;
    for (uint span = 0; span < GateSpans; ++span) {
      const uint dim = span * 512 + 2 * thread_index;
      const uint column = group.x * GroupWidth + dim;
      const float y0 = float(lane_hidden[token * InnerSize + column]);
      const float y1 = float(lane_hidden[token * InnerSize + column + 1]);
      const float z0 = float(packed[token * PackedWidth + column]);
      const float z1 = float(packed[token * PackedWidth + column + 1]);
      versus[2 * span] = y0 * richengine_silu(z0);
      versus[2 * span + 1] = y1 * richengine_silu(z1);
      sum += versus[2 * span] * versus[2 * span] +
             versus[2 * span + 1] * versus[2 * span + 1];
    }
    const float inverse = rms_inverse_of_sums<kMamba2DecodeSimdgroups>(
        sum, GroupWidth,
        shared.reductions + (token & 1) * kMamba2DecodeSimdgroups,
        thread_index, lane, simd_group, params.rms_eps);
    for (uint span = 0; span < GateSpans; ++span) {
      const uint dim = span * 512 + 2 * thread_index;
      const uint column = group.x * GroupWidth + dim;
      const bfloat gated0 =
          bfloat((versus[2 * span] * inverse) * float(norm_weight[column]));
      const bfloat gated1 =
          bfloat((versus[2 * span + 1] * inverse) *
                 float(norm_weight[column + 1]));
      lane_hidden[token * InnerSize + column] = gated0;
      lane_hidden[token * InnerSize + column + 1] = gated1;
      if (table) {
        // As in the chain gate: each 512-column span's simdgroup tiles sit
        // eight heads later within Bolt's sixteen-head groups.
        Table::write_row(
            table + ulong(batch) * InnerSize * Rows,
            sums + ulong(batch) * (Rows / q4sg::kRows) *
                       Table::sums_per_tile(InnerSize),
            InnerSize,
            group.x * HeadsPerGroup + span * kMamba2DecodeSimdgroups +
                simd_group,
            token, lane, gated0, gated1);
      }
    }
  }
}

// The chain kernel's binding layout plus the tree tables at 17 and 18; the
// params sit at 19 (21 with a linear-input table) and the conv bias rides
// the slot just past them, as the chain's buffer(21) does.
#define MAMBA2_TREE_DECODE_BUFFERS(W)                                         \
  device const bfloat *packed [[buffer(0)]],                                  \
      device const bfloat *conv_weights [[buffer(1)]],                        \
      device const uchar *current0 [[buffer(2)]],                             \
      device const uchar *current1 [[buffer(3)]],                             \
      device const uchar *current2 [[buffer(4)]],                             \
      device const uchar *current3 [[buffer(5)]],                             \
      device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],   \
      device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],   \
      device bfloat *mixed [[buffer(10)]],                                    \
      device const float *a_weights [[buffer(11)]],                           \
      device const bfloat *dt_bias [[buffer(12)]],                            \
      device float *step_sizes [[buffer(13)]],                                \
      device const float *skip_weights [[buffer(14)]],                        \
      device const W *norm_weight [[buffer(15)]],                             \
      device bfloat *hidden [[buffer(16)]],                                   \
      device const uint *tree_nodes [[buffer(17)]],                           \
      device const uint *tree_counts [[buffer(18)]]
#define MAMBA2_TREE_DECODE_THREADS                                            \
  uint2 group [[threadgroup_position_in_grid]],                               \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]
#define MAMBA2_TREE_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim,   \
                                PackedWidth, table, sums, Layout)             \
  threadgroup Mamba2DecodeShared<Heads / Groups, HeadDim, StateSize,          \
                                 RICHENGINE_TREE_VERIFY_NODES>                \
      shared;                                                                 \
  mamba2_decode_tree_batch_phase<Heads, HeadDim, Groups, StateSize, ConvDim,  \
                                 PackedWidth, 2, Layout>(                     \
      packed, conv_weights, conv_bias, current0, current1, current2,          \
      current3, next0, next1, next2, next3, mixed, a_weights, dt_bias,        \
      step_sizes, skip_weights, norm_weight, hidden, tree_nodes,              \
      tree_counts, params, group, lane, simd_group, shared, table, sums);
#define MAMBA2_TREE_DECODE_ENTRY(Name, Heads, HeadDim, Groups, StateSize,     \
                                 ConvDim, PackedWidth, W)                     \
  [[max_total_threads_per_threadgroup(256)]]                                  \
  kernel void Name(                                                           \
      MAMBA2_TREE_DECODE_BUFFERS(W),                                          \
      constant Mamba2DecodeBatchParams &params [[buffer(19)]],                \
      device const bfloat *conv_bias [[buffer(20)]],                          \
      MAMBA2_TREE_DECODE_THREADS) {                                           \
    MAMBA2_TREE_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim,       \
                            PackedWidth, (device bfloat *)nullptr,            \
                            (device float *)nullptr, q4sg::Table64)           \
  }
#define MAMBA2_TREE_DECODE_TABLE_ENTRY(Name, Heads, HeadDim, Groups,          \
                                       StateSize, ConvDim, PackedWidth,       \
                                       Layout, W)                             \
  [[max_total_threads_per_threadgroup(256)]]                                  \
  kernel void Name(                                                           \
      MAMBA2_TREE_DECODE_BUFFERS(W), device bfloat *table [[buffer(19)]],     \
      device float *sums [[buffer(20)]],                                      \
      constant Mamba2DecodeBatchParams &params [[buffer(21)]],                \
      device const bfloat *conv_bias [[buffer(22)]],                          \
      MAMBA2_TREE_DECODE_THREADS) {                                           \
    MAMBA2_TREE_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim,       \
                            PackedWidth, table, sums, Layout)                 \
  }
#define MAMBA2_TREE_DECODE_PACKED_ENTRY(Name, Heads, HeadDim, Groups,         \
                                        StateSize, ConvDim, PackedWidth, W)   \
  [[max_total_threads_per_threadgroup(256)]]                                  \
  kernel void Name(                                                           \
      MAMBA2_TREE_DECODE_BUFFERS(W), device half *plane [[buffer(19)]],       \
      device uchar *exponents [[buffer(20)]],                                 \
      constant Mamba2DecodeBatchParams &params [[buffer(21)]],                \
      device const bfloat *conv_bias [[buffer(22)]],                          \
      MAMBA2_TREE_DECODE_THREADS) {                                           \
    MAMBA2_TREE_DECODE_BODY(Heads, HeadDim, Groups, StateSize, ConvDim,       \
                            PackedWidth, plane, exponents, gguf_sg::Packed)   \
  }

// The tree variants of Nemotron 3.5's decode list: one per chain variant.
MAMBA2_TREE_DECODE_ENTRY(verify_tree_mamba2_fused, 64, 64, 8, 128, 6144,
                         10496, bfloat)
MAMBA2_TREE_DECODE_ENTRY(verify_tree_mamba2_fused_f32, 64, 64, 8, 128, 6144,
                         10496, float)
MAMBA2_TREE_DECODE_TABLE_ENTRY(verify_tree_mamba2_fused_table64, 64, 64, 8,
                               128, 6144, 10496, q4sg::Table64, bfloat)
MAMBA2_TREE_DECODE_TABLE_ENTRY(verify_tree_mamba2_fused_table16_f32, 64, 64,
                               8, 128, 6144, 10496, gguf_sg::Table16, float)
MAMBA2_TREE_DECODE_TABLE_ENTRY(verify_tree_mamba2_fused_table16, 64, 64, 8,
                               128, 6144, 10496, gguf_sg::Table16, bfloat)
MAMBA2_TREE_DECODE_PACKED_ENTRY(verify_tree_mamba2_fused_packed_f32, 64, 64,
                                8, 128, 6144, 10496, float)
MAMBA2_TREE_DECODE_PACKED_ENTRY(verify_tree_mamba2_fused_packed, 64, 64, 8,
                                128, 6144, 10496, bfloat)
#undef MAMBA2_TREE_DECODE_PACKED_ENTRY
#undef MAMBA2_TREE_DECODE_TABLE_ENTRY
#undef MAMBA2_TREE_DECODE_ENTRY
#undef MAMBA2_TREE_DECODE_BODY
#undef MAMBA2_TREE_DECODE_THREADS
#undef MAMBA2_TREE_DECODE_BUFFERS

// ---------------------------------------------------------------------------
// The commit: replays the retained verify rows onto the incoming state — the
// conv carry rewinds to the retained rows' raw conv inputs and the recurrent
// matrix recomputes over their post-conv x|B|C rows and dt'. Mirrors the GDN
// commit for spec-decode rejection; a full acceptance already holds the
// scan's provisional state.
// ---------------------------------------------------------------------------

// Grid x is the groups.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_commit_phase(
    device const bfloat *packed, device const bfloat *mixed,
    device const float *step_sizes, device const float *a_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device const float *state_in, device float *state_out, uint retained,
    uint group, uint thread_index, uint lane, uint simd_group) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint DimBatches = HeadDim / kMamba2DecodeSimdgroups;
  const uint count = retained;
  if (count == RICHENGINE_TARGET_VERIFY_ROWS)
    return;
  // The packed row's conv columns lead at InnerSize.
  device const bfloat *conv_rows = packed + InnerSize;
  for (uint element = group * 256 + thread_index; element < 3 * ConvDim;
       element += Groups * 256) {
    const uint row = element / ConvDim;
    const uint channel = element % ConvDim;
    conv_state_out[element] = gdn_conv_carry(
        conv_rows, conv_state_in, PackedWidth, ConvDim, count, row, channel);
  }

  // One (head, dim) row's four state columns per lane; tasks stride over the
  // heads' HeadDim/8 row batches so a simdgroup's lanes split the StateSize
  // columns.
  for (uint task = group; task < Heads * DimBatches; task += Groups) {
    const uint head = task / DimBatches;
    const uint dim = (task % DimBatches) * kMamba2DecodeSimdgroups + simd_group;
    const uint grp = head / HeadsPerGroup;
    const ulong state_base =
        (ulong(head) * HeadDim + dim) * StateSize;
    float local_state[4];
    for (uint i = 0; i < 4; ++i)
      local_state[i] = state_in[state_base + lane * 4 + i];
    const float a_w = a_weights[head];
    for (uint token = 0; token < count; ++token) {
      const ulong conv_base = ulong(token) * ConvDim;
      const float dtv = step_sizes[token * Heads + head];
      const float dA = fast::exp(dtv * a_w);
      const float x = float(mixed[conv_base + head * HeadDim + dim]);
      const float scale = dtv * x;
      for (uint i = 0; i < 4; ++i) {
        const uint n = lane * 4 + i;
        local_state[i] =
            fma(scale,
                float(mixed[conv_base + InnerSize + grp * StateSize + n]),
                local_state[i] * dA);
      }
    }
    for (uint i = 0; i < 4; ++i)
      state_out[state_base + lane * 4 + i] = local_state[i];
  }
}

// Grid {groups, layers, lanes}; each lane's rows of a layer's packed, mixed
// and dt' tensors follow the four lanes of the layer before.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_commit_batch_phase(
    device const bfloat *packed, device const bfloat *mixed,
    device const float *step_sizes, device const float *a_weights,
    device const uchar *current_0, device const uchar *current_1,
    device const uchar *current_2, device const uchar *current_3,
    device uchar *next_0, device uchar *next_1, device uchar *next_2,
    device uchar *next_3, device const uint *retained,
    constant Mamba2BatchCommitParams &params, uint3 group, uint thread_index,
    uint simd_lane, uint simd_group) {
  const uint batch = group.z;
  const uint layer = group.y;
  device const uchar *current = batch == 0   ? current_0
                                : batch == 1 ? current_1
                                : batch == 2 ? current_2
                                             : current_3;
  device uchar *next = batch == 0   ? next_0
                       : batch == 1 ? next_1
                       : batch == 2 ? next_2
                                    : next_3;
  const ulong rows = (ulong(layer) * RICHENGINE_MAXIMUM_BATCH_WIDTH + batch) *
                     params.verify_stride;
  packed += rows * PackedWidth;
  mixed += rows * ConvDim;
  step_sizes += rows * Heads;
  // One buffer holds every layer's ssm_a: the dispatch's own layer's Heads
  // floats lead it.
  a_weights += ulong(layer) * Heads;
  device const bfloat *conv_state_in = reinterpret_cast<device const bfloat *>(
      current + ulong(layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  mamba2_commit_phase<Heads, HeadDim, Groups, StateSize, ConvDim, PackedWidth>(
      packed, mixed, step_sizes, a_weights, conv_state_in, conv_state_out,
      state_in, state_out, retained[batch], group.x, thread_index, simd_lane,
      simd_group);
}

#define MAMBA2_COMMIT_ENTRY(Name, Heads, HeadDim, Groups, StateSize, ConvDim, \
                            PackedWidth)                                     \
  kernel void Name(                                                          \
      device const bfloat *packed [[buffer(0)]],                             \
      device const bfloat *mixed [[buffer(1)]],                              \
      device const float *step_sizes [[buffer(2)]],                          \
      device const float *a_weights [[buffer(3)]],                           \
      device const uchar *current_0 [[buffer(4)]],                           \
      device const uchar *current_1 [[buffer(5)]],                           \
      device const uchar *current_2 [[buffer(6)]],                           \
      device const uchar *current_3 [[buffer(7)]],                           \
      device uchar *next_0 [[buffer(8)]], device uchar *next_1 [[buffer(9)]],\
      device uchar *next_2 [[buffer(10)]],                                   \
      device uchar *next_3 [[buffer(11)]],                                   \
      device const uint *retained [[buffer(12)]],                            \
      constant Mamba2BatchCommitParams &params [[buffer(13)]],               \
      uint3 group [[threadgroup_position_in_grid]],                          \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint simd_lane [[thread_index_in_simdgroup]],                          \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    mamba2_commit_batch_phase<Heads, HeadDim, Groups, StateSize, ConvDim,    \
                              PackedWidth>(                                  \
        packed, mixed, step_sizes, a_weights, current_0, current_1,          \
        current_2, current_3, next_0, next_1, next_2, next_3, retained,      \
        params, group, thread_index, simd_lane, simd_group);                 \
  }

MAMBA2_COMMIT_ENTRY(verify_mamba2_commit, 64, 64, 8, 128, 6144, 10496)
MAMBA2_COMMIT_ENTRY(verify_mamba2_commit_h128, 128, 64, 8, 128, 10240, 18688)
#undef MAMBA2_COMMIT_ENTRY

// The tree commit: the carried convolution always moves to the retained
// path's tail (the tree decode wrote none), and the recurrent state replays
// the path's rows unless the path is the whole chain, in which case the
// scan's chain-end state already matches.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_commit_tree_phase(
    device const bfloat *packed, device const bfloat *mixed,
    device const float *step_sizes, device const float *a_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device const float *state_in, device float *state_out,
    device const uint *path, uint retained, uint group, uint thread_index,
    uint lane, uint simd_group) {
  constexpr uint HeadsPerGroup = Heads / Groups;
  constexpr uint InnerSize = Heads * HeadDim;
  constexpr uint DimBatches = HeadDim / kMamba2DecodeSimdgroups;
  const uint count = retained;
  // The packed row's conv columns lead at InnerSize.
  device const bfloat *conv_rows = packed + InnerSize;
  for (uint element = group * 256 + thread_index; element < 3 * ConvDim;
       element += Groups * 256) {
    const uint row = element / ConvDim;
    const uint channel = element % ConvDim;
    conv_state_out[element] =
        gdn_conv_carry_tree(conv_rows, conv_state_in, PackedWidth, ConvDim,
                            path, count, row, channel);
  }

  // A full retained chain (path[i] == i) already ended at the scan's
  // chain-row state; only a truncation or a leaf detour needs the replay.
  const bool full_chain =
      count == RICHENGINE_TARGET_VERIFY_ROWS &&
      path[RICHENGINE_TARGET_VERIFY_ROWS - 1] ==
          RICHENGINE_TARGET_VERIFY_ROWS - 1;
  if (full_chain)
    return;
  for (uint task = group; task < Heads * DimBatches; task += Groups) {
    const uint head = task / DimBatches;
    const uint dim = (task % DimBatches) * kMamba2DecodeSimdgroups + simd_group;
    const uint grp = head / HeadsPerGroup;
    const ulong state_base = (ulong(head) * HeadDim + dim) * StateSize;
    float local_state[4];
    for (uint i = 0; i < 4; ++i)
      local_state[i] = state_in[state_base + lane * 4 + i];
    const float a_w = a_weights[head];
    for (uint token = 0; token < count; ++token) {
      const ulong conv_base = ulong(path[token]) * ConvDim;
      const float dtv = step_sizes[path[token] * Heads + head];
      const float dA = fast::exp(dtv * a_w);
      const float x = float(mixed[conv_base + head * HeadDim + dim]);
      const float scale = dtv * x;
      for (uint i = 0; i < 4; ++i) {
        const uint n = lane * 4 + i;
        local_state[i] =
            fma(scale,
                float(mixed[conv_base + InnerSize + grp * StateSize + n]),
                local_state[i] * dA);
      }
    }
    for (uint i = 0; i < 4; ++i)
      state_out[state_base + lane * 4 + i] = local_state[i];
  }
}

// Grid {groups, layers, lanes}; the lane's rows of a layer's packed, mixed
// and dt' tensors are RICHENGINE_TREE_VERIFY_NODES apart, and retained_path
// holds the committed path's DFS rows.
template <uint Heads, uint HeadDim, uint Groups, uint StateSize, uint ConvDim,
          uint PackedWidth>
inline void mamba2_commit_tree_batch_phase(
    device const bfloat *packed, device const bfloat *mixed,
    device const float *step_sizes, device const float *a_weights,
    device const uchar *current_0, device const uchar *current_1,
    device const uchar *current_2, device const uchar *current_3,
    device uchar *next_0, device uchar *next_1, device uchar *next_2,
    device uchar *next_3, device const uint *retained,
    device const uint *retained_path,
    constant Mamba2BatchCommitParams &params, uint3 group, uint thread_index,
    uint simd_lane, uint simd_group) {
  const uint batch = group.z;
  const uint layer = group.y;
  device const uchar *current = batch == 0   ? current_0
                                : batch == 1 ? current_1
                                : batch == 2 ? current_2
                                             : current_3;
  device uchar *next = batch == 0   ? next_0
                       : batch == 1 ? next_1
                       : batch == 2 ? next_2
                                    : next_3;
  // The arena's mamba tensors hold verify_stride-row units per lane slot; a
  // tree lane's Nodes rows occupy Nodes/Verify units of the sixteen-row
  // stride a tree batch carries.
  const ulong rows =
      (ulong(layer) * RICHENGINE_MAXIMUM_BATCH_WIDTH +
       batch * (RICHENGINE_TREE_VERIFY_NODES / RICHENGINE_TARGET_VERIFY_ROWS)) *
      params.verify_stride;
  packed += rows * PackedWidth;
  mixed += rows * ConvDim;
  step_sizes += rows * Heads;
  // One buffer holds every layer's ssm_a: the dispatch's own layer's Heads
  // floats lead it.
  a_weights += ulong(layer) * Heads;
  device const bfloat *conv_state_in = reinterpret_cast<device const bfloat *>(
      current + ulong(layer) * params.conv_layer_bytes);
  device bfloat *conv_state_out = reinterpret_cast<device bfloat *>(
      next + ulong(layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  device float *state_out = reinterpret_cast<device float *>(
      next + params.convolution_state_bytes +
      ulong(layer) * params.recurrent_layer_bytes);
  mamba2_commit_tree_phase<Heads, HeadDim, Groups, StateSize, ConvDim,
                           PackedWidth>(
      packed, mixed, step_sizes, a_weights, conv_state_in, conv_state_out,
      state_in, state_out,
      retained_path + batch * RICHENGINE_TARGET_VERIFY_ROWS, retained[batch],
      group.x, thread_index, simd_lane, simd_group);
}

#define MAMBA2_TREE_COMMIT_ENTRY(Name, Heads, HeadDim, Groups, StateSize,    \
                                 ConvDim, PackedWidth)                        \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *mixed [[buffer(1)]],                               \
      device const float *step_sizes [[buffer(2)]],                           \
      device const float *a_weights [[buffer(3)]],                            \
      device const uchar *current_0 [[buffer(4)]],                            \
      device const uchar *current_1 [[buffer(5)]],                            \
      device const uchar *current_2 [[buffer(6)]],                            \
      device const uchar *current_3 [[buffer(7)]],                            \
      device uchar *next_0 [[buffer(8)]], device uchar *next_1 [[buffer(9)]], \
      device uchar *next_2 [[buffer(10)]],                                    \
      device uchar *next_3 [[buffer(11)]],                                    \
      device const uint *retained [[buffer(12)]],                             \
      device const uint *retained_path [[buffer(13)]],                        \
      constant Mamba2BatchCommitParams &params [[buffer(14)]],                \
      uint3 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint simd_lane [[thread_index_in_simdgroup]],                           \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    mamba2_commit_tree_batch_phase<Heads, HeadDim, Groups, StateSize,         \
                                   ConvDim, PackedWidth>(                     \
        packed, mixed, step_sizes, a_weights, current_0, current_1,           \
        current_2, current_3, next_0, next_1, next_2, next_3, retained,       \
        retained_path, params, group, thread_index, simd_lane, simd_group);   \
  }

MAMBA2_TREE_COMMIT_ENTRY(verify_mamba2_commit_tree, 64, 64, 8, 128, 6144,
                         10496)
#undef MAMBA2_TREE_COMMIT_ENTRY

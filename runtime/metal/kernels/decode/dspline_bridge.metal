#include "metal/abi/KernelABI.h"
#include "metal/abi/Dspline.h"
#include "metal/kernels/common/rms_inverse.h"

// The DSpline selected-prefix code bridge's elementwise phases — the
// trained write after every draft layer of the checkpoint's feature mask.
// Each phase keeps the rounding points of sglang's DSpineCodeBridge fused
// kernels (sglang_dspine/kernels/dspine.py, ported by
// dspine_mlx/sglang/kernels/bridge.py):
//   prepare  raw's per-row RMS at the fp32-tiny floor, the code read's
//            normalized input x / (rms + eps)
//   matmul   the three bf16 projections — read [code][hidden], packed
//            [3*rank][code] (three weight segments of one dispatch) and
//            feature write [hidden][rank] — fp32 accumulation on matmul2d
//   codes    the sqrt(code)-scaled L2 code normalize; the block's anchor
//            row substitutes sqrt(code) * codebook[anchor token]
//   message  sigmoid(q[r] + s[r-1] + bias) * v[r-1] over the packed
//            projection, zero on every block's anchor row
//   write    the feature projection fused with the elementwise phases its
//            delta feeds: the RMS-scaled write-back into the residual
//            stream, in place — and on the last write the refinement's
//            raw * final_norm stash; anchor rows keep their layer output
// The remaining elementwise kernels cannot merge across their matmuls:
// prepare's normalized rows feed the read projection, codes' input is the
// read projection's output, message's input the packed projection's, so
// no two adjacent elementwise phases share a dependency level. prepare,
// codes and message keep their dispatches; dspline_bridge_finish and
// dspline_refine_stash remain below as the unfused reference the write
// epilogue reproduces bit for bit.
// The fused next-layer norm the reference's finish also emits stays a
// dispatch of the ordinary norm (ops::Normalization::addRms): its
// single-rounded output is what every plain-draft layer consumes, and
// the projection input tables it can emit keep their layout.

// One threadgroup per row: the row's RMS (sum of squares in source
// order, the fp32-tiny floor on the mean) and the normalized row the
// read projection consumes.
kernel void dspline_bridge_prepare(
    device const bfloat *raw [[buffer(0)]],
    device bfloat *normalized [[buffer(1)]],
    device float *rms [[buffer(2)]],
    constant DsplineBridgeParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp reassociate(off)
  threadgroup float reductions[8];
  device const bfloat *input = raw + ulong(row) * p.hidden;
  float sum = row_squares(input, p.hidden, tid);
  sum = simd_sum(sum);
  if (lane == 0)
    reductions[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reductions[i];
    const float rms_value =
        sqrt(max(total / p.hidden, 1.1754943508222875e-38f));
    reductions[0] = rms_value;
    rms[row] = rms_value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float rms_value = reductions[0];
  for (uint column = tid; column < p.hidden; column += 256)
    normalized[ulong(row) * p.hidden + column] =
        bfloat(float(input[column]) / (rms_value + p.eps));
}

// One threadgroup per row: sqrt(code) * projected / (norm + eps), or the
// anchor token's scaled codebook row on every block's row 0.
kernel void dspline_bridge_codes(
    device const bfloat *projected [[buffer(0)]],
    device const bfloat *codebook [[buffer(1)]],
    device const uint *input_tokens [[buffer(2)]],
    device bfloat *codes [[buffer(3)]],
    constant DsplineBridgeParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp reassociate(off)
  const float sqrt_code = sqrt(float(p.code));
  device bfloat *output = codes + ulong(row) * p.code;
  if (row % p.block == 0) {
    // Each lane's anchor is the draft input's row-0 token: the packed
    // input holds `block` rows per lane.
    const uint anchor = input_tokens[(row / p.block) * p.block];
    device const bfloat *source = codebook + ulong(anchor) * p.code;
    for (uint column = tid; column < p.code; column += 256)
      output[column] = bfloat(sqrt_code * float(source[column]));
    return;
  }
  threadgroup float reductions[8];
  device const bfloat *input = projected + ulong(row) * p.code;
  float sum = row_squares(input, p.code, tid);
  sum = simd_sum(sum);
  if (lane == 0)
    reductions[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reductions[i];
    reductions[0] = sqrt(total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float norm = reductions[0];
  for (uint column = tid; column < p.code; column += 256)
    output[column] =
        bfloat(sqrt_code * float(input[column]) / (norm + p.eps));
}

// One thread per rank element of one row: the gated message the row
// receives from its predecessor's packed [q | s | v] projection. Anchor
// rows send nothing: their zero keeps the write's addition exact.
kernel void dspline_bridge_message(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *bias [[buffer(1)]],
    device bfloat *message [[buffer(2)]],
    constant DsplineBridgeParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint row = gid / p.rank;
  const uint r = gid - row * p.rank;
  if (row % p.block == 0) {
    message[gid] = bfloat(0.0f);
    return;
  }
  const uint previous = row - 1;
  const float q = float(packed[ulong(row) * (3 * p.rank) + r]);
  const float s =
      float(packed[ulong(previous) * (3 * p.rank) + p.rank + r]);
  const float v =
      float(packed[ulong(previous) * (3 * p.rank) + 2 * p.rank + r]);
  const float gate = 1.0f / (1.0f + exp(-(q + s + float(bias[r]))));
  message[gid] = bfloat(gate * v);
}

// The RMS-scaled write-back: hidden[m][h] += rms[m] * delta[m][h] on
// non-anchor rows, in place in the residual stream. Anchor rows keep
// their layer output. Unfused reference — dspline_bridge_write folds
// this epilogue into the delta projection; no live dispatch uses this.
kernel void dspline_bridge_finish(
    device bfloat *hidden [[buffer(0)]],
    device const bfloat *delta [[buffer(1)]],
    device const float *rms [[buffer(2)]],
    constant DsplineBridgeParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint row = gid / p.hidden;
  if (row % p.block == 0)
    return;
  hidden[gid] =
      bfloat(float(hidden[gid]) + rms[row] * float(delta[gid]));
}

// out[m][n] = bf16(sum_k in[m][k] * W[n][k]) on 16x64 output tiles of
// 64-input steps, fp32 accumulation. `segment` ranks the output columns
// among the three bound weights: a column tile of one segment reads one
// weight's rows, so the packed projection's three [rank][code] segments
// dispatch in one call while the read and feature-write passes bind one
// weight three times. The row tile clamps into the last full tile when
// the row count is not a multiple of 16 — rows already stored rewrite
// identical values, and callers pad the output's row allocation.
kernel void dspline_bridge_matmul(
    device bfloat *input [[buffer(0)]],
    device bfloat *w0 [[buffer(1)]],
    device bfloat *w1 [[buffer(2)]],
    device bfloat *w2 [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant DsplineMatmulParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr ushort M = 16, N = 64, K = 64;
  const uint column = group.x * N;
  const uint segment = column / p.segment;
  device bfloat *weights =
      (segment == 0 ? w0 : segment == 1 ? w1 : w2) +
      ulong(column - segment * p.segment) * p.input;
  const uint base = min(group.y * M, p.rows > M ? p.rows - M : 0u);
  auto a = tensor(input + ulong(base) * p.input,
                  dextents<int, 2>{int(p.input), M},
                  array<int, 2>{1, int(p.input)});
  auto b = tensor(weights, dextents<int, 2>{int(p.input), N},
                  array<int, 2>{1, int(p.input)});
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, K, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.template slice<K, M>(0, 0);
  auto b0 = b.template slice<K, N>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    accumulated[i] = 0.0f;
  for (uint k = 0; k + K <= p.input; k += K) {
    auto as = a.template slice<K, M>(k, 0);
    auto bs = b.template slice<K, N>(k, 0);
    operation.run(as, bs, accumulated);
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i))
      continue;
    const auto index = accumulated.get_multidimensional_index(i);
    output[ulong(base + index[1]) * p.output + column + index[0]] =
        bfloat(accumulated[i]);
  }
}

// The packed [gate; predecessor_gate; prev] projection fused with the
// message phase that consumes it: one threadgroup computes a 64-column
// slice of all three weight segments — the same 16x64 matmul2d tile as
// dspline_bridge_matmul, run three times against the three weights on
// one shared input slice, so the activation row loads a plain
// three-dispatch pass would repeat are paid once — then the same
// threadgroup emits dspline_bridge_message's sigmoid gate for the slice.
// The message values are computed from the bf16-rounded tiles the store
// produced, staged in threadgroup memory, so the fused kernel's output
// is bit-identical to the matmul plus message pair: message[m][r] =
// sigmoid(q[m][r] + s[m-1][r] + bias[r]) * v[m-1][r], zero on anchor
// rows. A threadgroup holds every row of its 16-row tile, but the tile's
// first row needs the previous row's s/v — another tile's work — so the
// host dispatches this only when every 16-row tile base is a block
// anchor (16 % block == 0: bases are multiples of 16, and the
// last-tile clamp base rows - 16 stays block-aligned too), which makes
// the first row of every tile send nothing.
kernel void dspline_bridge_pack(
    device bfloat *input [[buffer(0)]],
    device bfloat *w0 [[buffer(1)]],
    device bfloat *w1 [[buffer(2)]],
    device bfloat *w2 [[buffer(3)]],
    device const bfloat *bias [[buffer(4)]],
    device bfloat *packed [[buffer(5)]],
    device bfloat *message [[buffer(6)]],
    constant DsplinePackParams &p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr ushort M = 16, N = 64, K = 64;
  const uint column = group.x * N;
  const uint base = min(group.y * M, p.rows > M ? p.rows - M : 0u);
  auto a = tensor(input + ulong(base) * p.input,
                  dextents<int, 2>{int(p.input), M},
                  array<int, 2>{1, int(p.input)});
  auto b0 = tensor(w0 + ulong(column) * p.input,
                   dextents<int, 2>{int(p.input), N},
                   array<int, 2>{1, int(p.input)});
  auto b1 = tensor(w1 + ulong(column) * p.input,
                   dextents<int, 2>{int(p.input), N},
                   array<int, 2>{1, int(p.input)});
  auto b2 = tensor(w2 + ulong(column) * p.input,
                   dextents<int, 2>{int(p.input), N},
                   array<int, 2>{1, int(p.input)});
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, K, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.template slice<K, M>(0, 0);
  auto c0 = b0.template slice<K, N>(0, 0);
  auto accumulated_0 =
      operation.template get_destination_cooperative_tensor<decltype(a0),
                                                            decltype(c0),
                                                            float>();
  auto accumulated_1 = accumulated_0;
  auto accumulated_2 = accumulated_0;
#pragma unroll
  for (ushort i = 0; i < accumulated_0.get_capacity(); ++i) {
    accumulated_0[i] = 0.0f;
    accumulated_1[i] = 0.0f;
    accumulated_2[i] = 0.0f;
  }
  for (uint k = 0; k + K <= p.input; k += K) {
    auto as = a.template slice<K, M>(k, 0);
    auto bs0 = b0.template slice<K, N>(k, 0);
    auto bs1 = b1.template slice<K, N>(k, 0);
    auto bs2 = b2.template slice<K, N>(k, 0);
    operation.run(as, bs0, accumulated_0);
    operation.run(as, bs1, accumulated_1);
    operation.run(as, bs2, accumulated_2);
  }
  threadgroup bfloat staged[3][M][N];
  // The cooperative tile's valid elements may not cover every staged
  // position on a fragment layout that pads the destination.
  for (uint e = tid; e < 3 * M * N; e += 256)
    staged[e / (M * N)][(e / N) % M][e % N] = bfloat(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (ushort i = 0; i < accumulated_0.get_capacity(); ++i) {
    if (!accumulated_0.is_valid_element(i))
      continue;
    const auto index = accumulated_0.get_multidimensional_index(i);
    const uint row = base + index[1];
    const uint r = column + index[0];
    const bfloat q = bfloat(accumulated_0[i]);
    const bfloat s = bfloat(accumulated_1[i]);
    const bfloat v = bfloat(accumulated_2[i]);
    staged[0][index[1]][index[0]] = q;
    staged[1][index[1]][index[0]] = s;
    staged[2][index[1]][index[0]] = v;
    packed[ulong(row) * (3 * p.rank) + r] = q;
    packed[ulong(row) * (3 * p.rank) + p.rank + r] = s;
    packed[ulong(row) * (3 * p.rank) + 2 * p.rank + r] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < M * N; e += 256) {
    const uint row_local = e / N;
    const uint rr = e - row_local * N;
    const uint row = base + row_local;
    float value = 0.0f;
    if (row_local && (row % p.block)) {
      const float gate =
          1.0f / (1.0f + exp(-(float(staged[0][row_local][rr]) +
                               float(staged[1][row_local - 1][rr]) +
                               float(bias[column + rr]))));
      value = gate * float(staged[2][row_local - 1][rr]);
    }
    message[ulong(row) * p.rank + column + rr] = bfloat(value);
  }
}

// The feature write's fused projection: the same 16x64-tiled bf16 matmul
// as dspline_bridge_matmul, but its epilogue is the elementwise tail the
// delta buffer used to visit — dspline_bridge_finish's
// hidden[m][h] += rms[m] * delta[m][h] (anchor rows skipped) and, when
// p.stash is set, dspline_refine_stash's raw * final_norm row. The delta
// rounds to bf16 exactly where the unfused pass stored it, so the
// write-back's float(bf16(acc)) sees the same bits finish read; the stash
// reads each element's pre-write value in the same thread that updates
// it, capturing the residual stream finish used to see between the delta
// matmul and the write-back. The row-tile clamp duplicates stored rows
// when the count is not a multiple of 16 — harmless for a plain store,
// wrong for an in-place update — so an element's epilogue runs only in
// the tile whose natural row range [group.y*M, group.y*M+M) owns its row
// (group.y*M <= row < rows), and every row updates exactly once.
kernel void dspline_bridge_write(
    device bfloat *input [[buffer(0)]],
    device bfloat *w0 [[buffer(1)]],
    device bfloat *w1 [[buffer(2)]],
    device bfloat *w2 [[buffer(3)]],
    device bfloat *hidden [[buffer(4)]],
    device const bfloat *norm [[buffer(5)]],
    device bfloat *rawnorm [[buffer(6)]],
    device const float *rms [[buffer(7)]],
    constant DsplineWriteParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]]) {
  constexpr ushort M = 16, N = 64, K = 64;
  const uint column = group.x * N;
  const uint segment = column / p.segment;
  device bfloat *weights =
      (segment == 0 ? w0 : segment == 1 ? w1 : w2) +
      ulong(column - segment * p.segment) * p.input;
  const uint base = min(group.y * M, p.rows > M ? p.rows - M : 0u);
  auto a = tensor(input + ulong(base) * p.input,
                  dextents<int, 2>{int(p.input), M},
                  array<int, 2>{1, int(p.input)});
  auto b = tensor(weights, dextents<int, 2>{int(p.input), N},
                  array<int, 2>{1, int(p.input)});
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, K, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.template slice<K, M>(0, 0);
  auto b0 = b.template slice<K, N>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    accumulated[i] = 0.0f;
  for (uint k = 0; k + K <= p.input; k += K) {
    auto as = a.template slice<K, M>(k, 0);
    auto bs = b.template slice<K, N>(k, 0);
    operation.run(as, bs, accumulated);
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i))
      continue;
    const auto index = accumulated.get_multidimensional_index(i);
    const uint row = base + index[1];
    if (row < group.y * M || row >= p.rows)
      continue;
    const uint h = column + index[0];
    const ulong at = ulong(row) * p.output + h;
    const float value = float(hidden[at]);
    if (p.stash)
      rawnorm[at] = bfloat(value * float(norm[h]));
    if (row % p.block)
      hidden[at] =
          bfloat(value + rms[row] * float(bfloat(accumulated[i])));
  }
}

// ---------------------------------------------------------------------------
// The serial last-write refinement (upstream code_bridge_second_pass, mode
// serial_table — sglang_dspine/kernels/dspine.py _serial_table +
// _serial_chain, dspine_mlx RefinementTail.greedy). After the draft's plain
// top-16 has picked each mask row's candidates, every mask row's sixteen
// predecessor-conditioned score rows are rebuilt with the ACTUAL predecessor
// token's code instead of the forward's predicted sender:
//   score[j][k] = base[k] + rms * dot(
//       sigmoid(q + TSV[sid_j][0:R] + gate_bias) * TSV[sid_j][R:2R],
//       TG[cand[k]])
// with base[k] the row's unnormalized logit (the second head pass over
// raw * final_norm), sid_j the anchor for mask row 0 and the previous mask
// row's candidate j after, and the tables built once at load:
//   TSV[v] = [pred_gate; prev] @ bf16(sqrt(C) * codebook[v])  ([V][2R])
//   TG[v]  = bf16(head[v] * norm_w) @ write_w[last]           ([V][R])
// The walk then takes j = idx[0][0], token[r] = cand[r][j],
// j = idx[r + 1][j] into the draft_select_plain contract.
// ---------------------------------------------------------------------------

// One element per thread of the raw-times-norm input the refinement's base
// head pass consumes: raw is the pre-write residual stream the bridge's
// finish is about to update, so this runs between the last write's delta
// matmul and its finish dispatch. The norm weight fold keeps the second
// head pass a plain projection on bf16 rows. Unfused reference — the last
// write's dspline_bridge_write emits these rows in its epilogue; no live
// dispatch uses this.
kernel void dspline_refine_stash(
    device const bfloat *hidden [[buffer(0)]],
    device const bfloat *norm [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant DsplineBridgeParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint row = gid / p.hidden;
  if (row >= p.rows)
    return;
  output[gid] = bfloat(float(hidden[gid]) * float(norm[gid - row * p.hidden]));
}

// One element per thread of a chunk of the affine Q4 vocabulary head,
// dequantized and folded with the draft's final norm weight into the
// TG table's matmul input: out[r][h] = bf16(deq(head[r][h]) * norm[h])
// where deq = code * scale + bias of the StorageN=256-tiled planes
// (WeightStore's affine layout: codes [row/256][group][256][32B],
// scale/bias bf16 [(row/256 * groups + group) * 256 + row%256]).
kernel void dspline_table_dequant(
    device const uchar *codes [[buffer(0)]],
    device const bfloat *scales [[buffer(1)]],
    device const bfloat *biases [[buffer(2)]],
    device const bfloat *norm [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant DsplineTableParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint row = gid / p.hidden;
  if (row >= p.rows)
    return;
  const uint h = gid - row * p.hidden;
  const uint group = h / 64;
  const uint element = h - group * 64;
  const ulong parameter =
      (ulong(row / 256) * (p.hidden / 64) + group) * 256 + row % 256;
  const uchar packed_byte = codes[parameter * 32 + element / 2];
  const float code = (element & 1) ? float(packed_byte >> 4)
                                   : float(packed_byte & 0xF);
  const float value =
      code * float(scales[parameter]) + float(biases[parameter]);
  output[gid] = bfloat(value * float(norm[h]));
}

// The simdgroup merge of draft_select_plain (its top-16 sharded partials
// contract), copied so the refinement stays in this file: the partials of
// proposal row `row` become the lanes' sorted candidate list.
template <uint Count>
inline void ds_top_pop(thread float (&values)[Count],
                       thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

inline void ds_simd_best_head(float value, uint token, thread float &best,
                              thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

inline void ds_top16_merge(device const uint *partial_ids,
                           device const float *partial_values, uint row,
                           uint lane, thread float &value,
                           thread uint &token) {
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr uint Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    ds_simd_best_head(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      ds_top_pop<Entries>(values, ids);
  }
}

// The refinement's gather-column head pass. dspline_refine_rows reads
// each mask row's unnormalized base logits at the row's merged top-16
// candidate columns and nowhere else, so the second vocabulary-head pass
// over the staged raw * final_norm rows computes just those columns
// (~positions*16 of the vocabulary's ~250k) — one threadgroup per mask
// row n = batch * positions + r. Simdgroup 0 merges the row's shard
// partials exactly as dspline_refine_rows does, so the gather and the
// rescoring see the same candidate ids; every column's logit then lands
// at the logits slot the rescoring reads, and the columns no dispatch
// consumes keep the first pass's stale values.
// The arithmetic is the affine Q4 tiles' own: for each 64-input quant
// group the 4-bit codes dot the input row, then the group's scale and
// the group input sum times its bias accumulate in group order —
// acc += dot * scale + xsum * bias — matching
// sum_h x[h] * (code[h] * scale + bias[h]) without materializing the
// dequantized row. Each of the row's 16 candidates pairs with the 16
// thread slices of a 256-thread group: slice g of candidate c covers
// quant groups g, g + 16, ..., and the group's input sums (each computed
// once, in threadgroup memory) feed the bias term of every candidate.
kernel void dspline_head_gather(
    device const bfloat *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const bfloat *scales [[buffer(2)]],
    device const bfloat *biases [[buffer(3)]],
    device const uint *partial_ids [[buffer(4)]],
    device const float *partial_values [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant DsplineGatherParams &p [[buffer(7)]],
    uint n [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  static_assert(256 % K == 0, "the gather's thread slices tile the group");
  // The threadgroup arrays bound the head's input to 8192 (128 quant
  // groups); the dispatch falls back to the full pass past that.
  constexpr uint MaxGroups = 128;
  constexpr uint Slices = 256 / K;
  const uint batch = n / p.positions;
  const uint r = n - batch * p.positions;
  const uint draft_row = batch * p.query_rows + 1 + r;
  const uint groups = p.input / 64;
  threadgroup uint cand[K];
  threadgroup float input_sums[MaxGroups];
  threadgroup float partials[K][Slices];
  device const bfloat *row = input + ulong(draft_row) * p.input;
  if (sg == 0) {
    float value;
    uint token;
    ds_top16_merge(partial_ids, partial_values, n, lane, value, token);
    if (lane < K)
      cand[lane] = token;
  }
  for (uint g = tid; g < groups; g += 256) {
    float sum = 0.0f;
    for (uint e = 0; e < 64; ++e)
      sum += float(row[g * 64 + e]);
    input_sums[g] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tid / Slices;
  const uint g0 = tid - c * Slices;
  const uint token = cand[c];
  const uint tile = token / 256;
  const uint tile_offset = token - tile * 256;
  float accumulated = 0.0f;
  for (uint g = g0; g < groups; g += Slices) {
    const ulong parameter =
        (ulong(tile) * groups + g) * 256 + tile_offset;
    device const uchar *block = codes + parameter * 32;
    device const bfloat *x = row + g * 64;
    float dot = 0.0f;
    for (uint e = 0; e < 32; ++e) {
      const uint pair = block[e];
      dot += float(x[2 * e]) * float(pair & 0xF) +
             float(x[2 * e + 1]) * float(pair >> 4);
    }
    accumulated += dot * float(scales[parameter]) +
                   input_sums[g] * float(biases[parameter]);
  }
  partials[c][g0] = accumulated;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < K) {
    float base = 0.0f;
    for (uint i = 0; i < Slices; ++i)
      base += partials[tid][i];
    output[ulong(draft_row) * p.logits_stride + cand[tid]] = base;
  }
}

// One threadgroup per mask row n = batch * positions + r (the draft's row
// batch * query_rows + 1 + r): simdgroup 0 merges the row's top-16 into
// cand, simdgroup 1 merges the previous mask row's (the predecessor table's
// token set — mask row 0's sender is the anchor token instead), and each
// thread (j, k) = (tid / 16, tid % 16) scores one predecessor/candidate
// pair. The score table goes to the selector's partialValues tail, the
// argmax idx rows to `unary` (both read by dspline_refine_walk).
// The pair scores share two operands the scalar expression used to
// re-read and re-derive per pair: the gated sender column
// gate[j][rr] * TSV[sid_j][rank + rr] depends only on (j, rr) — the
// sixteen k threads of one j recomputed it sixteenfold, sigmoid and all —
// and each TG[cand[k]] row was gathered once per j. Per 256-column block
// of the rank, the eight simdgroups stage two senders' gated columns into
// fp32 threadgroup rows and two candidates' TG rows into bf16 ones
// (coalesced by lane); each pair then walks its operands in threadgroup
// memory in rr order. Both stages hold the exact values the scalar
// expression reads — gate * sender_value is the same fp32 product, the
// TG stage the same bf16 bits — so every pair's serial dot keeps the
// unfused accumulation order bit for bit.
kernel void dspline_refine_rows(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device const float *base_logits [[buffer(2)]],
    device const bfloat *tsv [[buffer(3)]],
    device const bfloat *tg [[buffer(4)]],
    device const bfloat *packed [[buffer(5)]],
    device const float *rms [[buffer(6)]],
    device const bfloat *gate_bias [[buffer(7)]],
    device const uint *draft_input [[buffer(8)]],
    device uint *candidates [[buffer(9)]],
    device uint *idx [[buffer(10)]],
    device float *scores [[buffer(11)]],
    constant DsplineRefineParams &p [[buffer(12)]],
    uint n [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint Stage = 256;
  const uint batch = n / p.positions;
  const uint r = n - batch * p.positions;
  const uint draft_row = batch * p.query_rows + 1 + r;
  threadgroup uint cand_self[K];
  threadgroup uint cand_prev[K];
  threadgroup float gated[K][Stage];
  threadgroup bfloat targets[K][Stage];
  if (sg == 0) {
    float value;
    uint token;
    ds_top16_merge(partial_ids, partial_values, n, lane, value, token);
    if (lane < K)
      cand_self[lane] = token;
  } else if (sg == 1 && r) {
    float value;
    uint token;
    ds_top16_merge(partial_ids, partial_values, n - 1, lane, value, token);
    if (lane < K)
      cand_prev[lane] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < K)
    candidates[n * K + tid] = cand_self[tid];
  const uint j = tid / K;
  const uint k = tid - j * K;
  device const bfloat *query = packed + ulong(draft_row) * (3 * p.rank);
  const uint anchor = draft_input[batch * p.query_rows];
  // Mask row 0's sender is the anchor token on every predecessor lane:
  // one staged row serves all sixteen j. Later rows stage all sixteen
  // predecessor candidates.
  const uint senders = r ? K : 1;
  const uint jrow = r ? j : 0;
  float dot = 0.0f;
  for (uint at = 0; at < p.rank; at += Stage) {
    const uint width = min(Stage, p.rank - at);
    for (uint c = sg; c < K; c += 8) {
      device const bfloat *target = tg + ulong(cand_self[c]) * p.rank;
      if (c < senders) {
        const uint sid = r ? cand_prev[c] : anchor;
        device const bfloat *sender = tsv + ulong(sid) * (2 * p.rank);
        for (uint i = lane; i < width; i += 32) {
          const uint rr = at + i;
          const float gate =
              1.0f / (1.0f + exp(-(float(query[rr]) + float(sender[rr]) +
                                   float(gate_bias[rr]))));
          gated[c][i] = gate * float(sender[p.rank + rr]);
        }
      }
      for (uint i = lane; i < width; i += 32)
        targets[c][i] = target[at + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < width; ++i)
      dot += gated[jrow][i] * float(targets[k][i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float base =
      base_logits[ulong(draft_row) * p.logits_stride + cand_self[k]];
  threadgroup float table[K][K];
  const float score = base + rms[draft_row] * dot;
  table[j][k] = score;
  scores[ulong(batch) * p.scores_lane + p.scores_base + r * K * K + j * K +
         k] = score;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < K) {
    float best = -INFINITY;
    uint argmax = 0;
    for (uint kk = 0; kk < K; ++kk) {
      if (table[tid][kk] > best) {
        best = table[tid][kk];
        argmax = kk;
      }
    }
    idx[n * K + tid] = argmax;
  }
}

// One threadgroup per mask row of a sampled lane: the refined RMS each
// predecessor lane's score row is normalized by before the sampled walk
// draws from it (upstream serial_rs_table — the sampling table's logits
// are the truly renormalized refined states, unlike the greedy table's
// rms-free score). For predecessor j the refined row is
//   refined[j] = raw + rms * delta[j],   delta[j] = m_j @ write_w.T
// with m_j the row's gated sender under predecessor token j, so
//   mean(refined^2) = mean(raw^2) + 2*rms*(m_j . W^T raw)/H
//                     + rms^2*(m_j^T (W^T W) m_j)/H.
// mean(raw^2) is the stashed write rms squared, W^T raw comes of the
// staged raw*norm rows through the norm-divided write table (rw), and
// W^T W is the load-time Gram table. Each mask row stages its sixteen
// senders' m once; every thread then walks its rank rows of the Gram
// against all sixteen staged senders, so the table is read once a row.
// threadgroup bound: bridge ranks past 512 do not fit the staged senders
// and take the plain walk instead.
kernel void dspline_refine_norm(
    device const uint *candidates [[buffer(0)]],
    device const uint *draft_input [[buffer(1)]],
    device const bfloat *packed [[buffer(2)]],
    device const float *rms [[buffer(3)]],
    device const bfloat *tsv [[buffer(4)]],
    device const bfloat *gate_bias [[buffer(5)]],
    device const bfloat *gram [[buffer(6)]],
    device float *pv [[buffer(7)]],
    constant DsplineRefineParams &p [[buffer(8)]],
    uint n [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  constexpr uint MaxRank = 512;
  const uint batch = n / p.positions;
  const uint r = n - batch * p.positions;
  if (!(p.sampling_mask & (1u << batch)))
    return;
  const uint draft_row = batch * p.query_rows + 1 + r;
  const uint anchor = draft_input[batch * p.query_rows];
  device const bfloat *query = packed + ulong(draft_row) * (3 * p.rank);
  device const bfloat *rw =
      reinterpret_cast<device const bfloat *>(pv + p.rw_base) +
      ulong(draft_row) * p.rank;
  threadgroup bfloat m[K][MaxRank];
  for (uint j = 0; j < K; ++j) {
    const uint sid = r ? candidates[(n - 1) * K + j] : anchor;
    device const bfloat *sender = tsv + ulong(sid) * (2 * p.rank);
    for (uint rr = tid; rr < p.rank; rr += 256) {
      const float gate =
          1.0f / (1.0f + exp(-(float(query[rr]) + float(sender[rr]) +
                               float(gate_bias[rr]))));
      m[j][rr] = bfloat(gate * float(sender[p.rank + rr]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float cross[K];
  float gram_dot[K];
  for (uint j = 0; j < K; ++j) {
    cross[j] = 0.0f;
    gram_dot[j] = 0.0f;
  }
  for (uint r1 = tid; r1 < p.rank; r1 += 256) {
    device const bfloat *srow = gram + ulong(r1) * p.rank;
    const float w = float(rw[r1]);
    for (uint j = 0; j < K; ++j) {
      float u = 0.0f;
      for (uint r2 = 0; r2 < p.rank; ++r2)
        u += float(srow[r2]) * float(m[j][r2]);
      gram_dot[j] += float(m[j][r1]) * u;
      cross[j] += float(m[j][r1]) * w;
    }
  }
  threadgroup float reduction[2][8][K];
  for (uint j = 0; j < K; ++j) {
    cross[j] = simd_sum(cross[j]);
    gram_dot[j] = simd_sum(gram_dot[j]);
    if (lane == 0) {
      reduction[0][sg][j] = cross[j];
      reduction[1][sg][j] = gram_dot[j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < K) {
    float a = 0.0f;
    float b = 0.0f;
    for (uint i = 0; i < 8; ++i) {
      a += reduction[0][i][tid];
      b += reduction[1][i][tid];
    }
    const float ri = rms[draft_row];
    const float mean =
        ri * ri + (2.0f * ri * a + ri * ri * b) / float(p.hidden);
    pv[ulong(batch) * p.scores_lane + p.rmsref_base + r * K + tid] =
        sqrt(mean + p.norm_eps);
  }
}

// One simdgroup per lane walks the table serially. A greedy lane takes
// the argmax chain: the proposal at mask row r is the argmax its score
// row takes under the predecessor lane the walk holds, j = idx[r][j_prev],
// and each row's candidates' probabilities are its score row's softmax at
// the upstream rms scale (confidence the chosen lane's share, as
// _serial_chain emits it). A lane in sampling_mask instead divides each
// score row by the refined RMS dspline_refine_norm computed — the
// normalized logits of upstream serial_rs_table — and draws under the
// lane's joint filter at its temperature (upstream serial_sample_chain:
// softmax, the min-p/top-k/top-p keep mask over rank order, the
// renormalized q, a CDF draw on the row's proposal uniform, chained on
// the SAMPLED predecessor). All outputs land exactly where
// draft_select_plain puts them; a sampled lane's probabilities are the
// filtered q its verify rows' residual draw expects.
kernel void dspline_refine_walk(
    device const uint *idx [[buffer(0)]],
    device const uint *candidates [[buffer(1)]],
    device const float *scores [[buffer(2)]],
    device const float *rms [[buffer(3)]],
    device const float *uniforms [[buffer(4)]],
    device uint *tokens [[buffer(5)]],
    device float *probabilities [[buffer(6)]],
    device float *confidence [[buffer(7)]],
    constant DsplineRefineParams &p [[buffer(8)]],
    uint batch [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  uint predecessor = 0;
  if (!(p.sampling_mask & (1u << batch))) {
    for (uint r = 0; r < p.positions; ++r) {
      const uint n = batch * p.positions + r;
      const uint draft_row = batch * p.query_rows + 1 + r;
      const uint j = idx[n * K + predecessor];
      device const float *row_scores = scores +
          ulong(batch) * p.scores_lane + p.scores_base + r * K * K +
          predecessor * K;
      const float best = row_scores[j];
      const float scale = max(rms[draft_row], 1e-6f);
      const float weight =
          lane < K ? fast::exp((row_scores[lane] - best) / scale) : 0.0f;
      const float mass = simd_sum(weight);
      if (lane < K)
        probabilities[n * K + lane] = weight / mass;
      if (lane == 0) {
        confidence[n] = 1.0f / mass;
        tokens[n] = candidates[n * K + j];
      }
      predecessor = j;
    }
    return;
  }
  const float inverse_temperature = 1.0f / p.temperature[batch];
  const uint top_k = p.top_k[batch];
  const float top_p = p.top_p[batch];
  const float min_p = p.min_p[batch];
  for (uint r = 0; r < p.positions; ++r) {
    const uint n = batch * p.positions + r;
    const uint draft_row = batch * p.query_rows + 1 + r;
    device const float *row_scores = scores +
        ulong(batch) * p.scores_lane + p.scores_base + r * K * K +
        predecessor * K;
    const float refined = scores[ulong(batch) * p.scores_lane +
                                 p.rmsref_base + r * K + predecessor];
    const float score =
        lane < K ? row_scores[lane] / refined : -INFINITY;
    const float maximum = simd_max(score);
    const float mass =
        lane < K ? fast::exp((score - maximum) * inverse_temperature)
                 : 0.0f;
    const float pv = mass / simd_sum(mass);
    // Upstream filter_probs: rank in (value desc, lane asc) order — the
    // number of entries strictly ahead — and their cumulative mass.
    float rank = 0.0f;
    float before = 0.0f;
    for (uint c = 0; c < K; ++c) {
      const float pc = simd_broadcast(pv, c);
      const bool ahead = pc > pv || (pc == pv && c < lane);
      rank += ahead ? 1.0f : 0.0f;
      before += ahead ? pc : 0.0f;
    }
    const float maximum_p = simd_max(pv);
    const bool keep = lane < K && rank < float(top_k) && before <= top_p &&
                      pv >= min_p * maximum_p;
    const float w = keep ? pv : 0.0f;
    const float q = w / simd_sum(w);
    if (lane < K)
      probabilities[n * K + lane] = q;
    // The CDF draw over index order, clamped to the last kept lane as
    // upstream's min(choice, last).
    const float prefix = simd_prefix_inclusive_sum(q);
    const float threshold =
        uniforms[ulong(batch) * RICHENGINE_SAMPLING_UNIFORMS +
                 RICHENGINE_UNIFORM_PROPOSALS + r] *
        simd_broadcast(prefix, K - 1);
    const bool hit =
        lane < K && prefix - q <= threshold && prefix > threshold;
    uint choice = simd_min(hit ? lane : 0xffffffffu);
    if (choice == 0xffffffffu)
      choice = simd_max(keep ? lane : 0u);
    const float scale = max(rms[draft_row], 1e-6f);
    const float weight =
        lane < K ? fast::exp((score - maximum) / scale) : 0.0f;
    if (lane == 0) {
      confidence[n] = 1.0f / simd_sum(weight);
      tokens[n] = candidates[n * K + choice];
    }
    predecessor = choice;
  }
}

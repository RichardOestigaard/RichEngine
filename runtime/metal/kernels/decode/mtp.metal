// The maxwell-moe MTP draft's kernels. The draft keeps a one-layer K/V
// ring whose slot s holds the normed+rotated key and the value of the
// pair (h_{s-1}, emb(x_s)): the MTP layer input of position s - 1, whose
// output predicts x_{s+1}. Fifteen chained passes run per decode step,
// each pass's argmax feeding the next pass's embedding and its own
// stream state feeding the eh rows.

#include "metal/abi/KernelABI.h"
#include "metal/abi/Mtp.h"
#include "metal/abi/Sampling.h"

// The NeoX half-split rotary of a head row whose float4s one lane each of
// a 32-lane group holds: lane l holds dims 4l..4l+3 and pairs with
// l + pairs/4, exchanged by simd_shuffle.
inline float4 mtp_head_rope(float4 v, uint dim, uint pairs, float theta,
                            float position, uint lane) {
  const uint quarter = pairs / 4;
  const uint partner = lane < quarter ? lane + quarter : lane - quarter;
  const float4 other = simd_shuffle(v, partner);
  const float4 low = lane < quarter ? v : other;
  const float4 high = lane < quarter ? other : v;
  const uint base = (lane < quarter ? lane : partner) * 4;
  const float inv = 1.0f / pow(theta, 2.0f / float(dim));
  const float4 angle = position *
      float4(pow(inv, float(base)), pow(inv, float(base + 1)),
             pow(inv, float(base + 2)), pow(inv, float(base + 3)));
  const float4 c = cos(angle), s = sin(angle);
  return lane < quarter ? low * c - high * s : low * s + high * c;
}

// The RMS norm of the head row the 32 lanes share one float4 of.
inline float4 mtp_head_norm(float4 v, device const float *norm, uint dim,
                            float eps, uint lane) {
  const float inv = fast::rsqrt(simd_sum(dot(v, v)) / float(dim) + eps);
  return v * inv * float4(norm[lane * 4], norm[lane * 4 + 1],
                          norm[lane * 4 + 2], norm[lane * 4 + 3]);
}

inline float4 mtp_load_head(device const bfloat *src, uint lane) {
  return float4(float(src[lane * 4]), float(src[lane * 4 + 1]),
                float(src[lane * 4 + 2]), float(src[lane * 4 + 3]));
}

inline void mtp_store_head(device bfloat *out, float4 v, uint lane) {
  out[lane * 4] = bfloat(v.x);
  out[lane * 4 + 1] = bfloat(v.y);
  out[lane * 4 + 2] = bfloat(v.z);
  out[lane * 4 + 3] = bfloat(v.w);
}

// The lanes' ring buffers, one layer of keys and values, as four
// buffer-pair arguments like the draft attention kernels'.
struct MtpLanes {
  device bfloat *k0;
  device bfloat *k1;
  device bfloat *k2;
  device bfloat *k3;
  device bfloat *v0;
  device bfloat *v1;
  device bfloat *v2;
  device bfloat *v3;
};

inline device bfloat *mtp_lane_key(thread const MtpLanes &l, uint lane) {
  return lane == 0 ? l.k0 : lane == 1 ? l.k1 : lane == 2 ? l.k2 : l.k3;
}
inline device bfloat *mtp_lane_value(thread const MtpLanes &l, uint lane) {
  return lane == 0 ? l.v0 : lane == 1 ? l.v1 : lane == 2 ? l.v2 : l.v3;
}

// The flat token list the embedding gather reads. Mode 0 is the decode
// anchor (the packed draft input's row 0 of each lane); mode 2 a prefill
// span's rolled tokens (row + 1, clamped — the kv store drops the row it
// repeats).
kernel void mtp_gather_tokens(device const uint *input [[buffer(0)]],
                              device uint *tokens [[buffer(1)]],
                              constant MtpGatherTokensParams &p [[buffer(2)]],
                              uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  if (p.mode == 0)
    tokens[row] = input[row * p.stride];
  else
    tokens[row] = input[min(row + p.offset + 1, p.end ? p.end - 1 : 0u)];
}

// The 1024-thread group's sum of a per-thread partial.
inline float mtp_group_sum(float v, uint lane, uint sg,
                           threadgroup float *partials) {
  const float s = simd_sum(v);
  if (!lane) partials[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < 32; ++i) total += partials[i];
  return total;
}

// The eh rows of `rows` MTP inputs whose stream state is bf16: columns
// [0, hidden) the normed embedding row, [hidden, 2 * hidden) the hnormed
// stream — the fused eh projection's input, one 1024-thread group per
// (row, stream). source 0 reads captured row sourceBase + row (the
// commit's verify rows and the prefill's capture rows alike); source 1
// the lane's anchor row (decode pass 0), one of four state-lane buffers.
kernel void mtp_ehtoken(device const bfloat *captured [[buffer(0)]],
                        device const bfloat *embed [[buffer(1)]],
                        device const float *hnorm [[buffer(2)]],
                        device const bfloat *a0 [[buffer(3)]],
                        device const bfloat *a1 [[buffer(4)]],
                        device const bfloat *a2 [[buffer(5)]],
                        device const bfloat *a3 [[buffer(6)]],
                        device bfloat *eh [[buffer(7)]],
                        constant MtpEhTokenParams &p [[buffer(8)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[32];
  const uint row = group.x, s = group.y;
  if (row >= p.rows) return;
  const uint width = p.streams * p.hidden;
  device const bfloat *anchor =
      row == 0 ? a0 : row == 1 ? a1 : row == 2 ? a2 : a3;
  device const bfloat *stream =
      (p.source ? anchor : captured) +
      ulong(p.source ? 0 : p.sourceBase + row) * width + s * p.hidden;
  float acc = 0.0f;
  for (uint i = tid; i < p.hidden; i += 1024) {
    const float v = float(stream[i]);
    acc += v * v;
  }
  const float total = mtp_group_sum(acc, lane, sg, partials);
  const float inv = fast::rsqrt(total / float(p.hidden) + 1e-6f);
  device bfloat *out = eh + (ulong(row) * p.streams + s) * 2 * p.hidden;
  for (uint i = tid; i < p.hidden; i += 1024) {
    out[i] = embed[ulong(row) * p.hidden + i];
    out[p.hidden + i] = bfloat(float(stream[i]) * inv * hnorm[i]);
  }
}

// The chain variant: the previous pass's fp32 decoder streams.
kernel void mtp_ehtoken_chain(
    device const float *streams [[buffer(0)]],
    device const bfloat *embed [[buffer(1)]],
    device const float *hnorm [[buffer(2)]],
    device bfloat *eh [[buffer(3)]],
    constant MtpEhTokenParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[32];
  const uint row = group.x, s = group.y;
  device const float *stream =
      streams + ulong(row) * p.streams * p.hidden + s * p.hidden;
  float acc = 0.0f;
  for (uint i = tid; i < p.hidden; i += 1024) acc += stream[i] * stream[i];
  const float total = mtp_group_sum(acc, lane, sg, partials);
  const float inv = fast::rsqrt(total / float(p.hidden) + 1e-6f);
  device bfloat *out = eh + (ulong(row) * p.streams + s) * 2 * p.hidden;
  for (uint i = tid; i < p.hidden; i += 1024) {
    out[i] = embed[ulong(row) * p.hidden + i];
    out[p.hidden + i] = bfloat(stream[i] * inv * hnorm[i]);
  }
}

// The eh projection's bf16 stream rows into the fp32 decoder streams.
kernel void mtp_stream_init(device const bfloat *input [[buffer(0)]],
                            device float *streams [[buffer(1)]],
                            constant MtpStreamInitParams &p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  const uint64_t total = uint64_t{p.rows} * p.width;
  if (gid >= total) return;
  streams[gid] = float(input[gid]);
}

// One 32-lane threadgroup per (row, head): mode 0 (decode pass)
// normalizes and rotates query heads into `outQueries` and the kv heads'
// keys into the ring slot, copying values in; mode 1 (commit rows) stores
// the retained rows' keys and values — row % stride < retained[lane] — at
// slot startPosition[lane] + row % stride.
kernel void mtp_kv_store(device const bfloat *queries [[buffer(0)]],
                         device const bfloat *keys [[buffer(1)]],
                         device const bfloat *values [[buffer(2)]],
                         device const float *queryNorm [[buffer(3)]],
                         device const float *keyNorm [[buffer(4)]],
                         device const uint *positions [[buffer(5)]],
                         device const uint *retained [[buffer(6)]],
                         device bfloat *outQueries [[buffer(7)]],
                         device bfloat *keys0 [[buffer(8)]],
                         device bfloat *keys1 [[buffer(9)]],
                         device bfloat *keys2 [[buffer(10)]],
                         device bfloat *keys3 [[buffer(11)]],
                         device bfloat *values0 [[buffer(12)]],
                         device bfloat *values1 [[buffer(13)]],
                         device bfloat *values2 [[buffer(14)]],
                         device bfloat *values3 [[buffer(15)]],
                         constant MtpKvStoreParams &p [[buffer(16)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x, head = group.y;
  if (row >= p.rows) return;
  const MtpLanes rings{keys0, keys1, keys2, keys3,
                      values0, values1, values2, values3};
  uint batch = row, position = 0;
  if (p.mode == 0) {
    position = positions[batch * p.posStride] - 1 + p.pass;
  } else {
    batch = row / p.stride;
    const uint pos = row % p.stride;
    if (pos >= retained[batch]) return;
    position = p.startPositions[batch] + pos;
  }
  const uint slot = position % p.ring;
  if (p.mode == 0 && head < p.heads) {
    float4 q = mtp_load_head(queries + ulong(row) * p.heads * p.headDim +
                                 head * p.headDim,
                             lane);
    q = mtp_head_norm(q, queryNorm, p.headDim, p.rmsEps, lane);
    q = mtp_head_rope(q, p.headDim, p.rotaryPairs, p.ropeTheta,
                      float(position), lane);
    mtp_store_head(outQueries + ulong(row) * p.heads * p.headDim +
                       head * p.headDim,
                   q, lane);
    return;
  }
  const uint kv = p.mode == 0 ? head - p.heads : head;
  float4 k = mtp_load_head(keys + ulong(row) * p.kvHeads * p.headDim +
                               kv * p.headDim,
                           lane);
  k = mtp_head_norm(k, keyNorm, p.headDim, p.rmsEps, lane);
  k = mtp_head_rope(k, p.headDim, p.rotaryPairs, p.ropeTheta,
                    float(position), lane);
  mtp_store_head(mtp_lane_key(rings, batch) +
                     (ulong(kv) * p.ring + slot) * p.headDim,
                 k, lane);
  device const bfloat *vsrc =
      values + ulong(row) * p.kvHeads * p.headDim + kv * p.headDim + lane * 4;
  device bfloat *vout = mtp_lane_value(rings, batch) +
      ulong(kv) * p.ring * p.headDim + lane * 4 * p.ring + slot;
  vout[0] = vsrc[0];
  vout[p.ring] = vsrc[1];
  vout[2 * p.ring] = vsrc[2];
  vout[3 * p.ring] = vsrc[3];
}

// One prefill span's keys and values into its ring: row r commits slot
// base + r, dropped when the slot reaches `end` — the sequence's last
// row's rolled token is not in the prompt.
kernel void mtp_kv_store_prefill(device const bfloat *keys [[buffer(0)]],
                                 device const bfloat *values [[buffer(1)]],
                                 device const float *keyNorm [[buffer(2)]],
                                 device bfloat *ringKeys [[buffer(3)]],
                                 device bfloat *ringValues [[buffer(4)]],
                                 constant MtpPrefillKvParams &p [[buffer(5)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x, kv = group.y;
  if (row >= p.rows) return;
  const uint position = p.base + row;
  if (position >= p.end) return;
  const uint slot = position % p.ring;
  float4 k = mtp_load_head(keys + ulong(row) * p.kvHeads * p.headDim +
                               kv * p.headDim,
                           lane);
  k = mtp_head_norm(k, keyNorm, p.headDim, p.rmsEps, lane);
  k = mtp_head_rope(k, p.headDim, p.rotaryPairs, p.ropeTheta,
                    float(position), lane);
  mtp_store_head(ringKeys + (ulong(kv) * p.ring + slot) * p.headDim, k, lane);
  device const bfloat *vsrc =
      values + ulong(row) * p.kvHeads * p.headDim + kv * p.headDim + lane * 4;
  device bfloat *vout = ringValues + ulong(kv) * p.ring * p.headDim +
      lane * 4 * p.ring + slot;
  vout[0] = vsrc[0];
  vout[p.ring] = vsrc[1];
  vout[2 * p.ring] = vsrc[2];
  vout[3 * p.ring] = vsrc[3];
}

// The decode pass's GQA: one 32-lane threadgroup per (lane, query head).
// Every lane scores its position tile while it accumulates the four
// output dims it owns; the online softmax folds each tile into the
// running max and sum.
kernel void mtp_attention(device const bfloat *queries [[buffer(0)]],
                          device bfloat *keys0 [[buffer(1)]],
                          device bfloat *keys1 [[buffer(2)]],
                          device bfloat *keys2 [[buffer(3)]],
                          device bfloat *keys3 [[buffer(4)]],
                          device bfloat *values0 [[buffer(5)]],
                          device bfloat *values1 [[buffer(6)]],
                          device bfloat *values2 [[buffer(7)]],
                          device bfloat *values3 [[buffer(8)]],
                          device const uint *positions [[buffer(9)]],
                          device bfloat *out [[buffer(10)]],
                          constant MtpAttentionParams &p [[buffer(11)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float q[128];
  const uint batch = group.x, head = group.y;
  const uint kv = head / (p.heads / p.kvHeads);
  const MtpLanes rings{keys0, keys1, keys2, keys3,
                      values0, values1, values2, values3};
  device const bfloat *rk = mtp_lane_key(rings, batch);
  device const bfloat *rv = mtp_lane_value(rings, batch);
  const uint length = positions[batch * p.posStride] + p.pass;
  {
    device const bfloat *src = queries + ulong(batch) * p.heads * p.headDim +
                               head * p.headDim + lane * 4;
    q[lane * 4] = float(src[0]);
    q[lane * 4 + 1] = float(src[1]);
    q[lane * 4 + 2] = float(src[2]);
    q[lane * 4 + 3] = float(src[3]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device const bfloat *khead = rk + ulong(kv) * p.ring * p.headDim;
  device const bfloat *vhead = rv + ulong(kv) * p.ring * p.headDim;
  const float scale = fast::rsqrt(float(p.headDim));
  float m = -INFINITY, sum = 0.0f;
  float4 acc = float4(0.0f);
  for (uint tile = 0; tile < length; tile += 32) {
    const uint position = tile + lane;
    float score = -INFINITY;
    if (position < length) {
      device const bfloat *k = khead + (position % p.ring) * p.headDim;
      float dot = 0.0f;
      for (uint d = 0; d < p.headDim; d += 4) {
        const float4 k4 = float4(float(k[d]), float(k[d + 1]),
                                 float(k[d + 2]), float(k[d + 3]));
        dot += q[d] * k4.x + q[d + 1] * k4.y + q[d + 2] * k4.z +
               q[d + 3] * k4.w;
      }
      score = dot * scale;
    }
    const float tileMax = simd_max(score);
    const float next = max(m, tileMax);
    const float rescale = exp(m - next);
    const float weight = position < length ? exp(score - next) : 0.0f;
    sum = sum * rescale + simd_sum(weight);
    acc *= rescale;
    // Every output quad accumulates all positions: lane i's weight is
    // broadcast with simd_shuffle while the lane reads its own dims'
    // ring column of that position.
    const uint count = min(32u, length - tile);
    for (uint i = 0; i < count; ++i) {
      const float wi = simd_shuffle(weight, i);
      const uint pos = tile + i;
      device const bfloat *v =
          vhead + lane * 4 * p.ring + pos % p.ring;
      const float4 v4 = float4(float(v[0]), float(v[p.ring]),
                               float(v[2 * p.ring]), float(v[3 * p.ring]));
      acc += wi * v4;
    }
    m = next;
  }
  // RICHENGINE_MTP_DUMP debug: head 0 writes diagnostics into row 1+ of
  // the 8-row output scratch — length, kv head, max score, weight sum.
  if (head == 0 && lane == 0) {
    device bfloat *dbg = out + ulong(p.heads) * p.headDim;
    dbg[0] = bfloat(float(length));
    dbg[1] = bfloat(float(m));
    dbg[2] = bfloat(float(sum));
    dbg[3] = bfloat(float(q[0]));
    dbg[4] = bfloat(float(rk[0]));
    dbg[5] = bfloat(float(rv[0]));
  }
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  device bfloat *dst =
      out + ulong(batch) * p.heads * p.headDim + head * p.headDim + lane * 4;
  const float4 o = acc * inv;
  dst[0] = bfloat(o.x);
  dst[1] = bfloat(o.y);
  dst[2] = bfloat(o.z);
  dst[3] = bfloat(o.w);
}

// draft_select_top16_sharded's pass-major variant: proposal position p of
// lane b reads logits row (p * lanes + b) * stride, the contiguous pass
// rows the MTP head writes. The partial layout stays lane-major for
// draft_select_plain.
kernel void mtp_select_top16_sharded(
    device const float *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant MtpTop16Params &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Positions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Shards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
  constexpr uint K = RICHENGINE_DRAFT_CANDIDATES;
  uint batch = group / (Positions * Shards);
  uint local = group % (Positions * Shards);
  uint position = local / Shards;
  uint shard = local % Shards;
  ulong row_start =
      ulong(position * params.lanes + batch) * params.stride;
  const uint shard_tokens = (params.vocabulary + Shards - 1) / Shards;
  const uint begin = min(shard * shard_tokens, params.vocabulary);
  const uint end = min(begin + shard_tokens, params.vocabulary);
  float best = -INFINITY;
  uint best_id = 0xffffffffu;
  for (uint token = begin + thread_index; token < end; token += 256) {
    const float value = logits[row_start + token];
    if (value > best || (value == best && token < best_id)) {
      best = value;
      best_id = token;
    }
  }
  threadgroup float round_values[2][8];
  threadgroup uint round_ids[2][8];
  for (uint rank = 0; rank < K; ++rank) {
    const float simd_value = simd_max(best);
    const uint simd_id =
        simd_min(best == simd_value ? best_id : 0xffffffffu);
    const uint slot = rank & 1;
    if (!lane) {
      round_values[slot][simd_group] = simd_value;
      round_ids[slot][simd_group] = simd_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float head = round_values[slot][0];
    uint head_id = round_ids[slot][0];
    for (uint other = 1; other < 8; ++other) {
      const float v = round_values[slot][other];
      const uint i = round_ids[slot][other];
      if (v > head || (v == head && i < head_id)) {
        head = v;
        head_id = i;
      }
    }
    if (thread_index == rank) {
      partial_ids[group * K + rank] = head_id;
      partial_values[group * K + rank] = head;
    }
    if (best == head && best_id == head_id) {
      best = -INFINITY;
      best_id = 0xffffffffu;
    }
  }
}

// The pass logits rows' greedy token: the next pass's embedding input.
kernel void mtp_argmax(device const float *logits [[buffer(0)]],
                       device uint *tokens [[buffer(1)]],
                       constant MtpArgmaxParams &p [[buffer(2)]],
                       uint batch [[threadgroup_position_in_grid]],
                       uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float values[256];
  threadgroup uint ids[256];
  if (batch >= p.lanes) return;
  float best = -INFINITY;
  uint best_id = 0xffffffffu;
  device const float *row = logits + ulong(batch) * p.stride;
  for (uint token = thread_index; token < p.vocabulary; token += 256) {
    const float value = row[token];
    if (value > best || (value == best && token < best_id)) {
      best = value;
      best_id = token;
    }
  }
  values[thread_index] = best;
  ids[thread_index] = best_id;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 128; step; step >>= 1) {
    if (thread_index < step) {
      const float other = values[thread_index + step];
      const uint other_id = ids[thread_index + step];
      if (other > best || (other == best && other_id < best_id)) {
        best = other;
        best_id = other_id;
        values[thread_index] = best;
        ids[thread_index] = best_id;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!thread_index) tokens[batch] = best_id;
}

// The pass-0 stream anchor, per state lane. mtp_anchor_commit refreshes
// every dispatch lane's anchor from its last retained capture row;
// mtp_anchor_prefill lands one sequence's last captured row on its state
// lane's slot.
kernel void mtp_anchor_commit(device const bfloat *captured [[buffer(0)]],
                              device const uint *retained [[buffer(1)]],
                              device bfloat *a0 [[buffer(2)]],
                              device bfloat *a1 [[buffer(3)]],
                              device bfloat *a2 [[buffer(4)]],
                              device bfloat *a3 [[buffer(5)]],
                              constant MtpAnchorParams &p [[buffer(6)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  if (group >= p.lanes) return;
  const MtpLanes lanes{a0, a1, a2, a3, a0, a1, a2, a3};
  device const bfloat *src =
      captured +
      (ulong(group) * p.sourceStride + retained[group] - 1) * p.width;
  device bfloat *dst = mtp_lane_key(lanes, group);
  for (uint i = tid; i < p.width; i += 256) dst[i] = src[i];
}

kernel void mtp_anchor_prefill(device const bfloat *captured [[buffer(0)]],
                               device bfloat *anchor [[buffer(1)]],
                               constant MtpAnchorParams &p [[buffer(2)]],
                               uint tid [[thread_index_in_threadgroup]]) {
  for (uint i = tid; i < p.width; i += 256) anchor[i] = captured[i];
}

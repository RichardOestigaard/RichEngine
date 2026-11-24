// The nemotron MTP draft's kernels (model/NemotronMtpDraft.cpp): the
// nextn block of a Nemotron-3.5-Lightning target — one NoPE GQA + ungated
// relu-squared MoE decoder layer over the trunk's 2688-wide hidden,
// chained one pass per proposal. The draft keeps the maxwell MTP draft's
// K/V ring semantics — slot s holds the key and value of the pair
// (h_s, x_{s+1}) — but its attention is NoPE and norm-free, its experts
// read bf16 slabs directly, and its norms take the target's 1e-5 epsilon.
// The pass's embedding, attention flash, top-16 select, argmax, stream
// copy and anchor commits reuse kernels/decode/mtp.metal's.

#include "metal/abi/KernelABI.h"
#include "metal/abi/Nmtp.h"
#include "metal/abi/Mtp.h"

// One lane's ring buffers (the draft runs one lane per dispatch; the
// four-way structure stays so a future batched call site can bind them).
struct NmtpLanes {
  device bfloat *k0;
  device bfloat *k1;
  device bfloat *k2;
  device bfloat *k3;
  device bfloat *v0;
  device bfloat *v1;
  device bfloat *v2;
  device bfloat *v3;
};

inline device bfloat *nmtp_lane_key(thread const NmtpLanes &l, uint lane) {
  return lane == 0 ? l.k0 : lane == 1 ? l.k1 : lane == 2 ? l.k2 : l.k3;
}
inline device bfloat *nmtp_lane_value(thread const NmtpLanes &l, uint lane) {
  return lane == 0 ? l.v0 : lane == 1 ? l.v1 : lane == 2 ? l.v2 : l.v3;
}

// The 1024-thread group's sum of a per-thread partial.
inline float nmtp_group_sum(float v, uint lane, uint sg,
                            threadgroup float *partials) {
  const float s = simd_sum(v);
  if (!lane) partials[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!sg && !lane) {
    float total = 0.0f;
    for (uint i = 0; i < 32; ++i) total += partials[i];
    partials[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return partials[0];
}

// The eh row of `rows` MTP inputs whose hidden is bf16: columns
// [0, hidden) the normed embedding row, [hidden, 2 * hidden) the hnormed
// hidden — the fused eh projection's input, one 1024-thread group per
// row. source 0 reads captured row sourceBase + row (the commit's verify
// rows and the prefill's capture rows alike); source 1 the lane's anchor
// row (decode pass 0), one of four state-lane buffers.
kernel void nmtp_ehtoken(device const bfloat *captured [[buffer(0)]],
                         device const bfloat *embed [[buffer(1)]],
                         device const bfloat *hnorm [[buffer(2)]],
                         device const bfloat *a0 [[buffer(3)]],
                         device const bfloat *a1 [[buffer(4)]],
                         device const bfloat *a2 [[buffer(5)]],
                         device const bfloat *a3 [[buffer(6)]],
                         device bfloat *eh [[buffer(7)]],
                         constant NmtpEhTokenParams &p [[buffer(8)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[32];
  if (row >= p.rows) return;
  device const bfloat *anchor =
      row == 0 ? a0 : row == 1 ? a1 : row == 2 ? a2 : a3;
  device const bfloat *hidden =
      (p.source ? anchor : captured + ulong(p.sourceBase + row) * p.hidden);
  float acc = 0.0f;
  for (uint i = tid; i < p.hidden; i += 1024) {
    const float v = float(hidden[i]);
    acc += v * v;
  }
  const float total = nmtp_group_sum(acc, lane, sg, partials);
  const float inv = fast::rsqrt(total / float(p.hidden) + p.rmsEps);
  device bfloat *out = eh + ulong(row) * 2 * p.hidden;
  for (uint i = tid; i < p.hidden; i += 1024) {
    out[i] = embed[ulong(row) * p.hidden + i];
    out[p.hidden + i] = bfloat(float(hidden[i]) * inv * float(hnorm[i]));
  }
}

// The chain variant: the previous pass's fp32 hidden.
kernel void nmtp_ehtoken_chain(
    device const float *streams [[buffer(0)]],
    device const bfloat *embed [[buffer(1)]],
    device const bfloat *hnorm [[buffer(2)]],
    device bfloat *eh [[buffer(3)]],
    constant NmtpEhTokenParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[32];
  if (row >= p.rows) return;
  device const float *hidden = streams + ulong(row) * p.hidden;
  float acc = 0.0f;
  for (uint i = tid; i < p.hidden; i += 1024) acc += hidden[i] * hidden[i];
  const float total = nmtp_group_sum(acc, lane, sg, partials);
  const float inv = fast::rsqrt(total / float(p.hidden) + p.rmsEps);
  device bfloat *out = eh + ulong(row) * 2 * p.hidden;
  for (uint i = tid; i < p.hidden; i += 1024) {
    out[i] = embed[ulong(row) * p.hidden + i];
    out[p.hidden + i] = bfloat(hidden[i] * inv * float(hnorm[i]));
  }
}

// One 32-lane threadgroup per (row, kv head): the NoPE layer's keys and
// values into the ring — no norms, no rotary. Mode 0 (decode pass): slot
// positions[batch * posStride] - 1 + pass; mode 1 (commit rows): slot
// startPosition[batch] + row % stride while row % stride < retained.
kernel void nmtp_kv_store(device const bfloat *keys [[buffer(0)]],
                          device const bfloat *values [[buffer(1)]],
                          device const uint *positions [[buffer(2)]],
                          device const uint *retained [[buffer(3)]],
                          device bfloat *keys0 [[buffer(4)]],
                          device bfloat *keys1 [[buffer(5)]],
                          device bfloat *keys2 [[buffer(6)]],
                          device bfloat *keys3 [[buffer(7)]],
                          device bfloat *values0 [[buffer(8)]],
                          device bfloat *values1 [[buffer(9)]],
                          device bfloat *values2 [[buffer(10)]],
                          device bfloat *values3 [[buffer(11)]],
                          constant NmtpKvStoreParams &p [[buffer(12)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x, kv = group.y;
  if (row >= p.rows) return;
  const NmtpLanes rings{keys0, keys1, keys2, keys3,
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
  device bfloat *kd = nmtp_lane_key(rings, batch) +
                      (ulong(kv) * p.ring + slot) * p.headDim;
  device const bfloat *ks = keys + ulong(row) * p.kvHeads * p.headDim +
                            kv * p.headDim;
  for (uint d = lane * 4; d < p.headDim; d += 128) {
    kd[d] = ks[d];
    kd[d + 1] = ks[d + 1];
    kd[d + 2] = ks[d + 2];
    kd[d + 3] = ks[d + 3];
  }
  device const bfloat *vsrc =
      values + ulong(row) * p.kvHeads * p.headDim + kv * p.headDim + lane * 4;
  device bfloat *vout = nmtp_lane_value(rings, batch) +
      ulong(kv) * p.ring * p.headDim + lane * 4 * p.ring + slot;
  vout[0] = vsrc[0];
  vout[p.ring] = vsrc[1];
  vout[2 * p.ring] = vsrc[2];
  vout[3 * p.ring] = vsrc[3];
}

// One prefill span's keys and values into its ring: row r commits slot
// base + r, dropped when the slot reaches `end` — the sequence's last
// row's rolled token is not in the prompt.
kernel void nmtp_kv_store_prefill(device const bfloat *keys [[buffer(0)]],
                                  device const bfloat *values [[buffer(1)]],
                                  device bfloat *ringKeys [[buffer(2)]],
                                  device bfloat *ringValues [[buffer(3)]],
                                  constant NmtpPrefillKvParams &p [[buffer(4)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x, kv = group.y;
  if (row >= p.rows) return;
  const uint position = p.base + row;
  if (position >= p.end) return;
  const uint slot = position % p.ring;
  device bfloat *kd = ringKeys + (ulong(kv) * p.ring + slot) * p.headDim;
  device const bfloat *ks = keys + ulong(row) * p.kvHeads * p.headDim +
                            kv * p.headDim;
  for (uint d = lane * 4; d < p.headDim; d += 128) {
    kd[d] = ks[d];
    kd[d + 1] = ks[d + 1];
    kd[d + 2] = ks[d + 2];
    kd[d + 3] = ks[d + 3];
  }
  device const bfloat *vsrc =
      values + ulong(row) * p.kvHeads * p.headDim + kv * p.headDim + lane * 4;
  device bfloat *vout = ringValues + ulong(kv) * p.ring * p.headDim +
      lane * 4 * p.ring + slot;
  vout[0] = vsrc[0];
  vout[p.ring] = vsrc[1];
  vout[2 * p.ring] = vsrc[2];
  vout[3 * p.ring] = vsrc[3];
}

// The sigmoid router: one 1024-thread group per row — each of the 32
// simdgroups scores four experts (its lane covering a strided 128-element
// window per round), then thread 0 picks the top-K. As HF and the shared
// moe_route_select_sigmoid kernels: the exp_probs_b correction bias orders
// the selection only — the routing weights are the winners' UNbiased
// sigmoid scores normalized over the winners, times the routed scale.
// routes[row * 16] holds the expert ids in [0, topK) and the fp32
// weights in [8, 8 + topK).
kernel void nmtp_moe_route(device const bfloat *input [[buffer(0)]],
                           device const bfloat *router [[buffer(1)]],
                           device const bfloat *bias [[buffer(2)]],
                           device uint *routes [[buffer(3)]],
                           constant NmtpMoeRouteParams &p [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float probs[128];    // the winners' unbiased sigmoid scores
  threadgroup float keys[128];     // probs + bias: the selection order
  const uint rounds = (p.experts + 31) / 32;
  device const bfloat *x = input + ulong(row) * p.hidden;
  for (uint r = 0; r < rounds; ++r) {
    const uint e = sg + r * 32;
    if (e >= p.experts) break;
    device const bfloat *w = router + ulong(e) * p.hidden;
    float acc = 0.0f;
    for (uint i = lane * 4; i < p.hidden; i += 128) {
      acc += float(x[i]) * float(w[i]) + float(x[i + 1]) * float(w[i + 1]) +
             float(x[i + 2]) * float(w[i + 2]) +
             float(x[i + 3]) * float(w[i + 3]);
    }
    const float prob = 1.0f / (1.0f + exp(-simd_sum(acc)));
    if (!lane && e < 128) {
      probs[e] = prob;
      keys[e] = prob + float(bias[e]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  device uint *out = routes + ulong(row) * 16;
  device float *weights = reinterpret_cast<device float *>(out + 8);
  float total = 0.0f;
  for (uint k = 0; k < p.topK; ++k) {
    float best = -INFINITY;
    uint bestId = 0;
    for (uint e = 0; e < p.experts; ++e) {
      const float s = keys[e];
      if (s > best) {
        best = s;
        bestId = e;
      }
    }
    keys[bestId] = -INFINITY;
    const float w = probs[bestId];
    out[k] = bestId;
    weights[k] = w;
    total += w;
  }
  const float inv = total > 0.0f ? p.weightScale / total : 0.0f;
  for (uint k = 0; k < p.topK; ++k) weights[k] *= inv;
}

// The ungated up pass: 256 threads per (column tile, route slot) — the
// routed slots' experts from the route record, slot topK the shared
// expert. intermediate[slot][n] = relu(x . W_e[n])^2 as bf16.
kernel void nmtp_moe_up(device const bfloat *input [[buffer(0)]],
                        device const bfloat *expertUp [[buffer(1)]],
                        device const bfloat *sharedUp [[buffer(2)]],
                        device const uint *routes [[buffer(3)]],
                        device bfloat *intermediate [[buffer(4)]],
                        constant NmtpMoeParams &p [[buffer(5)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
  const uint slot = group.y;
  const uint width = slot == p.topK ? p.sharedIntermediate : p.intermediate;
  const uint n = group.x * 256 + tid;
  if (n >= width) return;
  device const bfloat *w =
      slot == p.topK
          ? sharedUp + ulong(n) * p.hidden
          : expertUp + (ulong(routes[slot]) * p.intermediate + n) * p.hidden;
  float acc = 0.0f;
  for (uint i = 0; i < p.hidden; i += 4) {
    acc += float(input[i]) * float(w[i]) + float(input[i + 1]) * float(w[i + 1]) +
           float(input[i + 2]) * float(w[i + 2]) +
           float(input[i + 3]) * float(w[i + 3]);
  }
  const float r = max(acc, 0.0f);
  intermediate[slot * p.interStride + n] = bfloat(r * r);
}

// The ungated down pass: 256 threads per (hidden column tile, route
// slot) — fp32 per-slot partials the combine weights and sums.
kernel void nmtp_moe_down(device const bfloat *intermediate [[buffer(0)]],
                          device const bfloat *expertDown [[buffer(1)]],
                          device const bfloat *sharedDown [[buffer(2)]],
                          device const uint *routes [[buffer(3)]],
                          device float *partials [[buffer(4)]],
                          constant NmtpMoeParams &p [[buffer(5)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  const uint slot = group.y;
  const uint k = slot == p.topK ? p.sharedIntermediate : p.intermediate;
  const uint c = group.x * 256 + tid;
  if (c >= p.hidden) return;
  device const bfloat *w =
      slot == p.topK
          ? sharedDown + ulong(c) * p.sharedIntermediate
          : expertDown + (ulong(routes[slot]) * p.hidden + c) * p.intermediate;
  device const bfloat *x = intermediate + slot * p.interStride;
  float acc = 0.0f;
  for (uint i = 0; i < k; i += 4) {
    acc += float(x[i]) * float(w[i]) + float(x[i + 1]) * float(w[i + 1]) +
           float(x[i + 2]) * float(w[i + 2]) +
           float(x[i + 3]) * float(w[i + 3]);
  }
  partials[slot * p.hidden + c] = acc;
}

// output = residual + sum_s w_s * partial_s + partial_shared — the routed
// weights normalized and scaled by the router, the shared expert's unit
// weight implicit. Deterministic: the sum folds the slots in order.
kernel void nmtp_moe_combine(device const bfloat *residual [[buffer(0)]],
                             device const float *partials [[buffer(1)]],
                             device const uint *routes [[buffer(2)]],
                             device bfloat *output [[buffer(3)]],
                             constant NmtpMoeParams &p [[buffer(4)]],
                             uint tid [[thread_position_in_grid]]) {
  if (tid >= p.hidden) return;
  device const float *weights =
      reinterpret_cast<device const float *>(routes + 8);
  float acc = float(residual[tid]);
  for (uint slot = 0; slot < p.topK; ++slot)
    acc += weights[slot] * partials[slot * p.hidden + tid];
  acc += partials[p.topK * p.hidden + tid];
  output[tid] = bfloat(acc);
}

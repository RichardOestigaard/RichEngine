#include "metal/abi/KernelABI.h"
#include "metal/abi/Mhc.h"
#include "metal/kernels/common/activation.h"

#include <metal_stdlib>

using namespace metal;

// Manifold-constrained hyper-connections (metal/abi/Mhc.h): the
// maxwell_moe residual. The streams are fp32 [rows][streams][hidden].
// The mapping projections that produce these kernels' logits — a
// module's streams*hidden -> mixWidth and the head's -> streams — run
// through ops::Linear on the host into fp32 rows; everything here is
// fp32, and the gates' and Sinkhorn's epsilons are the reference
// module's hard-coded 1e-6.
constant constexpr uint kRowThreads = RICHENGINE_MHC_ROW_THREADS;
constant constexpr uint kSimdgroups = kRowThreads / 32;
constant constexpr uint kStreams = RICHENGINE_MHC_MAXIMUM_STREAMS;
constant constexpr uint kResMix = kStreams * kStreams;
constant constexpr uint kMixWidth = RICHENGINE_MHC_MAXIMUM_MIX_WIDTH;
constant constexpr float kEps = 1e-6f;

// rsqrt(mean(x^2) + eps) of a row's `width` fp32 stream values — the
// module's dynamic scale, 1 / (||x|| / sqrt(nC) + eps). Every thread
// leaves with it.
inline float mhc_rms_inverse(device const float *x, uint width, uint tid,
                             uint lane, uint sg,
                             threadgroup float *scratch) {
  float sum = 0.0f;
  for (uint i = tid; i < width; i += kRowThreads) {
    const float v = x[i];
    sum += v * v;
  }
  sum = simd_sum(sum);
  if (lane == 0)
    scratch[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  float total = 0.0f;
  for (uint g = 0; g < kSimdgroups; ++g)
    total += scratch[g];
  return rsqrt(total / float(width) + kEps);
}

// A row's mapping from its projection logits: h[c] = r * logits[c] *
// alpha(c) + bias[c] with alpha the pre/post/res scalar of the channel's
// range; h_pre = sigmoid(h) + eps, h_post = 2 * sigmoid(h), and h_res
// the residual logits' streams x streams matrix softmaxed per row then
// Sinkhorn-normalized to the manifold of doubly-stochastic matrices.
kernel void mhc_mappings(device const float *streams [[buffer(0)]],
                         device const float *logits [[buffer(1)]],
                         device const float *alphas [[buffer(2)]],
                         device const float *bias [[buffer(3)]],
                         device float *mappings [[buffer(4)]],
                         constant MhcParams &p [[buffer(5)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups];
  threadgroup float h[kMixWidth];
  if (row >= p.rows)
    return;
  const uint n = p.streams;
  const float r =
      mhc_rms_inverse(streams + ulong(row) * n * p.hidden, n * p.hidden,
                      tid, lane, sg, scratch);
  device const float *logit = logits + ulong(row) * p.mixWidth;
  device float *out = mappings + ulong(row) * p.mixWidth;
  for (uint c = tid; c < p.mixWidth; c += kRowThreads) {
    const float alpha = alphas[c < n ? 0 : c < 2 * n ? 1 : 2];
    const float v = r * logit[c] * alpha + bias[c];
    h[c] = v;
    if (c < n)
      out[c] = richengine_sigmoid(v) + kEps;
    else if (c < 2 * n)
      out[c] = 2.0f * richengine_sigmoid(v);
  }
  // The residual logits become M[t][s], the weight with which target
  // stream s reads source stream t: softmax over s in each row t, then
  // Sinkhorn's alternating row/column normalizations. Sixteen values —
  // one thread runs it.
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  if (tid != 0)
    return;
  float m[kResMix];
  for (uint t = 0; t < n; ++t)
    for (uint s = 0; s < n; ++s)
      m[t * n + s] = h[2 * n + t * n + s];
  for (uint t = 0; t < n; ++t) {
    float hi = m[t * n];
    for (uint s = 1; s < n; ++s)
      hi = max(hi, m[t * n + s]);
    float sum = 0.0f;
    for (uint s = 0; s < n; ++s) {
      m[t * n + s] = fast::exp2(1.44269504089f * (m[t * n + s] - hi));
      sum += m[t * n + s];
    }
    for (uint s = 0; s < n; ++s)
      m[t * n + s] /= sum;
  }
  // The reference's exact order: M + eps then a column normalization,
  // and (sinkhornIterations - 1) row/column pairs after it.
  for (uint i = 0; i < n * n; ++i)
    m[i] += kEps;
  for (uint s = 0; s < n; ++s) {
    float col = 0.0f;
    for (uint t = 0; t < n; ++t)
      col += m[t * n + s];
    for (uint t = 0; t < n; ++t)
      m[t * n + s] /= col + kEps;
  }
  for (uint it = 1; it < p.sinkhornIterations; ++it) {
    for (uint t = 0; t < n; ++t) {
      float rowsum = 0.0f;
      for (uint s = 0; s < n; ++s)
        rowsum += m[t * n + s];
      for (uint s = 0; s < n; ++s)
        m[t * n + s] /= rowsum + kEps;
    }
    for (uint s = 0; s < n; ++s) {
      float col = 0.0f;
      for (uint t = 0; t < n; ++t)
        col += m[t * n + s];
      for (uint t = 0; t < n; ++t)
        m[t * n + s] /= col + kEps;
    }
  }
  for (uint i = 0; i < n * n; ++i)
    out[2 * n + i] = m[i];
}

// The whole module stage of a row in one dispatch — mhc_project +
// mhc_mappings + the aggregate half of mhc_weighted_sum — which the
// decode step otherwise pays three launches and two scratch round-trips
// for. One threadgroup per row: the simdgroups split the mixWidth
// columns' dots over the streams, thread 0 runs the gates' Sinkhorn as
// mhc_mappings does, and the row's streams contract by h_pre into the
// bf16 aggregate the layer norm reads.
kernel void mhc_module(device const float *streams [[buffer(0)]],
                       device const float *weights [[buffer(1)]],
                       device const float *alphas [[buffer(2)]],
                       device const float *bias [[buffer(3)]],
                       device float *mappings [[buffer(4)]],
                       device bfloat *aggregated [[buffer(5)]],
                       constant MhcParams &p [[buffer(6)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups];
  threadgroup float logit[kMixWidth];
  threadgroup float h[kMixWidth];
  threadgroup float pre[kStreams];
  if (row >= p.rows)
    return;
  const uint n = p.streams;
  const uint width = n * p.hidden;
  device const float *x = streams + ulong(row) * width;
  const float r = mhc_rms_inverse(x, width, tid, lane, sg, scratch);
  // The mapping projection's dots: simdgroups take columns round-robin,
  // lanes striding the row's flattened streams.
  for (uint c = sg; c < p.mixWidth; c += kSimdgroups) {
    device const float *w = weights + ulong(c) * width;
    float part = 0.0f;
    for (uint k = lane; k < width; k += 32)
      part += x[k] * w[k];
    part = simd_sum(part);
    if (lane == 0)
      logit[c] = part;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  // mhc_mappings' gates, over the in-group logits.
  device float *out = mappings + ulong(row) * p.mixWidth;
  for (uint c = tid; c < p.mixWidth; c += kRowThreads) {
    const float alpha = alphas[c < n ? 0 : c < 2 * n ? 1 : 2];
    const float v = r * logit[c] * alpha + bias[c];
    h[c] = v;
    if (c < n) {
      const float gate = richengine_sigmoid(v) + kEps;
      out[c] = gate;
      pre[c] = gate;
    } else if (c < 2 * n) {
      out[c] = 2.0f * richengine_sigmoid(v);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  if (tid == 0) {
    float m[kResMix];
    for (uint t = 0; t < n; ++t)
      for (uint s = 0; s < n; ++s)
        m[t * n + s] = h[2 * n + t * n + s];
    for (uint t = 0; t < n; ++t) {
      float hi = m[t * n];
      for (uint s = 1; s < n; ++s)
        hi = max(hi, m[t * n + s]);
      float sum = 0.0f;
      for (uint s = 0; s < n; ++s) {
        m[t * n + s] = fast::exp2(1.44269504089f * (m[t * n + s] - hi));
        sum += m[t * n + s];
      }
      for (uint s = 0; s < n; ++s)
        m[t * n + s] /= sum;
    }
    // The reference's exact order: M + eps then a column normalization,
    // and (sinkhornIterations - 1) row/column pairs after it.
    for (uint i = 0; i < n * n; ++i)
      m[i] += kEps;
    for (uint s = 0; s < n; ++s) {
      float col = 0.0f;
      for (uint t = 0; t < n; ++t)
        col += m[t * n + s];
      for (uint t = 0; t < n; ++t)
        m[t * n + s] /= col + kEps;
    }
    for (uint it = 1; it < p.sinkhornIterations; ++it) {
      for (uint t = 0; t < n; ++t) {
        float rowsum = 0.0f;
        for (uint s = 0; s < n; ++s)
          rowsum += m[t * n + s];
        for (uint s = 0; s < n; ++s)
          m[t * n + s] /= rowsum + kEps;
      }
      for (uint s = 0; s < n; ++s) {
        float col = 0.0f;
        for (uint t = 0; t < n; ++t)
          col += m[t * n + s];
        for (uint t = 0; t < n; ++t)
          m[t * n + s] /= col + kEps;
      }
    }
    for (uint i = 0; i < n * n; ++i)
      out[2 * n + i] = m[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  // The aggregate: the streams' contraction by h_pre into the bf16 rows
  // the layer norm reads.
  device bfloat *dst = aggregated + ulong(row) * p.hidden;
  for (uint d = tid; d < p.hidden; d += kRowThreads) {
    float sum = 0.0f;
    for (uint s = 0; s < n; ++s)
      sum += pre[s] * x[s * p.hidden + d];
    dst[d] = bfloat(sum);
  }
}

// A weighted contraction of the streams into bf16: out[row][d] =
// sum_s weights[row][s] * streams[row][s][d]. Serves the aggregate
// (weights the mappings row's h_pre slice) and the head's reduce; the
// weights' row stride and slice offset are the parameters'.
kernel void mhc_weighted_sum(device const float *streams [[buffer(0)]],
                             device const float *weights [[buffer(1)]],
                             device bfloat *out [[buffer(2)]],
                             constant MhcSumParams &p [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden)
    return;
  device const float *x =
      streams + ulong(row) * p.streams * p.hidden + d;
  device const float *w =
      weights + ulong(row) * p.weightsStride + p.weightsOffset;
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s)
    sum += w[s] * x[s * p.hidden];
  out[ulong(row) * p.hidden + d] = bfloat(sum);
}

// The residual update, in place: stream s becomes the residual matrix's
// column-s contraction of the old streams — sum_t h_res[t][s] * old[t][d]
// — plus h_post[s] * branch[d]. A thread reads every source value it
// rewrites, so no staging is needed.
kernel void mhc_expand_mix(device float *streams [[buffer(0)]],
                           device const bfloat *branch [[buffer(1)]],
                           device const float *mappings [[buffer(2)]],
                           constant MhcParams &p [[buffer(3)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden)
    return;
  const uint n = p.streams;
  device float *x = streams + ulong(row) * n * p.hidden + d;
  device const float *map = mappings + ulong(row) * p.mixWidth;
  const float b = float(branch[ulong(row) * p.hidden + d]);
  float old[kStreams];
  for (uint t = 0; t < n; ++t)
    old[t] = x[t * p.hidden];
  for (uint s = 0; s < n; ++s) {
    float v = map[n + s] * b;
    for (uint t = 0; t < n; ++t)
      v += map[2 * n + t * n + s] * old[t];
    x[s * p.hidden] = v;
  }
}

// The head's contraction weights: weights[row][s] = sigmoid(proj[row][s]
// * r * scale + base[s]) + eps, r the streams' rms-inverse as in
// mhc_mappings.
kernel void mhc_head_weights(device const float *streams [[buffer(0)]],
                             device const float *proj [[buffer(1)]],
                             device const float *base [[buffer(2)]],
                             device const float *scale [[buffer(3)]],
                             device float *weights [[buffer(4)]],
                             constant MhcParams &p [[buffer(5)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups];
  if (row >= p.rows)
    return;
  const float r =
      mhc_rms_inverse(streams + ulong(row) * p.streams * p.hidden,
                      p.streams * p.hidden, tid, lane, sg, scratch);
  if (tid < p.streams)
    weights[row * p.streams + tid] =
        richengine_sigmoid(proj[row * p.streams + tid] * r * scale[0] +
                           base[tid]) +
        kEps;
}

// The fp32 mapping projection no GGUF float kernel can read: one
// threadgroup per (column, row) dots the row's streams with the column's
// fp32 weights. The mHC mapping projection and the HC head's contraction
// projection share it.
kernel void mhc_project(device const float *streams [[buffer(0)]],
                        device const float *weights [[buffer(1)]],
                        device float *logits [[buffer(2)]],
                        constant MhcProjectParams &p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups];
  const uint column = group.x, row = group.y;
  if (row >= p.rows || column >= p.columns)
    return;
  device const float *x = streams + ulong(row) * p.width;
  device const float *w = weights + ulong(column) * p.width;
  float part = 0.0f;
  for (uint k = tid; k < p.width; k += kRowThreads)
    part += x[k] * w[k];
  part = simd_sum(part);
  if (lane == 0) scratch[sg] = part;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  if (sg == 0 && lane == 0) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[g];
    logits[ulong(row) * p.columns + column] = total;
  }
}

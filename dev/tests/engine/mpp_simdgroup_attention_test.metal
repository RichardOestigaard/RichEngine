// Probe for a per-simdgroup MPP TensorOps attention tile. One threadgroup
// (8 simdgroups) owns one tile of M=48 fused rows (8 query rows x 6 query
// heads of one KV head), D=256, over pages of N=32 int8 tokens. Variant A
// keeps the score cooperative tensor in registers across QK^T, the online
// softmax (reduce_rows) and P x V via a left-input cooperative tensor.
// Variant B replicates the production structure: one execution_simdgroups<8>
// matmul per page staging scores and bf16 probabilities in threadgroup
// memory.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint ProbeM = 48;   // fused rows: 8 rows x 6 query heads
constant uint ProbeMG = 16;  // rows per simdgroup (3 of 8 groups active)
constant uint ProbeGroups = ProbeM / ProbeMG;
constant uint ProbeD = 256;  // head dimension
constant uint ProbeN = 32;   // tokens per page
constant uint ProbeQH = 6;   // query heads per KV head

// flags[0]: is_compatible_as_left_input(QK dest CT -> PV left)
// flags[1]: is_iterator_compatible(scores CT, row-reduction CT)
// flags[2]: scores CT capacity per thread
// flags[3]: running CT capacity per thread
// flags[4]: row-reduction CT capacity per thread

kernel void mpp_attention_simdgroup(
    device bfloat *queries [[buffer(0)]],
    device int8_t *keys [[buffer(1)]],
    device int8_t *values [[buffer(2)]],
    device float *key_scales [[buffer(3)]],
    device float *value_scales [[buffer(4)]],
    device float *output [[buffer(5)]],
    device float *statistics [[buffer(6)]],
    device uint *flags [[buffer(7)]],
    constant uint &pages [[buffer(8)]],
    constant uint &committed [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint M = ProbeM, MG = ProbeMG, D = ProbeD, N = ProbeN;
  device bfloat *q = queries + ulong(group) * M * D;
  device int8_t *k = keys + ulong(group) * pages * N * D;
  device int8_t *v = values + ulong(group) * pages * D * N;
  device float *ks = key_scales + ulong(group) * pages * N;
  device float *vs = value_scales + ulong(group) * pages * N;
  device float *out = output + ulong(group) * M * D;
  device float *st = statistics + ulong(group) * M * 2;

  const uint visible = committed + 8;

  threadgroup float row_max[8 * ProbeMG];  // indexed [simdgroup][row]
  threadgroup float row_sum[8 * ProbeMG];
  threadgroup float row_scale[8 * ProbeMG];
  threadgroup float row_next[8 * ProbeMG];
  if (thread_index < 8 * MG) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(q, dextents<int, 2>{int(D), int(M)}, array<int, 2>{1, int(D)});
  auto ot = tensor(out, dextents<int, 2>{int(D), int(M)}, array<int, 2>{1, int(D)});

  constexpr auto qk_d = matmul2d_descriptor(ProbeMG, ProbeN, ProbeD, false,
                                            true, false,
                                            matmul2d_descriptor::mode::multiply);
  constexpr auto pv_d = matmul2d_descriptor(ProbeMG, ProbeD, ProbeN, false,
                                            true, true,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_d, execution_simdgroup> qk;
  matmul2d<pv_d, execution_simdgroup> pv;

  const bool active = simdgroup < ProbeGroups;
  auto q0 = qt.slice<ProbeD, ProbeMG>(0, simdgroup * ProbeMG);
  auto scores = qk.get_destination_cooperative_tensor<
      decltype(q0), decltype(tensor(static_cast<device int8_t *>(nullptr),
                                    dextents<int, 2>{int(ProbeD), int(ProbeN)},
                                    array<int, 2>{1, int(ProbeD)})
                                 .template slice<ProbeD, ProbeN>(0, 0)),
      float>();
  auto page_max = qk.get_row_reduction_destination_cooperative_tensor<
      decltype(q0), decltype(tensor(static_cast<device int8_t *>(nullptr),
                                    dextents<int, 2>{int(ProbeD), int(ProbeN)},
                                    array<int, 2>{1, int(ProbeD)})
                                 .template slice<ProbeD, ProbeN>(0, 0)),
      float>();
  auto page_sum = qk.get_row_reduction_destination_cooperative_tensor<
      decltype(q0), decltype(tensor(static_cast<device int8_t *>(nullptr),
                                    dextents<int, 2>{int(ProbeD), int(ProbeN)},
                                    array<int, 2>{1, int(ProbeD)})
                                 .template slice<ProbeD, ProbeN>(0, 0)),
      float>();

  const bool left_ok =
      pv.template is_compatible_as_left_input<float, int8_t, float>(scores);
  const bool iter_ok = is_iterator_compatible(scores, page_max);
  auto pl0 = pv.template get_left_input_cooperative_tensor<float, int8_t,
                                                           float>(scores);
  auto vt_proto = tensor(static_cast<device int8_t *>(nullptr),
                         dextents<int, 2>{int(ProbeN), int(ProbeD)},
                         array<int, 2>{1, int(ProbeN)});
  auto vt_slice_proto = vt_proto.slice<ProbeN, ProbeD>(0, 0);
  auto running = pv.get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(pl0)>,
      remove_addrspace_t<decltype(vt_slice_proto)>, float>();

  if (group == 0 && thread_index == 0) {
    flags[0] = left_ok;
    flags[1] = iter_ok;
    flags[2] = scores.get_capacity();
    flags[3] = running.get_capacity();
    flags[4] = page_max.get_capacity();
  }

  const bool running_full =
      uint(running.get_capacity()) * 32u == MG * D;
#pragma unroll
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (running_full || running.is_valid_element(i))
      running[i] = 0.0f;

  // The CT coordinates are fixed for the kernel's lifetime; the impl call
  // costs more than a cached lookup once the page loop runs. Per element:
  // token column, local row, and the row's causal limit (row-invariant).
  const ushort score_cap = scores.get_capacity();
  uchar score_col[32];
  uchar score_row[32];
  uint score_limit[32];
  bool score_valid[32];
  const ushort red_cap = page_max.get_capacity();
  uchar red_row[8];
#pragma unroll
  for (ushort i = 0; i < score_cap && i < 32; ++i) {
    score_valid[i] = scores.is_valid_element(i);
    if (!score_valid[i]) continue;
    const auto c = scores.get_multidimensional_index(i);
    score_col[i] = uchar(c[0]);
    score_row[i] = uchar(c[1]);
    const uint fused_row = simdgroup * MG + uint(c[1]);
    score_limit[i] = min(visible, committed + min(fused_row / ProbeQH, 7u) + 1);
  }
#pragma unroll
  for (ushort i = 0; i < red_cap && i < 8; ++i)
    red_row[i] = page_max.is_valid_element(i)
                     ? uchar(page_max.get_multidimensional_index(i)[0])
                     : uchar(0);

  for (uint page = 0; active && page < pages; ++page) {
    device int8_t *kp = k + ulong(page) * N * D;
    device int8_t *vp = v + ulong(page) * D * N;
    device float *ksp = ks + page * N;
    device float *vsp = vs + page * N;
    const uint token_start = page * N;

    auto kt = tensor(kp, dextents<int, 2>{int(D), int(N)}, array<int, 2>{1, int(D)});
    auto vt = tensor(vp, dextents<int, 2>{int(N), int(D)}, array<int, 2>{1, int(N)});

    auto ks_slice = kt.slice<ProbeD, ProbeN>(0, 0);
    qk.run(q0, ks_slice, scores);

    // Key scales, 1/sqrt(D) and the causal mask fold into the scores CT.
#pragma unroll
    for (ushort i = 0; i < score_cap; ++i) {
      if (!score_valid[i])
        continue;
      const uint token = token_start + score_col[i];
      float s = scores[i] * ksp[score_col[i]] * 0.0625f;
      scores[i] = token < score_limit[i] ? s : -INFINITY;
    }

    reduce_rows(scores, page_max, reduction_operation::max, -INFINITY);

    bool any_scale = false;
#pragma unroll
    for (ushort i = 0; i < page_max.get_capacity(); ++i) {
      if (!page_max.is_valid_element(i))
        continue;
      const uint r = red_row[i];
      const float prev = row_max[simdgroup * MG + r];
      const float next = max(prev, page_max[i]);
      const float scale =
          (next == -INFINITY || next == prev) ? 1.0f : fast::exp(prev - next);
      row_scale[simdgroup * MG + r] = scale;
      row_next[simdgroup * MG + r] = next;
      any_scale |= scale != 1.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_any(any_scale)) {
#pragma unroll
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running_full && !running.is_valid_element(i))
          continue;
        const auto c = running.get_multidimensional_index(i);
        running[i] *= row_scale[simdgroup * MG + uint(c[1])];
      }
    }

    // Unscaled probabilities first: the row statistic is the plain exp sum,
    // as the split reduce expects; masked entries produce exact zeros.
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i) {
      if (!scores.is_valid_element(i))
        continue;
      const float s = scores[i];
      scores[i] =
          s == -INFINITY
              ? 0.0f
              : fast::exp(s - row_next[simdgroup * MG + score_row[i]]);
    }
    reduce_rows(scores, page_sum, reduction_operation::sum, 0.0f);
    // The value scales fold into the PV operand, not the statistic.
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i) {
      if (!scores.is_valid_element(i))
        continue;
      scores[i] *= vsp[score_col[i]];
    }
#pragma unroll
    for (ushort i = 0; i < page_sum.get_capacity(); ++i) {
      if (!page_sum.is_valid_element(i))
        continue;
      const uint r = red_row[i];
      row_sum[simdgroup * MG + r] =
          row_sum[simdgroup * MG + r] * row_scale[simdgroup * MG + r] +
          page_sum[i];
      row_max[simdgroup * MG + r] = row_next[simdgroup * MG + r];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (left_ok) {
      auto pl = pv.template get_left_input_cooperative_tensor<float, int8_t,
                                                              float>(scores);
      auto v_slice = vt.slice<ProbeN, ProbeD>(0, 0);
      pv.run(pl, v_slice, running);
    }
  }

#pragma unroll
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!running_full && !running.is_valid_element(i))
      continue;
    const auto c = running.get_multidimensional_index(i);
    const float denom = row_sum[simdgroup * MG + uint(c[1])];
    running[i] = denom > 0.0f ? running[i] / denom : 0.0f;
  }
  if (active)
    running.store(ot.slice<ProbeD, ProbeMG>(0, simdgroup * ProbeMG));
  if (active && lane < MG) {
    st[(simdgroup * MG + lane) * 2] = row_max[simdgroup * MG + lane];
    st[(simdgroup * MG + lane) * 2 + 1] = row_sum[simdgroup * MG + lane];
  }
}

// Production-structure baseline: one execution_simdgroups<8> matmul per page,
// scores and bf16 probabilities staged in threadgroup memory, the four-lane
// fused-row softmax of paged_attention_tile.h.
kernel void mpp_attention_threadgroup(
    device bfloat *queries [[buffer(0)]],
    device int8_t *keys [[buffer(1)]],
    device int8_t *values [[buffer(2)]],
    device float *key_scales [[buffer(3)]],
    device float *value_scales [[buffer(4)]],
    device float *output [[buffer(5)]],
    device float *statistics [[buffer(6)]],
    device uint *flags [[buffer(7)]],
    constant uint &pages [[buffer(8)]],
    constant uint &committed [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr uint M = ProbeM, D = ProbeD, N = ProbeN;
  device bfloat *q = queries + ulong(group) * M * D;
  device int8_t *k = keys + ulong(group) * pages * N * D;
  device int8_t *v = values + ulong(group) * pages * D * N;
  device float *ks = key_scales + ulong(group) * pages * N;
  device float *vs = value_scales + ulong(group) * pages * N;
  device float *out = output + ulong(group) * M * D;
  device float *st = statistics + ulong(group) * M * 2;

  const uint visible = committed + 8;

  alignas(16) threadgroup float scores[M * ProbeN];
  alignas(16) threadgroup bfloat probabilities[M * ProbeN];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float previous_scale[M];
  threadgroup atomic_uint rescale;

  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(q, dextents<int, 2>{int(D), int(M)}, array<int, 2>{1, int(D)});
  auto ot = tensor(out, dextents<int, 2>{int(D), int(M)}, array<int, 2>{1, int(D)});
  auto sct = tensor(scores, dextents<int, 2>{int(N), int(M)}, array<int, 2>{1, int(N)});
  auto pt = tensor(probabilities, dextents<int, 2>{int(N), int(M)},
                   array<int, 2>{1, int(N)});

  constexpr auto qk_d = matmul2d_descriptor(ProbeM, ProbeN, ProbeD, false, true,
                                            false,
                                            matmul2d_descriptor::mode::multiply);
  constexpr auto pv_d = matmul2d_descriptor(ProbeM, ProbeD, ProbeN, false, true,
                                            true,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_d, execution_simdgroups<8>> qk;
  matmul2d<pv_d, execution_simdgroups<8>> pv;
  auto q0 = qt.slice<ProbeD, ProbeM>(0, 0);
  auto p0 = pt.slice<ProbeN, ProbeM>(0, 0);
  auto vt_proto = tensor(static_cast<device int8_t *>(nullptr),
                         dextents<int, 2>{int(ProbeN), int(ProbeD)},
                         array<int, 2>{1, int(ProbeN)});
  auto v_slice_proto = vt_proto.slice<ProbeN, ProbeD>(0, 0);
  auto running = pv.get_destination_cooperative_tensor<
      decltype(p0), decltype(v_slice_proto), float>();
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == M * D;
#pragma unroll
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (running_full || running.is_valid_element(i))
      running[i] = 0.0f;
  if (group == 0 && thread_index == 0)
    flags[8] = running.get_capacity();

  for (uint page = 0; page < pages; ++page) {
    device int8_t *kp = k + ulong(page) * N * D;
    device int8_t *vp = v + ulong(page) * D * N;
    device float *ksp = ks + page * N;
    device float *vsp = vs + page * N;
    const uint token_start = page * N;

    auto kt = tensor(kp, dextents<int, 2>{int(D), int(N)}, array<int, 2>{1, int(D)});
    auto vt = tensor(vp, dextents<int, 2>{int(N), int(D)}, array<int, 2>{1, int(N)});

    auto page_scores = qk.get_destination_cooperative_tensor<
        decltype(q0), decltype(kt.slice<ProbeD, ProbeN>(0, 0)), float>();
    auto ks_slice = kt.slice<ProbeD, ProbeN>(0, 0);
    qk.run(q0, ks_slice, page_scores);
    page_scores.store(sct.slice<ProbeN, ProbeM>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Four lanes own one fused row, eight tokens each.
    constexpr uint FusedRows = ProbeM;
    constexpr uint TokensPerLane = 8;
    constexpr uint LanesPerRow = ProbeN / TokensPerLane;
    if (thread_index < LanesPerRow * FusedRows) {
      const uint fused_row = thread_index / LanesPerRow;
      const uint column = thread_index % LanesPerRow * TokensPerLane;
      const uint query_row = fused_row / ProbeQH;
      const uint limit = min(visible, committed + min(query_row, 7u) + 1);
      const uint token = token_start + column;
      threadgroup const float4 *scores4 =
          reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N +
                                                       column);
      const uint vector = column / 4;
      float4 low = scores4[0] * reinterpret_cast<device float4 *>(ksp)[vector];
      float4 high =
          scores4[1] * reinterpret_cast<device float4 *>(ksp)[vector + 1];
      low *= 0.0625f;
      high *= 0.0625f;
      float score[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
      float local_max = -INFINITY;
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        score[j] = token + j < limit ? score[j] : -INFINITY;
        local_max = max(local_max, score[j]);
      }
      local_max = max(local_max, simd_shuffle_xor(local_max, 1));
      local_max = max(local_max, simd_shuffle_xor(local_max, 2));
      const float prev = row_max[fused_row];
      const float next = max(prev, local_max);
      float probability[8];
      float local_sum = 0.0f;
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        probability[j] = token + j < limit ? fast::exp(score[j] - next) : 0.0f;
        local_sum += probability[j];
      }
      local_sum += simd_shuffle_xor(local_sum, 1);
      local_sum += simd_shuffle_xor(local_sum, 2);
      if (column == 0) {
        const float scale = next == -INFINITY || next == prev
                                ? 1.0f
                                : fast::exp(prev - next);
        previous_scale[fused_row] = scale;
        row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
        row_max[fused_row] = next;
        if (scale != 1.0f)
          atomic_store_explicit(&rescale, 1u, memory_order_relaxed);
      }
      float4 low_scales =
          reinterpret_cast<device float4 *>(vsp)[vector];
      float4 high_scales =
          reinterpret_cast<device float4 *>(vsp)[vector + 1];
      float4 plo(token + 0 < limit ? probability[0] * low_scales.x : 0.0f,
                 token + 1 < limit ? probability[1] * low_scales.y : 0.0f,
                 token + 2 < limit ? probability[2] * low_scales.z : 0.0f,
                 token + 3 < limit ? probability[3] * low_scales.w : 0.0f);
      float4 phi(token + 4 < limit ? probability[4] * high_scales.x : 0.0f,
                 token + 5 < limit ? probability[5] * high_scales.y : 0.0f,
                 token + 6 < limit ? probability[6] * high_scales.z : 0.0f,
                 token + 7 < limit ? probability[7] * high_scales.w : 0.0f);
      threadgroup bfloat4 *p4 = reinterpret_cast<threadgroup bfloat4 *>(
          probabilities + fused_row * N + column);
      p4[0] = bfloat4(plo);
      p4[1] = bfloat4(phi);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!running_full && !running.is_valid_element(i))
          continue;
        const auto c = running.get_multidimensional_index(i);
        running[i] *= previous_scale[c[1]];
      }
    }
    auto v_slice = vt.slice<ProbeN, ProbeD>(0, 0);
    pv.run(p0, v_slice, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

#pragma unroll
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!running_full && !running.is_valid_element(i))
      continue;
    const auto c = running.get_multidimensional_index(i);
    const float denom = row_sum[c[1]];
    running[i] = denom > 0.0f ? running[i] / denom : 0.0f;
  }
  running.store(ot.slice<ProbeD, ProbeM>(0, 0));
  if (thread_index < M) {
    st[thread_index * 2] = row_max[thread_index];
    st[thread_index * 2 + 1] = row_sum[thread_index];
  }
}

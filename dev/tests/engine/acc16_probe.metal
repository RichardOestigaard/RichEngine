// Phase-1 probe for the attention accumulator-precision A/B: does MPP
// matmul2d accept a bfloat cooperative-tensor destination for the PV product
// (bfloat probabilities x int8_t / int4b / bfloat values), and does
// multiply_accumulate round per page or hold a wider internal sum?
// Each kernel accumulates the same page loop into a bfloat CT and a float
// CT and stores both; the host compares them bitwise.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint ProbeM = 32;    // fused rows (multiple of 8)
constant uint ProbeD = 256;   // head dimension
constant uint ProbeN = 32;    // tokens per page
constant uint ProbePages = 8;
constant uint ProbeThreads = 256;

// Deterministic per-(page, token, row) probability in [0, 1).
inline float probe_prob(uint page, uint token, uint row) {
  uint h = page * 0x9e3779b1u + token * 0x85ebca6bu + row * 0xc2b2ae35u;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  return float(h & 0xffffffu) / float(0x1000000u);
}

template <typename Running>
inline void probe_store(device float *out, Running running) {
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (!running.is_valid_element(index))
      continue;
    const auto c = running.get_multidimensional_index(index);
    out[uint(c[1]) * ProbeD + uint(c[0])] = float(running[index]);
  }
}

// Values arrive in the production tile's operand layouts:
//   int8/bfloat: tensor {N, D} strides {1, N}, transpose_right = true
//   int4b:       tensor<device int4b_format> {D, N} strides {1, D}
template <typename CacheElement, bool Packed>
inline void probe_pv_body(device CacheElement *values, device float *out_bf,
                          device float *out_fp, uint thread_index,
                          threadgroup bfloat *probabilities) {
  constexpr ushort M = ProbeM, N = ProbeN, D = ProbeD;
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.template slice<N, M>(0, 0);

  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, !Packed, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  typedef decltype(p0) PSliced;

  // Right-operand prototype slices, exactly as the production tile builds
  // them (nullptr tensors fix the type; per-page tensors share it).
  auto running_bf = [&] {
    if constexpr (Packed) {
      typedef tensor<device int4b_format, dextents<int, 2>, tensor_inline>
          PackedTensor;
      return pv.template get_destination_cooperative_tensor<
          PSliced,
          decltype(PackedTensor(static_cast<device uchar *>(nullptr),
                                dextents<int, 2>{D, N}, array<int, 2>{1, D})
                       .template slice<D, N>(0, 0)),
          bfloat>();
    } else {
      return pv.template get_destination_cooperative_tensor<
          PSliced,
          decltype(tensor(static_cast<device CacheElement *>(nullptr),
                          dextents<int, 2>{N, D}, array<int, 2>{1, N})
                       .template slice<N, D>(0, 0)),
          bfloat>();
    }
  }();
  auto running_fp = [&] {
    if constexpr (Packed) {
      typedef tensor<device int4b_format, dextents<int, 2>, tensor_inline>
          PackedTensor;
      return pv.template get_destination_cooperative_tensor<
          PSliced,
          decltype(PackedTensor(static_cast<device uchar *>(nullptr),
                                dextents<int, 2>{D, N}, array<int, 2>{1, D})
                       .template slice<D, N>(0, 0)),
          float>();
    } else {
      return pv.template get_destination_cooperative_tensor<
          PSliced,
          decltype(tensor(static_cast<device CacheElement *>(nullptr),
                          dextents<int, 2>{N, D}, array<int, 2>{1, N})
                       .template slice<N, D>(0, 0)),
          float>();
    }
  }();

  const bool full_bf =
      uint(running_bf.get_capacity()) * (8u * 32u) == uint(M) * D;
  const bool full_fp =
      uint(running_fp.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort i = 0; i < running_bf.get_capacity(); ++i)
    if (full_bf || running_bf.is_valid_element(i))
      running_bf[i] = bfloat(0.0f);
#pragma unroll
  for (ushort i = 0; i < running_fp.get_capacity(); ++i)
    if (full_fp || running_fp.is_valid_element(i))
      running_fp[i] = 0.0f;

  for (uint page = 0; page < ProbePages; ++page) {
    // This page's value slab and probabilities.
    if constexpr (Packed) {
      typedef tensor<device int4b_format, dextents<int, 2>, tensor_inline>
          PackedTensor;
      device uchar *page_bytes =
          reinterpret_cast<device uchar *>(values) + ulong(page) * (D * N / 2);
      PackedTensor vt_page(page_bytes, dextents<int, 2>{D, N},
                           array<int, 2>{1, D});
      auto vs_page = vt_page.template slice<D, N>(0, 0);
      for (uint i = thread_index; i < N * M; i += ProbeThreads) {
        const uint row = i / N, token = i % N;
        probabilities[i] = bfloat(probe_prob(page, token, row));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
      const float s = page == 0 ? 1.0f : 0.9999f;
#pragma unroll
      for (ushort i = 0; i < running_bf.get_capacity(); ++i) {
        if (!full_bf && !running_bf.is_valid_element(i)) continue;
        running_bf[i] = bfloat(float(running_bf[i]) * s);
      }
#pragma unroll
      for (ushort i = 0; i < running_fp.get_capacity(); ++i) {
        if (!full_fp && !running_fp.is_valid_element(i)) continue;
        running_fp[i] *= s;
      }
      pv.run(p0, vs_page, running_bf);
      pv.run(p0, vs_page, running_fp);
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    } else {
      device CacheElement *page_values = values + ulong(page) * (D * N);
      auto vt_page = tensor(page_values, dextents<int, 2>{N, D},
                            array<int, 2>{1, N});
      auto vs_page = vt_page.template slice<N, D>(0, 0);
      for (uint i = thread_index; i < N * M; i += ProbeThreads) {
        const uint row = i / N, token = i % N;
        probabilities[i] = bfloat(probe_prob(page, token, row));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
      const float s = page == 0 ? 1.0f : 0.9999f;
#pragma unroll
      for (ushort i = 0; i < running_bf.get_capacity(); ++i) {
        if (!full_bf && !running_bf.is_valid_element(i)) continue;
        running_bf[i] = bfloat(float(running_bf[i]) * s);
      }
#pragma unroll
      for (ushort i = 0; i < running_fp.get_capacity(); ++i) {
        if (!full_fp && !running_fp.is_valid_element(i)) continue;
        running_fp[i] *= s;
      }
      pv.run(p0, vs_page, running_bf);
      pv.run(p0, vs_page, running_fp);
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    }
  }
  probe_store(out_bf, running_bf);
  probe_store(out_fp, running_fp);
}

// int8 values, dim-major slab [D][N] per page (production stride {1,N}).
kernel void acc16_probe_i8(device int8_t *values [[buffer(0)]],
                           device float *out_bf [[buffer(1)]],
                           device float *out_fp [[buffer(2)]],
                           uint thread_index [[thread_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat probabilities[ProbeN * ProbeM];
  probe_pv_body<int8_t, false>(values, out_bf, out_fp, thread_index,
                               probabilities);
}

// bf16 values, same layout.
kernel void acc16_probe_bf16(device bfloat *values [[buffer(0)]],
                             device float *out_bf [[buffer(1)]],
                             device float *out_fp [[buffer(2)]],
                             uint thread_index [[thread_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat probabilities[ProbeN * ProbeM];
  probe_pv_body<bfloat, false>(values, out_bf, out_fp, thread_index,
                               probabilities);
}

// Packed int4 values: device bytes, token-major pairs.
kernel void acc16_probe_i4(device uchar *values [[buffer(0)]],
                           device float *out_bf [[buffer(1)]],
                           device float *out_fp [[buffer(2)]],
                           uint thread_index [[thread_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat probabilities[ProbeN * ProbeM];
  probe_pv_body<uchar, true>(values, out_bf, out_fp, thread_index,
                             probabilities);
}

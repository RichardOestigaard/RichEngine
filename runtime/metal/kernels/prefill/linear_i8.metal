#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// RICHENGINE_PREFILL_FAST_INT8 path: activations quantize per 64-input group to
// a two-term asymmetric uint8 split (x ~= s*q + lo), the projections run the
// integer Neural-Accelerator matmul uint8 x uint4 -> int32, and the epilogue
// reproduces the affine Q4 scale algebra:
//   y = s_w * (s*I + lo*C) + b_w * Jx
// where I = sum(q*c) is the integer accumulation, C the weight column's code
// sum (the group's 64 nibbles, computed per tile in threadgroup memory), b_w
// the group bias and Jx = s*sum(q) + 64*lo the group's dequantized input
// sum. The result is not bit-identical to the bf16 path.

constant constexpr ushort kI8Group = 64;

// Per (row, quant group) parameters of one split term.
struct I8GroupParams {
  float s1;
  float s2;
  float offSum;
  float jxSum;
};

// Quantize a 32-row tile's 64-input groups into two split terms: hi covers
// [lo, hi] of the group's inputs, lo covers the hi term's signed residual
// (offset -rmax, scale rmax/255).
kernel void prefill_linear_i8_quant(
    device const bfloat *input [[buffer(0)]],
    device uchar *codes_hi [[buffer(1)]], device uchar *codes_lo [[buffer(2)]],
    device I8GroupParams *params_hi [[buffer(3)]],
    constant uint &input_size [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort TileM = 32;
  const uint groups = input_size / kI8Group;
  for (uint task = simd_group; task < TileM; task += 8) {
    const uint row = tile * TileM + task;
    for (uint g = simd_lane; g < groups; g += 32) {
      device const bfloat *src = input + ulong(row) * input_size + g * kI8Group;
      float v[kI8Group];
      float lo = INFINITY, hi = -INFINITY;
#pragma unroll
      for (ushort i = 0; i < kI8Group; ++i) {
        v[i] = float(src[i]);
        lo = min(lo, v[i]);
        hi = max(hi, v[i]);
      }
      const float s1 = max((hi - lo) / 255.0f, 1e-8f);
      const float rcp1 = 1.0f / s1;
      float j1 = 0.0f, rmax = 0.0f;
      float r[kI8Group];
#pragma unroll
      for (ushort i = 0; i < kI8Group; ++i) {
        const float q = clamp(round(v[i] * rcp1 - lo * rcp1), 0.0f, 255.0f);
        codes_hi[row * input_size + g * kI8Group + i] = uchar(q);
        j1 += q;
        r[i] = v[i] - (q * s1 + lo);
        rmax = max(rmax, abs(r[i]));
      }
      // Residual term: r in [-rmax, rmax] quantizes to [0,255] with offset
      // -rmax; Jx = s2*sum(q2) - 64*rmax = sum(r) estimate. One packed
      // record: (s1, s2, lo + -rmax, Jx1 + Jx2).
      const float s2 = max(rmax / 255.0f, 1e-12f);
      const float rcp2 = 1.0f / s2;
      float j2 = 0.0f;
#pragma unroll
      for (ushort i = 0; i < kI8Group; ++i) {
        const float q = clamp(round((r[i] + rmax) * rcp2), 0.0f, 255.0f);
        codes_lo[row * input_size + g * kI8Group + i] = uchar(q);
        j2 += q;
      }
      params_hi[row * groups + g] = {s1, s2, lo - rmax,
                                     s1 * j1 + kI8Group * lo + s2 * j2 -
                                         kI8Group * rmax};
    }
  }
}

// The 32 x TileN int8 tile of grid position (row tile, column tile):
// TileM=32 rows of uint8 activations against TileN of the projection's 256-
// column-packed Q4 weight tile, four simdgroups (the measured prefill policy).
template <ushort TileN, bool Residual, bool UpSiluSums>
__attribute__((always_inline)) inline void i8_prefill(
    device uchar *codes_hi, device uchar *codes_lo,
    device const I8GroupParams *params_hi, device uchar *weights,
    device bfloat *scales, device bfloat *biases, device bfloat *auxiliary,
    device bfloat *output, device float *output_sums, uint output_size,
    uint input_size, uint2 group, uint simd_lane, uint simd_group,
    threadgroup float *tg_c) {
  constexpr ushort TileM = 32;
  constexpr ushort Simdgroups = 8;
  const uint quant_groups = input_size / 64;
  const ulong row0 = group.x * TileM;
  codes_hi += row0 * input_size;
  codes_lo += row0 * input_size;
  params_hi += row0 * quant_groups;
  auto a_hi = tensor(codes_hi, dextents<int, 2>{int(input_size), TileM},
                     array<int, 2>{1, int(input_size)});
  auto a_lo = tensor(codes_lo, dextents<int, 2>{int(input_size), TileM},
                     array<int, 2>{1, int(input_size)});
  auto c = tensor(output + row0 * output_size,
                  dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> op;
  const uint output_origin = group.y * TileN;
  const uint tile = output_origin / kQ4StorageColumns;
  const uint tile_column = output_origin % kQ4StorageColumns;
  device uchar *tile_weights =
      weights +
      (ulong(tile) * quant_groups * kQ4StorageColumns + tile_column) * 64 / 2;
  auto b_proto =
      tensor<device uint4b_format, dextents<int, 2>, tensor_inline>(
          tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = b_proto.template slice<64, TileN>(0, 0);
  auto a0 = a_hi.template slice<64, TileM>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0),
                                                            decltype(b0),
                                                            float>();
  auto part_hi = op.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), int>();
  decltype(part_hi) part_lo;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    acc[i] = 0.0f;

  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    device uchar *group_weights =
        tile_weights + ulong(quant_group) * kQ4StorageColumns * 64 / 2;
    const uint thread_index = simd_group * 32 + simd_lane;
    for (uint col = thread_index; col < TileN; col += Simdgroups * 32) {
      device const uchar *wb = group_weights + col * 32;
      uint sum = 0;
#pragma unroll
      for (ushort byte = 0; byte < 32; ++byte)
        sum += (wb[byte] & 0xFu) + (wb[byte] >> 4);
      tg_c[col] = float(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto b_ten =
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline>(
            group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto bs = b_ten.template slice<64, TileN>(0, 0);
    auto ah = a_hi.template slice<64, TileM>(quant_group * 64, 0);
    op.run(ah, bs, part_hi);
    auto al = a_lo.template slice<64, TileM>(quant_group * 64, 0);
    op.run(al, bs, part_lo);

#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      auto index = acc.get_multidimensional_index(i);
      const uint row = index[1];
      const uint col = index[0];
      const ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
          tile_column + col;
      const I8GroupParams pa = params_hi[row * quant_groups + quant_group];
      const float sw = float(scales[parameter]);
      const float bw = float(biases[parameter]);
      acc[i] += sw * (pa.s1 * float(part_hi[i]) + pa.s2 * float(part_lo[i])) +
                pa.offSum * sw * tg_c[col] + pa.jxSum * bw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto converted = op.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), bfloat>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    float value = float(bfloat(acc[i]));
    auto index = acc.get_multidimensional_index(i);
    const uint out_index = index[1] * output_size + output_origin + index[0];
    if constexpr (UpSiluSums) {
      value = richengine_silu(float(auxiliary[out_index])) * value;
    } else if constexpr (Residual) {
      value += float(auxiliary[out_index]);
    }
    converted[i] = bfloat(value);
  }
  converted.store(c.template slice<TileN, TileM>(output_origin, 0));
  if constexpr (UpSiluSums) {
    constexpr uint QuantGroups = TileN / 64;
    threadgroup_barrier(mem_flags::mem_device);
    for (uint task = simd_group; task < TileM * QuantGroups;
         task += Simdgroups) {
      uint row = task / QuantGroups;
      uint local_group = task % QuantGroups;
      uint origin = row * output_size + output_origin + local_group * 64 +
                    simd_lane;
      float sum =
          simd_sum(float(output[origin]) + float(output[origin + 32]));
      if (simd_lane == 0) {
        uint quant_group = output_origin / 64 + local_group;
        output_sums[row * (output_size / 64) + quant_group] = sum;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Bindings: codes_hi/codes_lo (0-1), params_hi/params_lo (2-3), weights,
// scales, biases (4-6), then per-epilogue auxiliary/output/output_sums.
#define PREFILL_I8(Name, Residual_, UpSilu_)                                 \
  kernel void Name(device uchar *codes_hi [[buffer(0)]],                     \
                   device uchar *codes_lo [[buffer(1)]],                     \
                   device const I8GroupParams *params_hi [[buffer(2)]],      \
                   device uchar *weights [[buffer(3)]],                      \
                   device bfloat *scales [[buffer(4)]],                      \
                   device bfloat *biases [[buffer(5)]],                      \
                   device bfloat *auxiliary [[buffer(6)]],                   \
                   device bfloat *output [[buffer(7)]],                      \
                   device float *output_sums [[buffer(8)]],                  \
                   constant Q4Params &params [[buffer(9)]],                 \
                   uint2 group [[threadgroup_position_in_grid]],             \
                   uint simd_lane [[thread_index_in_simdgroup]],             \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {     \
    threadgroup float tg_c[256];                                             \
    i8_prefill<256, Residual_, UpSilu_>(                                     \
        codes_hi, codes_lo, params_hi, weights, scales, biases,           \
        auxiliary, output, output_sums, params.output_size,                  \
        params.input_size, group, simd_lane, simd_group, tg_c);              \
  }
PREFILL_I8(prefill_linear_i8_n256, false, false)
PREFILL_I8(prefill_linear_i8_n256_residual, true, false)
PREFILL_I8(prefill_linear_i8_n256_up_silu_sums, false, true)

// NAXR arm of the register-dequantized Q4 small-M GEMM A/B (q4_naxr_ab.mm),
// after locals/nax_kernel.py's locals_qmm_naxr: the weights are dequantized
// in registers into the matmul's right cooperative tensor and the fp32
// accumulators stay in their cooperative tensors for the whole K loop, so no
// per-group partial readout or scale/bias epilogue exists at all — scales and
// biases are folded into the weights before the matmul, once per element.
// Adapted to RichEngine's packed AF4P planes: 256-column storage tiles of
// [64-group][column][32 linear-nibble bytes], one bf16 scale and bias per
// (column, group) in the same tile order.
//
// Compiles at metal4.0 only: metal4.1 rejects get_destination_cooperative_tensor
// for cooperative-tensor operands (the Makefile's METAL40_FLAGS note). Bench
// code only — nothing here is wired into production.
//
// Measured (M5 Pro, q4_naxr_ab.mm, palindromic ABBA rounds): the candidate
// loses to the direct-packed tiles by 2.5-6x at every valid shape — the
// per-element dequant fill costs far more than the direct tile's per-group
// scale/bias epilogue, because dequant work scales with weight elements
// (TileN x K per tile) while the epilogue scales with output elements
// (Rows x TileN), and the neural accelerators consume the packed uint4
// operand natively at full rate. bench_naxr_s8_m64 additionally computes
// wrong values (the four-fragment accumulator chain at eight simdgroups;
// not root-caused — the design is discarded on the m16/m32 evidence) and is
// kept only so the harness line documents it.
#include "metal/abi/KernelABI.h"

// Rows x (32 * Simds) tile, one 16-row fragment chain per simdgroup of the
// threadgroup, column strip output_origin + simd_group * 32. The B operand of
// one quant group is 32 columns x 64 inputs of bf16 in the cooperative tensor
// cb, dequantized element by element through the tensor's own coordinate
// map (layout-agnostic; the probe kernel below prints the map for a
// vectorized rewrite).
template <ushort Rows, ushort Simds>
inline void naxr_q4_tile(device bfloat *input, device uchar *weights,
                         device bfloat *scales, device bfloat *biases,
                         device bfloat *output, uint output_size,
                         uint input_size, uint output_origin,
                         uint simd_group) {
  constexpr ushort Frags = Rows / 16;
  constexpr auto descriptor =
      matmul2d_descriptor(16, 32, 64, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> operation;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  const uint quant_groups = input_size / 64;
  const uint tile = output_origin / 256;
  const uint tile_offset = output_origin % 256;
  device uchar *tile_weights =
      weights + ulong(tile) * quant_groups * 256 * 32;
  const uint strip = tile_offset + simd_group * 32;

  using CtB = decltype(operation.template get_right_input_cooperative_tensor<
                       bfloat, bfloat, float>());
  auto probe_a = a.slice<64, 16>(0, 0);
  using Acc = decltype(operation.template get_destination_cooperative_tensor<
                       decltype(probe_a), CtB, float>());
  Acc acc0, acc1, acc2, acc3;
  auto zero = [](thread Acc &acc) {
    const ushort cap = acc.get_capacity();
#pragma unroll
    for (ushort i = 0; i < cap; ++i)
      if (acc.is_valid_element(i)) acc[i] = 0.0f;
  };
  zero(acc0);
  zero(acc1);
  zero(acc2);
  zero(acc3);

  CtB cb;
  const ushort bcap = cb.get_capacity();
  for (uint g = 0; g < quant_groups; ++g) {
    device uchar *group_weights = tile_weights + ulong(g) * 256 * 32;
    device const bfloat *sg =
        scales + (ulong(tile) * quant_groups + g) * 256 + strip;
    device const bfloat *bg =
        biases + (ulong(tile) * quant_groups + g) * 256 + strip;
    int cached_n = -1;
    float fs = 0.0f, fb = 0.0f;
#pragma unroll
    for (ushort i = 0; i < bcap; ++i) {
      if (!cb.is_valid_element(i)) continue;
      auto index = cb.get_multidimensional_index(i);
      const int k = int(index[0]);
      const int n = int(index[1]);
      if (n != cached_n) {
        fs = float(sg[n]);
        fb = float(bg[n]);
        cached_n = n;
      }
      const uint byte = group_weights[(strip + uint(n)) * 32 + uint(k >> 1)];
      const uint code = (k & 1) ? (byte >> 4) : (byte & 0xF);
      cb[i] = bfloat(fma(fs, float(code), fb));
    }
    auto a0 = a.slice<64, 16>(g * 64, 0);
    operation.run(a0, cb, acc0);
    if constexpr (Frags > 1) {
      auto a1 = a.slice<64, 16>(g * 64, 16);
      operation.run(a1, cb, acc1);
    }
    if constexpr (Frags > 2) {
      auto a2 = a.slice<64, 16>(g * 64, 32);
      operation.run(a2, cb, acc2);
    }
    if constexpr (Frags > 3) {
      auto a3 = a.slice<64, 16>(g * 64, 48);
      operation.run(a3, cb, acc3);
    }
  }

  const uint column_base = output_origin + simd_group * 32;
  auto store = [&](thread Acc &acc, uint row0) {
    const ushort cap = acc.get_capacity();
#pragma unroll
    for (ushort i = 0; i < cap; ++i) {
      if (!acc.is_valid_element(i)) continue;
      auto index = acc.get_multidimensional_index(i);
      output[(row0 + uint(index[1])) * output_size + column_base +
             uint(index[0])] = bfloat(acc[i]);
    }
  };
  store(acc0, 0);
  if constexpr (Frags > 1) store(acc1, 16);
  if constexpr (Frags > 2) store(acc2, 32);
  if constexpr (Frags > 3) store(acc3, 48);
}

#define NAXR_Q4_KERNEL(Name, Rows, Simds)                                      \
  [[max_total_threads_per_threadgroup(Simds * 32)]] kernel void Name(          \
                   device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4PersistentParams &params [[buffer(5)]],          \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    const uint tiles = params.output_size / (32 * Simds);                      \
    for (uint tile = group; tile < tiles; tile += params.groups)               \
      naxr_q4_tile<Rows, Simds>(input, weights, scales, biases, output,        \
                                params.output_size, params.input_size,         \
                                tile * 32 * Simds, simd_group);                \
  }

template <ushort Rows, ushort Simds>
inline void naxr_q4_ceiling_tile(device bfloat *input, device bfloat *output,
                                 uint output_size, uint input_size,
                                 uint output_origin, uint simd_group) {
  constexpr ushort Frags = Rows / 16;
  constexpr auto descriptor =
      matmul2d_descriptor(16, 32, 64, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> operation;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows},
                  array<int, 2>{1, int(input_size)});
  const uint quant_groups = input_size / 64;
  using CtB = decltype(operation.template get_right_input_cooperative_tensor<
                       bfloat, bfloat, float>());
  auto probe_a = a.slice<64, 16>(0, 0);
  using Acc = decltype(operation.template get_destination_cooperative_tensor<
                       decltype(probe_a), CtB, float>());
  Acc acc0, acc1, acc2, acc3;
  auto zero = [](thread Acc &acc) {
    const ushort cap = acc.get_capacity();
#pragma unroll
    for (ushort i = 0; i < cap; ++i)
      if (acc.is_valid_element(i)) acc[i] = 0.0f;
  };
  zero(acc0);
  zero(acc1);
  zero(acc2);
  zero(acc3);
  CtB cb;
  {
    const ushort bcap = cb.get_capacity();
#pragma unroll
    for (ushort i = 0; i < bcap; ++i)
      if (cb.is_valid_element(i)) cb[i] = bfloat(0.5f);
  }
  for (uint g = 0; g < quant_groups; ++g) {
    auto a0 = a.slice<64, 16>(g * 64, 0);
    operation.run(a0, cb, acc0);
    if constexpr (Frags > 1) {
      auto a1 = a.slice<64, 16>(g * 64, 16);
      operation.run(a1, cb, acc1);
    }
    if constexpr (Frags > 2) {
      auto a2 = a.slice<64, 16>(g * 64, 32);
      operation.run(a2, cb, acc2);
    }
    if constexpr (Frags > 3) {
      auto a3 = a.slice<64, 16>(g * 64, 48);
      operation.run(a3, cb, acc3);
    }
  }
  const uint column_base = output_origin + simd_group * 32;
  auto store = [&](thread Acc &acc, uint row0) {
    const ushort cap = acc.get_capacity();
#pragma unroll
    for (ushort i = 0; i < cap; ++i) {
      if (!acc.is_valid_element(i)) continue;
      auto index = acc.get_multidimensional_index(i);
      output[(row0 + uint(index[1])) * output_size + column_base +
             uint(index[0])] = bfloat(acc[i]);
    }
  };
  store(acc0, 0);
  if constexpr (Frags > 1) store(acc1, 16);
  if constexpr (Frags > 2) store(acc2, 32);
  if constexpr (Frags > 3) store(acc3, 48);
}

#define NAXR_Q4_CEILING_KERNEL(Name, Rows, Simds)                              \
  [[max_total_threads_per_threadgroup(Simds * 32)]] kernel void Name(          \
                   device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4PersistentParams &params [[buffer(5)]],          \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    (void)weights;                                                             \
    (void)scales;                                                              \
    (void)biases;                                                              \
    const uint tiles = params.output_size / (32 * Simds);                      \
    for (uint tile = group; tile < tiles; tile += params.groups)               \
      naxr_q4_ceiling_tile<Rows, Simds>(input, output, params.output_size,     \
                                        params.input_size, tile * 32 * Simds,  \
                                        simd_group);                           \
  }

NAXR_Q4_CEILING_KERNEL(bench_naxr_ceiling_s8_m16, 16, 8)
NAXR_Q4_CEILING_KERNEL(bench_naxr_ceiling_s8_m32, 32, 8)
NAXR_Q4_CEILING_KERNEL(bench_naxr_ceiling_s8_m64, 64, 8)

NAXR_Q4_KERNEL(bench_naxr_s4_m16, 16, 4)
NAXR_Q4_KERNEL(bench_naxr_s8_m16, 16, 8)
NAXR_Q4_KERNEL(bench_naxr_s4_m32, 32, 4)
NAXR_Q4_KERNEL(bench_naxr_s8_m32, 32, 8)
NAXR_Q4_KERNEL(bench_naxr_s4_m64, 64, 4)
NAXR_Q4_KERNEL(bench_naxr_s8_m64, 64, 8)

// Cooperative-tensor layout probe: writes each lane's element coordinates of
// the B operand, the A operand and the destination of the naxr descriptor as
// int4(index0, index1, valid, capacity) records, [which][lane][element] with
// 64 elements per lane. The harness prints the map behind the vectorized
// weight fill.
kernel void naxr_probe(device int4 *out [[buffer(0)]],
                       uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr auto descriptor =
      matmul2d_descriptor(16, 32, 64, false, true, false,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> operation;
  auto cb = operation.template get_right_input_cooperative_tensor<bfloat,
                                                                  bfloat, float>();
  auto ca = operation.template get_left_input_cooperative_tensor<bfloat,
                                                                 bfloat, float>();
  // The destination as the tile computes it: from a tensor-slice left operand.
  auto a = tensor((device bfloat *)out, dextents<int, 2>{64, 16},
                  array<int, 2>{1, 64});
  auto probe_a = a.slice<64, 16>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<
      decltype(probe_a), decltype(cb), float>();
  auto dump = [&](thread auto &t, uint which) {
    const ushort cap = t.get_capacity();
    for (ushort i = 0; i < cap && i < 64; ++i) {
      auto index = t.get_multidimensional_index(i);
      out[(which * 32 + simd_lane) * 64 + i] =
          int4(int(index[0]), int(index[1]), t.is_valid_element(i) ? 1 : 0,
               int(cap));
    }
  };
  dump(cb, 0);
  dump(ca, 1);
  dump(acc, 2);
}

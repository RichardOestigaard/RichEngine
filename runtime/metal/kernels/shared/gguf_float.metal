// fp32 projections of the GGUF float tensors llama.cpp keeps unquantized: the
// MoE router (ffn_gate_inp) and, when a GGUF stores them as F32, the GDN
// alpha/beta gates (ops/Linear.hpp float segments). out[r][n] = sum_k x[r][k]
// W[n][k] on simdgroup float 8x8 MMA with exact operands (bf16 activations
// widen to fp32, the weights as stored), so a result differs from the fp64
// product only by fp32 accumulation. llama.cpp's mul_mm stages F32 weights as
// half above eight rows; no phase here rounds them.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/sgmatrix.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

namespace gguf_float {

// K parts, one per simdgroup. A decode dispatch has few threadgroups (8 or 32 at the 35B's alpha/beta and router
// widths), so each simdgroup's chain of K / (8 Splits) dependent loads and MMAs sets its time: on the 40-core M3
// Max four parts took 26-28 us per decode router or alpha/beta dispatch.

// One threadgroup: output columns [8 tg.x, 8 tg.x + 8) of rows [8 Fragments tg.y, 8 Fragments tg.y + 8 Fragments). Simdgroup s accumulates
// K part s of every fragment; simdgroup f then adds fragment f's parts in order, so a result depends neither on
// the grid nor on the row count. Lane (fm, fn) holds x[row fm][k + fn, k + fn + 1] as A,
// W[column fn, fn + 1][k + fm] as B and the destination [row fm][column fn, fn + 1] (sgmatrix::lane_map).
template <typename T, typename W, uint Splits, uint Fragments>
inline void project(device const bfloat *input, device const W *weights, device T *output,
                    constant GgufFloatParams &p, uint2 tg, uint sg, uint lane, threadgroup float2 *parts) {
  static_assert(Splits >= Fragments, "simdgroup f reduces fragment f");
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint K = p.input_size, column = tg.x * 8 + l.fn, row0 = tg.y * (8 * Fragments);
  const uint live = min(8 * Fragments, p.rows - row0);
  const uint steps = K / 8, first = sg * steps / Splits, last = (sg + 1) * steps / Splits;
  device const W *w = weights + ulong(column) * K + l.fm;
  float2 acc[Fragments];
#pragma unroll
  for (uint f = 0; f < Fragments; ++f) acc[f] = float2(0);
#pragma unroll(4)
  for (uint step = first; step < last; ++step) {
    const uint k = step * 8;
    const float2 b = float2(w[k], w[K + k]);
#pragma unroll
    for (uint f = 0; f < Fragments; ++f) {
      if (f * 8 >= live) break;
      const uint row = f * 8 + l.fm;
      const float2 a = row < live
          ? float2(*reinterpret_cast<device const bfloat2 *>(input + ulong(row0 + row) * K + k + l.fn))
          : float2(0);
      sgmatrix::mma_acc<float>(acc[f], a, b);
    }
  }
#pragma unroll
  for (uint f = 0; f < Fragments; ++f) parts[(sg * Fragments + f) * 32 + lane] = acc[f];
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  const uint f = sg, row = f * 8 + l.fm;
  if (f >= Fragments || row >= live) return;
  float2 total = float2(0);
  for (uint s = 0; s < Splits; ++s) total += parts[(s * Fragments + f) * 32 + lane];
  device T *out = output + ulong(row0 + row) * p.out_stride + p.out_offset + column;
  out[0] = T(total.x);
  out[1] = T(total.y);
}

} // namespace gguf_float

// Grid (output_size / 8, ceil(rows / (8 Fragments))), 32 threads per K split.
#define GGUF_FLOAT_KERNEL(Name, T, W, Splits, Fragments)                                                       \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const W *weights [[buffer(1)]],           \
                   device T *output [[buffer(2)]], constant GgufFloatParams &p [[buffer(3)]],                 \
                   uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],     \
                   uint lane [[thread_index_in_simdgroup]]) {                                                 \
    threadgroup float2 parts[Splits * Fragments * 32];                                                        \
    gguf_float::project<T, W, Splits, Fragments>(input, weights, output, p, tg, sg, lane, parts);             \
  }
GGUF_FLOAT_KERNEL(gguf_float_bf16, bfloat, float, 16, 4)
GGUF_FLOAT_KERNEL(gguf_float_f32, float, float, 16, 4)
GGUF_FLOAT_KERNEL(gguf_float_wbf16_bf16, bfloat, bfloat, 16, 4)
GGUF_FLOAT_KERNEL(gguf_float_wbf16_f32, float, bfloat, 16, 4)
GGUF_FLOAT_KERNEL(gguf_float_wbf16_bf16_s8, bfloat, bfloat, 8, 4)
GGUF_FLOAT_KERNEL(gguf_float_wbf16_bf16_s4, bfloat, bfloat, 4, 4)
GGUF_FLOAT_KERNEL(gguf_float_wbf16_bf16_m8, bfloat, bfloat, 16, 1)
#undef GGUF_FLOAT_KERNEL

// The same projections on the neural accelerator (ops::FloatTile::NeuralAccelerator), whose matmul takes bf16
// operands: each F32 weight is split into three bf16 parts, its leading 8 significant bits, the next 8 and the last 8,
// so hi + mid + lo == w exactly (fp32 has 24) and each part's product with a bf16 activation is exact in fp32. The
// results again differ from the fp64 products only by fp32 accumulation. The accelerator adds a matmul's products to
// a destination with a few ulps of error (a probe: products 1 + 31 x 0.75 ulp(1) added to 1 + 23 ulp(1) came out
// 3 ulp(2) above the rounded sum), so the hi part accumulates apart from the mid and lo parts: that cuts the error
// against fp64 2.2x (mean 1.5e-8 -> 6.6e-9 of sum |x w| at the router's K = 2048, 512 rows; this file's fp32 kernel:
// 5.5e-9) for 3-6% of the kernel's time on a 20-core M5 Pro.
namespace gguf_float_na {

constant constexpr ushort kRowsPerSimdgroup = 16;   // the accelerator's row unit
constant constexpr ushort kSimdgroups = 4;
constant constexpr ushort kRows = kRowsPerSimdgroup * kSimdgroups;
constant constexpr ushort kColumns = 32;
constant constexpr ushort kParts = 3;

template <typename W, ushort Step>
struct StageBuffer {
  enum : uint { elements = (is_same<W, float>::value ? kParts : 1) * kColumns * Step };
};

// The parts of w as bf16 bits: each truncates the remainder to its leading 8 significant bits, so every subtraction
// is exact (Sterbenz) and the last remainder has at most 8 significant bits left.
inline ushort3 split(float w) {
  const uint hi = as_type<uint>(w) & 0xFFFF0000u;
  const float r = w - as_type<float>(hi);
  const uint mid = as_type<uint>(r) & 0xFFFF0000u;
  const float lo = r - as_type<float>(mid);
  return ushort3(ushort(hi >> 16), ushort(mid >> 16), ushort(as_type<uint>(lo) >> 16));
}

// One threadgroup: kColumns output columns of kRows rows, kRowsPerSimdgroup per simdgroup. Every thread splits
// kPerThread weights of a K step into the stage's three bf16 planes [part][column][input] (columns past the output
// stay zero); after the barrier each simdgroup with live rows runs the three matmuls, and one past the rows only
// stages. A simdgroup whose rows cross p.rows computes the 16 rows ending at p.rows instead and stores its own, so no
// row past the input is read (p.rows >= 16).
// W = float: stage each weight's three bf16 parts and run three matmuls. W =
// bfloat (a repacked draft segment, QuantizedSegment::kBfloat16): the weight
// is already the matmul's operand — stage it alone and run one, for a third
// of the matmul work and half the weight traffic.
template <typename T, typename W, ushort Step>
inline void project(device bfloat *input, device const W *weights, device T *output,
                    constant GgufFloatParams &p, uint2 tg, ushort sg, ushort lane, threadgroup bfloat *stage) {
  constexpr bool Split = is_same<W, float>::value;
  constexpr ushort PerThread = kColumns * Step / (kSimdgroups * 32);
  constexpr uint Plane = kColumns * Step;
  constexpr uint Buffer = StageBuffer<W, Step>::elements;
  static_assert(kColumns * Step % (kSimdgroups * 32) == 0 && PerThread % 4 == 0 && Step % PerThread == 0,
                "a thread stages whole vectorized weights of one column");
  using W4 = vec<W, 4>;
  const uint K = p.input_size, steps = K / Step;
  const uint row0 = tg.y * kRows + sg * kRowsPerSimdgroup, first_row = min(row0, p.rows - kRowsPerSimdgroup);
  const uint col0 = tg.x * kColumns;
  const ushort t = sg * 32 + lane, column = t / (Step / PerThread), first = t % (Step / PerThread) * PerThread;
  const bool live_column = col0 + column < p.output_size;
  device const W4 *w =
      reinterpret_cast<device const W4 *>(weights + ulong(live_column ? col0 + column : 0) * K + first);
  auto a = tensor(input + ulong(first_row) * K, dextents<int, 2>{int(K), kRowsPerSimdgroup}, array<int, 2>{1, int(K)});
  constexpr auto descriptor = matmul2d_descriptor(kRowsPerSimdgroup, kColumns, Step, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const auto plane = [&](uint buffer, uint part) {
    tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> b(
        stage + buffer * Buffer + part * Plane, dextents<int, 2>{Step, kColumns}, array<int, 2>{1, Step});
    return b.template slice<Step, kColumns>(0, 0);
  };
  auto h0 = plane(0, 0), h1 = plane(1, 0);
  auto m0 [[maybe_unused]] = plane(0, 1), l0 [[maybe_unused]] = plane(0, 2), m1 [[maybe_unused]] = plane(1, 1),
       l1 [[maybe_unused]] = plane(1, 2);
  auto a0 = a.template slice<Step, kRowsPerSimdgroup>(0, 0);
  auto high = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(h0), float>();
  auto low = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(h0), float>();
#pragma unroll
  for (ushort i = 0; i < high.get_capacity(); ++i) high[i] = low[i] = 0.0f;
  W4 next[PerThread / 4];
#pragma unroll
  for (ushort j = 0; j < PerThread / 4; ++j) next[j] = live_column ? w[j] : W4(0);
  const bool owns_rows = row0 < p.rows;   // uniform per simdgroup
  for (uint step = 0; step < steps; ++step) {
    threadgroup ushort *parts =
        reinterpret_cast<threadgroup ushort *>(stage + (step & 1) * Buffer) + column * Step + first;
    if constexpr (Split) {
#pragma unroll
      for (ushort j = 0; j < PerThread / 4; ++j) {
        const ushort3 x = split(next[j].x), y = split(next[j].y), z = split(next[j].z), v = split(next[j].w);
#pragma unroll
        for (ushort part = 0; part < kParts; ++part)
          *reinterpret_cast<threadgroup ushort4 *>(parts + part * Plane + 4 * j) =
              ushort4(x[part], y[part], z[part], v[part]);
      }
    } else {
#pragma unroll
      for (ushort j = 0; j < PerThread / 4; ++j)
        *reinterpret_cast<threadgroup ushort4 *>(parts + 4 * j) = as_type<ushort4>(next[j]);
    }
    if (step + 1 < steps && live_column) {
#pragma unroll
      for (ushort j = 0; j < PerThread / 4; ++j) next[j] = w[(step + 1) * (Step / 4) + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    if (owns_rows) {
      auto slice = a.template slice<Step, kRowsPerSimdgroup>(step * Step, 0);
      if constexpr (Split) {
        if (step & 1) {
          operation.run(slice, h1, high);
          operation.run(slice, m1, low);
          operation.run(slice, l1, low);
        } else {
          operation.run(slice, h0, high);
          operation.run(slice, m0, low);
          operation.run(slice, l0, low);
        }
      } else {
        if (step & 1) {
          operation.run(slice, h1, high);
        } else {
          operation.run(slice, h0, high);
        }
      }
    }
  }
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < high.get_capacity(); ++i) {
    if (!high.is_valid_element(i)) continue;
    const auto index = high.get_multidimensional_index(i);
    const uint row = first_row + index[1], column_out = col0 + index[0];
    if (row >= row0 && column_out < p.output_size)
      output[ulong(row) * p.out_stride + p.out_offset + column_out] = T(high[i] + low[i]);
  }
}

} // namespace gguf_float_na

// Grid (ceil(output_size / 32), ceil(rows / 64)), 128 threads; rows >= 16, input_size a multiple of Step.
#define GGUF_FLOAT_NA_KERNEL(Name, T, W, Step)                                                                 \
  kernel void Name(device bfloat *input [[buffer(0)]], device const W *weights [[buffer(1)]],                 \
                   device T *output [[buffer(2)]], constant GgufFloatParams &p [[buffer(3)]],                 \
                   uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],   \
                   ushort lane [[thread_index_in_simdgroup]]) {                                               \
    threadgroup bfloat stage[2 * gguf_float_na::StageBuffer<W, Step>::elements];                               \
    gguf_float_na::project<T, W, Step>(input, weights, output, p, tg, sg, lane, stage);                       \
  }
GGUF_FLOAT_NA_KERNEL(gguf_float_na_bf16, bfloat, float, 32)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_f32, float, float, 32)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_wbf16_bf16, bfloat, bfloat, 32)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_wbf16_f32, float, bfloat, 32)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_wbf16_bf16_k64, bfloat, bfloat, 64)
GGUF_FLOAT_NA_KERNEL(gguf_float_na_wbf16_bf16_k128, bfloat, bfloat, 128)
#undef GGUF_FLOAT_NA_KERNEL

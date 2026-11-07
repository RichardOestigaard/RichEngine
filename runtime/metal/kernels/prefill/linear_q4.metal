#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// Row-major sums of each 64-input group: sums[row * (input_size / 64) +
// group]. Every producer (this pass, prefill_norm_rms_sums32, the q4 prefill
// output-sums epilogues and the mixer gates) and consumer writes and reads
// this flat layout, which is also what lets a producer emit a row range
// starting at an unaligned row (the packed prefill sequences' rowBegin).
kernel void prefill_linear_q4_sums32(device const bfloat *input [[buffer(0)]],
                              device float *sums [[buffer(1)]],
                              constant uint &input_size [[buffer(2)]],
                              uint tile [[threadgroup_position_in_grid]],
                              uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group
                              [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TileM = 32;
  uint quant_groups = input_size / 64;
  input += ulong(tile) * TileM * input_size;
  sums += ulong(tile) * TileM * quant_groups;
  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    for (uint row = simd_group; row < TileM; row += 8) {
      uint origin = row * input_size + quant_group * 64 + simd_lane;
      float sum = simd_sum(float(input[origin]) + float(input[origin + 32]));
      if (simd_lane == 0) {
        sums[row * quant_groups + quant_group] = sum;
      }
    }
  }
}

// Eligible Apple9/Apple10 devices allow 32 KiB of static threadgroup memory.
// Keeping 256 row sums resident lets the 5120/6144-wide projections run with
// one load and the 17408-wide down projection with only one refill.
constant constexpr ushort PrefillSumBatch = 256;

// Eight simdgroups stage row sums in threadgroup memory. Four simdgroups read
// sums from device memory to avoid staging barriers and increase residency.
// Both paths use the same accumulation order.
template <ushort TileM, ushort TileN, ushort Simdgroups, bool AddResidual,
          bool MultiplySiluGate, ushort SumBatch = PrefillSumBatch>
inline void q4_mpp_prefill_tile(device bfloat *input, device uchar *weights,
                                device bfloat *scales, device bfloat *biases,
                                device bfloat *output, device bfloat *auxiliary,
                                uint output_size, uint input_size,
                                device const float *precomputed_sums,
                                uint output_origin, uint simd_lane,
                                uint simd_group,
                                threadgroup float *input_sums) {
  constexpr bool StagedSums = Simdgroups == 8;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  auto c = tensor(output, dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, TileM>(0, 0);
  uint quant_groups = input_size / 64;
  uint tile = output_origin / kQ4StorageColumns;
  uint tile_column = output_origin % kQ4StorageColumns;
  device uchar *tile_weights =
      weights +
      (ulong(tile) * quant_groups * kQ4StorageColumns + tile_column) * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    accumulated[i] = 0.0f;
  }

  auto load_sums = [&](uint start) {
    uint count = min(uint(SumBatch), quant_groups - start);
    uint thread_index = simd_group * 32 + simd_lane;
    for (uint index = thread_index; index < count * TileM;
         index += Simdgroups * 32) {
      uint quant_group = start + index / TileM;
      uint row = index % TileM;
      input_sums[index] = precomputed_sums[row * quant_groups + quant_group];
    }
  };
  if constexpr (StagedSums) {
    load_sums(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint quant_group = 0; quant_group < quant_groups; ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, TileM>(input_origin, 0);
    device uchar *group_weights =
        tile_weights + ulong(quant_group) * kQ4StorageColumns * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b_slice), float>();
    operation.run(a_slice, b_slice, partial);

#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
          tile_column + index[0];
      float sum = StagedSums
          ? input_sums[(quant_group % SumBatch) * TileM + row]
          : precomputed_sums[row * quant_groups + quant_group];
      accumulated[i] += partial[i] * float(scales[parameter]) +
                        sum * float(biases[parameter]);
    }
    if (StagedSums && quant_group % SumBatch == SumBatch - 1 &&
        quant_group + 1 < quant_groups) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_sums(quant_group + 1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  auto converted = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), bfloat>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    float value = float(bfloat(accumulated[i]));
    if constexpr (MultiplySiluGate) {
      auto index = accumulated.get_multidimensional_index(i);
      float gate =
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
      value = richengine_silu(gate) * value;
    }
    if constexpr (AddResidual) {
      auto index = accumulated.get_multidimensional_index(i);
      value +=
          float(auxiliary[index[1] * output_size + output_origin + index[0]]);
    }
    converted[i] = bfloat(value);
  }
  converted.store(c.slice<TileN, TileM>(output_origin, 0));
}

// The output's Q4 input sums of a TileM x TileN tile, for the projection that
// reads the output next.
template <ushort TileM, ushort TileN, ushort Simdgroups>
inline void q4_prefill_write_output_sums(device const bfloat *output,
                                         device float *output_sums,
                                         uint output_size, uint output_origin,
                                         uint simd_lane, uint simd_group) {
  constexpr uint QuantGroups = TileN / 64;
  threadgroup_barrier(mem_flags::mem_device);
  for (uint task = simd_group; task < TileM * QuantGroups;
       task += Simdgroups) {
    uint row = task / QuantGroups;
    uint local_group = task % QuantGroups;
    uint origin =
        row * output_size + output_origin + local_group * 64 + simd_lane;
    float sum = simd_sum(float(output[origin]) + float(output[origin + 32]));
    if (simd_lane == 0) {
      uint quant_group = output_origin / 64 + local_group;
      output_sums[row * (output_size / 64) + quant_group] = sum;
    }
  }
}

// What the prefill kernels do with a tile's sums: store them (Plain), add the
// residual at buffer 4 (Residual), or multiply silu of the gate at buffer 4
// in and also write the output's Q4 input sums at buffer 7 (UpSiluSums).
enum class PrefillQ4Epilogue { Plain, Residual, UpSiluSums };

// The TileM x TileN tile of grid position (row tile, column tile), taking
// the matrix's sizes directly so kernels whose parameters carry more than
// Q4Params (the persistent variants' Q4PersistentParams) can call it. The
// kernels inline it as their bodies: when prefill_linear_q4_n256_up_silu_sums
// only calls it, shader validation on macOS 26.5 (an M5 Pro) gets its output
// wrong.
template <ushort TileM, ushort TileN, ushort Simdgroups,
          PrefillQ4Epilogue Epilogue, ushort SumBatch = PrefillSumBatch>
__attribute__((always_inline)) inline void
q4_prefill_sizes(device bfloat *input, device uchar *weights,
                 device bfloat *scales, device bfloat *biases,
                 device bfloat *auxiliary, device bfloat *output,
                 device const float *sums, device float *output_sums,
                 uint output_size, uint input_size, uint2 group,
                 uint simd_lane, uint simd_group,
                 threadgroup float *input_sums) {
  constexpr bool UpSiluSums = Epilogue == PrefillQ4Epilogue::UpSiluSums;
  const ulong input_offset = ulong(group.x) * TileM * input_size;
  const ulong output_offset = ulong(group.x) * TileM * output_size;
  sums += ulong(group.x) * TileM * (input_size / 64);
  if constexpr (UpSiluSums)
    output_sums += ulong(group.x) * TileM * (output_size / 64);
  q4_mpp_prefill_tile<TileM, TileN, Simdgroups,
                      Epilogue == PrefillQ4Epilogue::Residual, UpSiluSums,
                      SumBatch>(
      input + input_offset, weights, scales, biases, output + output_offset,
      auxiliary + output_offset, output_size, input_size, sums,
      group.y * TileN, simd_lane, simd_group, input_sums);
  if constexpr (UpSiluSums)
    q4_prefill_write_output_sums<TileM, TileN, Simdgroups>(
        output + output_offset, output_sums, output_size,
        group.y * TileN, simd_lane, simd_group);
}

template <ushort TileM, ushort TileN, ushort Simdgroups,
          PrefillQ4Epilogue Epilogue, ushort SumBatch = PrefillSumBatch>
__attribute__((always_inline)) inline void
q4_prefill(device bfloat *input, device uchar *weights, device bfloat *scales,
           device bfloat *biases, device bfloat *auxiliary,
           device bfloat *output, device const float *sums,
           device float *output_sums, constant Q4Params &params, uint2 group,
           uint simd_lane, uint simd_group, threadgroup float *input_sums) {
  q4_prefill_sizes<TileM, TileN, Simdgroups, Epilogue, SumBatch>(
      input, weights, scales, biases, auxiliary, output, sums, output_sums,
      params.output_size, params.input_size, group, simd_lane, simd_group,
      input_sums);
}

// Each epilogue's bindings after the input, weights, scales and biases
// (0-3), and the arguments they give q4_prefill; the plain projection reads no
// auxiliary rows and passes its output in their place. The IO macros hold the
// epilogue buffers alone so kernels whose parameter payload differs (the
// persistent and K-split variants) can append their own.
#define PREFILL_Q4_IO_Plain                                                    \
  device bfloat *output [[buffer(4)]], device const float *sums [[buffer(5)]]
#define PREFILL_Q4_IO_Residual                                                 \
  device bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],  \
      device const float *sums [[buffer(6)]]
#define PREFILL_Q4_IO_UpSiluSums                                               \
  device bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]],      \
      device const float *sums [[buffer(6)]],                                  \
      device float *output_sums [[buffer(7)]]
#define PREFILL_Q4_BUFFERS_Plain                                               \
  PREFILL_Q4_IO_Plain, constant Q4Params &params [[buffer(6)]]
#define PREFILL_Q4_ARGUMENTS_Plain output, output, sums, nullptr
#define PREFILL_Q4_BUFFERS_Residual                                            \
  PREFILL_Q4_IO_Residual, constant Q4Params &params [[buffer(7)]]
#define PREFILL_Q4_ARGUMENTS_Residual residual, output, sums, nullptr
#define PREFILL_Q4_BUFFERS_UpSiluSums                                          \
  PREFILL_Q4_IO_UpSiluSums, constant Q4Params &params [[buffer(8)]]
#define PREFILL_Q4_ARGUMENTS_UpSiluSums gate, output, sums, output_sums
// Eight simdgroups stage the row sums in threadgroup memory; four read them
// from device memory.
#define PREFILL_Q4_INPUT_SUMS_8(M, B) threadgroup float input_sums[(M) * (B)]
#define PREFILL_Q4_INPUT_SUMS_4(M, B) threadgroup float *const input_sums = nullptr
#define PREFILL_Q4_M(Name, TileM, TileN, Simdgroups, Epilogue, SumBatch)       \
  [[max_total_threads_per_threadgroup((Simdgroups) * 32)]] kernel void Name(   \
                   device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   PREFILL_Q4_BUFFERS_##Epilogue,                              \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    PREFILL_Q4_INPUT_SUMS_##Simdgroups(TileM, SumBatch);                                        \
    q4_prefill<TileM, TileN, Simdgroups, PrefillQ4Epilogue::Epilogue, SumBatch>(         \
        input, weights, scales, biases, PREFILL_Q4_ARGUMENTS_##Epilogue,       \
        params, group, simd_lane, simd_group, input_sums);                     \
  }
#define PREFILL_Q4(Name, TileN, Simdgroups, Epilogue)                          \
  PREFILL_Q4_M(Name, 32, TileN, Simdgroups, Epilogue, PrefillSumBatch)
#define PREFILL_Q4_M64(Name, TileN, Epilogue)                                  \
  PREFILL_Q4_M(Name, 64, TileN, 8, Epilogue, PrefillSumBatch / 2)
PREFILL_Q4(prefill_linear_q4_n128, 128, 8, Plain)
PREFILL_Q4(prefill_linear_q4_n256, 256, 8, Plain)
PREFILL_Q4(prefill_linear_q4_n128_residual, 128, 8, Residual)
PREFILL_Q4(prefill_linear_q4_n256_residual, 256, 8, Residual)
PREFILL_Q4(prefill_linear_q4_n256_up_silu_sums, 256, 8, UpSiluSums)
// 64-row tiles halve the weight stream of the 32-row prefill tile on long
// prompts; selected by plan for rows > 32.
PREFILL_Q4_M64(prefill_linear_q4_n128_m64, 128, Plain)
PREFILL_Q4_M64(prefill_linear_q4_n256_m64, 256, Plain)
PREFILL_Q4_M64(prefill_linear_q4_n128_residual_m64, 128, Residual)
PREFILL_Q4_M64(prefill_linear_q4_n256_residual_m64, 256, Residual)
PREFILL_Q4_M64(prefill_linear_q4_n256_up_silu_sums_m64, 256, UpSiluSums)
PREFILL_Q4(prefill_linear_q4_n128_sg4, 128, 4, Plain)
PREFILL_Q4(prefill_linear_q4_n128_residual_sg4, 128, 4, Residual)
PREFILL_Q4(prefill_linear_q4_n128_up_silu_sums_sg4, 128, 4, UpSiluSums)
// The parameters (a scale, a bias and 64 weights per column and group) that
// a view of the leading input_size inputs of rows of plane_input_size leaves
// unread in the 256-column tiles before the one holding `column`.
inline ulong q4_unread_parameters(constant Q4PrefillLeadingParams &params,
                                  uint column) {
  return ulong(column / kQ4StorageColumns) *
         (params.plane_input_size / 64 - params.matrix.input_size / 64) *
         kQ4StorageColumns;
}

// The residual kernels over a view of the leading inputs of wider weight rows,
// for the Neural Engine FFN split's down projection. Each runs the tile of its
// residual kernel on the planes advanced past the parameters the view leaves
// unread before its column tile, which the tile then addresses as rows of
// input_size inputs.
#define PREFILL_Q4_LEADING_TM(Name, TileN, Simdgroups, TileM, SumBatch)                            \
  [[max_total_threads_per_threadgroup((Simdgroups) * 32)]] kernel void Name(   \
                   device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *residual [[buffer(4)]],                      \
                   device bfloat *output [[buffer(5)]],                        \
                   device const float *sums [[buffer(6)]],                     \
                   constant Q4PrefillLeadingParams &params [[buffer(7)]],      \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    PREFILL_Q4_INPUT_SUMS_##Simdgroups(TileM, SumBatch);                       \
    const ulong unread = q4_unread_parameters(params, group.y * TileN);        \
    q4_prefill<TileM, TileN, Simdgroups, PrefillQ4Epilogue::Residual,          \
              SumBatch>(                                                     \
        input, weights + unread * 64 / 2, scales + unread, biases + unread,    \
        residual, output, sums, nullptr, params.matrix, group, simd_lane,      \
        simd_group, input_sums);                                               \
  }
#define PREFILL_Q4_LEADING(Name, TileN, Simdgroups)                            \
  PREFILL_Q4_LEADING_TM(Name, TileN, Simdgroups, 32, PrefillSumBatch)
PREFILL_Q4_LEADING(prefill_linear_q4_n128_residual_leading_inputs, 128, 8)
PREFILL_Q4_LEADING(prefill_linear_q4_n256_residual_leading_inputs, 256, 8)
PREFILL_Q4_LEADING(prefill_linear_q4_n128_residual_sg4_leading_inputs, 128, 4)
#undef PREFILL_Q4_LEADING
PREFILL_Q4_LEADING_TM(prefill_linear_q4_n128_residual_m64_leading_inputs, 128, 8, 64, PrefillSumBatch / 2)
PREFILL_Q4_LEADING_TM(prefill_linear_q4_n256_residual_m64_leading_inputs, 256, 8, 64, PrefillSumBatch / 2)
#undef PREFILL_Q4_LEADING_TM

// --- Opt-in experiments -----------------------------------------------------
// RICHENGINE_PREFILL_Q4_KSPLIT picks the *_ksplit kernels: the grid is (row
// tiles, column tiles, splits) and threadgroup (x, y, z) sums the quant
// groups [z*G/splits, (z+1)*G/splits) of its tile — the same per-group terms
// the unsplit tile accumulates in order — then publishes them as fp32
// partials in the tile's block of `splits` TileM x TileN planes. The last
// split of a tile to arrive (kernels/common/split_reduce.h, as the decode
// Split128 tile) adds the partials in split order, applies the epilogue and
// stores bf16. Within a split a column's arithmetic is the unsplit tile's,
// so the result differs from it only by the association of the split sums
// at the boundary.
template <ushort TileM, ushort TileN, ushort Simdgroups, bool AddResidual,
          bool MultiplySiluGate, ushort SumBatch = PrefillSumBatch>
inline void q4_mpp_prefill_tile_split(
    device bfloat *input, device uchar *weights, device bfloat *scales,
    device bfloat *biases, device bfloat *auxiliary, device bfloat *output,
    device const float *precomputed_sums, device float *output_sums,
    device coherent(device) float *partials, device atomic_uint *counters,
    uint split, uint splits, uint output_size, uint input_size,
    uint output_origin, uint first_group, uint end_group, uint simd_lane,
    uint simd_group, threadgroup float *input_sums,
    threadgroup uint *arrival) {
  constexpr bool StagedSums = Simdgroups == 8;
  auto a = tensor(input, dextents<int, 2>{int(input_size), TileM},
                  array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a0 = a.slice<64, TileM>(0, 0);
  uint quant_groups = input_size / 64;
  uint tile = output_origin / kQ4StorageColumns;
  uint tile_column = output_origin % kQ4StorageColumns;
  device uchar *tile_weights =
      weights +
      (ulong(tile) * quant_groups * kQ4StorageColumns + tile_column) * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b(
      tile_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
  auto b0 = first_b.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    accumulated[i] = 0.0f;
  }

  // The unsplit tile batches the staged sums from group zero; the split's
  // batch runs from first_group, so the staging slot is the group's index
  // within the split's own range.
  auto load_sums = [&](uint start) {
    uint count = min(uint(SumBatch), end_group - start);
    uint thread_index = simd_group * 32 + simd_lane;
    for (uint index = thread_index; index < count * TileM;
         index += Simdgroups * 32) {
      uint quant_group = start + index / TileM;
      uint row = index % TileM;
      input_sums[index] = precomputed_sums[row * quant_groups + quant_group];
    }
  };
  if constexpr (StagedSums) {
    load_sums(first_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint quant_group = first_group; quant_group < end_group;
       ++quant_group) {
    uint input_origin = quant_group * 64;
    auto a_slice = a.slice<64, TileM>(input_origin, 0);
    device uchar *group_weights =
        tile_weights + ulong(quant_group) * kQ4StorageColumns * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a_slice), decltype(b_slice), float>();
    operation.run(a_slice, b_slice, partial);

#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
          tile_column + index[0];
      float sum = StagedSums
          ? input_sums[((quant_group - first_group) % SumBatch) * TileM + row]
          : precomputed_sums[row * quant_groups + quant_group];
      accumulated[i] += partial[i] * float(scales[parameter]) +
                        sum * float(biases[parameter]);
    }
    if (StagedSums && (quant_group - first_group) % SumBatch == SumBatch - 1 &&
        quant_group + 1 < end_group) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_sums(quant_group + 1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // The cooperative tensor's element ownership is the same in every split's
  // threadgroup, so the reducer's registers stand in for its own plane.
  const bool fullyOccupied =
      uint(accumulated.get_capacity()) * (uint(Simdgroups) * 32u) ==
      uint(TileM) * TileN;
  const auto traversal =
      fullyOccupied ? Q4Traversal::All : q4_traversal(accumulated);
  q4_visit(accumulated, traversal, [&](ushort i) {
    auto index = accumulated.get_multidimensional_index(i);
    partials[split * TileM * TileN + index[1] * TileN + index[0]] =
        accumulated[i];
  });
  const uint thread_index = simd_group * 32 + simd_lane;
  if (!split_arrive_last(counters, splits, thread_index, arrival))
    return;
  q4_visit(accumulated, traversal, [&](ushort i) {
    auto index = accumulated.get_multidimensional_index(i);
    const uint row = index[1], column = index[0];
    const uint element = row * output_size + output_origin + column;
    const float total = split_sum(
        accumulated[i], split, splits, [&](uint s) {
          return partials[s * TileM * TileN + row * TileN + column];
        });
    float value = float(bfloat(total));
    if constexpr (MultiplySiluGate) {
      value = richengine_silu(float(auxiliary[element])) * value;
    }
    if constexpr (AddResidual) {
      value += float(auxiliary[element]);
    }
    output[element] = bfloat(value);
  });
  if constexpr (MultiplySiluGate) {
    q4_prefill_write_output_sums<TileM, TileN, Simdgroups>(
        output, output_sums, output_size, output_origin, simd_lane,
        simd_group);
  }
  split_release(counters, thread_index);
}

template <ushort TileM, ushort TileN, ushort Simdgroups,
          PrefillQ4Epilogue Epilogue, ushort SumBatch = PrefillSumBatch>
__attribute__((always_inline)) inline void
q4_prefill_split(device bfloat *input, device uchar *weights,
                 device bfloat *scales, device bfloat *biases,
                 device bfloat *auxiliary, device bfloat *output,
                 device const float *sums, device float *output_sums,
                 constant Q4Params &params,
                 device coherent(device) float *partials,
                 device atomic_uint *counters, uint3 group, uint splits,
                 uint simd_lane, uint simd_group,
                 threadgroup float *input_sums, threadgroup uint *arrival) {
  constexpr bool UpSiluSums = Epilogue == PrefillQ4Epilogue::UpSiluSums;
  const ulong input_offset = ulong(group.x) * TileM * params.input_size;
  const ulong output_offset = ulong(group.x) * TileM * params.output_size;
  sums += ulong(group.x) * TileM * (params.input_size / 64);
  if constexpr (UpSiluSums)
    output_sums += ulong(group.x) * TileM * (params.output_size / 64);
  const uint quant_groups = params.input_size / 64;
  const uint first_group = group.z * quant_groups / splits;
  const uint end_group = (group.z + 1) * quant_groups / splits;
  const ulong tile_index =
      ulong(group.x) * (params.output_size / TileN) + group.y;
  q4_mpp_prefill_tile_split<TileM, TileN, Simdgroups,
                            Epilogue == PrefillQ4Epilogue::Residual,
                            UpSiluSums, SumBatch>(
      input + input_offset, weights, scales, biases,
      auxiliary + output_offset, output + output_offset, sums, output_sums,
      partials + tile_index * splits * TileM * TileN, counters + tile_index,
      group.z, splits, params.output_size, params.input_size, group.y * TileN,
      first_group, end_group, simd_lane, simd_group, input_sums, arrival);
}

// The split kernels' tail bindings: the epilogue buffers, then the partials
// and counters of the split reduction, then the parameters (the dispatch
// binds its payload after the last buffer).
#define PREFILL_Q4_SPLIT_TAIL_Plain                                            \
  device coherent(device) float *partials [[buffer(6)]],                       \
      device atomic_uint *counters [[buffer(7)]],                              \
      constant Q4Params &params [[buffer(8)]]
#define PREFILL_Q4_SPLIT_TAIL_Residual                                         \
  device coherent(device) float *partials [[buffer(7)]],                       \
      device atomic_uint *counters [[buffer(8)]],                              \
      constant Q4Params &params [[buffer(9)]]
#define PREFILL_Q4_SPLIT_TAIL_UpSiluSums                                       \
  device coherent(device) float *partials [[buffer(8)]],                       \
      device atomic_uint *counters [[buffer(9)]],                              \
      constant Q4Params &params [[buffer(10)]]
#define PREFILL_Q4_SPLIT_M(Name, TileM, TileN, Simdgroups, Epilogue, SumBatch) \
  [[max_total_threads_per_threadgroup((Simdgroups) * 32)]] kernel void Name(   \
                   device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   PREFILL_Q4_IO_##Epilogue,                                   \
                   PREFILL_Q4_SPLIT_TAIL_##Epilogue,                           \
                   uint3 group [[threadgroup_position_in_grid]],               \
                   uint3 grid [[threadgroups_per_grid]],                       \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    PREFILL_Q4_INPUT_SUMS_##Simdgroups(TileM, SumBatch);                       \
    threadgroup uint arrival;                                                \
    q4_prefill_split<TileM, TileN, Simdgroups,                                 \
                     PrefillQ4Epilogue::Epilogue, SumBatch>(                   \
        input, weights, scales, biases, PREFILL_Q4_ARGUMENTS_##Epilogue,       \
        params, partials, counters, group, grid.z, simd_lane, simd_group,      \
        input_sums, &arrival);                                                 \
  }
// The eight-simdgroup split variants stage one group fewer of row sums than
// the unsplit tile so the reduce's threadgroup arrival counter still fits
// the 32 KiB threadgroup-memory cap.
#define PREFILL_Q4_SPLIT(Name, TileN, Simdgroups, Epilogue)                    \
  PREFILL_Q4_SPLIT_M(Name, 32, TileN, Simdgroups, Epilogue, PrefillSumBatch - 1)
PREFILL_Q4_SPLIT(prefill_linear_q4_n128_ksplit, 128, 8, Plain)
PREFILL_Q4_SPLIT(prefill_linear_q4_n256_ksplit, 256, 8, Plain)
PREFILL_Q4_SPLIT(prefill_linear_q4_n128_residual_ksplit, 128, 8, Residual)
PREFILL_Q4_SPLIT(prefill_linear_q4_n256_residual_ksplit, 256, 8, Residual)
PREFILL_Q4_SPLIT(prefill_linear_q4_n256_up_silu_sums_ksplit, 256, 8, UpSiluSums)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n128_sg4_ksplit, 32, 128, 4, Plain, PrefillSumBatch)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n128_residual_sg4_ksplit, 32, 128, 4, Residual, PrefillSumBatch)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n128_up_silu_sums_sg4_ksplit, 32, 128, 4, UpSiluSums, PrefillSumBatch)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n128_m64_ksplit, 64, 128, 8, Plain, PrefillSumBatch / 2 - 1)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n256_m64_ksplit, 64, 256, 8, Plain, PrefillSumBatch / 2 - 1)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n128_residual_m64_ksplit, 64, 128, 8, Residual, PrefillSumBatch / 2 - 1)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n256_residual_m64_ksplit, 64, 256, 8, Residual, PrefillSumBatch / 2 - 1)
PREFILL_Q4_SPLIT_M(prefill_linear_q4_n256_up_silu_sums_m64_ksplit, 64, 256, 8, UpSiluSums, PrefillSumBatch / 2 - 1)
#undef PREFILL_Q4_SPLIT
#undef PREFILL_Q4_SPLIT_M
#undef PREFILL_Q4_SPLIT_TAIL_UpSiluSums
#undef PREFILL_Q4_SPLIT_TAIL_Residual
#undef PREFILL_Q4_SPLIT_TAIL_Plain

// RICHENGINE_PREFILL_Q4_PERSISTENT picks the *_persist kernels: the grid is
// (row tiles, params.groups) and each threadgroup strides over the column
// tiles — the decode tiles' persistent loop applied to the prefill tile.
// Every tile is still computed whole by one threadgroup in the same
// quant-group order, so the result is bit-identical to the tiled kernel's.
// The trailing barrier keeps a lagging simdgroup's reads of the staged sums
// (eight simdgroups) or the output-sums device fence apart from the next
// tile's writes.
#define PREFILL_Q4_PERSISTENT_M(Name, TileM, TileN, Simdgroups, Epilogue, SumBatch) \
  [[max_total_threads_per_threadgroup((Simdgroups) * 32)]] kernel void Name(   \
                   device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   PREFILL_Q4_IO_##Epilogue,                                   \
                   constant Q4PersistentParams &params                         \
                       [[buffer(PREFILL_Q4_PERSISTENT_PARAMS_##Epilogue)]],    \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    PREFILL_Q4_INPUT_SUMS_##Simdgroups(TileM, SumBatch);                       \
    const uint columns = params.output_size / TileN;                           \
    for (uint tile = group.y; tile < columns; tile += params.groups) {         \
      q4_prefill_sizes<TileM, TileN, Simdgroups,                               \
                       PrefillQ4Epilogue::Epilogue, SumBatch>(                 \
          input, weights, scales, biases, PREFILL_Q4_ARGUMENTS_##Epilogue,     \
          params.output_size, params.input_size, uint2(group.x, tile),         \
          simd_lane, simd_group, input_sums);                                  \
      threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    }                                                                          \
  }
#define PREFILL_Q4_PERSISTENT_PARAMS_Plain 6
#define PREFILL_Q4_PERSISTENT_PARAMS_Residual 7
#define PREFILL_Q4_PERSISTENT_PARAMS_UpSiluSums 8
#define PREFILL_Q4_PERSISTENT(Name, TileN, Simdgroups, Epilogue)               \
  PREFILL_Q4_PERSISTENT_M(Name, 32, TileN, Simdgroups, Epilogue, PrefillSumBatch)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n128_persist, 128, 8, Plain)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n256_persist, 256, 8, Plain)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n128_residual_persist, 128, 8, Residual)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n256_residual_persist, 256, 8, Residual)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n256_up_silu_sums_persist, 256, 8, UpSiluSums)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n128_sg4_persist, 128, 4, Plain)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n128_residual_sg4_persist, 128, 4, Residual)
PREFILL_Q4_PERSISTENT(prefill_linear_q4_n128_up_silu_sums_sg4_persist, 128, 4, UpSiluSums)
PREFILL_Q4_PERSISTENT_M(prefill_linear_q4_n128_m64_persist, 64, 128, 8, Plain, PrefillSumBatch / 2)
PREFILL_Q4_PERSISTENT_M(prefill_linear_q4_n256_m64_persist, 64, 256, 8, Plain, PrefillSumBatch / 2)
PREFILL_Q4_PERSISTENT_M(prefill_linear_q4_n128_residual_m64_persist, 64, 128, 8, Residual, PrefillSumBatch / 2)
PREFILL_Q4_PERSISTENT_M(prefill_linear_q4_n256_residual_m64_persist, 64, 256, 8, Residual, PrefillSumBatch / 2)
PREFILL_Q4_PERSISTENT_M(prefill_linear_q4_n256_up_silu_sums_m64_persist, 64, 256, 8, UpSiluSums, PrefillSumBatch / 2)
#undef PREFILL_Q4_PERSISTENT
#undef PREFILL_Q4_PERSISTENT_M
#undef PREFILL_Q4_PERSISTENT_PARAMS_UpSiluSums
#undef PREFILL_Q4_PERSISTENT_PARAMS_Residual
#undef PREFILL_Q4_PERSISTENT_PARAMS_Plain
#undef PREFILL_Q4
#undef PREFILL_Q4_INPUT_SUMS_4
#undef PREFILL_Q4_INPUT_SUMS_8
#undef PREFILL_Q4_ARGUMENTS_UpSiluSums
#undef PREFILL_Q4_BUFFERS_UpSiluSums
#undef PREFILL_Q4_ARGUMENTS_Residual
#undef PREFILL_Q4_BUFFERS_Residual
#undef PREFILL_Q4_ARGUMENTS_Plain
#undef PREFILL_Q4_BUFFERS_Plain

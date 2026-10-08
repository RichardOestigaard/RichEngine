// MoE expert-pass variants for moe_prefill_bench_metal_test.mm: the
// DiffusionGemma canvas step runs the split prefill passes
// (prefill_moe_expert_q4_n256_*_m32) at ~40 routed rows per expert, where
// the gate/up grid is only ~3 column tiles x ~100 row tiles and the pass
// reaches ~50 GB/s. These kernels probe the levers: narrower column tiles
// (more threadgroups), fewer simdgroups (more threadgroups resident),
// 16-row tiles, and a K-split that shortens each threadgroup's serial
// quant-group chain at the cost of fp32 partials. Bench-only; production
// dispatch never names them.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

// The indirect tile body of the production gate/up kernels, parameterized
// by tile rows (the grouped tile height), column tile, simdgroup count and
// the store: Gelu multiplies each output by richengine_gelu_tanh of the
// gate pass's bf16 result (the production GeGLU epilogue), plain stores
// the rounded sum.
template <ushort MaxRows, ushort TileN, ushort Simdgroups, bool Gelu>
inline void bench_moe_indirect(
    device const bfloat *input, device const uint *grouped_routes,
    device const MoeTileDescriptor *tiles, device const uint *tile_count,
    device uchar *packed, device uchar *shared, device bfloat *gate,
    device bfloat *output, constant MoeExpertParams &params, uint2 group,
    threadgroup bfloat *staged, threadgroup float *input_sums,
    uint simd_lane, uint simd_group, uint threads) {
  if (group.y >= *tile_count)
    return;
  // Slab resolution moved into the tile body is per-expert; keep it here:
  const MoeQ4Slab slab = moe_q4_slab(
      packed, shared, tiles[group.y].expert, params.experts,
      params.expert_stride_bytes_0, params.output_size, params.input_size);
  const ulong row = ulong(group.y) * MaxRows;
  moe_live_rows<MaxRows>(tiles[group.y].rows, [&](auto rows) {
    q4_mpp_tile_sums_indirect<decltype(rows)::value, TileN, false,
                              Simdgroups>(
        input, grouped_routes + row, params.reserved0, params.input_size,
        slab.weights, slab.scales, slab.biases, slab.weights, slab.scales,
        slab.biases, staged, input_sums, group.x * TileN, simd_lane,
        simd_group, threads,
        [&](thread auto &accumulated, thread auto &, Q4Traversal traversal)
            __attribute__((always_inline)) {
          q4_visit(accumulated, traversal, [&](ushort i) {
            auto index = accumulated.get_multidimensional_index(i);
            const uint output_index =
                index[1] * params.output_size + group.x * TileN + index[0];
            const ulong at = row * params.output_size + output_index;
            if constexpr (Gelu) {
              output[at] = bfloat(richengine_gelu_tanh(float(gate[at])) *
                                  float(bfloat(accumulated[i])));
            } else {
              output[at] = bfloat(accumulated[i]);
            }
          });
        });
  });
}

#define BENCH_MOE_INDIRECT(Name, MaxRows, TileN, Simdgroups, Gelu)           \
  kernel void Name(                                                        \
      device const bfloat *input [[buffer(0)]],                            \
      device const uint *grouped_routes [[buffer(1)]],                     \
      device const MoeTileDescriptor *tiles [[buffer(2)]],                 \
      device const uint *tile_count [[buffer(3)]],                         \
      device uchar *packed [[buffer(4)]],                                  \
      device uchar *shared [[buffer(5)]],                                  \
      device bfloat *gate [[buffer(6)]],                                   \
      device bfloat *output [[buffer(7)]],                                 \
      constant MoeExpertParams &params [[buffer(8)]],                      \
      uint2 group [[threadgroup_position_in_grid]],                        \
      uint simd_lane [[thread_index_in_simdgroup]],                        \
      uint simd_group [[simdgroup_index_in_threadgroup]],                  \
      uint2 tpg [[threads_per_threadgroup]]) {                             \
    threadgroup bfloat staged[MaxRows * 256];                              \
    threadgroup float input_sums[8 * MaxRows];                             \
    bench_moe_indirect<MaxRows, TileN, Simdgroups, Gelu>(                  \
        input, grouped_routes, tiles, tile_count, packed, shared, gate,    \
        output, params, group, staged, input_sums, simd_lane, simd_group,  \
        tpg.x);                                                            \
  }

// Up pass (GeGLU) variants.
BENCH_MOE_INDIRECT(bench_moe_n128_up_gelu_indirect_m32, 32, 128, 8, true)
BENCH_MOE_INDIRECT(bench_moe_n64_up_gelu_indirect_m32, 32, 64, 8, true)
BENCH_MOE_INDIRECT(bench_moe_n256_sg4_up_gelu_indirect_m32, 32, 256, 4, true)
BENCH_MOE_INDIRECT(bench_moe_n128_sg4_up_gelu_indirect_m32, 32, 128, 4, true)
BENCH_MOE_INDIRECT(bench_moe_n256_up_gelu_indirect_m16, 16, 256, 8, true)
// Gate pass (plain store) variants.
BENCH_MOE_INDIRECT(bench_moe_n128_gate_indirect_m32, 32, 128, 8, false)
BENCH_MOE_INDIRECT(bench_moe_n64_gate_indirect_m32, 32, 64, 8, false)

// Down pass: the dense grouped-input tile at a narrower column width.
template <ushort Rows, ushort TileN>
inline void bench_moe_down_tile(device bfloat *grouped_input,
                                device const MoeTileDescriptor *tiles,
                                device uchar *packed, device uchar *shared,
                                device bfloat *output,
                                constant MoeExpertParams &params, uint2 group,
                                ulong row, threadgroup float *input_sums,
                                uint simd_lane, uint simd_group) {
  const MoeQ4Slab slab = moe_q4_slab(
      packed, shared, tiles[group.y].expert, params.experts,
      params.expert_stride_bytes_0, params.output_size, params.input_size);
  q4_mpp_tile<Rows, TileN, false, false, false>(
      grouped_input + row * params.input_size, slab.weights, slab.scales,
      slab.biases, output + row * params.output_size, slab.weights,
      slab.scales, slab.biases, output + row * params.output_size,
      params.output_size, params.input_size, input_sums, group.x * TileN,
      simd_lane, simd_group);
}

#define BENCH_MOE_DOWN(Name, TileN)                                        \
  kernel void Name(                                                        \
      device bfloat *grouped_input [[buffer(0)]],                          \
      device const MoeTileDescriptor *tiles [[buffer(1)]],                 \
      device const uint *tile_count [[buffer(2)]],                         \
      device uchar *packed [[buffer(3)]],                                  \
      device uchar *shared [[buffer(4)]],                                  \
      device bfloat *output [[buffer(5)]],                                 \
      constant MoeExpertParams &params [[buffer(6)]],                      \
      uint2 group [[threadgroup_position_in_grid]],                        \
      uint simd_lane [[thread_index_in_simdgroup]],                        \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                \
    threadgroup float input_sums[8 * 32];                                  \
    if (group.y >= *tile_count)                                            \
      return;                                                              \
    const ulong row = ulong(group.y) * 32;                                 \
    moe_live_rows<32>(tiles[group.y].rows, [&](auto rows) {                \
      bench_moe_down_tile<decltype(rows)::value, TileN>(                   \
          grouped_input, tiles, packed, shared, output, params, group,     \
          row, input_sums, simd_lane, simd_group);                         \
    });                                                                    \
  }

BENCH_MOE_DOWN(bench_moe_n128_m32, 128)
BENCH_MOE_DOWN(bench_moe_n64_m32, 64)

// ---- K-split ------------------------------------------------------------
// A ranged copy of q4_mpp_tile_sums_indirect: blocks [first_block,
// end_block) of the input's 256-wide quant groups. Identical accumulation
// order inside the range; the caller reduces the splits' fp32 sums.
template <ushort Rows, ushort TileN, ushort Simdgroups, class Store>
__attribute__((always_inline)) inline void bench_indirect_range(
    device const bfloat *input, device const uint *row_routes,
    uint routes_per_row, ulong input_size, device uchar *weights_0,
    device bfloat *scales_0, device bfloat *biases_0,
    threadgroup bfloat *staged, threadgroup float *input_sums,
    uint output_origin, uint first_block, uint end_block, uint simd_lane,
    uint simd_group, uint threads, const thread Store &store) {
  auto a = tensor(staged, dextents<int, 2>{256, int(Rows)},
                  array<int, 2>{1, 256});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  const uint quant_groups = uint(input_size / 64);
  const uint tile = output_origin / kQ4StorageColumns;
  const uint tile_offset = output_origin % kQ4StorageColumns;
  device uchar *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * kQ4StorageColumns * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a.slice<64, Rows>(0, 0)), decltype(b00), float>();
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      uint(Rows) * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal, [&](ushort i) { accumulated_0[i] = 0.0F; });
  const uint stage_thread = simd_group * 32 + simd_lane;
  for (uint block = first_block; block < end_block; ++block) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = stage_thread; e < Rows * 32; e += threads) {
      const uint row = e / 32, col = (e % 32) * 8;
      const uint route = row_routes[row];
      threadgroup uint4 *dst =
          reinterpret_cast<threadgroup uint4 *>(staged + row * 256 + col);
      if (route == ~0u) {
        *dst = uint4(0);
      } else {
        *dst = *reinterpret_cast<device const uint4 *>(
            input + ulong(route / routes_per_row) * input_size + block * 256 +
            col);
      }
    }
    for (uint row = simd_group; row < Rows; row += Simdgroups) {
      threadgroup const bfloat *r = staged + row * 256;
      const float first =
          simd_sum(float(r[simd_lane]) + float(r[simd_lane + 32]));
      const float second =
          simd_sum(float(r[simd_lane + 64]) + float(r[simd_lane + 96]));
      const float third =
          simd_sum(float(r[simd_lane + 128]) + float(r[simd_lane + 160]));
      const float fourth =
          simd_sum(float(r[simd_lane + 192]) + float(r[simd_lane + 224]));
      if (simd_lane == 0) {
        const uint sum_origin = (block & 1) * (4 * Rows);
        input_sums[sum_origin + row] = first;
        input_sums[sum_origin + Rows + row] = second;
        input_sums[sum_origin + 2 * Rows + row] = third;
        input_sums[sum_origin + 3 * Rows + row] = fourth;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < 4; ++g) {
      const uint quant_group = block * 4 + g;
      auto a_slice = a.slice<64, Rows>(g * 64, 0);
      device uchar *group_weights_0 =
          tile_weights_0 +
          (ulong(quant_group) * kQ4StorageColumns + tile_offset) * 64 / 2;
      tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
          group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
      auto b0_slice = b0.slice<64, TileN>(0, 0);
      decltype(accumulated_0) partial_0;
      operation.run(a_slice, b0_slice, partial_0);
      q4_visit(accumulated_0, traversal,
               [&](ushort i) __attribute__((always_inline)) {
        auto index = accumulated_0.get_multidimensional_index(i);
        const uint row = index[1];
        const ulong parameter =
            (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
            tile_offset + index[0];
        const uint sum_offset = (block & 1) * (4 * Rows) + g * Rows;
        accumulated_0[i] += partial_0[i] * float(scales_0[parameter]) +
                            input_sums[sum_offset + row] *
                                float(biases_0[parameter]);
      });
    }
  }
  store(accumulated_0, traversal);
}

// Splits the quant-group blocks of the up_gelu indirect tile across
// grid.z: every split publishes its Rows x TileN fp32 sums to
// partials[((tile * colTiles + col) * Splits + split) * 32 * 256 + slot],
// and the last to arrive adds them in split order and applies the GeGLU
// store (split_reduce.h).
template <ushort Splits>
inline void bench_moe_up_gelu_ksplit(
    device const bfloat *input, device const uint *grouped_routes,
    device const MoeTileDescriptor *tiles, device const uint *tile_count,
    device uchar *up_packed, device uchar *shared_up, device bfloat *gate,
    device bfloat *output, constant MoeExpertParams &params,
    device float *partials, device atomic_uint *counters, uint3 group,
    threadgroup bfloat *staged, threadgroup float *input_sums,
    threadgroup uint *arrival, uint simd_lane, uint simd_group,
    uint threads) {
  if (group.y >= *tile_count)
    return;
  constexpr ushort TileN = 256;
  const MoeQ4Slab slab = moe_q4_slab(
      up_packed, shared_up, tiles[group.y].expert, params.experts,
      params.expert_stride_bytes_0, params.output_size, params.input_size);
  const ulong row = ulong(group.y) * 32;
  const uint blocks = params.input_size / 256;
  const uint first = group.z * blocks / Splits;
  const uint last = (group.z + 1) * blocks / Splits;
  const uint colTiles = params.output_size / TileN;
  device float *tile_partials =
      partials +
      (ulong(group.y) * colTiles + group.x) * Splits * 32 * TileN;
  device atomic_uint *counter = counters + group.y * colTiles + group.x;
  const uint slot = (simd_group * 32 + simd_lane);
  bench_indirect_range<32, TileN, 8>(
      input, grouped_routes + row, params.reserved0, params.input_size,
      slab.weights, slab.scales, slab.biases, staged, input_sums,
      group.x * TileN, first, last, simd_lane, simd_group, threads,
      [&](thread auto &accumulated, Q4Traversal traversal)
          __attribute__((always_inline)) {
        q4_visit(accumulated, traversal, [&](ushort i) {
          tile_partials[ulong(group.z) * 32 * TileN +
                        slot * accumulated.get_capacity() + i] =
              accumulated[i];
        });
      });
  if (!split_arrive_last(counter, Splits, simd_group * 32 + simd_lane,
                         arrival))
    return;
  // Reducing tile: add the splits in order, then the GeGLU store.
  constexpr auto descriptor =
      matmul2d_descriptor(32, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto staged_a = tensor(staged, dextents<int, 2>{256, 32},
                         array<int, 2>{1, 256})
                      .slice<64, 32>(0, 0);
  auto b0 = tensor<device uint4b_format, dextents<int, 2>, tensor_inline>(
                slab.weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64})
                .slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(staged_a), decltype(b0), float>();
  const uint capacity = accumulated.get_capacity();
  q4_visit(accumulated, Q4Traversal::All, [&](ushort i) {
    const uint at = slot * capacity + i;
    accumulated[i] = split_sum<float>(
        tile_partials[ulong(group.z) * 32 * TileN + at], uint(group.z),
        Splits, [&](uint s) {
          return tile_partials[ulong(s) * 32 * TileN + at];
        });
  });
  split_release(counter, simd_group * 32 + simd_lane);
  q4_visit(accumulated, Q4Traversal::All, [&](ushort i) {
    auto index = accumulated.get_multidimensional_index(i);
    const uint output_index =
        index[1] * params.output_size + group.x * TileN + index[0];
    const ulong at = row * params.output_size + output_index;
    output[at] = bfloat(richengine_gelu_tanh(float(gate[at])) *
                        float(bfloat(accumulated[i])));
  });
}

#define BENCH_MOE_UP_GELU_KSPLIT(Name, Splits)                             \
  kernel void Name(                                                        \
      device const bfloat *input [[buffer(0)]],                            \
      device const uint *grouped_routes [[buffer(1)]],                     \
      device const MoeTileDescriptor *tiles [[buffer(2)]],                 \
      device const uint *tile_count [[buffer(3)]],                         \
      device uchar *up_packed [[buffer(4)]],                               \
      device uchar *shared_up [[buffer(5)]],                               \
      device bfloat *gate [[buffer(6)]],                                   \
      device bfloat *output [[buffer(7)]],                                 \
      constant MoeExpertParams &params [[buffer(8)]],                      \
      device float *partials [[buffer(9)]],                                \
      device atomic_uint *counters [[buffer(10)]],                         \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint simd_lane [[thread_index_in_simdgroup]],                        \
      uint simd_group [[simdgroup_index_in_threadgroup]],                  \
      uint3 tpg [[threads_per_threadgroup]]) {                             \
    threadgroup bfloat staged[32 * 256];                                   \
    threadgroup float input_sums[8 * 32];                                  \
    threadgroup uint arrival;                                              \
    bench_moe_up_gelu_ksplit<Splits>(                                      \
        input, grouped_routes, tiles, tile_count, up_packed, shared_up,    \
        gate, output, params, partials, counters, group, staged,           \
        input_sums, &arrival, simd_lane, simd_group, tpg.x);               \
  }

BENCH_MOE_UP_GELU_KSPLIT(bench_moe_up_gelu_ksplit2_m32, 2)
BENCH_MOE_UP_GELU_KSPLIT(bench_moe_up_gelu_ksplit4_m32, 4)

// ---- Pipelined indirect tile -------------------------------------------
// The production indirect tile issues one group matmul then its epilogue
// per iteration (q4_mpp_tile_sums_indirect). The direct tile's Pipelined
// form issues two matmuls before either epilogue, hiding a group's weight
// load behind the other's; this applies it inside the staged-block loop.
template <ushort Rows, ushort TileN, ushort Simdgroups, class Store>
__attribute__((always_inline)) inline void bench_indirect_pipe(
    device const bfloat *input, device const uint *row_routes,
    uint routes_per_row, ulong input_size, device uchar *weights_0,
    device bfloat *scales_0, device bfloat *biases_0,
    threadgroup bfloat *staged, threadgroup float *input_sums,
    uint output_origin, uint simd_lane, uint simd_group, uint threads,
    const thread Store &store) {
  auto a = tensor(staged, dextents<int, 2>{256, int(Rows)},
                  array<int, 2>{1, 256});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  const uint quant_groups = uint(input_size / 64);
  const uint tile = output_origin / kQ4StorageColumns;
  const uint tile_offset = output_origin % kQ4StorageColumns;
  device uchar *tile_weights_0 =
      weights_0 + ulong(tile) * quant_groups * kQ4StorageColumns * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_b0(
      tile_weights_0 + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto b00 = first_b0.slice<64, TileN>(0, 0);
  auto accumulated_0 = operation.template get_destination_cooperative_tensor<
      decltype(a.slice<64, Rows>(0, 0)), decltype(b00), float>();
  const bool fullyOccupied =
      uint(accumulated_0.get_capacity()) * (uint(Simdgroups) * 32u) ==
      uint(Rows) * TileN;
  const auto traversal = fullyOccupied ? Q4Traversal::All
                                       : q4_traversal(accumulated_0);
  q4_visit(accumulated_0, traversal,
           [&](ushort i) { accumulated_0[i] = 0.0F; });
  const uint stage_thread = simd_group * 32 + simd_lane;
  for (uint block = 0; block < quant_groups / 4; ++block) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = stage_thread; e < Rows * 32; e += threads) {
      const uint row = e / 32, col = (e % 32) * 8;
      const uint route = row_routes[row];
      threadgroup uint4 *dst =
          reinterpret_cast<threadgroup uint4 *>(staged + row * 256 + col);
      if (route == ~0u) {
        *dst = uint4(0);
      } else {
        *dst = *reinterpret_cast<device const uint4 *>(
            input + ulong(route / routes_per_row) * input_size + block * 256 +
            col);
      }
    }
    for (uint row = simd_group; row < Rows; row += Simdgroups) {
      threadgroup const bfloat *r = staged + row * 256;
      const float first =
          simd_sum(float(r[simd_lane]) + float(r[simd_lane + 32]));
      const float second =
          simd_sum(float(r[simd_lane + 64]) + float(r[simd_lane + 96]));
      const float third =
          simd_sum(float(r[simd_lane + 128]) + float(r[simd_lane + 160]));
      const float fourth =
          simd_sum(float(r[simd_lane + 192]) + float(r[simd_lane + 224]));
      if (simd_lane == 0) {
        const uint sum_origin = (block & 1) * (4 * Rows);
        input_sums[sum_origin + row] = first;
        input_sums[sum_origin + Rows + row] = second;
        input_sums[sum_origin + 2 * Rows + row] = third;
        input_sums[sum_origin + 3 * Rows + row] = fourth;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto run_group = [&](uint g) {
      auto a_slice = a.slice<64, Rows>(g * 64, 0);
      const uint quant_group = block * 4 + g;
      device uchar *group_weights_0 =
          tile_weights_0 +
          (ulong(quant_group) * kQ4StorageColumns + tile_offset) * 64 / 2;
      tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
          group_weights_0, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
      auto b0_slice = b0.slice<64, TileN>(0, 0);
      decltype(accumulated_0) partial_0;
      operation.run(a_slice, b0_slice, partial_0);
      return partial_0;
    };
    auto finish_group = [&](uint g,
                            thread decltype(accumulated_0) &partial_0) {
      const uint quant_group = block * 4 + g;
      q4_visit(accumulated_0, traversal,
               [&](ushort i) __attribute__((always_inline)) {
        auto index = accumulated_0.get_multidimensional_index(i);
        const uint row = index[1];
        const ulong parameter =
            (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
            tile_offset + index[0];
        const uint sum_offset = (block & 1) * (4 * Rows) + g * Rows;
        accumulated_0[i] += partial_0[i] * float(scales_0[parameter]) +
                            input_sums[sum_offset + row] *
                                float(biases_0[parameter]);
      });
    };
    {
      auto p0 = run_group(0);
      auto p1 = run_group(1);
      finish_group(0, p0);
      finish_group(1, p1);
      auto p2 = run_group(2);
      auto p3 = run_group(3);
      finish_group(2, p2);
      finish_group(3, p3);
    }
  }
  store(accumulated_0, traversal);
}

kernel void bench_moe_n256_up_gelu_pipe_m32(
    device const bfloat *input [[buffer(0)]],
    device const uint *grouped_routes [[buffer(1)]],
    device const MoeTileDescriptor *tiles [[buffer(2)]],
    device const uint *tile_count [[buffer(3)]],
    device uchar *up_packed [[buffer(4)]],
    device uchar *shared_up [[buffer(5)]],
    device bfloat *gate [[buffer(6)]],
    device bfloat *output [[buffer(7)]],
    constant MoeExpertParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint2 tpg [[threads_per_threadgroup]]) {
  threadgroup bfloat staged[32 * 256];
  threadgroup float input_sums[8 * 32];
  if (group.y >= *tile_count)
    return;
  const MoeQ4Slab slab = moe_q4_slab(
      up_packed, shared_up, tiles[group.y].expert, params.experts,
      params.expert_stride_bytes_0, params.output_size, params.input_size);
  const ulong row = ulong(group.y) * 32;
  moe_live_rows<32>(tiles[group.y].rows, [&](auto rows) {
    bench_indirect_pipe<decltype(rows)::value, 256, 8>(
        input, grouped_routes + row, params.reserved0, params.input_size,
        slab.weights, slab.scales, slab.biases, staged, input_sums,
        group.x * 256, simd_lane, simd_group, tpg.x,
        [&](thread auto &accumulated, Q4Traversal traversal)
            __attribute__((always_inline)) {
          q4_visit(accumulated, traversal, [&](ushort i) {
            auto index = accumulated.get_multidimensional_index(i);
            const uint output_index =
                index[1] * params.output_size + group.x * 256 + index[0];
            const ulong at = row * params.output_size + output_index;
            output[at] = bfloat(richengine_gelu_tanh(float(gate[at])) *
                                float(bfloat(accumulated[i])));
          });
        });
  });
}

// ---- Dense prefill tile variants ----------------------------------------
// q4_mpp_prefill_tile lives in prefill/linear_q4.metal; this clone adds the
// Pipelined option q4_mpp_tile_sums already has (issue two groups' matmuls
// before either epilogue) and a DeviceSums path for TileM/sg4 forms whose
// staged sums would exceed the 32KB threadgroup limit.
constant constexpr ushort BenchSumBatch = 256;

template <ushort TileM, ushort TileN, ushort Simdgroups, bool Pipelined>
__attribute__((always_inline)) inline void bench_prefill_tile(
    device bfloat *input, device uchar *weights, device bfloat *scales,
    device bfloat *biases, device bfloat *output, uint output_size,
    uint input_size, device const float *precomputed_sums, uint output_origin,
    uint simd_lane, uint simd_group, threadgroup float *input_sums) {
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
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    accumulated[i] = 0.0f;

  auto load_sums = [&](uint start) {
    uint count = min(uint(BenchSumBatch), quant_groups - start);
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

  auto run_group = [&](uint quant_group)
      __attribute__((always_inline)) {
    auto a_slice = a.slice<64, TileM>(quant_group * 64, 0);
    device uchar *group_weights =
        tile_weights + ulong(quant_group) * kQ4StorageColumns * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto b_slice = b.slice<64, TileN>(0, 0);
    decltype(accumulated) partial;
    operation.run(a_slice, b_slice, partial);
    return partial;
  };
  auto finish_group = [&](uint quant_group,
                          thread decltype(accumulated) &partial)
                          __attribute__((always_inline)) {
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      auto index = accumulated.get_multidimensional_index(i);
      uint row = index[1];
      ulong parameter =
          (ulong(tile) * quant_groups + quant_group) * kQ4StorageColumns +
          tile_column + index[0];
      float sum = StagedSums
          ? input_sums[(quant_group % BenchSumBatch) * TileM + row]
          : precomputed_sums[row * quant_groups + quant_group];
      accumulated[i] += partial[i] * float(scales[parameter]) +
                        sum * float(biases[parameter]);
    }
  };
  uint quant_group = 0;
  if constexpr (Pipelined) {
    // Bench shapes keep every group in the first staged batch, so no
    // mid-batch refill barrier is needed inside the pairs.
    for (; quant_group + 1 < quant_groups; quant_group += 2) {
      auto first = run_group(quant_group);
      auto second = run_group(quant_group + 1);
      finish_group(quant_group, first);
      finish_group(quant_group + 1, second);
    }
  }
  for (; quant_group < quant_groups; ++quant_group) {
    auto partial = run_group(quant_group);
    finish_group(quant_group, partial);
  }

  auto converted = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), bfloat>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    converted[i] = bfloat(accumulated[i]);
  converted.store(c.slice<TileN, TileM>(output_origin, 0));
}

#define BENCH_DENSE_Q4(Name, TileM, TileN, Simdgroups, Pipelined, Sums)       \
  kernel void Name(device bfloat *input [[buffer(0)]],                        \
                   device uchar *weights [[buffer(1)]],                       \
                   device bfloat *scales [[buffer(2)]],                       \
                   device bfloat *biases [[buffer(3)]],                       \
                   device bfloat *output [[buffer(4)]],                       \
                   device const float *sums [[buffer(5)]],                    \
                   constant Q4Params &params [[buffer(6)]],                   \
                   uint2 group [[threadgroup_position_in_grid]],              \
                   uint simd_lane [[thread_index_in_simdgroup]],              \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {      \
    const ulong input_offset = ulong(group.x) * TileM * params.input_size;    \
    const ulong output_offset = ulong(group.x) * TileM * params.output_size;  \
    device const float *row_sums =                                            \
        sums + ulong(group.x) * TileM * (params.input_size / 64);             \
    Sums;                                                                     \
    bench_prefill_tile<TileM, TileN, Simdgroups, Pipelined>(                  \
        input + input_offset, weights, scales, biases,                        \
        output + output_offset, params.output_size, params.input_size,        \
        row_sums, group.y * TileN, simd_lane, simd_group, input_sums);        \
  }

#define BENCH_STAGED_SUMS threadgroup float input_sums[BenchSumBatch * 32]
#define BENCH_DEVICE_SUMS threadgroup float *const input_sums = nullptr

// Same tile as the production kernel, pipelined group issue.
BENCH_DENSE_Q4(bench_dense_q4_n256_pipe, 32, 256, 8, true, BENCH_STAGED_SUMS)
// Four-simdgroup tile: 128 threads, device sums (no staging barrier).
BENCH_DENSE_Q4(bench_dense_q4_n256_sg4, 32, 256, 4, false, BENCH_DEVICE_SUMS)
// 64-row tile: halves the per-projection weight re-reads at M=256 (four row
// tiles instead of eight); sg4 keeps the sums un-staged.
BENCH_DENSE_Q4(bench_dense_q4_n256_m64_sg4, 64, 256, 4, false,
               BENCH_DEVICE_SUMS)

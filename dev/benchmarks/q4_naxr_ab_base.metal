// Baseline arm of the NAXR A/B (q4_naxr_ab.mm): the production direct-packed
// matmul2d Q4 tiles of kernels/common/q4_mpp_tiles.h, instantiated exactly as
// decode/linear_q4.metal instantiates them, under bench names. This file
// holds no experiment code: it compiles at both metal4.1 (the production
// standard) and metal4.0 so the harness can measure whether the standard
// itself moves the baseline before any naxr number is read.
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/q4_mpp_tiles.h"

#define Q4NAXR_BASELINE_THREADS(Name, TileCall, Sums, TileN, Threads)          \
  [[max_total_threads_per_threadgroup(Threads)]] kernel void Name(             \
                   device bfloat *input [[buffer(0)]],                         \
                   device uchar *weights [[buffer(1)]],                        \
                   device bfloat *scales [[buffer(2)]],                        \
                   device bfloat *biases [[buffer(3)]],                        \
                   device bfloat *output [[buffer(4)]],                        \
                   constant Q4PersistentParams &params [[buffer(5)]],          \
                   uint group [[threadgroup_position_in_grid]],                \
                   uint simd_lane [[thread_index_in_simdgroup]],               \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float input_sums[Sums];                                        \
    uint tiles = params.output_size / TileN;                                   \
    for (uint tile = group; tile < tiles; tile += params.groups) {             \
      TileCall(input, weights, scales, biases, output, weights, scales,        \
               biases, input, params.output_size, params.input_size,           \
               input_sums, tile * TileN, simd_lane, simd_group);               \
    }                                                                          \
  }
#define Q4NAXR_BASELINE(Name, TileCall, Sums, TileN)                           \
  Q4NAXR_BASELINE_THREADS(Name, TileCall, Sums, TileN, 256)

Q4NAXR_BASELINE(bench_base_n128_m16, (q4_mpp_tile<16, 128, false, false>), 128, 128)
Q4NAXR_BASELINE(bench_base_n256_m16, (q4_mpp_tile<16, 256, false, false>), 128, 256)
Q4NAXR_BASELINE(bench_base_n128_m32, (q4_mpp_tile<32, 128, false, false>), 256, 128)
Q4NAXR_BASELINE(bench_base_n256_m32, (q4_mpp_tile<32, 256, false, false>), 256, 256)
Q4NAXR_BASELINE(bench_base_n128_m64, (q4_mpp_tile<64, 128, false, false>), 512, 128)
Q4NAXR_BASELINE(bench_base_n256_m64, (q4_mpp_tile<64, 256, false, false>), 512, 256)
// The production 64-row wide tile's spill relief: sixteen simdgroups halve
// the accumulator registers per thread (decode/linear_q4.metal).
Q4NAXR_BASELINE_THREADS(bench_base_n256_m64_sg16,
                        (q4_mpp_tile<64, 256, false, false, false, 16>), 512,
                        256, 512)

#undef Q4NAXR_BASELINE
#undef Q4NAXR_BASELINE_THREADS

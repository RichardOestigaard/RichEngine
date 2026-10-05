#include "metal/kernels/common/paged_store_row_fp8.h"

// FP8 E4M3 verify stores: the INT8 entries' group walk with the e4m3 row
// quantization of paged_store_row_fp8.h.
#if defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__) && \
    defined(__HAVE_PACKED_NUMERIC_TYPE_PACK_UNPACK__)

template <uint KVHeads, uint HeadDim = SPLASH_KV_HEAD_DIMENSION>
inline void splash_store_fp8_verify_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table0, device const SplashKvPage *page_table1,
    device const SplashKvPage *page_table2, device const SplashKvPage *page_table3,
    constant SplashChunkedPrefillParams *params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint GroupsPerLane = 2 * Rows * KVHeads;
  uint batch = group / GroupsPerLane;
  uint local_group = group % GroupsPerLane;
  constant SplashChunkedPrefillParams &lane_params = params[batch];
  if (!splash_chunk_contract_valid(lane_params) ||
      lane_params.chunk_tokens != Rows ||
      lane_params.chunk_stride != SPLASH_VERIFY_CHUNK_STRIDE ||
      thread_index >= HeadDim)
    return;
  device const SplashKvPage *page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  constexpr ulong lane_tensor_stride =
      ulong(KVHeads) * SPLASH_VERIFY_CHUNK_STRIDE * HeadDim;
  chunk_keys += batch * lane_tensor_stride;
  chunk_values += batch * lane_tensor_stride;

  uint rows = Rows * KVHeads;
  bool value_tensor = local_group >= rows;
  uint row = value_tensor ? local_group - rows : local_group;
  uint chunk_token = row % Rows;
  uint head = row / Rows;
  splash_store_kv_row_fp8<KVHeads, HeadDim>(
      chunk_keys, chunk_values, page_table, lane_params, maxima, value_tensor,
      head, chunk_token, thread_index, simd_lane, simd_group);
}

#define PAGED_FP8_STORE(Name, Heads, HeadDim)                                  \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const SplashKvPage *page_table0 [[buffer(2)]],                    \
      device const SplashKvPage *page_table1 [[buffer(3)]],                    \
      device const SplashKvPage *page_table2 [[buffer(4)]],                    \
      device const SplashKvPage *page_table3 [[buffer(5)]],                    \
      constant SplashChunkedPrefillParams *params [[buffer(6)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup float maxima[HeadDim / 32];                                    \
    splash_store_fp8_verify_phase<Heads, HeadDim>(                             \
        chunk_keys, chunk_values, page_table0, page_table1, page_table2,       \
        page_table3, params, maxima, group, thread_index, simd_lane,           \
        simd_group);                                                           \
  }

PAGED_FP8_STORE(verify_attention_fp8_store, 4, 256)
PAGED_FP8_STORE(verify_attention_fp8_store_kv2_g8, 2, 256)
PAGED_FP8_STORE(verify_attention_fp8_store_hd128, 2, 128)
PAGED_FP8_STORE(verify_attention_fp8_store_hd64, 8, 64)
#undef PAGED_FP8_STORE

#endif // __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__

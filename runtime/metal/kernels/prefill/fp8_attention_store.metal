#include "metal/kernels/common/paged_store_row_fp8.h"

// FP8 E4M3 prefill stores: every current row to its final page slot through
// the e4m3 row quantization of paged_store_row_fp8.h.
#if defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__) && \
    defined(__HAVE_PACKED_NUMERIC_TYPE_PACK_UNPACK__)

template <uint KVHeads, uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline void richengine_store_fp8_chunk_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const RichKvPage *page_table,
    constant RichChunkedPrefillParams &params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  uint rows = params.chunk_tokens * KVHeads;
  if (!richengine_chunk_contract_valid(params) || group >= 2 * rows ||
      thread_index >= HeadDim)
    return;

  bool value_tensor = group >= rows;
  uint row = value_tensor ? group - rows : group;
  uint chunk_token = row % params.chunk_tokens;
  uint head = row / params.chunk_tokens;
  richengine_store_kv_row_fp8<KVHeads, HeadDim>(
      chunk_keys, chunk_values, page_table, params, maxima, value_tensor, head,
      chunk_token, thread_index, simd_lane, simd_group);
}

#define PAGED_FP8_STORE(Name, Heads, HeadDim)                                  \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const RichKvPage *page_table [[buffer(2)]],                     \
      constant RichChunkedPrefillParams &params [[buffer(3)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup float maxima[HeadDim / 32];                                    \
    richengine_store_fp8_chunk_phase<Heads, HeadDim>(                              \
        chunk_keys, chunk_values, page_table, params, maxima, group,           \
        thread_index, simd_lane, simd_group);                                  \
  }

PAGED_FP8_STORE(prefill_attention_fp8_store, 4, 256)
PAGED_FP8_STORE(prefill_attention_fp8_store_kv2_g8, 2, 256)
PAGED_FP8_STORE(prefill_attention_fp8_store_hd128, 2, 128)
PAGED_FP8_STORE(prefill_attention_fp8_store_hd64, 8, 64)
PAGED_FP8_STORE(prefill_attention_fp8_store_k8d128, 8, 128)
#undef PAGED_FP8_STORE

#endif // __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__

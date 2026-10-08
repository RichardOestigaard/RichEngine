#include "metal/kernels/common/paged_store_row.h"

// Writes every current row directly into its final page slot. Decode writes
// all speculative rows; acceptance is represented solely by the host-visible
// committed length. A rejected suffix remains unreachable and is overwritten
// by the next command starting at the same logical position.
template <uint KVHeads, typename CacheElement,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline void richengine_store_chunk_phase(
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
  richengine_store_kv_row<KVHeads, CacheElement, HeadDim>(
      chunk_keys, chunk_values, page_table, params, maxima, value_tensor, head,
      chunk_token, thread_index, simd_lane, simd_group);
}

kernel void
prefill_attention_q8_store(device const bfloat *chunk_keys [[buffer(0)]],
                        device const bfloat *chunk_values [[buffer(1)]],
                        device const RichKvPage *page_table [[buffer(2)]],
                        constant RichChunkedPrefillParams &params
                        [[buffer(3)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  richengine_store_chunk_phase<4, int8_t>(
      chunk_keys, chunk_values, page_table, params, maxima, group,
      thread_index, simd_lane, simd_group);
}

kernel void prefill_attention_q8_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const RichKvPage *page_table [[buffer(2)]],
    constant RichChunkedPrefillParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  richengine_store_chunk_phase<2, int8_t>(
      chunk_keys, chunk_values, page_table, params, maxima, group,
      thread_index, simd_lane, simd_group);
}

kernel void
prefill_attention_int4_store(device const bfloat *chunk_keys [[buffer(0)]],
                        device const bfloat *chunk_values [[buffer(1)]],
                        device const RichKvPage *page_table [[buffer(2)]],
                        constant RichChunkedPrefillParams &params
                        [[buffer(3)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  richengine_store_chunk_phase<4, RichKvPacked4>(
      chunk_keys, chunk_values, page_table, params, maxima, group,
      thread_index, simd_lane, simd_group);
}

kernel void prefill_attention_int4_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const RichKvPage *page_table [[buffer(2)]],
    constant RichChunkedPrefillParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  richengine_store_chunk_phase<2, RichKvPacked4>(
      chunk_keys, chunk_values, page_table, params, maxima, group,
      thread_index, simd_lane, simd_group);
}

// BF16 entries copy the source bits and need no scale reduction.

kernel void
prefill_attention_bf16_store(device const bfloat *chunk_keys [[buffer(0)]],
                        device const bfloat *chunk_values [[buffer(1)]],
                        device const RichKvPage *page_table [[buffer(2)]],
                        constant RichChunkedPrefillParams &params
                        [[buffer(3)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  richengine_store_chunk_phase<4, bfloat>(
      chunk_keys, chunk_values, page_table, params, nullptr, group,
      thread_index, simd_lane, simd_group);
}

kernel void prefill_attention_bf16_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const RichKvPage *page_table [[buffer(2)]],
    constant RichChunkedPrefillParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  richengine_store_chunk_phase<2, bfloat>(
      chunk_keys, chunk_values, page_table, params, nullptr, group,
      thread_index, simd_lane, simd_group);
}

// Head-dimension variants: the dense target's KV2 pages of 128 (_hd128)
// and LFM2's KV8 pages of 64 (_hd64); their threadgroups are the head
// dimension, so the scale reduction covers headDim/32 simdgroups.
#define PREFILL_ATTENTION_STORE_HD(Name, Heads, CacheElement, HeadDim)      \
  kernel void Name(                                                       \
      device const bfloat *chunk_keys [[buffer(0)]],                      \
      device const bfloat *chunk_values [[buffer(1)]],                    \
      device const RichKvPage *page_table [[buffer(2)]],                \
      constant RichChunkedPrefillParams &params [[buffer(3)]],          \
      uint group [[threadgroup_position_in_grid]],                        \
      uint thread_index [[thread_index_in_threadgroup]],                  \
      uint simd_lane [[thread_index_in_simdgroup]],                       \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {               \
    threadgroup float maxima[HeadDim / 32];                               \
    richengine_store_chunk_phase<Heads, CacheElement, HeadDim>(               \
        chunk_keys, chunk_values, page_table, params, maxima, group,      \
        thread_index, simd_lane, simd_group);                             \
  }
PREFILL_ATTENTION_STORE_HD(prefill_attention_q8_store_hd128, 2, int8_t, 128)
PREFILL_ATTENTION_STORE_HD(prefill_attention_int4_store_hd128, 2, RichKvPacked4, 128)
PREFILL_ATTENTION_STORE_HD(prefill_attention_bf16_store_hd128, 2, bfloat, 128)
PREFILL_ATTENTION_STORE_HD(prefill_attention_q8_store_hd64, 8, int8_t, 64)
PREFILL_ATTENTION_STORE_HD(prefill_attention_int4_store_hd64, 8, RichKvPacked4, 64)
PREFILL_ATTENTION_STORE_HD(prefill_attention_bf16_store_hd64, 8, bfloat, 64)
PREFILL_ATTENTION_STORE_HD(prefill_attention_q8_store_k8d128, 8, int8_t, 128)
PREFILL_ATTENTION_STORE_HD(prefill_attention_int4_store_k8d128, 8, RichKvPacked4, 128)
PREFILL_ATTENTION_STORE_HD(prefill_attention_bf16_store_k8d128, 8, bfloat, 128)
// Gemma 4: the sliding layers' KV8 pages of 256 and the global layers' KV2
// pages of 512 (threadgroups of 256 and 512 threads).
PREFILL_ATTENTION_STORE_HD(prefill_attention_q8_store_gemma_h256, 8, int8_t, 256)
PREFILL_ATTENTION_STORE_HD(prefill_attention_int4_store_gemma_h256, 8, RichKvPacked4, 256)
PREFILL_ATTENTION_STORE_HD(prefill_attention_bf16_store_gemma_h256, 8, bfloat, 256)
PREFILL_ATTENTION_STORE_HD(prefill_attention_q8_store_gemma_hd512, 2, int8_t, 512)
PREFILL_ATTENTION_STORE_HD(prefill_attention_int4_store_gemma_hd512, 2, RichKvPacked4, 512)
PREFILL_ATTENTION_STORE_HD(prefill_attention_bf16_store_gemma_hd512, 2, bfloat, 512)
#undef PREFILL_ATTENTION_STORE_HD

#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

constant uint SplashChunkMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
constant uint SplashChunkMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;

inline bool
splash_chunk_contract_valid(constant SplashChunkedPrefillParams &params) {
  uint required_pages = (params.committed_tokens + params.chunk_tokens +
                         SplashKvPageTokens - 1) /
                        SplashKvPageTokens;
  return params.committed_tokens <= SplashChunkMaximumPhysicalTokens &&
         params.chunk_tokens > 0 &&
         params.chunk_tokens <= SplashChunkMaximumRows &&
         params.committed_tokens + params.chunk_tokens <=
             SplashChunkMaximumPhysicalTokens &&
         params.chunk_stride >= params.chunk_tokens &&
         params.chunk_stride <= SplashChunkMaximumRows &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >= required_pages &&
         params.kv.extent_pages > 0;
}

template <uint HeadDim = SPLASH_KV_HEAD_DIMENSION>
inline ulong splash_current_key_index(uint stride, uint head, uint token,
                                        uint dimension) {
  return (ulong(head) * stride + token) * HeadDim + dimension;
}

template <uint HeadDim = SPLASH_KV_HEAD_DIMENSION>
inline ulong splash_current_value_index(uint stride, uint head, uint token,
                                          uint dimension) {
  return (ulong(head) * HeadDim + dimension) * stride + token;
}

// One lane per dimension stores a current row in its final page slot.
// INT8 derives a per-row scale; BF16 copies the original bits. Slots are
// addressed inside the head's slab of the page: the page index functions at
// page zero and head zero.
template <uint KVHeads, typename CacheElement,
          uint HeadDim = SPLASH_KV_HEAD_DIMENSION>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashKvPageTensors<CacheElement> slab =
      SplashKvAddressing<KVHeads, CacheElement, HeadDim>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index<HeadDim>(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : splash_current_key_index<HeadDim>(params.chunk_stride, head,
                                                chunk_token, dimension);
  bfloat source = value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

  if constexpr (is_same<CacheElement, bfloat>::value) {
    if (value_tensor)
      slab.values[splash_kv_value_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] = source;
    else
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] = source;
    return;
  } else if constexpr (is_same<CacheElement, SplashKvPacked4>::value) {
    constexpr uint Groups = HeadDim / 32;
    float value = float(source);
    float local_maximum = simd_max(abs(value));
    if (simd_lane == 0)
      maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      for (uint offset = Groups / 2; offset != 0; offset >>= 1) {
        if (simd_lane < offset)
          maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    float scale = maximum == 0.0f ? 0.0f : maximum / 7.0f;
    int quantized = maximum == 0.0f
                        ? 0
                        : clamp(int(rint(value * 7.0f / maximum)), -7, 7);
    uint nibble = uint(quantized) & 0xFu;

    // Packed bytes hold a dim pair at one token for both tensors, so the
    // even lane writes a whole byte after taking the odd lane's code by
    // shuffle. Keys and values are both token-major [token][dim pair], the
    // element order of the attention tile's int4b matmul2d operands.
    const uint partner = simd_shuffle_xor(nibble, 1);
    if (value_tensor) {
      if ((dimension & 1) == 0)
        slab.values[(page_token * HeadDim + dimension) / 2] =
            uchar(nibble | (partner << 4));
      if (dimension == 0)
        slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else {
      if ((dimension & 1) == 0)
        slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension) / 2] =
            uchar(nibble | (partner << 4));
      if (dimension == 0)
        slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  } else {

    constexpr uint Groups = HeadDim / 32;
    float value = float(source);
    float local_maximum = simd_max(abs(value));
    if (simd_lane == 0)
      maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      for (uint offset = Groups / 2; offset != 0; offset >>= 1) {
        if (simd_lane < offset)
          maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    int quantized = maximum == 0.0f
                        ? 0
                        : clamp(int(rint(value * 127.0f / maximum)), -127, 127);

    if (value_tensor) {
      slab.values[splash_kv_value_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else {
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  }
}

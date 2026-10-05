#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include "metal/kernels/common/paged_attention_fp8_tile.h"
#include "metal/kernels/common/paged_store_row.h"
#include <metal_stdlib>

using namespace metal;

// FP8 E4M3 page stores: the INT8 row loop's geometry and scale reduction,
// with the hardware's saturating round-to-nearest-even fp8 pack in place of
// the int8 clamp. The stored scale is the row's absolute maximum over 448,
// e4m3's largest magnitude, so one row's codes fill the format's range.
#if defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__) && \
    defined(__HAVE_PACKED_NUMERIC_TYPE_PACK_UNPACK__)

inline uchar splash_fp8_encode(float scaled) {
  // pack converts four floats; this row element keeps the first byte. The
  // oracle encodes the same value through the same instruction, so the two
  // agree bit for bit.
  return pack<metal_fp8_e4m3_format>(float4(scaled)).as_storage_type()[0];
}

// One lane per dimension stores a current row in its final page slot, like
// splash_store_kv_row's INT8 branch with an fp8 code in place of the int8.
template <uint KVHeads, uint HeadDim = SPLASH_KV_HEAD_DIMENSION>
__attribute__((always_inline)) inline void splash_store_kv_row_fp8(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashKvPageTensors<int8_t> slab =
      SplashKvAddressing<KVHeads, int8_t, HeadDim>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index<HeadDim>(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : splash_current_key_index<HeadDim>(params.chunk_stride, head,
                                                chunk_token, dimension);
  bfloat source =
      value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

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
  float scale = maximum == 0.0f ? 0.0f : maximum / 448.0f;
  uchar code =
      maximum == 0.0f ? uchar(0) : splash_fp8_encode(value * 448.0f / maximum);

  if (value_tensor) {
    reinterpret_cast<device uchar *>(slab.values)[
        splash_kv_value_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] = code;
    if (dimension == 0)
      slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
  } else {
    reinterpret_cast<device uchar *>(slab.keys)[
        splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] = code;
    if (dimension == 0)
      slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
  }
}

#endif // __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__

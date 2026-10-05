#pragma once

#include "metal/abi/KvExtent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// Kernels reach a KV page through its entry (abi/KvExtent.h): the GPU address
// of the extent that holds it, with the page's index in the low bits. Every
// kernel that touches KV pages forms the pointers to one KV head's slab of
// each of a page's tensors in one layer the same way, with
// RichKvAddressing: richengine_kv_offset places the layer's regions in the
// extent, whose 64-bit offsets it adds to the extent's address, and the page
// index functions of kv_paging.h place the page and the head in each region.

inline uint richengine_kv_page_index(RichKvPage page) {
  return uint(page) & RICHENGINE_KV_PAGE_INDEX_MASK;
}

inline device uchar *richengine_kv_extent(RichKvPage page, uint index) {
  return reinterpret_cast<device uchar *>(page - index);
}

// Marker for 4-bit KV: two elements per byte (even element in the low
// nibble), one fp32 scale per (head, token) exactly as INT8. Both tensors
// are token-major, so a head's slab feeds the attention tile's matmul2d as a
// device int4b_format operand with no unpacking: the keys' element order is
// {D, N} strides {1, D} and the values' {N, D} strides {D, 1}.
struct RichKvPacked4 {};

// One page's bytes of keys (or values) and of their scales in one layer.
// HeadDim is the page's head dimension; 256 is the original Qwen geometry.
template <uint KVHeads, typename CacheElement,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
struct RichKvPageBytes {
  static constexpr constant bool Quantized = is_same<CacheElement, int8_t>::value;
  static constexpr constant uint Data =
      KVHeads * RichKvPageTokens * HeadDim * sizeof(CacheElement);
  static constexpr constant uint Scale =
      Quantized ? KVHeads * RichKvPageTokens * sizeof(float) : 0;
};

template <uint KVHeads, uint HeadDim>
struct RichKvPageBytes<KVHeads, RichKvPacked4, HeadDim> {
  static constexpr constant bool Quantized = true;
  static constexpr constant uint Data =
      KVHeads * RichKvPageTokens * HeadDim / 2;
  static constexpr constant uint Scale =
      KVHeads * RichKvPageTokens * sizeof(float);
};

// One KV head's slab of each tensor of a page; BF16 has no scales.
template <typename CacheElement> struct RichKvPageTensors {
  device CacheElement *keys;
  device CacheElement *values;
  device float *key_scales;
  device float *value_scales;
};

template <> struct RichKvPageTensors<RichKvPacked4> {
  device uchar *keys;
  device uchar *values;
  device float *key_scales;
  device float *value_scales;
};

template <uint KVHeads, typename CacheElement,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
struct RichKvAddressing {
  using Bytes = RichKvPageBytes<KVHeads, CacheElement, HeadDim>;
  uint layer_offset;
  uint head;
  ulong key_scales_offset;
  ulong values_offset;
  ulong value_scales_offset;

  RichKvAddressing(RichKvLayer kv, uint kv_head)
      : layer_offset(kv.offset), head(kv_head),
        key_scales_offset(richengine_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                           0, RICHENGINE_KV_KEY_SCALES, 0)),
        values_offset(richengine_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                       0, RICHENGINE_KV_VALUES, 0)),
        value_scales_offset(richengine_kv_offset(kv.extent_pages, Bytes::Data,
                                             Bytes::Scale, 0,
                                             RICHENGINE_KV_VALUE_SCALES, 0)) {}

  RichKvPageTensors<CacheElement> page(RichKvPage entry) const {
    const uint index = richengine_kv_page_index(entry);
    device uchar *region = richengine_kv_extent(entry, index) + layer_offset;
    RichKvPageTensors<CacheElement> tensors{};
    tensors.keys = reinterpret_cast<device CacheElement *>(region) +
                   richengine_kv_key_index<KVHeads, HeadDim>(index, head, 0, 0);
    tensors.values = reinterpret_cast<device CacheElement *>(region + values_offset) +
                     richengine_kv_value_index<KVHeads, HeadDim>(index, head, 0, 0);
    if constexpr (Bytes::Quantized) {
      const ulong scale_index = richengine_q8_scale_index<KVHeads>(index, head, 0);
      tensors.key_scales =
          reinterpret_cast<device float *>(region + key_scales_offset) + scale_index;
      tensors.value_scales =
          reinterpret_cast<device float *>(region + value_scales_offset) + scale_index;
    }
    return tensors;
  }
};

// Packed pages hold the same elements in half the bytes: the element index
// functions address pairs, so every slab offset halves. Both tensors are
// token-major inside a head's slab, so the same halved key index gives the
// values' slab base.
template <uint KVHeads, uint HeadDim>
struct RichKvAddressing<KVHeads, RichKvPacked4, HeadDim> {
  using Bytes = RichKvPageBytes<KVHeads, RichKvPacked4, HeadDim>;
  uint layer_offset;
  uint head;
  ulong key_scales_offset;
  ulong values_offset;
  ulong value_scales_offset;

  RichKvAddressing(RichKvLayer kv, uint kv_head)
      : layer_offset(kv.offset), head(kv_head),
        key_scales_offset(richengine_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                           0, RICHENGINE_KV_KEY_SCALES, 0)),
        values_offset(richengine_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                       0, RICHENGINE_KV_VALUES, 0)),
        value_scales_offset(richengine_kv_offset(kv.extent_pages, Bytes::Data,
                                             Bytes::Scale, 0,
                                             RICHENGINE_KV_VALUE_SCALES, 0)) {}

  RichKvPageTensors<RichKvPacked4> page(RichKvPage entry) const {
    const uint index = richengine_kv_page_index(entry);
    device uchar *region = richengine_kv_extent(entry, index) + layer_offset;
    RichKvPageTensors<RichKvPacked4> tensors{};
    tensors.keys = region + richengine_kv_key_index<KVHeads, HeadDim>(index, head, 0, 0) / 2;
    tensors.values = region + values_offset +
                     richengine_kv_value_index<KVHeads, HeadDim>(index, head, 0, 0) / 2;
    const ulong scale_index = richengine_q8_scale_index<KVHeads>(index, head, 0);
    tensors.key_scales =
        reinterpret_cast<device float *>(region + key_scales_offset) + scale_index;
    tensors.value_scales =
        reinterpret_cast<device float *>(region + value_scales_offset) + scale_index;
    return tensors;
  }
};

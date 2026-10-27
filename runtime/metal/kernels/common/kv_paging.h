#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/KvExtent.h"
#include <metal_stdlib>

using namespace metal;

// KV page geometry of both formats, shared by the attention and store
// kernels. A page holds RichKvPageTokens tokens of every KV head: keys
// token-major, values dimension-major. INT8 adds one fp32 scale per (KV head,
// token) for each tensor.
constant uint RichKvPageTokens = RICHENGINE_TARGET_KV_BLOCK_TOKENS;
constant uint RichKvHeadDimension = RICHENGINE_KV_HEAD_DIMENSION;

// An element's index in a layer's region of one tensor: its page's slab,
// then its place in the slab (abi/KvExtent.h). HeadDim is the slab's head
// dimension; the default keeps every page a 256-dimension one.
template <uint KVHeads, uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline ulong richengine_kv_key_index(uint page, uint head, uint token,
                                   uint dimension) {
  constexpr ulong ElementsPerPage =
      ulong(KVHeads) * RichKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         richengine_kv_key_element_dim(head, token, dimension, HeadDim);
}

// A layout may keep its values token-major like its keys
// (kv::Layout::valuesTokenMajor): a block of a head's tokens then holds its
// values contiguously, which a sparse attention's block gather needs.
template <uint KVHeads, uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION,
          bool ValuesTokenMajor = false>
inline ulong richengine_kv_value_index(uint page, uint head, uint token,
                                     uint dimension) {
  constexpr ulong ElementsPerPage =
      ulong(KVHeads) * RichKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         (ValuesTokenMajor
              ? richengine_kv_key_element_dim(head, token, dimension, HeadDim)
              : richengine_kv_value_element_dim(head, token, dimension, HeadDim));
}

template <uint KVHeads>
inline ulong richengine_q8_scale_index(uint page, uint head, uint token) {
  constexpr ulong ScalesPerPage = ulong(KVHeads) * RichKvPageTokens;
  return ulong(page) * ScalesPerPage + richengine_kv_scale_element(head, token);
}

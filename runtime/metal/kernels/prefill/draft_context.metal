#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/draft_context_kv.h"

// One instantiation per compiled draft head geometry: the 32x8x128 of the
// DFlash2 and plain drafts, MiniCPM5 DSpark's 16x2x128 and LFM2.5 DSpark's
// interleaved-rotary 32x8x64.
template <uint KVHeads, uint HeadDim, bool Interleaved, bool E5>
inline void prefill_draft_context_kv_impl(
    device const bfloat *context_kv, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *keys, device bfloat *values,
    constant DraftContextParams &params, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint task, uint thread_index, uint lane,
    uint simd_group) {
  draft_context_kv_phase<KVHeads, HeadDim, Interleaved, E5>(
      context_kv, k_norm, rope_cos, rope_sin, keys, values,
      params.start_position, params.tokens, task, thread_index, lane,
      simd_group, reductions, normalized);
}

#define PREFILL_DRAFT_CONTEXT_KERNEL(NAME, KV_HEADS, HEAD_DIM, INTERLEAVED,  \
                                   E5)                                     \
  kernel void NAME(                                                          \
      device const bfloat *context_kv [[buffer(0)]],                         \
      device const bfloat *k_norm [[buffer(1)]],                             \
      device const float *rope_cos [[buffer(2)]],                            \
      device const float *rope_sin [[buffer(3)]],                            \
      device bfloat *keys [[buffer(4)]], device bfloat *values [[buffer(5)]],\
      constant DraftContextParams &params [[buffer(6)]],                     \
      uint task [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    threadgroup float reductions[8];                                         \
    threadgroup bfloat normalized[HEAD_DIM];                                 \
    prefill_draft_context_kv_impl<KV_HEADS, HEAD_DIM, INTERLEAVED, E5>(      \
        context_kv, k_norm, rope_cos, rope_sin, keys, values, params,        \
        reductions, normalized, task, thread_index, lane, simd_group);       \
  }

PREFILL_DRAFT_CONTEXT_KERNEL(prefill_draft_context_kv, 8, 128, false, false)
PREFILL_DRAFT_CONTEXT_KERNEL(prefill_draft_context_kv_q16k2, 2, 128, false,
                             false)
PREFILL_DRAFT_CONTEXT_KERNEL(prefill_draft_context_kv_q32k8d64i, 8, 64, true,
                             true)

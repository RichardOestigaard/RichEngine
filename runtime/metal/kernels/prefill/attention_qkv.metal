#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32).
#define PREFILL_ATTENTION_QKV(Name, QHeads, KHeads, W)                        \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                             \
      device const float *rope_sin [[buffer(4)]],                             \
      device bfloat *queries [[buffer(5)]],                                   \
      device bfloat *chunk_keys [[buffer(6)]],                                \
      device bfloat *chunk_values [[buffer(7)]],                              \
      constant FullPrefillParams &params [[buffer(8)]],                       \
      uint task [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[8];                                          \
    threadgroup bfloat normalized[256];                                       \
    full_qkv_storage_phase<QHeads, KHeads>(                                   \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, chunk_keys,         \
        chunk_values, params, reductions, normalized, task, thread_index,     \
        lane, simd_group);                                                    \
  }
PREFILL_ATTENTION_QKV(prefill_attention_qkv, 24, 4, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv4_g4, 16, 4, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g8, 16, 2, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_f32, 24, 4, float)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv4_g4_f32, 16, 4, float)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g8_f32, 16, 2, float)
#undef PREFILL_ATTENTION_QKV

// Head-dimension variants of the same row walk:
// - hd128: the dense target's 16x2 heads of 128, all 64 pairs rotated,
//   no query gate and no per-head norms.
// - hd64: LFM2's 32x8 heads of 64, all 32 pairs rotated, no query gate,
//   per-head RMS norms of epsilon 1e-5 (_f32 for a GGUF's F32 norms).
kernel void
prefill_attention_qkv_hd128(device const bfloat *qkv [[buffer(0)]],
                            device const bfloat *q_norm [[buffer(1)]],
                            device const bfloat *k_norm [[buffer(2)]],
                            device const float *rope_cos [[buffer(3)]],
                            device const float *rope_sin [[buffer(4)]],
                            device bfloat *queries [[buffer(5)]],
                            device bfloat *chunk_keys [[buffer(6)]],
                            device bfloat *chunk_values [[buffer(7)]],
                            constant FullPrefillParams &params [[buffer(8)]],
                            uint task [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  threadgroup bfloat normalized[256];
  full_qkv_storage_phase<16, 2, 128, 64, false, false>(
      qkv, q_norm, k_norm, rope_cos, rope_sin, queries, chunk_keys,
      chunk_values, params, reductions, normalized, task, thread_index, lane,
      simd_group);
}

#define PREFILL_ATTENTION_QKV_HD64(Name, W) \
  kernel void Name( \
      device const bfloat *qkv [[buffer(0)]], \
      device const W *q_norm [[buffer(1)]], \
      device const W *k_norm [[buffer(2)]], \
      device const float *rope_cos [[buffer(3)]], \
      device const float *rope_sin [[buffer(4)]], \
      device bfloat *queries [[buffer(5)]], \
      device bfloat *chunk_keys [[buffer(6)]], \
      device bfloat *chunk_values [[buffer(7)]], \
      constant FullPrefillParams &params [[buffer(8)]], \
      uint task [[threadgroup_position_in_grid]], \
      uint thread_index [[thread_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]], \
      uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float reductions[8]; \
    threadgroup bfloat normalized[256]; \
    full_qkv_storage_phase<32, 8, 64, 32, false, true>( \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, chunk_keys, \
        chunk_values, params, reductions, normalized, task, thread_index, lane, \
        simd_group, 1e-5f); \
  }
PREFILL_ATTENTION_QKV_HD64(prefill_attention_qkv_hd64, bfloat)
PREFILL_ATTENTION_QKV_HD64(prefill_attention_qkv_hd64_f32, float)
#undef PREFILL_ATTENTION_QKV_HD64

template <uint QHeads, uint KHeads, uint HeadDim = 256, bool Gate = true>
inline void full_attention_gate_prefill_phase(
    device const bfloat *packed_qkv, device const bfloat *attention,
    device bfloat *hidden, constant FullPrefillParams &params, uint index,
    uint grid_size) {
  constexpr uint QStride = Gate ? 2 * HeadDim : HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint HeadsPerKV = QHeads / KHeads;
  uint count = params.tokens * QHeads * HeadDim;
  for (uint element = index; element < count; element += grid_size) {
    uint row = element / (QHeads * HeadDim);
    uint remainder = element % (QHeads * HeadDim);
    uint query_head = remainder / HeadDim;
    uint dim = remainder % HeadDim;
    float gate_scale = 1.0f;
    if constexpr (Gate) {
      float gate = float(packed_qkv[ulong(row) * PackedStride +
                                    query_head * QStride + HeadDim + dim]);
      gate_scale = splash_sigmoid(gate);
    }
    uint kv_head = query_head / HeadsPerKV;
    uint local_head = query_head % HeadsPerKV;
    hidden[element] = bfloat(
        float(attention[((ulong(kv_head) * params.stride + row) *
                             HeadsPerKV +
                         local_head) *
                            HeadDim +
                        dim]) *
        gate_scale);
  }
}

kernel void
prefill_attention_gate(device const bfloat *packed_qkv [[buffer(0)]],
                            device const bfloat *attention [[buffer(1)]],
                            device bfloat *hidden [[buffer(2)]],
                            constant FullPrefillParams &params [[buffer(3)]],
                            uint index [[thread_position_in_grid]],
                            uint grid_size [[threads_per_grid]]) {
  full_attention_gate_prefill_phase<24, 4>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

kernel void prefill_attention_gate_kv4_g4(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullPrefillParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_prefill_phase<16, 4>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

kernel void prefill_attention_gate_kv2_g8(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullPrefillParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_prefill_phase<16, 2>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

// The *_sums variants also write the Affine64 out-projection's input sums of
// the gated rows — the flat [row][64-group] layout and pairing of
// prefill_linear_q4_sums32, computed post-store like
// q4_prefill_write_output_sums — so no separate sums pass runs on the mixer
// output. Each threadgroup owns one 256-element block inside one row (the
// row width is a multiple of 256), which is four 64-wide groups.
#define PREFILL_ATTENTION_GATE_SUMS(Name, QHeads, KHeads)                   \
  kernel void Name(                                                       \
      device const bfloat *packed_qkv [[buffer(0)]],                      \
      device const bfloat *attention [[buffer(1)]],                       \
      device bfloat *hidden [[buffer(2)]],                                \
      device float *sums [[buffer(3)]],                                   \
      constant FullPrefillParams &params [[buffer(4)]],                   \
      uint task [[threadgroup_position_in_grid]],                         \
      uint index [[thread_position_in_grid]],                             \
      uint grid_size [[threads_per_grid]],                                \
      uint threads [[threads_per_threadgroup]],                           \
      uint lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {               \
    full_attention_gate_prefill_phase<QHeads, KHeads>(                    \
        packed_qkv, attention, hidden, params, index, grid_size);         \
    threadgroup_barrier(mem_flags::mem_device);                           \
    for (uint group = simd_group; group < threads / 64;                   \
         group += threads / 32) {                                         \
      const uint origin = task * threads + group * 64 + lane;             \
      const float sum =                                                   \
          simd_sum(float(hidden[origin]) + float(hidden[origin + 32]));   \
      if (lane == 0)                                                      \
        sums[task * (threads / 64) + group] = sum;                        \
    }                                                                     \
  }
PREFILL_ATTENTION_GATE_SUMS(prefill_attention_gate_sums, 24, 4)
PREFILL_ATTENTION_GATE_SUMS(prefill_attention_gate_sums_kv4_g4, 16, 4)
PREFILL_ATTENTION_GATE_SUMS(prefill_attention_gate_sums_kv2_g8, 16, 2)
#undef PREFILL_ATTENTION_GATE_SUMS

// The no-gate rows of the dense and LFM2 targets: the same elementwise
// gather of the attention output, without the packed gate or the sigmoid.
// Their out-projections' sums are the generic prefill_linear_q4_sums32
// pass (ops::Linear::addPrefillSums).
#define PREFILL_ATTENTION_GATHER(Name, QHeads, KHeads, HeadDim)             \
  kernel void Name(                                                       \
      device const bfloat *packed_qkv [[buffer(0)]],                      \
      device const bfloat *attention [[buffer(1)]],                       \
      device bfloat *hidden [[buffer(2)]],                                \
      constant FullPrefillParams &params [[buffer(3)]],                   \
      uint index [[thread_position_in_grid]],                             \
      uint grid_size [[threads_per_grid]]) {                              \
    full_attention_gate_prefill_phase<QHeads, KHeads, HeadDim, false>(    \
        packed_qkv, attention, hidden, params, index, grid_size);         \
  }
PREFILL_ATTENTION_GATHER(prefill_attention_gather_hd128, 16, 2, 128)
PREFILL_ATTENTION_GATHER(prefill_attention_gather_hd64, 32, 8, 64)
#undef PREFILL_ATTENTION_GATHER

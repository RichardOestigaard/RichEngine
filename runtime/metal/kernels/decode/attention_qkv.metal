#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

template <uint QHeads, uint KHeads, class W>
inline void full_qkv_decode_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *keys, device bfloat *values, threadgroup float *reductions,
    threadgroup bfloat *normalized, constant FullDecodeBatchParams &params,
    uint2 group, uint thread_index, uint lane,
    uint simd_group) {
  constexpr uint HeadDim = 256, RotaryPairs = 32, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint Stride = RICHENGINE_VERIFY_CHUNK_STRIDE;
  const uint rows = params.rows;
  uint batch = group.y;
  const ulong kv_lane_stride = ulong(KHeads) * Stride * HeadDim;
  FullPrefillParams lane_params{rows, Stride};
  full_qkv_storage_phase<QHeads, KHeads>(
      qkv + ulong(batch) * rows * PackedStride, q_norm, k_norm,
      rope_cos + ulong(batch) * rows * RotaryPairs,
      rope_sin + ulong(batch) * rows * RotaryPairs,
      queries + ulong(batch) * QHeads * Stride * HeadDim,
      keys + ulong(batch) * kv_lane_stride,
      values + ulong(batch) * kv_lane_stride, lane_params, reductions,
      normalized, group.x, thread_index, lane, simd_group);
}

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32).
#define VERIFY_ATTENTION_QKV(Name, QHeads, KHeads, W)                         \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                             \
      device const float *rope_sin [[buffer(4)]],                             \
      device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]], \
      device bfloat *values [[buffer(7)]],                                    \
      constant FullDecodeBatchParams &params [[buffer(8)]],                   \
      uint2 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[8];                                          \
    threadgroup bfloat normalized[256];                                       \
    full_qkv_decode_phase<QHeads, KHeads>(                                    \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values,       \
        reductions, normalized, params, group, thread_index, lane,            \
        simd_group);                                                          \
  }
VERIFY_ATTENTION_QKV(verify_attention_qkv, 24, 4, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv4_g4, 16, 4, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g8, 16, 2, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_f32, 24, 4, float)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv4_g4_f32, 16, 4, float)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g8_f32, 16, 2, float)
#undef VERIFY_ATTENTION_QKV

// The head-dimension variants' decode phase: no query gate, a full rotary of
// HeadDim/2 pairs and per-head norms of epsilon `Eps` (LFM2's 1e-5; the
// dense target runs none).
template <uint QHeads, uint KHeads, uint HeadDim, bool Norms, class W>
inline void full_qkv_decode_phase_hd(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *keys, device bfloat *values, threadgroup float *reductions,
    threadgroup bfloat *normalized, constant FullDecodeBatchParams &params,
    uint2 group, uint thread_index, uint lane,
    uint simd_group, float eps) {
  constexpr uint RotaryPairs = HeadDim / 2;
  constexpr uint PackedStride = QHeads * HeadDim + 2 * KHeads * HeadDim;
  constexpr uint Stride = RICHENGINE_VERIFY_CHUNK_STRIDE;
  const uint rows = params.rows;
  uint batch = group.y;
  const ulong kv_lane_stride = ulong(KHeads) * Stride * HeadDim;
  FullPrefillParams lane_params{rows, Stride};
  full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, false, Norms>(
      qkv + ulong(batch) * rows * PackedStride, q_norm, k_norm,
      rope_cos + ulong(batch) * rows * RotaryPairs,
      rope_sin + ulong(batch) * rows * RotaryPairs,
      queries + ulong(batch) * QHeads * Stride * HeadDim,
      keys + ulong(batch) * kv_lane_stride,
      values + ulong(batch) * kv_lane_stride, lane_params, reductions,
      normalized, group.x, thread_index, lane, simd_group, eps);
}

// The dense target's KV2/Group8 of 128 (no norms) and LFM2's KV8/Group4 of
// 64 (norms of epsilon 1e-5, in their stored type).
#define VERIFY_ATTENTION_QKV_HD(Name, QHeads, KHeads, HeadDim, Norms, W, Eps) \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                             \
      device const float *rope_sin [[buffer(4)]],                             \
      device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]], \
      device bfloat *values [[buffer(7)]],                                    \
      constant FullDecodeBatchParams &params [[buffer(8)]],                   \
      uint2 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[HeadDim / 32];                               \
    threadgroup bfloat normalized[HeadDim];                                   \
    full_qkv_decode_phase_hd<QHeads, KHeads, HeadDim, Norms, W>(              \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values,       \
        reductions, normalized, params, group, thread_index, lane,            \
        simd_group, Eps);                                                     \
  }
VERIFY_ATTENTION_QKV_HD(verify_attention_qkv_hd128, 16, 2, 128, false, bfloat, 1e-6f)
VERIFY_ATTENTION_QKV_HD(verify_attention_qkv_hd64, 32, 8, 64, true, bfloat, 1e-5f)
VERIFY_ATTENTION_QKV_HD(verify_attention_qkv_hd64_f32, 32, 8, 64, true, float, 1e-5f)
// Granite's KV8 groups: 3B's 40 query heads of 64 (group 5, full rotary of
// 32 pairs) and 8B's 32 of 128 (group 4, 64 pairs); neither carries norms.
VERIFY_ATTENTION_QKV_HD(verify_attention_qkv_k8q5d64, 40, 8, 64, false, bfloat, 1e-5f)
VERIFY_ATTENTION_QKV_HD(verify_attention_qkv_k8q4d128, 32, 8, 128, false, bfloat, 1e-5f)
#undef VERIFY_ATTENTION_QKV_HD

template <uint QHeads, uint KHeads, uint HeadDim = 256, bool Gate = true>
inline bfloat full_attention_gate_value(device const bfloat *packed_qkv,
                                        device const bfloat *attention,
                                        uint element, uint rows) {
  constexpr uint QStride = Gate ? 2 * HeadDim : HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint HeadsPerKV = QHeads / KHeads;
  const uint per_lane = rows * QHeads * HeadDim;
  uint batch = element / per_lane;
  uint lane_element = element % per_lane;
  uint row = lane_element / (QHeads * HeadDim);
  uint remainder = lane_element % (QHeads * HeadDim);
  uint query_head = remainder / HeadDim;
  uint dim = remainder % HeadDim;
  float gate_scale = 1.0f;
  if constexpr (Gate) {
    float gate = float(
        packed_qkv[(ulong(batch) * rows + row) * PackedStride +
                   query_head * QStride + HeadDim + dim]);
    gate_scale = richengine_sigmoid(gate);
  }
  uint kv_head = query_head / HeadsPerKV;
  uint local_head = query_head % HeadsPerKV;
  ulong attention_index =
      (((ulong(batch) * KHeads + kv_head) * RICHENGINE_VERIFY_CHUNK_STRIDE + row) *
           HeadsPerKV +
       local_head) *
          HeadDim +
      dim;
  return bfloat(float(attention[attention_index]) * gate_scale);
}

template <uint QHeads, uint KHeads, uint HeadDim = 256, bool Gate = true>
inline void full_attention_gate_decode_phase(
    device const bfloat *packed_qkv, device const bfloat *attention,
    device bfloat *hidden, constant FullDecodeBatchParams &params, uint index,
    uint grid_size) {
  const uint count =
      params.lanes * params.rows * QHeads * HeadDim;
  for (uint element = index; element < count; element += grid_size)
    hidden[element] =
        full_attention_gate_value<QHeads, KHeads, HeadDim, Gate>(packed_qkv, attention, element, params.rows);
}

kernel void verify_attention_gate(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullDecodeBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_decode_phase<24, 4>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

kernel void verify_attention_gate_kv4_g4(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullDecodeBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_decode_phase<16, 4>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

kernel void verify_attention_gate_kv2_g8(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullDecodeBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_decode_phase<16, 2>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

#define ATTENTION_GATE_TABLE(Name, QHeads, KHeads, Layout, HeadDim, Gate) \
  kernel void Name( \
      device const bfloat *packed [[buffer(0)]], \
      device const bfloat *attention [[buffer(1)]], \
      device bfloat *hidden [[buffer(2)]], \
      device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]], \
      constant FullDecodeBatchParams &params [[buffer(5)]], \
      uint index [[thread_position_in_grid]], \
      uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint width = QHeads * HeadDim; \
    const uint element = 2 * index; \
    const bfloat a = full_attention_gate_value<QHeads, KHeads, HeadDim, Gate>(packed, attention, element, params.rows); \
    const bfloat b = full_attention_gate_value<QHeads, KHeads, HeadDim, Gate>(packed, attention, element + 1, params.rows); \
    hidden[element] = a; hidden[element + 1] = b; \
    const uint row = element / width; \
    Layout::write(table + ulong(row / 8) * width * 8, sums + ulong(row / 8) * Layout::sums_per_tile(width), \
                  width, (element % width) / 64, row % 8, lane, a, b); \
  }
ATTENTION_GATE_TABLE(verify_attention_gate_table64, 24, 4, q4sg::Table64, 256, true)
ATTENTION_GATE_TABLE(verify_attention_gate_table64_kv4_g4, 16, 4, q4sg::Table64, 256, true)
ATTENTION_GATE_TABLE(verify_attention_gate_table64_kv2_g8, 16, 2, q4sg::Table64, 256, true)
ATTENTION_GATE_TABLE(verify_attention_gate_table16, 24, 4, gguf_sg::Table16, 256, true)
ATTENTION_GATE_TABLE(verify_attention_gate_table16_kv4_g4, 16, 4, gguf_sg::Table16, 256, true)
ATTENTION_GATE_TABLE(verify_attention_gate_table16_kv2_g8, 16, 2, gguf_sg::Table16, 256, true)
// The no-gate gather variants of the dense and LFM2 targets, head
// dimensions 128 and 64.
ATTENTION_GATE_TABLE(verify_attention_gather_table64_hd128, 16, 2, q4sg::Table64, 128, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table16_hd128, 16, 2, gguf_sg::Table16, 128, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table64_hd64, 32, 8, q4sg::Table64, 64, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table16_hd64, 32, 8, gguf_sg::Table16, 64, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table64_k8q5d64, 40, 8, q4sg::Table64, 64, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table16_k8q5d64, 40, 8, gguf_sg::Table16, 64, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table64_k8q4d128, 32, 8, q4sg::Table64, 128, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table16_k8q4d128, 32, 8, gguf_sg::Table16, 128, false)
// Gemma 4's no-gate shapes: the sliding layers' 16x8 of 256 and the global
// layers' 16x2 of 512.
ATTENTION_GATE_TABLE(verify_attention_gather_table64_gemma_h256, 16, 8, q4sg::Table64, 256, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table16_gemma_h256, 16, 8, gguf_sg::Table16, 256, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table64_gemma_hd512, 16, 2, q4sg::Table64, 512, false)
ATTENTION_GATE_TABLE(verify_attention_gather_table16_gemma_hd512, 16, 2, gguf_sg::Table16, 512, false)
#undef ATTENTION_GATE_TABLE

// The mxfp4p-operand variant (LinearInput::Packed): the same two elements per
// thread through gguf_sg::Packed, which takes the table and sums slots as the
// fp16 plane and exponent bytes — the table grid already covers every element.
#define ATTENTION_GATE_PACKED(Name, QHeads, KHeads, HeadDim, Gate) \
  kernel void Name( \
      device const bfloat *packed [[buffer(0)]], \
      device const bfloat *attention [[buffer(1)]], \
      device bfloat *hidden [[buffer(2)]], \
      device half *plane [[buffer(3)]], device uchar *exponents [[buffer(4)]], \
      constant FullDecodeBatchParams &params [[buffer(5)]], \
      uint index [[thread_position_in_grid]], \
      uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint width = QHeads * HeadDim; \
    const uint element = 2 * index; \
    const bfloat a = full_attention_gate_value<QHeads, KHeads, HeadDim, Gate>(packed, attention, element, params.rows); \
    const bfloat b = full_attention_gate_value<QHeads, KHeads, HeadDim, Gate>(packed, attention, element + 1, params.rows); \
    hidden[element] = a; hidden[element + 1] = b; \
    const uint row = element / width; \
    gguf_sg::Packed::write(plane + ulong(row / 8) * width * 8, \
                           exponents + ulong(row / 8) * gguf_sg::Packed::sums_per_tile(width), \
                           width, (element % width) / 64, row % 8, lane, a, b); \
  }
ATTENTION_GATE_PACKED(verify_attention_gate_packed, 24, 4, 256, true)
ATTENTION_GATE_PACKED(verify_attention_gate_packed_kv4_g4, 16, 4, 256, true)
ATTENTION_GATE_PACKED(verify_attention_gate_packed_kv2_g8, 16, 2, 256, true)
ATTENTION_GATE_PACKED(verify_attention_gather_packed_hd128, 16, 2, 128, false)
ATTENTION_GATE_PACKED(verify_attention_gather_packed_hd64, 32, 8, 64, false)
ATTENTION_GATE_PACKED(verify_attention_gather_packed_k8q5d64, 40, 8, 64, false)
ATTENTION_GATE_PACKED(verify_attention_gather_packed_k8q4d128, 32, 8, 128, false)
#undef ATTENTION_GATE_PACKED

// The no-gate gather of a Plain-input out-projection: attention rows as
// hidden, no packed gate or sigmoid.
#define VERIFY_ATTENTION_GATHER(Name, QHeads, KHeads, HeadDim) \
  kernel void Name( \
      device const bfloat *packed_qkv [[buffer(0)]], \
      device const bfloat *attention [[buffer(1)]], \
      device bfloat *hidden [[buffer(2)]], \
      constant FullDecodeBatchParams &params [[buffer(3)]], \
      uint index [[thread_position_in_grid]], \
      uint grid_size [[threads_per_grid]]) { \
    full_attention_gate_decode_phase<QHeads, KHeads, HeadDim, false>( \
        packed_qkv, attention, hidden, params, index, grid_size); \
  }
VERIFY_ATTENTION_GATHER(verify_attention_gather_hd128, 16, 2, 128)
VERIFY_ATTENTION_GATHER(verify_attention_gather_hd64, 32, 8, 64)
VERIFY_ATTENTION_GATHER(verify_attention_gather_k8q5d64, 40, 8, 64)
VERIFY_ATTENTION_GATHER(verify_attention_gather_k8q4d128, 32, 8, 128)
#undef VERIFY_ATTENTION_GATHER

// Gemma 4 26B-A4B's verify QKV phases, no query gate and per-head QK norms:
// the sliding-window layers' KV8 group 2 of 256 (full rotary of 128 pairs,
// theta 1e4, scaleless-normed V — NormalizeValues) and the global layers'
// KV2 group 8 of 512 (p-RoPE of 64 pairs, theta 1e6, k_eq_v — no V in the
// packed row; the V slot receives the scale-free RMS of the pre-norm K).
// The rope tables are bound per layer type (buffers 3/4); the hd512
// threadgroup is 512 threads.
template <uint QHeads, uint KHeads, uint HeadDim, uint RotaryPairs,
          bool KeyEqualsValue, bool NormalizeValues, class W,
          bool ProportionalRope = false>
inline void full_qkv_decode_phase_gemma(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *keys, device bfloat *values, threadgroup float *reductions,
    threadgroup bfloat *normalized, constant FullDecodeBatchParams &params,
    uint2 group, uint thread_index, uint lane,
    uint simd_group) {
  constexpr uint PackedStride = QHeads * HeadDim +
                                (KeyEqualsValue ? 1 : 2) * KHeads * HeadDim;
  constexpr uint Stride = RICHENGINE_VERIFY_CHUNK_STRIDE;
  const uint rows = params.rows;
  uint batch = group.y;
  const ulong kv_lane_stride = ulong(KHeads) * Stride * HeadDim;
  FullPrefillParams lane_params{rows, Stride};
  full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, false, true,
                         bfloat, KeyEqualsValue, NormalizeValues,
                         ProportionalRope>(
      qkv + ulong(batch) * rows * PackedStride, q_norm, k_norm,
      rope_cos + ulong(batch) * rows * RotaryPairs,
      rope_sin + ulong(batch) * rows * RotaryPairs,
      queries + ulong(batch) * QHeads * Stride * HeadDim,
      keys + ulong(batch) * kv_lane_stride,
      values + ulong(batch) * kv_lane_stride, lane_params, reductions,
      normalized, group.x, thread_index, lane, simd_group);
}

kernel void verify_attention_qkv_gemma_h256(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *q_norm [[buffer(1)]],
    device const bfloat *k_norm [[buffer(2)]],
    device const float *rope_cos [[buffer(3)]],
    device const float *rope_sin [[buffer(4)]],
    device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]],
    device bfloat *values [[buffer(7)]],
    constant FullDecodeBatchParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  threadgroup bfloat normalized[256];
  full_qkv_decode_phase_gemma<16, 8, 256, 128, false, true, bfloat>(
      qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values,
      reductions, normalized, params, group, thread_index, lane, simd_group);
}

kernel void verify_attention_qkv_gemma_hd512(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *q_norm [[buffer(1)]],
    device const bfloat *k_norm [[buffer(2)]],
    device const float *rope_cos [[buffer(3)]],
    device const float *rope_sin [[buffer(4)]],
    device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]],
    device bfloat *values [[buffer(7)]],
    constant FullDecodeBatchParams &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[16];
  threadgroup bfloat normalized[512];
  full_qkv_decode_phase_gemma<16, 2, 512, 64, true, false, bfloat, true>(
      qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values,
      reductions, normalized, params, group, thread_index, lane, simd_group);
}

// The no-gate gathers and affine-input table variants of both Gemma shapes.
#define VERIFY_ATTENTION_GATHER(Name, QHeads, KHeads, HeadDim) \
  kernel void Name( \
      device const bfloat *packed_qkv [[buffer(0)]], \
      device const bfloat *attention [[buffer(1)]], \
      device bfloat *hidden [[buffer(2)]], \
      constant FullDecodeBatchParams &params [[buffer(3)]], \
      uint index [[thread_position_in_grid]], \
      uint grid_size [[threads_per_grid]]) { \
    full_attention_gate_decode_phase<QHeads, KHeads, HeadDim, false>( \
        packed_qkv, attention, hidden, params, index, grid_size); \
  }
VERIFY_ATTENTION_GATHER(verify_attention_gather_gemma_h256, 16, 8, 256)
VERIFY_ATTENTION_GATHER(verify_attention_gather_gemma_hd512, 16, 2, 512)
#undef VERIFY_ATTENTION_GATHER

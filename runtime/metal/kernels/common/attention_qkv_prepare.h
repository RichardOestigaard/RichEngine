#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

// q_norm and k_norm are read in their stored type W: bfloat in the packed
// formats, float for a GGUF's F32 norms. HeadDim is the per-head dimension,
// RotaryPairs the rotated dimension pairs of each head (headDim/2 for the
// full-rotary variants), QueryGate whether every query head is stored as
// [query | gate] pairs (the hybrid families) or plain rows, Norms whether
// the per-head RMS norms run at all (the dense target has none), eps their
// epsilon.
// KeyEqualsValue (Gemma 4's k_eq_v global layers): the packed row carries no
// V region, and the V slot of chunk_values instead receives the scale-free
// RMS norm of the head's pre-norm K — element * inverse, the value before the
// learned k_norm scale multiplies in. NormalizeValues (Gemma 4's sliding
// layers): the packed row does carry V, and the stored value is its
// scale-free RMS norm. Both leave the layout and every other entry unchanged.
// ProportionalRope (HF rope_type "proportional"): the first RotaryPairs
// angles rotate as full-head NeoX pairs (i, i + HeadDim/2) — not the
// contiguous rotated-section pairs (i, i + RotaryPairs) a default partial
// rotary (Qwen's 256/32) uses.
template <uint QHeads, uint KHeads, uint HeadDim = 256, uint RotaryPairs = 32,
          bool QueryGate = true, bool Norms = true, class W = bfloat,
          bool KeyEqualsValue = false, bool NormalizeValues = false,
          bool ProportionalRope = false>
inline void full_qkv_storage_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *chunk_keys, device bfloat *chunk_values,
    FullPrefillParams params, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint task, uint thread_index, uint lane,
    uint simd_group, float eps = kRmsEpsilon) {
  static_assert(QHeads % KHeads == 0);
  static_assert(RotaryPairs * 2 <= HeadDim);
  static_assert(!(KeyEqualsValue && NormalizeValues));
  constexpr uint QStride = QueryGate ? 2 * HeadDim : HeadDim;
  constexpr uint PackedStride =
      QHeads * QStride + (KeyEqualsValue ? 1 : 2) * KHeads * HeadDim;
  constexpr uint QWidth = QHeads * QStride, KWidth = KHeads * HeadDim;
  constexpr uint Simdgroups = HeadDim / 32;
  uint query_tasks = params.tokens * QHeads;
  bool query = task < query_tasks;
  uint local_task = query ? task : task - query_tasks;
  uint heads = query ? QHeads : KHeads;
  uint row = local_task / heads;
  uint head_index = local_task % heads;
  uint position = row;
  device const bfloat *source =
      query ? qkv + ulong(row) * PackedStride + head_index * QStride
            : qkv + ulong(row) * PackedStride + QWidth + head_index * HeadDim;
  device const W *weight = query ? q_norm : k_norm;
  uint kv_head = head_index / (QHeads / KHeads);
  uint local_head = head_index % (QHeads / KHeads);
  device bfloat *destination =
      queries + ((ulong(kv_head) * params.stride + row) *
                     (QHeads / KHeads) +
                 local_head) *
                    HeadDim;
  if (!query) {
    ulong key_offset =
        (ulong(head_index) * params.stride + position) * HeadDim;
    destination = chunk_keys + key_offset;
  }

  float element = float(source[thread_index]);
  float inverse = 1.0f, scale = 1.0f;
  if constexpr (Norms) {
    inverse = rms_inverse_of_sums<Simdgroups>(element * element, HeadDim,
                                            reductions, thread_index, lane,
                                            simd_group, eps);
    scale = float(weight[thread_index]);
  } else {
    // Keep the callers' barrier count identical either way: no reduction
    // runs without norms, so no barrier is needed before the rope read.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  normalized[thread_index] = bfloat(element * inverse * scale);
  if (!query) {
    ulong value_offset =
        (ulong(head_index) * HeadDim + thread_index) * params.stride +
        position;
    if constexpr (KeyEqualsValue) {
      // With Norms, `inverse` is already the pre-norm K's inverse RMS;
      // without them a second reduction of the same elements computes it.
      float value_inverse = inverse;
      if constexpr (!Norms) {
        value_inverse = rms_inverse_of_sums<Simdgroups>(
            element * element, HeadDim, reductions, thread_index, lane,
            simd_group, eps);
      }
      chunk_values[value_offset] = bfloat(element * value_inverse);
    } else if constexpr (NormalizeValues) {
      const float value = float(source[KWidth + thread_index]);
      const float value_inverse = rms_inverse_of_sums<Simdgroups>(
          value * value, HeadDim, reductions, thread_index, lane, simd_group,
          eps);
      chunk_values[value_offset] = bfloat(value * value_inverse);
    } else {
      chunk_values[value_offset] = source[KWidth + thread_index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr uint partner =
      ProportionalRope ? HeadDim / 2 : RotaryPairs;
  if (thread_index < RotaryPairs) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + partner]);
    float cosine = rope_cos[ulong(row) * RotaryPairs + thread_index];
    float sine = rope_sin[ulong(row) * RotaryPairs + thread_index];
    destination[thread_index] = bfloat(first * cosine - second * sine);
    destination[thread_index + partner] =
        bfloat(second * cosine + first * sine);
  } else if (ProportionalRope
                 ? ((thread_index >= RotaryPairs &&
                     thread_index < HeadDim / 2) ||
                    thread_index >= HeadDim / 2 + RotaryPairs)
                 : (thread_index >= 2 * RotaryPairs)) {
    destination[thread_index] = normalized[thread_index];
  }
}

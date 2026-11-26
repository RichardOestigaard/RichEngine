#include "metal/abi/KernelABI.h"
#include "metal/abi/LiLiCorr.h"
#include "metal/kernels/common/sgmatrix.h"

// LiLiCorr's lattice scorer. The matrices are small and bias-heavy, so these
// kernels read the draft image's row-major f32 and nvfp4 tensors directly
// instead of the engine's tile projection machinery.

inline device float *lilicorr_section(device uchar *scratch, uint lane,
                                      uint lane_stride, uint offset) {
  return reinterpret_cast<device float *>(scratch + ulong(lane) * lane_stride + offset);
}

inline device const float *lilicorr_section(device const uchar *scratch, uint lane,
                                            uint lane_stride, uint offset) {
  return reinterpret_cast<device const float *>(scratch + ulong(lane) * lane_stride + offset);
}

inline device uint *lilicorr_tokens(device uchar *scratch, uint lane, uint lane_stride) {
  return reinterpret_cast<device uint *>(scratch + ulong(lane) * lane_stride + LILICORR_SCRATCH_TOKENS);
}

inline float lilicorr_ue4m3_half(uchar x) {
  if (x == 0 || x == 0x7f)
    return 0.0f;
  const uint exponent = (x >> 3) & 0xf;
  const uint mantissa = x & 7;
  const float raw = exponent
      ? ldexp(1.0f + float(mantissa) / 8.0f, int(exponent) - 7)
      : ldexp(float(mantissa), -9);
  return raw * 0.5f;
}

inline float lilicorr_fp4(uint code) {
  constexpr float values[16] = {
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return values[code & 15];
}

inline float lilicorr_nvfp4_row(device const uchar *row, uint column,
                                float global_scale) {
  const uint block = column >> 6;
  const uint element = column & 63;
  const uint sub = element >> 4;
  const uint within = element & 15;
  device const uchar *b = row + block * 36;
  const uchar packed = b[4 + sub * 8 + (within & 7)];
  const uint code = within < 8 ? packed & 15 : packed >> 4;
  return lilicorr_fp4(code) * lilicorr_ue4m3_half(b[sub]) * global_scale;
}

inline float lilicorr_nvfp4(device const uchar *weight, uint row, uint column,
                            uint input_size, float global_scale) {
  const uint blocks = input_size / 64;
  return lilicorr_nvfp4_row(weight + ulong(row) * blocks * 36, column,
                            global_scale);
}

inline float lilicorr_silu(float x) {
  return x / (1.0f + exp(-x));
}

// Full-vocabulary softmax statistics and the five rank features of one
// (lane, slot). `candidates` is the selector's sorted top-16 in the
// draft_select_plain layout — fifteen positions per lane of sixteen
// candidates each, slot s at position s — and LiLiCorr uses its first
// eight. Slot s reads draft row s + 1, as llama.cpp's view offset and the
// engine's DFlash row offset both do.
kernel void lilicorr_features(
    device const float *logits [[buffer(0)]],
    device const uint *candidates [[buffer(1)]],
    device float *probabilities [[buffer(2)]],
    device uchar *scratch [[buffer(3)]],
    constant LiLiCorrFeatureParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Slots = LILICORR_SLOTS;
  constexpr uint TopK = LILICORR_TOP_K;
  constexpr uint Candidates = RICHENGINE_DRAFT_CANDIDATES;
  const uint lane = group / Slots;
  const uint slot = group % Slots;
  if (lane >= params.lanes)
    return;
  const uint candidate_row = lane * RICHENGINE_DRAFT_PROPOSAL_TOKENS + slot;
  device const uint *row_candidates = candidates + ulong(candidate_row) * Candidates;
  const ulong logits_row = (ulong(lane) * params.query_rows + slot + 1) * params.logits_stride;
  device const float *row = logits + logits_row;
  const float maximum = row[row_candidates[0]];
  float partial = 0.0f;
  for (uint token = thread_index; token < params.vocabulary; token += 256)
    partial += exp(row[token] - maximum);
  partial = simd_sum(partial);
  threadgroup float partials[8];
  threadgroup float top_probability;
  if (lane_in_simd == 0)
    partials[simd_group] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  if (thread_index == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < 8; ++i)
      sum += partials[i];
    top_probability = exp(row[row_candidates[0]] - maximum) / sum;
    partials[0] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  if (thread_index >= TopK)
    return;
  const uint rank = thread_index;
  const uint token = row_candidates[rank];
  const float sum = partials[0];
  const float probability = exp(row[token] - maximum) / sum;
  const float lp = log(probability);
  const float top_lp = log(top_probability);
  const uint node = slot * TopK + rank;
  device uint *node_tokens = lilicorr_tokens(scratch, lane, LILICORR_SCRATCH_STRIDE);
  node_tokens[node] = token;
  device float *features = lilicorr_section(scratch, lane, LILICORR_SCRATCH_STRIDE,
                                            LILICORR_SCRATCH_FEATURES);
  device float *f = features + node * LILICORR_FEATURE_WIDTH;
  f[0] = lp;
  f[1] = probability;
  f[2] = lp - top_lp;
  f[3] = float(rank) / float(TopK - 1);
  f[4] = rank == 0 ? 1.0f : 0.0f;
  device float *out_probability = probabilities + ulong(candidate_row) * Candidates;
  out_probability[rank] = probability;
  if (thread_index < Candidates - TopK)
    out_probability[TopK + thread_index] = 0.0f;
}

// Gather the candidate token embeddings in fp32 and copy the pass rows out of
// the draft's bf16 final hidden state. The target table is GGUF q8_0 rows.
kernel void lilicorr_gather_inputs(
    device const uchar *embedding [[buffer(0)]],
    device const bfloat *final_hidden [[buffer(1)]],
    device uchar *scratch [[buffer(2)]],
    constant LiLiCorrFeatureParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  constexpr uint Hidden = LILICORR_TARGET_HIDDEN;
  const uint node_rows = params.lanes * LILICORR_NODES;
  const ulong node_elements = ulong(node_rows) * Hidden;
  if (index < node_elements) {
    const uint row = index / Hidden;
    const uint dim = index % Hidden;
    const uint lane = row / LILICORR_NODES;
    const uint node = row % LILICORR_NODES;
    device const uint *tokens = reinterpret_cast<device const uint *>(
        scratch + ulong(lane) * LILICORR_SCRATCH_STRIDE + LILICORR_SCRATCH_TOKENS);
    const uint token = min(tokens[node], params.vocabulary - 1);
    device const uchar *block = embedding + (ulong(token) * (Hidden / 32) + dim / 32) * 34;
    const half scale = as_type<half>(ushort(block[0] | block[1] << 8));
    const float value = float(scale) * float(as_type<char>(block[2 + dim % 32]));
    lilicorr_section(scratch, lane, LILICORR_SCRATCH_STRIDE,
                     LILICORR_SCRATCH_TOKEN_INPUT)[node * Hidden + dim] = value;
    return;
  }
  const ulong pass_index = ulong(index) - node_elements;
  const ulong pass_elements = ulong(params.lanes) * LILICORR_SLOTS * Hidden;
  if (pass_index >= pass_elements)
    return;
  const uint row = uint(pass_index / Hidden);
  const uint dim = uint(pass_index % Hidden);
  const uint lane = row / LILICORR_SLOTS;
  const uint slot = row % LILICORR_SLOTS;
  const ulong source = (ulong(lane) * params.query_rows + slot + 1) * Hidden + dim;
  lilicorr_section(scratch, lane, LILICORR_SCRATCH_STRIDE,
                   LILICORR_SCRATCH_PASS_INPUT)[row % LILICORR_SLOTS * Hidden + dim] =
      float(final_hidden[source]);
}

// row = linear % rows keeps the threads of one output column adjacent: they
// read the same weight element together, while the small input rows hit L2.
inline void lilicorr_gemm_f32_body(device const uchar *scratch,
                                   device const float *weight,
                                   device const float *bias,
                                   device uchar *output_scratch,
                                   constant LiLiCorrGemmParams &params,
                                   uint index) {
  if (index >= params.rows * params.output_size)
    return;
  const uint row = index % params.rows;
  const uint column = index / params.rows;
  const uint lane = row / params.rows_per_lane;
  const uint local = row % params.rows_per_lane;
  device const float *input = lilicorr_section(scratch, lane, params.lane_stride,
                                               params.input_stride) +
                              ulong(local) * params.input_size;
  device const float *w = weight + ulong(column) * params.input_size;
  float sum = bias ? bias[column] : 0.0f;
  for (uint k = 0; k < params.input_size; ++k)
    sum += input[k] * w[k];
  if (params.epilogue == 1)
    sum = lilicorr_silu(sum);
  lilicorr_section(output_scratch, lane, params.lane_stride,
                   params.output_stride)[ulong(local) * params.output_size + column] = sum;
}

kernel void lilicorr_gemm_f32(
    device const uchar *scratch [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device uchar *output_scratch [[buffer(3)]],
    constant LiLiCorrGemmParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  lilicorr_gemm_f32_body(scratch, weight, bias, output_scratch, params, index);
}

kernel void lilicorr_gemm_nvfp4(
    device const uchar *scratch [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device uchar *output_scratch [[buffer(3)]],
    constant LiLiCorrGemmParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.rows * params.output_size)
    return;
  const uint row = index % params.rows;
  const uint column = index / params.rows;
  const uint lane = row / params.rows_per_lane;
  const uint local = row % params.rows_per_lane;
  device const float *input = lilicorr_section(scratch, lane, params.lane_stride,
                                               params.input_stride) +
                              ulong(local) * params.input_size;
  float sum = bias ? bias[column] : 0.0f;
  for (uint k = 0; k < params.input_size; ++k)
    sum += input[k] * lilicorr_nvfp4(weight, column, k, params.input_size, params.scale);
  if (params.epilogue == 1)
    sum = lilicorr_silu(sum);
  lilicorr_section(output_scratch, lane, params.lane_stride,
                   params.output_stride)[ulong(local) * params.output_size + column] = sum;
}

inline float2 lilicorr_mma_weight2(device const float *weight, uint column,
                                   uint input_size, uint input_index, float) {
  device const float *w = weight + ulong(column) * input_size + input_index;
  return float2(w[0], w[input_size]);
}

inline float2 lilicorr_mma_weight2(device const uchar *weight, uint column,
                                   uint input_size, uint input_index,
                                   float scale) {
  const uint row_bytes = (input_size / 64) * 36;
  device const uchar *w = weight + ulong(column) * row_bytes;
  return float2(lilicorr_nvfp4_row(w, input_index, scale),
                lilicorr_nvfp4_row(w + row_bytes, input_index, scale));
}

// The wide projections on the GPU's fp32 8x8 MMA. One threadgroup covers
// eight output columns and all rows of one lane; its eight simdgroups split
// K. Lane (fm, fn) holds x[row fm][k + fn, k + fn + 1] as A, the matching
// elements of output columns fn and fn + 1 as B, and their destination as
// C. The weight operand is loaded once per step and reused by all seven
// eight-row fragments, so the NVFP4 projections decode each weight once per
// lane rather than once for every one of the lane's 56 rows.
template <typename W>
inline void lilicorr_gemm_mma_body(device const uchar *scratch,
                                   device const W *weight,
                                   device const float *bias,
                                   device uchar *output_scratch,
                                   constant LiLiCorrGemmParams &params,
                                   uint2 group, uint simd_group, uint lane_in_simd,
                                   threadgroup float2 *parts) {
  constexpr uint Splits = 8;
  constexpr uint Fragments = 7;
  constexpr uint Rows = 8 * Fragments;
  const uint live = min(params.rows_per_lane, Rows);
  const uint input_size = params.input_size;
  device const float *input = lilicorr_section(scratch, group.y, params.lane_stride,
                                               params.input_stride);
  device float *output = lilicorr_section(output_scratch, group.y, params.lane_stride,
                                          params.output_stride);
  const sgmatrix::Lane l = sgmatrix::lane_map(lane_in_simd);
  const uint column = group.x * 8 + l.fn;
  const uint steps = input_size / 8;
  const uint first = simd_group * steps / Splits;
  const uint last = (simd_group + 1) * steps / Splits;
  float2 acc[Fragments];
#pragma unroll
  for (uint f = 0; f < Fragments; ++f)
    acc[f] = float2(0);
#pragma unroll(4)
  for (uint step = first; step < last; ++step) {
    const uint k = step * 8;
    const float2 b = lilicorr_mma_weight2(weight, column, input_size,
                                          k + l.fm, params.scale);
#pragma unroll
    for (uint f = 0; f < Fragments; ++f) {
      if (f * 8 >= live)
        break;
      const uint row = f * 8 + l.fm;
      float2 a = float2(0);
      if (row < live)
        a = *reinterpret_cast<device const float2 *>(
            input + ulong(row) * input_size + k + l.fn);
      sgmatrix::mma_acc<float>(acc[f], a, b);
    }
  }
#pragma unroll
  for (uint f = 0; f < Fragments; ++f)
    parts[(simd_group * Fragments + f) * 32 + lane_in_simd] = acc[f];
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  const uint row_fragment = simd_group;
  if (row_fragment >= Fragments || row_fragment * 8 >= live)
    return;
  const uint row = row_fragment * 8 + l.fm;
  if (row >= live)
    return;
  float2 total = float2(0);
#pragma unroll
  for (uint split = 0; split < Splits; ++split)
    total += parts[(split * Fragments + row_fragment) * 32 + lane_in_simd];
  if (bias)
    total += float2(bias[column], bias[column + 1]);
  if (params.epilogue == 1) {
    total.x = lilicorr_silu(total.x);
    total.y = lilicorr_silu(total.y);
  }
  device float *out = output + ulong(row) * params.output_size + column;
  out[0] = total.x;
  out[1] = total.y;
}

kernel void lilicorr_gemm_f32_mma(
    device const uchar *scratch [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device uchar *output_scratch [[buffer(3)]],
    constant LiLiCorrGemmParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]]) {
  threadgroup float2 parts[8 * 7 * 32];
  lilicorr_gemm_mma_body(scratch, weight, bias, output_scratch, params,
                         group, simd_group, lane_in_simd, parts);
}

kernel void lilicorr_gemm_nvfp4_mma(
    device const uchar *scratch [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device uchar *output_scratch [[buffer(3)]],
    constant LiLiCorrGemmParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]]) {
  threadgroup float2 parts[8 * 7 * 32];
  lilicorr_gemm_mma_body(scratch, weight, bias, output_scratch, params,
                         group, simd_group, lane_in_simd, parts);
}

// ggml_norm over the five candidate features: layer normalization, then the
// feature affine row. One thread handles one node.
kernel void lilicorr_feature_norm(
    device const float *weight [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device uchar *scratch [[buffer(2)]],
    constant LiLiCorrNormParams &params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= params.rows)
    return;
  constexpr uint Width = LILICORR_FEATURE_WIDTH;
  const uint lane = row / LILICORR_NODES;
  const uint local = row % LILICORR_NODES;
  device float *features = lilicorr_section(scratch, lane, params.lane_stride,
                                            LILICORR_SCRATCH_FEATURES);
  device float *output = lilicorr_section(scratch, lane, params.lane_stride,
                                          LILICORR_SCRATCH_FEATURE_NORMED);
  device const float *x = features + local * Width;
  float mean = 0.0f;
  for (uint i = 0; i < Width; ++i)
    mean += x[i];
  mean /= Width;
  float variance = 0.0f;
  for (uint i = 0; i < Width; ++i) {
    const float d = x[i] - mean;
    variance += d * d;
  }
  variance /= Width;
  const float scale = 1.0f / sqrt(variance + params.epsilon);
  for (uint i = 0; i < Width; ++i)
    output[local * Width + i] = (x[i] - mean) * scale * weight[i] + bias[i];
}

// RMS normalization of one row per threadgroup. Input and output may be the
// same rows: every thread finishes its input reads before the reduction
// barrier, and writes only its own elements.
kernel void lilicorr_rmsnorm(
    device const float *weight [[buffer(0)]],
    device uchar *scratch [[buffer(1)]],
    constant LiLiCorrNormParams &params [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (row >= params.rows)
    return;
  const uint lane = row / params.rows_per_lane;
  const uint local = row % params.rows_per_lane;
  device const float *input = lilicorr_section(scratch, lane, params.lane_stride,
                                               params.input_stride) +
                              ulong(local) * params.width;
  device float *output = lilicorr_section(scratch, lane, params.lane_stride,
                                          params.output_stride) +
                         ulong(local) * params.width;
  float partial = 0.0f;
  for (uint i = thread_index; i < params.width; i += 256) {
    const float x = input[i];
    partial += x * x;
  }
  partial = simd_sum(partial);
  threadgroup float partials[8];
  if (lane_in_simd == 0)
    partials[simd_group] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  if (thread_index == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < 8; ++i)
      sum += partials[i];
    partials[0] = 1.0f / sqrt(sum / float(params.width) + params.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  const float scale = partials[0];
  for (uint i = thread_index; i < params.width; i += 256)
    output[i] = input[i] * scale * weight[i];
}

// The scorer node sum: token state, the slot's broadcast pass state, feature
// state, and the learned slot/rank embeddings.
kernel void lilicorr_assemble_nodes(
    device const float *slot_embedding [[buffer(0)]],
    device const float *rank_embedding [[buffer(1)]],
    device uchar *scratch [[buffer(2)]],
    constant LiLiCorrNormParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  const uint rows = params.rows;
  const uint elements = rows * LILICORR_HIDDEN;
  if (index >= elements)
    return;
  const uint row = index / LILICORR_HIDDEN;
  const uint dim = index % LILICORR_HIDDEN;
  const uint lane = row / LILICORR_NODES;
  const uint node = row % LILICORR_NODES;
  const uint slot = node / LILICORR_TOP_K;
  const uint rank = node % LILICORR_TOP_K;
  device const float *token = lilicorr_section(scratch, lane, params.lane_stride,
                                               LILICORR_SCRATCH_TOKEN_STATES);
  device const float *pass = lilicorr_section(scratch, lane, params.lane_stride,
                                              LILICORR_SCRATCH_PASS_STATES);
  device const float *feature = lilicorr_section(scratch, lane, params.lane_stride,
                                                 LILICORR_SCRATCH_FEATURE_STATES);
  device float *x = lilicorr_section(scratch, lane, params.lane_stride,
                                     LILICORR_SCRATCH_X);
  const uint at = node * LILICORR_HIDDEN + dim;
  x[at] = token[at] + pass[slot * LILICORR_HIDDEN + dim] + feature[at] +
          slot_embedding[slot * LILICORR_HIDDEN + dim] +
          rank_embedding[rank * LILICORR_HIDDEN + dim];
}

kernel void lilicorr_residual(
    device uchar *scratch [[buffer(0)]],
    constant LiLiCorrGemmParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  const uint elements = params.rows * params.input_size;
  if (index >= elements)
    return;
  const uint row = index / params.input_size;
  const uint dim = index % params.input_size;
  const uint lane = row / params.rows_per_lane;
  const uint local = row % params.rows_per_lane;
  device float *x = lilicorr_section(scratch, lane, params.lane_stride,
                                     params.output_stride) +
                    ulong(local) * params.input_size + dim;
  device const float *delta = lilicorr_section(scratch, lane, params.lane_stride,
                                               params.input_stride) +
                              ulong(local) * params.input_size + dim;
  x[0] += delta[0];
}

// One threadgroup per (lane, head, query node). Dense 56-node attention with
// the learned additive bias; no rope and no causal mask.
kernel void lilicorr_attention(
    device const float *bias [[buffer(0)]],
    device uchar *scratch [[buffer(1)]],
    constant LiLiCorrAttentionParams &params [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Nodes = LILICORR_NODES;
  constexpr uint Hidden = LILICORR_HIDDEN;
  constexpr uint HeadDim = LILICORR_HEAD_DIM;
  constexpr uint Qkv = 3 * Hidden;
  const uint lane = group / (LILICORR_HEADS * Nodes);
  const uint remainder = group % (LILICORR_HEADS * Nodes);
  const uint head = remainder / Nodes;
  const uint query = remainder % Nodes;
  if (lane >= params.lanes || thread_index >= HeadDim)
    return;
  device const float *qkv = lilicorr_section(scratch, lane, params.lane_stride,
                                             LILICORR_SCRATCH_QKV);
  device const float *q = qkv + ulong(query) * Qkv + head * HeadDim + thread_index;
  threadgroup float partials[4];
  threadgroup float scores[Nodes];
  for (uint key = 0; key < Nodes; ++key) {
    device const float *k = qkv + ulong(key) * Qkv + Hidden + head * HeadDim + thread_index;
    float part = q[0] * k[0];
    part = simd_sum(part);
    if (lane_in_simd == 0)
      partials[simd_group] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    if (thread_index == 0) {
      const float dot = partials[0] + partials[1] + partials[2] + partials[3];
      scores[key] = dot * params.scale + bias[key + Nodes * query + Nodes * Nodes * head];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  }
  if (thread_index == 0) {
    float maximum = scores[0];
    for (uint key = 1; key < Nodes; ++key)
      maximum = max(maximum, scores[key]);
    float sum = 0.0f;
    for (uint key = 0; key < Nodes; ++key) {
      scores[key] = exp(scores[key] - maximum);
      sum += scores[key];
    }
    partials[0] = 1.0f / sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  const float inverse = partials[0];
  float output = 0.0f;
  for (uint key = 0; key < Nodes; ++key) {
    device const float *v = qkv + ulong(key) * Qkv + 2 * Hidden + head * HeadDim + thread_index;
    output += scores[key] * inverse * v[0];
  }
  lilicorr_section(scratch, lane, params.lane_stride,
                   LILICORR_SCRATCH_ATTENTION_OUT)[query * Hidden + head * HeadDim + thread_index] = output;
}

kernel void lilicorr_factor_input(
    device uchar *scratch [[buffer(0)]],
    constant LiLiCorrNormParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  const uint elements = params.rows * LILICORR_HIDDEN;
  if (index >= elements)
    return;
  const uint row = index / LILICORR_HIDDEN;
  const uint dim = index % LILICORR_HIDDEN;
  const uint lane = row / LILICORR_NODES;
  const uint node = row % LILICORR_NODES;
  device const float *x = lilicorr_section(scratch, lane, params.lane_stride,
                                           LILICORR_SCRATCH_X);
  device const float *anchor = lilicorr_section(scratch, lane, params.lane_stride,
                                                LILICORR_SCRATCH_ANCHOR_NORMED);
  device float *factor = lilicorr_section(scratch, lane, params.lane_stride,
                                          LILICORR_SCRATCH_FACTOR_INPUT);
  const float xv = x[node * LILICORR_HIDDEN + dim];
  const float av = anchor[dim];
  factor[node * 3 * LILICORR_HIDDEN + dim] = xv;
  factor[node * 3 * LILICORR_HIDDEN + LILICORR_HIDDEN + dim] = av;
  factor[node * 3 * LILICORR_HIDDEN + 2 * LILICORR_HIDDEN + dim] = xv * av;
}

// In-place L2 normalization, one row per threadgroup.
kernel void lilicorr_l2norm(
    device uchar *scratch [[buffer(0)]],
    constant LiLiCorrNormParams &params [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane_in_simd [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (row >= params.rows)
    return;
  const uint lane = row / params.rows_per_lane;
  const uint local = row % params.rows_per_lane;
  device float *x = lilicorr_section(scratch, lane, params.lane_stride,
                                     params.input_stride) +
                    ulong(local) * params.width;
  float partial = 0.0f;
  for (uint i = thread_index; i < params.width; i += 256) {
    const float v = x[i];
    partial += v * v;
  }
  partial = simd_sum(partial);
  threadgroup float partials[8];
  if (lane_in_simd == 0)
    partials[simd_group] = partial;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  if (thread_index == 0) {
    float sum = 0.0f;
    for (uint i = 0; i < 8; ++i)
      sum += partials[i];
    partials[0] = 1.0f / max(sqrt(sum), params.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  const float scale = partials[0];
  for (uint i = thread_index; i < params.width; i += 256)
    x[i] *= scale;
}

// The Markov walk. Only the predecessor reached by the walk is scored at
// each slot, which is output-identical to materializing the full lattice.
kernel void lilicorr_walk(
    device const uchar *scratch [[buffer(0)]],
    device uint *proposed [[buffer(1)]],
    device float *confidence [[buffer(2)]],
    constant LiLiCorrWalkParams &params [[buffer(3)]],
    uint batch [[thread_position_in_grid]]) {
  if (batch >= params.lanes)
    return;
  device const float *in = lilicorr_section(scratch, batch, params.lane_stride,
                                            LILICORR_SCRATCH_IN);
  device const float *out = lilicorr_section(scratch, batch, params.lane_stride,
                                             LILICORR_SCRATCH_OUT);
  device const float *anchor = lilicorr_section(scratch, batch, params.lane_stride,
                                                LILICORR_SCRATCH_ANCHOR_OUT);
  device const uint *tokens = reinterpret_cast<device const uint *>(
      scratch + ulong(batch) * params.lane_stride + LILICORR_SCRATCH_TOKENS);
  uint predecessor = 0;
  for (uint slot = 0; slot < LILICORR_SLOTS; ++slot) {
    float scores[LILICORR_TOP_K];
    float best_score = -INFINITY;
    uint best = 0;
    for (uint rank = 0; rank < LILICORR_TOP_K; ++rank) {
      const uint node = slot * LILICORR_TOP_K + rank;
      device const float *left = in + node * LILICORR_HIDDEN;
      device const float *right = slot == 0
          ? anchor
          : out + ((slot - 1) * LILICORR_TOP_K + predecessor) * LILICORR_HIDDEN;
      float dot = 0.0f;
      for (uint dim = 0; dim < LILICORR_HIDDEN; ++dim)
        dot += left[dim] * right[dim];
      scores[rank] = dot * params.logit_scale;
      if (scores[rank] > best_score) {
        best_score = scores[rank];
        best = rank;
      }
    }
    float sum = 0.0f;
    for (uint rank = 0; rank < LILICORR_TOP_K; ++rank)
      sum += exp(scores[rank] - best_score);
    predecessor = best;
    const uint node = slot * LILICORR_TOP_K + best;
    proposed[batch * RICHENGINE_DRAFT_PROPOSAL_TOKENS + slot] = tokens[node];
    confidence[batch * RICHENGINE_DRAFT_PROPOSAL_TOKENS + slot] = 1.0f / sum;
  }
}

// Stage one state lane's bf16 anchor row as fp32 in the dispatch lane's
// scorer scratch. params.source_row carries the dispatch lane here.
kernel void lilicorr_stage_anchor(
    device const bfloat *source [[buffer(0)]],
    device uchar *scratch [[buffer(1)]],
    constant LiLiCorrCopyParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.width)
    return;
  lilicorr_section(scratch, params.source_row, LILICORR_SCRATCH_STRIDE,
                   LILICORR_SCRATCH_ANCHOR_INPUT)[index] = float(source[index]);
}

// Copy one bf16 row. The commit path uses retained - 1; the prefill path the
// prompt tail row. Output is the per-state lane's LiLiCorr anchor.
kernel void lilicorr_copy_row(
    device const bfloat *source [[buffer(0)]],
    device bfloat *destination [[buffer(1)]],
    constant LiLiCorrCopyParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.width)
    return;
  destination[index] = source[ulong(params.source_row) * params.width + index];
}

// Vision encoder kernels for the LFM3-VL tower (the lfm3vl clip projector of
// an LFM3-Omni mmproj): raster-order pixel patchify, axial-2D rope tables,
// RMSNorms, per-head q/k norm + rope + GQA scatter, non-causal flash
// attention at head_dim 64, the gated depthwise 3x3 conv block, the
// silu-gated FFN multiply and the projector's 2x2 merge gather.
//
// The dense GEMMs (m64n128 / m32n256 families) and the projector's input
// LayerNorm (eps 1e-6) are the shared vision_* kernels, bound by name; only
// the tower-specific passes live here. Token order everywhere is raster
// order (y, x) — unlike the Qwen tower's block-major order — and the 2x2
// interleave happens at lfm_vision_merge.

#include "metal/abi/Vision.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant ushort kLfmHeads = 16;      // query heads
constant ushort kLfmKvHeads = 8;     // key/value heads, duplicated 2x for GQA
constant ushort kLfmHeadDim = 64;    // per-head width; K/V tiles also 64
constant ushort kLfmHidden = 1024;
constant ushort kLfmPatch = 16;
constant ushort kLfmPatchDim = 1536; // 3 channels x 2 temporal x 16 x 16

constant float kLfmRopeTheta = 1000000.0f;
constant float kLfmNormEps = 1e-5f;

// Patchify: uint8 RGB pixels (H, W, 3) -> bf16 patch rows (tokens, 1536) in
// raster order. Feature order (channel, frame, row, col); a still image fills
// both temporal slots with the same pixels, a video pair's payload is frame 0
// then frame 1, one grid each, and each temporal slot reads its own frame.
// Normalization 0.5/0.5.

kernel void lfm_vision_patchify(
    device const uchar *pixels [[buffer(0)]],
    device bfloat *patches [[buffer(1)]],
    constant VisionPatchifyParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint image_width = params.grid_width * kLfmPatch;
    ulong frame_pixels =
        ulong(image_width) * params.grid_height * kLfmPatch;
    device bfloat *row = patches + ulong(token) * kLfmPatchDim;
    for (uint feature = thread_index; feature < kLfmPatchDim;
         feature += 256) {
        uint channel = feature / (2 * kLfmPatch * kLfmPatch);
        uint temporal = (feature / (kLfmPatch * kLfmPatch)) & 1;
        uint within = feature % (kLfmPatch * kLfmPatch);
        uint patch_row = within / kLfmPatch;
        uint patch_col = within % kLfmPatch;
        // A pair's slot 1 reads frame 1; a still payload has one frame, so
        // both slots read frame 0 as the HF temporal expand does.
        ulong frame = temporal < params.frames ? temporal : params.frames - 1;
        ulong pixel =
            frame * frame_pixels +
            (ulong(token / params.grid_width * kLfmPatch + patch_row) *
                 image_width +
             (token % params.grid_width) * kLfmPatch + patch_col) * 3 +
            channel;
        row[feature] = bfloat(float(pixels[pixel]) * (1.0f / 127.5f) - 1.0f);
    }
}

// Fused QKV projection: one m64n128 GEMM whose 2048-wide output rows are
// [Q 1024 | K 512 | V 512]. The sixteen column tiles pick their weight
// operand per tile — tiles 0-7 multiply the query weights, 8-11 the key
// weights, 12-15 the value weights — so no repack pass is needed and the
// narrow K/V projections stop paying a per-GEMM fixed cost. The output
// rows land across the contiguous Query/Key/Value scratch slots, and
// lfm_vision_qkv_prepare reads them with params.fused set.

kernel void lfm_vision_gemm_qkv(
    device bfloat *input [[buffer(0)]],
    device bfloat *query_weights [[buffer(1)]],
    device bfloat *key_weights [[buffer(2)]],
    device bfloat *value_weights [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant VisionGemmParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr ushort TileM = 64, TileN = 128, BK = 128;
    constexpr ushort q_tiles = kLfmHidden / TileN;            // 8
    constexpr ushort kv_tiles =
        (kLfmKvHeads * kLfmHeadDim) / TileN;                  // 4
    const uint tile = group.y;
    device bfloat *tile_weights;
    uint output_origin;
    if (tile < q_tiles) {
        tile_weights = query_weights + tile * TileN * params.input_size;
        output_origin = tile * TileN;
    } else if (tile < q_tiles + kv_tiles) {
        tile_weights = key_weights;
        output_origin = kLfmHidden + (tile - q_tiles) * TileN;
        tile_weights += (tile - q_tiles) * TileN * params.input_size;
    } else {
        tile_weights = value_weights;
        output_origin =
            kLfmHidden + kLfmKvHeads * kLfmHeadDim +
            (tile - q_tiles - kv_tiles) * TileN;
        tile_weights +=
            (tile - q_tiles - kv_tiles) * TileN * params.input_size;
    }
    const uint row_origin = group.x * TileM;
    device bfloat *rows = input + ulong(row_origin) * params.input_size;
    auto a = tensor(rows, dextents<int, 2>{int(params.input_size), TileM},
                    array<int, 2>{1, int(params.input_size)});
    auto b = tensor(tile_weights,
                    dextents<int, 2>{int(params.input_size), TileN},
                    array<int, 2>{1, int(params.input_size)});
    auto c = tensor(output + ulong(row_origin) * params.output_size,
                    dextents<int, 2>{int(params.output_size), TileM},
                    array<int, 2>{1, int(params.output_size)});
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> operation;
    auto a0 = a.template slice<BK, TileM>(0, 0);
    auto b0 = b.template slice<BK, TileN>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i)
        accumulated[i] = 0.0f;
    for (uint chunk = 0; chunk < params.input_size / BK; ++chunk) {
        auto a_slice = a.template slice<BK, TileM>(chunk * BK, 0);
        auto b_slice = b.template slice<BK, TileN>(chunk * BK, 0);
        operation.run(a_slice, b_slice, accumulated);
        threadgroup_barrier(mem_flags::mem_none);
    }
    auto converted = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), bfloat>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i)
        converted[i] = bfloat(accumulated[i]);
    converted.store(c.template slice<TileN, TileM>(output_origin, 0));
}

// Per-token setup: zero the padded residual rows and build the axial 2D rope
// tables — head_dim/4 = 16 frequencies per axis (y on dims 0-15 and 32-47,
// x on 16-31 and 48-63), theta 1e6 over spatial_dim = head_dim/2 = 32.

kernel void lfm_vision_prepare(
    device float *hidden [[buffer(0)]],
    device float *rope_cos [[buffer(1)]],
    device float *rope_sin [[buffer(2)]],
    constant VisionGridParams &params [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    if (token >= params.grid_height * params.grid_width) {
        device float *out = hidden + ulong(token) * kLfmHidden;
        for (uint dim = thread_index; dim < kLfmHidden; dim += 256)
            out[dim] = 0.0f;
        return;
    }
    if (thread_index < kLfmHeadDim) {
        constexpr uint frequencies = kLfmHeadDim / 4; // 16
        uint frequency = thread_index % frequencies;
        // cat([freq_h, freq_w]) duplicated over the 64-dim head: the dim's
        // quarter picks the axis, the frequency index its inverse frequency.
        bool x_axis = (thread_index / frequencies) & 1;
        uint position = x_axis ? token % params.grid_width
                               : token / params.grid_width;
        float inverse = pow(kLfmRopeTheta,
                            -float(2 * frequency) / float(kLfmHeadDim / 2));
        float angle = float(position) * inverse;
        rope_cos[ulong(token) * kLfmHeadDim + thread_index] = cos(angle);
        rope_sin[ulong(token) * kLfmHeadDim + thread_index] = sin(angle);
    }
}

// RMSNorm of bf16 rows into the fp32 residual stream (the patch embedding's
// norm; eps 1e-5, no bias).

kernel void lfm_vision_rmsnorm_embed(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant VisionNormParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    device const bfloat *row = input + ulong(group) * params.width;
    device float *out = output + ulong(group) * params.width;

    float squares = 0.0f;
    for (uint column = thread_index; column < params.width; column += 256) {
        float value = float(row[column]);
        squares += value * value;
    }
    squares = simd_sum(squares);
    if (simd_lane == 0) reductions[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) total += reductions[i];
    float inverse = rsqrt(total / float(params.width) + kLfmNormEps);

    for (uint column = thread_index; column < params.width; column += 256)
        out[column] = float(row[column]) * float(weight[column]) * inverse;
}

// RMSNorm of an fp32 residual row into bf16 (ln1/ln2 of every block).

kernel void lfm_vision_rmsnorm(
    device const float *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant VisionNormParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    device const float *row = input + ulong(group) * params.width;
    device bfloat *out = output + ulong(group) * params.width;

    float squares = 0.0f;
    for (uint column = thread_index; column < params.width; column += 256) {
        float value = row[column];
        squares += value * value;
    }
    squares = simd_sum(squares);
    if (simd_lane == 0) reductions[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) total += reductions[i];
    float inverse = rsqrt(total / float(params.width) + kLfmNormEps);

    for (uint column = thread_index; column < params.width; column += 256)
        out[column] = bfloat(row[column] * float(weight[column]) * inverse);
}

// QKV preparation, one 32-lane threadgroup per (token, head): per-head
// RMSNorm on q and k (dim 64, eps 1e-5), axial rope, and the scatter into
// per-head planes with the 8 kv heads duplicated onto the 16 query heads.
// Each lane owns rope pair (dim, dim+32), so both rotated halves are local.
// Padded tokens write zeros so whole 128-key tiles stay finite.

kernel void lfm_vision_qkv_prepare(
    device const bfloat *queries_in [[buffer(0)]],
    device const bfloat *keys_in [[buffer(1)]],
    device const bfloat *values_in [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device const float *rope_cos [[buffer(5)]],
    device const float *rope_sin [[buffer(6)]],
    device bfloat *queries [[buffer(7)]],
    device bfloat *keys [[buffer(8)]],
    device bfloat *values [[buffer(9)]],
    constant VisionQkvParams &params [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint token = group / kLfmHeads;
    uint head = group % kLfmHeads;
    constexpr ushort half_dim = kLfmHeadDim / 2;
    constexpr ushort kv_width = kLfmKvHeads * kLfmHeadDim;
    ulong plane = (ulong(head) * params.padded_tokens + token) * kLfmHeadDim;
    if (token >= params.tokens || lane >= half_dim) {
        if (lane < half_dim) {
            queries[plane + lane] = bfloat(0.0f);
            queries[plane + lane + half_dim] = bfloat(0.0f);
            keys[plane + lane] = bfloat(0.0f);
            keys[plane + lane + half_dim] = bfloat(0.0f);
            values[plane + lane] = bfloat(0.0f);
            values[plane + lane + half_dim] = bfloat(0.0f);
        }
        return;
    }
    uint kv = head / (kLfmHeads / kLfmKvHeads); // 2 query heads share a kv head
    device const bfloat *q, *k, *v;
    if (params.fused) {
        // The fused projection's rows are [Q | K | V] over 2048 dims and
        // all three bindings point at its base.
        device const bfloat *row = queries_in + ulong(token) * 2048;
        q = row + head * kLfmHeadDim;
        k = row + kLfmHidden + kv * kLfmHeadDim;
        v = row + kLfmHidden + kv_width + kv * kLfmHeadDim;
    } else {
        q = queries_in + ulong(token) * kLfmHidden + head * kLfmHeadDim;
        k = keys_in + ulong(token) * kv_width + kv * kLfmHeadDim;
        v = values_in + ulong(token) * kv_width + kv * kLfmHeadDim;
    }

    // Norm before rope: one lane holds dims (lane, lane+32) of the head.
    float q0 = float(q[lane]), q1 = float(q[lane + half_dim]);
    float k0 = float(k[lane]), k1 = float(k[lane + half_dim]);
    float q_inv = rsqrt(simd_sum(q0 * q0 + q1 * q1) / float(kLfmHeadDim) + kLfmNormEps);
    float k_inv = rsqrt(simd_sum(k0 * k0 + k1 * k1) / float(kLfmHeadDim) + kLfmNormEps);
    q0 *= float(query_norm[lane]) * q_inv;
    q1 *= float(query_norm[lane + half_dim]) * q_inv;
    k0 *= float(key_norm[lane]) * k_inv;
    k1 *= float(key_norm[lane + half_dim]) * k_inv;

    device const float *cos_row = rope_cos + ulong(token) * kLfmHeadDim;
    device const float *sin_row = rope_sin + ulong(token) * kLfmHeadDim;
    queries[plane + lane] =
        bfloat(q0 * cos_row[lane] - q1 * sin_row[lane]);
    queries[plane + lane + half_dim] =
        bfloat(q1 * cos_row[lane + half_dim] + q0 * sin_row[lane + half_dim]);
    keys[plane + lane] =
        bfloat(k0 * cos_row[lane] - k1 * sin_row[lane]);
    keys[plane + lane + half_dim] =
        bfloat(k1 * cos_row[lane + half_dim] + k0 * sin_row[lane + half_dim]);
    values[plane + lane] = v[lane];
    values[plane + lane + half_dim] = v[lane + half_dim];
}

// Non-causal flash attention, head_dim 64. Same online-softmax structure as
// vision_attention: grid {ceil(tokens / 64), heads}.

kernel void lfm_vision_attention(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionAttentionParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr ushort M = 64, N = 128, D = kLfmHeadDim;
    constexpr ushort RowsPerSimdgroup = M / 8;
    threadgroup float score_storage[M * N];
    threadgroup bfloat *probabilities =
        reinterpret_cast<threadgroup bfloat *>(score_storage);
    threadgroup float *row_sum = score_storage + M * N / 2;
    threadgroup float *previous_scale = row_sum + M;
    float running_max[RowsPerSimdgroup];
    float running_sum[RowsPerSimdgroup];
    for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
        running_max[i] = -INFINITY;
        running_sum[i] = 0.0f;
    }

    uint head = group.y;
    uint query_start = group.x * M;
    ulong qk_plane = ulong(head) * params.padded_tokens * D;
    device bfloat *q = queries + qk_plane + ulong(query_start) * D;
    device bfloat *k = keys + qk_plane;
    device bfloat *v = values + ulong(head) * D * params.padded_tokens;
    device bfloat *out = output +
        ulong(head) * params.padded_tokens * D + ulong(query_start) * D;
    uint context = params.tokens;
    uint tiles = (context + N - 1) / N;

    auto qt = tensor(q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    constexpr auto qk_descriptor =
        matmul2d_descriptor(M, N, D, false, true, false);
    matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
    auto q0 = qt.slice<D, M>(0, 0);

    auto pt = tensor(probabilities, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto p0 = pt.slice<N, M>(0, 0);
    auto st = tensor(score_storage, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto s0 = st.slice<N, M>(0, 0);
    auto first_v = tensor(v, dextents<int, 2>{D, N},
                          array<int, 2>{1, D});
    auto first_v0 = first_v.slice<D, N>(0, 0);
    constexpr auto pv_descriptor =
        matmul2d_descriptor(M, D, N, false, false, false);
    matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), float>();
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;
    for (uint tile = 0; tile < tiles; ++tile) {
        device bfloat *tile_k = k + ulong(tile) * N * D;
        auto kt = tensor(tile_k, dextents<int, 2>{D, N},
                         array<int, 2>{1, D});
        auto k0 = kt.slice<D, N>(0, 0);
        auto scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(k0), float>();
        qk.run(q0, k0, scores);
        scores.store(s0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float exponents[RowsPerSimdgroup][4];
        float scales[RowsPerSimdgroup];
        for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
            ushort row = simd_group * RowsPerSimdgroup + i;
            float values_local[4];
            float tile_max = -INFINITY;
            for (ushort j = 0; j < 4; ++j) {
                uint column = simd_lane + j * 32;
                uint key_position = tile * N + column;
                float score = score_storage[row * N + column] * params.scale;
                values_local[j] =
                    key_position < context ? score : -INFINITY;
                tile_max = max(tile_max, values_local[j]);
            }
            tile_max = simd_max(tile_max);
            float next_max = max(running_max[i], tile_max);
            float scale = fast::exp(running_max[i] - next_max);
            float tile_sum = 0.0f;
            for (ushort j = 0; j < 4; ++j) {
                exponents[i][j] = fast::exp(values_local[j] - next_max);
                tile_sum += exponents[i][j];
            }
            tile_sum = simd_sum(tile_sum);
            scales[i] = scale;
            running_sum[i] = running_sum[i] * scale + tile_sum;
            running_max[i] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
            ushort row = simd_group * RowsPerSimdgroup + i;
            if (simd_lane == 0) {
                previous_scale[row] = scales[i];
                row_sum[row] = running_sum[i];
            }
            for (ushort j = 0; j < 4; ++j) {
                uint column = simd_lane + j * 32;
                probabilities[row * N + column] = bfloat(exponents[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device bfloat *tile_v = v + ulong(tile) * N * D;
        auto vt = tensor(tile_v, dextents<int, 2>{D, N},
                         array<int, 2>{1, D});
        auto v0 = vt.slice<D, N>(0, 0);
        auto partial_output = pv.template get_destination_cooperative_tensor<
            decltype(p0), decltype(v0), float>();
        pv.run(p0, v0, partial_output);
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            auto index = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[index[1]] +
                partial_output[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto ot = tensor(out, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto converted = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), bfloat>();
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        auto index = running.get_multidimensional_index(i);
        converted[i] = bfloat(running[i] / row_sum[index[1]]);
    }
    converted.store(ot.slice<D, M>(0, 0));
}

// Gather padded attention output back to (tokens, 1024) rows.

kernel void lfm_vision_attention_pack(
    device const bfloat *padded [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant VisionQkvParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device bfloat *row = output + ulong(token) * kLfmHidden;
    for (uint index = thread_index; index < kLfmHidden; index += 256) {
        uint head = index / kLfmHeadDim;
        uint dim = index % kLfmHeadDim;
        row[index] = padded[
            (ulong(head) * params.padded_tokens + token) * kLfmHeadDim +
            dim];
    }
}

// The conv block's gated depthwise 3x3 over the patch grid. The in_proj
// output rows are [B | C | X] thirds of 1024; gated = B * X, the output is
// C * conv(gated) with one-pixel zero padding, tokens in raster order.
// Each thread owns four channels and moves B, X and C through bf16x4
// vectors; the kernel weights stay scalar gathers (1024 x 9 rows, hot in
// cache across a token's threads).

kernel void lfm_vision_conv2d(
    device const bfloat *proj [[buffer(0)]],
    device const bfloat *kernel_weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant VisionGridParams &params [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint grid_y = token / params.grid_width;
    uint grid_x = token % params.grid_width;
    for (uint channel = thread_index * 4; channel < kLfmHidden;
         channel += 256 * 4) {
        float4 conv = 0.0f;
        #pragma unroll
        for (ushort dy = 0; dy < 3; ++dy) {
            uint ny = grid_y + dy - 1;
            if (ny >= params.grid_height) continue;
            #pragma unroll
            for (ushort dx = 0; dx < 3; ++dx) {
                uint nx = grid_x + dx - 1;
                if (nx >= params.grid_width) continue;
                ulong n = ulong(ny * params.grid_width + nx);
                const float4 b_values =
                    float4(*reinterpret_cast<device const bfloat4 *>(
                        proj + n * 3 * kLfmHidden + channel));
                const float4 x_values =
                    float4(*reinterpret_cast<device const bfloat4 *>(
                        proj + n * 3 * kLfmHidden + 2 * kLfmHidden +
                        channel));
                const float4 gated = b_values * x_values;
                const ushort tap = dy * 3 + dx;
                const float4 w = float4(
                    float(kernel_weights[(channel + 0) * 9 + tap]),
                    float(kernel_weights[(channel + 1) * 9 + tap]),
                    float(kernel_weights[(channel + 2) * 9 + tap]),
                    float(kernel_weights[(channel + 3) * 9 + tap]));
                conv += gated * w;
            }
        }
        const float4 c_values =
            float4(*reinterpret_cast<device const bfloat4 *>(
                proj + ulong(token) * 3 * kLfmHidden + kLfmHidden + channel));
        *reinterpret_cast<device bfloat4 *>(
            output + ulong(token) * kLfmHidden + channel) =
            bfloat4(c_values * conv);
    }
}

// Dense bf16 GEMM whose epilogue is the gated FFN's multiply: it replaces
// the up projection's plain GEMM and the silu(gate) * up pass, so the
// output rows land ready for the down GEMM. Same tile structure as
// vision_gemm_m64n128 (M=64, N=128, BK=128, fp32 accumulation); the gate
// buffer is bound as `residual` and read elementwise — each thread touches
// only the elements it writes, so the output may overwrite the gate rows
// in place.

kernel void lfm_vision_gemm_m64n128_silu_up(
    device bfloat *input [[buffer(0)]],
    device bfloat *weights [[buffer(1)]],
    device bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device bfloat *gate [[buffer(4)]],
    constant VisionGemmParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr ushort TileM = 64, TileN = 128, BK = 128;
    const uint output_origin = group.y * TileN;
    const uint row_origin = group.x * TileM;
    device bfloat *rows = input + ulong(row_origin) * params.input_size;
    device bfloat *tile_weights =
        weights + ulong(output_origin) * params.input_size;
    auto a = tensor(rows, dextents<int, 2>{int(params.input_size), TileM},
                    array<int, 2>{1, int(params.input_size)});
    auto b = tensor(tile_weights,
                    dextents<int, 2>{int(params.input_size), TileN},
                    array<int, 2>{1, int(params.input_size)});
    auto c = tensor(output + ulong(row_origin) * params.output_size,
                    dextents<int, 2>{int(params.output_size), TileM},
                    array<int, 2>{1, int(params.output_size)});
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> operation;
    auto a0 = a.template slice<BK, TileM>(0, 0);
    auto b0 = b.template slice<BK, TileN>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i)
        accumulated[i] = 0.0f;
    for (uint chunk = 0; chunk < params.input_size / BK; ++chunk) {
        auto a_slice = a.template slice<BK, TileM>(chunk * BK, 0);
        auto b_slice = b.template slice<BK, TileN>(chunk * BK, 0);
        operation.run(a_slice, b_slice, accumulated);
        threadgroup_barrier(mem_flags::mem_none);
    }
    auto converted = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), bfloat>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        const ulong element = ulong(row_origin + index[1]) *
            params.output_size + output_origin + index[0];
        float g = float(gate[element]);
        const float gate_value = g / (1.0f + precise::exp(-g));
        // The up value rounds to bf16 before the multiply, exactly as the
        // separate up GEMM + silu_mul passes rounded it.
        const bfloat up = bfloat(
            accumulated[i] + float(bias[output_origin + index[0]]));
        converted[i] = bfloat(float(up) * gate_value);
    }
    converted.store(c.template slice<TileN, TileM>(output_origin, 0));
}

// silu(gate) * up, elementwise — the gated FFN's multiply between the two
// 2560-wide projections and the down GEMM. params.width carries the total
// element count (rows x 2560).

kernel void lfm_vision_silu_mul(
    device const bfloat *gate [[buffer(0)]],
    device const bfloat *up [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant VisionNormParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]])
{
    for (uint i = index; i < params.width; i += grid_size) {
        float g = float(gate[i]);
        output[i] = bfloat((g / (1.0f + precise::exp(-g))) * float(up[i]));
    }
}

// The projector's 2x2 merge gather: merged token (by, bx) of the
// (height/2, width/2) merged grid holds the four raster-order patches
// (2by+dy, 2bx+dx), dy*2+dx ordering, concatenated into 4096 dims.

kernel void lfm_vision_merge(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint merged [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint merged_width = params.grid_width / 2;
    uint by = merged / merged_width;
    uint bx = merged % merged_width;
    for (uint index = thread_index; index < 4 * kLfmHidden; index += 256) {
        uint group = index / kLfmHidden;
        uint channel = index % kLfmHidden;
        uint patch = (2 * by + group / 2) * params.grid_width + 2 * bx + group % 2;
        output[ulong(merged) * 4 * kLfmHidden + index] =
            input[ulong(patch) * kLfmHidden + channel];
    }
}

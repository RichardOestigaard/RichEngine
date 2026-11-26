// Nemotron Omni vision tower kernels. Token order is raster order; the ten
// register rows precede the patch rows inside the ViT, then the merger drops
// them by reading only the patch rows. Positions use PyTorch-style bilinear
// resampling (align_corners=false) of the learned 128x128 table.

#include "metal/abi/Vision.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant ushort kNemotronHeads = 16;
constant ushort kNemotronHeadDim = 80;
constant ushort kNemotronHidden = 1280;
constant ushort kNemotronPatch = 16;
constant ushort kNemotronPatchDim = 768;
constant ushort kNemotronGridSide = 128;
constant ushort kNemotronMerge = 2;

constant float3 kNemotronMean = float3(0.48145466f, 0.4578275f, 0.40821073f);
constant float3 kNemotronStd = float3(0.26862954f, 0.26130258f, 0.27577711f);

kernel void nemotron_vision_patchify(
    device const uchar *pixels [[buffer(0)]],
    device bfloat *patches [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint image_width = params.grid_width * kNemotronPatch;
    device bfloat *row = patches + ulong(token) * kNemotronPatchDim;
    for (uint feature = thread_index; feature < kNemotronPatchDim;
         feature += 256) {
        uint channel = feature / (kNemotronPatch * kNemotronPatch);
        uint within = feature % (kNemotronPatch * kNemotronPatch);
        uint patch_row = within / kNemotronPatch;
        uint patch_col = within % kNemotronPatch;
        ulong pixel =
            (ulong(token / params.grid_width * kNemotronPatch + patch_row) *
                 image_width +
             (token % params.grid_width) * kNemotronPatch + patch_col) * 3 +
            channel;
        float value = float(pixels[pixel]) * (1.0f / 255.0f);
        row[feature] = bfloat((value - kNemotronMean[channel]) /
                              kNemotronStd[channel]);
    }
}

// Initial hidden rows: registers first, then patch rows containing only the
// resampled learned position values. Padding rows are zero because residual
// GEMMs update whole tiles in place.
kernel void nemotron_vision_prepare(
    device const bfloat *table [[buffer(0)]],
    device const bfloat *registers [[buffer(1)]],
    device float *hidden [[buffer(2)]],
    constant NemotronVisionPrepareParams &params [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device float *out = hidden + ulong(token) * kNemotronHidden;
    if (token >= params.tokens) {
        for (uint dim = thread_index; dim < kNemotronHidden; dim += 256)
            out[dim] = 0.0f;
        return;
    }
    if (token < params.registers) {
        device const bfloat *source = registers + ulong(token) * kNemotronHidden;
        for (uint dim = thread_index; dim < kNemotronHidden; dim += 256)
            out[dim] = float(source[dim]);
        return;
    }

    uint patch = token - params.registers;
    uint row = patch / params.grid_width;
    uint col = patch % params.grid_width;
    float source_row = (float(row) + 0.5f) * float(kNemotronGridSide) /
        float(params.grid_height) - 0.5f;
    float source_col = (float(col) + 0.5f) * float(kNemotronGridSide) /
        float(params.grid_width) - 0.5f;
    float row_floor = floor(source_row);
    float col_floor = floor(source_col);
    int row0 = clamp(int(row_floor), 0, int(kNemotronGridSide - 1));
    int col0 = clamp(int(col_floor), 0, int(kNemotronGridSide - 1));
    int row1 = clamp(int(row_floor) + 1, 0, int(kNemotronGridSide - 1));
    int col1 = clamp(int(col_floor) + 1, 0, int(kNemotronGridSide - 1));
    float dy = clamp(source_row - row_floor, 0.0f, 1.0f);
    float dx = clamp(source_col - col_floor, 0.0f, 1.0f);
    for (uint dim = thread_index; dim < kNemotronHidden; dim += 256) {
        float a = float(table[(ulong(row0) * kNemotronGridSide + col0) *
                              kNemotronHidden + dim]);
        float b = float(table[(ulong(row0) * kNemotronGridSide + col1) *
                              kNemotronHidden + dim]);
        float c = float(table[(ulong(row1) * kNemotronGridSide + col0) *
                              kNemotronHidden + dim]);
        float d = float(table[(ulong(row1) * kNemotronGridSide + col1) *
                              kNemotronHidden + dim]);
        out[dim] = a * (1.0f - dx) * (1.0f - dy) + b * dx * (1.0f - dy) +
            c * (1.0f - dx) * dy + d * dx * dy;
    }
}

// Scatter separate Q/K/V rows into per-head planes. There is no rope and no
// q/k norm in this tower. Padded rows are zero so full attention tiles read
// finite values.
kernel void nemotron_vision_qkv_prepare(
    device const bfloat *queries_in [[buffer(0)]],
    device const bfloat *keys_in [[buffer(1)]],
    device const bfloat *values_in [[buffer(2)]],
    device bfloat *queries [[buffer(3)]],
    device bfloat *keys [[buffer(4)]],
    device bfloat *values [[buffer(5)]],
    constant VisionQkvParams &params [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    const bool valid = token < params.tokens;
    for (uint index = thread_index;
         index < uint(kNemotronHeads) * kNemotronHeadDim; index += 256) {
        uint head = index / kNemotronHeadDim;
        uint dim = index % kNemotronHeadDim;
        ulong plane = (ulong(head) * params.padded_tokens + token) *
            kNemotronHeadDim + dim;
        if (!valid) {
            queries[plane] = bfloat(0.0f);
            keys[plane] = bfloat(0.0f);
            values[plane] = bfloat(0.0f);
            continue;
        }
        queries[plane] = queries_in[ulong(token) * kNemotronHidden + index];
        keys[plane] = keys_in[ulong(token) * kNemotronHidden + index];
        values[plane] = values_in[ulong(token) * kNemotronHidden + index];
    }
}

// Non-causal flash attention for head dimension 80. Grid: {ceil(tokens/64),
// heads}; output: (heads, padded_tokens, 80).
kernel void nemotron_vision_attention(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionAttentionParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr ushort M = 64, N = 128, D = kNemotronHeadDim;
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
    ulong plane = ulong(head) * params.padded_tokens * D;
    device bfloat *q = queries + plane + ulong(query_start) * D;
    device bfloat *k = keys + plane;
    device bfloat *v = values + plane;
    device bfloat *out = output + plane + ulong(query_start) * D;
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
    auto first_v = tensor(v, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto first_v0 = first_v.slice<D, N>(0, 0);
    constexpr auto pv_descriptor =
        matmul2d_descriptor(M, D, N, false, false, false);
    matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), float>();
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;

    for (uint tile = 0; tile < tiles; ++tile) {
        device bfloat *tile_k = k + ulong(tile) * N * D;
        auto kt = tensor(tile_k, dextents<int, 2>{D, N}, array<int, 2>{1, D});
        auto k0 = kt.slice<D, N>(0, 0);
        auto scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(k0), float>();
        qk.run(q0, k0, scores);
        scores.store(s0);
        threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);

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
                values_local[j] = key_position < context ? score : -INFINITY;
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
        threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
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
        threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);

        device bfloat *tile_v = v + ulong(tile) * N * D;
        auto vt = tensor(tile_v, dextents<int, 2>{D, N}, array<int, 2>{1, D});
        auto v0 = vt.slice<D, N>(0, 0);
        auto partial_output = pv.template get_destination_cooperative_tensor<
            decltype(p0), decltype(v0), float>();
        pv.run(p0, v0, partial_output);
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            auto index = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[index[1]] +
                partial_output[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
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

kernel void nemotron_vision_attention_pack(
    device const bfloat *padded [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant VisionQkvParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device bfloat *row = output + ulong(token) * kNemotronHidden;
    for (uint index = thread_index; index < kNemotronHidden; index += 256) {
        uint head = index / kNemotronHeadDim;
        uint dim = index % kNemotronHeadDim;
        row[index] = padded[
            (ulong(head) * params.padded_tokens + token) * kNemotronHeadDim +
            dim];
    }
}

// Pixel-unshuffle one 2x2 block into one fp32 row. Feature order is channel
// fastest, then horizontal offset, then vertical offset, matching llama.cpp's
// build_patch_merge_permute.
kernel void nemotron_vision_merge(
    device const float *hidden [[buffer(0)]],
    device float *merged [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint merged_width = params.grid_width / kNemotronMerge;
    uint row_block = token / merged_width;
    uint col_block = token % merged_width;
    device float *out = merged + ulong(token) * kNemotronHidden * 4;
    for (uint feature = thread_index; feature < kNemotronHidden * 4;
         feature += 256) {
        uint channel = feature % kNemotronHidden;
        uint within = feature / kNemotronHidden;
        uint dx = within % kNemotronMerge;
        uint dy = within / kNemotronMerge;
        uint patch = (row_block * kNemotronMerge + dy) * params.grid_width +
            col_block * kNemotronMerge + dx;
        out[feature] = hidden[ulong(patch) * kNemotronHidden + channel];
    }
}

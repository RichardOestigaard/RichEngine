// The lfm3a audio encoder's kernels, dispatched by ops::LfmAudioEnc.mm.
// Activations stay bf16, the trunk's residual stream fp32 — the same
// convention ops::LfmVision keeps. Positions are the frontend's time
// axis; a stage's activation is (freq * time + time) positions of a
// channel row, and the trunk's rows are the positions' own.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/Audio.h"
#include "metal/abi/Vision.h"
using namespace metal;
using namespace mpp::tensor_ops;

constant ushort kAudioHeads = 8;
constant ushort kAudioKvHeads = 4;
constant ushort kAudioHeadDim = 96;
constant ushort kAudioHidden = 768;
constant ushort kAudioBcxWidth = 2304;   // [B|C|X]
constant ushort kAudioWindow = 32;
constant ushort kAudioRowTile = 64;    // GEMM row tile; padded rows write zeros
constant float kAudioRopeTheta = 1000000.0f;
constant float kAudioNormEps = 1e-5f;

// float32 mel -> bf16, for the frontend's first conv.
kernel void lfm_audio_mel_pack(
    device const float *mel [[buffer(0)]],
    device bfloat *out [[buffer(1)]],
    constant LfmAudioCountParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.count)
        out[gid] = bfloat(mel[gid]);
}

// A stride-2 3x3 conv over (freq, time) planes of channel rows, one thread
// per four output channels — bf16x4 vector loads/stores, and the channel
// dimension of the depthwise stages (and the dense stage's weights) is the
// vector lane. The time axis is zero-padded on both sides (the clip's left
// context is a zero state; the right is a zero pad), the freq axis
// symmetrically. Dense (in_channels 1, stage 1 — one scalar input shared by
// all four lanes) and depthwise (in_channels == out_channels, stages 2 and
// 3) share the weight layout out_channel * 9 + ky * 3 + kx.
// params.input_mode folds the standalone elementwise passes into the loads:
// 0 bf16 input, 1 bf16 input through SiLU (a silu pass folded away), 2
// float32 input (the raw mel — no pack pass).
kernel void lfm_audio_conv_down(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant LfmAudioConvParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint positions = params.out_freq * params.out_time;
    const uint channel_groups = params.out_channels / 4;
    if (gid >= positions * channel_groups)
        return;
    const uint oc = (gid % channel_groups) * 4;
    const uint pos = gid / channel_groups;
    const uint fo = pos / params.out_time;
    const uint to = pos % params.out_time;
    const bool dense = params.in_channels == 1;
    device const float *input_f32 =
        reinterpret_cast<device const float *>(input);
    float4 sum = 0.0f;
    #pragma unroll
    for (ushort ky = 0; ky < 3; ++ky) {
        const int fi = int(fo * 2) + ky - 1;
        if (fi < 0 || fi >= int(params.in_freq))
            continue;
        #pragma unroll
        for (ushort kx = 0; kx < 3; ++kx) {
            const int ti = int(to * 2) + kx - 1;
            if (ti < 0 || ti >= int(params.in_time))
                continue;
            const uint at = (uint(fi) * params.in_time + uint(ti)) *
                            params.in_channels;
            const ushort tap = ky * 3 + kx;
            float4 x;
            if (dense) {
                const float v = params.input_mode == 2
                                    ? input_f32[at]
                                    : float(input[at]);
                x = float4(v);
            } else {
                x = float4(*reinterpret_cast<device const bfloat4 *>(
                    input + at + oc));
            }
            if (params.input_mode == 1)
                x = x / (1.0f + exp(-x));
            // tap-major weights (the conv_wt repack): one aligned bf16x4
            // load carries all four output channels' tap weight.
            const float4 w = float4(*reinterpret_cast<device const bfloat4 *>(
                weight + params.weight_offset +
                uint(tap) * params.out_channels + oc));
            sum += x * w;
        }
    }
    const float4 b = float4(*reinterpret_cast<device const bfloat4 *>(
        bias + oc));
    *reinterpret_cast<device bfloat4 *>(
        output + pos * params.out_channels + oc) = bfloat4(sum + b);
}

// Stage 1's dense conv as a matmul: im2col gathers each output position's
// 3x3 float32 mel window into a K=16 bf16 row (seven zero columns; a zero
// column pair for the out-of-range taps the conv skips, so the pad is
// exact), and lfm_audio_stage1_mm multiplies it by the tap-major weights
// through an MPP tensor matmul — K=16 is the smallest legal K extent.
kernel void lfm_audio_stage1_im2col(
    device const float *mel [[buffer(0)]],
    device bfloat *columns [[buffer(1)]],
    constant LfmAudioConvParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint positions =
        (params.out_freq * params.out_time + 63) / 64 * 64;
    if (gid >= positions * 16)
        return;
    const uint pos = gid / 16;
    const uint k = gid % 16;
    float v = 0.0f;
    if (k < 9 && pos < params.out_freq * params.out_time) {
        const uint fo = pos / params.out_time;
        const uint to = pos % params.out_time;
        const int fi = int(fo * 2) + int(k / 3) - 1;
        const int ti = int(to * 2) + int(k % 3) - 1;
        if (fi >= 0 && fi < int(params.in_freq) && ti >= 0 &&
            ti < int(params.in_time))
            v = mel[uint(fi) * params.in_time + uint(ti)];
    }
    // The im2col rows live past the transposed weights in the same slot.
    columns[params.weight_offset + gid] = bfloat(v);
}

// Stage 1's matmul: out[pos, oc] = sum_k im2col[pos,k] * wt[k*256 + oc] +
// bias[oc]. The weight buffer is the conv_wt stage-0 region — the zeroed A
// columns 9..15 make B's taps past 9 dead, so the stage-0 region needs no
// padding and the reads past it stay inside the scratch slot.
kernel void lfm_audio_stage1_mm(
    device bfloat *columns [[buffer(0)]],
    device bfloat *weights [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant LfmAudioConvParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr ushort M = 64, N = 128, K = 16;
    const uint row_origin = group.x * M;
    const uint col_origin = group.y * N;
    auto a = tensor(columns + params.weight_offset + ulong(row_origin) * K,
                    dextents<int, 2>{K, M}, array<int, 2>{1, K});
    // NN layout: B's tensor is (n, k) over the tap-major rows — element
    // (oc, tap) sits at tap * out_channels + oc.
    auto b = tensor(weights + col_origin,
                    dextents<int, 2>{N, K},
                    array<int, 2>{1, int(params.out_channels)});
    auto c = tensor(output + ulong(row_origin) * params.out_channels,
                    dextents<int, 2>{int(params.out_channels), M},
                    array<int, 2>{1, int(params.out_channels)});
    constexpr auto descriptor = matmul2d_descriptor(
        M, N, K, false, false, false);
    matmul2d<descriptor, execution_simdgroups<8>> operation;
    auto a0 = a.template slice<K, M>(0, 0);
    auto b0 = b.template slice<N, K>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    operation.run(a0, b0, accumulated);
    auto converted = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), bfloat>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        converted[i] = bfloat(
            accumulated[i] + float(bias[col_origin + index[0]]));
    }
    converted.store(c.template slice<N, M>(col_origin, 0));
}

// Repack the three conv stages' oc-major kernels ([out][ky*3+kx]) into
// tap-major rows ([stage][tap][oc]) inside one scratch buffer, so
// lfm_audio_conv_down loads the four output channels' tap weight as one
// bfloat4. params.count carries per-stage element count (256 x 9).
kernel void lfm_audio_conv_wt(
    device const bfloat *w_in [[buffer(0)]],
    device const bfloat *w_mid [[buffer(1)]],
    device const bfloat *w_end [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant LfmAudioCountParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.count * 3)
        return;
    const uint stage = gid / params.count;
    const uint i = gid % params.count;
    const uint oc = i / 9;
    const uint tap = i % 9;
    device const bfloat *src =
        stage == 0 ? w_in : stage == 1 ? w_mid : w_end;
    out[stage * params.count + tap * (params.count / 9) + oc] = src[i];
}

// Elementwise SiLU over a bf16 row range.
kernel void lfm_audio_silu(
    device bfloat *data [[buffer(0)]],
    constant LfmAudioCountParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.count) {
        const float x = float(data[gid]);
        data[gid] = bfloat(x / (1.0f + exp(-x)));
    }
}

// The frontend's channel-major flatten: (freq * frames + time, channel)
// rows become one 4096-wide row per time position, index channel * freq
// + freq, matching the HF downsampler's flatten(1, 2).
kernel void lfm_audio_flatten(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant LfmAudioFlattenParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint row_width = params.channels * params.freq;
    const uint rows = (params.tokens + kAudioRowTile - 1) / kAudioRowTile *
                      kAudioRowTile;
    if (gid >= rows * row_width)
        return;
    const uint token = gid / row_width;
    const uint j = gid % row_width;
    const uint channel = j / params.freq;
    const uint freq = j % params.freq;
    if (token >= params.tokens) {
        output[gid] = bfloat(0.0f);
        return;
    }
    float v = float(
        input[(freq * params.frames + token) * params.channels + channel]);
    if (params.input_silu)
        v = v / (1.0f + exp(-v));
    output[gid] = bfloat(v);
}

// Fused QKV projection for the audio trunk: one m64n128 GEMM whose
// 1536-wide output rows are [Q 768 | K 384 | V 384]. The twelve column
// tiles pick their weight operand per tile — 0-5 query, 6-8 key, 9-11
// value — like lfm_vision_gemm_qkv; the output lands across the
// contiguous Query/Key/Value scratch slots.
kernel void lfm_audio_gemm_qkv(
    device bfloat *input [[buffer(0)]],
    device bfloat *query_weights [[buffer(1)]],
    device bfloat *key_weights [[buffer(2)]],
    device bfloat *value_weights [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant VisionGemmParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]])
{
    constexpr ushort TileM = 64, TileN = 128, BK = 128;
    constexpr ushort q_tiles = kAudioHidden / TileN;                 // 6
    constexpr ushort kv_width = kAudioKvHeads * kAudioHeadDim;       // 384
    constexpr ushort kv_tiles = kv_width / TileN;                    // 3
    const uint tile = group.y;
    device bfloat *tile_weights;
    uint output_origin;
    if (tile < q_tiles) {
        tile_weights = query_weights + tile * TileN * params.input_size;
        output_origin = tile * TileN;
    } else if (tile < q_tiles + kv_tiles) {
        tile_weights = key_weights;
        output_origin = kAudioHidden + (tile - q_tiles) * TileN;
        tile_weights += (tile - q_tiles) * TileN * params.input_size;
    } else {
        tile_weights = value_weights;
        output_origin =
            kAudioHidden + kv_width + (tile - q_tiles - kv_tiles) * TileN;
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

// The lin_out projection with fp32 output and bias, no residual pass: it
// opens the trunk's fp32 residual stream, so the zero-fill the residual
// GEMM used to accumulate into is gone.
kernel void lfm_audio_gemm_lin_out(
    device bfloat *input [[buffer(0)]],
    device bfloat *weights [[buffer(1)]],
    device bfloat *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant VisionGemmParams &params [[buffer(4)]],
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
        decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        converted[i] =
            accumulated[i] + float(bias[output_origin + index[0]]);
    }
    converted.store(c.template slice<TileN, TileM>(output_origin, 0));
}

// Zero the fp32 residual rows the first GEMM accumulates into.
kernel void lfm_audio_prepare(
    device float *hidden [[buffer(0)]],
    constant LfmAudioCountParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < params.count)
        hidden[gid] = 0.0f;
}

// Per-head q/k RMSNorm + NEOX rope over all 96 dims (pairs (d, d+48) at
// position t, theta 1e6) + the GQA scatter into per-head (padded_tokens,
// 96) planes; padded rows write zeros. One 96-thread group per
// (token, head); kv heads serve groups of two.
kernel void lfm_audio_qkv_prepare(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device bfloat *queries [[buffer(5)]],
    device bfloat *keys [[buffer(6)]],
    device bfloat *values [[buffer(7)]],
    constant LfmAudioQkvParams &params [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint token = group / kAudioHeads;
    const uint head = group % kAudioHeads;
    const bool kv_lane = head < kAudioKvHeads;
    threadgroup float reductions[3];
    device bfloat *q_out =
        queries + (ulong(head) * params.padded_tokens + token) * kAudioHeadDim;
    device bfloat *k_out =
        keys + (ulong(head) * params.padded_tokens + token) * kAudioHeadDim;
    device bfloat *v_out =
        values + (ulong(head) * params.padded_tokens + token) * kAudioHeadDim;
    if (token >= params.tokens) {
        if (lane < kAudioHeadDim) {
            q_out[lane] = bfloat(0.0f);
            if (kv_lane) {
                k_out[lane] = bfloat(0.0f);
                v_out[lane] = bfloat(0.0f);
            }
        }
        return;
    }
    const uint kv_width = kAudioHeadDim * kAudioKvHeads;
    const device bfloat *q_row;
    const device bfloat *k_row;
    const device bfloat *v_row;
    if (params.fused) {
        // The fused projection's rows are [Q | K | V] over 1536 dims; all
        // three bindings point at its base.
        const device bfloat *row =
            query + ulong(token) * (kAudioHidden + 2 * kv_width);
        q_row = row;
        k_row = row + kAudioHidden;
        v_row = row + kAudioHidden + kv_width;
    } else {
        q_row = query + ulong(token) * kAudioHeadDim * kAudioHeads;
        k_row = key + ulong(token) * kv_width;
        v_row = value + ulong(token) * kv_width;
    }
    const float q_val = float(q_row[head * kAudioHeadDim + lane]);
    const float qw = lane < kAudioHeadDim ? float(query_norm[lane]) : 0.0f;
    const float k_val = kv_lane ? float(k_row[head * kAudioHeadDim + lane]) : 0.0f;
    const float kw = kv_lane ? float(key_norm[lane]) : 0.0f;

    float squares = q_val * q_val;
    squares = simd_sum(squares);
    if (simd_lane == 0) reductions[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float q_total = reductions[0] + reductions[1] + reductions[2];
    const float q_inv = rsqrt(q_total / float(kAudioHeadDim) + kAudioNormEps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    squares = k_val * k_val;
    squares = simd_sum(squares);
    if (simd_lane == 0) reductions[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_total = reductions[0] + reductions[1] + reductions[2];
    const float k_inv = rsqrt(k_total / float(kAudioHeadDim) + kAudioNormEps);

    if (lane < kAudioHeadDim / 2) {
        const float freq = pow(kAudioRopeTheta, -2.0f * float(lane) / float(kAudioHeadDim));
        const float angle = float(token) * freq;
        const float cs = cos(angle), sn = sin(angle);
        const float q0 = q_val * q_inv * qw;
        const float q1 = float(q_row[head * kAudioHeadDim + lane + kAudioHeadDim / 2]) *
                         q_inv * float(query_norm[lane + kAudioHeadDim / 2]);
        q_out[lane] = bfloat(q0 * cs - q1 * sn);
        q_out[lane + kAudioHeadDim / 2] = bfloat(q1 * cs + q0 * sn);
        if (kv_lane) {
            const float k0 = k_val * k_inv * kw;
            const float k1 = float(k_row[head * kAudioHeadDim + lane + kAudioHeadDim / 2]) *
                             k_inv * float(key_norm[lane + kAudioHeadDim / 2]);
            k_out[lane] = bfloat(k0 * cs - k1 * sn);
            k_out[lane + kAudioHeadDim / 2] = bfloat(k1 * cs + k0 * sn);
            v_out[lane] = v_row[head * kAudioHeadDim + lane];
            v_out[lane + kAudioHeadDim / 2] = v_row[head * kAudioHeadDim + lane + kAudioHeadDim / 2];
        }
    }
}

// Causal sliding-window-32 attention, one 32-lane group per (token, head):
// a lane owns dims {lane*3 .. lane*3+2}, the score reduces over one simd
// group, and the softmax over at most 32 keys runs online in registers.
kernel void lfm_audio_attention(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant LfmAudioAttentionParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint token = group / kAudioHeads;
    const uint head = group % kAudioHeads;
    if (token >= params.tokens)
        return;
    const uint kv_head = head >> 1;
    const device bfloat *q =
        queries + (ulong(head) * params.padded_tokens + token) * kAudioHeadDim;
    const device bfloat *k_head = keys + ulong(kv_head) * params.padded_tokens * kAudioHeadDim;
    const device bfloat *v_head = values + ulong(kv_head) * params.padded_tokens * kAudioHeadDim;
    float q0 = float(q[lane * 3]), q1 = float(q[lane * 3 + 1]), q2 = float(q[lane * 3 + 2]);
    const uint lo = token >= kAudioWindow ? token + 1 - kAudioWindow : 0;
    float m = -INFINITY, denom = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;
    for (uint j = lo; j <= token; ++j) {
        const device bfloat *k_j = k_head + ulong(j) * kAudioHeadDim;
        const float partial = q0 * float(k_j[lane * 3]) +
                              q1 * float(k_j[lane * 3 + 1]) +
                              q2 * float(k_j[lane * 3 + 2]);
        const float score = simd_sum(partial) * params.scale;
        const device bfloat *v_j = v_head + ulong(j) * kAudioHeadDim;
        if (score > m) {
            const float shift = exp(m - score);
            denom = denom * shift + 1.0f;
            acc0 = acc0 * shift + float(v_j[lane * 3]);
            acc1 = acc1 * shift + float(v_j[lane * 3 + 1]);
            acc2 = acc2 * shift + float(v_j[lane * 3 + 2]);
            m = score;
        } else {
            const float p = exp(score - m);
            denom += p;
            acc0 += p * float(v_j[lane * 3]);
            acc1 += p * float(v_j[lane * 3 + 1]);
            acc2 += p * float(v_j[lane * 3 + 2]);
        }
    }
    device bfloat *o =
        out + (ulong(head) * params.padded_tokens + token) * kAudioHeadDim;
    o[lane * 3] = bfloat(acc0 / denom);
    o[lane * 3 + 1] = bfloat(acc1 / denom);
    o[lane * 3 + 2] = bfloat(acc2 / denom);
}

// Gather the per-head attention rows back into (token, head*96 + dim)
// context rows; padded rows write zeros.
kernel void lfm_audio_attention_pack(
    device const bfloat *attention [[buffer(0)]],
    device bfloat *context [[buffer(1)]],
    constant LfmAudioQkvParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint rows = (params.tokens + kAudioRowTile - 1) / kAudioRowTile *
                      kAudioRowTile;
    if (gid >= rows * kAudioHidden)
        return;
    const uint token = gid / kAudioHidden;
    const uint head = (gid % kAudioHidden) / kAudioHeadDim;
    const uint dim = gid % kAudioHeadDim;
    context[gid] =
        token < params.tokens
            ? attention[(ulong(head) * params.padded_tokens + token) * kAudioHeadDim + dim]
            : bfloat(0.0f);
}

// The conv block's causal taps-3 depthwise conv1d over the B*X half of the
// pw1 [B|C|X] rows, gated by C: y[t, c] = wide[t, 2H + c] * sum_k w[c, k] *
// (wide[t-2+k, c] * wide[t-2+k, H + c]); taps below t=0 read the zero
// state, so the block is causal.
kernel void lfm_audio_conv_bcx(
    device const bfloat *wide [[buffer(0)]],
    device const bfloat *dw_weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant LfmAudioQkvParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint rows = (params.tokens + kAudioRowTile - 1) / kAudioRowTile *
                      kAudioRowTile;
    if (gid >= rows * kAudioHidden)
        return;
    const uint token = gid / kAudioHidden;
    const uint c = gid % kAudioHidden;
    if (token >= params.tokens) {
        output[gid] = bfloat(0.0f);
        return;
    }
    float sum = 0.0f;
    #pragma unroll
    for (ushort k = 0; k < 3; ++k) {
        const int ti = int(token) - 2 + k;
        if (ti < 0)
            continue;
        const float b = float(wide[uint(ti) * kAudioBcxWidth + c]);
        const float x = float(wide[uint(ti) * kAudioBcxWidth + kAudioHidden + c]);
        sum += float(dw_weight[c * 3 + k]) * b * x;
    }
    const float gate = float(wide[token * kAudioBcxWidth + 2 * kAudioHidden + c]);
    output[gid] = bfloat(gate * sum);
}

// LayerNorm with bias over bf16 rows (the audio projector's eps is 1e-5,
// unlike the vision tower's 1e-6). One threadgroup per row.
kernel void lfm_audio_layer_norm(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant LfmAudioCountParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    device const bfloat *row = input + ulong(group) * params.count;
    device bfloat *out = output + ulong(group) * params.count;

    float sum = 0.0f;
    for (uint column = thread_index; column < params.count; column += 256)
        sum += float(row[column]);
    sum = simd_sum(sum);
    if (simd_lane == 0) reductions[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) total += reductions[i];
    const float mean = total / float(params.count);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float squares = 0.0f;
    for (uint column = thread_index; column < params.count; column += 256) {
        const float centered = float(row[column]) - mean;
        squares += centered * centered;
    }
    squares = simd_sum(squares);
    if (simd_lane == 0) reductions[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance = 0.0f;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) variance += reductions[i];
    variance /= float(params.count);
    const float inverse = rsqrt(variance + 1e-5f);

    for (uint column = thread_index; column < params.count; column += 256) {
        const float normalized = (float(row[column]) - mean) * inverse;
        out[column] = bfloat(normalized * float(weight[column]) +
                             float(bias[column]));
    }
}

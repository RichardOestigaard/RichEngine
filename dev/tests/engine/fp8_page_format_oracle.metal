#include <metal_stdlib>
#include "metal/abi/ExecutionGeometry.h"
using namespace metal;

// Native Metal oracle for the persistent FP8 E4M3 page format. Production
// writes pages directly from packed prefill/verify kernels and does not link
// these conversion or gather symbols. The layout is the INT8 format's:
// keys token-major, values dimension-major, one fp32 scale per (head,
// token); only the element encoding differs.
constant uint RichFp8PageTokens = RICHENGINE_TARGET_KV_BLOCK_TOKENS;
constant uint RichFp8KVHeads = 4;
constant uint RichFp8HeadDimension = 256;
constant uint RichFp8ElementsPerLayerPage = 32768;
constant uint RichFp8ScalesPerLayerPage = 128;

struct RichFp8KVPageParams {
    uint source_page;
    uint destination_page;
    uint valid_tokens;
    uint reserved;
};

inline ulong richengine_fp8_key_data_index(
    uint page, uint head, uint token, uint dimension)
{
    return ulong(page) * RichFp8ElementsPerLayerPage +
        (ulong(head) * RichFp8PageTokens + token) *
            RichFp8HeadDimension + dimension;
}

inline ulong richengine_fp8_scale_index(
    uint page, uint head, uint token)
{
    return ulong(page) * RichFp8ScalesPerLayerPage +
        ulong(head) * RichFp8PageTokens + token;
}

inline ulong richengine_fp8_value_data_index(
    uint page, uint head, uint token, uint dimension)
{
    return ulong(page) * RichFp8ElementsPerLayerPage +
        (ulong(head) * RichFp8HeadDimension + dimension) *
            RichFp8PageTokens + token;
}

inline float richengine_fp8_decode(uchar code)
{
    return unpack<float>(packed_metal_fp8_e4m3<4>(
        packed_uchar4(code, 0, 0, 0)))[0];
}

inline uchar richengine_fp8_encode(float value)
{
    // Saturating round-to-nearest-even, the same instruction the production
    // store rows use.
    return pack<metal_fp8_e4m3_format>(float4(value)).as_storage_type()[0];
}

inline float richengine_fp8_load_key(
    device const uchar *keys,
    device const float *scales,
    uint page, uint head, uint token, uint dimension)
{
    float scale = scales[richengine_fp8_scale_index(page, head, token)];
    return richengine_fp8_decode(keys[richengine_fp8_key_data_index(
        page, head, token, dimension)]) * scale;
}

inline float richengine_fp8_load_value(
    device const uchar *values,
    device const float *scales,
    uint page, uint head, uint token, uint dimension)
{
    float scale = scales[richengine_fp8_scale_index(page, head, token)];
    return richengine_fp8_decode(values[richengine_fp8_value_data_index(
        page, head, token, dimension)]) * scale;
}

// Converts one BF16 cache page for one attention layer into per-token/head
// scaled E4M3. Dispatch 256 threadgroups of 256 threads: 128 K rows followed
// by 128 V rows. Each row has exactly one FP32 scale.
kernel void richengine_fp8_quantize_kv_page(
    device const bfloat *source_keys [[buffer(0)]],
    device const bfloat *source_values [[buffer(1)]],
    device uchar *destination_keys [[buffer(2)]],
    device float *destination_key_scales [[buffer(3)]],
    device uchar *destination_values [[buffer(4)]],
    device float *destination_value_scales [[buffer(5)]],
    constant RichFp8KVPageParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (group >= 2 * RichFp8ScalesPerLayerPage ||
        thread_index >= RichFp8HeadDimension) {
        return;
    }
    bool value_group = group >= RichFp8ScalesPerLayerPage;
    uint local_group = value_group
        ? group - RichFp8ScalesPerLayerPage : group;
    uint token = local_group % RichFp8PageTokens;
    uint head = local_group / RichFp8PageTokens;
    uint dimension = thread_index;
    bool valid = token < min(params.valid_tokens, RichFp8PageTokens);
    ulong source_index = value_group
        ? richengine_fp8_value_data_index(
              params.source_page, head, token, dimension)
        : richengine_fp8_key_data_index(
              params.source_page, head, token, dimension);
    ulong destination_index = value_group
        ? richengine_fp8_value_data_index(
              params.destination_page, head, token, dimension)
        : richengine_fp8_key_data_index(
              params.destination_page, head, token, dimension);
    ulong scale_index = richengine_fp8_scale_index(
        params.destination_page, head, token);

    float value = valid ? float(
        value_group ? source_values[source_index] : source_keys[source_index])
        : 0.0f;
    float local_maximum = simd_max(abs(value));
    threadgroup float maxima[8];
    if (simd_lane == 0) maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (uint offset = 4; offset != 0; offset >>= 1) {
            if (simd_lane < offset) {
                maxima[simd_lane] = max(
                    maxima[simd_lane], maxima[simd_lane + offset]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = maxima[0] == 0.0f ? 0.0f : maxima[0] / 448.0f;
    uchar code = valid && maxima[0] != 0.0f
        ? richengine_fp8_encode(value * 448.0f / maxima[0]) : uchar(0);
    if (value_group) {
        destination_values[destination_index] = code;
        if (thread_index == 0) destination_value_scales[scale_index] = scale;
    } else {
        destination_keys[destination_index] = code;
        if (thread_index == 0) destination_key_scales[scale_index] = scale;
    }
}

// Native-oracle conversion back into BF16 physical layouts. This symbol is
// absent from the production metallib.
kernel void richengine_fp8_dequantize_kv_page(
    device const uchar *source_keys [[buffer(0)]],
    device const float *source_key_scales [[buffer(1)]],
    device const uchar *source_values [[buffer(2)]],
    device const float *source_value_scales [[buffer(3)]],
    device bfloat *destination_keys [[buffer(4)]],
    device bfloat *destination_values [[buffer(5)]],
    constant RichFp8KVPageParams &params [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= 2 * RichFp8ElementsPerLayerPage) return;
    bool value = index >= RichFp8ElementsPerLayerPage;
    uint local = value ? index - RichFp8ElementsPerLayerPage : index;
    uint head;
    uint token;
    uint dimension;
    if (!value) {
        dimension = local % RichFp8HeadDimension;
        uint row = local / RichFp8HeadDimension;
        token = row % RichFp8PageTokens;
        head = row / RichFp8PageTokens;
        ulong destination = richengine_fp8_key_data_index(
            params.destination_page, head, token, dimension);
        destination_keys[destination] = token < params.valid_tokens
            ? bfloat(richengine_fp8_load_key(
                  source_keys, source_key_scales, params.source_page,
                  head, token, dimension)) : bfloat(0.0f);
    } else {
        token = local % RichFp8PageTokens;
        uint column = local / RichFp8PageTokens;
        dimension = column % RichFp8HeadDimension;
        head = column / RichFp8HeadDimension;
        ulong destination = richengine_fp8_value_data_index(
            params.destination_page, head, token, dimension);
        destination_values[destination] = token < params.valid_tokens
            ? bfloat(richengine_fp8_load_value(
                  source_values, source_value_scales, params.source_page,
                  head, token, dimension)) : bfloat(0.0f);
    }
}

// Native-oracle logical gather. This symbol is absent from production.
kernel void richengine_fp8_gather_logical_kv_page(
    device const uchar *source_keys [[buffer(0)]],
    device const float *source_key_scales [[buffer(1)]],
    device const uchar *source_values [[buffer(2)]],
    device const float *source_value_scales [[buffer(3)]],
    device bfloat *logical_keys [[buffer(4)]],
    device bfloat *logical_values [[buffer(5)]],
    constant RichFp8KVPageParams &params [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= RichFp8ElementsPerLayerPage) return;
    uint dimension = index % RichFp8HeadDimension;
    uint row = index / RichFp8HeadDimension;
    uint head = row % RichFp8KVHeads;
    uint token = row / RichFp8KVHeads;
    bool valid = token < min(params.valid_tokens, RichFp8PageTokens);
    logical_keys[index] = valid ? bfloat(richengine_fp8_load_key(
        source_keys, source_key_scales, params.source_page,
        head, token, dimension)) : bfloat(0.0f);
    logical_values[index] = valid ? bfloat(richengine_fp8_load_value(
        source_values, source_value_scales, params.source_page,
        head, token, dimension)) : bfloat(0.0f);
}

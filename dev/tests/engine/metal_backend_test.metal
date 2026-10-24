#include <metal_stdlib>

using namespace metal;

kernel void test_copy_u32(device const uint *source [[buffer(0)]],
                         device uint *destination [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        destination[gid] = source[gid];
    }
}

kernel void test_add_u32(device uint *values [[buffer(0)]],
                       constant uint &count [[buffer(1)]],
                       constant uint &increment [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        values[gid] += increment;
    }
}

struct TestFillParams {
    uint value;
    uint count;
};

kernel void test_fill_u32(device uint *destination [[buffer(0)]],
                          constant TestFillParams &params [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < params.count) {
        destination[gid] = params.value;
    }
}

// Row r of the grid writes word x of the buffer at table[r], which no
// dispatch binds; addressed_check_u32 reads it back the same way.
kernel void addressed_write_u32(device const ulong *table [[buffer(0)]],
                                constant uint &words [[buffer(1)]],
                                constant uint &seed [[buffer(2)]],
                                uint2 id [[thread_position_in_grid]]) {
    if (id.x < words) {
        reinterpret_cast<device uint *>(table[id.y])[id.x] =
            seed ^ (id.y * 131071u + id.x);
    }
}

kernel void addressed_check_u32(device const ulong *table [[buffer(0)]],
                                constant uint &words [[buffer(1)]],
                                constant uint &seed [[buffer(2)]],
                                device atomic_uint *mismatches [[buffer(3)]],
                                uint2 id [[thread_position_in_grid]]) {
    if (id.x < words &&
        reinterpret_cast<device const uint *>(table[id.y])[id.x] !=
            (seed ^ (id.y * 131071u + id.x))) {
        atomic_fetch_add_explicit(mismatches, 1u, memory_order_relaxed);
    }
}

// The stitchable hook a stitched pipeline binds: dispatch
// "test_add_hook_u32[select=impl:1]". The impl's signature matches the
// hook's declared one.
[[stitchable]] uint test_select_addend(uint base);

[[stitchable]] uint test_addend_one(uint base) { return base + 1u; }
[[stitchable]] uint test_addend_seven(uint base) { return base + 7u; }

kernel void test_add_hook_u32(device uint *values [[buffer(0)]],
                              constant uint &count [[buffer(1)]],
                              constant uint &increment [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        values[gid] += test_select_addend(increment);
    }
}

// Indirect-dispatch fixtures: test_write_grid stores a
// MTLDispatchThreadgroupsIndirectArguments, test_write_range an
// MTLIndirectCommandBufferExecutionRange (location, length).
kernel void test_write_grid(device uint3 *grid [[buffer(0)]],
                            constant uint3 &value [[buffer(1)]]) {
    *grid = value;
}

kernel void test_write_range(device uint2 *range [[buffer(0)]],
                             constant uint2 &value [[buffer(1)]]) {
    *range = value;
}

// Every threadgroup writes its grid position into the row it owns, so a
// GPU-sized grid's actual launch shape is observable afterwards.
kernel void test_mark_grid_u32(device uint *marks [[buffer(0)]],
                               constant uint &limit [[buffer(1)]],
                               uint2 tgid [[threadgroup_position_in_grid]]) {
    if (tgid.x < limit && tgid.y < 4u) {
        marks[tgid.y * 64u + tgid.x] = 0x5afed00du + tgid.y * 131071u + tgid.x;
    }
}

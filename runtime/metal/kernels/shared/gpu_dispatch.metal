// The GPU-driven dispatch encoder (RICHENGINE_GPU_SPANS): one thread per
// command of a gpu span writes its indirect compute command — pipeline,
// buffer bindings, grid — into the span's indirect command buffer, which
// the same encoder then replays after a device-buffer memory barrier. Runs
// the host would otherwise encode dispatch by dispatch — the suspended
// GGUF spans of ops/MoE.cpp and ops/LinearGguf.cpp — move their encoding
// onto the GPU.
#include <metal_stdlib>
#include <metal_command_buffer>
#include "../../abi/GpuDispatch.h"

using namespace metal;

// The argument buffer MetalBackend::Impl::bakeGpuSpan fills: the span's ICB
// at index 0, its distinct compute pipelines at indices 1..32 — every
// pipeline created with supportIndirectCommandBuffers.
struct GpuEncodeArgs {
  command_buffer icb [[id(0)]];
  array<compute_pipeline_state, kGpuSpanPipelines> pipelines [[id(1)]];
};

kernel void gpu_encode(device GpuEncodeArgs *args [[buffer(0)]],
                       device const GpuDispatchDesc *descs [[buffer(1)]],
                       constant uint32_t &count [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
  if (tid >= count) return;
  device const GpuDispatchDesc &d = descs[tid];
  compute_command cmd(args->icb, tid);
  cmd.set_compute_pipeline_state(args->pipelines[d.pipeline]);
  for (uint32_t i = 0; i < d.bindings; ++i)
    cmd.set_kernel_buffer(reinterpret_cast<device void *>(d.buffers[i]),
                          d.binding_index[i]);
  // Serial ordering like the baked spans: every command but the first waits
  // on all commands before it.
  if (tid) cmd.set_barrier();
  cmd.concurrent_dispatch_threadgroups({d.groups_x, d.groups_y, d.groups_z},
                                       {d.threads_x, d.threads_y, d.threads_z});
}

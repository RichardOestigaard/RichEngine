#pragma once

// The GPU-encoded span ABI (shared/gpu_dispatch.metal's gpu_encode kernel,
// MetalBackend::Impl::bakeGpuSpan): the host stages one descriptor per
// dispatch of a gpu span — a run that cannot ride a baked indirect command
// buffer — and the encoder kernel writes one indirect compute command into
// the span's ICB per thread, whereupon the same encoder replays it. Scalar
// fields only: Metal aligns vector types to 16 bytes, so grids ride three
// uints.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

enum {
  // The argument buffer's pipeline table (gpu_encode's GpuEncodeArgs): a
  // gpu span's commands pick their pipeline by index into it. A run with
  // more distinct pipelines encodes directly instead.
  kGpuSpanPipelines = 32,
  // The most bindings (buffers and staged-parameter addresses alike) one
  // encodable command may carry.
  kGpuSpanBindings = 24,
  // The descriptor table's byte offset inside its staging buffer: the
  // first four bytes hold the run's command count.
  kGpuSpanDescHeader = 256,
};

struct GpuDispatchDesc {
  // The command's index into the span's pipeline table.
  uint32_t pipeline;
  uint32_t groups_x, groups_y, groups_z;
  uint32_t threads_x, threads_y, threads_z;
  // The (address, index) pairs the command binds. Parameter payloads ride
  // addresses too — they are staged into the span's parameter arena.
  uint32_t bindings;
  uint32_t reserved;
  uint64_t buffers[kGpuSpanBindings];
  uint8_t binding_index[kGpuSpanBindings];
};

static_assert(sizeof(GpuDispatchDesc) == 256,
              "GPU dispatch descriptors are 256 bytes on both sides");

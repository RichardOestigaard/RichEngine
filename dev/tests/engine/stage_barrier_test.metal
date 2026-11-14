// stage_barrier.h's self-test kernels: a mutual barrier across every
// worker threadgroup of one dispatch, and a producer/consumer pair split
// over two dispatches of one command buffer. Host: stage_barrier_test.mm.
#include "metal/kernels/common/stage_barrier.h"
#include <metal_stdlib>

using namespace metal;

constant uint kBarrierThreads = 64;

struct BarrierTestParams {
  uint epoch;
  uint workers;
};

// Mutual barrier: each worker threadgroup writes its slice of `data`,
// arrives and waits; afterwards every thread re-reads every worker's slice
// and adds its mismatches into `bad`. `ok` gets the sync verdict per worker.
kernel void test_stage_barrier_sync(
    device StageBarrier *barrier [[buffer(0)]],
    device uint *data [[buffer(1)]],
    device atomic_uint *bad [[buffer(2)]],
    device uint *ok [[buffer(3)]],
    constant BarrierTestParams &params [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint worker = group.x;
  const uint threads = kBarrierThreads;
  device uint *mine = data + worker * threads;
  for (uint i = tid; i < threads; i += threads)
    mine[i] = worker * 4096u + i + 1u;
  const bool released =
      stage_barrier_sync(barrier, worker, params.workers, params.epoch,
                         kStageBarrierSpins, tid);
  if (tid == 0)
    ok[worker] = released ? 1u : 0u;
  uint mismatches = 0;
  for (uint w = 0; w < params.workers; ++w)
    for (uint i = tid; i < threads; i += threads)
      if (data[w * threads + i] != w * 4096u + i + 1u)
        ++mismatches;
  if (mismatches)
    atomic_fetch_add_explicit(bad, mismatches, memory_order_relaxed);
}

// Producer half of the pair: each worker writes its slice and arrives, then
// leaves — the consumer dispatch's poll is what releases it. The delay
// keeps the arrival late enough for the consumer's poll to actually spin.
kernel void test_stage_barrier_produce(
    device StageBarrier *barrier [[buffer(0)]],
    device uint *data [[buffer(1)]],
    device uint *sink [[buffer(2)]],
    constant BarrierTestParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint worker = group.x;
  uint x = tid + 1u;
  for (uint i = 0; i < 100000u; ++i)
    x = x * 1664525u + 1013904223u;
  device uint *mine = data + worker * kBarrierThreads;
  for (uint i = tid; i < kBarrierThreads; i += kBarrierThreads)
    mine[i] = worker * 4096u + i + 1u;
  sink[worker] = x;
  stage_barrier_arrive(barrier, worker, params.epoch, tid);
}

// Consumer half: ONE simdgroup's first thread polls every producer flag,
// then every thread waits the release and verifies the produced slices.
kernel void test_stage_barrier_consume(
    device StageBarrier *barrier [[buffer(0)]],
    device const uint *data [[buffer(1)]],
    device atomic_uint *bad [[buffer(2)]],
    device uint *ok [[buffer(3)]],
    constant BarrierTestParams &params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (tid == 0)
    static_cast<void>(
        stage_barrier_poll(barrier, params.workers, params.epoch,
                           kStageBarrierSpins));
  const bool released =
      stage_barrier_wait(barrier, params.epoch, kStageBarrierSpins * 33u);
  uint mismatches = 0;
  for (uint w = 0; w < params.workers; ++w)
    for (uint i = tid; i < kBarrierThreads; i += 32u)
      if (data[w * kBarrierThreads + i] != w * 4096u + i + 1u)
        ++mismatches;
  if (mismatches)
    atomic_fetch_add_explicit(bad, mismatches, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0)
    ok[0] = released && atomic_load_explicit(bad, memory_order_acquire, mem_flags::mem_device) == 0u
                ? 1u
                : 0u;
}

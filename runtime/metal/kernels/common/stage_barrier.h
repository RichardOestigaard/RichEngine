#pragma once
#include <metal_stdlib>
using namespace metal;

// Cross-dispatch stage barrier for the GPU-resident decode loop: workers of
// one command (or of two overlapping commands) meet at a stage boundary
// without a host round trip. The state lives in device memory as a
// StageBarrier block the host zeroes once; each round's epoch distinguishes
// its arrivals and release from every earlier round's, so the block needs
// no reset between uses. `error` is the exception: it is sticky, and the
// host clears it (or a kernel does, once per round) before the next use.
//
// The contract, per round of `workers` workers (worker indices 0..workers-1,
// at most kStageBarrierMaxWorkers):
//
//   stage_barrier_arrive  every thread of a worker, when its stage's writes
//                         are done: the threadgroup fence publishes them
//                         and thread 0 stores the arrival epoch into the
//                         worker's flag.
//   stage_barrier_poll    ONE thread — worker 0's first, or a dedicated
//                         consumer dispatch's — spins every other worker's
//                         flag until all reach `epoch`, bounded by `spins`.
//                         Success stores `released = epoch`; `spins`
//                         exhausted stores the first missing worker's index
//                         + 1 into `error` and `released = epoch |
//                         kStageBarrierFail`, so waiters leave instead of
//                         hanging. Metal gives no threadgroup residency
//                         guarantee: an unscheduled worker must degrade to
//                         this bounded failure, never a hang.
//   stage_barrier_wait    every thread of every worker, bounded by `spins`:
//                         true once `released` reaches `epoch`, false on the
//                         failure marker or on its own bound. The bound must
//                         outlast the poller's `spins` scans of `workers`
//                         flags — stage_barrier_sync sizes it.
//
// Epochs must differ between rounds and stay below kStageBarrierFail.
// `thread_index` is the caller's thread_index_in_threadgroup.
constant uint kStageBarrierMaxWorkers = 32;
constant uint kStageBarrierFail = 0x80000000u;
constant uint kStageBarrierSpins = 1000000;

struct StageBarrier {
  // worker -> the epoch it last arrived at (0 = never).
  atomic_uint flags[kStageBarrierMaxWorkers];
  // The epoch the poller last released, epoch | kStageBarrierFail on a
  // bounded failure.
  atomic_uint released;
  // The first missing worker's index + 1 the poller reported, sticky.
  atomic_uint error;
};

inline void stage_barrier_arrive(device StageBarrier *barrier, uint worker,
                                 uint epoch, uint thread_index) {
  // Every writer publishes its stage before thread zero signals arrival.
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    atomic_store_explicit(&barrier->flags[worker], epoch,
                          memory_order_release, mem_flags::mem_device);
  }
}

inline bool stage_barrier_poll(device StageBarrier *barrier, uint workers,
                               uint epoch, uint spins) {
  if (workers == 0 || workers > kStageBarrierMaxWorkers)
    return false;
  uint missing = workers;
  for (uint spin = 0; spin < spins && missing != workers; ++spin) {
    missing = workers;
    for (uint worker = 0; worker < workers; ++worker)
      if (atomic_load_explicit(&barrier->flags[worker], memory_order_acquire,
                               mem_flags::mem_device) != epoch) {
        missing = worker;
        break;
      }
  }
  if (missing == workers) {
    atomic_store_explicit(&barrier->released, epoch, memory_order_release,
                          mem_flags::mem_device);
    return true;
  }
  atomic_store_explicit(&barrier->error, missing + 1, memory_order_relaxed,
                        mem_flags::mem_device);
  atomic_store_explicit(&barrier->released, epoch | kStageBarrierFail,
                        memory_order_release, mem_flags::mem_device);
  return false;
}

inline bool stage_barrier_wait(device StageBarrier *barrier, uint epoch,
                               uint spins) {
  for (uint spin = 0; spin < spins; ++spin) {
    const uint seen = atomic_load_explicit(
        &barrier->released, memory_order_acquire, mem_flags::mem_device);
    if (seen == epoch)
      return true;
    if (seen == (epoch | kStageBarrierFail))
      return false;
  }
  return false;
}

// The common mutual barrier: arrive, worker 0's first thread polls for the
// rest, every thread waits out the release. Returns the wait's verdict.
inline bool stage_barrier_sync(device StageBarrier *barrier, uint worker,
                               uint workers, uint epoch, uint spins,
                               uint thread_index) {
  if (worker >= workers || workers > kStageBarrierMaxWorkers)
    return false;
  stage_barrier_arrive(barrier, worker, epoch, thread_index);
  if (worker == 0 && thread_index == 0)
    static_cast<void>(stage_barrier_poll(barrier, workers, epoch, spins));
  // The poller's worst case is `spins` scans of `workers` flags; the wait
  // gets the same budget counted in single-flag loads.
  return stage_barrier_wait(barrier, epoch, spins * (workers + 1));
}

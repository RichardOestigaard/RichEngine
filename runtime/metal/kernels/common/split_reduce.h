#pragma once
#include <metal_stdlib>
using namespace metal;

// Cross-threadgroup reduction of a K split, shared by the split decode
// kernels. Each of the `splits` threadgroups of one output tile stores its
// fp32 partials, in its kernel's own layout, through a device
// coherent(device) pointer and then arrives at the tile's counter. One
// threadgroup adds the splits in split order, its own from registers, and
// returns the counter to zero for the next dispatch — the last to arrive on
// the fallback path, the first (parked on the counter until the last
// arrival's notify) with MSL 4.1 wait/notify. The result does not depend on
// scheduling.

// True in every thread of the threadgroup that ends up owning the merge.
// With MSL 4.1 wait/notify (RICHENGINE_SPLIT_WAIT_NOTIFY) the owner is the
// first threadgroup to arrive: it parks on the counter until the last
// arrival's notify, so the reducer sleeps instead of holding a seq_cst
// polling loop. Without it the owner is the last to arrive and no
// threadgroup waits at all. Either way exactly one threadgroup merges, the
// merge still happens after every split's partials are visible, and
// split_sum adds them in split order — the arithmetic, and therefore the
// bitwise result, is identical.
//
// The wait/notify path is opt-in via -DRICHENGINE_SPLIT_WAIT_NOTIFY: using
// it marks the kernel SIMDGroupParallelForwardProgress, which at least
// Apple-family-10 devices reject at pipeline creation ("feature is used in
// an unsupported shader stage and/or device"), so it cannot be the default.
inline bool split_arrive_last(device atomic_uint *counter, uint splits,
                              uint thread_index, threadgroup uint *arrival) {
  // Every writer publishes its partials before thread zero signals arrival.
#if defined(RICHENGINE_SPLIT_WAIT_NOTIFY) && \
    defined(__HAVE_ATOMIC_WAIT_NOTIFY__)
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0)
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_release,
                                         mem_flags::mem_device);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint order = *arrival;
  // The last arrival brings the counter to `splits` and wakes the parked
  // reducer. `order` is threadgroup-uniform, so whole simdgroups notify.
  if (order == splits - 1)
    atomic_notify_all_simdgroup(counter);
  if (order != 0)
    return false;
  // The first arrival merges: park until the counter reaches `splits`. A
  // notify that lands before the wait is not missed — atomic_wait rechecks
  // the value before sleeping and yields return here on spurious wakeups.
  for (uint seen = atomic_load_explicit(counter, memory_order_acquire,
                                        mem_flags::mem_device);
       seen < splits;
       seen = atomic_wait_value_explicit_simdgroup(
                  counter, seen, memory_order_acquire,
                  mem_flags::mem_device)) {
#if defined(__HAVE_YIELD_SIMDGROUP__)
    yield_simdgroup(64);
#endif
  }
  return true;
#else
  // The device-scope fences pair through the counter.
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
#endif
}

// The splits' partials added in split order: `own` for split `self`,
// `partial(s)` for every other split s.
template <typename T, typename Partial>
inline T split_sum(T own, uint self, uint splits, Partial partial) {
#pragma clang fp reassociate(off)
  T total = T(0);
  for (uint s = 0; s < splits; ++s)
    total += s == self ? own : partial(s);
  return total;
}

// Returns the tile's counter to zero; called by the reducing threadgroup.
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0)
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

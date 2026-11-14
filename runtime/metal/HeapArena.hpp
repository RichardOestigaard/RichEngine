#pragma once

// HeapArena.hpp: placement-heap suballocation for transient scratch
// buffers, behind RICHENGINE_HEAP_ARENA=1 (wired through
// MetalArena.mm's Impl::newBufferPlaced). One HeapArena owns a set of
// same-configuration MTLHeapTypePlacement heaps; allocate() carves a
// buffer out of a recycled free range or a heap's bump frontier, and
// freeRange() returns the range when the owning MetalAllocation detaches,
// so the next transient allocation may alias the same physical pages.
//
// Lifetime contract: a range returns to the free list only when the last
// MetalBuffer view of its allocation is gone AND every command that bound
// it has completed — CommandTicket::State::retainedAllocations holds each
// bound allocation until its ticket is consumed, which happens after GPU
// completion. Buffers reached only through GPU addresses (never bound)
// keep the caller's existing keep-the-view-alive contract; the residency
// set sees every heap-backed buffer as an ordinary MTLBuffer, so nothing
// here bypasses residency tracking.
//
// Hazard tracking: heaps default to MTLHazardTrackingModeUntracked —
// a tracked heap delays every access to ANY suballocated resource behind
// every other heap resource's in-flight modification, which would
// serialize exactly the parallel scratch work the arena exists to alias.
// RICHENGINE_HEAP_ARENA_TRACKED=1 creates tracked heaps for the A/B that
// prices the fences. Untracked aliasing is safe because a range returns
// to the free list only after its previous buffer's commands completed
// (the ticket retains bound allocations until consumption) and after the
// last view died; accesses Metal cannot see — buffers reached only
// through GPU addresses — keep the caller's existing
// keep-the-view-alive contract either way.

#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace richengine::metal {

class HeapArena final {
public:
    // What allocate() hands back and freeRange() takes back: the buffer
    // and the heap range it occupies. `size` is the requiredSize from
    // heapBufferSizeAndAlignWithLength — it can exceed the buffer's length
    // by alignment padding, and the whole range is what gets recycled.
    struct Placement {
        id<MTLBuffer> buffer = nil;
        std::shared_ptr<HeapArena> arena;
        id<MTLHeap> heap = nil;
        uint64_t offset = 0;
        uint64_t size = 0;
    };

    // `hazardTrackingMode` is stamped on every heap descriptor;
    // `growBytes` is the size of each new heap unless the request itself
    // needs more.
    HeapArena(id<MTLDevice> device, MTLStorageMode storage,
              MTLCPUCacheMode cpuCache,
              MTLHazardTrackingMode hazardTrackingMode, uint64_t growBytes)
        : device_(device),
          storage_(storage),
          cpuCache_(cpuCache),
          hazardTrackingMode_(hazardTrackingMode),
          growBytes_(growBytes) {}

    // A buffer suballocated from one of the arena's heaps, or an empty
    // Placement when no heap could serve the request — the caller then
    // allocates directly, so a heap failure never fails the allocation.
    Placement allocate(uint64_t bytes);

    // Returns a range allocate() handed out. Called from
    // MetalAllocation::detach() on whatever thread drops the last view —
    // hence noexcept and the mutex; a failed insert leaks the range inside
    // its heap rather than propagating through a destructor.
    void freeRange(id<MTLHeap> heap, uint64_t offset,
                   uint64_t size) noexcept;

private:
    struct FreeRange {
        uint64_t offset = 0;
        uint64_t size = 0;
    };

    struct HeapSlot {
        __strong id<MTLHeap> heap = nil;
        // Next never-allocated offset inside `heap`.
        uint64_t bump = 0;
        // Sorted by offset, non-overlapping, adjacent ranges merged.
        std::vector<FreeRange> free;

        // First-fit carve of `required` bytes at an `alignment` boundary
        // out of the lowest-offset free range that holds it.
        bool takeFree(uint64_t required, uint64_t alignment,
                      uint64_t *offset);
        // Carve off the bump frontier when it still fits inside the heap.
        bool takeBump(uint64_t required, uint64_t alignment,
                      uint64_t *offset);
        // Sorted insert with adjacency/overlap merge, so the free list
        // stays small however allocations cycle.
        void insertFree(uint64_t offset, uint64_t size);
    };

    static uint64_t alignUp(uint64_t value, uint64_t alignment) noexcept;
    MTLResourceOptions resourceOptions() const;
    Placement place(HeapSlot &slot, uint64_t bytes,
                    MTLResourceOptions options, uint64_t offset,
                    uint64_t required);

    __strong id<MTLDevice> device_ = nil;
    MTLStorageMode storage_ = MTLStorageModePrivate;
    MTLCPUCacheMode cpuCache_ = MTLCPUCacheModeDefaultCache;
    MTLHazardTrackingMode hazardTrackingMode_ =
        MTLHazardTrackingModeUntracked;
    uint64_t growBytes_ = 0;
    std::mutex mutex_;
    std::vector<HeapSlot> heaps_;
};

inline uint64_t HeapArena::alignUp(uint64_t value,
                                   uint64_t alignment) noexcept {
    if (!alignment) return value;
    const uint64_t remainder = value % alignment;
    if (!remainder) return value;
    // The addition must not wrap: a wrapped offset would pass every later
    // bound check while pointing outside the heap.
    if (value >
        std::numeric_limits<uint64_t>::max() - (alignment - remainder)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return value + (alignment - remainder);
}

inline MTLResourceOptions HeapArena::resourceOptions() const {
    MTLResourceOptions options = storage_ == MTLStorageModeShared
        ? MTLResourceStorageModeShared
        : MTLResourceStorageModePrivate;
    if (storage_ == MTLStorageModeShared &&
        cpuCache_ == MTLCPUCacheModeWriteCombined) {
        options |= MTLResourceCPUCacheModeWriteCombined;
    }
    return options;
}

inline bool HeapArena::HeapSlot::takeFree(uint64_t required,
                                          uint64_t alignment,
                                          uint64_t *offset) {
    for (size_t i = 0; i < free.size(); ++i) {
        const uint64_t rangeOffset = free[i].offset;
        const uint64_t rangeEnd = rangeOffset + free[i].size;
        const uint64_t start = alignUp(rangeOffset, alignment);
        if (start >= rangeEnd || required > rangeEnd - start) continue;
        const uint64_t finish = start + required;
        free.erase(free.begin() + static_cast<std::ptrdiff_t>(i));
        insertFree(finish, rangeEnd - finish);
        insertFree(rangeOffset, start - rangeOffset);
        *offset = start;
        return true;
    }
    return false;
}

inline bool HeapArena::HeapSlot::takeBump(uint64_t required,
                                          uint64_t alignment,
                                          uint64_t *offset) {
    const uint64_t start = alignUp(bump, alignment);
    const uint64_t heapSize = heap.size;
    if (start >= heapSize || required > heapSize - start) return false;
    bump = start + required;
    *offset = start;
    return true;
}

inline void HeapArena::HeapSlot::insertFree(uint64_t offset, uint64_t size) {
    if (!size) return;
    free.push_back({offset, size});
    std::sort(free.begin(), free.end(),
              [](const FreeRange &a, const FreeRange &b) {
                  return a.offset < b.offset;
              });
    size_t kept = 0;
    for (size_t i = 0; i < free.size(); ++i) {
        const FreeRange range = free[i];
        if (kept && free[kept - 1].offset + free[kept - 1].size >=
                       range.offset) {
            FreeRange &last = free[kept - 1];
            last.size = std::max(last.offset + last.size,
                                 range.offset + range.size) -
                        last.offset;
        } else {
            free[kept++] = range;
        }
    }
    free.resize(kept);
}

inline HeapArena::Placement HeapArena::place(
        HeapSlot &slot, uint64_t bytes, MTLResourceOptions options,
        uint64_t offset, uint64_t required) {
    Placement result;
    result.buffer = [slot.heap
        newBufferWithLength:bytes options:options offset:offset];
    if (!result.buffer) return result;
    result.heap = slot.heap;
    result.offset = offset;
    result.size = required;
    return result;
}

inline HeapArena::Placement HeapArena::allocate(uint64_t bytes) {
    Placement result;
    if (!device_ || !bytes) return result;
    const MTLResourceOptions options = resourceOptions();
    const MTLSizeAndAlign needed =
        [device_ heapBufferSizeAndAlignWithLength:bytes options:options];
    const uint64_t required = needed.size;
    const uint64_t alignment = needed.align ? needed.align : 1;
    if (!required) return result;
    std::lock_guard lock(mutex_);
    // Recycled ranges first: they are the aliasing the arena exists for.
    for (HeapSlot &slot : heaps_) {
        uint64_t offset = 0;
        if (slot.takeFree(required, alignment, &offset)) {
            result = place(slot, bytes, options, offset, required);
            if (result.buffer) return result;
        }
    }
    // Then each heap's bump frontier.
    for (HeapSlot &slot : heaps_) {
        uint64_t offset = 0;
        if (slot.takeBump(required, alignment, &offset)) {
            result = place(slot, bytes, options, offset, required);
            if (result.buffer) return result;
        }
    }
    // Nothing fit: grow with a new heap sized for this request or the
    // arena's quantum, whichever is larger. A creation failure leaves
    // `result` empty and the caller allocates directly.
    const uint64_t heapSize = std::max(growBytes_, required);
    MTLHeapDescriptor *descriptor = [MTLHeapDescriptor new];
    descriptor.type = MTLHeapTypePlacement;
    descriptor.storageMode = storage_;
    descriptor.cpuCacheMode = cpuCache_;
    descriptor.hazardTrackingMode = hazardTrackingMode_;
    descriptor.size = heapSize;
    id<MTLHeap> heap = [device_ newHeapWithDescriptor:descriptor];
    if (!heap) return result;
    heaps_.emplace_back();
    HeapSlot &slot = heaps_.back();
    slot.heap = heap;
    uint64_t offset = 0;
    if (!slot.takeBump(required, alignment, &offset)) return result;
    return place(slot, bytes, options, offset, required);
}

inline void HeapArena::freeRange(id<MTLHeap> heap, uint64_t offset,
                                 uint64_t size) noexcept {
    if (!heap || !size) return;
    try {
        std::lock_guard lock(mutex_);
        for (HeapSlot &slot : heaps_) {
            if (slot.heap == heap) {
                slot.insertFree(offset, size);
                return;
            }
        }
    } catch (...) {
        // A failed free leaks the range inside its heap; the arena keeps
        // serving the rest.
    }
}

} // namespace richengine::metal

// MetalArena.mm: allocation, views and the residency/release path.

#import "MetalBackend.hpp"
#include "AwakeClock.hpp"
#include "CommandWatchdog.hpp"
#include "Env.hpp"
#include "Residency.hpp"
#include "TestConfig.hpp"
#ifdef RICHENGINE_BACKEND_INSTRUMENTATION
#include "BackendInstrumentation.hpp"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Metal 4 command encoding exists from macOS 26, below the engine's macOS
// 27 floor, so header availability is the only gate.
#if __has_include(<Metal/MTL4CommandQueue.h>)
#define RICHENGINE_MTL4_AVAILABLE 1
#else
#define RICHENGINE_MTL4_AVAILABLE 0
#endif

#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <unistd.h>

#include "metal/MetalBackendImpl.hpp"

namespace richengine::metal {

namespace {

// RICHENGINE_HEAP_ARENA=1 opts allocateBuffer into placement-heap
// suballocation (HeapArena.hpp): transient scratch aliases physical
// pages. Every knob is read once; the defaults keep the experiment
// conservative and the whole path falls back to direct device
// allocations on any heap failure.
bool heapArenaEnabled() {
    static const bool enabled = envFlagOn("RICHENGINE_HEAP_ARENA");
    return enabled;
}
uint64_t heapArenaMinBytes() {
    static const uint64_t value =
        envUint("RICHENGINE_HEAP_ARENA_MIN_BYTES", 256 * 1024);
    return value;
}
uint64_t heapArenaMaxBytes() {
    static const uint64_t value =
        envUint("RICHENGINE_HEAP_ARENA_MAX_BYTES", 1024 * 1024 * 1024);
    return value;
}
uint64_t heapArenaGrowBytes() {
    static const uint64_t value =
        static_cast<uint64_t>(
            envUint("RICHENGINE_HEAP_ARENA_GROW_MB", 512)) <<
        20;
    return value;
}
MTLHazardTrackingMode heapArenaHazardTracking() {
    // Untracked by default: a tracked heap serializes every heap
    // resource's accesses against every other's, which would serialize
    // the parallel scratch work aliasing exists to enable. TRACKED=1 is
    // the A/B that prices the fences.
    static const MTLHazardTrackingMode mode =
        envFlagOn("RICHENGINE_HEAP_ARENA_TRACKED")
            ? MTLHazardTrackingModeTracked
            : MTLHazardTrackingModeUntracked;
    return mode;
}
bool heapArenaWriteCombined() {
    static const bool enabled = envFlagOn("RICHENGINE_HEAP_ARENA_WC");
    return enabled;
}
uint64_t heapArenaWriteCombinedMinBytes() {
    static const uint64_t value =
        envUint("RICHENGINE_HEAP_ARENA_WC_MIN_BYTES", 32 * 1024 * 1024);
    return value;
}

} // namespace

id<MTLBuffer> MetalBackend::Impl::newBuffer(uint64_t bytes, BufferStorage storage,
                            NSString *label) {
        MTLResourceOptions options = storage == BufferStorage::Shared
            ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes
                                                   options:options];
        if (!buffer)
            throw MetalAllocationError("Metal buffer allocation failed");
        if (label) buffer.label = label;
        return buffer;
    }

id<MTLBuffer> MetalBackend::Impl::newBufferPlaced(
        uint64_t bytes, BufferStorage storage, NSString *label,
        HeapArena::Placement *placement) {
        if (placement) *placement = HeapArena::Placement{};
        const uint64_t maxBytes = heapArenaMaxBytes();
        // A heap placement is only useful when the caller reports it back:
        // without `placement` the range could never return to the free
        // list, so the buffer allocates directly instead.
        if (placement && heapArenaEnabled() &&
            bytes >= heapArenaMinBytes() &&
            (!maxBytes || bytes <= maxBytes)) {
            if (std::shared_ptr<HeapArena> arena =
                    heapArenaFor(storage, bytes)) {
                HeapArena::Placement placed = arena->allocate(bytes);
                if (placed.buffer) {
                    placed.arena = arena;
                    if (label) placed.buffer.label = label;
                    if (placement) *placement = placed;
                    return placed.buffer;
                }
            }
        }
        return newBuffer(bytes, storage, label);
    }

std::shared_ptr<HeapArena> MetalBackend::Impl::heapArenaFor(
        BufferStorage storage, uint64_t bytes) {
        // heapArenas: [0] Private, [1] Shared default-cache, [2] Shared
        // write-combined — the last only for buffers past the WC floor
        // when RICHENGINE_HEAP_ARENA_WC=1 (large read-mostly Shared
        // buffers).
        const bool writeCombined =
            storage == BufferStorage::Shared && heapArenaWriteCombined() &&
            bytes >= heapArenaWriteCombinedMinBytes();
        const size_t index = storage == BufferStorage::Private
            ? 0
            : (writeCombined ? 2 : 1);
        std::shared_ptr<HeapArena> &arena = heapArenas[index];
        if (!arena) {
            arena = std::make_shared<HeapArena>(
                device,
                storage == BufferStorage::Private ? MTLStorageModePrivate
                                                  : MTLStorageModeShared,
                index == 2 ? MTLCPUCacheModeWriteCombined
                           : MTLCPUCacheModeDefaultCache,
                heapArenaHazardTracking(), heapArenaGrowBytes());
        }
        return arena;
    }

// The base allocation of a buffer, which must be a whole buffer of this
    // backend.
    MetalAllocation &MetalBackend::Impl::baseAllocation(const MetalBuffer &buffer) const {
        if (!buffer.impl_ ||
            buffer.impl_->allocation->accounting.get() != accounting.get())
            throw MetalBackendError("Metal buffer is not a buffer of this backend");
        MetalAllocation &allocation = *buffer.impl_->allocation;
        if (buffer.impl_->offsetBytes ||
            buffer.impl_->lengthBytes != allocation.length)
            throw MetalBackendError("a view's memory is its base buffer's");
        return allocation;
    }

MetalBuffer MetalBackend::allocateBuffer(uint64_t bytes,
                                         BufferStorage storage,
                                         std::string_view label) {
    checkOperation();
    if (!bytes) throw MetalBackendError("Metal buffer size must be positive");
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }
    auto allocation = std::make_shared<MetalAllocation>();
    allocation->accounting = impl_->accounting;
    allocation->length = bytes;
    allocation->storage = storage;
    if (!label.empty()) allocation->label = checkedNSString(label, "buffer label");
    allocation->residency = impl_->residency;
    HeapArena::Placement placement;
    allocation->attach(impl_->newBufferPlaced(bytes, storage,
                                              allocation->label,
                                              &placement));
    allocation->heapArena = std::move(placement.arena);
    allocation->heap = placement.heap;
    allocation->heapOffset = placement.offset;
    allocation->heapSize = placement.size;
    impl_->sampleDeviceMemory();
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->lengthBytes = bytes;
    result->allocation = std::move(allocation);
    return MetalBuffer(std::move(result));
}

MetalBuffer MetalBackend::wrapSharedMemory(void *address, uint64_t bytes,
                                           std::shared_ptr<void> owner,
                                           std::string_view label) {
    checkOperation();
    const uint64_t page = static_cast<uint64_t>(getpagesize());
    if (!address || !bytes || !owner ||
        reinterpret_cast<uintptr_t>(address) % page || bytes % page) {
        throw MetalBackendError(
            "wrapped memory needs an owner and whole pages");
    }
    if (bytes > impl_->capabilities.maxBufferLengthBytes) {
        throw MetalBackendError("Metal buffer exceeds maxBufferLength");
    }
    auto allocation = std::make_shared<MetalAllocation>();
    allocation->accounting = impl_->accounting;
    allocation->length = bytes;
    allocation->owner = owner;
    if (!label.empty()) allocation->label = checkedNSString(label, "buffer label");
    allocation->residency = impl_->residency;
    // Metal may keep the buffer past our last view, so its deallocator holds
    // the owner too.
    id<MTLBuffer> buffer = [impl_->device
        newBufferWithBytesNoCopy:address
                          length:bytes
                         options:MTLResourceStorageModeShared
                     deallocator:^(void *, NSUInteger) { (void)owner; }];
    if (!buffer)
        throw MetalAllocationError("zero-copy Metal buffer creation failed");
    if (allocation->label) buffer.label = allocation->label;
    allocation->attach(buffer);
    impl_->sampleDeviceMemory();
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->lengthBytes = bytes;
    result->allocation = std::move(allocation);
    return MetalBuffer(std::move(result));
}

void MetalBackend::releaseMemory(const MetalBuffer &buffer) {
    MetalAllocation &allocation = impl_->baseAllocation(buffer);
    if (allocation.owner)
        throw MetalBackendError("wrapped memory is its owner's to release");
    if (!allocation.buffer)
        throw MetalBackendError("Metal buffer memory is already released");
    if (commandInFlight())
        throw MetalBackendError(
            "Metal buffer memory is released while a command is in flight");
    allocation.detach();
    impl_->sampleDeviceMemory();
}

void MetalBackend::restoreMemory(const MetalBuffer &buffer) {
    checkOperation();
    MetalAllocation &allocation = impl_->baseAllocation(buffer);
    if (allocation.buffer)
        throw MetalBackendError("Metal buffer memory is not released");
    HeapArena::Placement placement;
    allocation.attach(impl_->newBufferPlaced(allocation.length,
                                             allocation.storage,
                                             allocation.label,
                                             &placement));
    allocation.heapArena = std::move(placement.arena);
    allocation.heap = placement.heap;
    allocation.heapOffset = placement.offset;
    allocation.heapSize = placement.size;
    impl_->sampleDeviceMemory();
}

MetalBuffer MetalBackend::view(const MetalBuffer &base,
                               uint64_t offsetBytes,
                               uint64_t lengthBytes) const {
    impl_->ensureHealthy();
    if (!base.impl_) {
        throw MetalBackendError("cannot view an empty Metal buffer");
    }
    if (base.impl_->allocation->accounting.get() != impl_->accounting.get()) {
        throw MetalBackendError("Metal buffer belongs to another backend");
    }
    if (!lengthBytes || offsetBytes > base.impl_->lengthBytes ||
        lengthBytes > base.impl_->lengthBytes - offsetBytes) {
        std::ostringstream message;
        message << "Metal buffer view is out of range: offset=" << offsetBytes
                << " length=" << lengthBytes
                << " base_length=" << base.impl_->lengthBytes;
        throw MetalBackendError(message.str());
    }
    auto result = std::make_shared<MetalBuffer::Impl>();
    result->allocation = base.impl_->allocation;
    result->offsetBytes = base.impl_->offsetBytes + offsetBytes;
    result->lengthBytes = lengthBytes;
    return MetalBuffer(std::move(result));
}

SharedEvent MetalBackend::newSharedEvent() {
    checkOperation();
    auto result = std::make_shared<SharedEvent::Impl>();
    result->event = [impl_->device newSharedEvent];
    if (!result->event) {
        throw MetalBackendError("unable to create Metal shared event");
    }
    result->listener = impl_->eventListener;
    return SharedEvent(std::move(result));
}

MetalMemoryStats MetalBackend::memoryStats() const noexcept {
    // Reading MTLDevice.currentAllocatedSize can synchronize with an active
    // command on some Apple GPUs. Every allocation and command lifecycle
    // boundary already samples it, so status must use the cached atomic value
    // rather than turning a control-plane query into a GPU barrier.
    return {
        impl_->accounting->allocatedBytes.load(std::memory_order_relaxed),
        impl_->accounting->peakAllocatedBytes.load(std::memory_order_relaxed),
        impl_->asyncState->deviceCurrentAllocatedBytes.load(
            std::memory_order_relaxed),
        impl_->asyncState->devicePeakAllocatedBytes.load(
            std::memory_order_relaxed),
    };
}

MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
    impl_->sampleDeviceMemory();
    return memoryStats();
}

// --- Blit commands ---------------------------------------------------
// The MTLBlitCommandEncoder groundwork: fill and copy operations under
// the same queue, gate, ticket and event-step contract as compute
// submissions. Kept out of MetalEncode.mm's dispatch path until the
// encoder there learns to interleave them; see BlitCommand in
// MetalBackend.hpp for the ordering contract.

void MetalBackend::Impl::blitRange(const MetalBuffer &view, uint64_t offset,
                                   uint64_t length, const char *role,
                                   std::shared_ptr<MetalAllocation> *allocation,
                                   uint64_t *absoluteOffset) {
        if (!view.impl_) {
            throw MetalBackendError(
                std::string("blit ") + role + " is an empty Metal buffer");
        }
        MetalBuffer::Impl &impl = *view.impl_;
        if (impl.allocation->accounting.get() != accounting.get()) {
            throw MetalBackendError(
                std::string("blit ") + role + " belongs to another backend");
        }
        if (!impl.allocation->buffer) {
            throw MetalBackendError(
                std::string("blit ") + role + " memory is released");
        }
        if (!length || offset > impl.lengthBytes ||
            length > impl.lengthBytes - offset) {
            std::ostringstream message;
            message << "blit " << role << " range is out of bounds: offset="
                    << offset << " length=" << length
                    << " view_length=" << impl.lengthBytes;
            throw MetalBackendError(message.str());
        }
        *allocation = impl.allocation;
        *absoluteOffset = impl.offsetBytes + offset;
    }

MetalBackend::Impl::PreparedBlitCommand
MetalBackend::Impl::prepareBlit(std::span<const BlitOperation> operations) {
        PreparedBlitCommand prepared;
        prepared.operations.reserve(operations.size());
        for (const BlitOperation &operation : operations) {
            PreparedBlit item;
            item.kind = operation.kind;
            item.length = operation.length;
            item.fillValue = operation.fillValue;
            blitRange(operation.destination, operation.destinationOffset,
                      operation.length, "destination", &item.destination,
                      &item.destinationOffset);
            if (operation.kind == BlitOperation::Kind::Copy) {
                blitRange(operation.source, operation.sourceOffset,
                          operation.length, "source", &item.source,
                          &item.sourceOffset);
            } else if (operation.kind != BlitOperation::Kind::Fill) {
                throw MetalBackendError("unknown blit operation kind");
            }
            prepared.retainedAllocations.push_back(item.destination);
            if (item.source)
                prepared.retainedAllocations.push_back(item.source);
            prepared.operations.push_back(std::move(item));
        }
        return prepared;
    }

CommandTicket MetalBackend::Impl::commitBlit(
        std::span<const PreparedBlit> operations,
        std::span<const EventStep> events,
        std::vector<std::shared_ptr<MetalAllocation>> retained,
        CommandCompletion completion) {
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        // The submit-ahead ring's bound, as commit(): a blit may be in
        // flight beside one compute command.
        ticketState->sequence =
            asyncState->beginSubmission(operations.size(), 2);
        ticketState->self = ticketState;
        ticketState->retainedAllocations = std::move(retained);
        ticketState->wallStart = AwakeClock::now();

        // The completed handler holds the ticket's state strongly, as
        // encodeSubmission's does: once a waiter gives up on the command,
        // it keeps the retained allocations until the GPU ends.
        const std::shared_ptr<CommandTicket::State> strongTicket =
            ticketState->self.lock();
        if (!strongTicket) {
            markUnhealthy(
                "Metal blit command lost its ticket before encoding");
            asyncState->releaseSubmission(ticketState->sequence);
            throw MetalBackendError(
                "Metal blit command lost its ticket before encoding");
        }

        // Metal may autorelease the command and its encoder, and the
        // serving loop's pool never drains, so their temporary ownership
        // ends with this submission, as encodeSubmission's does.
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!command) {
                markUnhealthy("unable to create Metal command buffer");
                asyncState->releaseSubmission(ticketState->sequence);
                throw MetalBackendError(
                    "unable to create Metal command buffer");
            }
            // The buffers before `command`, each ended by an event signal.
            std::vector<id<MTLCommandBuffer>> leading;
            id<MTLBlitCommandEncoder> encoder = nil;
            const auto failBeforeCommit = [&](std::string message) {
                markUnhealthy(message);
                asyncState->releaseSubmission(ticketState->sequence);
                throw MetalBackendError(std::move(message));
            };
            // Encodes the event steps that follow the first `encoded`
            // operations — encodeSubmission's rule: a wait holds the work
            // encoded next, a signal splits the buffer so later work never
            // waits behind it, and a failed buffer's signal is delivered
            // from the CPU through the completed handler.
            auto step = events.begin();
            const auto encodeSteps = [&](size_t encoded) {
                for (; step != events.end() && step->before == encoded;
                     ++step) {
                    if (encoder) {
                        [encoder endEncoding];
                        encoder = nil;
                    }
                    id<MTLSharedEvent> event =
                        (__bridge id<MTLSharedEvent>)
                            step->event.nativeHandle();
                    if (step->kind == EventStep::Kind::Wait) {
                        [command encodeWaitForEvent:event
                                              value:step->value];
                        continue;
                    }
                    [command encodeSignalEvent:event value:step->value];
                    const SharedEvent signaled = step->event;
                    const uint64_t value = step->value;
                    [command addCompletedHandler:
                        ^(id<MTLCommandBuffer> ended) {
                        if (ended.status ==
                            MTLCommandBufferStatusError)
                            signaled.signal(value);
                    }];
                    leading.push_back(command);
                    command = [queue commandBuffer];
                    if (!command) {
                        failBeforeCommit(
                            "unable to create Metal command buffer");
                    }
                }
            };
            for (size_t index = 0; index < operations.size(); ++index) {
                encodeSteps(index);
                if (!encoder) {
                    encoder = [command blitCommandEncoder];
                    if (!encoder) {
                        failBeforeCommit(
                            "unable to create Metal blit encoder");
                    }
                }
                const PreparedBlit &operation = operations[index];
                if (operation.kind == BlitOperation::Kind::Fill) {
                    [encoder fillBuffer:operation.destination->buffer
                                  range:NSMakeRange(
                                      operation.destinationOffset,
                                      operation.length)
                                  value:operation.fillValue];
                } else {
                    [encoder
                        copyFromBuffer:operation.source->buffer
                          sourceOffset:operation.sourceOffset
                             toBuffer:operation.destination->buffer
                    destinationOffset:operation.destinationOffset
                                  size:operation.length];
                }
            }
            encodeSteps(operations.size());
            if (encoder) [encoder endEncoding];
            ticketState->leadingCommands = leading;

            [command addCompletedHandler:^(id<MTLCommandBuffer> finished) {
                strongTicket->finishCommand(finished);
            }];
            residency->use();
            const auto recovered =
                [weakTicket = std::weak_ptr(strongTicket)](
                    id<MTLCommandBuffer> finished) {
                    if (auto ticket = weakTicket.lock())
                        ticket->finishCommand(finished);
                };
            asyncState->commitSubmission(ticketState->sequence, leading,
                                         command, recovered);
        }
        return CommandTicket(std::move(ticketState));
    }

CommandTiming MetalBackend::submitBlit(const BlitCommand &command) {
    return submitBlitAsync(command).wait();
}

CommandTicket MetalBackend::submitBlitAsync(const BlitCommand &command,
                                            CommandCompletion completion) {
    checkOperation();
    checkEvents(command.events, command.operations.size());
    Impl::PreparedBlitCommand prepared = impl_->prepareBlit(command.operations);
    return impl_->commitBlit(prepared.operations, command.events,
                             std::move(prepared.retainedAllocations),
                             std::move(completion));
}


} // namespace richengine::metal

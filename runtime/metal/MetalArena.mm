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
    allocation->attach(impl_->newBuffer(bytes, storage, allocation->label));
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
    allocation.attach(impl_->newBuffer(allocation.length, allocation.storage,
                                       allocation.label));
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


} // namespace richengine::metal

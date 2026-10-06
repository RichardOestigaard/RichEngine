// MetalBackend.mm: lifecycle, device probing, buffers and tickets. The
// Impl, allocation and submission bodies live in MetalBackendImpl.hpp,
// MetalArena.mm and MetalEncode.mm.

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

#include "metal/MetalBackendImpl.hpp"

namespace richengine::metal {



SharedEvent::SharedEvent() = default;
SharedEvent::~SharedEvent() = default;
SharedEvent::SharedEvent(const SharedEvent &) = default;
SharedEvent &SharedEvent::operator=(const SharedEvent &) = default;
SharedEvent::SharedEvent(SharedEvent &&) noexcept = default;
SharedEvent &SharedEvent::operator=(SharedEvent &&) noexcept = default;
SharedEvent::SharedEvent(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

SharedEvent::operator bool() const noexcept { return impl_ && impl_->event; }

void *SharedEvent::nativeHandle() const noexcept {
    return impl_ ? (__bridge void *)impl_->event : nullptr;
}

// Metal ignores a value below the event's, so the write alone raises it.
void SharedEvent::signal(uint64_t value) const noexcept {
    if (*this) impl_->event.signaledValue = value;
}

void SharedEvent::notify(uint64_t value, std::function<void()> callback) const {
    if (!*this) throw MetalBackendError("an empty shared event cannot notify");
    [impl_->event notifyListener:impl_->listener
                         atValue:value
                           block:^(id<MTLSharedEvent>, uint64_t) { callback(); }];
}

MetalBuffer::MetalBuffer() = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(const MetalBuffer &) = default;
MetalBuffer &MetalBuffer::operator=(const MetalBuffer &) = default;
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;

MetalBuffer::MetalBuffer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MetalBuffer::operator bool() const noexcept {
    return impl_ != nullptr;
}

uint64_t MetalBuffer::sizeBytes() const noexcept {
    return impl_ ? impl_->lengthBytes : 0;
}

uint64_t MetalBuffer::allocatedBytes() const noexcept {
    return impl_ ? impl_->allocation->bytes : 0;
}

bool MetalBuffer::sameView(const MetalBuffer &other) const noexcept {
    if (impl_ == other.impl_) return true;
    return impl_ && other.impl_ &&
           impl_->allocation == other.impl_->allocation &&
           impl_->offsetBytes == other.impl_->offsetBytes &&
           impl_->lengthBytes == other.impl_->lengthBytes;
}

BufferStorage MetalBuffer::storage() const noexcept {
    return impl_ ? impl_->allocation->storage : BufferStorage::Shared;
}

void *MetalBuffer::contents() const noexcept {
    if (!impl_ || impl_->allocation->storage != BufferStorage::Shared ||
        !impl_->allocation->buffer) {
        return nullptr;
    }
    return static_cast<uint8_t *>(impl_->allocation->buffer.contents) +
           impl_->offsetBytes;
}

uint64_t MetalBuffer::gpuAddress() const noexcept {
    if (!impl_ || !impl_->allocation->buffer) return 0;
    return impl_->allocation->buffer.gpuAddress + impl_->offsetBytes;
}

CommandTicket::CommandTicket() = default;

CommandTicket::CommandTicket(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

CommandTicket::~CommandTicket() {
    if (state_) state_->abandon();
}

CommandTicket::CommandTicket(CommandTicket &&) noexcept = default;

CommandTicket &CommandTicket::operator=(CommandTicket &&other) noexcept {
    if (this == &other) return *this;
    if (state_) state_->abandon();
    state_ = std::move(other.state_);
    return *this;
}

bool CommandTicket::ready() const noexcept {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->completed;
}

CommandTiming CommandTicket::wait() {
    if (!state_) throw MetalBackendError("Metal command ticket is empty");
    if (!state_->awaitCompletion(true)) {
        // Let go first, so that unwinding does not wait again.
        auto backend = state_->backend;
        state_.reset();
        backend->throwUnhealthy();
    }
    CommandTiming timing;
    std::string error;
    {
        std::lock_guard lock(state_->mutex);
        timing = state_->timing;
        error = state_->error;
    }
    state_->release();
    if (!error.empty()) throw MetalBackendError(error);
    return timing;
}

MetalBackend::MetalBackend(std::string metallibPath, double residencyKeepAliveSeconds)
    : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        if (metallibPath.empty()) {
            throw MetalBackendError("metallib path must not be empty");
        }
        if (!(residencyKeepAliveSeconds > 0.0)) {
            throw MetalBackendError("residency keep-alive must be positive");
        }
        // Check the OS floor before loading Metal resources so an unsupported
        // system reports the version requirement first.
        readMacosVersion(impl_->capabilities);
        if (!impl_->capabilities.meetsMinimumMacos()) {
            throw MetalBackendError(
                "RichEngine requires macOS " +
                std::to_string(DeviceCapabilities::kMinimumMacosMajor) + '.' +
                std::to_string(DeviceCapabilities::kMinimumMacosMinor) +
                " or newer; this Mac runs macOS " +
                impl_->capabilities.macosVersion());
        }
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw MetalBackendError("Metal device unavailable");
        }
        impl_->asyncState->device = impl_->device;
        // A command takes one Metal command buffer per event signal
        // (EventStep) and one more, all created before the first is
        // committed, and creating one waits while the queue's limit of them
        // is outstanding: at the default of 64, a 64-layer prefill with a
        // Neural Engine step each would wait forever.
        impl_->queue = [impl_->device
            newCommandQueueWithMaxCommandBufferCount:kMaximumCommandBuffers];
        if (!impl_->queue) {
            throw MetalBackendError("unable to create Metal command queue");
        }
        impl_->eventListener = [[MTLSharedEventListener alloc]
            initWithDispatchQueue:dispatch_queue_create(
                "richengine.metal.events",
                dispatch_queue_attr_make_with_autorelease_frequency(
                    DISPATCH_QUEUE_SERIAL,
                    DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM))];

        NSString *path = checkedNSString(metallibPath, "metallib path");
        NSError *error = nil;
        NSData *fileData = [NSData dataWithContentsOfFile:path
                                                 options:0
                                                   error:&error];
        if (!fileData) {
            throw MetalBackendError(
                "unable to read metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        // The library keeps the bytes read here, whatever later replaces the
        // path; the dispatch data retains them rather than copying them.
        dispatch_data_t data = dispatch_data_create(
            fileData.bytes, fileData.length, nullptr, ^{ (void)fileData; });
        error = nil;
        impl_->library =
            [impl_->device newLibraryWithData:data error:&error];
        if (!impl_->library) {
            throw MetalBackendError(
                "unable to load metallib " + metallibPath + ": " +
                errorDescription(error));
        }
        // Ending residency dispatches a kernel built here, so no pipeline or
        // driver program is compiled when a keep-alive lapses.
        impl_->residency = std::make_shared<Residency>(
            impl_->device, impl_->queue,
            impl_->newPipeline(Residency::kKickPipeline), residencyKeepAliveSeconds);

#if RICHENGINE_MTL4_AVAILABLE
        // Opt-in Metal 4 submission: queue, reused allocator and argument
        // table. Any failure leaves queue4 nil and submissions fall back to
        // the Metal 3 path.
        if (Impl::mtl4Enabled() &&
            [impl_->device respondsToSelector:
                               @selector(newMTL4CommandQueue)]) {
            impl_->queue4 = [impl_->device newMTL4CommandQueue];
            impl_->allocator4[0] = [impl_->device newCommandAllocator];
            impl_->allocator4[1] = [impl_->device newCommandAllocator];
            MTL4ArgumentTableDescriptor *tableDescriptor =
                [MTL4ArgumentTableDescriptor new];
            tableDescriptor.maxBufferBindCount = kBufferArgumentEntries;
            impl_->argumentTable4 = [impl_->device
                newArgumentTableWithDescriptor:tableDescriptor
                                         error:nil];
            if (impl_->queue4 && impl_->allocator4[0] &&
                impl_->allocator4[1] && impl_->argumentTable4) {
                impl_->residency->attach(impl_->queue4);
            } else {
                impl_->queue4 = nil;
                impl_->allocator4[0] = nil;
                impl_->allocator4[1] = nil;
                impl_->argumentTable4 = nil;
            }
        }
#endif

        readDeviceCapabilities(impl_->device, impl_->capabilities);
    }
    impl_->sampleDeviceMemory();
}

MetalBackend::~MetalBackend() = default;

void MetalBackend::stop() noexcept {
    std::lock_guard lock(impl_->asyncState->gateMutex);
    impl_->asyncState->stopping = true;
}

const DeviceCapabilities &MetalBackend::capabilities() const noexcept {
    return impl_->capabilities;
}

DeviceCapabilities probeDeviceCapabilities() {
    @autoreleasepool {
        DeviceCapabilities capabilities;
        readMacosVersion(capabilities);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw MetalBackendError("Metal device unavailable");
        readDeviceCapabilities(device, capabilities);
        return capabilities;
    }
}

void MetalBackend::checkOperation() const {
    impl_->ensureHealthy();
    if (impl_->operationGuard) impl_->operationGuard();
}

void MetalBackend::setOperationGuard(std::function<void()> guard) {
    impl_->operationGuard = std::move(guard);
}

void MetalBackend::setWaitInterrupt(std::function<bool()> shuttingDown) {
    impl_->asyncState->waitInterrupt = std::move(shuttingDown);
}

void MetalBackend::checkHealth() {
    impl_->asyncState->checkCommandHealth();
}

bool MetalBackend::healthy() const noexcept {
    return impl_->asyncState->healthy.load(std::memory_order_acquire);
}

std::string MetalBackend::unhealthyReason() const {
    std::lock_guard lock(impl_->asyncState->healthMutex);
    return impl_->asyncState->healthReason;
}

#ifdef RICHENGINE_BACKEND_INSTRUMENTATION
uint64_t BackendInstrumentation::submittedCommands(
    const MetalBackend &backend) {
    std::lock_guard lock(backend.impl_->asyncState->gateMutex);
    return backend.impl_->asyncState->nextSequence;
}

size_t BackendInstrumentation::cachedPipelines(const MetalBackend &backend) {
    return backend.impl_->pipelines.size();
}

void BackendInstrumentation::setDispatchProfiling(MetalBackend &backend,
                                                  bool enabled) {
    backend.impl_->dispatchProfiling = enabled;
}

std::vector<DispatchTiming>
BackendInstrumentation::takeDispatchProfile(MetalBackend &backend) {
    return std::exchange(backend.impl_->dispatchProfile, {});
}
#endif

}  // namespace richengine::metal

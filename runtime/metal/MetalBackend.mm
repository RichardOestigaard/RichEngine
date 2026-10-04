#import "MetalBackend.hpp"
#include "AwakeClock.hpp"
#include "CommandWatchdog.hpp"
#include "Env.hpp"
#include "Residency.hpp"
#include "TestConfig.hpp"
#ifdef SPLASH_BACKEND_INSTRUMENTATION
#include "BackendInstrumentation.hpp"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Metal 4 command encoding exists from macOS 26, below the engine's macOS
// 27 floor, so header availability is the only gate.
#if __has_include(<Metal/MTL4CommandQueue.h>)
#define SPLASH_MTL4_AVAILABLE 1
#else
#define SPLASH_MTL4_AVAILABLE 0
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

namespace splash::metal {
namespace {

// The accelerator entry that backs a Metal device publishes gpu-core-count.
// The device's registry ID names that entry or a child of it; the first
// IOAccelerator service is the fallback, since Apple silicon Macs have one
// GPU. Zero means the property was not found anywhere.
uint32_t gpuCoreCountForDevice(uint64_t registryId) noexcept {
    uint32_t count = 0;
    const auto read = [&](io_registry_entry_t entry) {
        if (!entry) return false;
        CFTypeRef value = IORegistryEntryCreateCFProperty(
            entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            int64_t number = 0;
            if (CFGetTypeID(value) == CFNumberGetTypeID() &&
                CFNumberGetValue(static_cast<CFNumberRef>(value),
                                 kCFNumberSInt64Type, &number) &&
                number > 0 && number <= 4096) {
                count = static_cast<uint32_t>(number);
            }
            CFRelease(value);
        }
        return count != 0;
    };
    io_registry_entry_t entry = IOServiceGetMatchingService(
        kIOMainPortDefault, IORegistryEntryIDMatching(registryId));
    for (int depth = 0; entry && depth < 4 && !read(entry); ++depth) {
        io_registry_entry_t parent = MACH_PORT_NULL;
        if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) !=
            KERN_SUCCESS) {
            parent = MACH_PORT_NULL;
        }
        IOObjectRelease(entry);
        entry = parent;
    }
    if (entry) IOObjectRelease(entry);
    if (!count) {
        io_registry_entry_t accelerator = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
        if (accelerator) {
            read(accelerator);
            IOObjectRelease(accelerator);
        }
    }
    return count;
}

std::string stringFromNSString(NSString *value) {
    if (!value) return {};
    const char *utf8 = value.UTF8String;
    return utf8 ? utf8 : "";
}

std::string errorDescription(NSError *error) {
    if (!error) return "unknown Metal error";
    std::string result = stringFromNSString(error.localizedDescription);
    return result.empty() ? "unknown Metal error" : result;
}

void readMacosVersion(DeviceCapabilities &capabilities) {
    const NSOperatingSystemVersion os =
        NSProcessInfo.processInfo.operatingSystemVersion;
    const auto component = [](NSInteger value) {
        return value > 0 ? static_cast<uint32_t>(value) : 0U;
    };
    capabilities.macosMajor = component(os.majorVersion);
    capabilities.macosMinor = component(os.minorVersion);
    capabilities.macosPatch = component(os.patchVersion);
}

// The backend and probeDeviceCapabilities() share one reading of the device,
// so the probe judges a Mac by the values the engine validates.
void readDeviceCapabilities(id<MTLDevice> device,
                            DeviceCapabilities &capabilities) {
    capabilities.deviceName = stringFromNSString(device.name);
    capabilities.gpuCoreCount = gpuCoreCountForDevice(device.registryID);
    // Apple GPU families nest, so the device's is the last one supported
    // counting up from Apple7.
    uint32_t family = 0;
    for (uint32_t next = 7;
         [device supportsFamily:static_cast<MTLGPUFamily>(1000 + next)]; ++next)
        family = next;
    capabilities.appleGpuFamily = family;
    capabilities.physicalMemoryBytes = NSProcessInfo.processInfo.physicalMemory;
    capabilities.recommendedMaxWorkingSetBytes =
        device.recommendedMaxWorkingSetSize;
    capabilities.maxBufferLengthBytes = device.maxBufferLength;
    capabilities.maxThreadgroupMemoryBytes = device.maxThreadgroupMemoryLength;
    MTLSize maximumThreads = device.maxThreadsPerThreadgroup;
    capabilities.maxThreadgroupWidth = maximumThreads.width;
    capabilities.hasUnifiedMemory = device.hasUnifiedMemory;
}

// Every uint64_t size, offset and length passes to Metal as it is.
static_assert(sizeof(NSUInteger) == sizeof(uint64_t),
              "Splash builds for arm64 only");

MTLSize metalSize(const DispatchSize &size, std::string_view field) {
    if (!size.x || !size.y || !size.z) {
        throw MetalBackendError(std::string(field) + " must be non-zero");
    }
    return MTLSizeMake(size.x, size.y, size.z);
}

bool multiplyOverflows(uint64_t left, uint64_t right) {
    return right && left > std::numeric_limits<uint64_t>::max() / right;
}

// Entries of a kernel's buffer argument table on every Apple GPU family.
constexpr uint32_t kBufferArgumentEntries = 31;
// How long a ticket waits for its command before it asks the watchdog.
constexpr auto kTicketWaitSlice = std::chrono::seconds(1);

double awakeSeconds() noexcept {
    return std::chrono::duration<double>(AwakeClock::now().time_since_epoch()).count();
}

const char *commandStatusName(MTLCommandBufferStatus status) noexcept {
    switch (status) {
    case MTLCommandBufferStatusNotEnqueued: return "not_enqueued";
    case MTLCommandBufferStatusEnqueued: return "enqueued";
    case MTLCommandBufferStatusCommitted: return "committed";
    case MTLCommandBufferStatusScheduled: return "scheduled";
    case MTLCommandBufferStatusCompleted: return "completed";
    case MTLCommandBufferStatusError: return "error";
    }
    return "unknown";
}

template <typename T>
void raisePeak(std::atomic<T> &peak, T value) noexcept {
    T current = peak.load(std::memory_order_relaxed);
    while (value > current &&
           !peak.compare_exchange_weak(current, value,
                                       std::memory_order_relaxed)) {}
}

// Hashes pipeline names as views, so a cache lookup builds no string.
struct PipelineNameHash {
    using is_transparent = void;
    size_t operator()(std::string_view name) const noexcept {
        return std::hash<std::string_view>{}(name);
    }
};

NSString *checkedNSString(std::string_view value, std::string_view field) {
    NSString *result = [[NSString alloc]
        initWithBytes:value.data()
        length:value.size()
        encoding:NSUTF8StringEncoding];
    if (!result) {
        throw MetalBackendError(std::string(field) + " is not UTF-8");
    }
    return result;
}

}  // namespace

struct AllocationAccounting {
    std::atomic<uint64_t> allocatedBytes{0};
    std::atomic<uint64_t> peakAllocatedBytes{0};
};

struct MetalAllocation {
    // Nil while the memory is released (MetalBackend::releaseMemory).
    __strong id<MTLBuffer> buffer = nil;
    std::shared_ptr<AllocationAccounting> accounting;
    // What the buffer adds to the accounting: its allocated size, or zero
    // while released.
    uint64_t bytes = 0;
    // The length, storage and label a restored buffer is allocated with.
    uint64_t length = 0;
    BufferStorage storage = BufferStorage::Shared;
    __strong NSString *label = nil;
    // The residency set the buffer belongs to, held weakly as allocations
    // may outlive the backend. The set retains the buffer, and with it its
    // memory, so the last view takes it out.
    std::weak_ptr<Residency> residency;
    // Submitting-thread dedupe stamps, one per prepared-command pass:
    // submitEpoch marks the retained set, matchEpoch marks an allocation
    // whose identity a command or span snapshot already proved this pass.
    // Compared, never synchronized.
    uint64_t submitEpoch = 0;
    uint64_t matchEpoch = 0;

    // Takes the buffer, which joins the residency set and the accounting.
    void attach(id<MTLBuffer> allocated) {
        buffer = allocated;
        bytes = allocated.allocatedSize;
        if (auto kept = residency.lock()) kept->add(buffer);
        raisePeak(accounting->peakAllocatedBytes,
                  accounting->allocatedBytes.fetch_add(
                      bytes, std::memory_order_relaxed) + bytes);
    }
    // Lets the buffer go: it leaves the residency set and the accounting.
    void detach() noexcept {
        if (!buffer) return;
        if (auto kept = residency.lock()) kept->remove(buffer);
        accounting->allocatedBytes.fetch_sub(bytes, std::memory_order_relaxed);
        buffer = nil;
        bytes = 0;
    }

    ~MetalAllocation() { detach(); }
};

// Every Impl has an allocation, whose buffer is nil only while its memory is
// released: only allocateBuffer and view create one.
struct MetalBuffer::Impl {
    std::shared_ptr<MetalAllocation> allocation;
    uint64_t offsetBytes = 0;
    uint64_t lengthBytes = 0;
};

struct BackendAsyncState {
    explicit BackendAsyncState(double commandTimeoutSeconds)
        : commandWatchdog(commandTimeoutSeconds) {}

    __strong id<MTLDevice> device = nil;
    mutable std::atomic<uint64_t> deviceCurrentAllocatedBytes{0};
    mutable std::atomic<uint64_t> devicePeakAllocatedBytes{0};
    std::atomic<bool> healthy{true};
    mutable std::mutex healthMutex;
    std::string healthReason;
    mutable std::mutex gateMutex;
    uint64_t nextSequence = 0;
    uint64_t activeSequence = 0;
    size_t activeDispatchCount = 0;
    __weak id<MTLCommandBuffer> activeCommand = nil;
    std::function<void(id<MTLCommandBuffer>)> activeCompletion;
    // The Metal 4 path has no command object whose status the watchdog can
    // read: its commit feedback is the only completion signal, so an
    // expired watch there is always a timeout.
    bool activeMtl4 = false;
    CommandWatchdog commandWatchdog;
    bool stopping = false;
    // MetalBackend::setWaitInterrupt's predicate.
    std::function<bool()> waitInterrupt;

    void sampleDeviceMemory() const noexcept {
        if (!device) return;
        uint64_t current = static_cast<uint64_t>(device.currentAllocatedSize);
        deviceCurrentAllocatedBytes.store(current, std::memory_order_relaxed);
        raisePeak(devicePeakAllocatedBytes, current);
    }

    [[noreturn]] void throwUnhealthy() const {
        std::lock_guard lock(healthMutex);
        throw MetalBackendError("Metal backend is unhealthy: " + healthReason);
    }

    void ensureHealthy() const {
        if (!healthy.load(std::memory_order_acquire)) throwUnhealthy();
    }

    void markUnhealthy(std::string reason) {
        {
            std::lock_guard lock(healthMutex);
            if (healthReason.empty()) healthReason = std::move(reason);
        }
        healthy.store(false, std::memory_order_release);
    }

    uint64_t beginSubmission(size_t dispatchCount) {
        ensureHealthy();
        std::lock_guard lock(gateMutex);
        if (stopping)
            throw MetalBackendError("Metal backend is stopping");
        if (activeSequence) {
            throw MetalBackendError(
                "Metal backend already has an in-flight command");
        }
        activeSequence = ++nextSequence;
        activeDispatchCount = dispatchCount;
        return activeSequence;
    }

    void commitSubmission(uint64_t sequence, id<MTLCommandBuffer> command,
                          std::function<void(id<MTLCommandBuffer>)> completion) {
        std::lock_guard lock(gateMutex);
        activeCommand = command;
        activeCompletion = std::move(completion);
        commandWatchdog.start(sequence, awakeSeconds());
        [command commit];
    }

    // The MTL4 equivalent of commitSubmission: the caller commits on its
    // queue right after this registers the watch.
    void commitSubmission4(uint64_t sequence) {
        std::lock_guard lock(gateMutex);
        activeMtl4 = true;
        commandWatchdog.start(sequence, steadySeconds());
    }

    void releaseSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
        if (activeSequence == sequence) {
            activeSequence = 0;
            activeCommand = nil;
            activeCompletion = {};
            activeMtl4 = false;
        }
    }

    void completeSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        commandWatchdog.complete(sequence);
    }

    // Runs the command watchdog. A terminal command whose callback is late
    // is completed here; one still running past its timeout marks the
    // backend unhealthy, and the answer is then true: the backend gave up on
    // that command.
    [[nodiscard]] bool commandAbandoned() noexcept {
        id<MTLCommandBuffer> command = nil;
        std::function<void(id<MTLCommandBuffer>)> complete;
        {
            std::lock_guard lock(gateMutex);
            if (!commandWatchdog.expired(awakeSeconds())) return false;
            command = activeCommand;
            const auto status = command ? command.status
                                        : MTLCommandBufferStatusNotEnqueued;
            // Recover terminal results even if the driver has not delivered
            // its callback. Finish outside the gate: it takes the ticket lock.
            // The Metal 4 path publishes completion only through its commit
            // feedback, which has already run finish() when it arrives, so a
            // still-watching Metal 4 command always counts as timed out.
            if (command && (status == MTLCommandBufferStatusCompleted ||
                            status == MTLCommandBufferStatusError)) {
                complete = activeCompletion;
            } else {
                // Waits that must not throw run this too: the reason drops
                // its details when they cannot be formatted.
                std::string reason = "Metal command completion timed out";
                try {
                    std::ostringstream message;
                    message << reason << " after "
                            << commandWatchdog.timeoutSeconds()
                            << " seconds (sequence=" << activeSequence
                            << ", status="
                            << (command ? commandStatusName(status)
                                        : (activeMtl4 ? "mtl4" : "unavailable"))
                            << ", dispatches=" << activeDispatchCount << ')';
                    reason = message.str();
                } catch (const std::bad_alloc &) {
                }
                markUnhealthy(std::move(reason));
                return true;
            }
        }
        if (complete) complete(command);
        return false;
    }

    void checkCommandHealth() {
        static_cast<void>(commandAbandoned());
        ensureHealthy();
    }

    // True when the process is shutting down; the waiter decides whether
    // that gives its command up.
    [[nodiscard]] bool waitInterrupted() const noexcept {
        return waitInterrupt && waitInterrupt();
    }

    [[nodiscard]] bool hasActiveSubmission() const noexcept {
        std::lock_guard lock(gateMutex);
        return activeSequence != 0;
    }
};

struct CommandTicket::State {
    std::shared_ptr<BackendAsyncState> backend;
    std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    CommandCompletion completion;
    mutable std::mutex mutex;
    std::condition_variable condition;
    uint64_t sequence = 0;
    CommandTiming timing;
    AwakeClock::time_point wallStart;
    std::string error;
    bool completed = false;
    bool released = false;

    void finishCommand(id<MTLCommandBuffer> command) {
        auto wallEnd = AwakeClock::now();
        CommandTiming timing;
        timing.gpuSeconds =
            command.GPUEndTime - command.GPUStartTime;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();

        std::string error;
        if (command.status != MTLCommandBufferStatusCompleted) {
            std::ostringstream message;
            message << "Metal command " << sequence << " failed";
            if (command.error) {
                message << ": " << errorDescription(command.error);
            }
            error = message.str();
        }

        finish(timing, std::move(error));
    }

#if SPLASH_MTL4_AVAILABLE
    // The Metal 4 commit feedback carries the same terminal state the Metal 3
    // completed handler reads off its command buffer.
    void finishCommand4(id<MTL4CommitFeedback> feedback) {
        auto wallEnd = std::chrono::steady_clock::now();
        CommandTiming timing;
        timing.gpuSeconds =
            feedback.GPUEndTime - feedback.GPUStartTime;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();

        std::string error;
        if (feedback.error) {
            std::ostringstream message;
            message << "Metal command " << sequence << " failed: "
                    << errorDescription(feedback.error);
            error = message.str();
        }

        finish(timing, std::move(error));
    }
#endif

    void finish(CommandTiming result, std::string failure) {
        CommandCompletion notify;
        {
            std::lock_guard lock(mutex);
            // Host recovery, late callbacks, and discarded commands all share
            // this completion path; only the first result may publish or notify.
            if (completed) return;
            backend->completeSubmission(sequence);
            if (!failure.empty()) backend->markUnhealthy(failure);
            timing = result;
            error = std::move(failure);
            completed = true;
            notify = completion;
        }
        if (notify) {
            try {
                notify();
            } catch (...) {
                backend->markUnhealthy(
                    "Metal completion callback threw an exception");
            }
        }
        condition.notify_all();
    }

    void release() noexcept {
        bool shouldRelease = false;
        {
            std::lock_guard lock(mutex);
            if (!released) {
                released = true;
                retainedAllocations.clear();
                shouldRelease = true;
            }
        }
        if (shouldRelease && backend) {
            // Refresh the cached device telemetry, which status reports, on
            // the consuming thread after GPU completion; admission samples
            // its own (refreshMemoryStats).
            if (backend->healthy.load(std::memory_order_acquire))
                backend->sampleDeviceMemory();
            backend->releaseSubmission(sequence);
        }
    }

    // Waits for the command in kTicketWaitSlice slices. Between them, outside
    // `mutex` (the watchdog may finish this ticket through finishCommand), it
    // asks the backend whether to stop, and with honorShutdown also whether
    // the process is shutting down. False when the backend gave up on a
    // command that never completed: the GPU may still use the retained
    // allocations, which the command's completion handler keeps alive with
    // this state.
    [[nodiscard]] bool awaitCompletion(bool honorShutdown) noexcept {
        std::unique_lock lock(mutex);
        while (!condition.wait_until(lock, AwakeClock::now() + kTicketWaitSlice,
                                     [this] { return completed; })) {
            lock.unlock();
            const bool abandoned = backend->commandAbandoned();
            const bool interrupted =
                !abandoned && honorShutdown && backend->waitInterrupted();
            lock.lock();
            if (completed) break;
            // A shutdown gives the command up only here, where the lock shows
            // it unfinished: one that completed meanwhile leaves the backend
            // healthy.
            if (interrupted) backend->markUnhealthy(shutdownReason());
            if (abandoned || interrupted) return false;
        }
        return true;
    }

    // Why the backend is unhealthy once a shutdown gave this command up.
    [[nodiscard]] std::string shutdownReason() const noexcept {
        std::string reason = "shutdown requested while waiting for a Metal command";
        try {
            reason = "shutdown requested while waiting for Metal command " +
                     std::to_string(sequence);
        } catch (const std::bad_alloc &) {
        }
        return reason;
    }

    // An abandoned command keeps its allocations until its completion
    // handler lets go of this state; the unhealthy backend admits no more.
    // A shutdown does not give up this wait: teardown waits for the command,
    // as long as the watchdog lets it.
    void abandon() noexcept {
        if (awaitCompletion(false)) release();
    }
};

struct MetalBackend::Impl {
    struct BakedSpan;

    // One dispatch of a command, validated and resolved to its pipeline.
    struct PreparedDispatch {
        const ComputeDispatch *source = nullptr;
        MTLSize groups{};
        MTLSize threads{};
        uint64_t threadCount = 0;
        // The argument table entries its buffers take, one bit each.
        uint32_t bufferIndices = 0;
        __strong id<MTLComputePipelineState> pipeline = nil;
        // When this dispatch opens a replayable span, the baked indirect
        // command buffer covering it; commit() skips the span's dispatches.
        const BakedSpan *span = nullptr;
    };

    // A baked indirect command buffer replaying one CommandGraph baked span,
    // plus the snapshot of the run it was baked from. Submission replays the
    // span only while the run still matches the snapshot field for field;
    // drift re-bakes. Indirect commands bind no bytes, so every parameter
    // payload of the run is staged in paramsAllocation, 256-byte aligned.
    struct BakedSpan {
        struct BufferSnapshot {
            // Weak so a cached span never keeps a dropped buffer's memory:
            // every replay binds only buffers already retained by the
            // in-flight ticket, validated identical to the snapshot.
            std::weak_ptr<MetalAllocation> allocation;
            // The MTLBuffer bound at bake time: memory released or restored
            // since then mismatches and re-bakes. Weak so the snapshot never
            // keeps released memory alive.
            __weak id<MTLBuffer> object = nil;
            uint64_t offsetBytes = 0;
            uint64_t lengthBytes = 0;
        };
        struct BytesSnapshot {
            uint32_t index = 0;
            // Compared only while the dispatch is not patchable.
            std::vector<std::byte> data;
            // The payload's staged copy inside paramsAllocation, which a
            // replay rewrites when the dispatch is patchable.
            uint64_t paramsOffset = 0;
            uint64_t sizeBytes = 0;
        };
        struct DispatchSnapshot {
            std::string pipelineName;
            DispatchSize threadgroups{};
            DispatchSize threadsPerThreadgroup{};
            // The run marks payloads that change between submissions:
            // matching skips their contents and replay rewrites the staged
            // bytes (ComputeDispatch::patchableBytes).
            bool patchableBytes = false;
            std::vector<uint32_t> bufferIndices;
            std::vector<BufferSnapshot> buffers;
            std::vector<BytesSnapshot> bytes;
        };
        __strong id<MTLIndirectCommandBuffer> icb = nil;
        std::shared_ptr<MetalAllocation> paramsAllocation;
        std::vector<DispatchSnapshot> dispatches;
        // The distinct buffers the span binds: the encoder declares their
        // usage at replay so its hazard tracking sees the span's accesses.
        std::vector<__weak id<MTLBuffer>> usedBuffers;
        uint64_t touched = 0;
    };

    // A command as submission encodes it: its dispatches and every
    // allocation they bind, which its ticket retains.
    struct PreparedCommand {
        std::vector<PreparedDispatch> dispatches;
        std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
        // Keeps the spans dispatches[].span points at alive: the baked
        // cache may evict one while a prepared command still names it.
        std::vector<std::shared_ptr<const BakedSpan>> spanRefs;
    };

    // A command's shape as last submitted, validated field-for-field like a
    // baked span: on a match the prepared command is reused — pipelines,
    // geometries and resolved spans — and only the new run's views and
    // payloads are read at encode, so validation, pipeline lookup, the
    // retained-set build and per-span resolution all skip. Patchable
    // payloads never take part in the match.
    struct CachedCommand {
        using BufferSnapshot = BakedSpan::BufferSnapshot;
        struct BytesSnapshot {
            uint32_t index = 0;
            uint64_t sizeBytes = 0;
            // Compared only for bakeable non-patchable payloads, whose
            // staged bytes a replayed span binds.
            std::vector<std::byte> data;
        };
        struct DispatchSnapshot {
            std::string pipelineName;
            DispatchSize threadgroups{};
            DispatchSize threadsPerThreadgroup{};
            bool bakeable = false;
            bool patchableBytes = false;
            std::vector<uint32_t> bufferIndices;
            std::vector<BufferSnapshot> buffers;
            std::vector<BytesSnapshot> bytes;
        };
        std::vector<DispatchSnapshot> dispatches;
        // The prepared form to reuse; its source pointers are rebound to
        // the new run on every hit.
        std::vector<PreparedDispatch> prepared;
        std::vector<std::shared_ptr<const BakedSpan>> spanRefs;
        uint64_t boundBuffers = 0;
        uint64_t touched = 0;
    };

    std::function<void()> operationGuard;

#ifdef SPLASH_BACKEND_INSTRUMENTATION
    bool dispatchProfiling = false;
    std::vector<DispatchTiming> dispatchProfile;
#endif
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
#if SPLASH_MTL4_AVAILABLE
    // The Metal 4 submission path, used only while mtl4Enabled() opted in.
    // One command in flight makes the allocator, argument table and staging
    // buffer safely reusable across submissions.
    __strong id<MTL4CommandQueue> queue4 = nil;
    __strong id<MTL4CommandAllocator> allocator4 = nil;
    __strong id<MTL4ArgumentTable> argumentTable4 = nil;
    // MTL4ArgumentTable binds GPU addresses and has no setBytes equivalent:
    // every BytesBinding payload is staged here, 256-byte aligned, and bound
    // as an address into this buffer. Grown to a submission's total.
    std::shared_ptr<MetalAllocation> staging4;
#endif
    // Allocations hold it weakly: they may outlive the backend.
    std::shared_ptr<Residency> residency;
    __strong id<MTLLibrary> library = nil;
    // Looked up for every dispatch when its command is prepared, which the
    // GPU may be waiting for; a hit allocates nothing. Used only by the
    // submitting thread.
    std::unordered_map<std::string, id<MTLComputePipelineState>,
                       PipelineNameHash, std::equal_to<>>
        pipelines;
    // Baked indirect-command spans of earlier submissions, keyed by the
    // run's position, length and first pipeline. Two alternates per key keep
    // interleaved graph shapes from re-baking each other every step.
    std::unordered_map<std::string,
                       std::vector<std::shared_ptr<BakedSpan>>>
        bakedSpans;
    // Prepared commands of earlier submissions, keyed by dispatch count and
    // the run's first and last pipeline; alternates per key as the spans.
    std::unordered_map<std::string,
                       std::vector<std::unique_ptr<CachedCommand>>>
        preparedCache;
    uint64_t bakedClock = 0;
    // Epoch counters stamping allocations once per pass: retained-set
    // dedupe and snapshot identity checks.
    uint64_t submitEpoch = 0;
    uint64_t matchEpoch = 0;
    // Pipelines baked spans bind: only an indirect-command-capable pipeline
    // (supportIndirectCommandBuffers) may be set on an indirect command.
    std::unordered_map<std::string, id<MTLComputePipelineState>,
                       PipelineNameHash, std::equal_to<>>
        icbPipelines;

    DeviceCapabilities capabilities;
    std::shared_ptr<AllocationAccounting> accounting =
        std::make_shared<AllocationAccounting>();
    std::shared_ptr<BackendAsyncState> asyncState =
        std::make_shared<BackendAsyncState>(
            testConfig().commandTimeoutSeconds.value_or(kCommandTimeoutSeconds));

    void sampleDeviceMemory() const noexcept {
        asyncState->sampleDeviceMemory();
    }

    void ensureHealthy() const {
        asyncState->ensureHealthy();
    }

    void markUnhealthy(std::string reason) {
        asyncState->markUnhealthy(std::move(reason));
    }

    id<MTLBuffer> newBuffer(uint64_t bytes, BufferStorage storage,
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
    MetalAllocation &baseAllocation(const MetalBuffer &buffer) const {
        if (!buffer.impl_ ||
            buffer.impl_->allocation->accounting.get() != accounting.get())
            throw MetalBackendError("Metal buffer is not a buffer of this backend");
        MetalAllocation &allocation = *buffer.impl_->allocation;
        if (buffer.impl_->offsetBytes ||
            buffer.impl_->lengthBytes != allocation.length)
            throw MetalBackendError("a view's memory is its base buffer's");
        return allocation;
    }

    id<MTLComputePipelineState> pipeline(std::string_view name) {
        if (name.empty()) {
            throw MetalBackendError("Metal pipeline name must not be empty");
        }
        if (const auto cached = pipelines.find(name); cached != pipelines.end())
            return cached->second;
        id<MTLComputePipelineState> result = newPipeline(name);
        pipelines.emplace(name, result);
        sampleDeviceMemory();
        return result;
    }

    id<MTLComputePipelineState> newPipeline(std::string_view name) {
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLFunction> function = [library newFunctionWithName:key];
        if (!function) {
            throw MetalBackendError(
                "missing Metal function: " + std::string(name));
        }
        NSError *error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) {
            throw MetalBackendError(
                "unable to create Metal pipeline " + std::string(name) +
                ": " + errorDescription(error));
        }
        return result;
    }

    // The pipeline a baked indirect command binds: the same function as
    // pipeline(), created with indirect command buffer support, which the
    // default pipeline object does not carry.
    id<MTLComputePipelineState> indirectPipeline(std::string_view name) {
        if (const auto cached = icbPipelines.find(name);
            cached != icbPipelines.end())
            return cached->second;
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLFunction> function = [library newFunctionWithName:key];
        if (!function) {
            throw MetalBackendError(
                "missing Metal function: " + std::string(name));
        }
        MTLComputePipelineDescriptor *descriptor =
            [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        descriptor.supportIndirectCommandBuffers = YES;
        NSError *error = nil;
        id<MTLComputePipelineState> result =
            [device newComputePipelineStateWithDescriptor:descriptor
                                                  options:MTLPipelineOptionNone
                                               reflection:nil
                                                    error:&error];
        if (!result) {
            throw MetalBackendError(
                "unable to create Metal pipeline " + std::string(name) +
                ": " + errorDescription(error));
        }
        icbPipelines.emplace(std::string(name), result);
#if SPLASH_MTL4_AVAILABLE
        // Metal 4 resolves an indirect command's pipeline through residency
        // sets, not the per-encoder usage declarations the Metal 3 replay
        // makes.
        if (mtl4Enabled()) residency->add(result);
#endif
        sampleDeviceMemory();
        return result;
    }

    // The escape hatch: baked indirect dispatch is off entirely.
    [[nodiscard]] static bool icbDisabled() noexcept {
        static const bool disabled = envFlag("SPLASH_ICB_OFF");
        return disabled;
    }

    // The Metal 4 encoder is opt-in (SPLASH_MTL4=1) while it is benchmarked
    // against the Metal 3 path it mirrors.
    [[nodiscard]] static bool mtl4Enabled() noexcept {
        static const bool enabled = envFlag("SPLASH_MTL4");
        return enabled;
    }

    [[nodiscard]] bool useMtl4() const noexcept {
#if SPLASH_MTL4_AVAILABLE
        return mtl4Enabled() && queue4 != nil;
#else
        return false;
#endif
    }

    // Proves a bound view's allocation identity once per allocation per
    // pass: a previous successful check's epoch stamp stands in for the
    // weak lock and object load below. The stamp must fail for released
    // memory so a later validation reports it — the nil check cannot tell
    // released from never-attached.
    [[nodiscard]] bool
    bindingIdentity(const std::weak_ptr<MetalAllocation> &saved,
                    __weak id<MTLBuffer> savedObject,
                    const MetalBuffer::Impl &view) noexcept {
        if (view.allocation->matchEpoch == matchEpoch) return true;
        if (view.allocation != saved.lock() || !view.allocation->buffer ||
            view.allocation->buffer != savedObject)
            return false;
        view.allocation->matchEpoch = matchEpoch;
        return true;
    }

    // Byte-for-byte identity of a run against a baked snapshot; the buffer
    // identity check catches released-and-restored memory the MetalBuffer
    // view metadata alone would miss.
    [[nodiscard]] bool
    snapshotMatches(const BakedSpan &span,
                    std::span<const ComputeDispatch> run) noexcept {
        if (span.dispatches.size() != run.size()) return false;
        for (size_t i = 0; i < run.size(); ++i) {
            const ComputeDispatch &dispatch = run[i];
            const BakedSpan::DispatchSnapshot &snapshot = span.dispatches[i];
            if (snapshot.pipelineName != dispatch.pipelineName ||
                snapshot.patchableBytes != dispatch.patchableBytes ||
                snapshot.threadgroups.x != dispatch.threadgroups.x ||
                snapshot.threadgroups.y != dispatch.threadgroups.y ||
                snapshot.threadgroups.z != dispatch.threadgroups.z ||
                snapshot.threadsPerThreadgroup.x !=
                    dispatch.threadsPerThreadgroup.x ||
                snapshot.threadsPerThreadgroup.y !=
                    dispatch.threadsPerThreadgroup.y ||
                snapshot.threadsPerThreadgroup.z !=
                    dispatch.threadsPerThreadgroup.z ||
                snapshot.buffers.size() != dispatch.buffers.size() ||
                snapshot.bytes.size() != dispatch.bytes.size()) {
                return false;
            }
            for (size_t binding = 0; binding < dispatch.buffers.size();
                 ++binding) {
                const BufferBinding &bound = dispatch.buffers[binding];
                const BakedSpan::BufferSnapshot &saved =
                    snapshot.buffers[binding];
                const MetalBuffer::Impl &view = *bound.buffer.impl_;
                if (snapshot.bufferIndices[binding] != bound.index ||
                    view.offsetBytes != saved.offsetBytes ||
                    view.lengthBytes != saved.lengthBytes ||
                    !bindingIdentity(saved.allocation, saved.object, view)) {
                    return false;
                }
            }
            for (size_t binding = 0; binding < dispatch.bytes.size();
                 ++binding) {
                const BytesBinding &bound = dispatch.bytes[binding];
                const BakedSpan::BytesSnapshot &saved =
                    snapshot.bytes[binding];
                if (saved.index != bound.index ||
                    saved.sizeBytes != bound.sizeBytes ||
                    (!snapshot.patchableBytes &&
                     std::memcmp(saved.data.data(), bound.data,
                                 bound.sizeBytes) != 0)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Bakes one validated run into an indirect command buffer. The commands
    // carry their pipeline and buffer objects directly (nothing is
    // inherited).
    std::shared_ptr<BakedSpan>
    bakeSpan(std::span<const ComputeDispatch> run) {
        auto span = std::make_shared<BakedSpan>();
        span->dispatches.reserve(run.size());
        uint64_t paramsBytes = 0;
        uint32_t maxBindCount = 0;
        for (const ComputeDispatch &dispatch : run) {
            for (const BufferBinding &binding : dispatch.buffers)
                maxBindCount = std::max(maxBindCount, binding.index + 1);
            for (const BytesBinding &binding : dispatch.bytes) {
                maxBindCount = std::max(maxBindCount, binding.index + 1);
                paramsBytes += (binding.sizeBytes + 255) & ~uint64_t{255};
            }
        }
        id<MTLBuffer> params = nil;
        if (paramsBytes) {
            auto allocation = std::make_shared<MetalAllocation>();
            allocation->accounting = accounting;
            allocation->length = paramsBytes;
            allocation->storage = BufferStorage::Shared;
            allocation->label = @"baked-span-params";
            allocation->residency = residency;
            allocation->attach(
                newBuffer(paramsBytes, BufferStorage::Shared,
                          allocation->label));
            span->paramsAllocation = std::move(allocation);
            params = span->paramsAllocation->buffer;
        }
        MTLIndirectCommandBufferDescriptor *descriptor =
            [MTLIndirectCommandBufferDescriptor new];
        descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        descriptor.inheritPipelineState = NO;
        descriptor.inheritBuffers = NO;
        descriptor.maxKernelBufferBindCount = maxBindCount;
        span->icb = [device
            newIndirectCommandBufferWithDescriptor:descriptor
                                   maxCommandCount:run.size()
                                           options:0];
        if (!span->icb)
            throw MetalBackendError(
                "unable to create a Metal indirect command buffer");
        span->icb.label = @"baked-span";
#if SPLASH_MTL4_AVAILABLE
        // executeCommandsInBuffer requires the buffer itself resident on the
        // Metal 4 path; the Metal 3 encoder declares its usage instead.
        if (mtl4Enabled()) residency->add(span->icb);
#endif
        uint64_t cursor = 0;
        for (size_t i = 0; i < run.size(); ++i) {
            const ComputeDispatch &dispatch = run[i];
            BakedSpan::DispatchSnapshot snapshot;
            snapshot.pipelineName = dispatch.pipelineName;
            snapshot.threadgroups = dispatch.threadgroups;
            snapshot.threadsPerThreadgroup = dispatch.threadsPerThreadgroup;
            snapshot.patchableBytes = dispatch.patchableBytes;
            id<MTLIndirectComputeCommand> command =
                [span->icb indirectComputeCommandAtIndex:i];
            [command setComputePipelineState:indirectPipeline(
                                                 dispatch.pipelineName)];
            for (const BufferBinding &binding : dispatch.buffers) {
                const MetalBuffer::Impl &view = *binding.buffer.impl_;
                [command setKernelBuffer:view.allocation->buffer
                                  offset:view.offsetBytes
                                 atIndex:binding.index];
                snapshot.bufferIndices.push_back(binding.index);
                snapshot.buffers.push_back(
                    {view.allocation, view.allocation->buffer,
                     view.offsetBytes, view.lengthBytes});
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                std::memcpy(
                    static_cast<uint8_t *>(params.contents) + cursor,
                    binding.data, binding.sizeBytes);
                [command setKernelBuffer:params
                                  offset:cursor
                                 atIndex:binding.index];
                BakedSpan::BytesSnapshot saved{binding.index, {}, cursor,
                                               binding.sizeBytes};
                if (!dispatch.patchableBytes) {
                    saved.data.resize(binding.sizeBytes);
                    std::memcpy(saved.data.data(), binding.data,
                                binding.sizeBytes);
                }
                snapshot.bytes.push_back(std::move(saved));
                cursor += (binding.sizeBytes + 255) & ~uint64_t{255};
            }
            // A barrier makes this command wait for all commands before it:
            // set on every command but the first, it preserves the serial
            // ordering the direct encoder's hazard tracking gives the run.
            if (i) [command setBarrier];
            [command concurrentDispatchThreadgroups:
                MTLSizeMake(dispatch.threadgroups.x, dispatch.threadgroups.y,
                            dispatch.threadgroups.z)
                threadsPerThreadgroup:
                    MTLSizeMake(dispatch.threadsPerThreadgroup.x,
                                dispatch.threadsPerThreadgroup.y,
                                dispatch.threadsPerThreadgroup.z)];
            span->dispatches.push_back(std::move(snapshot));
        }
        {
            std::vector<id<MTLBuffer>> bound;
            for (const BakedSpan::DispatchSnapshot &snapshot :
                 span->dispatches)
                for (const BakedSpan::BufferSnapshot &saved : snapshot.buffers)
                    bound.push_back(saved.object);
            std::ranges::sort(bound);
            const auto repeated = std::ranges::unique(bound);
            bound.erase(repeated.begin(), repeated.end());
            for (id<MTLBuffer> buffer : bound)
                span->usedBuffers.push_back(buffer);
        }
        sampleDeviceMemory();
        return span;
    }

    // Rewrites a span's staged parameters from the run's current patchable
    // payloads: the indirect commands bind the parameter arena by address,
    // so the same commands replay with the new parameters. One command in
    // flight means the GPU finished reading the arena of the previous
    // submission before this patch runs.
    static void patchSpanParams(const BakedSpan &span,
                                std::span<const PreparedDispatch> run) {
        if (!span.paramsAllocation || !span.paramsAllocation->buffer) return;
        auto *contents =
            static_cast<uint8_t *>(span.paramsAllocation->buffer.contents);
        for (size_t i = 0; i < run.size(); ++i) {
            const BakedSpan::DispatchSnapshot &snapshot = span.dispatches[i];
            if (!snapshot.patchableBytes) continue;
            const ComputeDispatch &dispatch = *run[i].source;
            for (size_t binding = 0; binding < snapshot.bytes.size();
                 ++binding)
                std::memcpy(contents + snapshot.bytes[binding].paramsOffset,
                            dispatch.bytes[binding].data,
                            snapshot.bytes[binding].sizeBytes);
        }
    }

    // Finds a baked span matching this run or bakes a new one. Returns
    // nullptr when the span cannot bake; the run then encodes directly.
    std::shared_ptr<const BakedSpan>
    resolveBakedSpan(size_t begin,
                     std::span<const ComputeDispatch> run) {
        if (bakedSpans.size() > 256) {
            // Pathological graph churn: bound the cache rather than grow it.
            bakedSpans.clear();
        }
        std::string key = std::to_string(begin) + '#' +
                          std::to_string(run.size()) + '#' +
                          run.front().pipelineName;
        std::vector<std::shared_ptr<BakedSpan>> &slots = bakedSpans[key];
        for (const auto &candidate : slots) {
            // Each candidate gets a fresh stamp epoch: a partial walk's
            // stamps must not vouch for a different candidate's objects.
            ++matchEpoch;
            if (!snapshotMatches(*candidate, run)) continue;
            candidate->touched = ++bakedClock;
            return candidate;
        }
        try {
            auto baked = bakeSpan(run);
            baked->touched = ++bakedClock;
            slots.push_back(baked);
            if (slots.size() > 2) {
                const auto stale = std::min_element(
                    slots.begin(), slots.end(),
                    [](const auto &left, const auto &right) {
                        return left->touched < right->touched;
                    });
                slots.erase(stale);
            }
            return baked;
        } catch (...) {
            // Whatever the direct encoder would report still reports: the
            // run just encodes directly.
            return nullptr;
        }
    }

    PreparedCommand prepare(std::span<const ComputeDispatch> dispatches) {
        if (dispatches.empty()) {
            throw MetalBackendError("Metal command must contain a dispatch");
        }
        PreparedCommand command;
        command.dispatches.reserve(dispatches.size());
        size_t bindings = 0;
        for (const ComputeDispatch &dispatch : dispatches) {
            PreparedDispatch item;
            item.source = &dispatch;
            item.groups = metalSize(dispatch.threadgroups, "threadgroups");
            item.threads = metalSize(
                dispatch.threadsPerThreadgroup, "threadsPerThreadgroup");
            if (multiplyOverflows(dispatch.threadsPerThreadgroup.x,
                                  dispatch.threadsPerThreadgroup.y) ||
                multiplyOverflows(dispatch.threadsPerThreadgroup.x *
                                      dispatch.threadsPerThreadgroup.y,
                                  dispatch.threadsPerThreadgroup.z)) {
                throw MetalBackendError("threadsPerThreadgroup size overflows");
            }
            item.threadCount = dispatch.threadsPerThreadgroup.x *
                dispatch.threadsPerThreadgroup.y *
                dispatch.threadsPerThreadgroup.z;

            // Each binding takes its own entry of the argument table.
            uint32_t indices = 0;
            const auto claim = [&](uint32_t index) {
                if (index >= kBufferArgumentEntries) {
                    throw MetalBackendError(
                        "compute binding index exceeds the argument table");
                }
                if (indices & (uint32_t{1} << index)) {
                    throw MetalBackendError("duplicate compute binding index");
                }
                indices |= uint32_t{1} << index;
            };
            for (const BufferBinding &binding : dispatch.buffers) {
                if (!binding.buffer.impl_) {
                    std::ostringstream message;
                    message << "compute dispatch '" << dispatch.pipelineName
                            << "' contains an empty buffer at index "
                            << binding.index;
                    throw MetalBackendError(message.str());
                }
                if (binding.buffer.impl_->allocation->accounting.get() !=
                    accounting.get()) {
                    throw MetalBackendError(
                        "compute dispatch buffer belongs to another backend");
                }
                if (!binding.buffer.impl_->allocation->buffer) {
                    std::ostringstream message;
                    message << "compute dispatch '" << dispatch.pipelineName
                            << "' binds released memory at index "
                            << binding.index;
                    throw MetalBackendError(message.str());
                }
                claim(binding.index);
                item.bufferIndices |= uint32_t{1} << binding.index;
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                if (!binding.data || !binding.sizeBytes) {
                    throw MetalBackendError("compute byte binding is empty");
                }
                claim(binding.index);
            }
            bindings += dispatch.buffers.size();
            command.dispatches.push_back(item);
        }

        for (PreparedDispatch &item : command.dispatches) {
            item.pipeline = pipeline(item.source->pipelineName);
            if (item.threadCount >
                item.pipeline.maxTotalThreadsPerThreadgroup) {
                throw MetalBackendError(
                    "threadsPerThreadgroup exceeds pipeline capability");
            }
        }

        // Each allocation the command binds, once.
        command.retainedAllocations.reserve(bindings);
        retainBound(dispatches, command.retainedAllocations);

        // Maximal runs of CommandGraph baked-span dispatches replay through a
        // cached indirect command buffer; a run that cannot bake encodes
        // directly like any other.
        if (!icbDisabled()) {
            for (size_t begin = 0; begin < dispatches.size();) {
                if (!dispatches[begin].bakeable) {
                    ++begin;
                    continue;
                }
                size_t end = begin + 1;
                while (end < dispatches.size() && dispatches[end].bakeable)
                    ++end;
                const std::span<const ComputeDispatch> run =
                    dispatches.subspan(begin, end - begin);
                if (std::shared_ptr<const BakedSpan> span =
                        resolveBakedSpan(begin, run)) {
                    command.dispatches[begin].span = span.get();
                    if (span->paramsAllocation)
                        command.retainedAllocations.push_back(
                            span->paramsAllocation);
                    command.spanRefs.push_back(std::move(span));
                }
                begin = end;
            }
        }
        return command;
    }

    // Stamps every bound allocation into the submission's retained set,
    // once. Cheaper than sort-and-unique over the run's bindings.
    void retainBound(std::span<const ComputeDispatch> dispatches,
                     std::vector<std::shared_ptr<MetalAllocation>> &retained) {
        ++submitEpoch;
        for (const ComputeDispatch &dispatch : dispatches)
            for (const BufferBinding &binding : dispatch.buffers) {
                const std::shared_ptr<MetalAllocation> &owner =
                    binding.buffer.impl_->allocation;
                if (owner->submitEpoch == submitEpoch) continue;
                owner->submitEpoch = submitEpoch;
                retained.push_back(owner);
            }
    }

    // Whole-command identity of a run against a cached prepared command;
    // covers every field prepare() resolves — so a match implies every
    // baked span it references still matches its run too.
    [[nodiscard]] bool
    commandMatches(const CachedCommand &cached,
                   std::span<const ComputeDispatch> run) noexcept {
        if (cached.dispatches.size() != run.size()) return false;
        for (size_t i = 0; i < run.size(); ++i) {
            const ComputeDispatch &dispatch = run[i];
            const CachedCommand::DispatchSnapshot &snapshot =
                cached.dispatches[i];
            if (snapshot.pipelineName != dispatch.pipelineName ||
                snapshot.bakeable != dispatch.bakeable ||
                snapshot.patchableBytes != dispatch.patchableBytes ||
                snapshot.threadgroups.x != dispatch.threadgroups.x ||
                snapshot.threadgroups.y != dispatch.threadgroups.y ||
                snapshot.threadgroups.z != dispatch.threadgroups.z ||
                snapshot.threadsPerThreadgroup.x !=
                    dispatch.threadsPerThreadgroup.x ||
                snapshot.threadsPerThreadgroup.y !=
                    dispatch.threadsPerThreadgroup.y ||
                snapshot.threadsPerThreadgroup.z !=
                    dispatch.threadsPerThreadgroup.z ||
                snapshot.buffers.size() != dispatch.buffers.size() ||
                snapshot.bytes.size() != dispatch.bytes.size()) {
                return false;
            }
            for (size_t binding = 0; binding < dispatch.buffers.size();
                 ++binding) {
                const BufferBinding &bound = dispatch.buffers[binding];
                const CachedCommand::BufferSnapshot &saved =
                    snapshot.buffers[binding];
                if (!bound.buffer.impl_ ||
                    snapshot.bufferIndices[binding] != bound.index)
                    return false;
                const MetalBuffer::Impl &view = *bound.buffer.impl_;
                if (view.offsetBytes != saved.offsetBytes ||
                    view.lengthBytes != saved.lengthBytes ||
                    !bindingIdentity(saved.allocation, saved.object, view)) {
                    return false;
                }
            }
            for (size_t binding = 0; binding < dispatch.bytes.size();
                 ++binding) {
                const BytesBinding &bound = dispatch.bytes[binding];
                const CachedCommand::BytesSnapshot &saved =
                    snapshot.bytes[binding];
                if (saved.index != bound.index ||
                    saved.sizeBytes != bound.sizeBytes ||
                    (dispatch.bakeable && !dispatch.patchableBytes &&
                     std::memcmp(saved.data.data(), bound.data,
                                 bound.sizeBytes) != 0)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Snapshots a run the way commandMatches validates it.
    std::unique_ptr<CachedCommand>
    snapshotCommand(std::span<const ComputeDispatch> run,
                    PreparedCommand &command) {
        auto cached = std::make_unique<CachedCommand>();
        cached->dispatches.reserve(run.size());
        for (const ComputeDispatch &dispatch : run) {
            CachedCommand::DispatchSnapshot snapshot;
            snapshot.pipelineName = dispatch.pipelineName;
            snapshot.threadgroups = dispatch.threadgroups;
            snapshot.threadsPerThreadgroup = dispatch.threadsPerThreadgroup;
            snapshot.bakeable = dispatch.bakeable;
            snapshot.patchableBytes = dispatch.patchableBytes;
            for (const BufferBinding &binding : dispatch.buffers) {
                const MetalBuffer::Impl &view = *binding.buffer.impl_;
                snapshot.bufferIndices.push_back(binding.index);
                snapshot.buffers.push_back(
                    {view.allocation, view.allocation->buffer,
                     view.offsetBytes, view.lengthBytes});
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                CachedCommand::BytesSnapshot saved{binding.index,
                                                   binding.sizeBytes, {}};
                if (dispatch.bakeable && !dispatch.patchableBytes) {
                    saved.data.resize(binding.sizeBytes);
                    std::memcpy(saved.data.data(), binding.data,
                                binding.sizeBytes);
                }
                snapshot.bytes.push_back(std::move(saved));
            }
            cached->boundBuffers += dispatch.buffers.size();
            cached->dispatches.push_back(std::move(snapshot));
        }
        cached->prepared = command.dispatches;
        for (PreparedDispatch &item : cached->prepared) item.source = nullptr;
        cached->spanRefs = command.spanRefs;
        cached->touched = ++bakedClock;
        return cached;
    }

    // The escape hatch for the prepared-command cache only.
    [[nodiscard]] static bool preparedCacheDisabled() noexcept {
        static const bool disabled = envFlag("SPLASH_PREPARED_CACHE_OFF");
        return disabled;
    }

    // prepare(), or its cached equivalent when the run is field-for-field
    // identical to a recent submission: reuse skips validation, pipeline
    // lookups, the retained-set build and every baked-span resolution.
    PreparedCommand prepared(std::span<const ComputeDispatch> dispatches) {
        if (dispatches.empty() || preparedCacheDisabled()) {
            return prepare(dispatches);
        }
        if (preparedCache.size() > 64) {
            // Pathological command churn: bound the cache rather than grow
            // it.
            preparedCache.clear();
        }
        const std::string key = std::to_string(dispatches.size()) + '#' +
                                dispatches.front().pipelineName + '#' +
                                dispatches.back().pipelineName;
        std::vector<std::unique_ptr<CachedCommand>> &slots =
            preparedCache[key];
        for (const auto &candidate : slots) {
            // Each candidate gets a fresh stamp epoch, as the span slots
            // do: a partial walk's stamps must not vouch for a different
            // candidate's objects.
            ++matchEpoch;
            if (!commandMatches(*candidate, dispatches)) continue;
            candidate->touched = ++bakedClock;
            PreparedCommand command;
            command.dispatches = candidate->prepared;
            command.spanRefs = candidate->spanRefs;
            for (size_t i = 0; i < dispatches.size(); ++i)
                command.dispatches[i].source = &dispatches[i];
            command.retainedAllocations.reserve(candidate->boundBuffers +
                                                candidate->spanRefs.size());
            retainBound(dispatches, command.retainedAllocations);
            for (const auto &span : command.spanRefs)
                if (span->paramsAllocation)
                    command.retainedAllocations.push_back(
                        span->paramsAllocation);
            return command;
        }
        PreparedCommand command = prepare(dispatches);
        auto cached = snapshotCommand(dispatches, command);
        slots.push_back(std::move(cached));
        if (slots.size() > 2) {
            const auto stale = std::min_element(
                slots.begin(), slots.end(),
                [](const auto &left, const auto &right) {
                    return left->touched < right->touched;
                });
            slots.erase(stale);
        }
        return command;
    }

    // Encodes and commits prepared dispatches; the ticket retains
    // `retained` until it is consumed.
    CommandTicket commit(std::span<const PreparedDispatch> dispatches,
                         std::vector<std::shared_ptr<MetalAllocation>> retained,
                         CommandCompletion completion) {
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        ticketState->retainedAllocations = std::move(retained);
        ticketState->sequence =
            asyncState->beginSubmission(dispatches.size());

        auto failBeforeCommit = [&](std::string message) {
            markUnhealthy(message);
            asyncState->releaseSubmission(ticketState->sequence);
            throw MetalBackendError(std::move(message));
        };

        auto wallStart = AwakeClock::now();
        // Metal may autorelease the command and its encoder, and the serving
        // loop's pool never drains, so their temporary ownership ends with
        // this submission (under the validation layer an autoreleased
        // command holds every member of the residency set). The command
        // retains everything the GPU still needs.
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            if (!command) {
                failBeforeCommit("unable to create Metal command buffer");
            }
            ticketState->wallStart = wallStart;
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) {
                failBeforeCommit("unable to create Metal compute encoder");
            }
            // Indexed by argument table entry. The ticket and the dispatches
            // keep the buffers alive.
            __unsafe_unretained id<MTLBuffer> buffers[kBufferArgumentEntries];
            NSUInteger offsets[kBufferArgumentEntries];
            for (size_t index = 0; index < dispatches.size(); ++index) {
                const PreparedDispatch &item = dispatches[index];
                // A baked span replays its whole run from the indirect
                // command buffer; the run's own dispatches are skipped.
                if (item.span &&
                    index + item.span->dispatches.size() <=
                        dispatches.size()) {
                    patchSpanParams(
                        *item.span, dispatches.subspan(
                                        index, item.span->dispatches.size()));
                    [encoder useResource:item.span->icb
                                   usage:MTLResourceUsageRead];
                    for (__weak id<MTLBuffer> buffer :
                         item.span->usedBuffers) {
                        if (id<MTLBuffer> resource = buffer)
                            [encoder useResource:resource
                                           usage:MTLResourceUsageRead |
                                                 MTLResourceUsageWrite];
                    }
                    [encoder executeCommandsInBuffer:item.span->icb
                                           withRange:NSMakeRange(
                                                         0, item.span->dispatches.size())];
                    index += item.span->dispatches.size() - 1;
                    continue;
                }
                const ComputeDispatch &dispatch = *item.source;
                [encoder setComputePipelineState:item.pipeline];
                for (const BufferBinding &binding : dispatch.buffers) {
                    const MetalBuffer::Impl &buffer = *binding.buffer.impl_;
                    buffers[binding.index] = buffer.allocation->buffer;
                    offsets[binding.index] = buffer.offsetBytes;
                }
                // One call per run of consecutive entries: a command graph's
                // dispatch binds a single run.
                for (uint32_t unbound = item.bufferIndices; unbound;) {
                    const uint32_t first = std::countr_zero(unbound);
                    const uint32_t count = std::countr_one(unbound >> first);
                    [encoder setBuffers:buffers + first
                                offsets:offsets + first
                              withRange:NSMakeRange(first, count)];
                    unbound &= ~(((uint32_t{1} << count) - 1) << first);
                }
                for (const BytesBinding &binding : dispatch.bytes) {
                    [encoder setBytes:binding.data
                               length:binding.sizeBytes
                              atIndex:binding.index];
                }
                [encoder dispatchThreadgroups:item.groups
                         threadsPerThreadgroup:item.threads];
            }
            [encoder endEncoding];

            // Driver callbacks only complete the ticket. Device-wide memory
            // telemetry is sampled on the host when consuming the result. The
            // handler holds the ticket's state strongly: once a waiter gives
            // up on the command, it keeps the retained allocations until the
            // GPU ends.
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                ticketState->finishCommand(completed);
            }];
            residency->use();
            asyncState->commitSubmission(ticketState->sequence, command,
                [weakTicket = std::weak_ptr(ticketState)](
                    id<MTLCommandBuffer> completed) {
                    if (auto ticket = weakTicket.lock())
                        ticket->finishCommand(completed);
                });
        }
        return CommandTicket(std::move(ticketState));
    }

#if SPLASH_MTL4_AVAILABLE
    // The Metal 4 encoding of commit(): same prepared dispatches, same
    // ticket contract, encoded through one reused allocator and argument
    // table and committed with a commit-feedback handler in place of the
    // completed handler.
    CommandTicket commit4(std::span<const PreparedDispatch> dispatches,
                          std::vector<std::shared_ptr<MetalAllocation>> retained,
                          CommandCompletion completion) {
        // Stage every BytesBinding payload: an argument table binds
        // addresses only. Each payload keeps the 256-byte alignment a
        // baked span's parameters use. Span-covered runs stage their
        // payloads in the span's own parameter arena instead.
        uint64_t stagingBytes = 0;
        for (size_t index = 0; index < dispatches.size(); ++index) {
            const PreparedDispatch &item = dispatches[index];
            if (item.span &&
                index + item.span->dispatches.size() <= dispatches.size()) {
                index += item.span->dispatches.size() - 1;
                continue;
            }
            for (const BytesBinding &binding : item.source->bytes)
                stagingBytes += (binding.sizeBytes + 255) & ~uint64_t{255};
        }
        if (stagingBytes && (!staging4 || staging4->length < stagingBytes)) {
            // The one-command-in-flight invariant means the GPU has
            // finished with the staging buffer this replaces.
            auto allocation = std::make_shared<MetalAllocation>();
            allocation->accounting = accounting;
            allocation->length = stagingBytes;
            allocation->storage = BufferStorage::Shared;
            allocation->label = @"mtl4-params";
            allocation->residency = residency;
            allocation->attach(
                newBuffer(stagingBytes, BufferStorage::Shared,
                          allocation->label));
            staging4 = std::move(allocation);
        }
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        // The staging buffer backs the command's byte bindings for as long
        // as the GPU may read it.
        if (staging4 && stagingBytes)
            retained.push_back(staging4);
        ticketState->retainedAllocations = std::move(retained);
        ticketState->sequence =
            asyncState->beginSubmission(dispatches.size());

        auto failBeforeCommit = [&](std::string message) {
            markUnhealthy(message);
            asyncState->releaseSubmission(ticketState->sequence);
            throw MetalBackendError(std::move(message));
        };

        auto wallStart = std::chrono::steady_clock::now();
        // The allocator may be reused once the previous command buffer ended
        // (and, per the one-in-flight invariant, finished on the GPU), and
        // the serving loop's pool never drains, so temporary ownership ends
        // with this submission like the Metal 3 path's.
        @autoreleasepool {
            [allocator4 reset];
            id<MTL4CommandBuffer> command = [device newCommandBuffer];
            if (!command) {
                failBeforeCommit("unable to create Metal command buffer");
            }
            [command beginCommandBufferWithAllocator:allocator4];
            ticketState->wallStart = wallStart;
            id<MTL4ComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (!encoder) {
                failBeforeCommit("unable to create Metal compute encoder");
            }
            uint8_t *stagingContents =
                staging4 ? static_cast<uint8_t *>(staging4->buffer.contents)
                         : nullptr;
            const MTLGPUAddress stagingAddress =
                staging4 ? staging4->buffer.gpuAddress : 0;
            uint64_t stagingCursor = 0;
            // Dependency-scoped ordering. Metal 4 issues no implicit hazard
            // tracking between the dispatches of an encoder, so a
            // dispatch-stage barrier must separate any unit that may touch
            // bytes a still-unordered earlier unit touched. Whether a
            // binding reads or writes is not recorded on ComputeDispatch, so
            // any byte-range overlap — read/read included — still orders
            // (pipeline reflection cannot reliably attribute writes either:
            // the same buffer argument may be read or written depending on
            // runtime parameters the encoder never sees). Each barrier ends
            // the frontier: a unit whose bound views are disjoint from every
            // range since the last barrier encodes without one, letting
            // independent branches of a command graph overlap. A baked
            // span's replay serializes internally, so it joins the frontier
            // as one unit at the whole-buffer granularity of its
            // usedBuffers snapshot. Byte bindings stage in staging4, which
            // the GPU only reads, so they take no part in the frontier.
            struct BoundRange {
                // The ticket retains every bound allocation, so the buffer
                // objects outlive this analysis.
                __unsafe_unretained id<MTLBuffer> buffer;
                uint64_t begin;
                uint64_t end;
            };
            std::vector<uint8_t> needsBarrier(dispatches.size(), 0);
            {
                std::vector<BoundRange> frontier;
                std::vector<BoundRange> ranges;
                for (size_t index = 0; index < dispatches.size(); ++index) {
                    const PreparedDispatch &item = dispatches[index];
                    const bool replaysSpan =
                        item.span &&
                        index + item.span->dispatches.size() <=
                            dispatches.size();
                    ranges.clear();
                    if (replaysSpan) {
                        for (__weak id<MTLBuffer> weak :
                             item.span->usedBuffers) {
                            if (id<MTLBuffer> buffer = weak) {
                                ranges.push_back(
                                    {buffer, 0, buffer.length});
                            }
                        }
                    } else {
                        for (const BufferBinding &binding :
                             item.source->buffers) {
                            const MetalBuffer::Impl &view =
                                *binding.buffer.impl_;
                            ranges.push_back({view.allocation->buffer,
                                              view.offsetBytes,
                                              view.offsetBytes +
                                                  view.lengthBytes});
                        }
                    }
                    bool dependent = false;
                    for (const BoundRange &earlier : frontier) {
                        for (const BoundRange &range : ranges) {
                            if (earlier.buffer == range.buffer &&
                                earlier.begin < range.end &&
                                range.begin < earlier.end) {
                                dependent = true;
                                break;
                            }
                        }
                        if (dependent) break;
                    }
                    if (dependent) {
                        needsBarrier[index] = 1;
                        frontier.clear();
                    }
                    frontier.insert(frontier.end(), ranges.begin(),
                                    ranges.end());
                    if (replaysSpan)
                        index += item.span->dispatches.size() - 1;
                }
            }
            const auto orderAfterPrevious = [&](size_t index) {
                if (!needsBarrier[index]) return;
                [encoder barrierAfterEncoderStages:MTLStageDispatch
                              beforeEncoderStages:MTLStageDispatch
                                 visibilityOptions:MTL4VisibilityOptionDevice];
            };
            for (size_t index = 0; index < dispatches.size(); ++index) {
                const PreparedDispatch &item = dispatches[index];
                // A baked span replays its whole run from the indirect
                // command buffer; the run's own dispatches are skipped.
                if (item.span &&
                    index + item.span->dispatches.size() <=
                        dispatches.size()) {
                    patchSpanParams(
                        *item.span, dispatches.subspan(
                                        index, item.span->dispatches.size()));
                    orderAfterPrevious(index);
                    [encoder executeCommandsInBuffer:item.span->icb
                                           withRange:NSMakeRange(
                                                         0, item.span->dispatches.size())];
                    index += item.span->dispatches.size() - 1;
                    continue;
                }
                const ComputeDispatch &dispatch = *item.source;
                orderAfterPrevious(index);
                [encoder setComputePipelineState:item.pipeline];
                for (const BufferBinding &binding : dispatch.buffers) {
                    const MetalBuffer::Impl &buffer = *binding.buffer.impl_;
                    [argumentTable4 setAddress:
                        buffer.allocation->buffer.gpuAddress +
                            buffer.offsetBytes
                                       atIndex:binding.index];
                }
                for (const BytesBinding &binding : dispatch.bytes) {
                    std::memcpy(stagingContents + stagingCursor,
                                binding.data, binding.sizeBytes);
                    [argumentTable4 setAddress:stagingAddress + stagingCursor
                                       atIndex:binding.index];
                    stagingCursor +=
                        (binding.sizeBytes + 255) & ~uint64_t{255};
                }
                // The table's contents are snapshotted at each dispatch, so
                // one table serves the whole encoder.
                [encoder setArgumentTable:argumentTable4];
                [encoder dispatchThreadgroups:item.groups
                         threadsPerThreadgroup:item.threads];
            }
            [encoder endEncoding];
            [command endCommandBuffer];

            // The commit feedback delivers the GPU timing and error the
            // Metal 3 path reads off its finished command buffer; there is
            // no separate completion handler to register.
            MTL4CommitOptions *options = [MTL4CommitOptions new];
            [options addFeedbackHandler:
                ^(id<MTL4CommitFeedback> feedback) {
                    ticketState->finishCommand4(feedback);
                }];
            residency->use();
            asyncState->commitSubmission4(ticketState->sequence);
            id<MTL4CommandBuffer> committed = command;
            [queue4 commit:&committed count:1 options:options];
        }
        return CommandTicket(std::move(ticketState));
    }
#endif

#ifdef SPLASH_BACKEND_INSTRUMENTATION
    // Commits every dispatch of the command as its own command and waits
    // for it, then hands back an already-completed ticket with the summed
    // timing, so callers observe the usual asynchronous contract.
    CommandTicket submitProfiled(std::span<const ComputeDispatch> dispatches,
                                 CommandCompletion completion) {
        const PreparedCommand command = prepare(dispatches);
        CommandTiming total;
        for (const PreparedDispatch &item : command.dispatches) {
            const CommandTiming timing =
                commit({&item, 1}, command.retainedAllocations, {}).wait();
            dispatchProfile.push_back(
                {item.source->pipelineName, timing.gpuSeconds});
            total.gpuSeconds += timing.gpuSeconds;
            total.wallSeconds += timing.wallSeconds;
        }
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->sequence =
            asyncState->beginSubmission(command.dispatches.size());
        ticketState->timing = total;
        ticketState->completed = true;
        if (completion) completion();
        return CommandTicket(std::move(ticketState));
    }
#endif
};

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

MetalBackend::MetalBackend(std::string metallibPath)
    : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        if (metallibPath.empty()) {
            throw MetalBackendError("metallib path must not be empty");
        }
        // Check the OS floor before loading Metal resources so an unsupported
        // system reports the version requirement first.
        readMacosVersion(impl_->capabilities);
        if (!impl_->capabilities.meetsMinimumMacos()) {
            throw MetalBackendError(
                "Splash requires macOS " +
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
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) {
            throw MetalBackendError("unable to create Metal command queue");
        }

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
            impl_->newPipeline(Residency::kKickPipeline),
            testConfig().residencyKeepAliveSeconds.value_or(
                kResidencyKeepAliveSeconds));

#if SPLASH_MTL4_AVAILABLE
        // Opt-in Metal 4 submission: queue, reused allocator and argument
        // table. Any failure leaves queue4 nil and submissions fall back to
        // the Metal 3 path.
        if (Impl::mtl4Enabled() &&
            [impl_->device respondsToSelector:
                               @selector(newMTL4CommandQueue)]) {
            impl_->queue4 = [impl_->device newMTL4CommandQueue];
            impl_->allocator4 = [impl_->device newCommandAllocator];
            MTL4ArgumentTableDescriptor *tableDescriptor =
                [MTL4ArgumentTableDescriptor new];
            tableDescriptor.maxBufferBindCount = kBufferArgumentEntries;
            impl_->argumentTable4 = [impl_->device
                newArgumentTableWithDescriptor:tableDescriptor
                                         error:nil];
            if (impl_->queue4 && impl_->allocator4 && impl_->argumentTable4) {
                impl_->residency->attach(impl_->queue4);
            } else {
                impl_->queue4 = nil;
                impl_->allocator4 = nil;
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

void MetalBackend::releaseMemory(const MetalBuffer &buffer) {
    MetalAllocation &allocation = impl_->baseAllocation(buffer);
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

CommandTiming MetalBackend::submit(const ComputeDispatch &dispatch) {
    return submitAsync(dispatch).wait();
}

CommandTiming MetalBackend::submitCommand(
    std::span<const ComputeDispatch> dispatches) {
    return submitCommandAsync(dispatches).wait();
}

CommandTicket MetalBackend::submitAsync(const ComputeDispatch &dispatch) {
    return submitCommandAsync(std::span<const ComputeDispatch>(&dispatch, 1));
}

CommandTicket MetalBackend::submitCommandAsync(
    std::span<const ComputeDispatch> dispatches,
    CommandCompletion completion) {
    checkOperation();
#ifdef SPLASH_BACKEND_INSTRUMENTATION
    if (impl_->dispatchProfiling)
        return impl_->submitProfiled(dispatches, std::move(completion));
#endif
    Impl::PreparedCommand command = impl_->prepared(dispatches);
#if SPLASH_MTL4_AVAILABLE
    if (impl_->useMtl4()) {
        return impl_->commit4(command.dispatches,
                              std::move(command.retainedAllocations),
                              std::move(completion));
    }
#endif
    return impl_->commit(command.dispatches,
                         std::move(command.retainedAllocations),
                         std::move(completion));
}

void MetalBackend::preparePipelines(
    std::span<const ComputeDispatch> dispatches) {
    checkOperation();
    static_cast<void>(impl_->prepare(dispatches));
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

bool MetalBackend::commandInFlight() const noexcept {
    return impl_->asyncState->hasActiveSubmission();
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

#ifdef SPLASH_BACKEND_INSTRUMENTATION
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

}  // namespace splash::metal

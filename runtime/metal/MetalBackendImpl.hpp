#pragma once

// MetalBackend's private implementation: the device/allocation helpers,
// MetalBuffer::Impl, MetalAllocation, CommandTicket::State and the complete
// MetalBackend::Impl. Shared by MetalBackend.mm (lifecycle, buffers,
// tickets), MetalArena.mm (allocation) and MetalEncode.mm (pipelines,
// baked spans, submission); everything here is TU-local per includer.

#import "MetalBackend.hpp"
#include "AwakeClock.hpp"
#include "CommandWatchdog.hpp"
#include "Env.hpp"
#include "Residency.hpp"
#include "TestConfig.hpp"
// DispatchTiming is part of Impl's layout, so the struct is included
// unconditionally: its methods exist only in the instrumented build, but the
// layout must match it in every translation unit.
#include "BackendInstrumentation.hpp"

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
#include <deque>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace richengine::metal {
namespace {

// The accelerator entry that backs a Metal device publishes gpu-core-count.
// The device's registry ID names that entry or a child of it; the first
// IOAccelerator service is the fallback, since Apple silicon Macs have one
// GPU. Zero means the property was not found anywhere.
[[maybe_unused]] uint32_t gpuCoreCountForDevice(uint64_t registryId) noexcept {
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
[[maybe_unused]] std::string stringFromNSString(NSString *value) {
    if (!value) return {};
    const char *utf8 = value.UTF8String;
    return utf8 ? utf8 : "";
}
[[maybe_unused]] std::string errorDescription(NSError *error) {
    if (!error) return "unknown Metal error";
    std::string result = stringFromNSString(error.localizedDescription);
    return result.empty() ? "unknown Metal error" : result;
}
[[maybe_unused]] void readMacosVersion(DeviceCapabilities &capabilities) {
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
[[maybe_unused]] void readDeviceCapabilities(id<MTLDevice> device,
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
              "RichEngine builds for arm64 only");
[[maybe_unused]] MTLSize metalSize(const DispatchSize &size, std::string_view field) {
    if (!size.x || !size.y || !size.z) {
        throw MetalBackendError(std::string(field) + " must be non-zero");
    }
    return MTLSizeMake(size.x, size.y, size.z);
}
[[maybe_unused]] bool multiplyOverflows(uint64_t left, uint64_t right) {
    return right && left > std::numeric_limits<uint64_t>::max() / right;
}

// Entries of a kernel's buffer argument table on every Apple GPU family.
constexpr uint32_t kBufferArgumentEntries = 31;
// How long a ticket waits for its command before it asks the watchdog.
constexpr auto kTicketWaitSlice = std::chrono::seconds(1);
// The queue's outstanding command buffers; a command takes one per event
// signal and one more.
constexpr NSUInteger kMaximumCommandBuffers = 512;

// Refuses the event steps of a command of `dispatches` dispatches that its
// submission cannot encode.
[[maybe_unused]] void checkEvents(std::span<const EventStep> events, size_t dispatches) {
    size_t before = 0;
    NSUInteger signals = 0;
    for (const EventStep &step : events) {
        if (!step.event || !step.value)
            throw MetalBackendError(
                "event step needs an event and a nonzero value");
        if (step.before < before)
            throw MetalBackendError("event steps are out of dispatch order");
        if (step.before > dispatches)
            throw MetalBackendError(
                "event step follows more dispatches than its command has");
        before = step.before;
        if (step.kind == EventStep::Kind::Signal) ++signals;
    }
    if (signals >= kMaximumCommandBuffers)
        throw MetalBackendError("Metal command signals more events than its "
                                "queue holds command buffers");
}
[[maybe_unused]] double awakeSeconds() noexcept {
    return std::chrono::duration<double>(AwakeClock::now().time_since_epoch()).count();
}
[[maybe_unused]] const char *commandStatusName(MTLCommandBufferStatus status) noexcept {
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
[[maybe_unused]] NSString *checkedNSString(std::string_view value, std::string_view field) {
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
    // The owner of wrapped memory (MetalBackend::wrapSharedMemory), which is
    // never released: our views keep it as well as Metal's deallocator.
    std::shared_ptr<void> owner;
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
// released: only allocateBuffer, wrapSharedMemory and view create one.
struct MetalBuffer::Impl {
    std::shared_ptr<MetalAllocation> allocation;
    uint64_t offsetBytes = 0;
    uint64_t lengthBytes = 0;
};

struct SharedEvent::Impl {
    __strong id<MTLSharedEvent> event = nil;
    // The listener of the backend that created the event, which runs its
    // notify() callbacks; the backend and each of its events hold it.
    __strong MTLSharedEventListener *listener = nil;
};

struct BackendAsyncState {
    explicit BackendAsyncState(double commandTimeoutSeconds)
        : commandTimeoutSeconds_(commandTimeoutSeconds) {
        if (!std::isfinite(commandTimeoutSeconds) || commandTimeoutSeconds <= 0.0)
            throw std::invalid_argument(
                "command timeout must be finite and positive");
    }

    // One GPU command submitted and not yet released. Commands complete in
    // submission order on the single queue; the ring admits a bounded number
    // so the next command can be committed while this one still runs.
    struct InFlight {
        uint64_t sequence = 0;
        size_t dispatchCount = 0;
        __weak id<MTLCommandBuffer> command = nil;
        std::function<void(id<MTLCommandBuffer>)> completion;
        // The Metal 4 path has no command object whose status the watchdog
        // can read: its commit feedback is the only completion signal, so an
        // expired watch there is always a timeout.
        bool mtl4 = false;
        double deadlineSeconds = 0.0;
        // Stops the watch on GPU completion even if the model ticket keeps
        // waiting for CPU work; release drops the entry itself.
        bool watching = false;
    };

    __strong id<MTLDevice> device = nil;
    mutable std::atomic<uint64_t> deviceCurrentAllocatedBytes{0};
    mutable std::atomic<uint64_t> devicePeakAllocatedBytes{0};
    std::atomic<bool> healthy{true};
    mutable std::mutex healthMutex;
    std::string healthReason;
    mutable std::mutex gateMutex;
    uint64_t nextSequence = 0;
    std::deque<InFlight> inFlight;
    const double commandTimeoutSeconds_;
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

    uint64_t beginSubmission(size_t dispatchCount, size_t maximumInFlight = 1) {
        ensureHealthy();
        std::lock_guard lock(gateMutex);
        if (stopping)
            throw MetalBackendError("Metal backend is stopping");
        if (inFlight.size() >= maximumInFlight) {
            throw MetalBackendError(
                "Metal backend already has an in-flight command");
        }
        const uint64_t sequence = ++nextSequence;
        InFlight entry;
        entry.sequence = sequence;
        entry.dispatchCount = dispatchCount;
        entry.deadlineSeconds = awakeSeconds() + commandTimeoutSeconds_;
        entry.watching = true;
        inFlight.push_back(std::move(entry));
        return sequence;
    }

    void commitSubmission(uint64_t sequence,
                          const std::vector<id<MTLCommandBuffer>> &leading,
                          id<MTLCommandBuffer> command,
                          std::function<void(id<MTLCommandBuffer>)> completion) {
        std::lock_guard lock(gateMutex);
        InFlight &entry = inFlightEntry(sequence);
        entry.command = command;
        entry.completion = std::move(completion);
        for (id<MTLCommandBuffer> earlier : leading) [earlier commit];
        [command commit];
    }

    // The MTL4 equivalent of commitSubmission: the caller commits on its
    // queue right after this registers the watch.
    void commitSubmission4(uint64_t sequence) {
        std::lock_guard lock(gateMutex);
        inFlightEntry(sequence).mtl4 = true;
    }

    void releaseSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        for (auto it = inFlight.begin(); it != inFlight.end(); ++it) {
            if (it->sequence == sequence) {
                inFlight.erase(it);
                return;
            }
        }
    }

    void completeSubmission(uint64_t sequence) noexcept {
        std::lock_guard lock(gateMutex);
        for (InFlight &entry : inFlight)
            if (entry.sequence == sequence) entry.watching = false;
    }

    // Runs the command watchdog. A terminal command whose callback is late
    // is completed here; one still running past its timeout marks the
    // backend unhealthy, and the answer is then true: the backend gave up on
    // that command.
    [[nodiscard]] bool commandAbandoned() noexcept {
        std::vector<std::pair<std::function<void(id<MTLCommandBuffer>)>,
                              id<MTLCommandBuffer>>>
            recovered;
        {
            std::lock_guard lock(gateMutex);
            const double now = awakeSeconds();
            bool expired = false;
            for (InFlight &entry : inFlight) {
                if (!entry.watching || now < entry.deadlineSeconds) continue;
                expired = true;
                id<MTLCommandBuffer> command = entry.command;
                const auto status = command ? command.status
                                            : MTLCommandBufferStatusNotEnqueued;
                // Recover terminal results even if the driver has not
                // delivered its callback. Finish outside the gate: it takes
                // the ticket lock. The Metal 4 path publishes completion only
                // through its commit feedback, which has already run finish()
                // when it arrives, so a still-watching Metal 4 command always
                // counts as timed out.
                if (command && (status == MTLCommandBufferStatusCompleted ||
                                status == MTLCommandBufferStatusError)) {
                    entry.watching = false;
                    recovered.emplace_back(entry.completion, command);
                } else {
                    // Waits that must not throw run this too: the reason
                    // drops its details when they cannot be formatted.
                    std::string reason = "Metal command completion timed out";
                    try {
                        std::ostringstream message;
                        message << reason << " after " << commandTimeoutSeconds_
                                << " seconds (sequence=" << entry.sequence
                                << ", status="
                                << (command ? commandStatusName(status)
                                            : (entry.mtl4 ? "mtl4" : "unavailable"))
                                << ", dispatches=" << entry.dispatchCount << ')';
                        reason = message.str();
                    } catch (const std::bad_alloc &) {
                    }
                    markUnhealthy(std::move(reason));
                    return true;
                }
            }
            if (!expired) return false;
        }
        for (auto &[complete, command] : recovered)
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
        return !inFlight.empty();
    }

private:
    InFlight &inFlightEntry(uint64_t sequence) {
        for (InFlight &entry : inFlight)
            if (entry.sequence == sequence) return entry;
        throw MetalBackendError("Metal submission sequence is not in flight");
    }
};

// RICHENGINE_OP_TIMINGS: merges one dispatch's measured GPU seconds into
// the per-pipeline table (MetalEncode.mm).
void recordOpTiming(const std::string &name, double gpuSeconds);

struct CommandTicket::State {
    std::shared_ptr<BackendAsyncState> backend;
    std::vector<std::shared_ptr<MetalAllocation>> retainedAllocations;
    CommandCompletion completion;
    mutable std::mutex mutex;
    std::condition_variable condition;
    uint64_t sequence = 0;
    CommandTiming timing;
    AwakeClock::time_point wallStart;
    // Command buffers committed before the last one, split at event signals.
    std::vector<id<MTLCommandBuffer>> leadingCommands;
    std::string error;
    bool completed = false;
    bool released = false;

    void finishCommand(id<MTLCommandBuffer> command) {
        auto wallEnd = AwakeClock::now();
        CommandTiming timing;
        const double gpuStart = leadingCommands.empty()
            ? command.GPUStartTime : leadingCommands.front().GPUStartTime;
        timing.gpuSeconds = command.GPUEndTime - gpuStart;
        if (!std::isfinite(timing.gpuSeconds) || timing.gpuSeconds < 0.0) {
            timing.gpuSeconds = 0.0;
        }
        timing.wallSeconds =
            std::chrono::duration<double>(wallEnd - wallStart).count();

        std::string error;
        // The first buffer that failed names the error: the buffers after a
        // failed one still run, as its signal is delivered (EventStep).
        id<MTLCommandBuffer> failed =
            command.status != MTLCommandBufferStatusCompleted ? command : nil;
        for (id<MTLCommandBuffer> earlier : leadingCommands) {
            if (earlier.status == MTLCommandBufferStatusError) {
                failed = earlier;
                break;
            }
        }
        if (failed) {
            std::ostringstream message;
            message << "Metal command " << sequence << " failed";
            if (failed.error) {
                message << ": " << errorDescription(failed.error);
            }
            error = message.str();
        }

        finish(timing, std::move(error));
    }

#if RICHENGINE_MTL4_AVAILABLE
    // The Metal 4 commit feedback carries the same terminal state the Metal 3
    // completed handler reads off its command buffer.
    void finishCommand4(id<MTL4CommitFeedback> feedback) {
        auto wallEnd = AwakeClock::now();
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

    // Always present, whatever the translation unit's instrumentation flag:
    // the instrumented and plain objects that share an Impl must agree on
    // its layout.
    bool dispatchProfiling = false;
    std::vector<DispatchTiming> dispatchProfile;
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    // Runs the notify() callbacks of the backend's shared events, on a
    // serial dispatch queue.
    __strong MTLSharedEventListener *eventListener = nil;
#if RICHENGINE_MTL4_AVAILABLE
    // The Metal 4 submission path, used only while mtl4Enabled() opted in.
    // The submit-ahead ring keeps two commands in flight, so the allocator
    // and the staging buffer the host writes come in two banks selected by
    // submission parity; the argument table snapshots at each dispatch and
    // is shared.
    __strong id<MTL4CommandQueue> queue4 = nil;
    __strong id<MTL4CommandAllocator> allocator4[2] = {nil, nil};
    __strong id<MTL4ArgumentTable> argumentTable4 = nil;
    // MTL4ArgumentTable binds GPU addresses and has no setBytes equivalent:
    // every BytesBinding payload is staged here, 256-byte aligned, and bound
    // as an address into this buffer. Grown to a submission's total.
    std::shared_ptr<MetalAllocation> staging4[2];
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
                            NSString *label);


    // The base allocation of a buffer, which must be a whole buffer of this
    // backend.
    MetalAllocation &baseAllocation(const MetalBuffer &buffer) const;


    id<MTLComputePipelineState> pipeline(std::string_view name);


    id<MTLComputePipelineState> newPipeline(std::string_view name);


    // The pipeline a baked indirect command binds: the same function as
    // pipeline(), created with indirect command buffer support, which the
    // default pipeline object does not carry.
    id<MTLComputePipelineState> indirectPipeline(std::string_view name);


    // The escape hatch: baked indirect dispatch is off entirely.
    [[nodiscard]] static bool icbDisabled() noexcept;


    // The Metal 4 encoder is opt-in (RICHENGINE_MTL4=1) while it is benchmarked
    // against the Metal 3 path it mirrors.
    [[nodiscard]] static bool mtl4Enabled() noexcept;


    [[nodiscard]] bool useMtl4() const noexcept;


    // Proves a bound view's allocation identity once per allocation per
    // pass: a previous successful check's epoch stamp stands in for the
    // weak lock and object load below. The stamp must fail for released
    // memory so a later validation reports it — the nil check cannot tell
    // released from never-attached.
    [[nodiscard]] bool
    bindingIdentity(const std::weak_ptr<MetalAllocation> &saved,
                    __weak id<MTLBuffer> savedObject,
                    const MetalBuffer::Impl &view) noexcept;


    // Byte-for-byte identity of a run against a baked snapshot; the buffer
    // identity check catches released-and-restored memory the MetalBuffer
    // view metadata alone would miss.
    [[nodiscard]] bool
    snapshotMatches(const BakedSpan &span,
                    std::span<const ComputeDispatch> run) noexcept;


    // Bakes one validated run into an indirect command buffer. The commands
    // carry their pipeline and buffer objects directly (nothing is
    // inherited).
    std::shared_ptr<BakedSpan>
    bakeSpan(std::span<const ComputeDispatch> run);


    // Rewrites a span's staged parameters from the run's current patchable
    // payloads: the indirect commands bind the parameter arena by address,
    // so the same commands replay with the new parameters. One command in
    // flight means the GPU finished reading the arena of the previous
    // submission before this patch runs.
    static void patchSpanParams(const BakedSpan &span,
                                std::span<const PreparedDispatch> run);


    // Finds a baked span matching this run or bakes a new one. Returns
    // nullptr when the span cannot bake; the run then encodes directly.
    std::shared_ptr<const BakedSpan>
    resolveBakedSpan(size_t begin,
                     std::span<const ComputeDispatch> run);


    PreparedCommand prepare(std::span<const ComputeDispatch> dispatches);


    // Stamps every bound allocation into the submission's retained set,
    // once. Cheaper than sort-and-unique over the run's bindings.
    void retainBound(std::span<const ComputeDispatch> dispatches,
                     std::vector<std::shared_ptr<MetalAllocation>> &retained);


    // Whole-command identity of a run against a cached prepared command;
    // covers every field prepare() resolves — so a match implies every
    // baked span it references still matches its run too.
    [[nodiscard]] bool
    commandMatches(const CachedCommand &cached,
                   std::span<const ComputeDispatch> run) noexcept;


    // Snapshots a run the way commandMatches validates it.
    std::unique_ptr<CachedCommand>
    snapshotCommand(std::span<const ComputeDispatch> run,
                    PreparedCommand &command);


    // The escape hatch for the prepared-command cache only.
    [[nodiscard]] static bool preparedCacheDisabled() noexcept;


    // prepare(), or its cached equivalent when the run is field-for-field
    // identical to a recent submission: reuse skips validation, pipeline
    // lookups, the retained-set build and every baked-span resolution.
    PreparedCommand prepared(std::span<const ComputeDispatch> dispatches);


    // Encodes and commits prepared dispatches and the event steps between
    // them; the ticket retains `retained` until it is consumed.
    CommandTicket commit(std::span<const PreparedDispatch> dispatches,
                         std::span<const EventStep> events,
                         std::vector<std::shared_ptr<MetalAllocation>> retained,
                         CommandCompletion completion);


#if RICHENGINE_MTL4_AVAILABLE
    // The Metal 4 encoding of commit(): same prepared dispatches, same
    // ticket contract, encoded through one reused allocator and argument
    // table and committed with a commit-feedback handler in place of the
    // completed handler.
    CommandTicket commit4(std::span<const PreparedDispatch> dispatches,
                          std::vector<std::shared_ptr<MetalAllocation>> retained,
                          CommandCompletion completion);

#endif

    // Commits every dispatch of the command as its own command and waits
    // for it, then hands back an already-completed ticket with the summed
    // timing, so callers observe the usual asynchronous contract. The flag
    // selecting it is unconditional, so the method is too.
    CommandTicket submitProfiled(std::span<const ComputeDispatch> dispatches,
                                 CommandCompletion completion);
};

} // namespace richengine::metal

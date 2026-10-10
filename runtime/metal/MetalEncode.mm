// MetalEncode.mm: pipeline cache, baked spans, command preparation and
// submission.

#import "MetalBackend.hpp"
#include "AwakeClock.hpp"
#include "CommandWatchdog.hpp"
#include "Tuning.hpp"
#include "metal/abi/ExecutionGeometry.h"
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

namespace {

// RICHENGINE_OP_TIMINGS=1 accumulates per-pipeline GPU seconds from the
// encoder's timestamp samples; dumpOpTimings() prints the running table.
struct OpTimingTable {
  std::mutex mutex;
  std::unordered_map<std::string, std::pair<double, uint64_t>> seconds;
};

OpTimingTable &opTimingTable() {
  static OpTimingTable *table = new OpTimingTable;
  return *table;
}

} // namespace

void recordOpTiming(const std::string &name, double gpuSeconds) {
  if (!std::isfinite(gpuSeconds) || gpuSeconds < 0.0) return;
  std::lock_guard lock(opTimingTable().mutex);
  opTimingTable().seconds[name].first += gpuSeconds;
  opTimingTable().seconds[name].second += 1;
}

void dumpOpTimings() {
  auto &table = opTimingTable();
  std::vector<std::pair<std::string, double>> rows;
  {
    std::lock_guard lock(table.mutex);
    rows.reserve(table.seconds.size());
    for (const auto &[name, totals] : table.seconds)
      rows.emplace_back(name + " x" + std::to_string(totals.second),
                        totals.first);
  }
  std::sort(rows.begin(), rows.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  double total = 0.0;
  for (const auto &[name, seconds] : rows) total += seconds;
  fprintf(stderr, "op timings (cumulative %.1fms):\n", total * 1000.0);
  for (size_t i = 0; i < rows.size() && i < 24; ++i)
    fprintf(stderr, "  %8.3fms  %s\n", rows[i].second * 1000.0,
            rows[i].first.c_str());
}

// The env-tuned kernel geometry (Tuning.hpp, ExecutionGeometry.h's
// function-constant table): built once like the tuning snapshot and
// supplied to every function — indices a kernel does not declare are
// ignored, so kernels without the constants compile unchanged.
static MTLFunctionConstantValues *functionConstantValues() {
    static MTLFunctionConstantValues *values = [] {
        MTLFunctionConstantValues *result = [MTLFunctionConstantValues new];
        uint32_t splits = tuning().draftAttentionSplits;
        [result setConstantValue:&splits
                            type:MTLDataTypeUInt
                         atIndex:RICHENGINE_DRAFT_SPLITS_FUNCTION_CONSTANT];
        return result;
    }();
    return values;
}

id<MTLComputePipelineState> MetalBackend::Impl::pipeline(std::string_view name) {
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

id<MTLComputePipelineState> MetalBackend::Impl::newPipeline(std::string_view name) {
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLFunction> function =
            [library newFunctionWithName:key
                          constantValues:functionConstantValues()
                                   error:nil];
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
    id<MTLComputePipelineState> MetalBackend::Impl::indirectPipeline(std::string_view name) {
        if (const auto cached = icbPipelines.find(name);
            cached != icbPipelines.end())
            return cached->second;
        NSString *key = checkedNSString(name, "pipeline name");
        id<MTLFunction> function =
            [library newFunctionWithName:key
                          constantValues:functionConstantValues()
                                   error:nil];
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
#if RICHENGINE_MTL4_AVAILABLE
        // Metal 4 resolves an indirect command's pipeline through residency
        // sets, not the per-encoder usage declarations the Metal 3 replay
        // makes.
        if (mtl4Enabled()) residency->add(result);
#endif
        sampleDeviceMemory();
        return result;
    }

// The escape hatch: baked indirect dispatch is off entirely.
    bool MetalBackend::Impl::icbDisabled() noexcept {
        return tuning().icbOff;
    }

// The Metal 4 encoder is opt-in (RICHENGINE_MTL4=1) while it is benchmarked
    // against the Metal 3 path it mirrors.
    bool MetalBackend::Impl::mtl4Enabled() noexcept {
        return tuning().mtl4;
    }

bool MetalBackend::Impl::useMtl4() const noexcept {
#if RICHENGINE_MTL4_AVAILABLE
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
    bool
    MetalBackend::Impl::bindingIdentity(const std::weak_ptr<MetalAllocation> &saved,
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
    bool
    MetalBackend::Impl::snapshotMatches(const BakedSpan &span,
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
    std::shared_ptr<MetalBackend::Impl::BakedSpan>
    MetalBackend::Impl::bakeSpan(std::span<const ComputeDispatch> run) {
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
#if RICHENGINE_MTL4_AVAILABLE
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
    void MetalBackend::Impl::patchSpanParams(const BakedSpan &span,
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
    std::shared_ptr<const MetalBackend::Impl::BakedSpan>
    MetalBackend::Impl::resolveBakedSpan(size_t begin,
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

MetalBackend::Impl::PreparedCommand MetalBackend::Impl::prepare(std::span<const ComputeDispatch> dispatches) {
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
    void MetalBackend::Impl::retainBound(std::span<const ComputeDispatch> dispatches,
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
    bool
    MetalBackend::Impl::commandMatches(const CachedCommand &cached,
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
    std::unique_ptr<MetalBackend::Impl::CachedCommand>
    MetalBackend::Impl::snapshotCommand(std::span<const ComputeDispatch> run,
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
    bool MetalBackend::Impl::preparedCacheDisabled() noexcept {
        return tuning().preparedCacheOff;
    }

// prepare(), or its cached equivalent when the run is field-for-field
    // identical to a recent submission: reuse skips validation, pipeline
    // lookups, the retained-set build and every baked-span resolution.
    MetalBackend::Impl::PreparedCommand MetalBackend::Impl::prepared(std::span<const ComputeDispatch> dispatches) {
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

// Encodes and commits prepared dispatches and the event steps between
    // them; the ticket retains `retained` until it is consumed.
    CommandTicket MetalBackend::Impl::commit(std::span<const PreparedDispatch> dispatches,
                         std::span<const EventStep> events,
                         std::vector<std::shared_ptr<MetalAllocation>> retained,
                         CommandCompletion completion) {
        // A second command may be committed while this one is still in
        // flight (the submit-ahead ring). Baked spans stay exclusive to the
        // GPU-idle case: patchSpanParams rewrites the shared parameter arena
        // an in-flight replay still reads, so a pipelined submission encodes
        // every dispatch directly instead. A command with event steps also
        // encodes directly: a replayed span would hide the dispatches its
        // steps order against the other agent (EventStep).
        const bool replaySpans =
            events.empty() && !asyncState->hasActiveSubmission();
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        ticketState->retainedAllocations = std::move(retained);
        ticketState->sequence =
            asyncState->beginSubmission(dispatches.size(), 2);

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
            // The buffers before `command`, each ended by an event signal.
            std::vector<id<MTLCommandBuffer>> leading;
            id<MTLComputeCommandEncoder> encoder = nil;
            // RICHENGINE_OP_TIMINGS: commit every dispatch as its own
            // command buffer so its GPU span is readable on completion.
            // Buffers serialize on the queue, so per-dispatch seconds are
            // kernel-time only — wall time inflates by the inter-buffer
            // gaps and adjacent-dispatch overlap disappears; a probe, not
            // a benchmark. (Encoder counter sampling is unsupported on this
            // GPU family.) Buffers commit eagerly: the queue caps
            // outstanding uncommitted buffers well below a command's
            // dispatch count.
            const bool opTimings = tuning().opTimings;
            // Encodes the event steps that follow the first `encoded`
            // dispatches.
            auto step = events.begin();
            const auto encodeSteps = [&](size_t encoded) {
                for (; step != events.end() && step->before == encoded; ++step) {
                    if (encoder) {
                        [encoder endEncoding];
                        encoder = nil;
                    }
                    id<MTLSharedEvent> event =
                        (__bridge id<MTLSharedEvent>)step->event.nativeHandle();
                    if (step->kind == EventStep::Kind::Wait) {
                        [command encodeWaitForEvent:event value:step->value];
                        continue;
                    }
                    [command encodeSignalEvent:event value:step->value];
                    // A buffer that fails may end without its signal; the
                    // work after it, waiting on an agent that waits on the
                    // signal, would then stall until a timeout ends it. The
                    // CPU delivers the signal instead, so the command ends
                    // with the failure at once.
                    const SharedEvent signaled = step->event;
                    const uint64_t value = step->value;
                    [command addCompletedHandler:^(id<MTLCommandBuffer> ended) {
                        if (ended.status == MTLCommandBufferStatusError)
                            signaled.signal(value);
                    }];
                    if (opTimings) {
                        [command commit];
                    } else {
                        leading.push_back(command);
                    }
                    command = [queue commandBuffer];
                    if (!command) {
                        failBeforeCommit("unable to create Metal command buffer");
                    }
                }
            };
            // Indexed by argument table entry. The ticket and the dispatches
            // keep the buffers alive.
            __unsafe_unretained id<MTLBuffer> buffers[kBufferArgumentEntries];
            NSUInteger offsets[kBufferArgumentEntries];
            const auto closeDispatch = [&](std::string_view name) {
                if (!opTimings || !encoder) return;
                [encoder endEncoding];
                encoder = nil;
                // The handler outlives this stack frame: copy the name.
                const std::string nameCopy(name);
                [command addCompletedHandler:^(id<MTLCommandBuffer> finished) {
                    recordOpTiming(nameCopy,
                                   finished.GPUEndTime - finished.GPUStartTime);
                }];
                [command commit];
                command = [queue commandBuffer];
            };
            for (size_t index = 0; index < dispatches.size(); ++index) {
                encodeSteps(index);
                const PreparedDispatch &item = dispatches[index];
                if (!encoder) {
                    encoder = [command computeCommandEncoder];
                    if (!encoder) {
                        failBeforeCommit("unable to create Metal compute encoder");
                    }
                }
                // A baked span replays its whole run from the indirect
                // command buffer; the run's own dispatches are skipped.
                if (item.span && replaySpans &&
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
                    closeDispatch(std::string(item.source->pipelineName));
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
                closeDispatch(std::string(dispatch.pipelineName));
            }
            encodeSteps(dispatches.size());
            if (encoder) [encoder endEncoding];
            ticketState->leadingCommands = leading;

            // Driver callbacks only complete the ticket. Device-wide memory
            // telemetry is sampled on the host when consuming the result. The
            // handler holds the ticket's state strongly: once a waiter gives
            // up on the command, it keeps the retained allocations until the
            // GPU ends.
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                ticketState->finishCommand(completed);
            }];
            residency->use();
            asyncState->commitSubmission(ticketState->sequence, leading, command,
                [weakTicket = std::weak_ptr(ticketState)](
                    id<MTLCommandBuffer> completed) {
                    if (auto ticket = weakTicket.lock())
                        ticket->finishCommand(completed);
                });
        }
        return CommandTicket(std::move(ticketState));
    }

#if RICHENGINE_MTL4_AVAILABLE
    // The Metal 4 encoding of commit(): same prepared dispatches, same
    // ticket contract, encoded through one reused allocator and argument
    // table and committed with a commit-feedback handler in place of the
    // completed handler.
    CommandTicket MetalBackend::Impl::commit4(std::span<const PreparedDispatch> dispatches,
                          std::vector<std::shared_ptr<MetalAllocation>> retained,
                          CommandCompletion completion) {
        // The submit-ahead ring may commit a second command while this one
        // runs. Baked spans stay exclusive to the GPU-idle case:
        // patchSpanParams rewrites the shared parameter arena an in-flight
        // replay still reads, so a pipelined submission encodes every
        // dispatch directly instead.
        const bool replaySpans = !asyncState->hasActiveSubmission();
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        ticketState->sequence =
            asyncState->beginSubmission(dispatches.size(), 2);
        // Submission parity selects the staging and allocator bank the
        // in-flight command does not use.
        const uint32_t bank = ticketState->sequence & 1;
        auto &staging = staging4[bank];
        // Stage every BytesBinding payload: an argument table binds
        // addresses only. Each payload keeps the 256-byte alignment a
        // baked span's parameters use. Span-covered runs stage their
        // payloads in the span's own parameter arena instead.
        uint64_t stagingBytes = 0;
        for (size_t index = 0; index < dispatches.size(); ++index) {
            const PreparedDispatch &item = dispatches[index];
            if (item.span && replaySpans &&
                index + item.span->dispatches.size() <= dispatches.size()) {
                index += item.span->dispatches.size() - 1;
                continue;
            }
            for (const BytesBinding &binding : item.source->bytes)
                stagingBytes += (binding.sizeBytes + 255) & ~uint64_t{255};
        }
        if (stagingBytes && (!staging || staging->length < stagingBytes)) {
            auto allocation = std::make_shared<MetalAllocation>();
            allocation->accounting = accounting;
            allocation->length = stagingBytes;
            allocation->storage = BufferStorage::Shared;
            allocation->label = @"mtl4-params";
            allocation->residency = residency;
            allocation->attach(
                newBuffer(stagingBytes, BufferStorage::Shared,
                          allocation->label));
            staging = std::move(allocation);
        }
        // The staging buffer backs the command's byte bindings for as long
        // as the GPU may read it.
        if (staging && stagingBytes)
            retained.push_back(staging);
        ticketState->retainedAllocations = std::move(retained);

        auto failBeforeCommit = [&](std::string message) {
            markUnhealthy(message);
            asyncState->releaseSubmission(ticketState->sequence);
            throw MetalBackendError(std::move(message));
        };

        auto wallStart = AwakeClock::now();
        // The allocator bank may be reused once the command two submissions
        // back ended on the GPU, and the serving loop's pool never drains,
        // so temporary ownership ends with this submission like the
        // Metal 3 path's.
        @autoreleasepool {
            [allocator4[bank] reset];
            id<MTL4CommandBuffer> command = [device newCommandBuffer];
            if (!command) {
                failBeforeCommit("unable to create Metal command buffer");
            }
            [command beginCommandBufferWithAllocator:allocator4[bank]];
            ticketState->wallStart = wallStart;
            id<MTL4ComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            if (!encoder) {
                failBeforeCommit("unable to create Metal compute encoder");
            }
            uint8_t *stagingContents =
                staging ? static_cast<uint8_t *>(staging->buffer.contents)
                        : nullptr;
            const MTLGPUAddress stagingAddress =
                staging ? staging->buffer.gpuAddress : 0;
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
                        item.span && replaySpans &&
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
                if (item.span && replaySpans &&
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

    // Commits every dispatch of the command as its own command and waits
    // for it, then hands back an already-completed ticket with the summed
    // timing, so callers observe the usual asynchronous contract. Built
    // unconditionally: this file's submitCommandAsync is one object for the
    // instrumented and plain Impls that share it.
    CommandTicket MetalBackend::Impl::submitProfiled(std::span<const ComputeDispatch> dispatches,
                                 CommandCompletion completion) {
        const PreparedCommand command = prepare(dispatches);
        CommandTiming total;
        for (const PreparedDispatch &item : command.dispatches) {
            const CommandTiming timing =
                commit({&item, 1}, {}, command.retainedAllocations, {}).wait();
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
    return submitCommandAsync(Command{dispatches, {}}, std::move(completion));
}

CommandTicket MetalBackend::submitCommandAsync(const Command &command,
                                               CommandCompletion completion) {
    checkOperation();
    // Always present, whatever this translation unit's instrumentation flag:
    // submitProfiled is only reachable in objects that carry it.
    if (impl_->dispatchProfiling) {
        if (!command.events.empty())
            throw MetalBackendError(
                "dispatch profiling does not replay event steps");
        return impl_->submitProfiled(command.dispatches, std::move(completion));
    }
    checkEvents(command.events, command.dispatches.size());
    Impl::PreparedCommand prepared = impl_->prepared(command.dispatches);
#if RICHENGINE_MTL4_AVAILABLE
    // The Metal 4 path has no shared-event encoding: a command with event
    // steps takes the Metal 3 encoder.
    if (command.events.empty() && impl_->useMtl4()) {
        return impl_->commit4(prepared.dispatches,
                              std::move(prepared.retainedAllocations),
                              std::move(completion));
    }
#endif
    return impl_->commit(prepared.dispatches, command.events,
                         std::move(prepared.retainedAllocations),
                         std::move(completion));
}

void MetalBackend::preparePipelines(
    std::span<const ComputeDispatch> dispatches) {
    checkOperation();
    static_cast<void>(impl_->prepare(dispatches));
}

bool MetalBackend::commandInFlight() const noexcept {
    return impl_->asyncState->hasActiveSubmission();
}


} // namespace richengine::metal

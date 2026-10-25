// MetalEncode.mm: pipeline cache, baked spans, command preparation and
// submission.

#import "MetalBackend.hpp"
#include "AwakeClock.hpp"
#include "CommandWatchdog.hpp"
#include "Tuning.hpp"
#include "model/ModelPolicy.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GpuDispatch.h"
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
#include <cassert>
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
MTLFunctionConstantValues *functionConstantValues() {
    // Model-scoped policy values are baked at call time — a second Runtime
    // on a different maker must not inherit a stale constant.
    MTLFunctionConstantValues *result = [MTLFunctionConstantValues new];
    uint32_t splits = model::modelPolicy().draftAttentionSplits;
    [result setConstantValue:&splits
                        type:MTLDataTypeUInt
                     atIndex:RICHENGINE_DRAFT_SPLITS_FUNCTION_CONSTANT];
    return result;
}

id<MTLComputePipelineState> MetalBackend::Impl::pipeline(std::string_view name) {
        if (name.empty()) {
            throw MetalBackendError("Metal pipeline name must not be empty");
        }
        if (const auto cached = pipelines.find(name); cached != pipelines.end())
            return cached->second;
        id<MTLComputePipelineState> result =
            name.find_first_of("[|") == std::string_view::npos
                ? newPipeline(name)
                : newSpecializedPipeline(name);
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

// The extended pipeline-name grammar, resolved by newSpecializedPipeline:
//
//   entry[hook=impl:arity,...]   call-site function stitching: every
//     [[stitchable]] function `hook` declared in `entry` is bound to a
//     single-node stitch graph whose output calls the library function
//     `impl`, forwarded the hook's `arity` leading arguments. Multiple
//     hooks are comma-separated. The kernel is looked up in the stitched
//     library, whose synthesized hook functions carry the bindings.
//   entry|option,...             pipeline options compiled through the
//     Metal 4 compiler (MTL4Compiler): `persistent` sets
//     optimizeForPersistentKernel, `fp=weak`/`fp=simd` set
//     forwardProgressUsage (for device-side atomic wait/notify kernels),
//     `relief=none` sets contentionRelief none.
//
// Stitches and options compose: `entry[hook=impl:1]|persistent`.
namespace {

struct PipelineOptionSpec {
    std::string_view entry;
    struct Hook {
        std::string_view name;
        std::string_view impl;
        uint32_t arity = 0;
    };
    std::vector<Hook> hooks;
    bool persistent = false;
    bool reliefNone = false;
    // 0 unset, else an MTLForwardProgressUsage value.
    NSInteger forwardProgress = 0;
};

[[maybe_unused]] PipelineOptionSpec parsePipelineSpec(std::string_view name) {
    PipelineOptionSpec spec;
    const auto take = [](std::string_view &text, char stop) {
        const size_t at = text.find(stop);
        std::string_view part = text.substr(0, at);
        text = at == std::string_view::npos ? std::string_view{}
                                            : text.substr(at + 1);
        return part;
    };
    const size_t hooksAt = name.find('[');
    const size_t hooksEnd =
        hooksAt == std::string_view::npos ? std::string_view::npos
                                          : name.find(']', hooksAt);
    const size_t optionsAt = name.find('|');
    spec.entry = name.substr(0, std::min({hooksAt, optionsAt}));
    if (hooksAt != std::string_view::npos) {
        if (hooksEnd == std::string_view::npos)
            throw MetalBackendError(
                "unterminated stitch list in pipeline name " +
                std::string(name));
        std::string_view list = name.substr(hooksAt + 1,
                                            hooksEnd - hooksAt - 1);
        while (!list.empty()) {
            std::string_view binding = take(list, ',');
            PipelineOptionSpec::Hook hook;
            hook.name = take(binding, '=');
            std::string_view arity = binding;
            hook.impl = take(arity, ':');
            if (hook.name.empty() || hook.impl.empty() || arity.empty())
                throw MetalBackendError(
                    "stitch binding needs hook=impl:arity in " +
                    std::string(name));
            hook.arity = static_cast<uint32_t>(
                std::stoul(std::string(arity)));
            spec.hooks.push_back(hook);
        }
    }
    if (optionsAt != std::string_view::npos) {
        std::string_view list = name.substr(optionsAt + 1);
        while (!list.empty()) {
            const std::string_view option = take(list, ',');
            if (option == "persistent") {
                spec.persistent = true;
            } else if (option == "relief=none") {
                spec.reliefNone = true;
            } else if (option == "fp=weak") {
                spec.forwardProgress = MTLForwardProgressUsageWeak;
            } else if (option == "fp=simd") {
                spec.forwardProgress =
                    MTLForwardProgressUsageSIMDGroupParallel;
            } else {
                throw MetalBackendError(
                    "unknown pipeline option " + std::string(option) +
                    " in " + std::string(name));
            }
        }
    }
    return spec;
}

} // namespace

// A stitched or optioned pipeline (the grammar above). Stitching resolves
// through a stitched library, which both encoder paths can dispatch; the
// options go through the Metal 4 compiler's pipeline descriptor, with the
// Metal 3 descriptor as the same-fields fallback.
    id<MTLComputePipelineState>
    MetalBackend::Impl::newSpecializedPipeline(std::string_view name) {
        const PipelineOptionSpec spec = parsePipelineSpec(name);
        // The kernel itself comes from the main library; hooks are bound
        // through a stitched library's synthesized functions, linked into
        // the pipeline as private functions.
        id<MTLFunction> function =
            [library newFunctionWithName:
                         checkedNSString(spec.entry, "pipeline name")
                            constantValues:functionConstantValues()
                                     error:nil];
        if (!function)
            throw MetalBackendError(
                "missing Metal function: " + std::string(spec.entry));
        NSMutableArray<id<MTLFunction>> *bound = nil;
        id<MTLLibrary> fnLibrary = nil;
        if (!spec.hooks.empty()) {
            MTLStitchedLibraryDescriptor *stitch =
                [MTLStitchedLibraryDescriptor new];
            NSMutableArray<id<MTLFunction>> *pool = [NSMutableArray new];
            NSMutableArray<MTLFunctionStitchingGraph *> *graphs =
                [NSMutableArray new];
            for (const PipelineOptionSpec::Hook &hook : spec.hooks) {
                NSString *implName =
                    checkedNSString(hook.impl, "stitch implementation");
                id<MTLFunction> impl =
                    [library newFunctionWithName:implName];
                if (!impl)
                    throw MetalBackendError(
                        "missing Metal stitch function: " +
                        std::string(hook.impl));
                NSMutableArray<MTLFunctionStitchingInputNode *> *args =
                    [NSMutableArray new];
                for (uint32_t a = 0; a < hook.arity; ++a)
                    [args addObject:[[MTLFunctionStitchingInputNode alloc]
                                        initWithArgumentIndex:a]];
                MTLFunctionStitchingFunctionNode *node =
                    [[MTLFunctionStitchingFunctionNode alloc]
                              initWithName:implName
                                 arguments:args
                       controlDependencies:@[]];
                [graphs addObject:
                    [[MTLFunctionStitchingGraph alloc]
                        initWithFunctionName:
                            checkedNSString(hook.name, "stitch hook")
                                       nodes:@[node]
                                  outputNode:node
                                  attributes:
                            @[ [MTLFunctionStitchingAttributeAlwaysInline
                                   new] ]]];
                [pool addObject:impl];
            }
            stitch.functionGraphs = graphs;
            stitch.functions = pool;
            NSError *error = nil;
            fnLibrary =
                [device newLibraryWithStitchedDescriptor:stitch error:&error];
            if (!fnLibrary)
                throw MetalBackendError(
                    "unable to create stitched library for " +
                    std::string(name) + ": " + errorDescription(error));
            stitchedLibraries.emplace(std::string(name), fnLibrary);
            // Each synthesized hook function is what the kernel's
            // [[stitchable]] call site binds by name.
            bound = [NSMutableArray new];
            for (const PipelineOptionSpec::Hook &hook : spec.hooks) {
                id<MTLFunction> hooked = [fnLibrary newFunctionWithName:
                    checkedNSString(hook.name, "stitch hook")];
                if (!hooked)
                    throw MetalBackendError(
                        "stitched library produced no " +
                        std::string(hook.name));
                [bound addObject:hooked];
            }
        }
#if RICHENGINE_MTL4_AVAILABLE
        // The Metal 4 compiler carries the options and the private-function
        // links in one descriptor.
        if (bound || spec.persistent || spec.reliefNone ||
            spec.forwardProgress) {
            if (!compiler4) {
                NSError *error = nil;
                compiler4 =
                    [device newCompilerWithDescriptor:
                                [MTL4CompilerDescriptor new]
                                                error:&error];
                if (!compiler4)
                    throw MetalBackendError(
                        "unable to create a Metal 4 compiler: " +
                        errorDescription(error));
            }
            MTL4LibraryFunctionDescriptor *entry =
                [MTL4LibraryFunctionDescriptor new];
            entry.name = function.name;
            entry.library = library;
            MTL4SpecializedFunctionDescriptor *specialized =
                [MTL4SpecializedFunctionDescriptor new];
            specialized.functionDescriptor = entry;
            specialized.constantValues = functionConstantValues();
            MTL4ComputePipelineDescriptor *descriptor =
                [MTL4ComputePipelineDescriptor new];
            descriptor.computeFunctionDescriptor = specialized;
            descriptor.supportIndirectCommandBuffers =
                MTL4IndirectCommandBufferSupportStateEnabled;
            descriptor.optimizeForPersistentKernel = spec.persistent;
            descriptor.forwardProgressUsage =
                static_cast<MTLForwardProgressUsage>(spec.forwardProgress);
            descriptor.contentionRelief =
                spec.reliefNone ? MTLContentionReliefNone
                                : MTLContentionReliefAutomatic;
            if (bound) {
                MTL4StaticLinkingDescriptor *linking =
                    [MTL4StaticLinkingDescriptor new];
                NSMutableArray<MTL4LibraryFunctionDescriptor *> *hooks4 =
                    [NSMutableArray new];
                for (id<MTLFunction> hooked in bound) {
                    MTL4LibraryFunctionDescriptor *hookDesc =
                        [MTL4LibraryFunctionDescriptor new];
                    hookDesc.name = hooked.name;
                    hookDesc.library = fnLibrary;
                    [hooks4 addObject:hookDesc];
                }
                linking.privateFunctionDescriptors = hooks4;
                descriptor.staticLinkingDescriptor = linking;
            }
            NSError *error = nil;
            id<MTLComputePipelineState> result =
                [compiler4 newComputePipelineStateWithDescriptor:descriptor
                                             compilerTaskOptions:nil
                                                           error:&error];
            if (!result)
                throw MetalBackendError(
                    "unable to create Metal 4 pipeline " + std::string(name) +
                    ": " + errorDescription(error));
            sampleDeviceMemory();
            return result;
        }
#else
        if (spec.persistent || spec.reliefNone || spec.forwardProgress)
            throw MetalBackendError(
                "pipeline options need Metal 4: " + std::string(name));
#endif
        NSError *error = nil;
        id<MTLComputePipelineState> result;
        if (bound) {
            MTLComputePipelineDescriptor *descriptor =
                [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            MTLLinkedFunctions *linked = [MTLLinkedFunctions new];
            linked.privateFunctions = bound;
            descriptor.linkedFunctions = linked;
            result =
                [device newComputePipelineStateWithDescriptor:descriptor
                                                      options:0
                                                   reflection:nil
                                                        error:&error];
        } else {
            result = [device newComputePipelineStateWithFunction:function
                                                           error:&error];
        }
        if (!result)
            throw MetalBackendError(
                "unable to create Metal pipeline " + std::string(name) +
                ": " + errorDescription(error));
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

// The Metal 4 encoder is on by default (RICHENGINE_MTL4_OFF disables) —
    // measured never slower than the Metal 3 path it mirrors.
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

// The barrier each command of a run needs, from the byte ranges it binds:
// the same frontier rule commit4 applies to direct dispatches, resolved once
// here because a span's bindings are fixed for its lifetime. A command whose
// ranges are disjoint from every range since the last barrier encodes
// without one, so independent branches of the run overlap.
std::vector<uint8_t>
MetalBackend::Impl::spanBarrierPlan(std::span<const ComputeDispatch> run) {
    struct BoundRange {
        id<MTLBuffer> buffer;
        uint64_t begin;
        uint64_t end;
    };
    std::vector<uint8_t> needsBarrier(run.size(), 0);
    std::vector<BoundRange> frontier;
    std::vector<BoundRange> ranges;
    for (size_t index = 0; index < run.size(); ++index) {
        ranges.clear();
        for (const BufferBinding &binding : run[index].buffers) {
            const MetalBuffer::Impl &view = *binding.buffer.impl_;
            ranges.push_back({view.allocation->buffer, view.offsetBytes,
                              view.offsetBytes + view.lengthBytes});
        }
        bool dependent = false;
        for (const BoundRange &earlier : frontier) {
            for (const BoundRange &range : ranges) {
                if (earlier.buffer == range.buffer &&
                    earlier.begin < range.end && range.begin < earlier.end) {
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
        frontier.insert(frontier.end(), ranges.begin(), ranges.end());
    }
    return needsBarrier;
}

// Bakes one validated run into an indirect command buffer. The commands
// carry their pipeline and buffer objects directly (nothing is
// inherited).
    std::shared_ptr<MetalBackend::Impl::BakedSpan>
    MetalBackend::Impl::bakeSpan(std::span<const ComputeDispatch> run) {
        auto span = std::make_shared<BakedSpan>();
        span->dispatches.reserve(run.size());
        for (const ComputeDispatch &dispatch : run) {
            for (const BufferBinding &binding : dispatch.buffers)
                span->maxBindCount =
                    std::max(span->maxBindCount, binding.index + 1);
            for (const BytesBinding &binding : dispatch.bytes) {
                span->maxBindCount =
                    std::max(span->maxBindCount, binding.index + 1);
                span->paramsBytes += (binding.sizeBytes + 255) & ~uint64_t{255};
            }
        }
        // Snapshot pass: record what each command binds, so every bank can
        // be written from the snapshot alone.
        uint64_t cursor = 0;
        for (size_t i = 0; i < run.size(); ++i) {
            const ComputeDispatch &dispatch = run[i];
            BakedSpan::DispatchSnapshot snapshot;
            snapshot.pipelineName = dispatch.pipelineName;
            snapshot.threadgroups = dispatch.threadgroups;
            snapshot.threadsPerThreadgroup = dispatch.threadsPerThreadgroup;
            snapshot.patchableBytes = dispatch.patchableBytes;
            for (const BufferBinding &binding : dispatch.buffers) {
                const MetalBuffer::Impl &view = *binding.buffer.impl_;
                snapshot.bufferIndices.push_back(binding.index);
                snapshot.buffers.push_back(
                    {view.allocation, view.allocation->buffer,
                     view.offsetBytes, view.lengthBytes});
            }
            for (const BytesBinding &binding : dispatch.bytes) {
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
            span->dispatches.push_back(std::move(snapshot));
        }
        // A barrier only where a later command's bytes overlap an earlier
        // one's since the last barrier (RICHENGINE_SPAN_BARRIERS keeps the
        // default serialize-everything bake otherwise).
        span->barrierPlan =
            tuning().spanBarriers ? spanBarrierPlan(run)
                                  : std::vector<uint8_t>(run.size(), 1);
        encodeBank(*span, 0, run);
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

// Builds one bank: its parameter arena, staged from the run's payloads, and
// the indirect command buffer whose commands bind that arena's address. The
// run's payloads are the ones the snapshot was validated against (identical
// for non-patchable bindings by snapshotMatches, current for patchable ones),
// so a later bank stages the same bytes the bake did.
void MetalBackend::Impl::encodeBank(const BakedSpan &span, uint32_t bank,
                                    std::span<const ComputeDispatch> run) {
    if (span.icb[bank]) return;
    if (span.paramsBytes && !span.paramsAllocation[bank]) {
        auto allocation = std::make_shared<MetalAllocation>();
        allocation->accounting = accounting;
        allocation->length = span.paramsBytes;
        allocation->storage = BufferStorage::Shared;
        allocation->label = @"baked-span-params";
        allocation->residency = residency;
        allocation->attach(
            newBuffer(span.paramsBytes, BufferStorage::Shared,
                      allocation->label));
        span.paramsAllocation[bank] = std::move(allocation);
    }
    id<MTLBuffer> params = span.paramsAllocation[bank]
                               ? span.paramsAllocation[bank]->buffer
                               : nil;
    if (params) {
        auto *contents = static_cast<uint8_t *>(params.contents);
        for (size_t i = 0; i < run.size() && i < span.dispatches.size(); ++i) {
            const BakedSpan::DispatchSnapshot &snapshot = span.dispatches[i];
            for (size_t binding = 0; binding < snapshot.bytes.size();
                 ++binding)
                std::memcpy(contents + snapshot.bytes[binding].paramsOffset,
                            run[i].bytes[binding].data,
                            snapshot.bytes[binding].sizeBytes);
        }
    }
    MTLIndirectCommandBufferDescriptor *descriptor =
        [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritPipelineState = NO;
    descriptor.inheritBuffers = NO;
    descriptor.maxKernelBufferBindCount = span.maxBindCount;
    id<MTLIndirectCommandBuffer> icb = [device
        newIndirectCommandBufferWithDescriptor:descriptor
                               maxCommandCount:span.dispatches.size()
                                       options:0];
    if (!icb)
        throw MetalBackendError(
            "unable to create a Metal indirect command buffer");
    icb.label = @"baked-span";
    span.icb[bank] = icb;
#if RICHENGINE_MTL4_AVAILABLE
    // executeCommandsInBuffer requires the buffer itself resident on the
    // Metal 4 path; the Metal 3 encoder declares its usage.
    if (mtl4Enabled()) residency->add(icb);
#endif
    for (size_t i = 0; i < span.dispatches.size(); ++i) {
        const BakedSpan::DispatchSnapshot &snapshot = span.dispatches[i];
        id<MTLIndirectComputeCommand> command =
            [icb indirectComputeCommandAtIndex:i];
        [command setComputePipelineState:
                     indirectPipeline(snapshot.pipelineName)];
        for (size_t bound = 0; bound < snapshot.buffers.size(); ++bound) {
            const BakedSpan::BufferSnapshot &saved = snapshot.buffers[bound];
            [command setKernelBuffer:saved.object
                              offset:saved.offsetBytes
                             atIndex:snapshot.bufferIndices[bound]];
        }
        for (const BakedSpan::BytesSnapshot &saved : snapshot.bytes)
            [command setKernelBuffer:params
                              offset:saved.paramsOffset
                             atIndex:saved.index];
        if (i && span.barrierPlan[i]) [command setBarrier];
        [command concurrentDispatchThreadgroups:
            MTLSizeMake(snapshot.threadgroups.x, snapshot.threadgroups.y,
                        snapshot.threadgroups.z)
            threadsPerThreadgroup:
                MTLSizeMake(snapshot.threadsPerThreadgroup.x,
                            snapshot.threadsPerThreadgroup.y,
                            snapshot.threadsPerThreadgroup.z)];
    }
}

// Rewrites a span's staged parameters from the run's current patchable
    // payloads: the indirect commands bind the parameter arena by address,
    // so the same commands replay with the new parameters. The arena is the
    // submission's parity bank: the only command that can still be reading
    // an arena is the one two submissions back, which used the other bank.
    void MetalBackend::Impl::patchSpanParams(const BakedSpan &span,
                                std::span<const PreparedDispatch> run,
                                uint32_t bank) {
        const auto &allocation = span.paramsAllocation[bank & 1];
        if (!allocation || !allocation->buffer) return;
        auto *contents = static_cast<uint8_t *>(allocation->buffer.contents);
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

// The escape hatch: GPU-encoded spans stay opt-in and internal
// (RICHENGINE_GPU_SPANS only — never a tuning-table default) until the
// MTL_SHADER_VALIDATION anomaly in ISSUES.md is resolved.
    bool MetalBackend::Impl::gpuSpansEnabled() noexcept {
        if (tuning().gpuSpans && getenv("MTL_SHADER_VALIDATION")) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr,
                        "richengine: GPU-encoded spans disabled under "
                        "MTL_SHADER_VALIDATION — an unresolved encode→"
                        "barrier→execute ordering hazard can corrupt an "
                        "expert tile (ISSUES.md).\n");
            }
            return false;
        }
        return tuning().gpuSpans;
    }

// Identity of a run against a gpu span's snapshot; identical to
    // snapshotMatches except that payload contents never take part — every
    // bytes binding is restaged into the span's arena on replay, so a
    // payload change never re-bakes.
    bool
    MetalBackend::Impl::gpuSpanMatches(const GpuSpan &span,
                   std::span<const ComputeDispatch> run) noexcept {
        if (span.dispatches.size() != run.size()) return false;
        for (size_t i = 0; i < run.size(); ++i) {
            const ComputeDispatch &dispatch = run[i];
            const BakedSpan::DispatchSnapshot &snapshot = span.dispatches[i];
            if (snapshot.pipelineName != dispatch.pipelineName ||
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
                    saved.sizeBytes != bound.sizeBytes)
                    return false;
            }
        }
        return true;
    }

// Builds a gpu span for the run. The ICB's commands are written on device
    // by gpu_encode from the descriptor table staged here: every binding —
    // buffer or staged parameter payload — rides its gpu address, so the
    // table stays valid while the run's snapshot matches.
    std::shared_ptr<MetalBackend::Impl::GpuSpan>
    MetalBackend::Impl::bakeGpuSpan(std::span<const ComputeDispatch> run) {
        auto span = std::make_shared<GpuSpan>();
        span->dispatches.reserve(run.size());
        // The run's distinct pipelines, in first-use order: a command names
        // its table index.
        std::unordered_map<std::string_view, uint32_t> pipelineIndices;
        uint64_t paramsBytes = 0;
        uint32_t maxBindCount = 0;
        for (const ComputeDispatch &dispatch : run) {
            const auto [_, fresh] = pipelineIndices.try_emplace(
                dispatch.pipelineName,
                static_cast<uint32_t>(pipelineIndices.size()));
            (void)fresh;
            for (const BufferBinding &binding : dispatch.buffers)
                maxBindCount = std::max(maxBindCount, binding.index + 1);
            for (const BytesBinding &binding : dispatch.bytes) {
                maxBindCount = std::max(maxBindCount, binding.index + 1);
                paramsBytes += (binding.sizeBytes + 255) & ~uint64_t{255};
            }
        }
        if (pipelineIndices.size() > kGpuSpanPipelines) return nullptr;
        const auto allocate = [&](uint64_t bytes, NSString *label) {
            auto allocation = std::make_shared<MetalAllocation>();
            allocation->accounting = accounting;
            allocation->length = bytes;
            allocation->storage = BufferStorage::Shared;
            allocation->label = label;
            allocation->residency = residency;
            allocation->attach(newBuffer(bytes, BufferStorage::Shared, label));
            return allocation;
        };
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
        span->icb.label = @"gpu-span";
#if RICHENGINE_MTL4_AVAILABLE
        if (mtl4Enabled()) residency->add(span->icb);
#endif
        // The encoder kernel's argument buffer: the ICB at index 0, the
        // run's indirect-capable pipelines at 1..kGpuSpanPipelines.
        id<MTLFunction> encodeFunction =
            [library newFunctionWithName:@"gpu_encode"
                          constantValues:functionConstantValues()
                                   error:nil];
        if (!encodeFunction)
            throw MetalBackendError("missing Metal function: gpu_encode");
        id<MTLArgumentEncoder> argEncoder =
            [encodeFunction newArgumentEncoderWithBufferIndex:0];
        span->argsAllocation = allocate(argEncoder.encodedLength,
                                        @"gpu-span-args");
        [argEncoder setArgumentBuffer:span->argsAllocation->buffer offset:0];
        [argEncoder setIndirectCommandBuffer:span->icb atIndex:0];
        for (const auto &[name, index] : pipelineIndices)
            [argEncoder setComputePipelineState:indirectPipeline(name)
                                        atIndex:1 + index];
        span->descAllocation = allocate(
            kGpuSpanDescHeader + run.size() * sizeof(GpuDispatchDesc),
            @"gpu-span-descs");
        if (paramsBytes)
            span->paramsAllocation =
                allocate(paramsBytes, @"gpu-span-params");
        auto *header = static_cast<uint8_t *>(
            span->descAllocation->buffer.contents);
        std::memset(header, 0,
                    kGpuSpanDescHeader +
                        run.size() * sizeof(GpuDispatchDesc));
        const uint32_t count = static_cast<uint32_t>(run.size());
        std::memcpy(header, &count, sizeof(uint32_t));
        auto *descs = reinterpret_cast<GpuDispatchDesc *>(
            header + kGpuSpanDescHeader);
        uint8_t *params = span->paramsAllocation
            ? static_cast<uint8_t *>(span->paramsAllocation->buffer.contents)
            : nullptr;
        const uint64_t paramsAddress =
            span->paramsAllocation ? span->paramsAllocation->buffer.gpuAddress : 0;
        uint64_t cursor = 0;
        for (size_t i = 0; i < run.size(); ++i) {
            const ComputeDispatch &dispatch = run[i];
            GpuDispatchDesc &desc = descs[i];
            desc.pipeline = pipelineIndices.at(dispatch.pipelineName);
            desc.groups_x = static_cast<uint32_t>(dispatch.threadgroups.x);
            desc.groups_y = static_cast<uint32_t>(dispatch.threadgroups.y);
            desc.groups_z = static_cast<uint32_t>(dispatch.threadgroups.z);
            desc.threads_x =
                static_cast<uint32_t>(dispatch.threadsPerThreadgroup.x);
            desc.threads_y =
                static_cast<uint32_t>(dispatch.threadsPerThreadgroup.y);
            desc.threads_z =
                static_cast<uint32_t>(dispatch.threadsPerThreadgroup.z);
            BakedSpan::DispatchSnapshot snapshot;
            snapshot.pipelineName = dispatch.pipelineName;
            snapshot.threadgroups = dispatch.threadgroups;
            snapshot.threadsPerThreadgroup = dispatch.threadsPerThreadgroup;
            snapshot.patchableBytes = dispatch.patchableBytes;
            uint32_t bindings = 0;
            for (const BufferBinding &binding : dispatch.buffers) {
                if (bindings == kGpuSpanBindings || binding.index > 0xFF)
                    return nullptr;
                const MetalBuffer::Impl &view = *binding.buffer.impl_;
                desc.buffers[bindings] =
                    view.allocation->buffer.gpuAddress + view.offsetBytes;
                desc.binding_index[bindings] =
                    static_cast<uint8_t>(binding.index);
                ++bindings;
                snapshot.bufferIndices.push_back(binding.index);
                snapshot.buffers.push_back(
                    {view.allocation, view.allocation->buffer,
                     view.offsetBytes, view.lengthBytes});
            }
            for (const BytesBinding &binding : dispatch.bytes) {
                if (bindings == kGpuSpanBindings || binding.index > 0xFF)
                    return nullptr;
                std::memcpy(params + cursor, binding.data,
                            binding.sizeBytes);
                desc.buffers[bindings] = paramsAddress + cursor;
                desc.binding_index[bindings] =
                    static_cast<uint8_t>(binding.index);
                ++bindings;
                snapshot.bytes.push_back({binding.index, {}, cursor,
                                          binding.sizeBytes});
                cursor += (binding.sizeBytes + 255) & ~uint64_t{255};
            }
            desc.bindings = bindings;
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
            if (span->paramsAllocation)
                span->usedBuffers.push_back(span->paramsAllocation->buffer);
        }
        sampleDeviceMemory();
        return span;
    }

// Finds a gpu span matching this run or bakes a new one; nullptr means the
    // run encodes directly.
    std::shared_ptr<const MetalBackend::Impl::GpuSpan>
    MetalBackend::Impl::resolveGpuSpan(size_t begin,
                   std::span<const ComputeDispatch> run) {
        if (gpuSpans.size() > 256) gpuSpans.clear();
        std::string key = std::to_string(begin) + '#' +
                          std::to_string(run.size()) + '#' +
                          run.front().pipelineName;
        std::vector<std::shared_ptr<GpuSpan>> &slots = gpuSpans[key];
        for (const auto &candidate : slots) {
            ++matchEpoch;
            if (!gpuSpanMatches(*candidate, run)) continue;
            candidate->touched = ++bakedClock;
            return candidate;
        }
        try {
            auto baked = bakeGpuSpan(run);
            if (!baked) return nullptr;
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
            return nullptr;
        }
    }

// Restages every bytes payload of the run into the span's parameter arena:
    // the descriptors bind it by address, so only contents change.
    void MetalBackend::Impl::stageGpuSpanParams(const GpuSpan &span,
                            std::span<const PreparedDispatch> run) {
        if (!span.paramsAllocation || !span.paramsAllocation->buffer) return;
        auto *contents =
            static_cast<uint8_t *>(span.paramsAllocation->buffer.contents);
        for (size_t i = 0; i < run.size(); ++i) {
            const BakedSpan::DispatchSnapshot &snapshot = span.dispatches[i];
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
            const auto checkIndirect = [](const MetalBuffer &view,
                                          uint64_t minimum,
                                          const char *what) {
                if (!view.impl_ || !view.impl_->allocation->buffer) {
                    throw MetalBackendError(
                        std::string("compute dispatch binds an empty ") +
                        what);
                }
                if (view.impl_->offsetBytes % 4 ||
                    view.impl_->lengthBytes < minimum) {
                    throw MetalBackendError(
                        std::string("compute dispatch ") + what +
                        " is misaligned or too small");
                }
            };
            if (dispatch.indirectGrid) {
                if (dispatch.bakeable || dispatch.gpuSpan)
                    throw MetalBackendError(
                        "indirect dispatch cannot join a span");
                checkIndirect(dispatch.indirectGrid, 12, "indirect grid");
            }
            if (dispatch.indirectRange) {
                if (!dispatch.bakeable && !dispatch.gpuSpan)
                    throw MetalBackendError(
                        "indirect span range on a dispatch outside a span");
                checkIndirect(dispatch.indirectRange, 8,
                              "indirect span range");
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
        for (const ComputeDispatch &dispatch : dispatches) {
            if (dispatch.indirectGrid)
                command.retainedAllocations.push_back(
                    dispatch.indirectGrid.impl_->allocation);
            if (dispatch.indirectRange)
                command.retainedAllocations.push_back(
                    dispatch.indirectRange.impl_->allocation);
        }

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
                    for (const auto &arena : span->paramsAllocation)
                        if (arena)
                            command.retainedAllocations.push_back(arena);
                    command.spanRefs.push_back(std::move(span));
                }
                begin = end;
            }
        }

        // Maximal runs of gpu-span dispatches replay through a GPU-encoded
        // indirect command buffer; a run that cannot bake one encodes
        // directly like any other.
        if (gpuSpansEnabled()) {
            for (size_t begin = 0; begin < dispatches.size();) {
                if (!dispatches[begin].gpuSpan) {
                    ++begin;
                    continue;
                }
                size_t end = begin + 1;
                while (end < dispatches.size() && dispatches[end].gpuSpan)
                    ++end;
                const std::span<const ComputeDispatch> run =
                    dispatches.subspan(begin, end - begin);
                if (std::shared_ptr<const GpuSpan> span =
                        resolveGpuSpan(begin, run)) {
                    command.dispatches[begin].gpuSpan = span.get();
                    if (span->argsAllocation)
                        command.retainedAllocations.push_back(
                            span->argsAllocation);
                    if (span->descAllocation)
                        command.retainedAllocations.push_back(
                            span->descAllocation);
                    if (span->paramsAllocation)
                        command.retainedAllocations.push_back(
                            span->paramsAllocation);
                    command.gpuSpanRefs.push_back(std::move(span));
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
            const auto viewIdentity =
                [&](const CachedCommand::BufferSnapshot &saved,
                    const MetalBuffer &view) {
                    if (!view != !saved.object) return false;
                    if (!view) return true;
                    const MetalBuffer::Impl &impl = *view.impl_;
                    return impl.offsetBytes == saved.offsetBytes &&
                           impl.lengthBytes == saved.lengthBytes &&
                           bindingIdentity(saved.allocation, saved.object,
                                           impl);
                };
            if (snapshot.pipelineName != dispatch.pipelineName ||
                snapshot.bakeable != dispatch.bakeable ||
                snapshot.gpuSpan != dispatch.gpuSpan ||
                snapshot.patchableBytes != dispatch.patchableBytes ||
                !viewIdentity(snapshot.indirectGrid, dispatch.indirectGrid) ||
                !viewIdentity(snapshot.indirectRange,
                              dispatch.indirectRange) ||
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
            snapshot.gpuSpan = dispatch.gpuSpan;
            const auto saveView = [](const MetalBuffer &view,
                                     CachedCommand::BufferSnapshot &saved) {
                if (!view) return;
                const MetalBuffer::Impl &impl = *view.impl_;
                saved = {impl.allocation, impl.allocation->buffer,
                         impl.offsetBytes, impl.lengthBytes};
            };
            saveView(dispatch.indirectGrid, snapshot.indirectGrid);
            saveView(dispatch.indirectRange, snapshot.indirectRange);
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
        cached->gpuSpanRefs = command.gpuSpanRefs;
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
            command.gpuSpanRefs = candidate->gpuSpanRefs;
            for (size_t i = 0; i < dispatches.size(); ++i)
                command.dispatches[i].source = &dispatches[i];
            command.retainedAllocations.reserve(
                candidate->boundBuffers + 2 * candidate->spanRefs.size() +
                3 * candidate->gpuSpanRefs.size());
            retainBound(dispatches, command.retainedAllocations);
            for (const auto &span : command.spanRefs)
                for (const auto &arena : span->paramsAllocation)
                    if (arena) command.retainedAllocations.push_back(arena);
            for (const auto &span : command.gpuSpanRefs) {
                command.retainedAllocations.push_back(span->argsAllocation);
                command.retainedAllocations.push_back(span->descAllocation);
                if (span->paramsAllocation)
                    command.retainedAllocations.push_back(
                        span->paramsAllocation);
            }
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
        // flight (the submit-ahead ring). Baked spans replay then: the bank
        // they use is built on first use, so the arena this submission
        // patches is never one an in-flight replay is reading. A command
        // with event steps encodes directly: a replayed span would hide the
        // dispatches its steps order against the other agent (EventStep).
        const bool overlapping = asyncState->hasActiveSubmission();
        const bool replaySpans =
            events.empty() && (!tuning().spanBankOff || !overlapping);
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        ticketState->retainedAllocations = std::move(retained);
        ticketState->sequence =
            asyncState->beginSubmission(dispatches.size(), 2);
        // A baked span replays the bank the in-flight command is not using,
        // and reuses the last bank when nothing is in flight, so bank 1 is
        // built the first time a submission actually overlaps.
        const uint32_t spanBank =
            tuning().spanBankOff ? 0u
                                 : (overlapping ? lastSpanBank ^ 1u
                                                : lastSpanBank);
        lastSpanBank = spanBank;
        // GPU-encoded spans restage their one arena per submission, so they
        // keep the GPU-idle requirement.
        const bool replayGpuSpans = replaySpans && !overlapping;

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
                    // The run's dispatches are contiguous in the submitting
                    // command, and `source` of the span's first one is its
                    // head, so the bank builds from the same objects the
                    // snapshot was validated against.
                    const size_t spanLength = item.span->dispatches.size();
                    if (!item.span->icb[spanBank]) {
                        encodeBank(*item.span, spanBank,
                                   {item.source, spanLength});
                        if (const auto &arena =
                                item.span->paramsAllocation[spanBank])
                            ticketState->retainedAllocations.push_back(arena);
                    }
                    patchSpanParams(
                        *item.span, dispatches.subspan(index, spanLength),
                        spanBank);
                    id<MTLIndirectCommandBuffer> icb = item.span->icb[spanBank];
                    [encoder useResource:icb
                                   usage:MTLResourceUsageRead];
                    if (const auto &arena =
                            item.span->paramsAllocation[spanBank])
                        [encoder useResource:arena->buffer
                                       usage:MTLResourceUsageRead];
                    for (__weak id<MTLBuffer> buffer :
                         item.span->usedBuffers) {
                        if (id<MTLBuffer> resource = buffer)
                            [encoder useResource:resource
                                           usage:MTLResourceUsageRead |
                                                 MTLResourceUsageWrite];
                    }
                    if (item.source->indirectRange) {
                        const MetalBuffer::Impl &range =
                            *item.source->indirectRange.impl_;
                        [encoder useResource:range.allocation->buffer
                                       usage:MTLResourceUsageRead];
                        [encoder executeCommandsInBuffer:icb
                                          indirectBuffer:range.allocation->buffer
                                     indirectBufferOffset:range.offsetBytes];
                    } else {
                        [encoder executeCommandsInBuffer:icb
                                               withRange:
                            NSMakeRange(0, item.span->dispatches.size())];
                    }
                    index += item.span->dispatches.size() - 1;
                    closeDispatch(std::string(item.source->pipelineName));
                    continue;
                }
                // A GPU-encoded span replays the same way once gpu_encode
                // has written its commands on device: the descriptors it
                // reads were staged at bake time, and every bytes payload
                // is restaged into the span's parameter arena now.
                if (item.gpuSpan && replayGpuSpans && !opTimings &&
                    index + item.gpuSpan->dispatches.size() <=
                        dispatches.size()) {
                    const GpuSpan &span = *item.gpuSpan;
                    stageGpuSpanParams(
                        span, dispatches.subspan(
                                  index, span.dispatches.size()));
                    const NSUInteger count = span.dispatches.size();
                    // gpu_encode writes commands that are a pure function
                    // of the span's descriptor table; the match that found
                    // this span keeps the table static, so only the first
                    // replay encodes — later ones replay the ICB as it
                    // stands, with only the params arena restaged above.
                    if (!span.encoded) {
                        [encoder setComputePipelineState:pipeline("gpu_encode")];
                        [encoder setBuffer:span.argsAllocation->buffer
                                    offset:0
                                   atIndex:0];
                        [encoder setBuffer:span.descAllocation->buffer
                                    offset:kGpuSpanDescHeader
                                   atIndex:1];
                        [encoder setBuffer:span.descAllocation->buffer
                                    offset:0
                                   atIndex:2];
                        [encoder useResource:span.icb
                                       usage:MTLResourceUsageWrite];
                        [encoder
                            dispatchThreadgroups:MTLSizeMake(
                                (count + 255) / 256, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        span.encoded = true;
                    }
                    [encoder useResource:span.icb
                                   usage:MTLResourceUsageRead];
                    for (__weak id<MTLBuffer> buffer : span.usedBuffers) {
                        if (id<MTLBuffer> resource = buffer)
                            [encoder useResource:resource
                                           usage:MTLResourceUsageRead |
                                                 MTLResourceUsageWrite];
                    }
                    if (item.source->indirectRange) {
                        const MetalBuffer::Impl &range =
                            *item.source->indirectRange.impl_;
                        [encoder useResource:range.allocation->buffer
                                       usage:MTLResourceUsageRead];
                        [encoder executeCommandsInBuffer:span.icb
                                          indirectBuffer:range.allocation->buffer
                                     indirectBufferOffset:range.offsetBytes];
                    } else {
                        [encoder executeCommandsInBuffer:span.icb
                                               withRange:
                            NSMakeRange(0, count)];
                    }
                    index += span.dispatches.size() - 1;
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
                if (dispatch.indirectGrid) {
                    const MetalBuffer::Impl &grid =
                        *dispatch.indirectGrid.impl_;
                    [encoder dispatchThreadgroupsWithIndirectBuffer:
                                 grid.allocation->buffer
                                               indirectBufferOffset:
                                 grid.offsetBytes
                                          threadsPerThreadgroup:
                                 item.threads];
                } else {
                    [encoder dispatchThreadgroups:item.groups
                             threadsPerThreadgroup:item.threads];
                }
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
    // completed handler. Event steps map to queue-level signalEvent /
    // waitForEvent between the split's commits, where commit() encoded
    // them into command buffers instead.
    CommandTicket MetalBackend::Impl::commit4(std::span<const PreparedDispatch> dispatches,
                          std::span<const EventStep> events,
                          std::vector<std::shared_ptr<MetalAllocation>> retained,
                          CommandCompletion completion) {
        // The submit-ahead ring may commit a second command while this one
        // runs. Baked spans replay then: the bank they use is built on first
        // use, so the arena this submission patches is never one an in-flight
        // replay is reading (RICHENGINE_SPAN_BANK_OFF restores the GPU-idle
        // requirement). A command with event steps encodes directly, as
        // commit() does: a replayed span skips the encodeSteps calls of the
        // indices it covers, and its steps would never order against the
        // other agent (EventStep).
        const bool overlapping = asyncState->hasActiveSubmission();
        const bool replaySpans =
            events.empty() && (!tuning().spanBankOff || !overlapping);
        auto ticketState = std::make_shared<CommandTicket::State>();
        ticketState->backend = asyncState;
        ticketState->completion = std::move(completion);
        ticketState->sequence =
            asyncState->beginSubmission(dispatches.size(), 2);
        // Submission parity selects the staging and allocator bank the
        // in-flight command does not use; a baked span's arena and ICB take
        // the bank the in-flight command is not reading, or the last bank
        // when nothing is in flight.
        const uint32_t bank = ticketState->sequence & 1;
        const uint32_t spanBank =
            tuning().spanBankOff ? 0u
                                 : (overlapping ? lastSpanBank ^ 1u
                                                : lastSpanBank);
        lastSpanBank = spanBank;
        // GPU-encoded spans restage their one arena per submission, so they
        // keep the GPU-idle requirement.
        const bool replayGpuSpans = replaySpans && !overlapping;
        // RICHENGINE_OP_TIMINGS on Metal 4 brackets each encoded unit with a
        // counter-heap timestamp pair, resolved in the final commit's
        // feedback — the per-op probe commit() gets by committing every
        // dispatch in its own buffer, except the buffer never splits here:
        // a timestamp still orders against the work before it, so the
        // serialized attribution is the same.
        const bool opTimings = tuning().opTimings;
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
            ticketState->wallStart = wallStart;
            // Event steps commit leading buffers during the encode below;
            // hold the residency set from the first of them, not the last.
            residency->use();
            // Commit feedback is per commit and covers only that commit's
            // buffers, so a command split at event steps shares one record
            // across its pieces: the final commit's finishCommand4 then
            // reports the whole command the way finishCommand aggregates
            // its leading buffers. The feedbacks run on the queue's single
            // serial feedback queue in commit order, so the record is
            // complete before the last one reads it.
            std::shared_ptr<SplitFeedback4> split4;
            if (!events.empty()) {
                split4 = std::make_shared<SplitFeedback4>();
                ticketState->split4 = split4;
            }
            // `command` exists only while work is being encoded toward the
            // next commit: an event boundary ends it, and the next dispatch
            // starts a fresh buffer on the same allocator bank.
            id<MTL4CommandBuffer> command = nil;
            id<MTL4ComputeCommandEncoder> encoder = nil;
            const auto newCommand = [&]() -> id<MTL4CommandBuffer> {
                id<MTL4CommandBuffer> next = [device newCommandBuffer];
                if (!next)
                    failBeforeCommit(
                        "unable to create Metal command buffer");
                [next beginCommandBufferWithAllocator:allocator4[bank]];
                return next;
            };
            uint8_t *stagingContents =
                staging ? static_cast<uint8_t *>(staging->buffer.contents)
                        : nullptr;
            const MTLGPUAddress stagingAddress =
                staging ? staging->buffer.gpuAddress : 0;
            uint64_t stagingCursor = 0;
            // Two heap entries per encoded unit; a span replay is one unit
            // however many dispatches it covers. The heap is banked on
            // submission parity like the allocator: its feedback resolves
            // it before the bank's next reuse.
            id<MTL4CounterHeap> timestamps4 = nil;
            NSUInteger timingIndex = 0;
            std::vector<std::string> timingNames;
            if (opTimings && !dispatches.empty()) {
                const NSUInteger needed = 2 * dispatches.size();
                if (!counterHeap4[bank] ||
                    counterHeap4Count[bank] < needed) {
                    MTL4CounterHeapDescriptor *descriptor =
                        [MTL4CounterHeapDescriptor new];
                    descriptor.type = MTL4CounterHeapTypeTimestamp;
                    descriptor.count = needed;
                    NSError *error = nil;
                    counterHeap4[bank] =
                        [device newCounterHeapWithDescriptor:descriptor
                                                       error:&error];
                    counterHeap4Count[bank] =
                        counterHeap4[bank] ? needed : 0;
                    if (!counterHeap4[bank])
                        failBeforeCommit(std::string(
                            "unable to create MTL4 counter heap: ") +
                            error.localizedDescription.UTF8String);
                }
                timestamps4 = counterHeap4[bank];
            }
            const auto timestampPair = [&](std::string_view name) {
                if (!opTimings) return;
                timingNames.emplace_back(name);
                [encoder writeTimestampWithGranularity:
                             MTL4TimestampGranularityPrecise
                                              intoHeap:timestamps4
                                               atIndex:timingIndex * 2];
            };
            const auto timestampPairEnd = [&] {
                if (!opTimings) return;
                [encoder writeTimestampWithGranularity:
                             MTL4TimestampGranularityPrecise
                                              intoHeap:timestamps4
                                               atIndex:timingIndex * 2 + 1];
                ++timingIndex;
            };
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
                    const bool replaysGpuSpan =
                        item.gpuSpan && replayGpuSpans &&
                        index + item.gpuSpan->dispatches.size() <=
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
                    } else if (replaysGpuSpan) {
                        ranges.push_back(
                            {item.gpuSpan->descAllocation->buffer, 0,
                             item.gpuSpan->descAllocation->buffer.length});
                        for (__weak id<MTLBuffer> weak :
                             item.gpuSpan->usedBuffers) {
                            if (id<MTLBuffer> buffer = weak) {
                                ranges.push_back(
                                    {buffer, 0, buffer.length});
                            }
                        }
                    }
                    if ((replaysSpan || replaysGpuSpan) &&
                        item.source->indirectRange) {
                        const MetalBuffer::Impl &range =
                            *item.source->indirectRange.impl_;
                        ranges.push_back({range.allocation->buffer,
                                          range.offsetBytes,
                                          range.offsetBytes +
                                              range.lengthBytes});
                    }
                    if (!replaysSpan && !replaysGpuSpan) {
                        for (const BufferBinding &binding :
                             item.source->buffers) {
                            const MetalBuffer::Impl &view =
                                *binding.buffer.impl_;
                            ranges.push_back({view.allocation->buffer,
                                              view.offsetBytes,
                                              view.offsetBytes +
                                                  view.lengthBytes});
                        }
                        for (const MetalBuffer &indirect :
                             {item.source->indirectGrid,
                              item.source->indirectRange}) {
                            if (!indirect) continue;
                            const MetalBuffer::Impl &view = *indirect.impl_;
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
                    else if (replaysGpuSpan)
                        index += item.gpuSpan->dispatches.size() - 1;
                }
            }
            const auto orderAfterPrevious = [&](size_t index) {
                if (!needsBarrier[index]) return;
                [encoder barrierAfterEncoderStages:MTLStageDispatch
                              beforeEncoderStages:MTLStageDispatch
                                 visibilityOptions:MTL4VisibilityOptionDevice];
            };
            // An execute's span range: literal unless the run's first
            // dispatch carries a GPU-written indirect range view. `icb` is
            // the span's bank: the arena the host patches is the one bound
            // here.
            const auto executeSpan = [&](id<MTLIndirectCommandBuffer> icb,
                                         size_t count, size_t at) {
                const MetalBuffer &range = dispatches[at].source->indirectRange;
                if (range) {
                    const MetalBuffer::Impl &view = *range.impl_;
                    [encoder executeCommandsInBuffer:icb
                                      indirectBuffer:
                        view.allocation->buffer.gpuAddress +
                            view.offsetBytes];
                } else {
                    [encoder executeCommandsInBuffer:icb
                                           withRange:NSMakeRange(0, count)];
                }
            };
            // Encodes the event steps that follow the first `encoded`
            // dispatches. Signals and waits are queue-level operations on
            // Metal 4 — signalEvent orders after the work committed so far,
            // waitForEvent before the work committed next — so a boundary
            // carrying any step first commits the in-progress buffer, the
            // split commit() makes only at signals: a queue wait taken while
            // the buffer was still open would wrongly hold the work already
            // encoded into it.
            auto step = events.begin();
            const auto encodeSteps = [&](size_t encoded) {
                if (step == events.end() || step->before != encoded) return;
                // Every signal of this boundary, gathered before the steps
                // run: a queue signalEvent after a failed buffer may never
                // execute — the same loss commit() guards against with the
                // completed handler of the buffer that encoded the signal —
                // so the commit's feedback delivers them from the CPU on
                // error. Signals at a later boundary always have encoded
                // work before them, so this boundary's commit is the last
                // that can cover them.
                std::vector<std::pair<SharedEvent, uint64_t>> covered;
                for (auto signal = step;
                     signal != events.end() && signal->before == encoded;
                     ++signal) {
                    if (signal->kind == EventStep::Kind::Signal)
                        covered.emplace_back(signal->event, signal->value);
                }
                if (command || !covered.empty()) {
                    if (!command) command = newCommand();
                    // A signal with nothing encoded still gets its empty
                    // carrier, as commit()'s signal-only buffer did: the
                    // failure fallback needs a commit to ride.
                    if (encoder) {
                        [encoder endEncoding];
                        encoder = nil;
                    }
                    [command endCommandBuffer];
                    MTL4CommitOptions *options = [MTL4CommitOptions new];
                    [options addFeedbackHandler:
                        ^(id<MTL4CommitFeedback> feedback) {
                            split4->record(feedback);
                            if (feedback.error)
                                for (const auto &[event, value] : covered)
                                    event.signal(value);
                        }];
                    id<MTL4CommandBuffer> committed = command;
                    [queue4 commit:&committed count:1 options:options];
                    command = nil;
                }
                for (; step != events.end() && step->before == encoded;
                     ++step) {
                    id<MTLSharedEvent> event =
                        (__bridge id<MTLSharedEvent>)
                            step->event.nativeHandle();
                    if (step->kind == EventStep::Kind::Wait) {
                        [queue4 waitForEvent:event value:step->value];
                    } else {
                        [queue4 signalEvent:event value:step->value];
                    }
                }
            };
            for (size_t index = 0; index < dispatches.size(); ++index) {
                encodeSteps(index);
                const PreparedDispatch &item = dispatches[index];
                if (!command) command = newCommand();
                if (!encoder) {
                    encoder = [command computeCommandEncoder];
                    if (!encoder) {
                        failBeforeCommit(
                            "unable to create Metal compute encoder");
                    }
                }
                // A baked span replays its whole run from the indirect
                // command buffer; the run's own dispatches are skipped.
                if (item.span && replaySpans &&
                    index + item.span->dispatches.size() <=
                        dispatches.size()) {
                    // The run's dispatches are contiguous in the submitting
                    // command, and `source` of the span's first one is its
                    // head, so the bank builds from the same objects the
                    // snapshot was validated against.
                    const size_t spanLength = item.span->dispatches.size();
                    if (!item.span->icb[spanBank]) {
                        encodeBank(*item.span, spanBank,
                                   {item.source, spanLength});
                        if (const auto &arena =
                                item.span->paramsAllocation[spanBank])
                            ticketState->retainedAllocations.push_back(arena);
                    }
                    patchSpanParams(
                        *item.span, dispatches.subspan(index, spanLength),
                        spanBank);
                    orderAfterPrevious(index);
                    timestampPair(item.source->pipelineName);
                    executeSpan(item.span->icb[spanBank], spanLength, index);
                    timestampPairEnd();
                    index += spanLength - 1;
                    continue;
                }
                // A GPU-encoded span replays like a baked one: gpu_encode
                // fills its indirect command buffer once (its commands are a
                // pure function of the span's static descriptor table), the
                // dispatch-stage barrier orders the write before the replay,
                // and every bytes payload restages into the span's arena.
                // GPU-encoded spans stay direct under opTimings, as in
                // commit(): a gpu_encode + replay pair would attribute the
                // whole run to the head pipeline.
                if (item.gpuSpan && replayGpuSpans && !opTimings &&
                    index + item.gpuSpan->dispatches.size() <=
                        dispatches.size()) {
                    const GpuSpan &span = *item.gpuSpan;
                    stageGpuSpanParams(
                        span, dispatches.subspan(
                                  index, span.dispatches.size()));
                    orderAfterPrevious(index);
                    const NSUInteger count = span.dispatches.size();
                    if (!span.encoded) {
                        [encoder setComputePipelineState:
                                     pipeline("gpu_encode")];
                        [argumentTable4 setAddress:
                            span.argsAllocation->buffer.gpuAddress
                                           atIndex:0];
                        [argumentTable4 setAddress:
                            span.descAllocation->buffer.gpuAddress +
                                kGpuSpanDescHeader
                                           atIndex:1];
                        [argumentTable4 setAddress:
                            span.descAllocation->buffer.gpuAddress
                                           atIndex:2];
                        [encoder setArgumentTable:argumentTable4];
                        [encoder dispatchThreadgroups:
                            MTLSizeMake((count + 255) / 256, 1, 1)
                             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [encoder barrierAfterEncoderStages:MTLStageDispatch
                                    beforeEncoderStages:MTLStageDispatch
                                       visibilityOptions:
                            MTL4VisibilityOptionDevice];
                        span.encoded = true;
                    }
                    executeSpan(span.icb, count, index);
                    index += count - 1;
                    continue;
                }
                const ComputeDispatch &dispatch = *item.source;
                orderAfterPrevious(index);
                timestampPair(dispatch.pipelineName);
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
                if (dispatch.indirectGrid) {
                    const MetalBuffer::Impl &grid =
                        *dispatch.indirectGrid.impl_;
                    [encoder dispatchThreadgroupsWithIndirectBuffer:
                        grid.allocation->buffer.gpuAddress + grid.offsetBytes
                                              threadsPerThreadgroup:
                        item.threads];
                } else {
                    [encoder dispatchThreadgroups:item.groups
                             threadsPerThreadgroup:item.threads];
                }
                timestampPairEnd();
            }
            encodeSteps(dispatches.size());
            if (encoder) [encoder endEncoding];
            // A command ending on an event boundary already committed its
            // last work; the ticket's feedback still needs a carrier, so a
            // fresh — possibly empty — buffer takes the final commit, as
            // commit()'s last buffer does.
            if (!command) command = newCommand();
            [command endCommandBuffer];

            // The commit feedback delivers the GPU timing and error the
            // Metal 3 path reads off its finished command buffer; there is
            // no separate completion handler to register.
            MTL4CommitOptions *options = [MTL4CommitOptions new];
            [options addFeedbackHandler:
                ^(id<MTL4CommitFeedback> feedback) {
                    // Feedback runs after this commit's GPU work, and the
                    // split commits of the same submission finished before
                    // it, so every timestamp written for the submission is
                    // resolvable here. Entries are nanoseconds; a missing or
                    // unwritten pair (a failed commit resolves 0) is skipped.
                    if (timestamps4 && timingIndex) {
                        NSData *resolved = [timestamps4 resolveCounterRange:
                            NSMakeRange(0, timingIndex * 2)];
                        const auto *stamps = static_cast<
                            const MTL4TimestampHeapEntry *>(resolved.bytes);
                        const NSUInteger entries =
                            resolved.length / sizeof(MTL4TimestampHeapEntry);
                        for (NSUInteger unit = 0; unit < timingIndex; ++unit) {
                            if (unit * 2 + 1 >= entries ||
                                unit >= timingNames.size())
                                break;
                            const uint64_t begin = stamps[unit * 2].timestamp;
                            const uint64_t end =
                                stamps[unit * 2 + 1].timestamp;
                            if (end > begin)
                                recordOpTiming(timingNames[unit],
                                               (end - begin) * 1e-9);
                        }
                    }
                    ticketState->finishCommand4(feedback);
                }];
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
    // Event-bearing commands stay on the Metal 3 path unless opted in: the
    // commit4 event split misattributed GPU timing and regressed prefix-hit
    // prefill output and wall time once. RICHENGINE_MTL4_EVENTS is the A/B
    // switch that re-enables it.
    if (impl_->useMtl4() && (command.events.empty() || tuning().mtl4Events)) {
        return impl_->commit4(prepared.dispatches, command.events,
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

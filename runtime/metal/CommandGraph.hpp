#pragma once

#include "MetalBackend.hpp"
#include "Tuning.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace richengine::metal {

// An ordered dispatch list for one command, with the event steps that order
// it against another agent (EventStep); MetalBackend splits a command into
// Metal command buffers at its event signals. Buffers bind at indices
// 0..n-1; an optional parameter struct binds at index n and is copied into
// graph-owned storage until submission.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Dispatches point into payloads_; a copy would keep pointing at the source.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&) noexcept = default;
  CommandGraph &operator=(CommandGraph &&) noexcept = default;

  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(std::move(pipeline), std::move(buffers), groups, threads);
  }

  template <class Params>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    payloads_.emplace_back(sizeof(Params));
    std::memcpy(payloads_.back().data(), &params, sizeof(Params));
    ComputeDispatch &dispatch =
        push(std::move(pipeline), std::move(buffers), groups, threads);
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              payloads_.back().data(), sizeof(Params)});
  }

  // A dispatch whose kernel takes two constant payloads after its buffers,
  // `params` at index n and `tail` at n + 1 (the paged attention splits'
  // window_tokens argument).
  template <class Params, class Tail>
  void addTail(std::string pipeline, std::vector<MetalBuffer> buffers,
               const Params &params, const Tail &tail, DispatchSize groups,
               DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params> &&
                      std::is_trivially_copyable_v<Tail>,
                  "dispatch parameters must be plain data");
    payloads_.emplace_back(sizeof(Params));
    std::vector<std::byte> &first = payloads_.back();
    std::memcpy(first.data(), &params, sizeof(Params));
    payloads_.emplace_back(sizeof(Tail));
    std::vector<std::byte> &second = payloads_.back();
    std::memcpy(second.data(), &tail, sizeof(Tail));
    ComputeDispatch &dispatch =
        push(std::move(pipeline), std::move(buffers), groups, threads);
    const uint32_t at = static_cast<uint32_t>(dispatch.buffers.size());
    dispatch.bytes.push_back({at, first.data(), sizeof(Params)});
    dispatch.bytes.push_back({at + 1, second.data(), sizeof(Tail)});
  }

  // A dispatch whose parameter payload binds at `paramsIndex`, which may sit
  // before the last buffer: `buffers` holds the payload-free indices in
  // order — the entry that would land at `paramsIndex` binds at
  // paramsIndex + 1 instead (canvas_entropy_accept keeps a device buffer at
  // index 7 behind its constant parameters at index 6).
  template <class Params>
  void addParamsAt(std::string pipeline, std::vector<MetalBuffer> buffers,
                   const Params &params, uint32_t paramsIndex,
                   DispatchSize groups,
                   DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    if (paramsIndex > buffers.size()) {
      throw std::invalid_argument(
          "dispatch parameter index past the buffer list");
    }
    ComputeDispatch &dispatch = push(std::move(pipeline), {},
                                   groups, threads);
    dispatch.buffers.reserve(buffers.size());
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      const uint32_t at = index < paramsIndex ? index : index + 1;
      dispatch.buffers.push_back({at, std::move(buffers[index])});
    }
    payloads_.emplace_back(sizeof(Params));
    std::memcpy(payloads_.back().data(), &params, sizeof(Params));
    dispatch.bytes.push_back(
        {paramsIndex, payloads_.back().data(), sizeof(Params)});
  }

  // The two-payload form of addPatchable().
  template <class Params, class Tail>
  void addPatchableTail(std::string pipeline,
                        std::vector<MetalBuffer> buffers,
                        const Params &params, const Tail &tail,
                        DispatchSize groups,
                        DispatchSize threads = {kDefaultThreads, 1, 1}) {
    addTail(std::move(pipeline), std::move(buffers), params, tail, groups,
            threads);
    static const bool disabled = tuning().patchableOff;
    if (disabled) {
      dispatches_.back().bakeable = false;
    } else {
      dispatches_.back().patchableBytes = true;
    }
  }

  // Event steps after the dispatches added so far (EventStep): signal once
  // all earlier work has completed, or hold all later work until the event
  // reaches value.
  void signal(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Signal);
  }
  void wait(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Wait);
  }

  // Like add(), but marks the dispatch's parameter payload as changing
  // between submissions: a baked span containing it replays with the
  // payload rewritten in place rather than re-baking (see
  // ComputeDispatch::patchableBytes). RICHENGINE_PATCHABLE_OFF makes the
  // dispatch non-bakeable instead — the pre-patchable behavior of a
  // suspension — for A/B measurement.
  template <class Params>
  void addPatchable(std::string pipeline, std::vector<MetalBuffer> buffers,
                    const Params &params, DispatchSize groups,
                    DispatchSize threads = {kDefaultThreads, 1, 1}) {
    add(std::move(pipeline), std::move(buffers), params, groups, threads);
    static const bool disabled = tuning().patchableOff;
    if (disabled) {
      dispatches_.back().bakeable = false;
    } else {
      dispatches_.back().patchableBytes = true;
    }
  }

  // Marks the dispatches added between the calls as one replayable span.
  // Every buffer binding, parameter payload and geometry of the span must be
  // identical on each submission that reuses it; the backend revalidates and
  // falls back to direct encoding when it is not. The exception is a
  // dispatch added through addPatchable(): its payload may change and is
  // rewritten into the span's staged parameters on every replay. Spans nest
  // (an inner begin/end pair leaves the outer span open).
  void beginBakedSpan() { ++bakedSpanDepth_; }
  void endBakedSpan() {
    if (!bakedSpanDepth_)
      throw std::logic_error("CommandGraph baked span was not open");
    --bakedSpanDepth_;
  }
  // Excludes dispatches from any enclosing span — spans nest, so ending an
  // inner span cannot remove dispatches from an outer one; a suspension
  // leaves them out of every span. Nests like the spans.
  void suspendBakedSpan() { ++bakedSpanSuspend_; }
  void resumeBakedSpan() {
    if (!bakedSpanSuspend_)
      throw std::logic_error("CommandGraph baked span was not suspended");
    --bakedSpanSuspend_;
  }

  [[nodiscard]] bool empty() const noexcept { return dispatches_.empty(); }
  // The dispatches alone; command() has the event steps too.
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
  }
  [[nodiscard]] Command command() const noexcept {
    return {dispatches_, events_};
  }

private:
  ComputeDispatch &push(std::string pipeline, std::vector<MetalBuffer> buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = std::move(pipeline);
    dispatch.threadgroups = groups;
    dispatch.threadsPerThreadgroup = threads;
    dispatch.bakeable = bakedSpanDepth_ > 0 && !bakedSpanSuspend_;
    dispatch.buffers.reserve(buffers.size());
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      dispatch.buffers.push_back({index, std::move(buffers[index])});
    }
    dispatches_.push_back(std::move(dispatch));
    return dispatches_.back();
  }

  void step(SharedEvent event, uint64_t value, EventStep::Kind kind) {
    events_.push_back({dispatches_.size(), std::move(event), value, kind});
  }

  uint32_t bakedSpanDepth_ = 0;
  uint32_t bakedSpanSuspend_ = 0;
  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
  std::vector<EventStep> events_;
};

} // namespace richengine::metal

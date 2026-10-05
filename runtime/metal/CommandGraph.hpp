#pragma once

#include "MetalBackend.hpp"

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

// An ordered dispatch list for one command buffer. Buffers bind at indices
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
    static const bool disabled =
        std::getenv("RICHENGINE_PATCHABLE_OFF") != nullptr;
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
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
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

  uint32_t bakedSpanDepth_ = 0;
  uint32_t bakedSpanSuspend_ = 0;
  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
};

} // namespace richengine::metal

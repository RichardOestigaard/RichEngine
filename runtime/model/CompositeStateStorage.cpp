#include "model/CompositeStateStorage.hpp"

#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace richengine::model {
namespace {

using metal::MetalBuffer;

void *writableContents(const MetalBuffer &buffer, const char *name) {
  void *contents = buffer.contents();
  if (!contents) {
    throw std::logic_error(std::string(name) + " is not CPU-visible");
  }
  return contents;
}

void copyExact(const MetalBuffer &destination, const MetalBuffer &source,
               const char *name) {
  if (!destination.sizeBytes() && !source.sizeBytes())
    return;
  if (destination.sizeBytes() != source.sizeBytes()) {
    throw std::logic_error(std::string(name) + " shape mismatch");
  }
  std::memcpy(writableContents(destination, name),
              writableContents(source, name), destination.sizeBytes());
}

void clear(const MetalBuffer &buffer, const char *name) {
  if (!buffer.sizeBytes())
    return;
  std::memset(writableContents(buffer, name), 0, buffer.sizeBytes());
}

std::vector<std::span<std::byte>> stateSpans(
    const GdnParityBuffers &gdn, const std::vector<DFlashDraftRingLayer> &draft) {
  std::vector<std::span<std::byte>> spans;
  const auto append = [&](const MetalBuffer &buffer) {
    if (buffer.sizeBytes())
      spans.emplace_back(static_cast<std::byte *>(writableContents(buffer, "state IO")),
                         buffer.sizeBytes());
  };
  append(gdn.stateBase);
  for (const auto &layer : draft) {
    append(layer.keys);
    append(layer.values);
  }
  return spans;
}

class FileOffload final : public StateOffload {
public:
  FileOffload(std::shared_ptr<SlotFile::Operation> operation,
              std::shared_ptr<const CompositeState> state,
              std::shared_ptr<StateStaging> staging)
      : operation_(std::move(operation)), state_(std::move(state)),
        staging_(std::move(staging)) {}
  // The staging copy is the write's source until the worker has stopped.
  ~FileOffload() override { operation_->drain(); }
  bool ready() const noexcept override { return operation_->ready(); }
  bool finish() override { return operation_->wait(); }
  const std::shared_ptr<const CompositeState> &state() const noexcept override {
    return state_;
  }
private:
  std::shared_ptr<SlotFile::Operation> operation_;
  std::shared_ptr<const CompositeState> state_;
  std::shared_ptr<StateStaging> staging_;
};

class FileRestore final : public StateRestore {
public:
  FileRestore(std::shared_ptr<SlotFile::Operation> operation,
              std::function<void()> committed,
              std::function<std::shared_ptr<const CompositeState>()> snapshot)
      : operation_(std::move(operation)), committed_(std::move(committed)),
        snapshot_(std::move(snapshot)) {}
  ~FileRestore() override { operation_->drain(); }
  bool ready() const noexcept override { return operation_->ready(); }
  void cancel() noexcept override { operation_->cancel(); }
  bool finish() override {
    if (!operation_->wait()) return false;
    committed_();
    finished_ = true;
    return true;
  }
  std::shared_ptr<const CompositeState> snapshot() override {
    return finished_ ? snapshot_() : nullptr;
  }
private:
  std::shared_ptr<SlotFile::Operation> operation_;
  std::function<void()> committed_;
  std::function<std::shared_ptr<const CompositeState>()> snapshot_;
  bool finished_ = false;
};

} // namespace

GdnCell::GdnCell(metal::MetalBackend &backend,
                         std::shared_ptr<StateAllocationTracker> tracker,
                         GdnStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)) {
  if (!tracker_)
    throw std::invalid_argument("state allocation tracker is empty");
  if (!layout.valid())
    throw std::invalid_argument("GDN state layout is invalid");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  // A stateless target's cell is empty: no layers, no allocation.
  buffers_.stateBase =
      layout.cellBytes()
          ? backend.allocateBuffer(layout.cellBytes(), metal::BufferStorage::Shared, label)
          : metal::MetalBuffer{};
  // The cell holds every layer's convolution state, then every layer's
  // recurrent state.
  buffers_.convolutionLayers.resize(layout.layers);
  buffers_.recurrentLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.convolutionLayers[layer] =
        backend.view(buffers_.stateBase,
                     uint64_t{layer} * layout.convolutionLayerBytes(),
                     layout.convolutionLayerBytes());
    if (layout.recurrentLayerBytes())
      buffers_.recurrentLayers[layer] = backend.view(
          buffers_.stateBase,
          layout.convolutionBytes() +
              uint64_t{layer} * layout.recurrentLayerBytes(),
          layout.recurrentLayerBytes());
  }
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.cellBytes()) {
    throw std::logic_error("GDN allocation is below declared bytes");
  }
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

GdnCell::~GdnCell() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

CompositeStateImpl::CompositeStateImpl(std::shared_ptr<StateBufferPool> pool,
                                       CachedStateBuffers buffers,
                                       CompositeStateLayout layout,
                                       LogicalLengths lengths,
                                       std::shared_ptr<SlotFile> file,
                                       std::shared_ptr<StateStaging> staging)
    : pool_(std::move(pool)), buffers_(std::move(buffers)), layout_(layout),
      lengths_(lengths), file_(std::move(file)), staging_(std::move(staging)) {
  if (!pool_ || !buffers_.gdn || !buffers_.draft) {
    throw std::invalid_argument("composite state buffers are empty");
  }
}

CompositeStateImpl::CompositeStateImpl(CompositeStateLayout layout,
    LogicalLengths lengths, std::shared_ptr<SlotFile> file,
    std::shared_ptr<SlotFile::Slot> disk)
    : layout_(layout), lengths_(lengths), file_(std::move(file)), disk_(std::move(disk)) {}

std::unique_ptr<StateOffload>
CompositeStateImpl::offload(std::function<void()> completion) const {
  if (!canOffload()) return {};
  return write(file_, staging_,
               stateSpans(buffers_.gdn->buffers(), buffers_.draft->layers()),
               layout_, lengths_, std::move(completion));
}

std::unique_ptr<StateOffload> CompositeStateImpl::write(
    const std::shared_ptr<SlotFile> &file, const std::shared_ptr<StateStaging> &staging,
    const std::vector<std::span<std::byte>> &spans, CompositeStateLayout layout,
    LogicalLengths lengths, std::function<void()> completion) {
  auto disk = file->acquire();
  if (!disk) return {};
  auto result = std::shared_ptr<const CompositeStateImpl>(
      new CompositeStateImpl(layout, lengths, file, disk));
  const std::span<std::byte> staged = staging->bytes();
  std::byte *cursor = staged.data();
  for (auto span : spans)
    cursor = std::copy(span.begin(), span.end(), cursor);
  std::shared_ptr<SlotFile::Operation> operation;
  try {
    operation = file->write(
        std::move(disk), {std::span<const std::byte>(staged)},
        std::move(completion));
    // Callers check that the file takes writes, and only a failed write of
    // its own closes it: none is in flight beside this one.
    if (!operation)
      throw std::logic_error("the state file closed with no state write in flight");
    return std::make_unique<FileOffload>(operation, std::move(result), staging);
  } catch (...) {
    if (operation)
      operation->drain();
    throw;
  }
}

std::unique_ptr<StateOffload>
CompositeStateImpl::persist(std::function<void()> completion) const {
  if (!canOffload())
    return {};
  auto disk = file_->acquire();
  if (!disk)
    return {};
  auto copy = std::shared_ptr<const CompositeState>(
      new CompositeStateImpl(layout_, lengths_, file_, disk));
  const auto spans = stateSpans(buffers_.gdn->buffers(), buffers_.draft->layers());
  auto operation = file_->write(std::move(disk), {spans.begin(), spans.end()},
                                std::move(completion));
  // As for write(): only a failed write of its own closes the file.
  if (!operation)
    throw std::logic_error("the state file closed with no state write in flight");
  return std::make_unique<FileOffload>(operation, std::move(copy), nullptr);
}

void CompositeStateImpl::label(std::vector<std::byte> label) const {
  if (disk_)
    file_->label(disk_, std::move(label));
}

CompositeStateImpl::~CompositeStateImpl() {
  if (!pool_ || !pool_->open)
    return;
  pool_->cells.push_back(std::move(buffers_.gdn));
  pool_->rings.push_back(std::move(buffers_.draft));
}

CompositeStateStorage::CompositeStateStorage(metal::MetalBackend &backend,
                                   metal::AllocationAdmission admitAllocation,
                                   CompositeStateLayout layout,
                                   std::shared_ptr<SlotFile> file)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout),
      allocations_(std::make_shared<StateAllocationTracker>()),
      pool_(std::make_shared<StateBufferPool>()) {
  if (!admitAllocation_)
    throw std::invalid_argument("state allocation admission is required");
  if (!layout_.valid())
    throw std::invalid_argument("composite state layout is invalid");
  if (file) {
    file_ = std::move(file);
    staging_ = std::make_shared<StateStaging>();
    staging_->buffer = backend_.allocateBuffer(
        layout_.cachedBytes(), metal::BufferStorage::Shared, "state-staging");
  }
}

CompositeStateStorage::~CompositeStateStorage() {
  pool_->open = false;
  pool_->cells.clear();
  pool_->rings.clear();
}

const LaneMetadata &CompositeStateStorage::metadata(uint32_t index) const {
  return lane(index).metadata;
}

const GdnParityBuffers &CompositeStateStorage::current(uint32_t index) const {
  const Lane &assigned = lane(index);
  requireAssigned(assigned);
  return assigned.cells.gdn[assigned.metadata.activeParity]->buffers();
}

const GdnParityBuffers &CompositeStateStorage::next(uint32_t index) const {
  const Lane &assigned = lane(index);
  requireAssigned(assigned);
  return assigned.cells.gdn[assigned.metadata.activeParity ^ 1]->buffers();
}

const std::vector<DFlashDraftRingLayer> &
CompositeStateStorage::draft(uint32_t index) const {
  const Lane &assigned = lane(index);
  requireAssigned(assigned);
  return assigned.cells.draft->layers();
}

metal::AllocationResult
CompositeStateStorage::tryActivateLane(uint32_t index, uint64_t requestId, uint64_t extraBytes,
                                  const std::function<void()> &allocateExtra) {
  if (!requestId)
    throw std::invalid_argument("request id must be non-zero");
  Lane &current = lane(index);
  if (current.metadata.assigned()) {
    throw std::logic_error("lane is already assigned");
  }
  Buffers buffers;
  if (auto admission = acquire(CompositeStateLayout::kLaneGdnCells,
                               "state-lane-" + std::to_string(index),
                               buffers, extraBytes, allocateExtra);
      !admission)
    return admission;
  current.cells = std::move(buffers);
  current.metadata = {requestId, 0, {}};
  return {};
}

void CompositeStateStorage::releaseLane(uint32_t index, uint64_t requestId) {
  Lane &current = lane(index);
  requireAssigned(current);
  if (!requestId || current.metadata.requestId != requestId) {
    throw std::logic_error("lane owner mismatch");
  }
  // Parity one first, so the next activation pops parity zero first and a
  // reactivated lane gets its previous buffers back in the same order.
  for (uint32_t parity = current.cells.gdn.size(); parity > 0;) {
    --parity;
    pool_->cells.push_back(std::move(current.cells.gdn[parity]));
  }
  pool_->rings.push_back(std::move(current.cells.draft));
  current.metadata = {};
}

void CompositeStateStorage::clearForColdStart(uint32_t index) {
  const Lane &target = lane(index);
  requireAssigned(target);
  if (target.metadata.lengths != LogicalLengths{})
    throw std::logic_error("a cold start begins at logical length zero");
  clear(current(index).stateBase, "cold-start GDN state");
}

uint64_t CompositeStateStorage::releaseOneIdle(bool keepLane) noexcept {
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  if (idleCells() > (keepLane ? CompositeStateLayout::kLaneGdnCells : 0))
    pool_->cells.pop_back();
  else if (idleRings() > (keepLane ? 1U : 0U))
    pool_->rings.pop_back();
  else
    return 0;
  const uint64_t after = backend_.memoryStats().allocatedBytes;
  return before >= after ? before - after : 0;
}

uint32_t CompositeStateStorage::idleCells() const noexcept {
  return static_cast<uint32_t>(pool_->cells.size());
}

uint32_t CompositeStateStorage::idleRings() const noexcept {
  return static_cast<uint32_t>(pool_->rings.size());
}

uint64_t CompositeStateStorage::missingBytes(uint32_t cells) const noexcept {
  const uint64_t missing = cells - std::min<uint64_t>(pool_->cells.size(), cells);
  return missing * layout_.target.cellBytes() +
         (pool_->rings.empty() ? layout_.draft.ringBytes() : 0);
}

uint32_t CompositeStateStorage::statesToActivate() const noexcept {
  constexpr uint32_t laneCells = CompositeStateLayout::kLaneGdnCells;
  const uint32_t cells = laneCells - std::min(idleCells(), laneCells);
  const uint32_t rings = idleRings() ? 0 : 1;
  return std::max(cells, rings);
}

void CompositeStateStorage::updateLengths(uint32_t index,
                                     LogicalLengths lengths) {
  validateLengths(lengths, false);
  Lane &current = lane(index);
  requireAssigned(current);
  current.metadata.lengths = lengths;
}

void CompositeStateStorage::swapParity(uint32_t index) {
  Lane &current = lane(index);
  requireAssigned(current);
  current.metadata.activeParity ^= 1;
}

std::shared_ptr<const CompositeStateImpl>
CompositeStateStorage::snapshot(uint32_t index) {
  return snapshot(index, lane(index).metadata.lengths);
}

std::shared_ptr<const CompositeStateImpl>
CompositeStateStorage::snapshot(uint32_t index, LogicalLengths lengths) {
  const GdnParityBuffers &gdn = current(index);
  const std::vector<DFlashDraftRingLayer> &ring = draft(index);
  validateLengths(lengths, true);
  Buffers buffers;
  if (!acquire(1, "state-cache", buffers))
    return nullptr;
  CachedStateBuffers cached{std::move(buffers.gdn[0]), std::move(buffers.draft)};
  copyExact(cached.gdn->buffers().stateBase, gdn.stateBase,
            "cached GDN state");
  for (uint32_t layer = 0; layer < ring.size(); ++layer) {
    copyExact(cached.draft->layers()[layer].keys, ring[layer].keys,
              "cached draft keys");
    copyExact(cached.draft->layers()[layer].values, ring[layer].values,
              "cached draft values");
  }
  return std::shared_ptr<const CompositeStateImpl>(new CompositeStateImpl(
      pool_, std::move(cached), layout_, lengths, file_, staging_));
}

std::unique_ptr<StateOffload>
CompositeStateStorage::snapshotToDisk(uint32_t index, std::function<void()> completion) {
  const Lane &source = lane(index);
  requireAssigned(source);
  validateLengths(source.metadata.lengths, true);
  if (!canSnapshotToDisk())
    return {};
  return CompositeStateImpl::write(file_, staging_,
                                   stateSpans(current(index), draft(index)),
                                   layout_, source.metadata.lengths,
                                   std::move(completion));
}

metal::AllocationResult
CompositeStateStorage::acquire(uint32_t cells, std::string_view label, Buffers &buffers,
                          uint64_t extraBytes, const std::function<void()> &allocateExtra) {
  const uint32_t pooledCells =
      std::min(cells, static_cast<uint32_t>(pool_->cells.size()));
  const bool pooledRing = !pool_->rings.empty();
  Buffers fresh;
  if (const uint64_t bytes = missingBytes(cells) + extraBytes) {
    const auto admission = admitAllocation_(bytes, [&] {
      if (allocateExtra)
        allocateExtra();
      for (uint32_t cell = pooledCells; cell < cells; ++cell) {
        fresh.gdn[cell] = std::shared_ptr<GdnCell>(new GdnCell(
            backend_, allocations_, layout_.target,
            std::string(label) + "-gdn-" + std::to_string(cell)));
      }
      if (!pooledRing) {
        fresh.draft = std::shared_ptr<DFlashDraftRing>(new DFlashDraftRing(
            backend_, allocations_, layout_.draft,
            std::string(label) + "-draft"));
      }
    });
    // What a driver's refusal left of the attempt goes with `fresh`.
    if (!admission)
      return admission;
  }
  for (uint32_t cell = 0; cell < pooledCells; ++cell) {
    buffers.gdn[cell] = std::move(pool_->cells.back());
    pool_->cells.pop_back();
  }
  for (uint32_t cell = pooledCells; cell < cells; ++cell)
    buffers.gdn[cell] = std::move(fresh.gdn[cell]);
  // A zero-byte layout admits without running the allocation lambda, so its
  // cells would stay null; every lane still binds them.
  for (uint32_t cell = 0; cell < cells; ++cell)
    if (!buffers.gdn[cell])
      buffers.gdn[cell] = std::shared_ptr<GdnCell>(new GdnCell(
          backend_, allocations_, layout_.target,
          std::string(label) + "-gdn-" + std::to_string(cell)));
  if (pooledRing) {
    buffers.draft = std::move(pool_->rings.back());
    pool_->rings.pop_back();
  } else {
    buffers.draft = std::move(fresh.draft);
    if (!buffers.draft)
      buffers.draft = std::shared_ptr<DFlashDraftRing>(new DFlashDraftRing(
          backend_, allocations_, layout_.draft,
          std::string(label) + "-draft"));
  }
  return {};
}

void CompositeStateStorage::restore(uint32_t index, const CompositeStateImpl &state,
                               bool restoreDraftState) {
  copyExact(current(index).stateBase, state.buffers_.gdn->buffers().stateBase,
            "restored GDN state");
  if (restoreDraftState) {
    const std::vector<DFlashDraftRingLayer> &ring = draft(index);
    for (uint32_t layer = 0; layer < ring.size(); ++layer) {
      copyExact(ring[layer].keys, state.buffers_.draft->layers()[layer].keys,
                "restored draft keys");
      copyExact(ring[layer].values, state.buffers_.draft->layers()[layer].values,
                "restored draft values");
    }
  }
  restoreLengths(index, state.lengths_, restoreDraftState);
}

void CompositeStateStorage::restoreLengths(uint32_t index, LogicalLengths lengths,
                                     bool restoreDraftState) {
  Lane &destination = lane(index);
  if (!restoreDraftState) {
    lengths.draftBase = lengths.targetTokens;
    lengths.draftLength = 0;
  }
  destination.metadata.lengths = lengths;
}

std::unique_ptr<StateRestore> CompositeStateStorage::beginRestore(
    uint32_t index, const CompositeState &state, bool restoreDraftState,
    std::function<void()> completion, std::function<void()> committed) {
  const auto *typed = dynamic_cast<const CompositeStateImpl *>(&state);
  if (!typed || typed->layout_ != layout_)
    throw std::invalid_argument("incompatible composite state");
  validateLengths(typed->lengths_, true);
  requireAssigned(lane(index));
  if (!typed->disk_) {
    restore(index, *typed, restoreDraftState);
    committed();
    return {};
  }
  auto spans = stateSpans(current(index), draft(index));
  auto commit = [this, index, lengths = typed->lengths_, restoreDraftState,
                 committed = std::move(committed)] {
    restoreLengths(index, lengths, restoreDraftState);
    committed();
  };
  auto operation = typed->file_->read(typed->disk_, std::move(spans), std::move(completion));
  try {
    return std::make_unique<FileRestore>(operation, std::move(commit),
        [this, index, lengths = typed->lengths_] { return snapshot(index, lengths); });
  } catch (...) {
    operation->drain();
    throw;
  }
}

std::shared_ptr<const CompositeState> CompositeStateStorage::adopt(const SlotRecord &record,
                                                              uint32_t tokens) {
  if (!file_)
    throw std::logic_error("states are taken back only from a disk tier");
  const uint32_t window = std::min(tokens, layout_.draft.window);
  const LogicalLengths lengths{tokens, tokens - window, window};
  validateLengths(lengths, true);
  if (record.payloadBytes != layout_.cachedBytes())
    throw std::invalid_argument("a recorded state does not match the state layout");
  return std::shared_ptr<const CompositeState>(
      new CompositeStateImpl(layout_, lengths, file_, file_->adopt(record)));
}

CompositeStateStorage::Lane &CompositeStateStorage::lane(uint32_t index) {
  if (index >= lanes_.size()) {
    throw std::out_of_range("invalid lane");
  }
  return lanes_[index];
}

const CompositeStateStorage::Lane &CompositeStateStorage::lane(uint32_t index) const {
  if (index >= lanes_.size()) {
    throw std::out_of_range("invalid lane");
  }
  return lanes_[index];
}

void CompositeStateStorage::validateLengths(const LogicalLengths &lengths,
                                       bool cacheSnapshot) const {
  if (lengths.draftLength > ExecutionLimits::draftRingCapacity ||
      lengths.draftEnd() > lengths.targetTokens) {
    throw std::invalid_argument("invalid draft ring metadata");
  }
  if (cacheSnapshot &&
      (!lengths.targetTokens ||
       !lengths.hasCompleteDraftWindow(layout_.draft.window) ||
       lengths.targetTokens % kv::kPageTokens)) {
    throw std::invalid_argument(
        "composite snapshot requires equal page-aligned committed lengths");
  }
}

void CompositeStateStorage::requireAssigned(const Lane &current) {
  if (!current.metadata.assigned()) {
    throw std::logic_error("lane is not assigned");
  }
}

} // namespace richengine::model

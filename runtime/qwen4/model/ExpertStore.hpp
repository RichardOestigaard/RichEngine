#pragma once

// Routed experts that stay in their source files and are read into slots when
// a step routes them (--expert-memory), so a MoE target whose experts do not
// fit in memory runs with part of them resident. The kernels read every
// routed expert through its block's slot table (ops::BlockMoeWeights), so
// resident and streamed experts run the same kernels.
//
// Experts of one size (the formats and shapes of their gate, up and down
// projections) share its slots, which chunks hold, each a buffer of its own
// (ops::ExpertChunks, model/ExpertChunkLayout.hpp): the floor chunk holds the
// size's floor (ExpertStoreOptions::floorBytes, 0.85 of its slots by
// default), and elastic chunks of about 1 GiB each the rest, up to the cap
// (ExpertStoreOptions::memoryBytes). While memory is short the store gives
// its last live elastic chunk back (releaseAboveFloor), the experts it holds
// moved to free slots below it and the coldest evicted to make room, and once
// memory is ample takes it back with its slots free (growTowardCap): the live
// chunks are always the first ones, which the kernels' address tables name
// alone. The kernels reach the chunks through their GPU addresses, which the
// backend writes per dispatch and declares resident once per encoder
// (metal::AddressTable); each chunk an encoder makes resident costs GPU time,
// so they are few. A size of several blocks streams: its slots are recycled
// across its blocks by adaptive replacement (SlotRecycler), which keeps the
// experts decode steps route again and again, and recycles a long prefill
// chunk's loads first. A size of one block (a layer whose experts take
// formats no other layer's do, as a dynamic quantization leaves some) holds
// every one of its experts, in its floor chunk alone: they are read when the
// store is built or restored, and nothing loads them again.
//
// A streamed block's load runs after its steps' grouping (ops::
// MoeExpertStream), in rounds: each a host step of the command, which claims
// a slot for each routed expert that is not resident (in the first round),
// never one of an expert the same load routes, and reads the native rows of
// some of them into places of the store's staging, then a repack
// (moe_expert_load) of those places into their slots. A decode step's load
// is one round over every place; any other load stages half the places a
// round, as many rounds as its rows may need, so that the other half is free
// for the read-ahead below while the GPU repacks. A run of consecutive
// experts' rows that a file stores as they are is read straight from it
// (F_NOCACHE without the buffer cache): its bytes widened to whole pages, into
// places that start on a page, and the repack starts from the rows' offset
// in the first page. Rows a loader builds (a quantized safetensors tensor's)
// are built into the places instead.
//
// Once a round has staged its experts, the next experts are read ahead while
// the GPU runs, into the places the round left free, an expert at a time in
// pieces of at most kAheadPieceBytes, most likely first: the load's experts
// of its next round, or after a decode step's round the next block's experts
// that are likely routed and not resident, those the next block's router
// picks from this block's input (the rows change little from block to
// block), by their summed routing weights. A long prefill chunk (rows * topK
// >= 3 * experts) routes nearly every expert of the next block: one more host
// step after its last repack reads them ahead into the whole staging, while
// the GPU runs this block's experts and the next block's mixing. The next
// round stops the read-ahead and takes over the experts it routes where they
// are, reading what the readers had not (or failed to: a read ahead of an
// expert no step routes is no failure).
//
// The live chunks and the staging are memory the engine gives back while
// idle (model::WeightMemory): released, every slot is free. A restore takes
// the floor chunks and the staging back in one part, as memory the next
// request needs, then each elastic chunk in a part of its own while memory
// admits it as growth it may do without, the rest left to a later
// growTowardCap(); it reads the experts of the sizes of one block again, and
// the loads after it every expert they route.
//
// Reads run on the engine's SSD readers (model::SsdReaders), which try a
// failed read once more. A load checks its sources unchanged after it read
// them. A load that fails (a read of an expert it routes fails twice, or a
// source was written) frees the slots of the experts it had not repacked and
// fails its command; every later load fails too.
//
// Threads: bind() runs while the model loads, before any command; addLoad()
// on the thread that encodes a command, reading only what construction and
// bind() set and which chunks are live, which changes between commands; the
// loads on the backend's host-step thread, one at a time
// (metal::EventStep); the reads of a read-ahead on the readers, until the
// next load, a release or the store's end stops them; release(),
// restore(), releaseAboveFloor(), releaseToFloor(), growTowardCap() and
// growthBytes() on the engine thread while no command is in flight
// (model::WeightMemory), so never beside a load, under the store's mutex,
// which orders them against a read-ahead's start; a read-ahead writes the
// staging alone, so they let it run on; released() on the engine thread and
// stats() on any. Neither of those two waits for a load: a status query
// never waits behind a command's reads.
//
// Shrinks and growth (releaseAboveFloor(), releaseToFloor(), growTowardCap())
// run only between commands that succeeded. A failed command ends the engine
// (engine::NativeRuntime's failure boundary), so they never see what a load
// left unfinished: its routed experts still waiting to be staged (pending),
// or a round's staged experts whose repack never ran, whose slots are claimed
// but hold nothing. The store cannot tell that a command failed after a
// load's host step ran, and need not. Its guards refuse only where it knows
// of an unfinished load: each once a load failed, and a shrink also while
// routed experts wait to be staged.

#include "qwen4/metal/MetalBackend.hpp"
#include "qwen4/model/ExpertChunkLayout.hpp"
#include "qwen4/model/ExpertStoreStats.hpp"
#include "qwen4/model/GgufImage.hpp"
#include "qwen4/model/SsdReaders.hpp"
#include "qwen4/model/WeightMemory.hpp"
#include "qwen4/ops/MoE.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace splash::model {

// The blocks of a target whose routed experts stream (gguf::PlanOptions::
// streamExperts): each block's experts as its plan left them out of its
// image, in block order (the target's `layers` layers), what holds the
// sources their rows name open (the store keeps it), and a check of each
// source, which throws once it was written since it was opened.
struct StreamedBlocks final {
  std::vector<gguf::StreamedExperts> blocks;
  std::shared_ptr<const void> sources;
  std::vector<std::function<void()>> checks;
};

struct ExpertStoreOptions final {
  // Every byte the store allocates at its cap: its staging and tables
  // (ExpertStore::fixedBytes), and in the rest the chunks' slots, which hold
  // at least a block's experts of each size and all of a size of one block.
  uint64_t memoryBytes = 0;
  // The floor in the same bytes (--expert-memory SIZE), what the store keeps
  // however short memory is, from minimumBytes to memoryBytes: each streamed
  // size's floor is the share of its slots those bytes hold, as memoryBytes'
  // slots are. Zero for kExpertFloorShare of each streamed size's slots.
  uint64_t floorBytes = 0;
  // The readers its reads run on: the engine's one pool, which the other
  // streamed weights and KV reads share.
  std::shared_ptr<SsdReaders> readers;
  // The most bytes one chunk takes, the floor chunk too: the device's
  // largest buffer (maxBufferLength) when zero or more. A streamed size whose
  // slots it holds only in more than MOE_EXPERT_MAX_CHUNKS chunks, and a
  // size of one block whose experts it does not hold, are refused
  // (streamedExpertChunkLayout, wholeExpertChunkLayout).
  uint64_t bufferBytes = 0;
  // About the most bytes a streamed size's elastic chunk takes, unless the
  // size needs larger ones to fit the expert kernels' table
  // (streamedExpertChunkLayout).
  uint64_t elasticChunkBytes = kExpertElasticChunkBytes;
};

class ExpertStore final : public ops::MoeExpertStream, public WeightMemory {
public:
  // Allocates the chunks, the staging and the slot tables, and reads the
  // experts of the sizes of one block. Throws std::invalid_argument when a
  // block's experts differ from `shape`, when options.memoryBytes, less
  // the staging and tables (fixedBytes), holds fewer slots than a block's
  // experts of each streamed size and every expert of a size of one block,
  // when a streamed size's slots would take more than MOE_EXPERT_MAX_CHUNKS
  // chunks of the largest buffer (options.bufferBytes or the device's
  // maxBufferLength), when a size of one block's experts take more than it,
  // or when options.floorBytes is neither zero nor from the smallest store
  // (minimumBytes) to options.memoryBytes.
  ExpertStore(metal::MetalBackend &backend, StreamedBlocks blocks, ops::MoeShape shape, ExpertStoreOptions options);
  ~ExpertStore() override;
  ExpertStore(const ExpertStore &) = delete;
  ExpertStore &operator=(const ExpertStore &) = delete;

  // Block `block`'s routed segments (the format and size of its size's
  // slots), chunks and slot table, which `weights` reads them through, and,
  // when its experts stream, the store, which loads them; the store keeps the
  // block's router and shared-expert gate, set before, which predict its
  // routes.
  void bind(uint32_t block, ops::BlockMoeWeights &weights);
  void addLoad(metal::CommandGraph &graph, std::span<const ops::MoeLoadPart> parts, uint32_t block) const override;
  [[nodiscard]] ExpertStoreStats stats() const;
  // The live chunks and the staging, while no command is in flight;
  // restore() takes the floor chunks and the staging back in one part under
  // `admit`, then each elastic chunk under an admission of its own by
  // `optional` until one is refused, which stays released with those after
  // it, and reads the experts of the sizes of one block again.
  [[nodiscard]] bool released() const noexcept override;
  void release() override;
  [[nodiscard]] bool restore(const metal::AllocationAdmission &admit,
                             const metal::AllocationAdmission &optional) override;
  // While memory is short: gives back the last live elastic chunk of the
  // streamed size with the largest share of its slots above its floor. The
  // experts it holds move to free slots below it, after as many of the
  // size's experts as they outnumber those slots are evicted, those its
  // recycling would recycle first; then no slot table names it, and no
  // address table from the next command on. Returns the chunk's bytes; zero
  // when every size is at its floor, while released, while a command is in
  // flight, and where the store knows a load is unfinished (above). A
  // reclaim that needs more calls it again. Each chunk it gives back logs a
  // line: how long it took, the experts moved and evicted, and what the
  // store holds after.
  [[nodiscard]] uint64_t releaseAboveFloor();
  // While memory is critically short: gives back every elastic chunk above
  // each size's floor, as releaseAboveFloor() would one at a time, but in
  // one compaction per size, which moves each expert that stays once rather
  // than again with each chunk. Returns their bytes, zero where
  // releaseAboveFloor() gives none; logs a line per size that shrank.
  [[nodiscard]] uint64_t releaseToFloor();
  // The bytes growTowardCap() takes next; zero at the cap, while released
  // and after a failed load.
  [[nodiscard]] uint64_t growthBytes() const;
  // While memory is ample: takes back the next elastic chunk of the streamed
  // size with the smallest share of its slots above its floor, under
  // `admit`, its slots free (nothing is read). Its refusal leaves the chunk
  // released and is returned; at the cap, while released, while a command is
  // in flight and after a failed load the store stays as it is. Each chunk
  // it takes back logs a line.
  [[nodiscard]] metal::AllocationResult growTowardCap(const metal::AllocationAdmission &admit);
  // The bytes a store of these blocks allocates besides its chunks: its
  // staging, slot tables, jobs and the decode prediction's buffers.
  [[nodiscard]] static uint64_t fixedBytes(const StreamedBlocks &blocks, ops::MoeShape shape);
  // The smallest store of these blocks (ExpertStoreOptions::memoryBytes and
  // floorBytes), which the constructor refuses a byte less than: its fixed
  // bytes, a block's experts of each streamed size and every expert of a
  // size of one block.
  [[nodiscard]] static uint64_t minimumBytes(const StreamedBlocks &blocks, ops::MoeShape shape);

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// The slot table of a block whose routed experts are all resident: expert e
// in slot e. A model's blocks share one, of residentExpertSlotsBytes.
[[nodiscard]] metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts);
[[nodiscard]] constexpr uint64_t residentExpertSlotsBytes(uint32_t experts) noexcept {
  return uint64_t{experts} * sizeof(uint32_t);
}

} // namespace splash::model

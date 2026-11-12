#pragma once

// One MoE block's streamed routed experts on the device: a bounded arena of
// expert slots laid out in chunks (model::ExpertChunkLayout), a slot table
// (expert -> slot) and a chunk address table (chunk -> region GPU addresses)
// the expert kernels read, and the slot management (model::ExpertSlotRecycler)
// that decides which expert a missing one evicts. model::PackedExpertStore
// supplies the bytes; stage() copies them into the slots, so the arena needs
// no GPU command of its own — its buffers are Shared, which the CPU writes
// and the GPU reads.
//
// The production port of the staged qwen4 reference's device side
// (runtime/qwen4/model/ExpertStore.cpp + ops::ExpertChunks). The packed
// Q8_0 slabs copy verbatim: an expert's file slice is the byte range the
// kernels read at local slot * rows of its chunk's region, so no repacking
// kernel is needed (the staged reference's moe_expert_load exists because
// its GGUF sources needed repacking).
//
// Bound today: Qwen4Exp's prefill MoE blocks bind slotTable() and
// chunkAddresses() through ops::MoE's route-first split
// (MoE::addRouting/addGgufGather/addGgufPasses/addCombine), staging the
// routed union between flushed submissions. Verify and decode still run
// the resident ops::MoE::add — their submissions drop a graph's event
// steps, so they have no host sync point for the union yet.

#include "model/ExpertChunkLayout.hpp"
#include "model/ExpertSlotRecycler.hpp"
#include "model/PackedExpertStore.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace richengine::model {

class PackedExpertStore;

struct ExpertArenaOptions final {
  // The device budget of every chunk's buffers together. With `slots` 0 the
  // arena holds as many slots as fit, at most `experts`.
  uint64_t bytes = 0;
  // The backend's largest buffer (capabilities.maxBufferLengthBytes).
  uint64_t largestBuffer = 0;
  // The block's routed experts (512 for Qwen3.8-Flash-Next).
  uint32_t experts = 0;
  // The slots the arena holds; 0 derives them from `bytes`.
  uint32_t slots = 0;
  // The floor's slots; 0 takes kExpertFloorShare of them.
  uint32_t floor = 0;
  // An elastic chunk's bytes; 0 takes kExpertElasticChunkBytes.
  uint64_t elasticChunkBytes = 0;
};

class ExpertArena final {
public:
  // `planes` is the per-slot bytes of each projection's plane0, plane1 and
  // meta ([projection][plane], the PackedExpertTable's region order); it
  // must equal the table's per-expert slice bytes. Throws
  // std::invalid_argument when the geometry disagrees or no slot fits.
  ExpertArena(metal::MetalBackend &backend, const ExpertSlotPlanes &planes,
              const PackedExpertTable &table, const ExpertArenaOptions &options);
  ~ExpertArena() = default;
  ExpertArena(const ExpertArena &) = delete;
  ExpertArena &operator=(const ExpertArena &) = delete;

  [[nodiscard]] const ExpertChunkLayout &layout() const noexcept { return layout_; }
  [[nodiscard]] uint32_t experts() const noexcept { return table_.experts; }
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }

  // The device slot of `expert`, or UINT32_MAX while it holds none.
  [[nodiscard]] uint32_t slotOf(uint32_t expert) const;

  // The slot table the expert kernels read: experts entries of uint32, the
  // global slot of each resident expert, UINT32_MAX for the rest.
  [[nodiscard]] const metal::MetalBuffer &slotTable() const noexcept { return slotTable_; }
  // The chunk address table: chunks * kExpertChunkAddresses uint64 GPU
  // addresses — chunk c's entries from c * kExpertChunkAddresses are the
  // gate's, up's and down's plane0, plane1 and meta regions, a format
  // without plane1 holding its meta region's address in plane1's entry.
  [[nodiscard]] const metal::MetalBuffer &chunkAddresses() const noexcept { return chunkAddresses_; }
  // Chunk `chunk`'s region of projection `z`, plane `p` (0..2, empty planes
  // have no buffer): the GPU-visible bytes its slots occupy.
  [[nodiscard]] const metal::MetalBuffer &region(uint32_t chunk, uint32_t z,
                                                 uint32_t p) const;

  // Makes every id of `ids` resident: hits touch their slots, misses claim
  // slots (the recycler evicts as ARC decides), read their slices through
  // `store` and copy them into place; returns the global slot of each id in
  // order. Throws PackedExpertError when the arena cannot hold the step's
  // distinct experts or a read fails. Stages are serialized; call before the
  // command that routes them is submitted. The slices `store` returns are
  // consumed before this returns, so no prefetch may run concurrently.
  std::vector<uint32_t> stage(PackedExpertStore &store,
                              std::span<const uint32_t> ids, uint64_t load,
                              PackedExpertUse use);

private:
  PackedExpertTable table_;
  ExpertSlotPlanes planes_;
  ExpertChunkLayout layout_;
  ExpertArenaOptions options_;
  uint64_t allocatedBytes_ = 0;

  std::vector<metal::MetalBuffer> regions_; // chunks * kPackedExpertRegions
  metal::MetalBuffer slotTable_;
  metal::MetalBuffer chunkAddresses_;

  std::optional<ExpertSlotRecycler> recycler_;
  std::vector<uint32_t> slotOfExpert_; // experts entries, UINT32_MAX empty
  std::vector<uint32_t> expertOfSlot_; // layout_.slots entries, UINT32_MAX free
  std::vector<uint64_t> addresses_;    // chunks * kExpertChunkAddresses
  mutable std::mutex mutex_;
};

} // namespace richengine::model

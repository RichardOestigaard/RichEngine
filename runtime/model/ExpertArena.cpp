// ExpertArena.cpp: the device side of a block's streamed routed experts.

#include "model/ExpertArena.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace richengine::model {
namespace {

constexpr uint32_t kNoSlot = std::numeric_limits<uint32_t>::max();

} // namespace

ExpertArena::ExpertArena(metal::MetalBackend &backend,
                         const ExpertSlotPlanes &planes,
                         const PackedExpertTable &table,
                         const ExpertArenaOptions &options)
    : table_(table), planes_(planes), options_(options) {
  if (!options_.experts || table_.experts != options_.experts)
    throw std::invalid_argument("an expert arena's table and options disagree on experts");
  if (!options_.largestBuffer) throw std::invalid_argument("an expert arena needs the largest buffer");
  // The slot planes are the table's per-expert slices, region by region.
  uint64_t slotBytes = 0;
  for (uint32_t z = 0; z < kPackedExpertProjections; ++z)
    for (uint32_t p = 0; p < kPackedExpertPlanes; ++p) {
      const uint64_t slice = table_.regions[z * kPackedExpertPlanes + p].sliceBytes;
      if (planes_[z][p] != slice)
        throw std::invalid_argument("an expert arena's planes are not the table's slices");
      slotBytes += planes_[z][p];
    }
  if (!slotBytes) throw std::invalid_argument("an expert arena has no expert bytes");
  uint32_t slots = options_.slots;
  if (!slots) {
    slots = static_cast<uint32_t>(std::min<uint64_t>(
        options_.bytes / slotBytes, options_.experts));
  }
  slots = std::min(slots, options_.experts);
  if (!slots) throw std::invalid_argument("an expert arena holds no slot");
  const uint32_t floor =
      options_.floor ? options_.floor
                     : static_cast<uint32_t>(uint64_t{slots} * kExpertFloorShare);
  layout_ = streamedExpertChunkLayout(planes_, slots, floor, 1,
                                      options_.largestBuffer,
                                      options_.elasticChunkBytes
                                          ? options_.elasticChunkBytes
                                          : kExpertElasticChunkBytes);
  // One Shared buffer per (chunk, region) with bytes: the CPU writes staged
  // slices into it and the GPU reads them, so no upload command is needed.
  regions_.reserve(size_t{layout_.chunks} * kPackedExpertRegions);
  for (uint32_t c = 0; c < layout_.chunks; ++c) {
    const uint32_t held = layout_.held(c);
    for (uint32_t r = 0; r < kPackedExpertRegions; ++r) {
      const uint64_t bytes = uint64_t{held} * planes_[r / kPackedExpertPlanes][r % kPackedExpertPlanes];
      metal::MetalBuffer buffer;
      if (bytes) {
        buffer = backend.allocateBuffer(
            bytes, metal::BufferStorage::Shared,
            "expert-chunk-" + std::to_string(c) + "-region-" + std::to_string(r));
        allocatedBytes_ += buffer.allocatedBytes();
      }
      regions_.push_back(std::move(buffer));
    }
  }
  // The chunk address table: gpuAddress() of every region, once — the
  // buffers are never released or moved.
  addresses_.assign(size_t{layout_.chunks} * kExpertChunkAddresses, 0);
  for (uint32_t c = 0; c < layout_.chunks; ++c)
    for (uint32_t z = 0; z < kPackedExpertProjections; ++z)
      for (uint32_t p = 0; p < kPackedExpertPlanes; ++p) {
        const uint32_t r = z * kPackedExpertPlanes + p;
        const metal::MetalBuffer &buffer = regions_[size_t{c} * kPackedExpertRegions + r];
        uint64_t address = 0;
        if (buffer)
          address = buffer.gpuAddress();
        else if (p == 1)
          // A format without plane1 holds its meta region's address there,
          // the staged convention (runtime/qwen4/metal/abi/MoE.h).
          address = regions_[size_t{c} * kPackedExpertRegions +
                             z * kPackedExpertPlanes + 2]
                        .gpuAddress();
        addresses_[size_t{c} * kExpertChunkAddresses + r] = address;
      }
  chunkAddresses_ = backend.allocateBuffer(
      addresses_.size() * sizeof(uint64_t), metal::BufferStorage::Shared,
      "expert-chunk-addresses");
  allocatedBytes_ += chunkAddresses_.allocatedBytes();
  std::memcpy(chunkAddresses_.contents(), addresses_.data(),
              addresses_.size() * sizeof(uint64_t));
  slotTable_ = backend.allocateBuffer(size_t{table_.experts} * sizeof(uint32_t),
                                      metal::BufferStorage::Shared,
                                      "expert-slot-table");
  allocatedBytes_ += slotTable_.allocatedBytes();
  auto *slotsOut = static_cast<uint32_t *>(slotTable_.contents());
  std::fill(slotsOut, slotsOut + table_.experts, kNoSlot);
  recycler_.emplace(layout_.slots);
  slotOfExpert_.assign(table_.experts, kNoSlot);
  expertOfSlot_.assign(layout_.slots, kNoSlot);
}

const metal::MetalBuffer &ExpertArena::region(uint32_t chunk, uint32_t z,
                                              uint32_t p) const {
  return regions_[(size_t{chunk} * kPackedExpertRegions) +
                  z * kPackedExpertPlanes + p];
}

uint32_t ExpertArena::slotOf(uint32_t expert) const {
  const std::lock_guard lock(mutex_);
  return expert < slotOfExpert_.size() ? slotOfExpert_[expert] : kNoSlot;
}

std::vector<uint32_t> ExpertArena::stage(PackedExpertStore &store,
                                         std::span<const uint32_t> ids,
                                         uint64_t load,
                                         PackedExpertUse use) {
  const std::lock_guard lock(mutex_);
  std::vector<uint32_t> slots(ids.size(), kNoSlot);
  std::vector<uint32_t> missing;
  std::vector<uint32_t> missingSlots;
  std::unordered_map<uint32_t, uint32_t> claimed;
  missing.reserve(ids.size());
  claimed.reserve(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint32_t expert = ids[i];
    if (expert >= table_.experts)
      throw std::invalid_argument("an expert id is past the arena's table");
    const uint32_t held = slotOfExpert_[expert];
    if (held != kNoSlot) {
      recycler_->touch(held, load, false);
      slots[i] = held;
      continue;
    }
    if (const auto it = claimed.find(expert); it != claimed.end()) {
      slots[i] = it->second;
      continue;
    }
    uint32_t fresh;
    try {
      fresh = recycler_->claim(expert, load, false);
    } catch (const std::exception &) {
      throw PackedExpertError(
          "the expert arena cannot hold the step's distinct experts");
    }
    // The claim may have evicted the slot's previous expert: it is no
    // longer resident.
    const uint32_t evicted = expertOfSlot_[fresh];
    if (evicted != kNoSlot) slotOfExpert_[evicted] = kNoSlot;
    expertOfSlot_[fresh] = expert;
    claimed.emplace(expert, fresh);
    missing.push_back(expert);
    missingSlots.push_back(fresh);
    slots[i] = fresh;
  }
  if (missing.empty()) return slots;
  PackedExpertSlices slices;
  try {
    slices = store.stage(missing, use);
  } catch (...) {
    // A failed load claims nothing: the slots go back to the recycler.
    for (size_t j = 0; j < missing.size(); ++j) {
      recycler_->free(missingSlots[j]);
      expertOfSlot_[missingSlots[j]] = kNoSlot;
      slotOfExpert_[missing[j]] = kNoSlot;
    }
    throw;
  }
  // Copy every slice into its slot's region: chunk c, local slot l, region
  // r's bytes at l * planes[z][p] of the region's buffer.
  size_t copied = 0;
  try {
    for (; copied < missing.size(); ++copied) {
      const uint32_t expert = missing[copied];
      const uint32_t slot = missingSlots[copied];
      const uint32_t chunk = layout_.chunkOf(slot);
      const uint32_t local = slot - layout_.first(chunk);
      for (uint32_t r = 0; r < kPackedExpertRegions; ++r) {
        const uint64_t sliceBytes = table_.regions[r].sliceBytes;
        if (!sliceBytes) continue;
        const uint32_t z = r / kPackedExpertPlanes;
        const uint32_t p = r % kPackedExpertPlanes;
        metal::MetalBuffer &buffer =
            regions_[size_t{chunk} * kPackedExpertRegions + r];
        if (!buffer)
          throw PackedExpertError("an expert arena region is not allocated");
        uint8_t *dst = static_cast<uint8_t *>(buffer.contents()) +
                       uint64_t{local} * planes_[z][p];
        std::memcpy(dst, slices[copied][r], sliceBytes);
      }
      slotOfExpert_[expert] = slot;
      recycler_->touch(slot, load, false);
    }
  } catch (...) {
    // Experts whose copies completed stay resident; the rest give their
    // slots back.
    for (size_t j = copied; j < missing.size(); ++j) {
      if (slotOfExpert_[missing[j]] == kNoSlot) {
        recycler_->free(missingSlots[j]);
        expertOfSlot_[missingSlots[j]] = kNoSlot;
      }
      slotOfExpert_[missing[j]] = kNoSlot;
    }
    throw;
  }
  // Publish the slot table: experts entries of the global slot or kNoSlot.
  auto *table = static_cast<uint32_t *>(slotTable_.contents());
  for (uint32_t expert = 0; expert < table_.experts; ++expert)
    table[expert] = slotOfExpert_[expert];
  return slots;
}

} // namespace richengine::model

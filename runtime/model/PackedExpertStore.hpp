#pragma once

// The routed experts of one packed MoE block, kept in their source file and
// read an expert at a time: a bounded host cache of expert slices, read on
// the engine's SSD readers (model::SsdReaders), which model::ExpertArena
// copies into its device slots. The production port of the staged qwen4
// reference's host side (runtime/qwen4/model/ExpertStore.cpp); the packed
// Q8_0 slabs need no repacking — an expert's plane slice in the file is
// already the byte range the expert kernels read, so a slice copies
// verbatim into a slot.
//
// A block's routed gate, up and down segments each hold `experts` experts
// of outputSize / experts rows: plane0, plane1 (none for Q8_0) and meta,
// each one contiguous file range per expert (the packed layout's
// expert-major slabs). The store knows the nine (projection, plane) regions
// of a block; stage() makes every region's slice of every requested expert
// CPU-resident and returns pointers into the cache, valid until the next
// stage() or prefetch() that evicts them — the arena copies them before
// either runs.
//
// A slice enters the cache only after the source file was checked unchanged
// after it was read (RowReader). A stage's failed read, a short file, or a
// source written since it was opened poisons the store: that stage and every
// later one throw PackedExpertError. A prefetch is a hint: a slice it fails
// to read is dropped, and the stage that needs it reads it again.

#include "model/SsdReaders.hpp"
#include "model/SparseRowStore.hpp" // RowReader, openRowReader
#include "ops/Weights.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace richengine::model {

class WeightSource;

class PackedExpertError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// The nine (projection, plane) regions of a block's routed experts, in the
// kernels' order: gate, up, down, each plane0, plane1, meta. A plane a
// format does not have (Q8_0's plane1) is an empty region.
inline constexpr uint32_t kPackedExpertProjections = 3;
inline constexpr uint32_t kPackedExpertPlanes = 3;
inline constexpr uint32_t kPackedExpertRegions =
    kPackedExpertProjections * kPackedExpertPlanes;

// One region's file range and per-expert slice: expert e's bytes are
// [offset + e * sliceBytes, offset + (e + 1) * sliceBytes).
struct PackedExpertRegion final {
  uint64_t offset = 0;     // absolute file offset of expert 0's slice
  uint64_t sliceBytes = 0; // bytes per expert, contiguous
  uint64_t regionBytes = 0;// experts * sliceBytes
};

// A block's routed-expert file layout, derived from its three QuantizedSegments
// (gate, up, down routed) by packedExpertTable(): region (z * 3 + plane).
struct PackedExpertTable final {
  std::string name;
  uint32_t experts = 0;
  std::array<PackedExpertRegion, kPackedExpertRegions> regions{};
  // Bytes one expert occupies over all its regions.
  [[nodiscard]] uint64_t expertBytes() const noexcept;
  void validate() const;
};

// The table of a block whose routed segments are `gate`, `up`, `down`, each
// holding `experts` experts of QuantizedSegment::outputSize / experts rows.
// Throws std::invalid_argument when a segment is a float or its planes do
// not hold whole expert slices.
[[nodiscard]] PackedExpertTable
packedExpertTable(std::string name, const ops::QuantizedSegment &gate,
                  const ops::QuantizedSegment &up,
                  const ops::QuantizedSegment &down, uint32_t experts);

struct PackedExpertStoreOptions final {
  // The cache and working-memory reservation (committedBytes): the cached
  // slices with their index and recency links, plus read staging. Not a
  // bound on all host allocations.
  uint64_t cacheBytes = 0;
  // The readers its reads run on besides the calling thread: the engine's
  // one pool, which the other streamed weights and KV reads share.
  std::shared_ptr<SsdReaders> readers;
  // The most expert slices prefetches may have queued or in flight.
  uint32_t prefetchSlices = 0;
};

// What a stage or prefetch was for, which its statistics are kept by.
enum class PackedExpertUse : uint8_t { Prefill, Decode };
inline constexpr uint32_t kPackedExpertUses = 2;

struct PackedExpertStageResult final {
  uint32_t uniqueExperts = 0; // experts staged: distinct ids
  uint32_t hitExperts = 0;    // of them, wholly cached
  uint32_t readExperts = 0;   // of them, with slices read from the source
  uint64_t nanoseconds = 0;
};

struct PackedExpertStats final {
  std::string table;
  uint64_t tableBytes = 0;   // experts * expertBytes
  uint64_t cacheBytes = 0;   // cache and working-memory reservation
  uint64_t cachedSlices = 0;
  uint64_t cachedBytes = 0;
  struct Use final {
    uint64_t stages = 0;
    uint64_t requestedExperts = 0;
    uint64_t uniqueExperts = 0;
    uint64_t hitExperts = 0;
    uint64_t readExperts = 0;
  };
  Use uses[kPackedExpertUses];
  uint64_t readBytes = 0;
  uint64_t readRetries = 0;
  uint64_t readErrors = 0;
  uint64_t evictions = 0;
  uint64_t prefetchRequestedExperts = 0;
  uint64_t prefetchReadExperts = 0;
  uint64_t prefetchDroppedExperts = 0;
  bool poisoned = false;
  std::string failure;
};

// expertIds.size() slices of a stage, one per requested id in order; slice
// r of expert i is regions[r]'s bytes.
using PackedExpertSlices =
    std::vector<std::array<const uint8_t *, kPackedExpertRegions>>;

class PackedExpertStore final {
public:
  // Allocates the cache now. Throws std::invalid_argument when the table is
  // invalid, its regions end past the reader's file, the options name no
  // readers, or no expert fits in options.cacheBytes.
  PackedExpertStore(PackedExpertTable table, std::unique_ptr<RowReader> reader,
                    PackedExpertStoreOptions options);
  ~PackedExpertStore();
  PackedExpertStore(const PackedExpertStore &) = delete;
  PackedExpertStore &operator=(const PackedExpertStore &) = delete;

  [[nodiscard]] const PackedExpertTable &table() const noexcept;
  // Cache bytes plus the fixed working-memory reservation, at most
  // options.cacheBytes; not the store's measured allocation high-water mark.
  [[nodiscard]] uint64_t committedBytes() const noexcept;

  // Makes every region's slice of every distinct id of `ids` (any order,
  // repeats allowed) CPU-resident and returns slices[i][r] for ids[i]'s
  // region r, valid until the next stage() or prefetch(). Blocks until every
  // slice is in place. Throws PackedExpertError when a slice cannot be read
  // (and poisons the store) or an id is past the table, std::invalid_argument
  // on misuse. Stages are serialized.
  PackedExpertSlices stage(std::span<const uint32_t> ids, PackedExpertUse use);

  // Starts reading the slices of `ids` that are not cached into the cache,
  // in the background. Returns at once. Slices beyond the bound of queued
  // slices, and slices it fails to read, are dropped and counted. Ids past
  // the table are ignored (a prefetch is only a hint). A source written
  // since it was opened poisons the store, which the next stage reports.
  void prefetch(std::span<const uint32_t> ids);

  // Finishes the prefetches queued or in flight: tests wait for prefetched
  // slices so.
  void drainPrefetch();

  [[nodiscard]] PackedExpertStats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace richengine::model

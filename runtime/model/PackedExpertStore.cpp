// PackedExpertStore.cpp: the bounded host cache of a block's routed experts.

#include "model/PackedExpertStore.hpp"

#include "AwakeClock.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"

#include <algorithm>
#include <chrono>
#include <list>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace richengine::model {
namespace {

using Clock = AwakeClock;

uint64_t nanosecondsSince(Clock::time_point start) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

// Cache key of (region, expert): regions are few, experts below 2^32.
uint64_t sliceKey(uint32_t region, uint32_t expert) noexcept {
  return (uint64_t{region} << 32) | expert;
}

struct CacheEntry {
  std::unique_ptr<uint8_t[]> bytes;
  std::list<uint64_t>::iterator recency;
};

// A slice read in flight: its destination and its cache key.
struct ReadJob {
  uint32_t region;
  uint32_t expert;
  std::unique_ptr<uint8_t[]> bytes;
};

} // namespace

uint64_t PackedExpertTable::expertBytes() const noexcept {
  uint64_t bytes = 0;
  for (const PackedExpertRegion &region : regions) bytes += region.sliceBytes;
  return bytes;
}

void PackedExpertTable::validate() const {
  if (!experts) throw std::invalid_argument("a packed expert table has no experts");
  bool any = false;
  for (const PackedExpertRegion &region : regions) {
    if (!region.sliceBytes) continue;
    any = true;
    if (region.regionBytes != uint64_t{experts} * region.sliceBytes)
      throw std::invalid_argument("a packed expert region does not hold its experts' slices");
  }
  if (!any) throw std::invalid_argument("a packed expert table has no regions");
}

PackedExpertTable
packedExpertTable(std::string name, const ops::QuantizedSegment &gate,
                  const ops::QuantizedSegment &up,
                  const ops::QuantizedSegment &down, uint32_t experts) {
  if (!experts) throw std::invalid_argument("a packed expert table has no experts");
  if (gate.isFloat() || up.isFloat() || down.isFloat())
    throw std::invalid_argument("routed expert segments must be quantized");
  if (!gate.outputSize || !up.outputSize || !down.outputSize ||
      gate.outputSize % experts || up.outputSize % experts ||
      down.outputSize % experts)
    throw std::invalid_argument("routed expert segments do not hold whole experts");
  PackedExpertTable table;
  table.name = std::move(name);
  table.experts = experts;
  const ops::QuantizedSegment *segments[3] = {&gate, &up, &down};
  for (uint32_t z = 0; z < kPackedExpertProjections; ++z) {
    const ops::QuantizedSegment &segment = *segments[z];
    const QuantFormat &format = segment.format();
    const uint64_t rows = segment.outputSize / experts;
    const uint64_t groups = segment.inputSize / 32;
    if (segment.inputSize % 32 || groups % format.meta_groups)
      throw std::invalid_argument("routed expert segment inputs are not whole quant groups");
    const uint64_t slices[3] = {
        rows * groups * format.plane0_bytes,
        rows * groups * format.plane1_bytes,
        rows * (groups / format.meta_groups) * format.meta_bytes,
    };
    const metal::MetalBuffer *planes[3] = {&segment.plane0, &segment.plane1,
                                            &segment.meta};
    for (uint32_t p = 0; p < kPackedExpertPlanes; ++p) {
      PackedExpertRegion &region = table.regions[z * kPackedExpertPlanes + p];
      region.sliceBytes = slices[p];
      region.regionBytes = region.sliceBytes * experts;
      if (!region.sliceBytes) continue;
      const metal::MetalBuffer &plane = *planes[p];
      if (!plane || plane.sizeBytes() < region.regionBytes)
        throw std::invalid_argument("a routed expert plane does not hold its experts' slices");
      region.offset = plane.offsetBytes();
    }
  }
  table.validate();
  return table;
}

struct PackedExpertStore::Impl {
  Impl(PackedExpertTable table, std::unique_ptr<RowReader> reader,
       PackedExpertStoreOptions options)
      : table(std::move(table)), reader(std::move(reader)),
        options(std::move(options)) {
    this->table.validate();
    if (!this->options.readers)
      throw std::invalid_argument("a packed expert store needs SSD readers");
    // Every slice must fit the cache, and every region must end inside the
    // file the metadata came from.
    const uint64_t fileBytes = this->reader->size();
    for (const PackedExpertRegion &region : this->table.regions) {
      if (!region.sliceBytes) continue;
      if (region.sliceBytes > this->options.cacheBytes)
        throw std::invalid_argument("a packed expert slice is larger than the cache");
      if (region.offset > fileBytes || region.regionBytes > fileBytes - region.offset)
        throw std::invalid_argument("a packed expert region ends past its file");
    }
    stats.table = this->table.name;
    stats.tableBytes = uint64_t{this->table.experts} * this->table.expertBytes();
    stats.cacheBytes = this->options.cacheBytes;
  }

  PackedExpertTable table;
  std::unique_ptr<RowReader> reader;
  PackedExpertStoreOptions options;

  std::mutex mutex;
  // The cache: key -> entry, and the recency list, most recent first.
  std::unordered_map<uint64_t, CacheEntry> cache;
  std::list<uint64_t> recency;
  uint64_t cachedBytes_ = 0;
  uint64_t load_ = 0;

  // The prefetch in flight, if any; its jobs stay alive with it.
  std::shared_ptr<SsdReaders::Batch> prefetchBatch;
  std::mutex prefetchMutex;

  PackedExpertStats stats;
  bool poisoned = false;
  std::string failure;

  // Whether every region's slice of `expert` is cached; touches the hits.
  bool cached(uint32_t expert, bool touch) {
    for (uint32_t r = 0; r < kPackedExpertRegions; ++r) {
      const PackedExpertRegion &region = table.regions[r];
      if (!region.sliceBytes) continue;
      const uint64_t key = sliceKey(r, expert);
      auto it = cache.find(key);
      if (it == cache.end()) return false;
      if (touch) {
        recency.erase(it->second.recency);
        recency.push_front(key);
        it->second.recency = recency.begin();
      }
    }
    return true;
  }

  void evictOne() {
    // The least recently used slice goes; entries of one expert evict
    // independently, so a hot expert's planes stay together by recency.
    const uint64_t key = recency.back();
    recency.pop_back();
    auto it = cache.find(key);
    if (it != cache.end()) {
      cachedBytes_ -= table.regions[uint32_t(key >> 32)].sliceBytes;
      cache.erase(it);
      ++stats.evictions;
    }
  }

  // Inserts a read slice into the cache, evicting as a stage would.
  void insert(uint32_t region, uint32_t expert, std::unique_ptr<uint8_t[]> bytes) {
    const uint64_t key = sliceKey(region, expert);
    if (cache.find(key) != cache.end()) return;
    const uint64_t sliceBytes = table.regions[region].sliceBytes;
    while (cachedBytes_ + sliceBytes > options.cacheBytes && !recency.empty()) evictOne();
    if (cachedBytes_ + sliceBytes > options.cacheBytes)
      throw PackedExpertError("a packed expert slice is larger than the cache");
    recency.push_front(key);
    cache.emplace(key, CacheEntry{std::move(bytes), recency.begin()});
    cachedBytes_ += sliceBytes;
    stats.readBytes += sliceBytes;
    stats.cachedSlices = cache.size();
    stats.cachedBytes = cachedBytes_;
  }
};

PackedExpertStore::PackedExpertStore(PackedExpertTable table,
                                     std::unique_ptr<RowReader> reader,
                                     PackedExpertStoreOptions options)
    : impl_(std::make_unique<Impl>(std::move(table), std::move(reader),
                                   std::move(options))) {}

PackedExpertStore::~PackedExpertStore() {
  std::shared_ptr<SsdReaders::Batch> batch;
  {
    const std::lock_guard lock(impl_->prefetchMutex);
    batch = impl_->prefetchBatch;
    impl_->prefetchBatch.reset();
  }
  // A read in progress finishes first: its tasks still name this store.
  if (batch) impl_->options.readers->finish(*batch);
}

const PackedExpertTable &PackedExpertStore::table() const noexcept { return impl_->table; }

uint64_t PackedExpertStore::committedBytes() const noexcept {
  return impl_->options.cacheBytes;
}

PackedExpertSlices PackedExpertStore::stage(std::span<const uint32_t> ids,
                                            PackedExpertUse use) {
  const auto start = Clock::now();
  Impl &s = *impl_;
  PackedExpertSlices result(ids.size());
  PackedExpertStageResult stage;
  {
    const std::lock_guard lock(s.mutex);
    if (s.poisoned) throw PackedExpertError(s.failure);
    // Distinct ids, first-seen order.
    std::vector<uint32_t> unique;
    unique.reserve(ids.size());
    std::unordered_map<uint32_t, uint32_t> slotOf;
    for (uint32_t id : ids) {
      if (id >= s.table.experts)
        throw std::invalid_argument("a packed expert id is past the table");
      if (!slotOf.count(id)) {
        slotOf.emplace(id, static_cast<uint32_t>(unique.size()));
        unique.push_back(id);
      }
    }
    stage.uniqueExperts = static_cast<uint32_t>(unique.size());
    // Which experts need which slices.
    std::vector<uint32_t> missing;
    for (uint32_t expert : unique) {
      const bool hit = s.cached(expert, true);
      stage.hitExperts += hit;
      stage.readExperts += !hit;
      if (!hit) missing.push_back(expert);
    }
    // The slices to read, in runs bounded so a stage's read buffers stay a
    // fraction of the cache beside it. flush() runs with s.mutex held: the
    // batch's tasks only read into their own job buffers and never take it,
    // and the inserts run after finish() on this thread.
    std::vector<ReadJob> pending;
    const auto flush = [&]() {
      if (pending.empty()) return;
      const auto jobs = std::make_shared<std::vector<ReadJob>>(std::move(pending));
      pending.clear();
      const std::shared_ptr<SsdReaders::Batch> batch =
          s.options.readers->runBatch(jobs->size(), [&](size_t i) {
            const ReadJob &job = (*jobs)[i];
            const PackedExpertRegion &region = s.table.regions[job.region];
            s.reader->read(region.offset + uint64_t{job.expert} * region.sliceBytes,
                           std::span<uint8_t>(job.bytes.get(), region.sliceBytes));
          });
      s.options.readers->finish(*batch);
      if (const std::exception_ptr failure = batch->failure())
        std::rethrow_exception(failure);
      for (ReadJob &job : *jobs)
        s.insert(job.region, job.expert, std::move(job.bytes));
    };
    try {
      s.reader->checkUnchanged();
      uint64_t runBytes = 0;
      const uint64_t runBound = std::max<uint64_t>(s.options.cacheBytes / 2, 64 << 20);
      for (uint32_t expert : missing) {
        for (uint32_t r = 0; r < kPackedExpertRegions; ++r) {
          const PackedExpertRegion &region = s.table.regions[r];
          if (!region.sliceBytes) continue;
          const uint64_t key = sliceKey(r, expert);
          if (s.cache.find(key) != s.cache.end()) continue;
          if (runBytes && runBytes + region.sliceBytes > runBound) {
            flush();
            runBytes = 0;
          }
          ReadJob job{r, expert,
                      std::make_unique<uint8_t[]>(region.sliceBytes)};
          runBytes += region.sliceBytes;
          pending.push_back(std::move(job));
        }
      }
      flush();
      s.reader->checkUnchanged();
    } catch (...) {
      s.poisoned = true;
      s.failure = "a packed expert read failed";
      s.stats.poisoned = true;
      s.stats.failure = s.failure;
      throw PackedExpertError(s.failure);
    }
    // Answer in request order; repeats share the cached slices.
    for (size_t i = 0; i < ids.size(); ++i) {
      const uint32_t expert = ids[i];
      for (uint32_t r = 0; r < kPackedExpertRegions; ++r) {
        const PackedExpertRegion &region = s.table.regions[r];
        result[i][r] = nullptr;
        if (!region.sliceBytes) continue;
        auto it = s.cache.find(sliceKey(r, expert));
        if (it == s.cache.end())
          throw PackedExpertError("a staged packed expert slice is not cached");
        result[i][r] = it->second.bytes.get();
      }
    }
    PackedExpertStats::Use &useStats =
        s.stats.uses[static_cast<uint8_t>(use)];
    ++useStats.stages;
    useStats.requestedExperts += ids.size();
    useStats.uniqueExperts += stage.uniqueExperts;
    useStats.hitExperts += stage.hitExperts;
    useStats.readExperts += stage.readExperts;
  }
  stage.nanoseconds = nanosecondsSince(start);
  return result;
}

void PackedExpertStore::prefetch(std::span<const uint32_t> ids) {
  Impl &s = *impl_;
  {
    const std::lock_guard lock(s.mutex);
    if (s.poisoned) return; // the next stage reports it
  }
  // Distinct valid ids not wholly cached, then their missing slices, bounded
  // by prefetchSlices.
  struct Job { uint32_t region; uint32_t expert; };
  std::vector<Job> jobs;
  {
    const std::lock_guard lock(s.mutex);
    std::unordered_set<uint32_t> seen;
    const uint64_t bound = s.options.prefetchSlices;
    for (uint32_t id : ids) {
      if (id >= s.table.experts || !seen.insert(id).second) continue;
      if (s.cached(id, false)) continue;
      s.stats.prefetchRequestedExperts += 1;
      for (uint32_t r = 0; r < kPackedExpertRegions; ++r) {
        const PackedExpertRegion &region = s.table.regions[r];
        if (!region.sliceBytes || region.sliceBytes > s.options.cacheBytes)
          continue;
        if (s.cache.find(sliceKey(r, id)) == s.cache.end())
          jobs.push_back({r, id});
      }
      if (bound && jobs.size() >= bound) break;
    }
  }
  if (jobs.empty()) return;
  const auto held = std::make_shared<std::vector<Job>>(std::move(jobs));
  std::shared_ptr<SsdReaders::Batch> batch;
  try {
    batch = s.options.readers->start(
        held->size(),
        [this, held](size_t i) {
          Impl &s = *impl_;
          const Job &job = (*held)[i];
          const PackedExpertRegion &region = s.table.regions[job.region];
          auto bytes = std::make_unique<uint8_t[]>(region.sliceBytes);
          try {
            s.reader->read(region.offset + uint64_t{job.expert} * region.sliceBytes,
                           std::span<uint8_t>(bytes.get(), region.sliceBytes));
          } catch (...) {
            // A prefetch is a hint: a failed slice is dropped, counted.
            const std::lock_guard lock(s.mutex);
            ++s.stats.prefetchDroppedExperts;
            return;
          }
          const std::lock_guard lock(s.mutex);
          s.insert(job.region, job.expert, std::move(bytes));
          ++s.stats.prefetchReadExperts;
        },
        std::min<unsigned>(s.options.readers->readers(), 8));
  } catch (...) {
    return; // a prefetch that cannot start is no failure
  }
  const std::lock_guard lock(s.prefetchMutex);
  if (s.prefetchBatch) s.options.readers->finish(*s.prefetchBatch);
  s.prefetchBatch = std::move(batch);
}

void PackedExpertStore::drainPrefetch() {
  Impl &s = *impl_;
  std::shared_ptr<SsdReaders::Batch> batch;
  {
    const std::lock_guard lock(s.prefetchMutex);
    batch = s.prefetchBatch;
    s.prefetchBatch.reset();
  }
  if (batch) s.options.readers->finish(*batch);
}

PackedExpertStats PackedExpertStore::stats() const {
  const std::lock_guard lock(impl_->mutex);
  return impl_->stats;
}

} // namespace richengine::model

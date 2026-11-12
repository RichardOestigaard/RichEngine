// The order a store's slots are recycled in (model::ExpertSlotRecycler): a load
// never takes a slot of a key it uses, and takes none when every slot is its
// own; free slots are taken first, a freed one before them; a slot's key is
// found until it is recycled, freed or resized away; a long prefill chunk's
// scan of more experts than the slots leaves the experts decode steps routed
// again in place; and when more lanes add experts to each cycle until a cycle
// routes more than the slots hold, the experts every cycle routes keep
// hitting (recycling the least recently used slot keeps a fifth of them). A
// claim allocates nothing. A compaction to fewer slots evicts the keys ARC
// would recycle, T1's past its target and T2's at it, without a ghost, until
// the keys past the slots fit the free slots below them, moves those keys
// there in their lists' order, keeps as many keys as fit and the ghosts
// within ARC's bounds, and leaves no key or claim past the slots; one straight
// to fewer slots keeps the keys compactions a chunk at a time keep, moving
// each once at most. On the
// decode selections of real requests
// (dev/tests/fixtures/kv-selection-traces, the directory the test takes), the
// hot block cache's resolves miss at most 2% more than recycling the least
// recently used block would.
#include "TestChecks.hpp"
#include "model/ExpertSlotRecycler.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Every allocation of the test's thread, which a claim must not add to.
thread_local uint64_t allocations = 0;
void *operator new(size_t size) {
  ++allocations;
  if (void *memory = std::malloc(size ? size : 1)) return memory;
  throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }

using richengine::model::ExpertSlotRecycler;
using richengine::test::rejects;
using richengine::test::require;

namespace {

// The slots of a store's experts, as the store keeps them: a load routes
// experts, those resident are touched, the others claim slots.
struct Arena final {
  explicit Arena(uint32_t slots) : order(slots), owner(slots, kFree) {}
  static constexpr uint64_t kFree = UINT64_MAX;
  ExpertSlotRecycler order;
  std::map<uint64_t, uint32_t> slotOf;
  std::vector<uint64_t> owner;
  uint64_t load = 0;
  // How often compactions moved each key.
  std::map<uint64_t, uint32_t> moves;
  // Routes `experts` in one load; returns how many were resident.
  uint32_t route(const std::vector<uint64_t> &experts, bool chunk) {
    ++load;
    uint32_t hits = 0;
    std::vector<uint64_t> missing;
    for (const uint64_t expert : experts)
      if (const auto found = slotOf.find(expert); found != slotOf.end()) {
        order.touch(found->second, load, chunk);
        ++hits;
      } else {
        missing.push_back(expert);
      }
    for (const uint64_t expert : missing) {
      const uint32_t slot = order.claim(expert, load, chunk);
      if (owner[slot] != kFree) slotOf.erase(owner[slot]);
      owner[slot] = expert;
      slotOf[expert] = slot;
    }
    return hits;
  }
  // Compacts to `slots` slots, as the store follows the evictions and moves.
  void compact(uint32_t slots) {
    order.compact(
        slots,
        [&](uint32_t slot) {
          require(slot < owner.size() && owner[slot] != kFree, "a compaction evicted a free slot");
          slotOf.erase(owner[slot]);
          owner[slot] = kFree;
        },
        [&](uint32_t from, uint32_t to) {
          require(from >= slots && to < slots && owner[from] != kFree && owner[to] == kFree,
                  "a compaction moved a key from below the slots or into a held slot");
          ++moves[owner[from]];
          slotOf[owner[from]] = to;
          owner[to] = std::exchange(owner[from], kFree);
        });
    for (uint32_t slot = slots; slot < owner.size(); ++slot)
      require(owner[slot] == kFree, "a compaction left a key past the slots");
    owner.resize(slots);
    for (const auto &[key, slot] : slotOf) require(order.find(key) == slot, "a compacted key is not where it moved");
  }
};

void neverTakesTheLoadsOwn() {
  ExpertSlotRecycler slots(8);
  std::set<uint32_t> taken;
  for (uint64_t expert = 0; expert < 8; ++expert) taken.insert(slots.claim(expert, 1, false));
  require(taken.size() == 8, "one load took a slot twice");
  rejects([&] { static_cast<void>(slots.claim(8, 1, false)); }, "fewer slots", "a load took a slot of its own");
  // The next load takes a slot of an expert it does not route.
  for (uint32_t slot = 0; slot < 8; ++slot)
    if (slot % 2) slots.touch(slot, 2, false);
  for (uint32_t slot = 0; slot < 8; ++slot)
    require(slots.pinned(slot, 2) == (slot % 2 == 1), "a touch did not pin the slot to its load");
  for (uint64_t expert = 9; expert < 13; ++expert)
    require(slots.claim(expert, 2, false) % 2 == 0, "a load took a slot of an expert it routes");
  rejects([&] { static_cast<void>(slots.claim(13, 2, false)); }, "fewer slots", "a load took a slot of its own");
}

void freeSlotsFirst() {
  ExpertSlotRecycler slots(4);
  for (uint64_t load = 1; load <= 4; ++load) static_cast<void>(slots.claim(load, load, false));
  slots.free(2);
  require(slots.claim(5, 5, false) == 2, "a freed slot was not taken first");
  slots.reset();
  std::set<uint32_t> taken;
  for (uint64_t expert = 0; expert < 4; ++expert) taken.insert(slots.claim(expert, 6, false));
  require(taken.size() == 4, "a reset left a slot held");
}

void findsWhatItHolds() {
  ExpertSlotRecycler slots(2);
  const uint32_t first = slots.claim(10, 1, false);
  slots.touch(first, 2, false);
  const uint32_t second = slots.claim(11, 3, false);
  require(slots.find(10) == first && slots.find(11) == second && !slots.find(12), "a held key was not found");
  // T1's key goes for a third, as a ghost, whose claim grows T1 and takes
  // T2's slot.
  require(slots.claim(12, 4, false) == second && !slots.find(11) && slots.find(12) == second,
          "a recycled slot's key was still found");
  require(slots.claim(11, 5, false) == first && !slots.find(10) && slots.ghostHits() == 1,
          "a ghost's claim was not counted");
  slots.free(first);
  require(!slots.find(11), "a freed slot's key was still found");
  // A scan's claim of T2's ghost adapts nothing.
  static_cast<void>(slots.claim(10, 6, true));
  require(slots.ghostHits() == 1, "a scan's claim of a ghost counted as a ghost's");
}

void resizes() {
  ExpertSlotRecycler slots(4);
  for (uint64_t key = 0; key < 4; ++key) static_cast<void>(slots.claim(key, 1, false));
  // More slots: the load that holds every slot takes the new ones, and the
  // old keys stay.
  slots.resize(8);
  std::set<uint32_t> added;
  for (uint64_t key = 4; key < 8; ++key) added.insert(slots.claim(key, 1, false));
  require(added == std::set<uint32_t>{4, 5, 6, 7}, "a grown recycler did not take its new slots");
  for (uint64_t key = 0; key < 8; ++key) require(slots.find(key).has_value(), "growing lost a key");
  // Ghosts survive growth: with keys 0-3 frequent, 8 and 9 recycle 4 and 5,
  // and 4 comes back as a ghost.
  for (uint64_t key = 0; key < 4; ++key) slots.touch(*slots.find(key), 2, false);
  static_cast<void>(slots.claim(8, 3, false));
  static_cast<void>(slots.claim(9, 4, false));
  require(!slots.find(4) && !slots.find(5), "T1's oldest keys were not recycled");
  slots.resize(12);
  static_cast<void>(slots.claim(4, 5, false));
  require(slots.ghostHits() == 1, "growing forgot a ghost");
  // Fewer slots: the keys of the slots past them go, and the ghosts within
  // the slots stay: 5, which T1 gave up before.
  std::vector<uint64_t> kept, dropped;
  for (uint64_t key = 0; key < 10; ++key)
    if (const auto slot = slots.find(key)) (*slot < 5 ? kept : dropped).push_back(key);
  slots.resize(5);
  require(slots.size() == 5 && !dropped.empty(), "a shrunk recycler kept its slots");
  for (const uint64_t key : kept) require(slots.find(key).has_value(), "shrinking lost a key it kept");
  for (const uint64_t key : dropped) require(!slots.find(key), "shrinking kept a key past its slots");
  static_cast<void>(slots.claim(5, 6, false));
  require(slots.ghostHits() == 2, "shrinking forgot a ghost within the slots");
  for (uint64_t load = 7; load < 40; ++load)
    require(slots.claim(100 + load, load, false) < 5, "a shrunk recycler took a slot past its size");
}

// A compaction evicts REPLACE's victims: T1's oldest keys while T1 holds
// more than its target, and moves the keys past the slots into the lowest
// free slots below them, where they keep their places in T2's order, which
// the claims after it recycle.
void compactsInRecencyOrder() {
  Arena arena(8);
  std::vector<uint64_t> keys{0, 1, 2, 3, 4, 5, 6, 7};
  static_cast<void>(arena.route(keys, false));
  for (uint64_t key = 0; key < 8; ++key) require(arena.slotOf.at(key) == key, "a fresh recycler's claims moved");
  // T2 holds 7, 2, 5, 0 and 6 in that order, T1 1, 3 and 4.
  for (const uint64_t key : {7, 2, 5, 0, 6}) arena.order.touch(arena.slotOf.at(key), 2, false);
  std::vector<uint32_t> evicted;
  std::vector<std::pair<uint32_t, uint32_t>> moved;
  arena.order.compact(
      5, [&](uint32_t slot) { evicted.push_back(slot); },
      [&](uint32_t from, uint32_t to) { moved.emplace_back(from, to); });
  require(evicted == std::vector<uint32_t>{1, 3, 4}, "a compaction evicted other keys than T1's oldest");
  require(moved == std::vector<std::pair<uint32_t, uint32_t>>{{5, 1}, {6, 3}, {7, 4}},
          "a compaction moved the keys past the slots elsewhere than the lowest free slots");
  require(arena.order.size() == 5 && arena.order.find(0) == 0 && arena.order.find(2) == 2 &&
              arena.order.find(5) == 1 && arena.order.find(6) == 3 && arena.order.find(7) == 4 &&
              !arena.order.find(1) && !arena.order.find(3) && !arena.order.find(4),
          "a compacted key is not where it moved, or an evicted one is still found");
  // One load's claims recycle T2 in its order: 7, 2, 5, 0, 6.
  const std::vector<uint64_t> recycled{7, 2, 5, 0, 6};
  for (uint64_t claim = 0; claim < recycled.size(); ++claim) {
    static_cast<void>(arena.order.claim(100 + claim, 3, false));
    require(!arena.order.find(recycled[claim]), "a compaction changed T2's order: claim " + std::to_string(claim) +
                                                    " did not recycle key " + std::to_string(recycled[claim]));
  }
  for (uint64_t load = 4; load < 40; ++load)
    require(arena.order.claim(1000 + load, load, false) < 5, "a compacted recycler took a slot past its size");
}

// With T1 at its target, a compaction evicts T2's oldest keys, and a ghost
// T1 gave up stays one in the fewer slots.
void compactsAtTheTarget() {
  ExpertSlotRecycler slots(4);
  for (uint64_t key = 0; key < 4; ++key) require(slots.claim(key, 1, false) == key, "a fresh claim moved");
  slots.touch(0, 2, false);
  slots.touch(1, 2, false);
  // Key 4 recycles T1's 2, whose claim as a ghost grows T1's target to one
  // and recycles T1's 3 into a ghost: T1 holds 4, T2 0, 1 and 2.
  require(slots.claim(4, 3, false) == 2 && slots.claim(2, 4, false) == 3 && slots.ghostHits() == 1,
          "the claims before the compaction recycled other slots");
  std::vector<uint32_t> evicted;
  std::vector<std::pair<uint32_t, uint32_t>> moved;
  slots.compact(
      2, [&](uint32_t slot) { evicted.push_back(slot); },
      [&](uint32_t from, uint32_t to) { moved.emplace_back(from, to); });
  require(evicted == std::vector<uint32_t>{0, 1} &&
              moved == std::vector<std::pair<uint32_t, uint32_t>>{{2, 0}, {3, 1}},
          "a compaction with T1 at its target evicted other keys than T2's oldest");
  require(slots.find(4) == 0 && slots.find(2) == 1 && !slots.find(0) && !slots.find(1),
          "a compacted key is not where it moved");
  // 3 is claimed as the ghost T1 gave up, and the victims, which left none,
  // as new keys.
  static_cast<void>(slots.claim(3, 5, false));
  require(slots.ghostHits() == 2, "a compaction forgot a ghost within the slots");
  static_cast<void>(slots.claim(0, 6, false));
  static_cast<void>(slots.claim(1, 7, false));
  require(slots.ghostHits() == 2, "a compaction's victim left a ghost");
}

// Compactions of a recycler whose ghosts fill ARC's bounds, between loads
// of keys used again and again and others in turn, keep as many keys as fit,
// leave no key or claim past the slots and the ghosts within the bounds (a
// claim throws past them), which keep hitting; a compaction to no slot
// evicts every key, and the recycler grows again.
void compactionsKeepTheBounds() {
  Arena arena(64);
  uint64_t fresh = 0;
  const auto loads = [&](uint32_t count) {
    for (uint32_t load = 0; load < count; ++load) {
      std::vector<uint64_t> keys;
      for (uint64_t key = 0; key < arena.order.size() / 2; ++key) keys.push_back(key);
      // More others in turn than the slots the keys used again leave.
      for (uint64_t other = 0; other < arena.order.size() / 4; ++other)
        keys.push_back(1000 + fresh++ % arena.order.size());
      static_cast<void>(arena.route(keys, load % 7 == 3));
      if (load % 11 == 5 && !arena.slotOf.empty()) {
        const auto [key, slot] = *arena.slotOf.begin();
        arena.order.free(slot);
        arena.slotOf.erase(key);
        arena.owner[slot] = Arena::kFree;
      }
    }
  };
  for (const uint32_t slots : {48u, 47u, 20u, 9u, 1u}) {
    loads(60);
    const size_t keys = arena.slotOf.size();
    const uint64_t hits = arena.order.ghostHits();
    arena.compact(slots);
    require(arena.order.size() == slots && arena.slotOf.size() == std::min<size_t>(keys, slots),
            "a compaction to " + std::to_string(slots) + " slots kept " + std::to_string(arena.slotOf.size()) +
                " of " + std::to_string(keys) + " keys");
    loads(40);
    require(slots == 1 || arena.order.ghostHits() > hits,
            "the loads after a compaction to " + std::to_string(slots) + " slots hit no ghost");
    for (const auto &[key, slot] : arena.slotOf)
      require(slot < slots, "a claim after a compaction took a slot past its size");
  }
  arena.compact(0);
  require(arena.order.size() == 0 && arena.slotOf.empty(), "a compaction to no slot kept a key");
  arena.order.resize(16);
  arena.owner.assign(16, Arena::kFree);
  loads(20);
  require(!arena.slotOf.empty(), "a recycler compacted to no slot did not grow again");
  rejects([&] { arena.order.compact(17, [](uint32_t) {}, [](uint32_t, uint32_t) {}); }, "more slots",
          "a compaction to more slots than the recycler holds was taken");
}

// A compaction straight to fewer slots, as critical pressure takes the store
// down to its floor (model::ExpertStore::releaseToFloor), keeps the keys
// that compactions a chunk at a time down to the same slots keep, since
// moves keep the keys' places in their lists, and moves each key once at
// most, no more keys in all than those compactions move, whether the slots
// were full or partly free.
void compactsToTheFloorOnce() {
  for (const uint64_t held : {64u, 48u}) {
    Arena once(64), steps(64);
    for (Arena *arena : {&once, &steps}) {
      for (uint64_t key = 0; key < held; ++key) static_cast<void>(arena->route({key}, false));
      // Every third key routed again, into T2.
      std::vector<uint64_t> again;
      for (uint64_t key = 0; key < held; key += 3) again.push_back(key);
      static_cast<void>(arena->route(again, false));
    }
    once.compact(16);
    for (const uint32_t slots : {52u, 40u, 28u, 16u}) steps.compact(slots);
    const auto keys = [](const Arena &arena) {
      std::set<uint64_t> result;
      for (const auto &[key, slot] : arena.slotOf) result.insert(key);
      return result;
    };
    uint64_t movedOnce = 0, movedSteps = 0;
    for (const auto &[key, count] : once.moves) {
      require(count == 1, "a compaction straight to fewer slots moved key " + std::to_string(key) + " twice");
      movedOnce += count;
    }
    for (const auto &[key, count] : steps.moves) movedSteps += count;
    require(keys(once) == keys(steps) && once.slotOf.size() == 16,
            "a compaction straight to fewer slots kept other keys than compactions a chunk at a time");
    require(movedOnce <= movedSteps, "a compaction straight to fewer slots moved " + std::to_string(movedOnce) +
                                         " keys, more than compactions a chunk at a time, " +
                                         std::to_string(movedSteps));
  }
}

void aScanSparesTheFrequent() {
  Arena arena(6);
  const std::vector<uint64_t> frequent{0, 1, 2};
  static_cast<void>(arena.route(frequent, false));
  require(arena.route(frequent, false) == 3, "a decode step's experts were not resident for the next");
  // A prompt's chunks route forty experts, two a load.
  for (uint64_t expert = 100; expert < 140; expert += 2) static_cast<void>(arena.route({expert, expert + 1}, true));
  require(arena.route(frequent, false) == 3, "a long chunk's scan evicted experts decode steps route again");
}

void aRoutedCoreOutlivesMoreLanes() {
  // Six layers, each routing its ten core experts every cycle besides fresh
  // ones: two a cycle (one lane), then eight (more lanes), when a cycle
  // routes 108 experts and the arena holds 100.
  Arena arena(100);
  uint64_t fresh = 1000;
  double rate = 0;
  for (const uint32_t extra : {2u, 8u}) {
    uint64_t hits = 0, routes = 0;
    for (int cycle = 0; cycle < 30; ++cycle)
      for (uint64_t layer = 0; layer < 6; ++layer) {
        std::vector<uint64_t> experts;
        for (uint64_t core = 0; core < 10; ++core) experts.push_back(layer << 32 | core);
        for (uint32_t other = 0; other < extra; ++other) experts.push_back(layer << 32 | fresh++);
        const uint32_t resident = arena.route(experts, false);
        if (cycle >= 5) {
          hits += resident;
          routes += 10;
        }
      }
    rate = double(hits) / double(routes);
    require(rate > 0.95, "the experts every cycle routes stopped hitting: " + std::to_string(rate));
  }
}

void claimsAllocateNothing() {
  // Loads of 32 keys used again and again and 16 of 56 others in turn: the
  // 64 slots recycle the others, which come back as ghosts; every seventh
  // load is a scan, and every fiftieth frees a slot.
  std::vector<std::vector<uint64_t>> loads(400);
  for (size_t load = 0; load < loads.size(); ++load) {
    for (uint64_t key = 0; key < 32; ++key) loads[load].push_back(key);
    for (uint64_t other = 0; other < 16; ++other) loads[load].push_back(1000 + (load * 16 + other) % 56);
  }
  ExpertSlotRecycler slots(64);
  uint64_t before = 0;
  for (uint64_t load = 1; load <= loads.size(); ++load) {
    if (load == 100) before = allocations;
    const std::vector<uint64_t> &keys = loads[load - 1];
    const bool chunk = load % 7 == 0;
    for (const uint64_t key : keys)
      if (const auto slot = slots.find(key)) slots.touch(*slot, load, chunk);
      else static_cast<void>(slots.claim(key, load, chunk));
    if (load % 50 == 0) slots.free(*slots.find(keys.front()));
  }
  require(allocations == before, "a claim allocated: " + std::to_string(allocations - before) + " allocations");
  require(slots.ghostHits() > 0, "the loads hit no ghost");
}

// The decode selections of real requests: the blocks each attention layer
// of Qwen3.8-Flash-Next's decode selected, step by step, as a build that
// recorded every layer's block lists wrote them (not part of Splash), for a
// novel at 126K and 252K tokens and a source tree at 126K, two questions
// each; the committed traces keep layers 0 and 6 of each request's first 100
// steps. A trace (NAME.kvst) is the magic "KVS1", then unsigned LEB128
// varints: the requests and the layers; per request its steps; per step and
// layer the step's first position, its blocks' count and the blocks in
// ascending order, the first as it is and each other as its distance from
// the one before. Per request, its steps; per step and layer, its first
// position and blocks.
struct TraceStep final {
  uint32_t first = 0;
  std::vector<std::vector<uint32_t>> layers;
};
using Trace = std::vector<std::vector<TraceStep>>;

Trace readTrace(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  size_t at = 4;
  const auto varint = [&] {
    uint64_t value = 0;
    for (uint32_t shift = 0;; shift += 7) {
      require(at < bytes.size(), "a selection trace ends early: " + path.string());
      value |= uint64_t{bytes[at] & 0x7fu} << shift;
      if (!(bytes[at++] & 0x80)) return static_cast<uint32_t>(value);
    }
  };
  require(bytes.size() >= 4 && std::string(bytes.begin(), bytes.begin() + 4) == "KVS1",
          "not a selection trace: " + path.string());
  Trace trace(varint());
  const uint32_t layers = varint();
  for (std::vector<TraceStep> &request : trace) {
    request.resize(varint());
    for (TraceStep &step : request) {
      step.layers.resize(layers);
      for (std::vector<uint32_t> &blocks : step.layers) {
        step.first = varint();
        blocks.resize(varint());
        uint32_t block = 0;
        for (uint32_t &entry : blocks) entry = block += varint();
      }
    }
  }
  require(at == bytes.size(), "a selection trace has bytes past its steps: " + path.string());
  return trace;
}

// The misses of one layer of a request's decode as the hot block cache's
// resolves place its blocks in `slots` slots (model::KvBlockCache): those
// of the open page and after it are the page table's; a step touches the
// blocks it holds, then claims slots for the others, never one it touched;
// the first step, of an empty cache, does not count. `recency` recycles the
// least recently used slot instead.
uint64_t misses(const std::vector<TraceStep> &request, uint32_t layer, uint32_t slots, bool recency) {
  constexpr uint32_t kPageBlocks = 8, kPageTokens = 32;
  ExpertSlotRecycler order(slots);
  std::list<uint32_t> used;
  std::unordered_map<uint32_t, std::pair<std::list<uint32_t>::iterator, uint64_t>> lru;
  uint64_t total = 0;
  for (uint64_t load = 1; load <= request.size(); ++load) {
    const TraceStep &step = request[load - 1];
    const uint32_t paged = step.first / kPageTokens * kPageBlocks;
    std::vector<uint32_t> missing;
    for (const uint32_t block : step.layers[layer]) {
      if (block >= paged) continue;
      if (!recency) {
        if (const auto slot = order.find(block)) order.touch(*slot, load, false);
        else missing.push_back(block);
      } else if (const auto held = lru.find(block); held != lru.end()) {
        used.splice(used.end(), used, held->second.first);
        held->second.second = load;
      } else {
        missing.push_back(block);
      }
    }
    if (load > 1) total += missing.size();
    for (const uint32_t block : missing) {
      if (!recency) {
        static_cast<void>(order.claim(block, load, false));
        continue;
      }
      if (used.size() == slots) {
        auto oldest = used.begin();
        while (lru.at(*oldest).second == load) ++oldest;
        lru.erase(*oldest);
        used.erase(oldest);
      }
      lru[block] = {used.insert(used.end(), block), load};
    }
  }
  return total;
}

void adaptiveReplacementKeepsUpWithRecency(const std::filesystem::path &traces) {
  // One lane's floor, two lanes' selections a layer: the knee of the hot
  // block cache's hit rate, as measured on such traces.
  constexpr uint32_t kSlots = 8192;
  std::vector<std::filesystem::path> paths;
  for (const auto &entry : std::filesystem::directory_iterator(traces))
    if (entry.path().extension() == ".kvst") paths.push_back(entry.path());
  std::ranges::sort(paths);
  require(!paths.empty(), "no selection trace in " + traces.string());
  uint64_t adaptive = 0, recency = 0;
  for (const std::filesystem::path &path : paths) {
    const Trace trace = readTrace(path);
    uint64_t traceAdaptive = 0, traceRecency = 0;
    for (const std::vector<TraceStep> &request : trace)
      for (uint32_t layer = 0; layer < request.front().layers.size(); ++layer) {
        traceAdaptive += misses(request, layer, kSlots, false);
        traceRecency += misses(request, layer, kSlots, true);
      }
    std::cout << "expert_slot_recycler_test: " << path.filename().string() << ": " << traceAdaptive << " misses, "
              << traceRecency << " least recently used\n";
    adaptive += traceAdaptive;
    recency += traceRecency;
  }
  require(adaptive * 100 <= recency * 102, "adaptive replacement missed " + std::to_string(adaptive) +
                                               " blocks, recency " + std::to_string(recency));
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc > 2) throw std::invalid_argument("usage: slot_recycler_test [SELECTION_TRACES]");
    neverTakesTheLoadsOwn();
    freeSlotsFirst();
    findsWhatItHolds();
    resizes();
    compactsInRecencyOrder();
    compactsAtTheTarget();
    compactionsKeepTheBounds();
    compactsToTheFloorOnce();
    aScanSparesTheFrequent();
    aRoutedCoreOutlivesMoreLanes();
    claimsAllocateNothing();
    // The trace-driven comparison needs upstream's fixture directory
    // (dev/tests/fixtures/kv-selection-traces), which this tree does not
    // carry; with none given, the section is skipped.
    if (argc == 2)
      adaptiveReplacementKeepsUpWithRecency(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << "expert_slot_recycler_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "expert_slot_recycler_test: PASS\n";
  return 0;
}

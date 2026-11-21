// Cache hot-path micro-benchmarks. Not a test: prints ns/op for A/B runs.
#include "Checked.hpp"
#include "TestCache.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/cache/Cache.hpp"
#include "model/SlotFile.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace richengine;
using namespace richengine::engine;
using namespace std::chrono;

class BenchState final : public CompositeState {
public:
  explicit BenchState(uint64_t byteCount) : byteCount_(byteCount) {}
  [[nodiscard]] uint64_t bytes() const noexcept override { return byteCount_; }
private:
  uint64_t byteCount_;
};

static void bench(const char *name, uint32_t iterations, std::function<void()> fn) {
  fn(); // warm
  const auto start = steady_clock::now();
  for (uint32_t i = 0; i < iterations; ++i) fn();
  const double ns = duration_cast<nanoseconds>(steady_clock::now() - start).count();
  std::printf("%-40s %10.1f ns/op\n", name, ns / iterations);
}

// Publishes a full chain under one request: admits pages, then commits.
static void publishChain(Cache &cache, uint64_t requestId,
                         std::span<const uint32_t> tokens) {
  cache.beginRequest(requestId);
  if (!cache.ensureTokens(requestId, tokens.size()).granted())
    throw std::runtime_error("page budget too small");
  cache.publishCommittedBlocks(requestId, tokens, tokens.size(), {});
  cache.endRequest(requestId);
}

int main() {
  const uint32_t depth = 512; // blocks per chain
  const uint32_t pageTokens = KvCache::pageTokens;
  test::TestKvStorage storage(20000, 4096, 16);
  KvPool pool(storage, 19000);

  std::vector<uint32_t> tokens(size_t(depth) * pageTokens);
  for (uint32_t i = 0; i < tokens.size(); ++i) tokens[i] = i * 2654435761u;

  Cache cache(pool, nullptr, nullptr);
  publishChain(cache, 1, tokens);

  bench("probe+lookup (512-block chain)", 300, [&] {
    auto probe = cache.probe(tokens, {});
    auto lookup = cache.lookup(tokens, {}, &probe);
    (void)lookup;
  });

  {
    auto probe = cache.probe(tokens, {});
    uint64_t rid = 100;
    bench("probe.refresh (gen bump)", 100, [&] {
      ++rid;
      std::vector<uint32_t> other(pageTokens);
      for (uint32_t i = 0; i < pageTokens; ++i) other[i] = rid * 977 + i;
      publishChain(cache, rid, other);
      cache.refresh(probe, tokens, {});
    });
  }

  {
    test::TestKvStorage storage2(8192, 4096, 16);
    KvPool pool2(storage2, 8000);
    Cache cache2(pool2, nullptr, nullptr);
    std::vector<StateUse> uses;
    for (uint32_t c = 0; c < 16; ++c) {
      std::vector<uint32_t> t(size_t(depth) * pageTokens);
      for (uint32_t i = 0; i < t.size(); ++i) t[i] = (c + 1) * 123456789u + i;
      uint64_t rid = 1000 + c;
      cache2.beginRequest(rid);
      if (!cache2.ensureTokens(rid, t.size()).granted())
        throw std::runtime_error("page budget too small");
      cache2.publishCommittedBlocks(rid, t, t.size(), {});
      uint64_t block = cache2.blockAt(rid, t.size() / pageTokens * pageTokens);
      cache2.endRequest(rid);
      cache2.publishCompositeState(block, std::make_shared<BenchState>(100), false);
      uses.push_back(cache2.useState(block));
    }
    bench("reusablePages (16 used, 512-deep)", 300, [&] {
      (void)cache2.reusablePages(ReclaimClass::Ordinary);
    });
  }

  {
    const uint32_t requests = 200;
    test::TestKvStorage storage3(4096, 4096, 16);
    KvPool pool3(storage3, 3900);
    Cache cache3(pool3, nullptr, nullptr);
    std::vector<uint32_t> t(size_t(depth) * pageTokens);
    for (uint32_t i = 0; i < t.size(); ++i) t[i] = i * 31 + 7;
    publishChain(cache3, 5000, t); // shared chain; later requests adopt it
    double total = 0;
    for (uint32_t r = 0; r < requests; ++r) {
      cache3.beginRequest(9000 + r);
      (void)cache3.ensureTokens(9000 + r, t.size());
      const auto start = steady_clock::now();
      cache3.endRequest(9000 + r);
      total += duration_cast<nanoseconds>(steady_clock::now() - start).count();
    }
    std::printf("%-40s %10.1f ns/op\n", "endRequest (adopted 512-chain)", total / requests);
  }

  {
    const uint64_t slotBytes = 64ull << 20;
    for (uint32_t round = 0; round < 3; ++round) {
      auto budget = std::make_shared<model::DiskBudget>(slotBytes * 4);
      model::SlotFile file(slotBytes, budget);
      auto slotA = file.acquire(), slotB = file.acquire();
      std::vector<std::byte> big(slotBytes);
      std::vector<std::byte> small(4096);
      (void)file.write(slotB, {std::span<const std::byte>(small)}, {})->wait();
      const auto t0 = steady_clock::now();
      auto w = file.write(slotA, {std::span<const std::byte>(big)}, {});
      auto r = file.read(slotB, {std::span<std::byte>(small)}, {});
      (void)r->wait();
      const auto readDone = steady_clock::now();
      (void)w->wait();
      const auto allDone = steady_clock::now();
      std::printf("%-40s read=%7.1f ms  total=%7.1f ms\n",
                  "slotfile read-behind-write",
                  duration_cast<nanoseconds>(readDone - t0).count() / 1e6,
                  duration_cast<nanoseconds>(allDone - t0).count() / 1e6);
    }
  }

  return 0;
}

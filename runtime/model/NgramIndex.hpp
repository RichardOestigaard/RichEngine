#pragma once

// The n-gram predraft's per-lane state and the pure logic over it: the
// 3-gram index, the chain lookup, the alternate-continuation pick a comb
// tree's leaf uses and the comb table's emission. Runtime::Impl wraps
// these on its Request; the functions take the table directly so tests
// can drive them without a runtime.

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Sampling.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <span>
#include <vector>

namespace richengine::model::ngram {

constexpr uint32_t kNone = ~0u;

// A lane's 3-gram starts in an open-addressed table: lookups run once per
// decode step per lane (and once per corpus doc per step), where
// unordered_map's per-node indirection costs more than the probe. Keys
// pack three <2^18 tokens, so ~0ull can never collide with a real key.
// Nothing deletes — seed() and clear() drop the table whole.
struct Starts {
  struct Slot {
    uint64_t key = ~0ull;
    std::array<uint32_t, 4> starts{};
  };
  std::vector<Slot> slots;
  size_t count = 0;

  static uint64_t mix(uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  }

  void rehash(size_t capacity) {
    std::vector<Slot> next(capacity);
    for (const Slot &slot : slots) {
      if (slot.key == ~0ull)
        continue;
      size_t i = mix(slot.key) & (capacity - 1);
      while (next[i].key != ~0ull)
        i = (i + 1) & (capacity - 1);
      next[i] = slot;
    }
    slots = std::move(next);
  }

  const std::array<uint32_t, 4> *find(uint64_t k) const {
    if (slots.empty())
      return nullptr;
    const size_t mask = slots.size() - 1;
    size_t i = mix(k) & mask;
    while (slots[i].key != ~0ull) {
      if (slots[i].key == k)
        return &slots[i].starts;
      i = (i + 1) & mask;
    }
    return nullptr;
  }

  std::array<uint32_t, 4> *entry(uint64_t k) {
    if (slots.empty())
      rehash(1024);
    if ((count + 1) * 10 >= slots.size() * 7)
      rehash(slots.size() * 2);
    const size_t mask = slots.size() - 1;
    size_t i = mix(k) & mask;
    while (slots[i].key != ~0ull) {
      if (slots[i].key == k)
        return &slots[i].starts;
      i = (i + 1) & mask;
    }
    slots[i].key = k;
    slots[i].starts = {kNone, kNone, kNone, kNone};
    ++count;
    return &slots[i].starts;
  }

  void clear() {
    slots.clear();
    count = 0;
  }
};

// A lane's token stream and, per 3-gram key, its last four starts — a
// key's newest start is the stream's own tail, so lookups skip it to the
// earlier occurrences.
struct Table {
  std::vector<uint32_t> history;
  Starts starts;
};

// The vocabulary is under 2^18, so three tokens pack into one key.
inline uint64_t key(const uint32_t *tokens) {
  return uint64_t{tokens[0]} | (uint64_t{tokens[1]} << 18) |
         (uint64_t{tokens[2]} << 36);
}

inline void noteAt(Table &table, uint32_t start, uint32_t *drops = nullptr) {
  auto *seen = table.starts.entry(key(table.history.data() + start));
  // A fifth+ start evicts the oldest recorded one — the occurrence a
  // variable-order matcher would still have found. `drops` counts them.
  if (drops && (*seen)[3] != kNone)
    ++*drops;
  *seen = {start, (*seen)[0], (*seen)[1], (*seen)[2]};
}

// (Re)seeds a table from its lane's prompt; emitted tokens then append.
inline void seed(Table &table, std::span<const uint32_t> prompt,
                 uint32_t *drops = nullptr) {
  table.history.assign(prompt.begin(), prompt.end());
  table.starts.clear();
  for (uint32_t i = 0; i + 3 <= table.history.size(); ++i)
    noteAt(table, i, drops);
}

inline void append(Table &table, std::span<const uint32_t> tokens) {
  for (const uint32_t token : tokens) {
    table.history.push_back(token);
    const uint32_t size = static_cast<uint32_t>(table.history.size());
    if (size >= 3)
      noteAt(table, size - 3);
  }
}

// Follows the most recent earlier occurrences of the stream's closing
// 3-gram — the stream's own tail is a key's newest start, so each key
// keeps two. The candidate with the longest backward extension wins.
// `extensionOut`, when set, gets the winning occurrence's backward
// extension — the matched pattern's effective length is 3 + extension.
inline uint32_t lookup(const Table &table, uint32_t *out,
                       uint32_t *extensionOut = nullptr) {
  const std::vector<uint32_t> &history = table.history;
  const uint32_t size = static_cast<uint32_t>(history.size());
  if (size < 4)
    return 0;
  const auto *found = table.starts.find(key(history.data() + size - 3));
  if (!found)
    return 0;
  bool have = false;
  uint32_t best = 0, bestExtension = 0, bestFollowers = 0;
  for (const uint32_t start : *found) {
    if (start == kNone || start + 3 >= size)
      continue;
    // An occurrence's extension is bounded by `start` — one under the best
    // found already can neither beat it nor tie, so its scan is dead work.
    if (start < bestExtension)
      continue;
    // Two tokens per compare while a whole pair fits, the scalar loop
    // handling the odd remainder — identical extension values.
    uint32_t extension = 0;
    while (extension + 2 <= start) {
      uint64_t a, b;
      std::memcpy(&a, history.data() + (start - 2 - extension), 8);
      std::memcpy(&b, history.data() + (size - 5 - extension), 8);
      if (a != b)
        break;
      extension += 2;
    }
    while (extension < start &&
           history[start - 1 - extension] == history[size - 4 - extension])
      ++extension;
    const uint32_t followers = size - (start + 3);
    if (!have || extension > bestExtension ||
        (extension == bestExtension && followers > bestFollowers)) {
      have = true;
      best = start;
      bestExtension = extension;
      bestFollowers = followers;
    }
  }
  if (!have)
    return 0;
  const uint32_t followers =
      std::min<uint32_t>(RICHENGINE_DRAFT_PROPOSAL_TOKENS, bestFollowers);
  std::copy_n(history.data() + best + 3, followers, out);
  if (extensionOut)
    *extensionOut = bestExtension;
  return followers;
}

// The cross-request corpus (RICHENGINE_NGRAM_CORPUS): a FIFO of per-request
// tables, each a finished request's committed token stream. A lane drafts
// from it only when its corpus match beats the live match's pattern length
// by the preference margin — a long hit in unrelated reference text is
// likely coincidental (AgSpec's δ; docs/NGRAM_V2_RESEARCH.md).
struct Corpus {
  // Every doc is one finished stream; eviction is whole-doc LRU — the
  // least recently *hit* doc leaves first, so a hot session's corpus
  // survives unrelated chatter. hitStamp is bumped on any doc that
  // contributed a lookup hit.
  struct Doc {
    Table table;
    uint64_t hitStamp = 0;
    // Who produced this stream: the request id that committed it, or 0 for
    // externally pushed text (POST /v1/corpus). A lane never drafts from
    // its own committed stream — its self table already covers it.
    uint64_t origin = 0;
    // Weak session key: a hash of the request's opening prompt tokens —
    // requests extending the same conversation share it. Same-session docs
    // win lookup ties.
    uint64_t session = 0;
  };
  std::deque<Doc> docs;
  // In-flight requests' streams, appended as they commit: a concurrent
  // lane drafting the same content sees it without waiting for the
  // request's end. Never evicted and not counted in `tokens` — the set
  // is bounded by the lane count, and each entry is dropped when its
  // request finishes or suspends (the finished stream is pushed normally).
  std::deque<Doc> liveDocs;
  size_t tokens = 0;
  uint64_t tick = 0;
  // Lifetime counters for /status observability. droppedStarts tracks
  // occurrences pushed past a key's last-4 window — the signal for
  // whether a variable-order matcher would pay. weakMatches counts
  // lookups that found the key but whose best pattern fell below
  // kCorpusArmMin (misses where the key wasn't found at all).
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t weakMatches = 0;
  uint64_t pushes = 0;
  uint64_t evictions = 0;
  uint64_t droppedStarts = 0;
};

// FNV-1a over the opening prompt tokens; stable across turns of one
// conversation, which is all the session tier needs.
inline uint64_t sessionKey(std::span<const uint32_t> prompt) {
  uint64_t hash = 14695981039346656037ull;
  for (const uint32_t token : prompt.first(std::min<size_t>(prompt.size(), 64)))
    hash = (hash ^ token) * 1099511628211ull;
  return hash;
}

// A corpus match with this effective pattern length (3 + backward
// extension) arms a lane for corpus lookups: an 8-token verbatim suffix
// of the prompt already in earlier traffic means the reply likely
// continues known text.
constexpr uint32_t kCorpusArmMin = 8;

inline void push(Corpus &corpus, std::span<const uint32_t> stream,
                 uint32_t maxDocs = 64, size_t maxTokens = 1u << 20,
                 uint64_t origin = 0, uint64_t session = 0) {
  if (stream.size() < 4)
    return;
  Corpus::Doc doc;
  doc.origin = origin;
  doc.session = session;
  uint32_t drops = 0;
  seed(doc.table, stream, &drops);
  corpus.droppedStarts += drops;
  doc.hitStamp = corpus.tick;
  corpus.docs.push_back(std::move(doc));
  corpus.tokens += stream.size();
  ++corpus.pushes;
  while (corpus.docs.size() > maxDocs ||
         (!corpus.docs.empty() &&
          corpus.tokens - corpus.docs.front().table.history.size() >
              maxTokens - corpus.docs.front().table.history.size())) {
    // LRU: evict the least-recently-hit doc, not the oldest.
    auto oldest = corpus.docs.begin();
    for (auto it = corpus.docs.begin() + 1; it != corpus.docs.end(); ++it)
      if (it->hitStamp < oldest->hitStamp)
        oldest = it;
    corpus.tokens -= oldest->table.history.size();
    corpus.docs.erase(oldest);
    ++corpus.evictions;
  }
}

// Drops an in-flight request's live doc — at end (its stream joins docs
// through the normal push) and at suspend (its stale prefix leaves the
// searchable set).
inline void liveEnd(Corpus &corpus, uint64_t origin) {
  for (auto it = corpus.liveDocs.begin(); it != corpus.liveDocs.end(); ++it)
    if (it->origin == origin) {
      corpus.liveDocs.erase(it);
      return;
    }
}

// Opens an in-flight request's live doc: seeded with the stream the
// request will eventually push — its whole prompt+output stream under
// CORPUS_ALL, otherwise an empty table only committed tokens join. A
// resume re-seeds wholesale: the stale doc is dropped first.
inline void liveBegin(Corpus &corpus, uint64_t origin, uint64_t session,
                      std::span<const uint32_t> stream) {
  liveEnd(corpus, origin);
  Corpus::Doc doc;
  doc.origin = origin;
  doc.session = session;
  seed(doc.table, stream);
  doc.hitStamp = corpus.tick;
  corpus.liveDocs.push_back(std::move(doc));
}

// Appends newly committed tokens to a request's live doc. A request
// without one (opted out, or live docs off) is skipped by the scan.
inline void liveAppend(Corpus &corpus, uint64_t origin,
                       std::span<const uint32_t> tokens) {
  for (Corpus::Doc &doc : corpus.liveDocs)
    if (doc.origin == origin) {
      append(doc.table, tokens);
      return;
    }
}

// The lane's closing 3-gram looked up across every corpus doc — live
// in-flight docs first (the freshest wins a full tie), then the finished
// FIFO; the winning occurrence's continuation fills `out`. Returns the
// follower count and the winning backward extension measured against the
// lane's own history — that is the corpus match's effective pattern
// length, minus the fixed 3. `extensionLimit` bounds the per-occurrence
// backward scan: the reported extension never exceeds it, so pass the
// smallest extension the caller's decision still needs (an extension
// past it can never change the outcome).
inline uint32_t lookupCorpus(Corpus &corpus,
                             std::span<const uint32_t> laneHistory,
                             uint32_t *out, uint32_t &extensionOut,
                             uint64_t selfOrigin = 0,
                             uint64_t selfSession = 0,
                             bool sessionGate = false,
                             uint32_t extensionLimit = kNone) {
  const uint32_t laneSize = static_cast<uint32_t>(laneHistory.size());
  if (laneSize < 3)
    return 0;
  const uint64_t pattern =
      key(laneHistory.data() + laneSize - 3);
  Corpus::Doc *bestDoc = nullptr;
  uint32_t best = kNone, bestExtension = 0, bestFollowers = 0;
  bool bestSession = false;
  const auto consider = [&](Corpus::Doc &doc) {
    if (selfOrigin && doc.origin == selfOrigin)
      return;
    // Hard gate: another conversation's docs are invisible, not just
    // outranked. Global docs (session 0) stay open to everyone.
    if (sessionGate && selfSession && doc.session &&
        doc.session != selfSession)
      return;
    const auto *found = doc.table.starts.find(pattern);
    if (!found)
      return;
    const auto &dhistory = doc.table.history;
    const uint32_t dsize = static_cast<uint32_t>(dhistory.size());
    for (const uint32_t start : *found) {
      if (start == kNone || start + 3 >= dsize)
        continue;
      // An occurrence's extension is bounded by `start` — one under the
      // best found already can neither beat it nor tie, so its scan is
      // dead work.
      if (start < bestExtension)
        continue;
      const uint32_t bound =
          std::min({extensionLimit, start, laneSize - 3});
      // Two tokens per compare while a whole pair fits, the scalar loop
      // handling the odd remainder — identical extension values.
      uint32_t extension = 0;
      while (extension + 2 <= bound) {
        uint64_t a, b;
        std::memcpy(&a, dhistory.data() + (start - 2 - extension), 8);
        std::memcpy(&b, laneHistory.data() + (laneSize - 5 - extension), 8);
        if (a != b)
          break;
        extension += 2;
      }
      while (extension < bound &&
             dhistory[start - 1 - extension] ==
                 laneHistory[laneSize - 4 - extension])
        ++extension;
      const uint32_t followers = dsize - (start + 3);
      const bool sameSession =
          selfSession && doc.session == selfSession;
      // Extension, then same-session, then followers: session affinity
      // breaks ties — a same-conversation continuation beats an equal
      // match from unrelated traffic.
      const bool better =
          best == kNone || extension > bestExtension ||
          (extension == bestExtension &&
           (sameSession > bestSession ||
            (sameSession == bestSession && followers > bestFollowers)));
      if (better) {
        best = start;
        bestDoc = &doc;
        bestExtension = extension;
        bestFollowers = followers;
        bestSession = sameSession;
      }
    }
  };
  for (Corpus::Doc &doc : corpus.liveDocs)
    consider(doc);
  for (Corpus::Doc &doc : corpus.docs)
    consider(doc);
  if (best == kNone) {
    ++corpus.misses;
    return 0;
  }
  // Key found but pattern too short to arm/displace — the lane's corpus
  // is close but the last-4 window hid the carrying occurrence.
  if (3 + bestExtension < kCorpusArmMin)
    ++corpus.weakMatches;
  // A contributing doc earns recency — the LRU keeps it past untouched
  // docs under pressure.
  bestDoc->hitStamp = ++corpus.tick;
  ++corpus.hits;
  const uint32_t followers =
      std::min<uint32_t>(RICHENGINE_DRAFT_PROPOSAL_TOKENS, bestFollowers);
  std::copy_n(bestDoc->table.history.data() + best + 3, followers, out);
  extensionOut = bestExtension;
  return followers;
}

// The best-ranked alternate continuation for one tree position: looks up
// the 3-gram that closes the chain's context at that position and returns
// the first follower of the best earlier occurrence other than the chain's
// own token — a same-token leaf rescues nothing. `chain`/`chainLen` is the
// lane's accepted-so-far prefix; context = history + chain, so a
// position's key may straddle the history/chain seam.
inline uint32_t alternate(const Table &table, const uint32_t *keyTail,
                          uint32_t chainToken, const uint32_t *chain,
                          uint32_t chainLen) {
  const std::vector<uint32_t> &history = table.history;
  const uint32_t size = static_cast<uint32_t>(history.size());
  const uint32_t ctxLen = size + chainLen;
  const auto *found = table.starts.find(key(keyTail));
  if (!found || ctxLen < 3)
    return kNone;
  bool have = false;
  uint32_t best = 0, bestExtension = 0, bestFollowers = 0;
  for (const uint32_t start : *found) {
    if (start == kNone || start + 3 >= size ||
        history[start + 3] == chainToken)
      continue;
    // An occurrence's extension is bounded by `start` — one under the
    // best found already can neither beat it nor tie, so its scan is
    // dead work.
    if (start < bestExtension)
      continue;
    uint32_t extension = 0;
    while (extension < start && extension < ctxLen - 3) {
      const uint32_t i = ctxLen - 4 - extension;
      const uint32_t context = i < size ? history[i] : chain[i - size];
      if (history[start - 1 - extension] != context)
        break;
      ++extension;
    }
    const uint32_t followers = size - (start + 3);
    if (!have || extension > bestExtension ||
        (extension == bestExtension && followers > bestFollowers)) {
      have = true;
      best = start;
      bestExtension = extension;
      bestFollowers = followers;
    }
  }
  return have ? history[best + 3] : kNone;
}

// One lane's comb table: the anchor at row 0 and the chain in the node
// block's front half, one sibling leaf per leading position at row
// half + position where an alternate continuation exists. Dead rows keep
// an all-NONE descriptor; the returned tree count covers only the last
// live leaf so a leafless lane verifies like a chain. `found` is how many
// of the proposal slots carry real matches — only those positions can
// offer an alternate.
inline uint32_t combTable(const Table &table, uint32_t anchor,
                          const uint32_t *proposals, uint32_t found,
                          uint32_t *tokens, uint32_t *nodes) {
  constexpr uint32_t kDead =
      RICHENGINE_TREE_NODE_NONE | (RICHENGINE_TREE_NODE_NONE << 8) |
      (RICHENGINE_TREE_NODE_NONE << 16);
  constexpr uint32_t kChain = RICHENGINE_TREE_VERIFY_NODES / 2;
  const uint32_t size = static_cast<uint32_t>(table.history.size());
  std::fill_n(tokens, RICHENGINE_TREE_VERIFY_NODES, 0u);
  std::fill_n(nodes, RICHENGINE_TREE_VERIFY_NODES, kDead);
  tokens[0] = anchor;
  // The anchor row's depth is zero: verify_input adds a dead
  // descriptor's 255 to the base position.
  nodes[0] =
      RICHENGINE_TREE_NODE_NONE | (RICHENGINE_TREE_NODE_NONE << 16);
  for (uint32_t p = 0; p + 1 < kChain && p < RICHENGINE_DRAFT_PROPOSAL_TOKENS;
       ++p) {
    tokens[1 + p] = proposals[p];
    nodes[1 + p] = p | ((p + 1) << 8) | (p << 16);
  }
  uint32_t lastLeaf = kNone;
  for (uint32_t p = 0; p < found && p < kChain; ++p) {
    // The position's context is history + the chain prefix, so its key
    // may straddle the seam.
    const uint32_t ctxLen = size + p;
    uint32_t tail[3];
    for (uint32_t j = 0; j < 3; ++j) {
      const uint32_t i = ctxLen - 3 + j;
      tail[j] = i < size ? table.history[i] : proposals[i - size];
    }
    const uint32_t alt =
        alternate(table, tail, proposals[p], proposals, p);
    if (alt == kNone)
      continue;
    const uint32_t row = kChain + p;
    tokens[row] = alt;
    nodes[row] = p | ((p + 1) << 8) | (p << 16);
    lastLeaf = p;
  }
  return lastLeaf == kNone ? kChain : kChain + lastLeaf + 1;
}

} // namespace richengine::model::ngram

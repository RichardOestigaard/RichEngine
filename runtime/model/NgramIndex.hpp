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
#include <cstdint>
#include <deque>
#include <span>
#include <unordered_map>
#include <vector>

namespace richengine::model::ngram {

// A lane's token stream and, per 3-gram key, its last four starts — a
// key's newest start is the stream's own tail, so lookups skip it to the
// earlier occurrences.
struct Table {
  std::vector<uint32_t> history;
  std::unordered_map<uint64_t, std::array<uint32_t, 4>> starts;
};

constexpr uint32_t kNone = ~0u;

// The vocabulary is under 2^18, so three tokens pack into one key.
inline uint64_t key(const uint32_t *tokens) {
  return uint64_t{tokens[0]} | (uint64_t{tokens[1]} << 18) |
         (uint64_t{tokens[2]} << 36);
}

inline void noteAt(Table &table, uint32_t start) {
  auto &seen =
      table.starts
          .try_emplace(key(table.history.data() + start),
                       std::array{kNone, kNone, kNone, kNone})
          .first->second;
  seen = {start, seen[0], seen[1], seen[2]};
}

// (Re)seeds a table from its lane's prompt; emitted tokens then append.
inline void seed(Table &table, std::span<const uint32_t> prompt) {
  table.history.assign(prompt.begin(), prompt.end());
  table.starts.clear();
  for (uint32_t i = 0; i + 3 <= table.history.size(); ++i)
    noteAt(table, i);
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
  const auto found = table.starts.find(key(history.data() + size - 3));
  if (found == table.starts.end())
    return 0;
  bool have = false;
  uint32_t best = 0, bestExtension = 0, bestFollowers = 0;
  for (const uint32_t start : found->second) {
    if (start == kNone || start + 3 >= size)
      continue;
    uint32_t extension = 0;
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
  // Every doc is one finished stream; eviction is whole-doc FIFO.
  std::deque<Table> docs;
  size_t tokens = 0;
};

// A corpus match with this effective pattern length (3 + backward
// extension) arms a lane for corpus lookups: an 8-token verbatim suffix
// of the prompt already in earlier traffic means the reply likely
// continues known text.
constexpr uint32_t kCorpusArmMin = 8;

inline void push(Corpus &corpus, std::span<const uint32_t> stream,
                 uint32_t maxDocs = 64, size_t maxTokens = 1u << 20) {
  if (stream.size() < 4)
    return;
  Table doc;
  seed(doc, stream);
  corpus.docs.push_back(std::move(doc));
  corpus.tokens += stream.size();
  while (corpus.docs.size() > maxDocs ||
         (!corpus.docs.empty() &&
          corpus.tokens - corpus.docs.front().history.size() >
              maxTokens - corpus.docs.front().history.size())) {
    corpus.tokens -= corpus.docs.front().history.size();
    corpus.docs.pop_front();
  }
}

// The lane's closing 3-gram looked up across every corpus doc; the winning
// occurrence's continuation fills `out`. Returns the follower count and the
// winning backward extension measured against the lane's own history —
// that is the corpus match's effective pattern length, minus the fixed 3.
inline uint32_t lookupCorpus(const Corpus &corpus,
                             std::span<const uint32_t> laneHistory,
                             uint32_t *out, uint32_t &extensionOut) {
  const uint32_t laneSize = static_cast<uint32_t>(laneHistory.size());
  if (laneSize < 3)
    return 0;
  const uint64_t pattern =
      key(laneHistory.data() + laneSize - 3);
  const Table *bestDoc = nullptr;
  uint32_t best = kNone, bestExtension = 0, bestFollowers = 0;
  for (const Table &doc : corpus.docs) {
    const auto found = doc.starts.find(pattern);
    if (found == doc.starts.end())
      continue;
    const auto &dhistory = doc.history;
    const uint32_t dsize = static_cast<uint32_t>(dhistory.size());
    for (const uint32_t start : found->second) {
      if (start == kNone || start + 3 >= dsize)
        continue;
      uint32_t extension = 0;
      while (extension < start && extension < laneSize - 3 &&
             dhistory[start - 1 - extension] ==
                 laneHistory[laneSize - 4 - extension])
        ++extension;
      const uint32_t followers = dsize - (start + 3);
      if (best == kNone || extension > bestExtension ||
          (extension == bestExtension && followers > bestFollowers)) {
        best = start;
        bestDoc = &doc;
        bestExtension = extension;
        bestFollowers = followers;
      }
    }
  }
  if (best == kNone)
    return 0;
  const uint32_t followers =
      std::min<uint32_t>(RICHENGINE_DRAFT_PROPOSAL_TOKENS, bestFollowers);
  std::copy_n(bestDoc->history.data() + best + 3, followers, out);
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
  const auto found = table.starts.find(key(keyTail));
  if (found == table.starts.end() || ctxLen < 3)
    return kNone;
  bool have = false;
  uint32_t best = 0, bestExtension = 0, bestFollowers = 0;
  for (const uint32_t start : found->second) {
    if (start == kNone || start + 3 >= size ||
        history[start + 3] == chainToken)
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

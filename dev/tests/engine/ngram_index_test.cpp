#include "TestChecks.hpp"
#include "model/NgramIndex.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace richengine;
using namespace richengine::model;

namespace {

using richengine::test::require;

constexpr uint32_t kDead =
    RICHENGINE_TREE_NODE_NONE | (RICHENGINE_TREE_NODE_NONE << 8) |
    (RICHENGINE_TREE_NODE_NONE << 16);

ngram::Table table(std::initializer_list<uint32_t> tokens) {
  ngram::Table table;
  ngram::seed(table, tokens);
  return table;
}

// The closing 3-gram's earlier occurrence supplies the followers.
void testLookupFollowsEarlierOccurrence() {
  ngram::Table t =
      table({5, 1, 2, 3, 40, 50, 60, 1, 2, 3});
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  const uint32_t found = ngram::lookup(t, out.data());
  require(found == 6, "lookup should follow the earlier occurrence");
  require((out == std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS>{40u, 50u, 60u, 1u, 2u, 3u, 0u}),
          "lookup should copy the occurrence's followers");
}

// Between earlier occurrences the longest backward extension wins even
// when another occurrence carries more followers.
void testLookupPicksLongestExtension() {
  ngram::Table t =
      table({1, 2, 3, 77, 0, 0, 42, 1, 2, 3, 88, 42, 1, 2, 3});
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  const uint32_t found = ngram::lookup(t, out.data());
  require(found == 5, "lookup should take the better-extended occurrence");
  require((out == std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS>{88u, 42u, 1u, 2u, 3u, 0u, 0u}),
          "lookup should emit the better-extended followers");
}

// The stream's own tail is a key's newest start; it is not a candidate.
void testLookupNeedsFourTokens() {
  ngram::Table t = table({1, 2, 3});
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  require(ngram::lookup(t, out.data()) == 0,
          "lookup should refuse a three-token stream");
}

// An occurrence whose follower equals the chain token rescues nothing, so
// the alternate picks the next occurrence's follower.
void testAlternateSkipsChainToken() {
  ngram::Table t = table({1, 2, 3, 10, 1, 2, 3, 20, 1, 2, 3});
  const uint32_t tail[] = {1, 2, 3};
  require(ngram::alternate(t, tail, 10, nullptr, 0) == 20,
          "alternate should skip the chain token's occurrence");
  require(ngram::alternate(t, tail, 20, nullptr, 0) == 10,
          "alternate should pick the remaining occurrence");
}

// When every earlier occurrence's follower is the chain token there is no
// alternate.
void testAlternateExhausted() {
  ngram::Table t = table({1, 2, 3, 10, 1, 2, 3});
  const uint32_t tail[] = {1, 2, 3};
  require(ngram::alternate(t, tail, 10, nullptr, 0) == ngram::kNone,
          "alternate should decline a same-token follower");
}

// A deep position's backward extension reads the chain prefix once the
// walk crosses out of history.
void testAlternateExtendsIntoChain() {
  // {1,2,3} occurs at start 0 followed by 55; the preceding run 9,9 gives
  // it extension depth.
  ngram::Table t =
      table({9, 9, 9, 1, 2, 3, 55, 0, 0, 0, 0, 0});
  const uint32_t tail[] = {1, 2, 3};
  // chainLen 6 puts the context's last three tokens at history indices
  // size+2-…, so the extension walk indexes into `chain`.
  const uint32_t chain[] = {0, 0, 9, 9, 9, 0};
  require(ngram::alternate(t, tail, 99, chain, 6) == 55,
          "alternate should follow a context spanning the seam");
}

// The leaf descriptor is the chain node's sibling: same parent, depth
// and position; only the token differs. The anchor keeps depth zero —
// verify_input adds the descriptor's depth byte to the base position.
void testCombTableDescriptors() {
  ngram::Table t = table({1, 2, 3, 55, 1, 2});
  const uint32_t proposals[] = {3, 90, 91, 92, 93, 94, 95};
  std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> tokens{};
  std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> nodes{};
  const uint32_t count =
      ngram::combTable(t, 77, proposals, 7, tokens.data(), nodes.data());

  require(tokens[0] == 77, "anchor row should hold the pending token");
  require(nodes[0] ==
              (RICHENGINE_TREE_NODE_NONE | (RICHENGINE_TREE_NODE_NONE << 16)),
          "anchor depth must be zero");
  for (uint32_t p = 0; p < 7; ++p) {
    require(tokens[1 + p] == proposals[p], "chain row should carry the proposal");
    require(nodes[1 + p] == (p | ((p + 1) << 8) | (p << 16)),
            "chain descriptor should be (parent, depth, position)");
  }
  // Position 1's tail {1,2,proposals[0]=3} hits the earlier {1,2,3}
  // occurrence, so row 9 is the live leaf: a sibling of chain row 2.
  require(tokens[9] == 55, "leaf row should hold the alternate follower");
  require(nodes[9] == nodes[2], "leaf should share its sibling's descriptor");
  require(nodes[8] == kDead, "leafless position's row stays dead");
  for (uint32_t row = 10; row < RICHENGINE_TREE_VERIFY_NODES; ++row)
    require(nodes[row] == kDead, "rows past the last leaf stay dead");
  require(count == 10, "tree count should cover the last live leaf");
}

// A lane with no alternates verifies like a bare chain.
void testCombTableLeafless() {
  ngram::Table t = table({7, 8, 9, 10, 11});
  const uint32_t proposals[] = {3, 90, 91, 92, 93, 94, 95};
  std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> tokens{};
  std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> nodes{};
  const uint32_t count =
      ngram::combTable(t, 77, proposals, 7, tokens.data(), nodes.data());
  require(count == RICHENGINE_TREE_VERIFY_NODES / 2,
          "a leafless comb verifies like a chain");
  for (uint32_t row = 8; row < RICHENGINE_TREE_VERIFY_NODES; ++row)
    require(nodes[row] == kDead, "leaf rows stay dead without alternates");
}

// Leaves are only searched under matched positions: found bounds the walk.
void testCombTableHonoursFoundBound() {
  ngram::Table t = table({1, 2, 3, 55, 1, 2});
  const uint32_t proposals[] = {3, 90, 91, 92, 93, 94, 95};
  std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> tokens{};
  std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> nodes{};
  const uint32_t count =
      ngram::combTable(t, 77, proposals, 1, tokens.data(), nodes.data());
  require(count == RICHENGINE_TREE_VERIFY_NODES / 2,
          "positions past the real match emit no leaves");
}

// A corpus doc's closing-3-gram match against the lane history returns
// the occurrence's followers and the backward extension — the pair the
// margin gate compares against the live match.
void testCorpusLookupFindsContinuation() {
  ngram::Corpus corpus;
  const uint32_t doc[] = {9, 9, 1, 2, 3, 40, 50, 60};
  ngram::push(corpus, doc);
  const uint32_t lane[] = {7, 8, 1, 2, 3};
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  uint32_t extension = 99;
  const uint32_t found = ngram::lookupCorpus(corpus, lane, out.data(),
                                             extension);
  require(found == 3 && out[0] == 40 && out[1] == 50 && out[2] == 60,
          "corpus lookup should return the matched doc's followers");
  require(extension == 0,
          "no shared prefix before the key means no backward extension");
}

// The extension walk compares the corpus doc's tokens before the key
// against the lane's own history — cross-doc context counts.
void testCorpusLookupCountsBackwardExtension() {
  ngram::Corpus corpus;
  const uint32_t doc[] = {11, 12, 13, 1, 2, 3, 40, 50};
  ngram::push(corpus, doc);
  const uint32_t lane[] = {5, 11, 12, 13, 1, 2, 3};
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  uint32_t extension = 0;
  const uint32_t found = ngram::lookupCorpus(corpus, lane, out.data(),
                                             extension);
  require(found == 2 && extension == 3,
          "extension should count the three shared context tokens");
}

// Short lane histories and an empty corpus both decline.
void testCorpusLookupRefusesShortLane() {
  ngram::Corpus corpus;
  const uint32_t lane[] = {1, 2, 3};
  uint32_t extension = 0;
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  require(ngram::lookupCorpus(corpus, lane, out.data(), extension) == 0,
          "empty corpus should return no followers");
  const uint32_t doc[] = {1, 2, 3, 4};
  ngram::push(corpus, doc);
  const uint32_t short_lane[] = {1, 2};
  require(ngram::lookupCorpus(corpus, short_lane, out.data(),
                              extension) == 0,
          "a lane shorter than the key should return no followers");
}

// Between docs the longest backward extension wins, then follower count.
void testCorpusLookupRanksAcrossDocs() {
  ngram::Corpus corpus;
  const uint32_t docA[] = {1, 2, 3, 70};
  const uint32_t docB[] = {9, 9, 9, 1, 2, 3, 80};
  ngram::push(corpus, docA);
  ngram::push(corpus, docB);
  const uint32_t lane[] = {9, 9, 9, 1, 2, 3};
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  uint32_t extension = 0;
  const uint32_t found = ngram::lookupCorpus(corpus, lane, out.data(),
                                             extension);
  require(found == 1 && out[0] == 80 && extension == 3,
          "the better-extended doc's continuation should win");
}

// Whole-doc FIFO eviction keeps the corpus bounded on both counts.
void testCorpusPushEvictsWholeDocs() {
  ngram::Corpus corpus;
  const uint32_t a[] = {1, 2, 3, 4, 5};
  const uint32_t b[] = {6, 7, 8, 9, 10};
  const uint32_t c[] = {11, 12, 13, 14, 15};
  ngram::push(corpus, a, /*maxDocs=*/2, /*maxTokens=*/1000);
  ngram::push(corpus, b, 2, 1000);
  ngram::push(corpus, c, 2, 1000);
  require(corpus.docs.size() == 2 && corpus.tokens == 10,
          "the doc cap should evict the oldest document whole");
  // The first doc's key is gone; the retained docs still answer.
  const uint32_t lane[] = {0, 1, 2, 3};
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  uint32_t extension = 0;
  require(ngram::lookupCorpus(corpus, lane, out.data(), extension) == 0,
          "an evicted doc's key must not survive eviction");
}

void testCorpusPushHonoursTokenCap() {
  ngram::Corpus corpus;
  const uint32_t big[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint32_t small[] = {9, 10, 11, 12, 13};
  ngram::push(corpus, big, /*maxDocs=*/64, /*maxTokens=*/10);
  ngram::push(corpus, small, 64, 10);
  require(corpus.docs.size() == 1 && corpus.tokens == 5,
          "the token cap should evict the oversized older doc");
}

// Streams under the 4-token minimum never enter the corpus.
void testCorpusPushDropsShortStreams() {
  ngram::Corpus corpus;
  const uint32_t short_stream[] = {1, 2, 3};
  ngram::push(corpus, short_stream);
  require(corpus.docs.empty() && corpus.tokens == 0,
          "sub-key-length streams should not become docs");
}

// Tokens appended after seeding index their new 3-grams.
void testAppendIndexesNewGrams() {
  ngram::Table t = table({1, 2, 3, 40});
  const uint32_t emitted[] = {50, 1, 2, 3};
  ngram::append(t, emitted);
  std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> out{};
  const uint32_t found = ngram::lookup(t, out.data());
  require(found == 5 && out[0] == 40 && out[1] == 50,
          "appended tokens should feed the index");
}

} // namespace

int main() {
  try {
    testLookupFollowsEarlierOccurrence();
    testLookupPicksLongestExtension();
    testLookupNeedsFourTokens();
    testAlternateSkipsChainToken();
    testAlternateExhausted();
    testAlternateExtendsIntoChain();
    testCombTableDescriptors();
    testCombTableLeafless();
    testCombTableHonoursFoundBound();
    testAppendIndexesNewGrams();
    testCorpusLookupFindsContinuation();
    testCorpusLookupCountsBackwardExtension();
    testCorpusLookupRefusesShortLane();
    testCorpusLookupRanksAcrossDocs();
    testCorpusPushEvictsWholeDocs();
    testCorpusPushHonoursTokenCap();
    testCorpusPushDropsShortStreams();
    std::cout << "ngram index tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ngram index tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

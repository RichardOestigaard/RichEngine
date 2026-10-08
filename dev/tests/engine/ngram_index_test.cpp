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
  require((out == std::array{40u, 50u, 60u, 1u, 2u, 3u, 0u}),
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
  require((out == std::array{88u, 42u, 1u, 2u, 3u, 0u, 0u}),
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
  for (uint32_t p = 0; p < RICHENGINE_DRAFT_PROPOSAL_TOKENS; ++p) {
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
  require(count == RICHENGINE_TARGET_VERIFY_ROWS,
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
  require(count == RICHENGINE_TARGET_VERIFY_ROWS,
          "positions past the real match emit no leaves");
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
    std::cout << "ngram index tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ngram index tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  // (Re)seeds a lane's n-gram state from its prompt at admission; emitted
  // tokens then append through commitSelected.
  void Runtime::Impl::seedNgramHistory(Request &entry, std::span<const uint32_t> prompt) {
    if (!ngramPredraft_ && !std::holds_alternative<NullDraft>(draftModel))
      return;
    ngram::seed(entry.ngram, prompt);
    entry.ngramRounds = 0;
    entry.ngramAcceptedAvg = 0;
    entry.ngramProbeAt = 0;
    entry.ngramInFlight = false;
  }

  void Runtime::Impl::appendNgramTokens(Request &entry, std::span<const uint32_t> tokens) {
    if (!ngramPredraft_ && !std::holds_alternative<NullDraft>(draftModel))
      return;
    ngram::append(entry.ngram, tokens);
  }

  // A lane's expected accepted tokens if its n-gram chain is used now:
  // no match contributes nothing; a lane still warming up or due a probe
  // is scored at the draft's expected rate so it can prove itself; after
  // that its observed EWMA speaks.
  double Runtime::Impl::ngramLaneScore(const Request &entry, bool found) const {
    if (!found)
      return 0;
    if (entry.ngramRounds < ngramWarmup_ ||
        entry.generatedTokens >= entry.ngramProbeAt)
      return ngramDraftExpect_;
    return entry.ngramAcceptedAvg;
  }

  // The learning-free predraft: each lane's chain comes from the followers
  // of its closing 3-gram's last earlier occurrence; a lane without a match
  // repeats its anchor (a wrong guess only wastes its verify rows). The
  // batch goes predrafted when the lanes' expected scores total what the
  // GPU draft would have accepted — mixed-quality lanes no longer veto.
  // Greedy lanes only: injected proposals carry no probabilities.
  bool Runtime::Impl::applyNgramPredraft(std::span<Request *const> entries,
                          uint32_t width, bool emitTree) {
    // A NullDraft model (Granite) has no GPU draft: the n-gram predraft is
    // its only proposer, so the gates below do not veto it.
    const bool nullDraft = std::holds_alternative<NullDraft>(draftModel);
    if (!ngramPredraft_ && !nullDraft)
      return false;
    predraftedTree_ = false;
    predraftedTreeNodes_ = RICHENGINE_TREE_VERIFY_NODES;
    std::array<std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS>, kLaneCount>
        proposals{};
    std::array<uint32_t, kLaneCount> foundCounts{};
    double score = 0;
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = *entries[lane];
      // Sampled lanes cannot consume probability-free proposals; a Null
      // draft still writes them (anchor repeats verify as the anchor or are
      // rejected, wasting only rows).
      if (samplingEnabled(entry) && !nullDraft)
        return false;
      const uint32_t found = ngram::lookup(entry.ngram, proposals[lane].data());
      foundCounts[lane] = found;
      if (found) {
        // Repeat the last real candidate: a duplicate only loses its row.
        for (uint32_t j = found; j < RICHENGINE_DRAFT_PROPOSAL_TOKENS; ++j)
          proposals[lane][j] = proposals[lane][found - 1];
      } else if (entry.pendingToken) {
        std::fill_n(proposals[lane].data(), RICHENGINE_DRAFT_PROPOSAL_TOKENS,
                    *entry.pendingToken);
      }
      score += ngramLaneScore(entry, found != 0);
      if (ngramDebug_)
        fprintf(stderr, "ngram-predraft lane=%u match=%u score=%.2f\n", lane,
                found, ngramLaneScore(entry, found != 0));
    }
    if (score < width * ngramDraftExpect_ && !nullDraft)
      return false;
    // The comb costs a wider verify per step; emit it only while some lane's
    // acceptance EWMA says the proposer is landing often enough for sibling
    // leaves to rescue. A cold batch falls back to the bare chain.
    if (emitTree) {
      bool hot = false;
      for (uint32_t lane = 0; lane < width; ++lane)
        hot |= entries[lane]->ngramAcceptedAvg >= ngramTreeMin_;
      if (!hot)
        emitTree = false;
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      std::memcpy(contents<uint32_t>(
                      decodeArena->get(lane, DecodeTensor::ProposedTokens),
                      "ngram proposals"),
                  proposals[lane].data(),
                  RICHENGINE_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t));
      entries[lane]->ngramInFlight = true;
    }
    if (!emitTree)
      return true;
    uint32_t liveNodes = RICHENGINE_TARGET_VERIFY_ROWS;
    for (uint32_t lane = 0; lane < width; ++lane) {
      const Request &entry = *entries[lane];
      std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> tokens;
      std::array<uint32_t, RICHENGINE_TREE_VERIFY_NODES> nodes;
      const uint32_t count =
          ngram::combTable(entry.ngram,
                           entry.pendingToken ? *entry.pendingToken : 0,
                           proposals[lane].data(), foundCounts[lane],
                           tokens.data(), nodes.data());
      liveNodes = std::max(liveNodes, count);
      std::memcpy(contents<uint32_t>(
                      decodeArena->get(lane, DecodeTensor::TreeTokens),
                      "ngram tree tokens"),
                  tokens.data(), tokens.size() * sizeof(uint32_t));
      std::memcpy(contents<uint32_t>(
                      decodeArena->get(lane, DecodeTensor::TreeNodes),
                      "ngram tree nodes"),
                  nodes.data(), nodes.size() * sizeof(uint32_t));
      *contents<uint32_t>(
          decodeArena->get(lane, DecodeTensor::TreeCounts),
          "ngram tree counts") = count;
    }
    predraftedTreeNodes_ = liveNodes;
    predraftedTree_ = true;
    return true;
  }

} // namespace richengine::model

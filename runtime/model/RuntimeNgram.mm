#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  // (Re)seeds a lane's n-gram state from its prompt at admission; emitted
  // tokens then append through commitSelected.
  void Runtime::Impl::seedNgramHistory(Request &entry, std::span<const uint32_t> prompt) {
    entry.corpusArmed = false;
    entry.corpusProbeAt = 0;
    entry.corpusSession = ngram::sessionKey(prompt);
    if (!ngramPredraftEnabled() && !std::holds_alternative<NullDraft>(draftModel))
      return;
    ngram::seed(entry.ngram, prompt);
    entry.ngramRounds = 0;
    entry.ngramAcceptedAvg = 0;
    entry.ngramProbeAt = 0;
    entry.ngramInFlight = false;
    if (!ngram_.ngramCorpus_)
      return;
    if (ngram_.ngramCorpus_ == 1) {
      entry.corpusArmed = true;
      return;
    }
    // Mode 2: the hint flag arms outright; otherwise the admission probe —
    // the prompt's closing 3-gram occurring deep inside the corpus — says
    // this request likely re-emits known text.
    if (entry.flags & RequestNgramCorpusHint) {
      entry.corpusArmed = true;
      return;
    }
    if (ngram_.ngramCorpusDocs_.docs.empty())
      return;
    std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> scratch;
    uint32_t extension = 0;
    if (ngram::lookupCorpus(ngram_.ngramCorpusDocs_, prompt, scratch.data(),
                            extension, entry.id, entry.corpusSession,
                            ngram_.ngramCorpusSessionGate_) &&
        3 + extension >= ngram::kCorpusArmMin)
      entry.corpusArmed = true;
  }

  void Runtime::Impl::appendNgramTokens(Request &entry, std::span<const uint32_t> tokens) {
    if (!ngramPredraftEnabled() && !std::holds_alternative<NullDraft>(draftModel))
      return;
    ngram::append(entry.ngram, tokens);
  }

  void Runtime::Impl::noteNgramCorpus(const Request &entry) {
    if (!ngram_.ngramCorpus_)
      return;
    const auto &history = entry.ngram.history;
    if (ngram_.ngramCorpusAll_)
      ngram::push(ngram_.ngramCorpusDocs_, history, 64, 1u << 20, entry.id,
                  entry.corpusSession);
    else if (history.size() > entry.promptTokens)
      ngram::push(ngram_.ngramCorpusDocs_,
                  {history.data() + entry.promptTokens,
                   history.size() - entry.promptTokens},
                  64, 1u << 20, entry.id, entry.corpusSession);
  }

  void Runtime::Impl::corpusPush(std::span<const uint32_t> tokens, uint64_t session) {
    if (ngram_.ngramCorpus_)
      ngram::push(ngram_.ngramCorpusDocs_, tokens, 64, 1u << 20, 0, session);
  }

  // A lane's expected accepted tokens if its n-gram chain is used now:
  // no match contributes nothing; a lane still warming up or due a probe
  // is scored at the draft's expected rate so it can prove itself; after
  // that its observed EWMA speaks.
  double Runtime::Impl::ngramLaneScore(const Request &entry, bool found) const {
    if (!found)
      return 0;
    if (entry.ngramRounds < ngram_.ngramWarmup_ ||
        entry.generatedTokens >= entry.ngramProbeAt)
      return ngram_.ngramDraftExpect_;
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
    if (!ngramPredraftEnabled() && !nullDraft)
      return false;
    ngram_.predraftedTree_ = false;
    ngram_.predraftedTreeNodes_ = RICHENGINE_TREE_VERIFY_NODES;
    std::array<std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS>, kLaneCount>
        proposals{};
    std::array<uint32_t, kLaneCount> foundCounts{};
    double score = 0;
    // The bar the n-gram chain must clear is the GPU draft's own measured
    // expectation — its acceptance EWMA — not a constant: a proposer that
    // lands less than the draft it displaces only burns verify rows.
    double draftExpect = 0;
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = *entries[lane];
      // Sampled lanes cannot consume probability-free proposals; a Null
      // draft still writes them (anchor repeats verify as the anchor or are
      // rejected, wasting only rows).
      if (samplingEnabled(entry) && !nullDraft)
        return false;
      uint32_t extension = 0;
      uint32_t found =
          ngram::lookup(entry.ngram, proposals[lane].data(), &extension);
      // The cross-request corpus offers its own continuation; it displaces
      // the lane's match only past the preference margin — the corpus
      // pattern (3 + its backward extension against the lane) must be that
      // much longer, the measured defense against coincidental long hits
      // in unrelated earlier traffic.
      // Starvation probe: a lane whose own 3-gram found nothing — where
      // the per-request matcher can only emit anchor repeats — probes the
      // corpus each kNgramProbeTokens and arms itself on a real match.
      if (ngram_.ngramCorpus_ && !found && !entry.corpusArmed &&
          entry.generatedTokens >= entry.corpusProbeAt &&
          !ngram_.ngramCorpusDocs_.docs.empty()) {
        // Young outputs probe often — a lane that starts re-emitting a
        // pushed or earlier document loses most of the win if arming
        // waits 256 tokens. Steady-state stays at the slow cadence.
        const uint32_t cadence =
            entry.generatedTokens < 256 ? 32 : NgramPredraft::kNgramProbeTokens;
        entry.corpusProbeAt = entry.generatedTokens + cadence;
        std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> probe;
        uint32_t probeExtension = 0;
        if (ngram::lookupCorpus(ngram_.ngramCorpusDocs_, entry.ngram.history,
                                probe.data(), probeExtension, entry.id,
                                entry.corpusSession,
                                ngram_.ngramCorpusSessionGate_) &&
            3 + probeExtension >= ngram::kCorpusArmMin)
          entry.corpusArmed = true;
      }
      if (ngram_.ngramCorpus_ && ngram_.corpusLane(entry) &&
          !ngram_.ngramCorpusDocs_.docs.empty()) {
        std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> corpus;
        uint32_t corpusExtension = 0;
        const uint32_t corpusFound =
            ngram::lookupCorpus(ngram_.ngramCorpusDocs_, entry.ngram.history,
                                corpus.data(), corpusExtension, entry.id,
                                entry.corpusSession,
                                ngram_.ngramCorpusSessionGate_);
        const uint32_t corpusPattern =
            corpusFound ? 3 + corpusExtension : 0;
        const uint32_t selfPattern = found ? 3 + extension : 0;
        if (corpusPattern > selfPattern + ngram_.ngramCorpusMargin_) {
          std::copy(corpus.begin(), corpus.end(), proposals[lane].begin());
          found = corpusFound;
        }
      }
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
      draftExpect += entry.proposalAcceptedAvg;
      if (ngramDebug_)
        fprintf(stderr, "ngram-predraft lane=%u match=%u score=%.2f\n", lane,
                found, ngramLaneScore(entry, found != 0));
    }
    if (score < draftExpect && !nullDraft)
      return false;
    // The comb costs a wider verify per step; emit it only while some lane's
    // acceptance EWMA says the proposer is landing often enough for sibling
    // leaves to rescue. A cold batch falls back to the bare chain.
    if (emitTree) {
      bool hot = false;
      for (uint32_t lane = 0; lane < width; ++lane)
        hot |= entries[lane]->ngramAcceptedAvg >= ngram_.ngramTreeMin_;
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
    ngram_.predraftedTreeNodes_ = liveNodes;
    ngram_.predraftedTree_ = true;
    return true;
  }

} // namespace richengine::model

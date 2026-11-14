#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  // (Re)seeds a lane's n-gram state from its prompt at admission; emitted
  // tokens then append through commitSelected.
  void Runtime::Impl::seedNgramHistory(Request &entry, std::span<const uint32_t> prompt) {
    entry.corpusArmed = false;
    entry.corpusProbeAt = 0;
    entry.corpusRounds = 0;
    entry.corpusAcceptedAvg = 0;
    entry.corpusInFlight = false;
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
    // Opted-out traffic neither drafts from the corpus nor feeds it.
    if (entry.flags & RequestNgramCorpusOptOut)
      return;
    // The live doc opens at admission: other lanes see this stream's
    // prefix (under CORPUS_ALL) and each committed token as it lands.
    if (ngram_.ngramCorpusLive_)
      ngram::liveBegin(ngram_.ngramCorpusDocs_, entry.id, entry.corpusSession,
                       ngram_.ngramCorpusAll_ ? prompt
                                              : std::span<const uint32_t>{});
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
    if (ngram_.ngramCorpusDocs_.docs.empty() &&
        ngram_.ngramCorpusDocs_.liveDocs.empty())
      return;
    std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> scratch;
    uint32_t extension = 0;
    // Arming only needs pattern >= kCorpusArmMin — cap the backward
    // extension scan at what the decision can use.
    if (ngram::lookupCorpus(ngram_.ngramCorpusDocs_, prompt, scratch.data(),
                            extension, entry.id, entry.corpusSession,
                            ngram_.ngramCorpusSessionGate_,
                            ngram::kCorpusArmMin - 3) &&
        3 + extension >= ngram::kCorpusArmMin)
      entry.corpusArmed = true;
  }

  void Runtime::Impl::appendNgramTokens(Request &entry, std::span<const uint32_t> tokens) {
    if (!ngramPredraftEnabled() && !std::holds_alternative<NullDraft>(draftModel))
      return;
    ngram::append(entry.ngram, tokens);
    if (ngram_.ngramCorpus_ && ngram_.ngramCorpusLive_)
      ngram::liveAppend(ngram_.ngramCorpusDocs_, entry.id, tokens);
  }

  void Runtime::Impl::removeNgramLiveDoc(const Request &entry) {
    if (ngram_.ngramCorpus_)
      ngram::liveEnd(ngram_.ngramCorpusDocs_, entry.id);
  }

  void Runtime::Impl::noteNgramCorpus(const Request &entry) {
    if (!ngram_.ngramCorpus_)
      return;
    // The live doc's job is done; the push below commits the whole stream.
    ngram::liveEnd(ngram_.ngramCorpusDocs_, entry.id);
    if (entry.flags & RequestNgramCorpusOptOut)
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
  // is scored at the draft's own measured rate so the mixer's tie-break
  // lets it prove itself; a constant bar would deadlock against a strong
  // draft or hand a weak one free displacement. After the warmup (or
  // between probes) its observed EWMA speaks.
  double Runtime::Impl::ngramLaneScore(const Request &entry, bool found,
                                       bool drafted) const {
    if (!found)
      return 0;
    if (entry.ngramRounds < ngram_.ngramWarmup_ ||
        entry.generatedTokens >= entry.ngramProbeAt)
      return drafted ? entry.proposalAcceptedAvg : ngram_.ngramDraftExpect_;
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
    ngram_.mixMask_ = 0;
    ngram_.mixCorpusMask_ = 0;
    std::array<std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS>, kLaneCount>
        proposals{};
    std::array<uint32_t, kLaneCount> foundCounts{};
    std::array<bool, kLaneCount> corpusLanes{};
    std::array<uint32_t, kLaneCount> corpusPatterns{};
    double score = 0;
    // The bar the n-gram chain must clear is the GPU draft's own measured
    // expectation — its acceptance EWMA — not a constant: a proposer that
    // lands less than the draft it displaces only burns verify rows.
    double draftExpect = 0;
    // Σ max(nᵢ, dᵢ) over the batch — what a drafted batch with per-lane
    // mixing scores, the total displacement actually has to beat.
    double mixExpect = 0;
    bool sampledLane = false;
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = *entries[lane];
      // Sampled lanes cannot consume probability-free proposals; a Null
      // draft still writes them (anchor repeats verify as the anchor or are
      // rejected, wasting only rows). On a drafted model one sampled lane
      // vetoes all-ngram displacement but keeps its own draft — the other
      // lanes may still mix.
      if (samplingEnabled(entry) && !nullDraft) {
        sampledLane = true;
        draftExpect += entry.proposalAcceptedAvg;
        mixExpect += entry.proposalAcceptedAvg;
        continue;
      }
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
          !(entry.flags & RequestNgramCorpusOptOut) &&
          entry.generatedTokens >= entry.corpusProbeAt &&
          !(ngram_.ngramCorpusDocs_.docs.empty() &&
            ngram_.ngramCorpusDocs_.liveDocs.empty())) {
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
                                ngram_.ngramCorpusSessionGate_,
                                ngram::kCorpusArmMin - 3) &&
            3 + probeExtension >= ngram::kCorpusArmMin)
          entry.corpusArmed = true;
      }
      // A proven-bad corpus loses its consult: armed on a shared prefix, a
      // doc whose continuations keep failing the verify wastes the
      // displacement's margin — the starvation probe's cadence may re-arm
      // the lane on a fresh match.
      if (entry.corpusArmed && entry.corpusRounds >= ngram_.ngramWarmup_ &&
          entry.corpusAcceptedAvg < ngram_.ngramTreeMin_)
        entry.corpusArmed = false;
      if (ngram_.ngramCorpus_ && ngram_.corpusLane(entry) &&
          !(ngram_.ngramCorpusDocs_.docs.empty() &&
            ngram_.ngramCorpusDocs_.liveDocs.empty())) {
        const uint32_t selfPattern = found ? 3 + extension : 0;
        // Displacement needs corpusPattern > selfPattern + margin, i.e.
        // extension > selfPattern + margin - 3 — cap the scan at the
        // threshold (floored at the arming minimum so weakMatch counting
        // stays accurate). Extensions past it cannot change the decision.
        const uint32_t extensionLimit =
            std::max(selfPattern + ngram_.ngramCorpusMargin_,
                     2 + ngram::kCorpusArmMin - 3) -
            2;
        std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS> corpus;
        uint32_t corpusExtension = 0;
        const uint32_t corpusFound =
            ngram::lookupCorpus(ngram_.ngramCorpusDocs_, entry.ngram.history,
                                corpus.data(), corpusExtension, entry.id,
                                entry.corpusSession,
                                ngram_.ngramCorpusSessionGate_,
                                extensionLimit);
        const uint32_t corpusPattern =
            corpusFound ? 3 + corpusExtension : 0;
        if (corpusPattern > selfPattern + ngram_.ngramCorpusMargin_) {
          std::copy(corpus.begin(), corpus.end(), proposals[lane].begin());
          found = corpusFound;
          corpusLanes[lane] = true;
          corpusPatterns[lane] = corpusPattern;
          if (ngramDebug_)
            fprintf(stderr, "  corpus-displaced lane=%u cfound=%u cext=%u\n",
                    lane, corpusFound, corpusExtension);
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
      double nScore = ngramLaneScore(entry, found != 0, !nullDraft);
      // A corpus-displaced chain carries its own evidence — the pattern
      // length that won the margin is a direct measurement, not an EWMA
      // estimate — so it scores itself instead of waiting on a prior the
      // draft-blind content could never clear.
      if (corpusLanes[lane])
        nScore = std::max(
            nScore, std::min<double>(corpusPatterns[lane],
                                     RICHENGINE_DRAFT_PROPOSAL_TOKENS));
      const double dScore = entry.proposalAcceptedAvg;
      // Unproven lanes (warmup or probe-due) score at parity so they can
      // mix — free exploration inside a drafted batch — but contribute
      // nothing to the displacement bar: a blind all-ngram skip pays full
      // verify cost for an unmeasured guess.
      const bool unproven =
          entry.ngramRounds < ngram_.ngramWarmup_ ||
          entry.generatedTokens >= entry.ngramProbeAt;
      if (!unproven || corpusLanes[lane])
        score += nScore;
      draftExpect += dScore;
      mixExpect += std::max(nScore, dScore);
      // Mix arm: the n-gram at least matches what the draft would score on
      // this lane — parity is the warmup prior, so ties explore; a batch
      // that stays drafted gets applyNgramMix's row swap for free.
      if (!nullDraft && ngram_.ngramMix_ && nScore >= dScore &&
          entry.constraint != ConstraintMode::TokenMask) {
        ngram_.mixMask_ |= uint32_t{1} << lane;
        if (corpusLanes[lane])
          ngram_.mixCorpusMask_ |= uint32_t{1} << lane;
        std::memcpy(static_cast<uint32_t *>(ngramMixTokens().contents()) +
                        lane * RICHENGINE_DRAFT_PROPOSAL_TOKENS,
                    proposals[lane].data(),
                    RICHENGINE_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t));
      }
      if (ngramDebug_)
        fprintf(stderr, "ngram-predraft lane=%u match=%u score=%.2f\n", lane,
                found, nScore);
    }
    // All-ngram clears the mixed total discounted by ρ: skipping the draft
    // forward refunds its cost share, so a sum modestly under Σ max can
    // still be the cheaper step. With mixing off the bar is the draft's.
    const double expect =
        (ngram_.ngramMix_ ? mixExpect : draftExpect) *
        (1.0 - ngram_.ngramDraftCost_);
    if ((sampledLane || score < expect) && !nullDraft)
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
      entries[lane]->corpusInFlight = corpusLanes[lane];
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

  // Post-selector per-lane mix: overwrite the proposal rows of lanes whose
  // n-gram score beat the draft EWMA — the draft pass already ran, so the
  // better chain costs one row copy. Verify stays authoritative: a wrong
  // n-gram row loses only itself, exactly like a losing draft pick.
  void Runtime::Impl::applyNgramMix(CommandGraph &graph,
                                    std::span<Request *const> entries,
                                    uint32_t width) {
    const MetalBuffer packed =
        decodeArena->packed(DecodeTensor::ProposedTokens, width);
    for (uint32_t lane = 0; lane < width; ++lane) {
      if (!(ngram_.mixMask_ & (uint32_t{1} << lane)))
        continue;
      ops::RowCopy::add(
          graph, ngramMixTokens(),
          {lane, 2 * RICHENGINE_DRAFT_PROPOSAL_TOKENS, 0}, packed,
          {lane, 2 * RICHENGINE_DRAFT_PROPOSAL_TOKENS, 0}, 1,
          2 * RICHENGINE_DRAFT_PROPOSAL_TOKENS);
      entries[lane]->ngramInFlight = true;
      entries[lane]->corpusInFlight =
          (ngram_.mixCorpusMask_ & (uint32_t{1} << lane)) != 0;
      if (ngramDebug_)
        fprintf(stderr, "ngram-mix lane=%u\n", lane);
    }
  }

} // namespace richengine::model

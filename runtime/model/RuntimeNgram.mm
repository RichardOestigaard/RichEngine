#include "model/RuntimeImpl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace richengine::model {
namespace {

template <class F> struct ScopeExit final {
  explicit ScopeExit(F action) : action(std::move(action)) {}
  ~ScopeExit() { action(); }
  ScopeExit(const ScopeExit &) = delete;
  ScopeExit &operator=(const ScopeExit &) = delete;
  F action;
};

template <class F> ScopeExit<F> scopeExit(F action) {
  return ScopeExit<F>(std::move(action));
}

} // namespace


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
    // The suffix-automaton lookup indexes the same stream (Splash 1.7.0
    // port); a reseed rebuilds it wholesale like the 3-gram table.
    entry.lookup = LookupIndex();
    entry.lookup.append(prompt);
    entry.lookupRounds = 0;
    entry.lookupAcceptedAvg = 0;
    entry.lookupInFlight = false;
    entry.lookupSourced = false;
    entry.lookupMatchLength = 0;
    entry.lookupChainLength = 0;
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
    entry.lookup.append(tokens);
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

  // The chain length that can actually occupy verify rows. Draft
  // checkpoints commonly train fewer proposals than the compiled fifteen;
  // scoring fifteen retrieved tokens against a five-proposal neural draft
  // otherwise overvalues every long exact match.
  uint32_t Runtime::Impl::ngramScoringDepth(const Request &entry,
                                            uint32_t found) const {
    const uint32_t proposalCap =
        modelPolicy().proposalCap.value_or(geometry.draft.proposalLimit());
    const uint32_t budget =
        adaptiveProposals_ && entry.constraint != ConstraintMode::TokenMask
            ? std::clamp(entry.proposalBudget, std::min<uint32_t>(1, proposalCap),
                         proposalCap)
            : proposalCap;
    const uint32_t remaining = entry.maxNewTokens > entry.generatedTokens
                                   ? entry.maxNewTokens - entry.generatedTokens
                                   : 0;
    const uint32_t retention = remaining ? remaining - 1 : 0;
    return std::min({found, budget, retention});
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

  uint32_t Runtime::Impl::draftCandidateStride() const {
    return std::holds_alternative<DSparkDraft>(draftModel)
               ? RICHENGINE_DSPARK_POOL
               : RICHENGINE_DRAFT_CANDIDATES;
  }

  std::span<const uint32_t> Runtime::Impl::draftCandidateRow(
      uint32_t lane, uint32_t width) const {
    if (!width || lane >= width)
      return {};
    const bool dspark = std::holds_alternative<DSparkDraft>(draftModel);
    const DecodeTensor tensor =
        dspark ? DecodeTensor::TopPartialIds : DecodeTensor::Candidates;
    const MetalBuffer packedBuffer = decodeArena->packed(tensor, width);
    if (!packedBuffer)
      return {};
    const uint32_t stride = draftCandidateStride();
    const size_t perLane =
        size_t{RICHENGINE_DRAFT_PROPOSAL_TOKENS} * stride;
    if (packedBuffer.sizeBytes() < uint64_t{width} * perLane * sizeof(uint32_t))
      return {};
    const uint32_t *base = contents<uint32_t>(packedBuffer, "draft candidates");
    return {base + size_t{lane} * perLane, perLane};
  }

  bool Runtime::Impl::ngramSkipsGpuPredraft(
      std::span<Request *const> entries, uint32_t width) const {
    if (!ngram_.skipGpuPredraft_ || !width)
      return false;
    for (uint32_t lane = 0; lane < width; ++lane) {
      const Request &entry = *entries[lane];
      // Suppressing the overlapped draft is not free exploration: when the
      // later displacement gate declines, the lane pays a synchronous draft
      // in the decode command. Require measured proposer evidence first;
      // corpus chains may qualify through their match-length prior.
      const bool unproven =
          entry.ngramRounds + entry.lookupRounds + entry.corpusRounds <
          ngram_.ngramWarmup_;
      if (unproven &&
          ngram_.chainSources_[lane] != NgramPredraft::ChainSource::Corpus)
        return false;
      if (ngram_.chainSources_[lane] == NgramPredraft::ChainSource::None ||
          ngram_.chainLengths_[lane] < ngram_.skipMinChain_ ||
          ngram_.chainScores_[lane] + 1e-9 < entry.proposalAcceptedAvg)
        return false;
    }
    return true;
  }

  // The learning-free predraft: each lane's chain comes from the followers
  // of its closing 3-gram's last earlier occurrence; a lane without a match
  // repeats its anchor (a wrong guess only wastes its verify rows). The
  // batch goes predrafted when the lanes' expected scores total what the
  // GPU draft would have accepted — mixed-quality lanes no longer veto.
  // Greedy lanes only: injected proposals carry no probabilities.
  bool Runtime::Impl::applyNgramPredraft(std::span<Request *const> entries,
                          uint32_t width, bool emitTree, bool mixOnly) {
    const auto hostStart = std::chrono::steady_clock::now();
    const auto hostTimer = scopeExit([&] {
      counters.ngramHostMicroseconds +=
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - hostStart)
                                    .count());
      ++counters.ngramHostEvaluations;
    });
    // A NullDraft model (Granite) has no GPU draft: the n-gram predraft is
    // its only proposer, so the gates below do not veto it.
    const bool nullDraft = std::holds_alternative<NullDraft>(draftModel);
    if (!ngramPredraftEnabled() && !nullDraft)
      return false;
    ngram_.predraftedTree_ = false;
    ngram_.predraftedTreeNodes_ = RICHENGINE_TREE_VERIFY_NODES;
    ngram_.mixMask_ = 0;
    ngram_.mixCorpusMask_ = 0;
    ngram_.suppressMask_ = 0;
    ngram_.resetEvaluation();
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
    const auto acceptanceSum = [&](const Request &entry, uint32_t matchLength,
                                   uint32_t chainLength) {
      double survival = 1.0;
      double total = 0;
      for (uint32_t depth = 0; depth < chainLength; ++depth) {
        survival *= ngram_.lookupAcceptance_.rate(
            samplingEnabled(entry), matchLength, depth,
            LookupAgreement::Unknown);
        total += survival;
      }
      return total;
    };
    // O6: blend the lane's residual EWMA with this step's match-depth prior.
    // scoreBlend is the prior's weight; 0.5 halves stale-EWMA domination.
    const auto blendLaneScore = [&](const Request &entry, double base,
                                    uint32_t matchLength,
                                    uint32_t chainLength) {
      if (!ngram_.stepScore_ || !chainLength)
        return base;
      // The prior is a cumulative-survival sum: rate() is conditional on
      // the parent surviving, so depth d contributes only after every
      // earlier conditional acceptance. Summing the conditionals would
      // overestimate deep chains.
      double survival = 1.0;
      double prior = 0.0;
      for (uint32_t depth = 0; depth < chainLength; ++depth) {
        survival *= ngram_.lookupAcceptance_.rate(
            samplingEnabled(entry), matchLength, depth,
            LookupAgreement::Unknown);
        prior += survival;
      }
      if (base <= 0)
        return prior;
      const double blend = std::clamp(ngram_.scoreBlend_, 0.0, 1.0);
      return (1.0 - blend) * base + blend * prior;
    };
    const auto pseudoConfidence = [&](uint32_t lane, uint32_t depth) {
      const uint32_t matchLength = std::max<uint32_t>(ngram_.chainPatterns_[lane], 3);
      double acceptance = ngram_.lookupAcceptance_.rate(
          false, matchLength, depth, LookupAgreement::Unknown);
      acceptance = std::clamp(acceptance, 1e-5, 1.0 - 1e-5);
      const VerifyBudget::Acceptance &fit = verifyBudget_.acceptance();
      const double slope = fit.slope == 0.0F ? 1.0 : fit.slope;
      const double logitAcceptance =
          std::log(acceptance / (1.0 - acceptance));
      const double z = (logitAcceptance - fit.intercepts[depth]) / slope;
      const double confidence = 1.0 / (1.0 + std::exp(-z));
      if (!std::isfinite(confidence))
        return 0.0F;
      return static_cast<float>(std::clamp(confidence, 0.0, 1.0));
    };
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
      // NullDraft self-suppression: a lane that finished its warmup with a
      // dead proposer — acceptance EWMA under kNgramSuppressAcceptance —
      // verifies its anchor alone instead of burning rows on proposals
      // that never land. The probe cadence still owes it one measurement
      // round each kNgramProbeTokens (generatedTokens >= ngramProbeAt),
      // bounding the worst-case waste to one round in 256 tokens; a probe
      // that lands lifts the EWMA back over the bar and the lane proposes
      // continuously again.
      // Rounds and acceptance count both proposers: a lookup-displaced
      // round never increments ngramRounds, so the 3-gram's own counters
      // alone would leave a fully displaced lane warming up — and
      // proposing — forever. Either source landing keeps the lane alive.
      const uint32_t proposingRounds =
          entry.ngramRounds + entry.lookupRounds;
      const double bestAccepted =
          std::max(entry.ngramAcceptedAvg, entry.lookupAcceptedAvg);
      if (nullDraft && proposingRounds >= ngram_.ngramWarmup_ &&
          entry.generatedTokens < entry.ngramProbeAt &&
          bestAccepted < NgramPredraft::kNgramSuppressAcceptance) {
        ngram_.suppressMask_ |= uint32_t{1} << lane;
        entry.lookupSourced = false;
        if (ngramDebug_)
          fprintf(stderr, "ngram-predraft lane=%u suppressed ewma=%.3f\n",
                  lane, bestAccepted);
        continue;
      }
      ++counters.ngramEvaluations;
      uint32_t extension = 0;
      uint32_t found =
          ngram::lookup(entry.ngram, proposals[lane].data(), &extension,
                        tuning().ngramExtensionCap);
      if (found) {
        ngram_.chainSources_[lane] = NgramPredraft::ChainSource::Self;
        ngram_.chainPatterns_[lane] = 3 + extension;
      }
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
          ngram_.chainSources_[lane] = NgramPredraft::ChainSource::Corpus;
          ngram_.chainPatterns_[lane] = corpusPattern;
          if (ngramDebug_)
            fprintf(stderr, "  corpus-displaced lane=%u cfound=%u cext=%u\n",
                    lane, corpusFound, corpusExtension);
        }
      }
      // Prompt lookup (Splash 1.7.0 LookupIndex port): the suffix
      // automaton's longest earlier occurrence offers its own continuation
      // beside the 3-gram chain — up to 64 tokens of match context where
      // the 3-gram keys on three. The fitted acceptance prior scores it
      // until the chain's own EWMA proves it; the better chain takes the
      // lane's proposal rows and the same verify forward checks it, so a
      // losing displacement wastes nothing but the rows it replaced. A
      // corpus displacement already won on measured pattern length and is
      // left alone.
      entry.lookupSourced = false;
      entry.lookupMatchLength = 0;
      entry.lookupChainLength = 0;
      double lookupLaneScore = 0;
      const LookupIndex::Match lookupMatch = entry.lookup.match();
      if (lookupMatch.length && !corpusLanes[lane]) {
        const std::span<const uint32_t> lookupTokens =
            entry.lookup.proposals(RICHENGINE_DRAFT_PROPOSAL_TOKENS);
        if (!lookupTokens.empty()) {
          const uint32_t lookupDepth = ngramScoringDepth(
              entry, static_cast<uint32_t>(lookupTokens.size()));
          const double lookupBase =
              entry.lookupRounds >= ngram_.ngramWarmup_
                  ? entry.lookupAcceptedAvg
                  : acceptanceSum(entry, lookupMatch.length, lookupDepth);
          lookupLaneScore = blendLaneScore(
              entry, lookupBase, lookupMatch.length, lookupDepth);
          double selfCompare =
              ngramLaneScore(entry, found != 0, !nullDraft);
          if (ngram_.chainSources_[lane] == NgramPredraft::ChainSource::Self)
            selfCompare = blendLaneScore(
                entry, selfCompare, ngram_.chainPatterns_[lane],
                ngramScoringDepth(entry, found));
          if (lookupLaneScore > selfCompare) {
            found = static_cast<uint32_t>(lookupTokens.size());
            std::copy(lookupTokens.begin(), lookupTokens.end(),
                      proposals[lane].begin());
            entry.lookupSourced = true;
            entry.lookupMatchLength = lookupMatch.length;
            entry.lookupChainLength = found;
            ngram_.chainSources_[lane] = NgramPredraft::ChainSource::Lookup;
            ngram_.chainPatterns_[lane] = lookupMatch.length;
            std::copy(lookupTokens.begin(), lookupTokens.end(),
                      entry.lookupChain.begin());
            if (ngramDebug_)
              fprintf(stderr,
                      "  lookup-displaced lane=%u match=%u len=%u score=%.2f\n",
                      lane, lookupMatch.length, found, lookupLaneScore);
          }
        }
      }
      foundCounts[lane] = found;
      ngram_.chainLengths_[lane] = found;
      counters.ngramChainTokens += found;
      if (found < 16)
        ++counters.ngramChainLengthsExact[found];
      if (!found)
        ++counters.ngramChainLength0;
      else if (found < 4)
        ++counters.ngramChainLength1To3;
      else if (found < 8)
        ++counters.ngramChainLength4To7;
      else
        ++counters.ngramChainLength8To15;
      if (found) {
        // Repeat the last real candidate: a duplicate only loses its row.
        for (uint32_t j = found; j < RICHENGINE_DRAFT_PROPOSAL_TOKENS; ++j)
          proposals[lane][j] = proposals[lane][found - 1];
      } else if (entry.pendingToken) {
        std::fill_n(proposals[lane].data(), RICHENGINE_DRAFT_PROPOSAL_TOKENS,
                    *entry.pendingToken);
      }
      double nScore = entry.lookupSourced
                          ? lookupLaneScore
                          : ngramLaneScore(entry, found != 0, !nullDraft);
      if (!entry.lookupSourced &&
          ngram_.chainSources_[lane] == NgramPredraft::ChainSource::Self)
        nScore = blendLaneScore(entry, nScore, ngram_.chainPatterns_[lane],
                                ngramScoringDepth(entry, found));
      // A corpus-displaced chain carries its own evidence — the pattern
      // length that won the margin is a direct measurement, not an EWMA
      // estimate — so it scores itself instead of waiting on a prior the
      // draft-blind content could never clear.
      if (corpusLanes[lane]) {
        if (ngram_.stepScore_) {
          // O6 prices corpus chains by the same conditional-survival
          // evidence as lookup chains. The legacy pattern-length score is
          // nearly always fifteen for a long hit, so it displaces a strong
          // neural draft even when only the first few continuations are
          // likely. A trained corpus lane blends its own measured residual.
          const uint32_t corpusDepth = ngramScoringDepth(entry, found);
          const double corpusBase =
              entry.corpusRounds
                  ? entry.corpusAcceptedAvg
                  : acceptanceSum(entry, corpusPatterns[lane], corpusDepth);
          nScore = blendLaneScore(entry, corpusBase, corpusPatterns[lane],
                                  corpusDepth);
        } else {
          nScore = std::max(
              nScore, std::min<double>(corpusPatterns[lane],
                                       RICHENGINE_DRAFT_PROPOSAL_TOKENS));
        }
      }
      ngram_.chainScores_[lane] = nScore;
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
      const bool corpusOnlyBlocked =
          ngram_.mixCorpusOnly_ &&
          ngram_.chainSources_[lane] != NgramPredraft::ChainSource::Corpus;
      if (!nullDraft && ngram_.ngramMix_ && nScore >= dScore &&
          !corpusOnlyBlocked && entry.constraint != ConstraintMode::TokenMask) {
        ngram_.mixMask_ |= uint32_t{1} << lane;
        if (corpusLanes[lane])
          ngram_.mixCorpusMask_ |= uint32_t{1} << lane;
        std::memcpy(static_cast<uint32_t *>(ngramMixTokens().contents()) +
                        lane * RICHENGINE_DRAFT_PROPOSAL_TOKENS,
                    proposals[lane].data(),
                    RICHENGINE_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t));
        if (ngram_.pseudoConf_) {
          float *confidence =
              static_cast<float *>(ngramMixConfidence().contents()) +
              lane * RICHENGINE_DRAFT_PROPOSAL_TOKENS;
          std::fill_n(confidence, RICHENGINE_DRAFT_PROPOSAL_TOKENS, 0.0F);
          for (uint32_t depth = 0; depth < found; ++depth)
            confidence[depth] = pseudoConfidence(lane, depth);
        }
      }
      if (ngramDebug_)
        fprintf(stderr, "ngram-predraft lane=%u match=%u score=%.2f\n", lane,
                found, nScore);
    }
    // A GPU-predraft adoption only asks for per-lane mixing: no host writes
    // to the shared proposal rows may happen while that forward is active.
    if (mixOnly)
      return false;
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
      // A comb sizes its verify batch-wide; a suppressed lane's rows can
      // only shrink inside the chain path, so a batch carrying one falls
      // back to it.
      if (ngram_.suppressMask_)
        emitTree = false;
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      // Suppressed lanes write nothing and owe no measurement: their rows
      // are dead by the decode path's zeroed proposal count, and a round
      // they never offered must not train the EWMA.
      if (ngram_.suppressMask_ & (uint32_t{1} << lane))
        continue;
      std::memcpy(contents<uint32_t>(
                      decodeArena->get(lane, DecodeTensor::ProposedTokens),
                      "ngram proposals"),
                  proposals[lane].data(),
                  RICHENGINE_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t));
      entries[lane]->ngramInFlight = true;
      entries[lane]->corpusInFlight = corpusLanes[lane];
      entries[lane]->lookupInFlight = entries[lane]->lookupSourced;
    }
    ++counters.ngramPredraftedBatches;
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
      entries[lane]->ngramMixedInFlight = true;
      const NgramPredraft::ChainSource source = ngram_.chainSources_[lane];
      entries[lane]->corpusInFlight =
          source == NgramPredraft::ChainSource::Corpus;
      entries[lane]->lookupInFlight =
          source == NgramPredraft::ChainSource::Lookup;
      ++counters.ngramMixedLanes;
      if (source == NgramPredraft::ChainSource::Self)
        ++counters.ngramMixedSelfLanes;
      else if (source == NgramPredraft::ChainSource::Lookup)
        ++counters.ngramMixedLookupLanes;
      else if (source == NgramPredraft::ChainSource::Corpus)
        ++counters.ngramMixedCorpusLanes;
      if (ngram_.pseudoConf_) {
        const MetalBuffer confidence =
            decodeArena->packed(DecodeTensor::VerifyConfidence, width);
        ops::RowCopy::add(
            graph, ngramMixConfidence(),
            {lane, 2 * RICHENGINE_DRAFT_PROPOSAL_TOKENS, 0}, confidence,
            {lane, 2 * RICHENGINE_DRAFT_PROPOSAL_TOKENS, 0}, 1,
            2 * RICHENGINE_DRAFT_PROPOSAL_TOKENS);
      }
      if (ngramDebug_)
        fprintf(stderr, "ngram-mix lane=%u\n", lane);
    }
  }

} // namespace richengine::model

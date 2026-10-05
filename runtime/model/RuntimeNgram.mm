#include "model/RuntimeImpl.hpp"

namespace richengine::model {

  void Runtime::Impl::noteNgramAt(Request &entry, uint32_t start) {
    auto &seen = entry.ngramIndex
                     .try_emplace(ngramKey(entry.ngramHistory.data() + start),
                                  std::array{kNgramNone, kNgramNone,
                                             kNgramNone, kNgramNone})
                     .first->second;
    seen = {start, seen[0], seen[1], seen[2]};
  }

  // (Re)seeds a lane's n-gram state from its prompt at admission; emitted
  // tokens then append through commitSelected.
  void Runtime::Impl::seedNgramHistory(Request &entry, std::span<const uint32_t> prompt) {
    if (!ngramPredraft_ && !std::holds_alternative<NullDraft>(draftModel))
      return;
    entry.ngramHistory.assign(prompt.begin(), prompt.end());
    entry.ngramIndex.clear();
    entry.ngramRounds = 0;
    entry.ngramAcceptedAvg = 0;
    entry.ngramProbeAt = 0;
    entry.ngramInFlight = false;
    for (uint32_t i = 0; i + 3 <= entry.ngramHistory.size(); ++i)
      noteNgramAt(entry, i);
  }

  void Runtime::Impl::appendNgramTokens(Request &entry, std::span<const uint32_t> tokens) {
    if (!ngramPredraft_ && !std::holds_alternative<NullDraft>(draftModel))
      return;
    for (const uint32_t token : tokens) {
      entry.ngramHistory.push_back(token);
      const uint32_t size = static_cast<uint32_t>(entry.ngramHistory.size());
      if (size >= 3)
        noteNgramAt(entry, size - 3);
    }
  }

  // Follows the most recent earlier occurrences of the stream's closing
  // 3-gram — the stream's own tail is a key's newest start, so each key
  // keeps two. The candidate with the longest backward extension wins.
  uint32_t Runtime::Impl::ngramLookup(const Request &entry, uint32_t *out) const {
    const std::vector<uint32_t> &history = entry.ngramHistory;
    const uint32_t size = static_cast<uint32_t>(history.size());
    if (size < 4)
      return 0;
    const auto found =
        entry.ngramIndex.find(ngramKey(history.data() + size - 3));
    if (found == entry.ngramIndex.end())
      return 0;
    bool have = false;
    uint32_t best = 0, bestExtension = 0, bestFollowers = 0;
    for (const uint32_t start : found->second) {
      if (start == kNgramNone || start + 3 >= size)
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
    return followers;
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
                          uint32_t width) {
    // A NullDraft model (Granite) has no GPU draft: the n-gram predraft is
    // its only proposer, so the gates below do not veto it.
    const bool nullDraft = std::holds_alternative<NullDraft>(draftModel);
    if (!ngramPredraft_ && !nullDraft)
      return false;
    std::array<std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS>, kLaneCount>
        proposals{};
    double score = 0;
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = *entries[lane];
      // Sampled lanes cannot consume probability-free proposals; a Null
      // draft still writes them (anchor repeats verify as the anchor or are
      // rejected, wasting only rows).
      if (samplingEnabled(entry) && !nullDraft)
        return false;
      const uint32_t found = ngramLookup(entry, proposals[lane].data());
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
    for (uint32_t lane = 0; lane < width; ++lane) {
      std::memcpy(contents<uint32_t>(
                      decodeArena->get(lane, DecodeTensor::ProposedTokens),
                      "ngram proposals"),
                  proposals[lane].data(),
                  RICHENGINE_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t));
      entries[lane]->ngramInFlight = true;
    }
    return true;
  }

} // namespace richengine::model

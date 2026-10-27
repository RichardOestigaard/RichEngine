"""N-gram proposer implementations shared by the model A/B harness
(ngram_v2_ab.py) and the offline trace replay (ngram_replay.py). Pure
Python, no mlx dependency.

Interface: proposer.propose(pending, ctx, k) -> list[int], where ctx is the
committed token stream (prompt + emitted) and pending the token about to be
verified; the returned chain proposes what follows pending.
"""

P_MIN = 3  # shortest pattern allowed to fire (v1's implicit threshold)
P_MAX = 64  # deepest suffix pattern tried


class NgramV1:
    """Port of runtime/model/NgramIndex.hpp: 3-gram key, last-4 starts,
    backward-extension ranking, fixed-length chain."""

    def __init__(self):
        self.starts = {}
        self._indexed = 0

    def propose(self, pending, ctx, k):
        seq = ctx + [pending]
        if len(seq) < 4:
            return []
        for i in range(self._indexed, len(seq) - 2):
            key = tuple(seq[i : i + 3])
            seen = self.starts.setdefault(key, [])
            seen.insert(0, i)
            del seen[4:]
        self._indexed = len(seq) - 2
        size = len(seq)
        found = self.starts.get(tuple(seq[size - 3 : size]))
        if not found:
            return []
        best, best_ext, best_fol = None, 0, 0
        for start in found:
            if start + 3 >= size:
                continue
            ext = 0
            while ext < start and seq[start - 1 - ext] == seq[size - 4 - ext]:
                ext += 1
            fol = size - (start + 3)
            if (
                best is None
                or ext > best_ext
                or (ext == best_ext and fol > best_fol)
            ):
                best, best_ext, best_fol = start, ext, fol
        if best is None:
            return []
        return list(seq[best + 3 : best + 3 + min(k, best_fol)])


class NgramV2E:
    """v2fg with expected-accept scoring (SuffixDecoding SCORE applied to
    chains) and AgSpec's margin on the source pick.

    Per candidate chain, expected accepted tokens ~= sum over positions of
    the product of conditional frequencies up to that position. Sources
    compete on that score; a corpus source must additionally clear the
    self match's pattern length by `margin` tokens (AgSpec δ).
    `min_expected` drops proposals whose expected accept can't cover a
    verify round's cost.
    """

    def __init__(self, global_corpus=None, theta=0.6, margin=5,
                 min_expected=0.0):
        self.corpus = list(global_corpus or [])
        self.theta = theta
        self.margin = margin
        self.min_expected = min_expected
        self.last_match_p = 0
        self.last_match_source = ""

    @staticmethod
    def _best_chain(source, seq, k, theta):
        """Best chain in one source across pattern lengths, scored by
        expected accepted tokens."""
        best_chain, best_exp, best_p = [], 0.0, 0
        for p in range(
            min(len(seq) - 1, len(source) - 1, P_MAX), P_MIN - 1, -1
        ):
            pat = seq[-p:]
            occs = [
                i
                for i in range(len(source) - p)
                if source[i : i + p] == pat
            ]
            if not occs:
                continue
            ends = NgramV2F._group_occs(source, occs, p)
            chain, expected = NgramV2F._chain(source, ends, 0, k, theta,
                                              expected=True)
            if expected > best_exp:
                best_chain, best_exp, best_p = chain, expected, p
            if expected >= k:
                break
        return best_chain, best_exp, best_p

    def propose(self, pending, ctx, k):
        seq = ctx + [pending]
        s_chain, s_exp, s_p = self._best_chain(seq, seq, k, self.theta)
        cand, cand_exp, cand_p, src = s_chain, s_exp, s_p, "self"
        for corpus in self.corpus:
            c_chain, c_exp, c_p = self._best_chain(corpus, seq, k,
                                                 self.theta)
            if c_p <= s_p + self.margin:
                continue
            if c_exp > cand_exp:
                cand, cand_exp, cand_p, src = c_chain, c_exp, c_p, "corpus"
        self.last_match_p = cand_p
        self.last_match_source = src if cand else ""
        if not cand or cand_exp < self.min_expected:
            return []
        return cand


class NgramV2P:
    """Production-faithful corpus proposer: mirrors NgramIndex.hpp —
    fixed-3 key, per-doc last-4 starts, best backward extension against
    the lane history, followers capped at k. The corpus consult fires only
    for armed lanes and displaces the self match when its pattern
    (3 + extension) beats the self pattern by `margin`.

    `arm_min`: pattern length that arms a lane at admission (prompt tail
    probe) or at a starvation probe (self miss). Armed lanes consult the
    corpus every step; unarmed lanes probe only every `probe_every`
    generated tokens.
    """

    def __init__(self, global_corpus=None, margin=5, arm_min=8,
                 probe_every=256, mode=2, occ_all=False, grouped=False,
                 session=None):
        self.corpus = list(global_corpus or [])
        self.margin = margin
        self.arm_min = arm_min
        self.probe_every = probe_every
        self.mode = mode
        self.armed = mode == 1
        self.probe_at = 0
        self.last_match_p = 0
        self.last_match_source = ""
        self._admission_checked = False
        # occ_all: rank every occurrence, not the doc's last-4 starts.
        # grouped: AgSpec continuation selection — occurrences grouped by
        # their next-3 tokens, majority group wins (ties → most recent).
        # session: when set, docs that are (tokens, session_id) pairs try
        # same-session docs before the global pool.
        self.occ_all = occ_all
        self.grouped = grouped
        self.session = session
        self._corpus_docs = [
            c[0] if isinstance(c, tuple) else c for c in self.corpus
        ]
        self._corpus_sess = [
            c[1] if isinstance(c, tuple) else None for c in self.corpus
        ]
        self.corpus = self._corpus_docs

    @staticmethod
    def _doc_hit(doc, lane, key_end, occ_all=False, grouped=False):
        """v1 lookup semantics on one doc for the lane's closing 3-gram:
        last-4 starts, best backward extension against the lane history."""
        pat = lane[key_end - 3 : key_end]
        starts = [
            i
            for i in range(len(doc) - 3)
            if doc[i : i + 3] == pat
        ]
        if grouped and len(starts) > 1:
            # Most-frequent next-3-token group; rank by extension inside it.
            groups = {}
            for s in starts:
                sig = tuple(doc[s + 3 : s + 6])
                groups.setdefault(sig, []).append(s)
            starts = max(groups.values(), key=lambda g: (len(g), max(g)))
        if not occ_all:
            starts = starts[-4:]
        best, best_ext, best_fol = None, 0, 0
        for s in starts:
            ext = 0
            while (
                ext < s
                and ext < key_end - 3
                and doc[s - 1 - ext] == lane[key_end - 4 - ext]
            ):
                ext += 1
            fol = len(doc) - (s + 3)
            if (
                best is None
                or ext > best_ext
                or (ext == best_ext and fol > best_fol)
            ):
                best, best_ext, best_fol = s, ext, fol
        return best, best_ext, best_fol

    def _scan(self, lane, k, sessions_only):
        best, best_ext, best_fol, best_doc = None, 0, 0, None
        for doc, sess in zip(self.corpus, self._corpus_sess):
            if sessions_only is not None and sess != sessions_only:
                continue
            hit, ext, fol = self._doc_hit(
                doc, lane, len(lane), self.occ_all, self.grouped
            )
            if hit is not None and (
                best is None or ext > best_ext
                or (ext == best_ext and fol > best_fol)
            ):
                best, best_ext, best_fol, best_doc = hit, ext, fol, doc
        return best, best_ext, best_fol, best_doc

    def _corpus_lookup(self, lane, k):
        # Session tier: a same-session doc's hit wins outright (AgSpec's
        # live-over-global preference); fall back to the global pool.
        if self.session is not None:
            best, ext, fol, doc = self._scan(lane, k, self.session)
            if best is not None:
                return (
                    list(doc[best + 3 : best + 3 + min(k, fol)]),
                    3 + ext,
                )
        best, ext, fol, doc = self._scan(lane, k, None)
        if best is None:
            return [], 0
        return (list(doc[best + 3 : best + 3 + min(k, fol)]), 3 + ext)

    def propose(self, pending, ctx, k):
        seq = ctx + [pending]
        if not self._admission_checked and len(seq) >= 3:
            self._admission_checked = True
            if self.mode == 2 and not self.armed and self.corpus:
                _, p = self._corpus_lookup(seq[: len(seq) - 0], k)
                if p >= self.arm_min:
                    self.armed = True
        # Self match (v1 semantics).
        found, best, ext, fol = 0, None, 0, 0
        if len(seq) >= 4:
            pat = seq[-3:]
            starts = [
                i for i in range(len(seq) - 3) if seq[i : i + 3] == pat
            ]
            for s in starts[-4:]:
                e = 0
                while (
                    e < s
                    and e < len(seq) - 3
                    and seq[s - 1 - e] == seq[len(seq) - 4 - e]
                ):
                    e += 1
                f = len(seq) - (s + 3)
                if best is None or e > ext or (e == ext and f > fol):
                    best, ext, fol = s, e, f
            if best is not None:
                found = min(k, fol)
        self_p = 3 + ext if found else 0
        cand = (
            list(seq[best + 3 : best + 3 + found]) if found else []
        )
        # Starvation probe for unarmed lanes.
        emitted = len(ctx)  # ctx = prompt + output[:i]; counts advance anyway
        if (
            self.corpus
            and not self.armed
            and not found
            and emitted >= self.probe_at
        ):
            self.probe_at = emitted + self.probe_every
            _, p = self._corpus_lookup(seq, k)
            if p >= self.arm_min:
                self.armed = True
        # Armed lanes: margin-gated displacement.
        if self.armed and self.corpus:
            corpus_cand, corpus_p = self._corpus_lookup(seq, k)
            if corpus_p > self_p + self.margin and len(corpus_cand) >= 1:
                self.last_match_p = corpus_p
                self.last_match_source = "corpus"
                return corpus_cand
        self.last_match_p = self_p
        self.last_match_source = "self" if cand else ""
        return cand


class NgramV2F:
    """Corrected v2 chain proposer: longest suffix match >= P_MIN over the
    stream (and optionally a global corpus), all occurrences, per-position
    majority extension (theta), full-k chain.

    Corpus selection follows AgSpec's preference margin (arXiv 2610.01108
    §3.1, App. C): a corpus match must beat the live match by `margin`
    pattern tokens, because a long hit in unrelated reference text is
    likely coincidental. Without it, "longest p wins" regressed mixed
    traffic — a long match inside an unrelated earlier output displaced
    better self matches (measured: json_fmt 72.9→64.6 tok/s).

    Continuation selection for corpus matches groups occurrences by their
    next-3 tokens and follows the most frequent group (AgSpec §3.1, App.
    D): the global corpus lost 18% agreement under first-occurrence
    selection in their replay.

    The naive port (alpha*p chain cap + product-score gate, no p floor)
    measured strictly worse than v1 — see docs/NGRAM_V2_RESEARCH.md.
    """

    def __init__(self, global_corpus=None, theta=0.6, margin=5):
        self.corpus = list(global_corpus or [])
        self.theta = theta
        self.margin = margin
        self.last_match_p = 0
        self.last_match_source = ""

    @staticmethod
    def _match(corpus, seq):
        """Longest p whose closing pattern of seq occurs earlier in corpus."""
        for p in range(min(len(seq) - 1, len(corpus) - 1, P_MAX), P_MIN - 1, -1):
            pat = seq[-p:]
            occs = [
                i for i in range(len(corpus) - p) if corpus[i : i + p] == pat
            ]
            if occs:
                return p, occs
        return None

    @classmethod
    def _chain(cls, corpus, occs, p, cap, theta, expected=False):
        """Greedy frequency-ranked chain; extends while the majority
        successor keeps >= theta of the remaining occurrences. Score is the
        product of conditional frequencies; expected=True returns the
        expected accepted count (sum of prefix products) instead."""
        chain, score, expected_sum = [], 1.0, 0.0
        ends = [o + p for o in occs]
        for depth in range(cap):
            counts = {}
            for e in ends:
                if e + depth < len(corpus):
                    t = corpus[e + depth]
                    counts[t] = counts.get(t, 0) + 1
            if not counts:
                break
            tok, cnt = max(counts.items(), key=lambda kv: kv[1])
            frac = cnt / len(ends)
            if frac < theta:
                break
            score *= frac
            expected_sum += score
            chain.append(tok)
            ends = [
                e
                for e in ends
                if e + depth < len(corpus) and corpus[e + depth] == tok
            ]
        return chain, (expected_sum if expected else score)

    @classmethod
    def _group_occs(cls, corpus, occs, p):
        """AgSpec continuation selection: group the matched occurrences by
        their next-3 continuation tokens, keep the most frequent group,
        ties to the group holding the most recent occurrence. Returns the
        winning group's continuation-start positions."""
        groups = {}
        for o in occs:
            e = o + p
            if e >= len(corpus):
                continue
            sig = tuple(corpus[e : e + 3])
            groups.setdefault(sig, []).append(e)
        if not groups:
            return []
        return max(groups.values(), key=lambda g: (len(g), max(g)))

    def propose(self, pending, ctx, k):
        seq = ctx + [pending]
        self_p, cand = 0, []
        m = self._match(seq, seq)
        if m is not None:
            p, occs = m
            chain, _ = self._chain(seq, occs, p, k, self.theta)
            if len(chain) >= 2:
                self_p, cand = p, chain
        # The corpus must clear the live match by the preference margin.
        best_p, best_chain = 0, []
        for s in self.corpus:
            m = self._match(s, seq)
            if m is None:
                continue
            p, occs = m
            if p <= self_p + self.margin or p <= best_p:
                continue
            ends = self._group_occs(s, occs, p)
            chain, _ = self._chain(s, ends, 0, k, self.theta)
            if len(chain) >= 2:
                best_p, best_chain = p, chain
        if best_chain:
            self.last_match_p = best_p
            self.last_match_source = "corpus"
            return best_chain
        self.last_match_p = self_p
        self.last_match_source = "self" if cand else ""
        return cand

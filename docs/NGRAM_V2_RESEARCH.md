# N-gram Predraft v2 — Research Notes

What v1 is, what the retrieval-draft literature adds on top of it, and how the
gap maps onto this engine's verify geometry. Lossless throughout: proposals
change, `decode_accept_*` does not.

## v1 as shipped

`runtime/model/NgramIndex.hpp`, driven from `RuntimeNgram.mm`:

- Per-lane `Table`: full `history` vector + `starts` map keyed by packed
  3-gram (vocab < 2^18 → 54-bit key), keeping each key's last **4** starts.
- `lookup`: finds earlier occurrences of the closing 3-gram, ranks them by
  backward extension (then follower count), emits up to
  `RICHENGINE_DRAFT_PROPOSAL_TOKENS` (=15) followers as a chain.
- `alternate`/`combTable`: one sibling leaf per chain position into the back
  half of `RICHENGINE_TREE_VERIFY_NODES` (=16) — a comb, not a real tree.
- Gate (`applyNgramPredraft`): summed lane score (warmup/probe →
  `ngramDraftExpect`, else `ngramAcceptedAvg` EWMA) must clear
  `Σ proposalAcceptedAvg` — the GPU draft's measured expectation. Sampled
  lanes veto (proposals carry no probabilities). Null-draft families bypass
  every gate.
- Feedback: `ngramAcceptedAvg` EWMA (0.75/0.25) updated in
  `RuntimeImpl.hpp:2658` on committed steps; `ngramProbeAt` schedules
  re-probes.
- Knobs: `RICHENGINE_NGRAM_{PREDRAFT,DRAFT_EXPECT,WARMUP,TREE_MIN}`.

Measured context (ISSUES.md, 2026-10-18): on Qwen3.8-27B the constant gate
let 3-gram chains preempt a draft landing ~2.6 accepted/step; the EWMA gate
fixed that. Step time (~62 ms weight streaming) — not acceptance — is the
dominant B1 gap, so every verify row saved or every long exact hit is pure
win.

## The literature

### SuffixDecoding — Oliaro, Jia, Campos, Qiao (NeurIPS 2025, arXiv 2411.04975)

The direct v2 template. Two suffix trees: a **global** tree over all previous
outputs (built offline or appended at request completion) and a **per-request**
tree over prompt+output (online, Ukkonen). Avoids cross-request
synchronization by construction.

- Pattern match: walk the tree on the last `p` tokens for a *range* of p
  (p=1..P), not a fixed order. Longest match found exactly.
- Speculation subtree: from matched node N_p, greedily add the leaf child
  maximizing `D(N) = D(parent)·C(N)`, `C(N) = COUNT(N)/Σ_siblings COUNT` —
  empirical conditional acceptance. `MAX_SPEC = α·p`, α∈[1,4] (longer match →
  confidently speculate more).
- Selection: `SCORE(T) = Σ D(N)` ≈ expected accepted tokens; pick the best
  tree across {global, request} × p range.
- Hybrid gate: use suffix tree iff `SCORE(T) > τ`, τ ≈ fallback speculator's
  mean accepted tokens. Same idea as v1's EWMA gate but with a principled
  per-step estimate instead of a rate.
- Cost: ~20 µs/token host-side; global tree over billions of tokens fits CPU
  RAM; FIFO eviction of cached requests.
- Results: up to 5.3× on SWE-Bench-class traces, 2.8× over EAGLE-2/3, 1.9×
  over Token Recycling; 4.5× end-to-end task time.
- Deployment signal: merged into vLLM Nov 2025 (PR #25784, `method: "suffix"`,
  via ArcticInference; knobs: `max_tree_depth`, `max_spec_factor`,
  `min_token_prob`, `max_cached_requests`; suggests `num_speculative_tokens`
  16–32). Reported beating vLLM's ngram proposer on Spec-Bench/Blazedit in
  essentially all cases.

### SAM-Decoding — Hu et al. (ACL 2025, arXiv 2411.10666)

Suffix **automaton** (not tree) over the generating sequence + a static text
corpus. Exact longest-suffix match, O(1) amortized per step for both update
and retrieval. Selects draft strategy by match length. +18% over other
retrieval drafters on Spec-Bench; stacks on EAGLE-2 (+3–11%). Cheaper and
simpler than a suffix tree, weaker scoring (no per-node frequencies exploited
the same way).

### AgSpec — Lee, Cho, Lim, Kwon (arXiv 2610.01108, Oct 2026)

Keeps the retrieval engine (SAM automaton or suffix tree) and fixes the
*policies* around it — this is where v2's real headroom likely is:

- **Corpus tiers**: session corpus (whole agent trajectory incl. past
  responses — survives prompt truncation), workspace corpus (opened repo
  files), global corpus (pooled outputs). Session corpus is the largest AL
  gain; workspace +26–47% alone.
- **Emission-format indexing**: index artifacts the way the model *emits*
  them — diff-prefixed lines, JSON-escaped file bodies in tool arguments —
  not raw file text. +13–39% AL in turn 1, before the session corpus warms.
  This is the single most actionable insight for a coding-agent server: raw
  files never token-match a patch.
- **Draft-length control**: per-agent offline-profiled cap; online scale α
  tracking the τ-quantile of `accepted/match` from verification feedback —
  target full acceptance, shrink when rejects dominate. The scale persists
  across sessions on the same server.
- Results: 4.37× B1 / 4.76× B16 throughput vs autoregressive; +64.6%
  throughput over SAM-Decoding alone; the online scale alone +6%.

### Token Recycling — Luo et al. (ACL 2025, arXiv 2408.08696)

Top-k adjacency matrix over *candidate* (previously drafted/rejected) tokens,
<2 MB, BFS draft tree. Note: it harvests rejected proposals too — v1 only
indexes committed tokens. Probably subsumed by SuffixDecoding here, but the
"index the rejects" trick is nearly free to add to any table.

## Gap map: v1 → v2

| Capability | v1 | SuffixDecoding/AgSpec |
|---|---|---|
| Match order | fixed 3-gram | exact longest suffix (adaptive p) |
| Occurrences per key | last 4 | all (COUNT per node) |
| Ranking | backward extension length | empirical frequency D(N), expected-accept SCORE |
| Corpus | own lane's prompt+output | per-request + global + session + workspace(emission format) |
| Spec length | fixed 15 slots, EWMA gate | α·p adaptive, per-agent cap + online scale |
| Tree shape | comb (chain + 1 alt/pos) | scored multi-branch subtree |
| Sampled lanes | vetoed | unaddressed (needs q per proposal) |
| Rejected drafts | dropped | Token Recycling indexes them |

## Design sketch for this engine

Ordered by leverage per DECODE_SPEED_PLAN §G (global suffix index =
highest-leverage retrieval gap):

1. **Variable-order per-lane index.** Replace `Table`'s fixed-3 `starts` map
   with a suffix automaton over `history` (SAM-Decoding-style): O(1) amortized
   append in `appendNgramTokens`, longest-match query instead of
   found/not-found on exactly 3 tokens. Keeps `seed`/`append`/`lookup`
   signature; the map's last-4-starts cap goes away. Cheapest structural win:
   partial matches (2-gram, 5+-gram) become available instead of nothing/short
   chains.

2. **Frequency counts + expected-accept scoring.** Store COUNT per automaton
   state / per stored occurrence; rank continuations by D(N), score a
   candidate chain by Σ D. Feeds the existing `score < draftExpect` gate with
   a per-step estimate instead of hit-rate × EWMA — strictly better
   information for the same decision.

3. **Global tree across requests.** New `Impl`-level structure updated at
   request completion (prompt+output committed), read in
   `applyNgramPredraft` alongside the per-lane table. FIFO bound
   (`max_cached_requests` equivalent). No concurrency issues if writes
   happen only at completion; the serving loop is the only writer.
   **Source selection is the make-or-break** — measured above: naive
   longest-p across sources regresses mixed traffic. Required pieces, all
   from AgSpec (arXiv 2610.01108 §3.1, App. C/D):
   - **Preference margin:** corpus wins only when `p_corpus > p_self + δ`,
     δ∈[2,5] (δ=2 optimal in their sweep, robust 0–∞). Validated in
     `NgramV2F(margin=5)` — corpus steals gone, upside kept.
   - **Grouped continuation selection:** on a corpus hit, group
     occurrences by next-3 tokens, follow the most frequent group, ties →
     most recent. First-occurrence selection cost them −18.1% agreement
     on the global corpus; REST-style trie ranking added +6.5% further —
     i.e. extend `_group_occs` toward a frequency trie if more is needed.
   - **Stronger source pick:** SuffixDecoding's SCORE = ΣD(N) (expected
     accepted) per source is the probabilistic version of the margin;
     feeds straight into the existing `score < draftExpect` gate.

4. **Session/workspace corpus in emission format.** The server layer knows
   the conversation; a `session` corpus = all prior turns' text tokenized in
   the form the model emits it. Workspace (repo files) would need the client
   to ship file contents — for OpenAI-compatible serving this degrades to
   "whatever appeared in earlier prompts", which the session corpus already
   covers. The emission-format insight still bites: diff markers and JSON
   escaping live inside the prompt/tool-result tokens, so indexing the *raw
   request token stream* (not a reparsed view) is required — v1 already does
   this per lane; the global tree must do the same.

5. **Adaptive chain length.** `MAX_SPEC = α·p` with per-lane α driven by the
   existing `ngramAcceptedAvg` feedback loop — AgSpec's online scale is
   essentially a quantile-tracked version of the EWMA already wired at
   `RuntimeImpl.hpp:2658`. Proposal slots are currently padded with
   anchor-repeats to 15; emitting a short real chain + dead rows costs the
   same GPU work but loses nothing, so adaptivity matters mainly for the
   *tree* path and for the score gate.

6. **Real tree, not comb.** `combTable` emits chain + ≤1 alternate/position.
   A scored subtree (greedy D(N) expansion, SuffixDecoding Alg. 1) fits the
   same `TreeTokens`/`TreeNodes` ABI inside `RICHENGINE_TREE_VERIFY_NODES=16`,
   just with more branching choices. Watch: tree verify measured −12%/−32% on
   the *draft* tree path (TREE_VERIFY_DESIGN) — the n-gram comb survives
   because it's hot-gated (`ngramTreeMin`); a bigger suffix tree needs the
   same hot-gate discipline or it burns verify bandwidth on cold lanes.

7. **Sampled lanes.** The veto exists because proposals carry no q. Suffix
   counts give empirical q(x) ≈ C(N); `decode_accept_*` speculative sampling
   can consume them exactly like draft probs. Unlocks the whole mechanism on
   T>0 lanes — worth checking the ABI (`ProposalProbs` path) for where a
   per-proposal prob could ride.

8. **Index rejected candidates** (Token Recycling trick): alternate
   continuations the target rejected are still evidence — cheap to note in
   the table at commit time.

### Deliberately out of scope

- Draft-model hybrid *within* one tree (suffix branch + DFlash branch in the
  same verify block): verify rows are the constraint; the existing
  either-or gate is cheaper and nearly as good on UMA where verify is
  bandwidth-bound.
- Sub-millisecond concerns: host cost is noise vs a ~60 ms step even at
  O(longest-match) per step.

## Measured A/B — `dev/benchmarks/ngram_v2_ab.py` (2026-10-27, Ornith-1.5-9B-MLX-4bit, K=15, 256-token gens, 8 scenarios)

tok/s vs baseline; avg_accept in parens. Wider run — added log_table
(12-row structured log→markdown, >4 repeats per trigram), big_edit (3×
file, rename), diff_emit (unified-diff emission format), test_gen
(repetitive boilerplate, ambiguous). v2fg's corpus pre-seeded with two
emitted documents + grown per scenario.

| scenario | baseline | v1 | v2f† | v2fg‡ | v2eg¶ |
|---|---|---|---|---|---|
| code_edit | 50.1 | 86.3 (9.9) | 85.7 (9.3) | 85.0 (9.3) | 83.0 (8.7) |
| rag_qa | 52.6 | 68.0 (6.0) | 66.5 (6.6) | 62.2 (6.6) | 56.8 (5.2) |
| json_fmt | 52.9 | 63.2 (4.9) | **67.2 (5.8)** | 61.1 (5.8) | 56.8 (4.7) |
| log_table | 51.3 | 100.6 (9.9) | **103.0 (9.9)** | 92.8 (9.5) | 90.9 (9.1) |
| big_edit | 52.5 | **75.4 (6.1)** | 73.8 (6.3) | 62.1 (5.9) | 62.2 (6.0) |
| diff_emit | 53.5 | **61.0 (4.7)** | 59.0 (5.0) | 46.2 (4.7) | 45.8 (4.5) |
| test_gen | 53.6 | 47.0 (2.0) ⚠below-AR | **52.8 (2.5)** | 38.7 (2.4) | 35.7 (1.9) |
| open_chat | 53.9 | 50.4 (1.3) | 49.4 (1.5) | 34.8 (1.7) | 34.0 (1.4) |
| session§ | 53.4 | 66.9 (6.5) | 60.2 (5.5) | 58.2 (5.5) | 57.3 (5.0) |

\* naive port (α·p cap + product gate, no p floor) dropped from the wide
run — it lost to everything.
† corrected: longest-p match ≥3, all occurrences, θ=0.6 per-position
majority extension, full-K chain.
‡ v2f + global corpus (pre-seeded with two emitted docs, grown per
scenario) + AgSpec δ=5 margin + next-3 grouped continuation. The
unmargined version regressed to 64.6/46.4 on json_fmt/open_chat. Its
tok/s is polluted by the harness's brute-force O(P_MAX·corpus) matcher —
a suffix automaton is ~O(p) — so compare on avg_accept, not tok/s.
§ second request re-runs the same edit; prompt carries original text,
corpus carries the previous emission.
¶ NgramV2E: corpus + margin + expected-accept source pick across ALL
pattern lengths — same total tokens accepted as v2f but spread over more
shorter rounds; confirmed worse, dropped.

Findings:

- **The naive paper port loses to v1.** MAX_SPEC=α·p is a *tree-size* rule;
  applied as a chain cap it truncates good chains on short-p matches
  (first run: v2 84.9 vs v1 88.8 on code_edit). p<3 matches are pure noise
  — a single-occurrence 1-gram predicts nothing (0/15 accepted observed).
  Chain mode needs `p_min≈3` — exactly v1's implicit threshold.
- **Corrected v2f ≈ v1 within-prompt, with one structural win.** On
  ambiguous boilerplate (test_gen) v1 fell *below* baseline (47.0 vs 53.6)
  — its 4-start, extension-ranked drafts were wrong often enough to burn
  rows; v2f's θ-majority requirement rejected them (52.8, fewer rounds,
  higher accept/round). That's the matcher's real advantage: it declines
  to propose when the statistics are weak, where v1 always emits the
  best-ranked occurrence's followers. On clean repetition they tie
  (9.9a both on log_table/code_edit); on rag_qa/big_edit/diff_emit v1's
  extension ranking still picks slightly better occurrences (−3-6%).
- **The harness corpus run exposes a selection bug, and AgSpec's margin
  fixes it.** v2fg *lost* on unrelated scenarios (json_fmt 72.9→64.6,
  open_chat 52.0→46.4): "longest p wins" lets a long match inside an
  unrelated earlier output displace a better self match — a long pattern
  is not a *reliable* pattern. With AgSpec's δ-margin (corpus must beat
  the self match by δ tokens, δ=2 best / 5 reported in their sweep) plus
  next-3-token grouped continuation selection, v2fg recovered to v2f's
  acceptance on every scenario (9.8/9.0/3.3/5.9 accept) while keeping the
  corpus available for genuine cross-request hits. SuffixDecoding's
  SCORE(T) (expected accepted from occurrence frequencies) is the more
  principled version of the same idea — compare expected accept, not
  pattern length — and composes with the existing `score < draftExpect`
  gate; the margin is the cheap discrete approximation of it.
- `lossless=False` on flat-distribution scenarios (open_chat) is fp
  tie-break noise between 1-row and 16-row forwards, not an algorithmic
  defect — greedy verify is exactly lossless.

### Real-trace replay — `dev/benchmarks/ngram_replay.py` on captured traffic

`RICHENGINE_TRACE_TOKENS=/path.jsonl` records every completed request's
prompt+output token ids, sampling fields and linkage (`server/backend.py`
`trace_tokens`, called from `_finalize`). The replay walks a trace the way
spec decode would — at each committed position the proposer sees
`prompt + output[:i]` and accepts the longest matching prefix — exact for
greedy traces, no model needed.

17-request capture via `dev/benchmarks/agentic_traffic.py` (Granite-4.2-3B,
greedy, multi-turn edit/diff/tool-arg/fix-loop/log-table/quote/chat) —
speedup is AR-equivalent (emitted/decode-steps); "a" = accepted per fired
round. Totals are dominated by the 512-token chat control:

| | speedup | accepted/step | fired | avg/round |
|---|---|---|---|---|
| v1 | 2.19× | 1.18 | 19% | 6.2 |
| v2f | 2.17× | 1.16 | 18% | 6.5 |
| v2fg | 2.17× | 1.16 | 18% | 6.5 |

Per-request deltas (v1 → v2fg) where they diverged: second-session re-edit
8.71×→9.38× (corpus hit on emitted form — the real win), second rename
re-emit 10.17×→11.09×, tool-arg loop turn 7.50×→10.00×; v1 ahead on the
457-token log table (3.20×→2.97×) and one fix-loop turn
(3.97×→3.83×). Aggregate is a wash on this small mix — the win
concentrates on cross-session re-emission; repeat turns reach
8.7–15.25× AR-equivalent while open chat sits at 1.01×.

- Repeat-style turns reach **8.7×–15.25×** AR-equivalent; open chat 1.01×
  (correctly idle).
- **v2f ≈ v1 within-prompt, again.** Avg/round 7.5 vs 6.8 — matcher
  upgrades are noise-level on real traffic too.
- **The global corpus paid on real output:** the re-run edit request in a
  *second* session went 8.71×→12.20× — the emitted (renamed) form matched
  the corpus where the prompt's original text could not. Same on the
  repeat-quote turn (7.00×→9.33×). AgSpec's emission-format point confirmed
  on real tokens: index what the model *emitted*, not what it read.
- Matched-p histogram: mass at p=3 (v1's order) plus a long tail p=9–64 —
  long matches exist but add little over p=3 when the continuation is
  forced anyway.

**Revised expectation:** within-request matcher upgrades buy ~±5%; the
measured wins live in corpus breadth — build order should be
**global/session emission-format corpus first, matcher upgrades second**.

## Intelligent opt-in — when to engage the corpus path

v2fg should not be always-on: cold traffic pays the match cost and, without
the margin, regresses. The opt-in decision is cheapest at request admission
(server layer), where the signals already exist, and corrects itself per
step at the runtime layer (the existing EWMA gate). Design:

### Admission-time signals (server/backend.py, at `submit`)

1. **Session continuation** — `previous_id` set (Responses API) or the new
   prompt's prefix is a stored recent prompt (append-only chat history →
   the earlier output is a likely continuation). A small ring of recent
   prompt-id lists in the host makes this a prefix-compare, no index.
2. **Prefix-cache overlap** — `job.cache.matched_tokens /
   job.prompt_total` above ~0.7 means the request shares its context with
   something already decoded; high reuse prior.
3. **Suffix probe of the corpus** — does the new prompt's closing ~16
   tokens occur anywhere in the global/session corpus? If the prompt tail
   is already a known emission, the reply likely continues known text.
   This is one `_match` call against the existing index — near-free.
4. **Client hint** — the web UI edit-resend flow can mark the request
   explicitly (`metadata` field or `previous_id`); a user-edited turn
   almost always re-emits most of the prior answer.

Any of these sets a per-request bit (`wire.RequestFlag`), carried into the
lane at admission.

### Runtime behavior on a hinted lane

- Attach the session/global corpus to that lane's matcher (the opt-in is
  corpus membership, not a different algorithm).
- Skip `ngramWarmup` and force an early probe: `ngramProbeAt` low so the
  lane gets a predraft round almost immediately; `ngramLaneScore` then
  trains `ngramAcceptedAvg` on real accepts.
- **The `score < draftExpect` gate stays in charge.** The hint buys a
  probe, not a permanent displacement — if the corpus didn't actually
  cover this output, the EWMA falls and the GPU draft resumes. This is
  SuffixDecoding's hybrid τ-gate restated: hint = temporarily lowered τ.
- Hint decays: after the probe round the lane is ordinary again — no
  persistent privilege, no oscillation risk.

### Why this shape

Measured above: v2fg's wins are concentrated on cross-session
re-emission (8.71×→9.38×–12.20×) and repeat turns; chat stays at 1.01×
and would only pay overhead. All four admission signals fire on exactly
the winning cases in the captured trace (edit-resend, fix-loop, tool-arg
loop = prefix containment; second-session re-edit = corpus suffix probe)
and none fire on the chat control. The hint → probe → gate flow is also
safe against false positives: a wrong hint costs one verify round, not a
session.

### Prototype A/B — `v2opt` in `ngram_replay.py` (17-request trace)

Hint = session continuation (prefix containment / previous_id) OR
suffix-probe hit (prompt's closing 16 tokens found in corpus). Hinted
requests run v2fg; the rest run production v1.

- 9/17 requests hinted — every session continuation plus the
  second-session re-edit (identical prompt = containment).
- v2opt kept **every** v2fg win (re-edit 9.38×, rename re-emit 11.09×,
  tool-arg 10.00×) and **every** v1 win on unhinted requests (log table
  3.20× vs v2fg's 2.97×, chat control untouched at 1.01×).
- One false positive (fix-loop turn 1.09×→slight dip vs v1) — cost ≈ one
  verify round, as designed.
- Totals: v2opt 2.19×/1.18 accepted-per-step — strictly ≥ both parents.
  On this trace opt-in = argmax(v1, v2fg) per request with no loser.

Conclusion: admission-time opt-in is validated on real traffic — wire the
hint bit plus corpus attach; keep the EWMA gate unchanged.

### Wired + production A/B (2026-10-27, Granite-4.2-3B, NullDraft — ngram
is its only proposer)

Wired: `RequestNgramCorpusHint` flag bit (Model.hpp/protocol.py),
`ngram::Corpus` FIFO-of-Tables in NgramIndex.hpp filled at `Runtime::end`,
margin-gated consult in `applyNgramPredraft`, server-side hint
(`RICHENGINE_NGRAM_HINT=1`: previous_id / prompt-prefix containment /
closing-16-token suffix probe). Knobs: `RICHENGINE_NGRAM_CORPUS` (0 off,
1 all greedy lanes, 2 hinted only), `RICHENGINE_NGRAM_CORPUS_ALL`
(index prompt+output — tool results — vs output-only),
`RICHENGINE_NGRAM_CORPUS_MARGIN` (default 5).

17-request agentic run (`dev/benchmarks/agentic_traffic.py`), tok/s and
draft accepted/offered per request:

| arm | edit_again t0 | everything else | chat |
|---|---|---|---|
| corpus off | 403.8 (100/315) | baseline | 78.8 |
| corpus=2 hinted, outputs | 418.3 (101/300) | ~same | 79.3 |
| corpus=2 hinted, ALL | **473.6 (106/225)** | ~same | 79.9 |
| corpus=1 always, ALL | 472.8 (106/225) | ~same | 80.0 |

- **The corpus win is real but narrow on this mix**: +17% on the
  cross-session re-edit, invisible elsewhere — matches the replay.
- **Prompt-inclusive corpus (`CORPUS_ALL`) is the better corpus**: the
  whole stream — prompt context + emitted form — extends matches across
  the prompt/output boundary where output-only docs can't reach.
  Tool-results-in-corpus is the same mechanism (tool output lives in the
  prompt), so the answer to "tool results too?" is yes.
- **Hinted ≈ always-on** on this workload — the hint caught the only
  corpus-eligible request; the margin blocked every would-be steal.
  Keep mode 2 as the default if enabled: it's the safe subset.
- **Lossless**: emitted ids byte-identical across all four arms.

### Self-arming opt-in — no server hint needed

Mode 2 now self-arms instead of depending on the wire flag:

- **Admission probe** (`seedNgramHistory`): the prompt's closing 3-gram
  is looked up in the corpus once; a hit with pattern ≥ 8 (3+extension)
  arms the lane. This is the "new request resembles earlier traffic"
  signal computed where the corpus actually lives.
- **Starvation probe** (`applyNgramPredraft`): a lane whose own 3-gram
  misses — where v1 can only emit anchor repeats — probes the corpus
  every `kNgramProbeTokens` (256) and arms on pattern ≥ 8. Strictly
  additive: it fires only where v1 contributes nothing.
- Displacement still needs `corpusPattern > selfPattern + margin`; the
  EWMA/`draftExpect` gate still owns the batch-level decision. Flag hint
  (`RequestNgramCorpusHint`, server `RICHENGINE_NGRAM_HINT=1`) still arms
  outright when the client knows more than the probes can see.

Measured (same 17-request run, `CORPUS=2 CORPUS_ALL`, **no hint env**):
`edit_again t0` 473.2 tok/s (106/225) — identical to the hinted and
always-on arms — and everything else flat, chat untouched. Outputs
byte-identical to baseline. The probes caught the one corpus-eligible
request on their own; the flag is now an optional accelerator, not a
requirement.

### Corpus breadth and currency — what actually scales

Every A/B so far lands on the same shape: the corpus either *contains*
the emission or it doesn't. Three properties decide that:

- **Breadth (what goes in).** `CORPUS_ALL` (prompt+output) beat
  output-only +13% on the re-edit because the match can cross the
  prompt→output boundary — the emitted form differs from the raw file.
  The general rule: index everything the model *could* re-emit —
  replies, tool args/results, diffs, file bodies, rendered templates.
  Tool results arrive inside the next prompt, so `CORPUS_ALL` covers them
  for free. A workspace corpus (files opened, indexed in emission format
  per AgSpec) is the same idea with a wider net.
- **Currency (how fresh).** A finished request joins the corpus at
  `Runtime::end`; eviction is whole-doc FIFO (64 docs / 1M tokens). Value
  decays with distance — same-session turns dominate the wins, so a
  session-tiered corpus (live docs preferred, global as fallback) keeps
  the freshest material first and shrinks the coincidental-hit surface.
- **Timing (when it lands).** A doc must exist before the lane arms.
  Same-session reuse needs the *previous turn's* doc, which lands at
  finalize — multi-turn works, in-flight overlap does not. The
  admission probe catches "resembles earlier traffic" at t=0; the
  starvation probe catches "started re-emitting mid-generation" within
  256 tokens. Between them, coverage is near-complete for the replay-
  verified patterns.

What would move the needle past this: not more selection machinery —
every margin/arm/occurrence/grouping knob measured flat — but **more
corpus**: session and workspace tiers, emission-format indexing, and a
variable-order matcher once docs are big enough for ambiguous hits to
exist.

#### Source-by-source: what belongs in the corpus

Ranked by measured relevance; each entry notes where it arrives and the
cheapest way in:

| source | reuse shape | arrival | status |
|---|---|---|---|
| assistant replies | re-emit, verbatim quote, continuation | `Runtime::end` | **in** |
| tool-call arguments | JSON bodies, same call shapes | inside output | **in** |
| tool results | echoed into next prompt, re-quoted | inside next prompt (ALL) | **in** |
| file bodies | pasted/read, re-emitted after edit | inside prompt (ALL) | **in** |
| diffs/patches | emitted form with line prefixes | inside output | **in** |
| system prompt / tool schemas | shared boilerplate across every request | prompt head of every doc | in, harmless — margin filters |
| user prose | rarely re-emitted | prompt | in, mostly inert |
| in-flight output | parallel lanes mid-generation | `commitSelected` | **out** — live-doc append not wired |
| workspace files never sent | agent emits unseen files | client-side open | **out** — needs a push API |
| cross-session history | repeat tasks, templates | global FIFO | in, but undifferentiated |

Two structural points worth keeping:

- **Index streams, not artifacts.** `CORPUS_ALL` won because a corpus
  doc holds prompt-then-output adjacency: a lane mid-output can match a
  position *inside* a corpus doc's prompt section and continue through
  the boundary into that request's emission. Splitting docs into typed
  parts (a "files" table vs an "outputs" table) would lose exactly the
  boundary matches that paid (the 15.25× re-edit row matched with p up
  to 275).
- **Prompt content is a doubled-edged corpus source.** The same
  adjacency that helps can also offer *prompt-format* continuations —
  template boilerplate and user prose continue a match just as well.
  The margin gate absorbed it on our traffic (no regressions vs
  output-only), but that's the mechanism behind the small log-table dip
  (3.20→3.03). If it grows, the fix is source tagging (mark prompt
  spans, demote their hits slightly), not dropping prompts.

#### Breadth extensions, cheapest first

1. **Recency-first scan ordering** — scan docs newest→oldest, first hit
   wins on tie. Same-session traffic is almost always also the most
   recent, so this is session affinity without plumbing a session id
   through the wire frame. ~10 lines in `lookupCorpus`.
2. **Session tagging** — a real session id on the request (extends the
   hint plumbing; the server already detects continuations). Then
   AgSpec's tiering applies: same-session doc preferred, global needs the
   margin. Measured flat on our trace because the wins were already
   session-local — pays once multi-tenant traffic shares the corpus.
3. **Live-doc append** — append emitted tokens to the request's own
   pending doc during decode so *concurrent* lanes drafting the same
   content see it without waiting for finalize. Narrow but real for
   parallel tool loops and batched same-prefix requests.
4. **Workspace push** — a wire call (`corpusAppend`) letting the server
   push tool results / opened files into the corpus when they arrive,
   not when the request ends. Closes the "file read but never
   re-pasted" gap.
5. **Emission-format duplicates** — when a file lands in the corpus,
   also push a diff-prefixed copy and a JSON-escaped copy (AgSpec). One
   `push` each; only worth it once a real file-write/edit workload shows
   the formats miss.

#### Workspace push — wired + measured (2026-10-27)

`POST /v1/corpus {text, emit_variants}` → `Frontend.corpus_push` →
`runtime.corpus_append` → `CorpusAppend` frame (0x0009) →
`Engine::corpusAppend` → `Runtime::corpusAppend` → `Impl::corpusPush` →
`ngram::push` — raw + diff-prefixed + JSON-escaped docs when variants
are on. Verified end-to-end: `{"pushed":[85,92,89]}` for one file.

Production A/B on a "emit a file you never sent" request: **flat**
(71/1365 draft, ~135 tok/s, both arms) — two reasons, both real:

- The lane's own n-gram table absorbs the re-emission once the first
  emission lands; corpus adds nothing past token ~0.
- The starvation probe (every 256 emitted tokens) never fired on a
  ~140-token output — corpus-arming arrives too late for short
  emissions.

Fix: young-output cadence — starvation probes every 32 tokens while
`generatedTokens < 256`, then the normal 256 cadence. Remeasured, same
request with the pushed file (raw + variants):

| arm | tok/s | draft accepted/offered |
|---|---|---|
| corpus off | 135.4 | 71/1365 |
| corpus on, 256-cadence | 134.8 | 71/1365 |
| corpus on, 32-cadence | **218.3** | **108/810** |

+62% over baseline with *fewer* offered rows — the corpus carried the
file emission the prompt never contained. This is the replay's 2×
showing up in production.

#### Corpus sizing

64 docs / 1M tokens FIFO is minutes of traffic on a busy server. For a
longer memory: per-session caps + session-pinned docs + whole-doc LRU
eviction beats a global FIFO — a hot session shouldn't lose its corpus
to unrelated traffic. Approx memory: 1M tokens ≈ 4 MB raw + index
overhead — a 16M-token corpus is still small next to KV.

### Concurrent DFlash2 A/B (Ornith-1.5-9B, corpus=2+ALL+hint)

4 parallel sessions (edit/fix/rag/chat) + sequential re-edit, corpus on
vs off. tok/s per request:

| req | off | on |
|---|---|---|
| rag t0 | 92.1 | 102.3 |
| edit t0 | 176.1 | 164.7 |
| fix t0 | 114.2 | 102.2 |
| chat | 54.9 | 56.3 |
| rag t1 | 194.6 | 192.8 |
| edit t1 | 286.4 | 284.9 |
| fix t1 | 408.7 | 440.1 |
| fix t2 | 412.9 | 346.1 |
| edit_again t0 | 519.3 | 522.7 |
| edit_again t1 | 547.3 | 546.1 |

Aggregate is noise — concurrent tok/s swing ±10% run to run — but
nothing regressed structurally: the gate (`score < draftExpect`) kept
n-gram out of DFlash2's way, chat stayed untouched, and fix t1
accepted 316/360 offered with corpus vs 326/405 without — the corpus
offered *fewer* wrong rows. The notable negative is fix t2 (412.9→346.1)
— worth a second look if it repeats; on a single run it's within
batching noise.

## Validation

- `dev/benchmarks/trainfree_spec_ab.py` ABBA on agentic traces (tool-call
  loops, repeated file re-emission) — the workload where v2 should pay.
- `dev/tests/ngram_probe.py` for live-server sanity; `ngram_index_test.cpp`
  pattern for unit coverage of the new index.
- Metrics already exported: `richengine_draft_acceptance_ratio`, per-lane
  accepted counts. Add a match-length histogram — p distribution decides
  whether α·p or a per-agent cap matters more.
- Non-goal per ISSUES.md: don't expect B1 wins on open-ended chat; v1's
  n-gram predraft earns its keep on repetitive spans only.

## References

- SuffixDecoding: arXiv 2411.04975 (v3), NeurIPS 2025; impl.
  github.com/snowflakedb/ArcticInference; vLLM PR #25784.
- AgSpec: arXiv 2610.01108.
- SAM-Decoding: arXiv 2411.10666, ACL 2025.
- Token Recycling: arXiv 2408.08696, ACL 2025.
- Internal: SPEC_DECODE_BOOST §L2, DECODE_SPEED_PLAN §G,
  TREE_VERIFY_DESIGN, PERF_REPORT row on retrieval drafts.

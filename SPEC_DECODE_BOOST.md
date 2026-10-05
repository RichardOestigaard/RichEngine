# Lossless Speculative Decoding — Decode Boost Analysis

Codebase-level findings on the draft→verify→accept pipeline, mapped against the
speculative-decoding literature. Complements PROPOSED_PLAN.md (host/dispatch
path), TO_EXPLORE.md / TO_EXPLORE2.md (Metal 4/4.1) and TODO.md. "Lossless"
throughout means: greedy output identical to the target's argmax path, sampled
output exactly from the target distribution (Leviathan et al. 2023; Chen et
al. 2023). Everything below preserves that — the proposals change, the
verification rule does not.

## What runs today

- DFlash2 block-diffusion draft: one parallel pass produces logits for 7
  positions (`SPLASH_DRAFT_PROPOSAL_TOKENS`, `ExecutionGeometry.h:6`), 8 query
  rows per lane. 5 layers + two-tap dynamic convolutions, conditioned on
  captured target hidden states (`DFlashDraft.cpp:213–256`).
- Selector keeps top-16 candidates per position
  (`draft_select_top16_sharded`, `sampling.metal:1040`), scores the 16×16
  predecessor/successor edge tables (`draft_select_edges`), then walks ONE
  chain (`draft_select_dflash`, `sampling.metal:1297`). `draft_select_plain`
  for non-DFlash2 drafts.
- Target verify: `lanes × SPLASH_TARGET_VERIFY_ROWS` (=8) rows, strictly
  row-index-causal inside the block (`causal_end`,
  `paged_attention_tile.h:104–106`). KV written to 7 scratch slots,
  `retained` committed (`SPLASH_SPECULATIVE_SCRATCH_TOKENS`).
- Acceptance (`decode_accept_dflash`, `sampling.metal:1576`): greedy =
  argmax match; sampled = `u·q < p` with residual correction drawn by
  `vocabulary_draw` over `max(0, w − q·kept_mass)` (speculative-sampling
  exactness, incl. penalties/min-p/top-k/top-p per row).
- Decode step (unconstrained): draft+select command → verify+accept+commit
  command (`Runtime.mm:1390–1411`, `ConstrainedDecodeTicket` shows the
  3-submission split when a host grammar mask intervenes).
- Draft context: ring buffer `SPLASH_DRAFT_SLIDING_WINDOW` = 2048;
  context commit replays the `retained` rows' KV
  (`addContextCommit`, `DFlashDraft.cpp:259–294`).

## The headroom, measured by the draft itself

From the DFlash 2 write-up (inco.ai/blog/dflash2): per-position recall@16 is
87.8–99.5%. The selector's best single chain accepts 4.61 tokens (T=0) /
4.25 (T=1); an oracle over the same top-16 candidate lists reaches 6.79.
**~2.2 tokens/step of selection headroom is already computed and sitting in
`buffers.candidates`** — the engine just only verifies one path through it.

Cost framing: the 27B Q4 verify pass is weight-bandwidth-bound (~16 GB/step
over ~270 GB/s ≈ 55–60 ms; draft ≈ 2–3 GB + its share of the shared LM head).
At B1–B2, widening the row count costs almost nothing in the FFN/attention
GEMMs (weights read once), and the paged attention splits read history KV
once per tile regardless of row count — KV bytes do NOT grow with tree width.
The scaling constraint is the shared LM head: 5120→248320 GEMM goes
compute-bound around ~24–32 rows, so tree width should stay ≤16 nodes/lane
without head-side work (see L5).

## Ranked levers

### L1. Tree verification over the existing candidate DAG — the big one

Verify a pruned tree of the selector's 16×7 DAG instead of the single best
chain. SpecInfer (Miao 2023), EAGLE-2, Sequoia and HSD (Zhou et al. 2026,
+12% over EAGLE-3 via sequence-level verify) all show chain→tree lifts
expected accepted length ~25–50% at modest width. Here the tree is free:
candidates + edge scores exist; `draft_select_dflash` walks best-first
anyway — emit the top-K nodes by cumulative score instead of one path.

What changes:

- **ABI**: `SPLASH_TARGET_VERIFY_ROWS` 8 → 16 (≤16/lane keeps the LM head
  sub-compute-bound; `SPLASH_VERIFY_CHUNK_STRIDE` and scratch slots follow).
  `SPLASH_UNIFORM_ACCEPTANCE`/`PROPOSALS` counts follow node count.
- **Attention**: per-row ancestor bitmask replaces `causal_end`
  (`paged_attention_tile.h:104`). A uint32 mask per row covers ≤32 nodes;
  committed-history pages are untouched. The `params.rows`-per-lane contract
  already tolerates variable rows.
- **GDN layers — the real cost.** `gdn_decode_scan` unrolls the delta
  recurrence over 8 chain rows per (lane, value head)
  (`decode/gdn.metal:133+`). A tree needs each node's ancestor state.
  Practical shape: DFS-ordered rows with a small branch-state stack — push
  state at each branch point, pop on backtrack; siblings adjacent. Cheaper
  still: **late-branch trees** (branch only at positions ≥3–4, where the
  selector is least confident) bound the rescan to short tails. The commit
  path gathers the accepted path (non-contiguous rows) instead of a
  `retained` prefix.
- **Acceptance**: greedy = DFS following the child's argmax match (SpecInfer
  tree-greedy, trivially lossless). Sampled = per-node `u·q < p` with
  per-node residual sets — `accept_sampled_lane`/`vocabulary_draw` extend to
  node-indexed candidate lists; or ship tree for greedy lanes first and keep
  chains on sampled lanes (per-lane row counts already supported).
- **Commit**: KV scratch slots for ≤16 nodes/lane; draft-context commit and
  GDN state commit take a node-path index list, not a count.
- **MoE caveat (35B-A3B)**: wider trees fan out more routed experts per
  layer — extra expert weight traffic where dense models pay none. Cap
  width lower on MoE, or dedupe routed experts across sibling rows.

Expected: chain 4.61 → ~5.5–6.0 retained/step at 16 nodes (half the oracle
gap is reachable at modest width per EAGLE-2/Sequoia curves); verify cost
+0–15% → net decode gain ~15–30% at B1 on the 27B. Largest single lever in
this document.

### L2. Retrieval/suffix proposals — nearly free on agentic workloads

The served workload is coding agents: tool-call loops, re-emitted file
bodies, repeated patches. SuffixDecoding (NeurIPS 2025) reports up to 5.3×
on SWE-bench-class traces, 2.8× over EAGLE, using only a host-side suffix
tree over prompt+output tokens. AgSpec (arXiv:2610.01108) adds the key
deployment detail: index what the model *emits* (diff lines with ` `/`-`
prefixes, JSON-escaped file bodies in tool arguments), not raw files — the
token stream never matches otherwise.

Two increments, both lossless (proposals verified identically):

- **L2a — proposal-source switch within today's 8 rows**: when the suffix
  match is long and the draft's top-1 mass at the anchor is low, substitute
  the suffix continuation as the lane's chain. Host-side decision before
  encoding; zero kernel work. Worthwhile only if draft-weak + repetition is
  common — measure the draft's per-lane proposal probability first (already
  in `ProposalProbs`).
- **L2b — merge into the tree (after L1)**: suffix continuations become
  extra branches at zero draft cost; this is where the big SuffixDecoding
  wins live (adaptive long matches).

Host cost: a suffix automaton over committed tokens per lane, updated in
`finalizeDecode`/`apply`. No GPU state. Deterministic; verify keeps
exactness.

### L3. Adaptive proposal length per lane (SpecDec++, AdaEAGLE, DISCO)

Block=7 is fixed for every lane and context. Acceptance is context- and
policy-dependent (T=0 vs T=1: 4.61 vs 4.25; degenerate/repetitive spans
reject early). Cheap version needs no learned predictor: track each lane's
rolling accepted-count (`AcceptedCount` is already read back in
`finalizeDecode`, `Runtime.mm:1642`) and shrink the lane's active rows when
the last N steps underperform. `VerifyAttentionPlan` params are per-lane
(`PagedAttention.cpp:151–166`); the draft still computes all 7 positions —
savings land in verify rows, LM-head rows and attention. Gains a few % at
B≥2 where lanes contend; mostly subsumed by L1 if it lands (a tree with
per-node scores IS an adaptive structure).

### L4. int4-KV acceptance tax — recover the 8-point gap

Measured (`README`, `DEVELOPMENT.md`): int8 KV → 63% draft acceptance vs
int4 → 55% at ~67K context, because quantization noise flips near-tie
argmaxes inside verify. Mitigations short of full int8:

- **Asymmetric K/V**: int8 K / int4 V. Attention-score error comes
  overwhelmingly from K; V precision affects output magnitude, not argmax
  order as strongly. Halves the byte penalty of full int8.
- **Recency tier**: keep the newest ~2–8 pages per lane bf16/int8, older
  pages int4. Recent keys dominate the near-tie decisions that flip draft
  tokens; storage is page-granular already (`KvPageTier`). Promotion =
  requantize on write, which page commit already touches.
- Either restores most acceptance at ~half the byte cost of int8-everywhere;
  int4 still wins at extreme context. Needs per-page format flags in
  `PagedKv`/`PageStorage` + per-format verify kernels (which exist — the
  format is kernel-selected per layer today).

### L5. Draft LM head — the draft's largest single read

The draft head IS the target's `vocabularyProjection` — 5120→248320 over 8
draft rows every step (`Runtime.mm:1391`, `DFlashDraft.cpp:245–251`), then
the selector keeps top-16. A dedicated small candidate head (low-rank or
coarse-vocab) trained for recall@16, not exact logits, would remove the
biggest draft-side weight read; recall tolerant to noise. Draft-training
item, not engine work — flag to the draft pipeline. Related: L1's ≤16-node
cap exists because this same head serves verify; a sparse "rescoring" head
for tree nodes (score only the ~64 union-of-candidates + argmax-check the
rest) could enable wider trees.

### L6. Ornith draft window 2048 → 4090-range

ISSUES.md: engine ring is `SPLASH_DRAFT_SLIDING_WINDOW` = 2048 while Ornith
drafts declare 4096 — draft quality (hence acceptance) degrades past 2048
context. Growing the ring is ring-buffer sizing + layout validation; direct
acceptance win on long contexts for the families that declare it.

### L7. Host turnaround — already specced, interacts with L1

PROPOSED_PLAN #1/#2 (submit-ahead ring depth 2, device-buffer params so the
attention section bakes) remain the top non-algorithmic levers. Note the
interaction: L1 widens rows, not dispatch count, so its win stacks; but
`executeCommandsInBuffer:indirectBuffer:` (TO_EXPLORE2) gains a second use —
the GPU could size the committed path / active row set without a host
readback.

### L8. HSD sequence-level verification (sampled lanes, post-L1)

Hierarchical Speculative Decoding (arXiv:2601.05724) is provably lossless
and lifts accepted tokens ~12% over per-token rejection sampling by
balancing excess/deficient mass across branches — but only makes sense once
there ARE branches. Revisit after L1 for the sampled path.

## Ruled out

- **Lossy acceptance** (typical acceptance / Medusa-style thresholds,
  relaxed `u·q<p`): breaks the engine's exactness contract — README promises
  greedy-identical output, and DEVELOPMENT documents exact min-p/top-k/top-p
  handling the tests presumably pin. The 2026 lossy-verification analysis
  (arXiv:2607.26627) shows the quality degradation is real, not just
  theoretical.
- **PEARL-style draft/verify overlap**: draft is already one parallel pass,
  and the next block's draft needs this block's captured target hidden
  states (`CapturedTargetHidden` written by verify). Serial dependency;
  nothing to overlap within a step. Submit-ahead (L7) is the correct
  version of this idea here.
- **Pre-verify of the anchor row** (PEARL pre-verify): splits the target
  pass in two — doubles weight reads per step. Never pays at B1.
- **Self-speculation / early-exit target layers**: hybrid GDN+attention
  architecture and quantized kernels make layer-skip verification a new
  training problem; DFlash2 already dominates on acceptance per draft-FLOP.
- **Bigger block size**: `SPLASH_DRAFT_PROPOSAL_TOKENS` is the trained
  block-1; the checkpoint fixes it at 8. Raising it is a draft-side
  (training) decision, and suffix decay (recall@16 → 87.8% by position 7)
  caps the return without draft changes anyway. L1's tree width is the
  better version of "more rows".

## Suggested order

1. **Measure the split** — `SPLASH_BACKEND_INSTRUMENTATION` /
   `benchmark-decode-profile`: draft pass vs verify pass vs host turnaround
   per step, at B1–B4 and short/long context. Sizes every claim above.
   Also log the existing `acceptedCount` distribution per lane to confirm
   the 4.3–4.6 figure on real traffic.
2. **L2a** (proposal-source switch) — small, host-only, independent of L1.
3. **PROPOSED_PLAN #1/#2** — submit-ahead + baked attention; multiplies
   everything else and is already designed.
4. **L1 tree verify** — the headline change. Prototype at width 12–16,
   late-branch shape, greedy lanes first; gate on
   `test-performance-real` ABBA + the `gguf_projection`-class numeric tests.
5. **L4 asymmetric/recency KV** — independent, restores int4 viability.
6. **L6** Ornith ring — mechanical, small.

## References

- Leviathan et al. 2023 / Chen et al. 2023 — speculative sampling, the
  lossless accept rule `sampling.metal` already implements.
- Chen, Liang, Liu — DFlash (arXiv:2602.06036, ICML 2026); Inco AI —
  DFlash 2 (inco.ai/blog/dflash2, Aug 2026): recall@16 vs oracle numbers.
- Miao et al. — SpecInfer (tree attention + token-tree sampling);
  Chen et al. — Sequoia; Li et al. — EAGLE-2/EAGLE-3 (dynamic draft trees).
- Zhou et al. — HSD (arXiv:2601.05724): lossless sequence-level verify,
  +12% on EAGLE-3; contrast arXiv:2607.26627 (lossy-verify failure modes).
- SuffixDecoding (NeurIPS 2025, ArcticInference); AgSpec
  (arXiv:2610.01108): retrieval drafts for coding-agent pipelines —
  index emitted formats.
- SpecDec++ (arXiv:2405.19715), AdaEAGLE (arXiv:2412.18910), PEARL
  (arXiv:2408.11850): adaptive/overlapped speculation policies.

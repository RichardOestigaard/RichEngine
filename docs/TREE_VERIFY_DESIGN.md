# DFlash Trees — Tree Verification Design

Extends verify from one 7-token chain to a ≤16-node tree per lane, built from
the DFlash2 candidate DAG the selector already computes. Lossless: acceptance
still walks target argmax (greedy) or u·q<p (sampled); the tree only widens
what is verified. Baseline (M5 Pro, 27B RichEngine, greedy): B1 ~100 tok/s,
7.11 retained/step, 87.3% draft acceptance.

## Tree shape (v1): the comb

Per greedy lane: the selector's chain (7 nodes) + at every proposal position
the 2nd-best successor of the chain's chosen predecessor (a leaf, ≤7).
1 anchor + 7 chain + ≤7 leaves = ≤15 nodes, DFS order:
`[anchor?, chain n0..n6, leaf p0..p6]`. A leaf at position i has parent =
chain node i−1 (anchor for i=0). Sampled/constrained/plain-draft lanes emit a
degenerate linear tree (chain) — one pipeline serves both.

Node table per lane, emitted by `draft_select_tree` (GPU) into
`TreeNodes[16] u32` packed {i8 parent, u8 depth, u8 position, u8 childCount?}
plus `TreeTokens[16] u32` (node[0] = anchor) + `TreeCount` u32.
A node is "drafted" iff it has children (child position = depth−1? — see
below). Anchor row: input=anchor, depth=0; proposal at position p has
depth=p+1 and its logits are checked by its parent's row.

## What changes, by subsystem

- **ABI**: `RICHENGINE_TREE_VERIFY_NODES=16` row stride per lane for every
  verify-row tensor (was 8). `RICHENGINE_SPECULATIVE_SCRATCH_TOKENS` stays 7:
  tree lanes skip `verify_attention_*_store` and commit K/V from staging
  post-acceptance. `RICHENGINE_VERIFY_CHUNK_STRIDE=32` covers ≤16 staged rows.
- **Verify input**: `InputTokens[row] = TreeTokens[row]` (grid lanes×16).
  Mask: per-row uint32 ancestor bitset over block rows, built by walking
  parent links (kernel-side, ≤7 hops).
- **Positions/RoPE**: row position = `logicalPosition + depth` (was +row).
  Depth is GPU-emitted → `rope_build_tables` reads the depth buffer.
- **Attention**: `causal_end` replaced by ancestor-mask bit test on block
  columns in `paged_attention_tile.h` (and fp8 tile). History pages
  unchanged. Per-lane `active_rows` (1..16) already parametric.
- **GDN**: prologue conv taps read per-node tap indices (3 ancestors:
  parent chain or `conv_state_in` for depth<3). Scan: post-state snapshot
  per row to scratch (`Snapshots[lane][head][16]`), leaf rows load
  `snapshots[parent]` as state-in instead of continuing the chain.
  Commit: replay `RetainedPath[i]` rows instead of prefix `0..retained`.
- **KV commit**: tree lanes skip the per-row page store; new
  `kv_commit_tree` copies `path[i]`'s staged K/V → page slot
  `committed + i` for i<retained. Chain lanes keep the existing store.
- **Sampling**: strides `lane*8→lane*16`, masks `lane*9→lane*17`,
  `drafted`/position via node table (linear table ⇒ identical for chain
  lanes). `vocabulary_draw` untouched (runs only on chain rows in v1).
- **Acceptance**: `decode_accept_tree` (greedy): DFS walk argmax→matching
  child, emit path + final argmax; writes `RetainedPath`, `RetainedCount`,
  `AcceptedCount`, `OutputTokens` (same contract: path tokens + last =
  correction/bonus). Sampled: existing `accept_sampled_lane` on its linear
  table — `draft_ids + pos*16` maps 1:1.
- **Draft context commit**: iterate `RetainedPath` (source row) → ring
  `start + i`; context projection runs all ≤16 rows, commit gathers.
- **Constrained decode**: tree off (mask simulation needs the chain;
  `simulationTokens` unchanged).

## Gating

`VerifyMode` in bootstrap config: `RICHENGINE_VERIFY_TREE` env (0/1) or
`--verify-tree` flag. Auto-on only when `DraftKind::DFlash2` AND the batch
has ≥1 unconstrained greedy lane AND width ≤ tree cap (B4 rows=64 → LM head
compute-bound; cap may fall back to chain by width — decide on benchmark).
`fp8e4m3` KV → off (near-tie noise compounds width). Escape: env kill
switch + per-lane fallback to chain table.

## Risk register

- GDN snapshot/restore is the novel kernel work (DFS state plumbing).
- LM head at B4×16=64 rows is compute-bound → width-capped gating.
- `commitSelected`/penalty words count tokens via the retained path — path
  order must be preserved into `outputTokens`.
- `decode_sample_penalize_verify` counts draft occurrences via input-row
  prefix — chain lanes unaffected (linear table); tree lanes with active
  penalties must map ancestors, not prefix rows (v1: tree lanes use
  ancestor-walk popcount — small kernel change).

## Measured outcome (M5 Pro, Qwen3.8-27B-RichEngine, decode benchmark)

Correctness achieved: tree output is bit-identical to chain (output hash
8845173885570541823 at widths 1-2 under tree, 3-4 gated to chain; strict
physical-width, lane-equality and DFlash accounting assertions all pass).
The tree never rescued a draft miss on this prompt — 87% top-1 acceptance
left no leaf accepts — so identical tokens at ~2x verify rows is a net
loss: B1 93 vs 106 tok/s (-12%), B2 120 vs 184 tok/s (-32%).

`RICHENGINE_VERIFY_TREE` is therefore opt-in (default chain, `=1` enables).
The machinery stays: it is exact, and it pays off only on workloads where
the draft's second-best rescues a meaningful share of misses.

Bugs found by the correctness work:

- `richengine_attention_page_softmax` loaded/stored only 8 of `TokensPerLane`
  elements; the tree tile's 16-token lanes read uninitialized scores —
  forward corrupted at every row (extended the loop to TokensPerLane).
- `verify_tree_attention_reduce_gate` used `active_rows` (15) as the
  per-lane packed/hidden stride instead of the 16-row capacity — lane 1
  read lane 0's dead row.
- `richengine_compact_verify_tree_phase` treated packed-INT4 values as
  dimension-major nibbles; values are token-major [token][dim pair]
  bytes, exactly like keys — RMW corrupted the value slab.

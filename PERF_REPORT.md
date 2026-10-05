# Splash Performance & Paper Survey

Consolidated research: decode, prefill, KV cache. Generated 2026-10-05.

---



# A. Decode Architecture (codebase at `/Users/richardgoestigaard/Documents/GitHub/splash`)

## A.1 End-to-end decode step

`runtime/model/Runtime.mm:2852` `Runtime::decodeAsync` is the single entry. Per step it builds one `CommandGraph` and submits it via `backend.submitCommandAsync` (line 3018). The step sequence:

1. `prepareDecodeLane` (line 1406) — host writes anchor token + RoPE positions per lane.
2. `applyAnePredraft` / `applyNgramPredraft` (lines 1693/1832) — optional proposal injection that skips the GPU draft entirely.
3. `treeVerifyBatch` decides chain vs. tree mode (line 2920); tree requires all-greedy, unconstrained, unpenalized DFlash lanes, ≤2 lanes wide.
4. `addRopeTables`, `encodeBatchEmbedding` (2087) — input embedding.
5. `encodeDraftBatchGraph` (1464) → `DFlashDraft::addDecode` + `addSelection` — draft forward + candidate selection.
6. `encodeBatchVerifyInput` (2098) or `encodeBatchVerifyTreeInput` (2110) — builds verify input rows (chain = anchor+7 proposals; tree = DFS node tokens + ancestor masks).
7. `encodeTargetVerifyBatchForward` (1862) → `QwenTarget::addVerify` — full target forward over 8 (chain) or 16 (tree) rows/lane.
8. `encodeTargetVerifyBatchPolicy` (1972) → `ops::Sampling::addVerify`/`addVerifyTree` — penalties, argmax or sampled draws.
9. `encodeBatchAcceptance` (2042) → `decode_accept_dflash`/`decode_accept_tree` — acceptance walk.
10. `encodeBatchTreeKvCompact` (2133) — tree lanes compact retained-path K/V into pages.
11. `encodeBatchGdnCommit` (2162) — replay accepted-path rows into GDN recurrent state.
12. `encodeDraftStateCommitBatch` (1999) → `DFlashDraft::addContextCommit` — write retained rows' K/V into the draft ring.

Serialization: `runtime/engine/Engine.cpp:253–288` — engine holds exactly one `pending_` ticket; `tick()` waits `ticket->ready()` → `wait()` → `apply()` → `scheduler_.next()` → `model_.submit()`. GPU idles during host turnaround; `PROPOSED_PLAN.md` ranks submit-ahead ring depth 2 as the largest lever.

## A.2 Speculative decoding — DFlash 2 draft

- `runtime/model/DFlashDraft.{hpp,cpp}` — `DraftKind::DFlash2` (also `Plain`, `DSpark`). 5-layer block-diffusion draft: per layer = input RMSNorm → dynamic projection (hidden→dynamicSize 1280) → two-tap dynamic convolution (`draft_conv`) → fused QKV → block-bidirectional attention over a KV ring → MLP with same conv pattern. Conditioned on `CapturedTargetHidden` written by the verify pass (`DFlashDraft.cpp:160–264`).
- Geometry (`runtime/metal/abi/ExecutionGeometry.h`): `SPLASH_DRAFT_QUERY_ROWS=8`, `SPLASH_DRAFT_PROPOSAL_TOKENS=7`, `SPLASH_TARGET_VERIFY_ROWS=8`, `SPLASH_TREE_VERIFY_NODES=16`, `SPLASH_MAXIMUM_BATCH_WIDTH=4`, `SPLASH_DRAFT_SLIDING_WINDOW=2048`, `SPLASH_DRAFT_CANDIDATES=16`, `SPLASH_DRAFT_SELECTOR_RANK=256`.
- Selector (all on GPU, `runtime/metal/kernels/decode/sampling.metal`, host glue `runtime/ops/DraftSelector.cpp`):
  - `draft_select_top16_sharded` (:1170) — top-16 per position over 248K vocab.
  - `draft_select_edges` (:1247) — 16×16 predecessor/successor edge scores via rank-256 codebooks (`predecessorCodebook`/`successorCodebook`).
  - `draft_select_dflash` (:1336) — one-thread-per-lane best-first chain walk; sampled lanes draw per-position softmax over the 16 candidates.
  - `draft_select_tree` (:1406) — additionally emits "comb" tree: chain rows 1–7 plus 2nd-best sibling leaves at rows 8–14.
  - `draft_select_plain` (:1515), `draft_select_dspark` (:1568) — other draft families; DSpark adds a low-rank Markov head.
- Acceptance (`sampling.metal`):
  - `decode_accept_dflash` (:1805): greedy = argmax-match loop (`accept_greedy_lane` :1791); sampled = `u·q < p` rejection with residual correction (`accept_sampled_lane` :1677) drawn by `vocabulary_draw` (:695) over `max(0, w − q·kept_mass)` — exact Leviathan/Chen speculative sampling, including min-p/top-k/top-p and penalties per row.
  - `decode_accept_tree` (:1844): DFS walk, child = argmax match; emits `RetainedPath`/`RetainedCount`/`AcceptedCount`. Greedy lanes only.
  - `tree_leaf_patch` (:1914): splices ANE-produced leaf alternates gated on an atomic serial.
- Sampled-row pipeline: `decode_sample_mass_sharded` (:131) → `decode_sample_vocabulary_search` (:853) → `decode_sample_vocabulary_draw` (:877); greedy: `decode_sample_argmax_sharded`/`_reduce` (:1744/:1777). Host: `ops/Sampling.cpp` `addVerify` (:141), `addVerifyTree` (:158), `addSelection` (:173).
- Tree verify: implemented end-to-end (ancestor bitmasks in `paged_attention_tile.h`, tree GDN kernels `verify_tree_gdn_fused*` at `decode/gdn.metal:976–985`, `verify_gdn_commit_tree` :543, `kv_commit_tree` via `PagedAttention::addVerifyTreeCompact`, `Runtime.mm:2958–2999` wiring) but **opt-in**: `SPLASH_VERIFY_TREE=1`/`--verify-tree`. `TREE_VERIFY_DESIGN.md` reports bit-identical output but −12%/−32% tok/s — the 87% top-1 acceptance leaves nothing for leaves to rescue.
- ANE/n-gram alternates (`ANE_DRAFTING.md`, `runtime/model/AnePredictor.mm`, `Runtime.mm:1567–1859`): `SPLASH_ANE_MEDUSA` (CoreML leaf alternates), `SPLASH_ANE_PREDRAFT` (predicts next step's chain), `SPLASH_NGRAM_PREDRAFT` (host-side 3-gram prompt-lookup with last-two-occurrences index and EWMA acceptance gating). Draft-tail reuse (Ouroboros-style) was implemented, measured (~1% tail acceptance under greedy), and removed.

## A.3 Metal decode kernels (`runtime/metal/kernels/decode/`)

- `paged_attention.metal` — `verify_attention_{q8,int4,bf16}_split*` (:106–120): split-KV flash decode over 32-token pages, `causal_end` row-causal masking for chains / ancestor bitset for trees; `verify_attention_reduce*` (:217+) merge splits. `fp8_attention.metal` + `fp8_attention_store.metal` for `fp8e4m3` KV.
- `gdn.metal` — Gated DeltaNet decode scan over the 8 verify rows (`gdn_decode_scan` :140) fused with weight reads (`verify_gdn_fused*` :656–676, including `table64`/`table16` Q4-input variants), plus `verify_gdn_commit*` (:388–389) prefix-commit and tree variants.
- `linear_q4.metal`, `linear_q4_sgmatrix.metal`, `linear_q4_grid_split.metal`, `linear_gguf_sgmatrix.metal` — bandwidth-bound weight GEMVs using Metal 4 `matmul2d`/simdgroup cooperative tensors (`common/q4_mpp_tiles.h`, `common/sgmatrix.h`, `common/gguf_sgmatrix.h`).
- `attention_qkv.metal` — verify QKV prepare/gate; `draft.metal` — draft conv/QKV/split-attention kernels listed above.
- `sampling.metal` — selector + policy + acceptance (above).
- Ops layer: `ops/PagedAttention.cpp` (`verifyParams`/`verifyTreeParams`, kernel-name assembly `verify_attention*`/`verify_tree_attention*`, `:153–166,:264–273,:346–377,:508`); `ops/GDN.cpp`, `ops/DraftAttention.cpp`, `ops/Linear.cpp` (`decodePlan`), `ops/Sampling.cpp`, `ops/DraftSelector.cpp`.

## A.4 Batching, sampling, quantization

- Batching: `runtime/engine/Scheduler.cpp` — continuous batching, max 4 lanes (`nextDecode` :363); decode batches require equal priority + same `constrained` flag + `decodeStage`; decode-vs-prefill arbitration via `decodeDebtMilliseconds_` (`next` :237). Constrained (grammar-mask) lanes take `ConstrainedDecodeTicket` (`Runtime.mm:2354`) — a 3-submission split letting the host inject masks.
- Sampling: full min-p/top-k/top-p/temperature + repetition/presence/frequency penalties (`decode_sample_penalize*`, `rebuildPenaltyWords`), per-lane uniform streams (`SPLASH_SAMPLING_UNIFORMS` layout, `abi/Sampling.h:11–24`).
- Weight quantization: 19 GGUF formats in `abi/QuantFormat.h` (Q4_K, IQ4_XS/NL, Q5K, Q6K, Q3K, Q8_0, IQ3_S, Q2K, IQ3_XXS, IQ2_XXS/XS, IQ2_S, IQ1_S/M, Q4_0/1, MXFP4, PQ2_0) in a tiled repack layout; native "affine Q4" planes for packaged models.
- KV quantization: `ops/PagedKv.hpp:16` — `Int8`/`BFloat16`/`Int4`/`Float8E4M3` per 32-token page, fp32 scale per token-head. `TurboQuant_ANLYSIS.md` documents measured rejection of TurboQuant, asymmetric K/V, recency windows, and mxfp4/nvfp4 KV: V-precision dominates on this model (4 KV heads × 256 dim); int4 costs ~8 pts draft acceptance at 67K ctx (63%→55%).
- Server layer (`server/`): Python frontend streams via HF `DecodeStream` (`server/backend.py:134`); engine is the C++ runtime behind `server/runtime.py`/`backend.py` job submission — all decode logic is native.

# B. Paper ↔ code mapping

| Paper | Authors/Year | Relevance → code |
|---|---|---|
| Fast Inference from Transformers via Speculative Decoding | Leviathan, Kalman, Matias — ICML 2023 (arXiv 2211.17192) | Speculative decoding + lossless accept rule → `accept_sampled_lane`/`decode_accept_dflash` (`sampling.metal:1677,1805`) |
| Accelerating LLM Decoding with Speculative Sampling | Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper — DeepMind 2023 (arXiv 2302.01318) | Modified rejection sampling `u·q<p` + residual draw → `vocabulary_draw` (`sampling.metal:695`) |
| SpecInfer: tree-based speculative inference & token tree verification | Miao, Oliaro, …, Jia — ASPLOS 2024 (arXiv 2305.09781) | Tree attention masks, token-tree verify → `draft_select_tree`, `decode_accept_tree`, tree ancestor masks (`TREE_VERIFY_DESIGN.md`) |
| Medusa | Cai, Li, Geng, Peng, Lee, Chen, Dao — ICML 2024 (arXiv 2401.10774) | Multi-head leaf proposals → `SPLASH_ANE_MEDUSA`, `tree_leaf_patch` (`sampling.metal:1914`); its "typical acceptance" (lossy) explicitly ruled out |
| EAGLE / EAGLE-2 / EAGLE-3 | Li, Wei, Zhang, Zhang — ICML 2024 / EMNLP 2024 (2406.16858) / NeurIPS 2025 (2503.01840) | Hidden-state-conditioned drafting + dynamic draft trees → DFlash conditioning on `CapturedTargetHidden`; comb tree in `draft_select_tree`; EAGLE-3 is the cited baseline |
| DFlash: Block Diffusion for Flash Speculative Decoding | Chen, Liang, Liu — arXiv 2602.06036 (ICML 2026); DFlash 2 = Inco AI blog | The draft architecture itself: one parallel pass, masked block, KV injection of target features → `DFlashDraft::addDecode` |
| Sequoia | Chen, May, Svirschevski, Huang, Ryabinin, Jia, Chen — NeurIPS 2024 (arXiv 2402.12374) | Optimal tree shape + robust sampled verify → informs tree-width analysis in SPEC_DECODE_BOOST L1 |
| HSD — Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding | Zhou, Huang, Li, Wu, Wang, Zhang, Lin, Cheng — arXiv 2601.05724, ICLR 2026 | Sequence-level lossless verify (+12% on EAGLE-3) → cited as L8, not implemented |
| SuffixDecoding | Oliaro, Jia, Campos, Qiao — NeurIPS 2025 (arXiv 2411.04975) | Suffix-tree retrieval drafts for agentic workloads → `SPLASH_NGRAM_PREDRAFT` is a simpler version (single-request 3-gram index vs. global suffix trees) |
| AgSpec | Lee, Cho, Lim, Kwon — arXiv 2610.01108 | Retrieval drafts over *emitted* formats for coding agents → SPEC_DECODE_BOOST L2, partially realized by n-gram predraft |
| Prompt-lookup decoding | Saxena 2023 (GitHub; in HF/vLLM) | Model-free n-gram proposals → `applyNgramPredraft` (`Runtime.mm:1832`) |
| Ouroboros | Zhao, Huang, Han, Xiao, Liu, Sun — EMNLP 2024 (arXiv 2402.13720) | Draft-tail reuse → implemented, measured, removed (`ANE_DRAFTING.md` "Rejected") |
| PEARL | (smart-lty) — arXiv 2408.11850, ICLR 2025 | Draft/verify overlap + adaptive draft length → explicitly ruled out (parallel draft already; pre-verify doubles weight reads) |
| SpecDec++ | Huang, Guo, Wang — ICML 2024 (arXiv 2405.19715) | Learned adaptive candidate length → L3 (host tracks `AcceptedCount` instead — heuristic, unlearned) |
| AdaEAGLE | Zhang, Wang, Ma, Zhu, Chen, Lan, Yu — arXiv 2412.18910 | Draft-length predictor → same L3 lever |
| Lookahead decoding (Jacobi) | Fu, Bailis, Stoica, Zhang — ICML 2024 (arXiv 2402.02057) | n-gram mining from verify trajectory → adjacent to n-gram predraft; not implemented |
| TurboQuant | Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026 (arXiv 2504.19874) | Rotation+QJL KV quant → tested and rejected (`TurboQuant_ANLYSIS.md`, `dev/turboquant_experiment.py`) |
| KIVI | Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu — ICML 2024 (arXiv 2402.02750) | Per-channel K / per-token V + residual window → recency-window variant tested and *hurt*; K-asymmetry found backwards on this model |
| KVQuant | Hooper, Kim, Mohammadzadeh, Mahoney, Shao, Keutzer, Gholami — NeurIPS 2024 (arXiv 2401.18079) | Pre-RoPE non-uniform KV quant → considered, rejected (calibration + paged-layout conflict) |
| QuaRot / PolarQuant | Ashkboos et al. NeurIPS 2024 (2404.00456) / AISTATS 2026 | Rotation-family quant → same family as TurboQuant; rejected as invasive model-level change |
| Gated DeltaNet | Yang, Kautz, Hatamizadeh — ICLR 2025 (arXiv 2412.06464) | The hybrid target's GDN layers → `decode/gdn.metal` scan/commit kernels; tree-verify DFS state snapshotting is the novel complication |
| LLM in a flash | Alizadeh et al. — Apple, ACL 2024 (arXiv 2312.11514) | Bandwidth-budget framing for on-device LLM → conceptual basis; splash adds SSD KV/state offload (`KvPageTier`, `WriteBehind`) rather than weight paging |
| On-device SD on Apple Silicon (atomgradient study) | ~2025 preprint | Draft:target speed ratio > acceptance rate on unified memory → consistent with splash's parallel-draft (near-zero marginal draft cost) choice |
| Lossy-verification analysis | arXiv 2607.26627 | Cited to justify keeping exactness contract |

# C. Gaps and opportunities

1. **Sequence-level (HSD) acceptance — not implemented.** Tree lanes are greedy-only; sampled lanes keep token-wise rejection on a linear table. HSD claims +12% accepted tokens losslessly; SPEC_DECODE_BOOST L8 flags it as post-L1 work. `vocabulary_draw`/`accept_sampled_lane` are the extension points.
2. **Richer tree shapes.** Comb (chain + 2nd-best leaves) is implemented but gated off — measured net loss at 87% top-1 acceptance. Sequoia-style DP tree selection or EAGLE-2-style value-weighted expansion over the already-computed 16×7 candidate DAG (the ~2.2-token oracle gap in `buffers.candidates`) is the documented big lever; needs workload where top-1 fails (sampled lanes, longer horizons, weaker draft).
3. **Adaptive proposal length** — fixed 7-token block; SpecDec++/AdaEAGLE-style per-lane shrinkage is sketched (L3) but unimplemented; `AcceptedCount` readback already exists (`Runtime.mm:2219`).
4. **Global/cross-request suffix index** — n-gram predraft is per-lane 3-gram only; SuffixDecoding's global suffix tree over past outputs and AgSpec's emission-format corpora are not indexed. Highest-leverage retrieval gap for the agentic workload.
5. **Submit-ahead pipelining** — one command in flight (`Engine.cpp:253–288`, `MetalBackend.mm:302`); PEARL-style overlap is ruled out, but ring-depth-2 submit-ahead + device-buffer params (PROPOSED_PLAN #1/#2) are designed, not built.
6. **Draft LM head cost** — draft shares target's 5120→248320 head; a dedicated low-rank/sparse "rescoring" head (L5) for wider trees is a draft-training item, not in code.
7. **KV asymmetry opportunity inverted by measurement** — literature says K-needs-bits; this model needs V≥6–8 bits (`TurboQuant_ANLYSIS.md`). Per-page format flags/`KvPageTier` mixed formats exist as a lever but the asymmetric split was measured unprofitable.
8. **ANE pipeline not overlapping target** — ANE predictors run between steps (serial queue, bounded wait); no true GPU↔ANE in-step overlap of draft vs. verify (Apple "mirror"/cross-burst pipelining idea in the wild, not present here).
9. **`vocabulary_draw` serial sections** (`sampling.metal:746–794`) and `decode_linear_q4_prepare` fusion (`linear_q4_sgmatrix.metal:137`) — known micro-gaps, PROPOSED_PLAN #3/#4.



# PREFILL Research Report — splash

## (A) Prefill Architecture Summary

### Request lifecycle & scheduling

- Phases: `Queued → WaitingResources/WaitingPrefix → Prefill → Decode → WaitingMask → Completed` — `runtime/engine/Scheduler.hpp:15-25`.
- The `Engine` runs **one GPU command at a time**: `tick()` waits `pending_->ticket->ready()` → `wait()` → `apply()` → `scheduler_.next()` → `model_.submit()` — `runtime/engine/Engine.cpp:253-288`. GPU idles during host turnaround (flagged in `PROPOSED_PLAN.md:8-24`).

### Chunked prefill + continuous batching

- **Chunk budget**: `ExecutionLimits::prefillTokenBudget = 2048` rows (`runtime/model/Model.hpp:379`; mirrored `SPLASH_PREFILL_TOKEN_BUDGET 2048` in `runtime/metal/abi/ExecutionGeometry.h:23`). 8,058-token prompt = 4 chunks.
- **Packed multi-sequence prefill**: `Scheduler::planPrefill` (`runtime/engine/Scheduler.cpp:282-321`) packs up to `maximumBatchWidth = 4` sequences' chunk rows into one 2048-row command, ordered shortest-remaining-first with anti-starvation (`kMaximumOvertakes`, `Scheduler.cpp:12-13`).
- **Adaptive chunk size**: `Scheduler::prefillBudget` (`Scheduler.cpp:323-361`) halves the row budget while `rows × measuredMsPerToken > kContendedPrefillMilliseconds (500)` whenever decoders or short prefills contend — a measured-time analogue of Sarathi-Serve's token-budget throttle.
- **Prefill↔decode interleave, not fusion**: `Scheduler::next` (`Scheduler.cpp:237-264`) — *"Prefill and decode use different Metal graphs and cannot be packed into one command"* — alternates whole commands: priority first, then a `decodeDebtMilliseconds_` ledger (decode time owed per unit prefill time, `--decode-share`, default 0.5, `runtime/engine/Engine.hpp:44`, `server/serve_options.py:307`), else strict alternation.

### Packed prefill encoding

- `Runtime::submit` → `prefillAsync` → `Impl::encodePackedPrefillGraph` (`runtime/model/Runtime.mm:2700-2723, 1259-1404`). All sequences share one row-space (`PackedPrefillBatch`), each with a `ChunkedPrefillParams` (`committed_tokens`, `chunk_tokens`, `chunk_stride`, page table) — `runtime/ops/PagedAttention.hpp:65-95`.
- `QwenTarget::addPrefill` (`runtime/model/QwenTarget.cpp:353-395`) encodes all layers in one `CommandGraph`: norm → input projection → mixer → FFN, with per-sequence `PrefillAttentionPlan`s (`ops::PagedAttention::prefillPlan`, `runtime/ops/PagedAttention.cpp:208-228`).

### Kernels used in prefill

| Stage | Kernel | Notes |
|---|---|---|
| GEMM (affine Q4) | `prefill/linear_q4.metal` — `prefill_linear_q4_n128/n256[_residual|_up_silu_sums]` | MPP `matmul2d` on `uint4b` weights (Neural Accelerators on Apple10/M5), TileM=32, fused epilogues (residual, SiLU-gate, output sums). Input sums precomputed by `prefill_linear_q4_sums32` / norm sums variants instead of dequantizing A. |
| GEMM (GGUF) | `shared/gguf_linear.metal` — `gguf_prefill_<format>`, `common/gguf_staged_tile.h` | Stage dequantized tiles, run `matmul2d`. |
| Attention | `prefill/attention_qkv.metal` (fused QK-norm+RoPE+layout), `prefill/paged_attention.metal` — `prefill_attention_{bf16,q8,int4}_split` + `prefill_attention_reduce`, plus fp8 variants in `fp8_attention*.metal` | **FlashAttention-style online softmax** over Page32 paged KV (`common/paged_attention_tile.h`, `splash_paged_attention_tile`), with **split-KV** (`prefillSplits = clamp(32/tiles,1,32)`, `PagedAttention.cpp:23-26`) and a fixed-order fp32 partials+statistics reduce — Flash-Decoding structure applied to prefill. KV store kernels: `paged_attention_store.metal`. |
| GDN (Gated DeltaNet; 48/64 layers of Qwen3.8-27B) | `prefill/gdn.metal` — `prefill_gdn_prepare` (conv+norm+gates), `prefill_gdn_scan`, `prefill_gdn_gate[_sums]` | **Serial per-token recurrence**: one threadgroup carries 16 rows of a 128×128 fp32 state token-by-token, 16-token staged blocks (`gdn.metal:6-18`). Sequential depth = full 2048-token chunk. |
| GDN chunked (parallel) | `prefill/gdn_chunked.metal` — `gdn_chunked_prep_c*`, `gdn_chunked_scan_c*` | Full WY/UT DPLR chunkwise form on `matmul2d` (gram products, `(I+A)^{-1}` column solve, `W=B·M`, `X=M'·WK`, hi/lo bf16 state). **Implemented but unbound/abandoned** — `ISSUES.md:52-57`: latency-bound at T=2048 (~1.0–1.15×), `matmul2d` N>16 silent-wrong bug, breaks bitwise split-invariance test. |
| Norms | `shared/normalization.metal`, `prefill/normalization.metal` | RMS norm + sums-emitting variants. |
| MoE (35B-A3B) | `prefill/moe.metal`, `shared/moe_gguf.metal` | Routed expert GEMM. |

### Prefix/prompt caching reuse

- **Content-addressed KV blocks** at `kPageTokens = 32` (`SPLASH_TARGET_KV_BLOCK_TOKENS`, `ExecutionGeometry.h:25`): `engine/KvCache.cpp:74` hashes `(parent block hash, tokens, image identity)`; blocks form a tree (`firstChild`/siblings) — hash-chained radix-like prefix cache. `Cache::probe`/`lookup` (`Cache.hpp:266-284`) return `CacheLookup{kvBoundary, state, junctionBoundary}`.
- **Composite states for the hybrid model**: GDN layers need recurrent+conv state, so KV alone can't resume. `Engine::plannedCheckpoints` (`Engine.hpp:61-72`) places state boundaries every `kPrefillCheckpointTokens = 4096` (`Engine.hpp:25`); `addSharedPrefillBoundaries`/`pendingSharedPrefill`/`sharedPrefillBoundary` (`Engine.cpp:601-653, 974-987`) let co-resident requests share junction states — a Marconi-style hybrid-model prefix cache.
- `publishCommittedBlocks` canonicalizes each new Page32 block — duplicate content re-points to the existing immutable page (`Cache.hpp:303-306`). Disk tier: `KvTier`, `WriteBehind`, `--persistent-cache` survives restarts.
- Measured: cold 8K prefill ~515 tok/s, replay TTFT 0.46 s; ~89% of cold TTFT is GPU prefill; **compute-bound**, weight-streaming floor ~0.2 s (`PREFILL_OPTIMIZATION_PLAN.md:8-25`).

## (B) Papers

1. **Orca — Yu, Jeong, Kim, Kim, Chun (OSDI 2022)**. Iteration-level scheduling / continuous batching. → Maps to `Scheduler`'s per-command batch replanning; Splash dispatches every ready decode lane each command (`Scheduler.hpp:57-61`).
2. **vLLM / PagedAttention — Kwon et al. (SOSP 2023)** + vLLM automatic prefix caching. → Paged KV (`kv::kPageTokens=32`, `KvPool` extents, per-request page tables `ModelBatchItem::pageTable`) and content-hashed blocks (`KvCache::find/insert`).
3. **SGLang / RadixAttention — Zheng et al. (NeurIPS 2024, arXiv 2312.07104)**. LRU radix-tree KV reuse. → Splash's block tree gives equivalent prefix reuse at page granularity, plus a second state-cache layer RadixAttention lacks.
4. **Sarathi-Serve — Agrawal et al. (OSDI 2024, arXiv 2403.02310)**. Chunked prefill + stall-free scheduling. → 2048-row chunks packed across up to 4 lanes; `prefillBudget` halving under contention is its token-budget analogue. Splash cannot fuse prefill+decode into one command, so it's "chunked interleave," not split-fuse.
5. **FlashAttention — Dao et al. (NeurIPS 2022)**; **Flash-Decoding — Dao et al. (2023)**. → `splash_paged_attention_tile` online softmax + `prefill_attention_*_split`/`_reduce` partials+statistics over KV-history splits.
6. **"Parallelizing Linear Transformers with the Delta Rule over Sequence Length" — Yang, Wang, Zhang et al. (NeurIPS 2024, arXiv 2406.06484)**. WY-representation chunkwise DeltaNet. → Direct template for `gdn_chunked.metal`'s `(I+A)^{-1}`/WY derivation.
7. **"Gated DeltaNet" — Yang, Kautz, Hatamizadeh (ICLR 2025, arXiv 2412.06464)**. Gated delta rule + chunkwise parallel form. → The exact recurrence `prefill_gdn_scan` implements serially; the chunked file is this paper's algorithm.
8. **Kimi Linear / KDA — Kimi team (arXiv 2510.26692, Oct 2025)**. Specialized-DPLR chunkwise kernels for the same hybrid family. → Cited in `PREFILL_OPTIMIZATION_PLAN.md:42-44` as the form's source; KDA's fused kernel design is the FlashQLA-class reference for reviving `gdn_chunked`.
9. **FlashQLA (Alibaba/TileLang, 2025)**. Fused memory-bound pieces of chunked linear attention, 2–3× over FLA Triton. → Plan lever 0 cites it; Splash's scan similarly re-reads staged blocks.
10. **ChunkAttention — Ye, Tao, Huang, Li (ACL 2024, arXiv 2402.15220)**. Prefix-tree KV + two-phase partition shares *attention compute* across requests. → Plan lever 5 (`PREFILL_OPTIMIZATION_PLAN.md:122-127`): Splash shares KV pages but **not** compute — unimplemented.
11. **Hydragen — Juravsky et al. (ICML 2024 ES-FoMo, arXiv 2402.05099)**; **FlashInfer cascade attention — Ye et al. (MLSys 2025, arXiv 2501.01005)**. Shared-prefix decode attention via softmax merge. → Splash's per-sequence `prefill_attention` and `verify_attention` plans read shared pages independently; merge-partials machinery (`partials`/`statistics` buffers) already exists as the substrate.
12. **POD-Attention — Kamath et al. (ASPLOS 2025, arXiv 2410.18038)**. Single kernel computing hybrid prefill+decode attention. → Directly targets Splash's "different Metal graphs, cannot pack" limitation (`Scheduler.cpp:256`).
13. **NanoFlow — Zhu et al. (OSDI 2025, arXiv 2408.12757)**. Intra-device overlap of memory-bound and compute-bound ops via nano-batches. → Splash serializes everything behind one pending ticket; alternating decode/prefill commands never co-run — relevant to both the scheduler and `PROPOSED_PLAN.md`'s depth-2 ring.
14. **Marconi — Pan et al. (MLSys 2025)**. Prefix caching for hybrid SSM/Transformer models; SSM-state admission/eviction policy. → Splash's composite-state layer (`StateCache`, junction boundaries, `plannedCheckpoints`) is this problem solved differently; Marconi's reuse-value-aware admission is a potential upgrade over LRU recency.
15. **Splitwise — Patel et al. (ISCA 2024)**; **DistServe — Zhong et al. (OSDI 2024)**. Phase disaggregation. → Single-device analogue is `decodeShare`; ANE-offload section of the plan (`PREFILL_OPTIMIZATION_PLAN.md:147-174`) evaluated and rejected a same-machine version.
16. **MInference — Jiang et al. (NeurIPS 2024, arXiv 2407.02490)**; **CompactAttention (arXiv 2605.16839, cited in plan)**. Sparse long-context prefill. → Plan lever 5 notes fit with paged KV but flags conflict with byte-exact numerics policy.
17. **FP8 activation quantization — e.g., FP8-LM (Peng et al., 2023)**. → Plan lever 1 / TODO #3: would double `matmul2d` throughput on Apple10, but `ISSUES.md:40-44` records e4m3/int8 activations violating the engine's fp64 projection bound by ~10–70× — dead unless the numerics policy relaxes.

## (C) Gaps / Opportunities

1. **Chunked-parallel GDN scan remains unshipped** (plan lever 0 — self-identified highest impact). `gdn_chunked.metal` is complete but abandoned for latency-boundness at C=32–64 and the `matmul2d` N>16 bug blocking C≥128. FlashQLA-style fusion and 32K+ prompts (where serial depth hurts most) are untested angles; the report's own lattice reference measured 1.55–1.65×.
2. **No fused hybrid batch**: prefill and decode never share a command (`Scheduler.cpp:256-264`). POD-Attention / Sarathi-Serve full split-fuse is the largest structural gap; today `--decode-share` alternation is the mitigation.
3. **Shared-prefix compute absent** (ChunkAttention/Hydragen/Cascade): KV pages dedupe but each request re-attends them. Junction boundaries + split/reduce partials are a natural substrate.
4. **No submit-ahead/pipelining** (`PROPOSED_PLAN.md` #1): one pending ticket, GPU idle in host turnaround — NanoFlow-style intra-device overlap or a depth-2 command ring addresses it.
5. **GPU-driven dispatch** (TODO #1): ~10 ops/layer × 64 layers × 4 chunks host-encoded; `MTL4MachineLearningCommandEncoder`/`compute_command` untried.
6. **fp8 prefill activations**: gated on accuracy policy, not kernel work (ISSUES.md).
7. **Chunk/policy sweep** (`prefillTokenBudget` fixed 2048, `kPrefillCheckpointTokens` 4096): un-A/B-tested per plan lever 4.
8. **Sparse long-context attention** (MInference/CompactAttention): noted as accuracy-incompatible with byte-exact guarantee — a policy decision, not a technical block.
---


# KV Cache Subsystem Report — splash

## A. KV cache architecture

**Scope note:** there is no `splash/` directory in the repo — the native engine lives in `runtime/`, plus `server/` (Python HTTP layer) and `dev/` (tests/benchmarks). `TO_EXPLORE.md` is listed by glob but unreadable (dangling/stale entry).

### A.1 Page format and physical layout

- **Page = 32 tokens**, same unit for allocation and prefix matching: `SPLASH_TARGET_KV_BLOCK_TOKENS 32` in `runtime/metal/abi/ExecutionGeometry.h:25`. Context cap 262,144 tokens + 15 speculative scratch rows (`SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS`, lines 15–21).
- **`runtime/metal/abi/KvExtent.h`** — the core addressing ABI. A `SplashKvPage` is a `uint64_t` = GPU address of its extent OR'd with a 14-bit page index (lines 18–20, 109–111). Within an extent, each attention layer gets a region holding **all pages' keys, then key scales, values, value scales** (`splash_kv_offset`, lines 80–89). Keys are **token-major**, values **dimension-major** (`splash_kv_key_element_dim` / `splash_kv_value_element_dim`, lines 44–58). Default head dim 256; 128 (dense) and 64 (LFM2) variants supported.
- **`runtime/ops/PagedKv.hpp`** — `kv::Format`: `Int8=1, BFloat16=2, Int4=3, Float8E4M3=4` (line 16). All quantized formats use **symmetric per-(token, KV-head) quantization with one fp32 scale per head·token** (`storageFormatName`, lines 33–45). `kv::Layout` computes bytes/page and extent geometry: extents target ~128 MiB (`kAllocationExtentTargetBytes`, line 94), every tensor region 64 KiB-aligned (line 89), `extentPagesFor()` picks a uniform extent size that minimizes leftover pages (lines 205–225).
- **`runtime/ops/PageStorage.{hpp,mm}`** — extents are ordinary shared Metal buffers; only `KvPool` allocates/releases them. `writeEntries()` fills per-request GPU page tables; `copyPages()` supports compaction; `spans()` gives host access for disk IO. Pages of committed blocks are immutable while shared (lines 24–26).

### A.2 Pool / block management

- **`runtime/engine/KvPool.{hpp,cpp}`** — sole owner of page references and extent allocation. Per-page refcount split into *active* (request) and *prefixOwner* (cache) references (`PageRecord`, lines 120–126). Free pages are handed out **fullest-extent-first** so cold extents drain to empty (`packingExtent`, lines 70–74, 155). `compactExtent()` (line 114) empties the sparsest extent by moving its pages — a defragmentation pass; `reclaimEmptyExtents()` releases whole extents to the OS, keeping one runway extent warm (lines 106, 80).
- **`runtime/engine/KvCache.{hpp,cpp}`** — **content-addressed radix-style DAG** of 32-token blocks. Block key = (parent block id, exact tokens, `ImageIdentity` for multimodal content) — `find()`/`insert()` (lines 64–72), `blockImageIdentity()` (line 31). A block holds a pool page, a `KvDiskSlot`, or both; resident blocks form a subtree at the root, disk-only blocks hang below (lines 34–39). O(1) leaf ops via sibling links; `generation_` counter lets scheduling probes stay valid across ticks (lines 228–233). Four LRU orders (`RecencyOrder` sets): `ramLeaves_`, `duplicates_`, `diskLeaves_`, `unneeded_` (lines 234–237).
- **`runtime/engine/Cache.{hpp,cpp}`** — orchestrates it all:
  - `probe()`/`refresh()`/`lookup()` — incremental prefix match; leaves one token unmatched to regenerate anchor logits (`Cache.cpp:183`).
  - `publishCommittedBlocks()` (Cache.cpp:293–332) — canonicalizes completed blocks; **duplicate content swaps the request onto the existing immutable page** (dedup, lines 318–327).
  - `ensureTokens()`/`admitPages()` — page admission with `Pending` verdict when demotions-in-flight will free pages (lines 380–412).
  - `reclaimOne()`/`evictOne()`/`reclaimKvLeaf()`/`demoteKv()` — ordered eviction: disposable checkpoints → KV leaves no state restores through → ordinary states/KV leaves → in-use class (`Cache.hpp:346–385`, `Cache.cpp:435–478`). Eviction is **page-granular LRU**, never token-granular dropping.
  - Shared-prefix scheduling: `junctionBoundary` in `CacheLookup` + `Engine::sharedPrefillBoundary`/`addSharedPrefillBoundaries`/`waitForPrefix` (Engine.cpp:601–669) — concurrent requests sharing a prefix plan a junction state and queue behind the producer.

### A.3 Disk tier (SSD offload + persistence)

- **`runtime/engine/KvTier.hpp`** — tier interface: `demote()`/`restore()` move whole pages between extent memory and a slot file via `KvTransfer` objects; pages stay valid throughout (lines 34–63).
- **`runtime/engine/KvPageTier.{hpp,cpp}`** — implementation over `model::SlotFile`; up to 128 transfers in flight, demotions ≤½, restores ≤¾ (`kTransfers`, line 28).
- **Persistence**: `Cache::persist()` writes restore points root-first while resident (`Cache.hpp:241`); `Cache::adopt()` takes back chains an earlier process left (`--persistent-cache`); quota eviction keeps whole restore points (`dropOldestPoint`, `Cache.hpp:582`). `WriteBehind.cpp` handles delayed durability.

### A.4 Attention kernels (KV reads)

- **`runtime/metal/kernels/common/paged_attention_tile.h`** — the main tile: one threadgroup per (KV head, query tile, history split). Pages are read **in place as MPP `matmul2d` cooperative-tensor operands — no staging/dequant buffer**: INT8 keys as `{D,N}` NT operand, values `{N,D}` NN operand; INT4 pages use native `int4b_format` operands directly on packed slabs (lines 187–337, comment at 250–256). Per-page online softmax with fused-row lanes, atomic rescale flag, probability buffer ping-pong (tensorop ordering workaround, lines 236–239). **Split-K over pages** (FlashDecoding-style): `split_count` splits write fp32 partials + max/sum statistics, merged by a fixed-order reduce (`splash_attention_reduce_value`, lines 360–399). Verify lanes get history-scaled splits (32–128, `PagedAttention.hpp:44–63`).
- **Stores**: `paged_store_row.h` quantizes each row (simd-max → scale → round) and writes it to its **final page slot before attention**; rejected verify rows are overwritten next command (commit index = `committed_tokens`). Tree-verify lanes commit via `addVerifyTreeCompact` (`PagedAttention.hpp:305–316`).
- **FP8 tier**: `paged_attention_fp8_tile.h` — E4M3 bytes sharing INT8 geometry; fp16 query staging, two half-M passes to fit threadgroup budget. Known ~0.96–0.98 cosine ceiling (ISSUES.md:46–50); tree verify unsupported for fp8.
- Kernel entry points: `verify_attention_{q8,int4,bf16}_split[_kv4_g4|_kv2_g8|_hd128|_hd64]`, `prefill_attention_*_split`, matching `*_store` kernels — `decode/paged_attention.metal`, `decode/paged_attention_store.metal`, `prefill/paged_attention.metal`, `prefill/paged_attention_store.metal`.
- **GPU page tables**: `Runtime.mm` `synchronizedPageTable()`/`PageTableBinding` (lines 770–796) — incremental `writeEntries` from `firstChanged` on revision bump.

### A.5 Layer-specific and recurrent-state handling

- **Hybrid targets** (Qwen3.5/3.6 families): `QwenHybridLayout.hpp` — full attention every `fullAttentionPeriod`-th layer, the rest are **GDN (Gated DeltaNet) recurrent layers** (lines 62–80). The KV pool only covers `attentionLayerCount()` layers; GDN state lives in `StateLayout`/`QwenState` and is snapshotted as `CompositeState` at block boundaries in `StateCache` (`engine/StateCache.hpp`). Consequence: **a hybrid model cannot resume from KV alone** — KV blocks are only worth keeping if a state restores through them (`Cache::kvNeededByState`, Cache.cpp:549).
- **Draft model KV is separate**: a per-lane **ring buffer of 2048 tokens** (`SPLASH_DRAFT_SLIDING_WINDOW`, ExecutionGeometry.h:24; `common/draft_context_kv.h`, `decode/draft.metal`) — the only sliding-window structure in the codebase; the target cache is full-history.
- **Speculative decode**: verify rows are pre-stored into page slots ahead of acceptance (`SPLASH_SPECULATIVE_SCRATCH_TOKENS=15`); tree verify uses per-row ancestor bitmasks (`TREE_VERIFY_DESIGN.md`).

### A.6 Memory planning

- **`runtime/engine/MemoryPlan.{hpp,cpp}`** — hard budget = `recommendedMaxWorkingSetSize` − max(1 GiB, 2%); host reserve = min(10% RAM, 2 GiB). One dynamic budget covers KV extents + states + lane state; extents grow/shrink on demand ("elastic" KV pool). `kvCapacityPages/Tokens` caps per-request context.
- `MemoryGovernor`, `MemoryControl`, `MemoryAudit` — host admission and pressure handling; `Engine::reclaimMemory()` runs extent release → eviction → compaction between commands.

### A.7 KV quantization findings (in-repo analysis)

- **`TurboQuant_ANLYSIS.md`** — evaluated TurboQuant, asymmetric K/V splits, KIVI-style recency windows, and mxfp4/nvfp4/mxfp8 block-fp on Qwen3.8-27B. Findings: V precision dominates (opposite of KIVI folklore); recency window *hurt*; mxfp4 loses to int4 at equal bytes; ~6–7 bits/element is the practical floor. **Recommendation implemented: keep symmetric per-(token,head) scalar quant; int4 is now the default** (`main.mm:57`), int8/bf16/fp8e4m3 selectable via `--kv-format` (README.md:103–134 documents the verify-acceptance trade: int8 62 tok/s/63% vs int4 56 tok/s/55% at ~67K ctx).

## B. Paper list → code mapping

| Paper | Authors/Year | Relevance | Code mapping |
|---|---|---|---|
| **PagedAttention / vLLM** — "Efficient Memory Management for LLM Serving with PagedAttention" | Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica — SOSP 2023 (arXiv 2309.06180) | OS-style paging for KV: fixed-size pages, page tables, sharing | Direct analog: `KvExtent.h` page entries (GPU addr + index), `PageStorage::writeEntries` GPU page tables, `KvPool` refcounts, `KvCache` shared immutable pages |
| **RadixAttention / SGLang** — "Efficient Execution of Structured Language Model Programs" | Zheng, Yin, Xie, Sun, Huang, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng — NeurIPS 2024 | Radix tree of cached token prefixes, LRU eviction | `KvCache` is exactly this: chained content-keyed 32-token blocks form a DAG; `find`/`insert`, `RecencyOrder` LRU eviction, `generation_` probe validity. Differs: fixed 32-token blocks (no partial-edge splits) |
| **FlashDecoding / FlashAttention-2 split-K** | Dao et al. 2023 (blog) / Dao, NeurIPS 2022 | Parallelize attention over KV length, merge partials | `splash_paged_attention_tile` writes per-split fp32 partials + {max,sum} stats; `splash_attention_reduce_value` fixed-order merge (PagedAttention.hpp:44–63, paged_attention_tile.h:360–399) |
| **KIVI** — "A Tuning-Free Asymmetric 2bit Quantization for KV Cache" | Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu — ICML 2024 (arXiv 2402.02750) | Per-channel K, per-token V, fp16 residual window | **Evaluated and rejected** (TurboQuant_ANLYSIS.md §"Recency window", alternatives table): per-token-head symmetric quant kept; residual window tested and worse |
| **KVQuant** — "Towards 10 Million Context Length" | Hooper, Kim, Mohammadzadeh, Mahoney, Shao, Keutzer, Gholami — NeurIPS 2024 (arXiv 2401.18079) | Pre-RoPE K quant, non-uniform, outlier isolation | Listed as alternative, not implemented — "needs calibration; complicates paged layout" (TurboQuant_ANLYSIS.md:139) |
| **TurboQuant** — "Online Vector Quantization with Near-optimal Distortion Rate" | Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026 (arXiv 2504.19874) | Rotation + Lloyd-Max + QJL residual, unbiased inner products | **Evaluated and rejected**: no separation from scalar int4 at equal bytes on this model (TurboQuant_ANLYSIS.md:81–84); vLLM's own study agrees |
| **H2O** — "Heavy-Hitter Oracle for Efficient Generative Inference" | Zhang et al. — NeurIPS 2023 (arXiv 2306.14048) | Attention-score-based token eviction | **Not implemented** — eviction is page-granular recency (`RecencyOrder`), not token importance; pages only evicted when unpinned |
| **StreamingLLM** — "Efficient Streaming with Attention Sinks" | Xiao et al. — ICLR 2024 (arXiv 2309.17453) | Sink + sliding-window KV | Only in the *draft* model: 2048-token ring (`draft_context_kv.h`); target KV is full-history exact |
| **SnapKV** / CaM / token pruning | Li et al. — NeurIPS 2024 (arXiv 2404.14469) | Per-head prompt token selection | Not implemented; noted as "separate investigation" (TurboQuant_ANLYSIS.md:142) |
| **DeepSeek-V2 / MLA** | DeepSeek-AI — 2024 (arXiv 2405.04434) | Latent low-rank KV compression | N/A architecturally — KV layout is per-layer/head paged slabs of served GGUF models; no MLA or cross-layer sharing. GQA already exploited: layouts `kv4_g6`, `kv4_g4`, `kv2_g8` (4 KV heads × 256 dim ≈ 33 KB/token int8) |
| **GQA** — "Training Generalized Multi-Query Transformers" | Ainslie et al. — EMNLP 2023 (arXiv 2305.13245) | Fewer KV heads → smaller KV | Exploited structurally: `QueryHeadsPerKVHead` template param, fused GQA rows per tile (paged_attention_tile.h:85–103) |
| **Chunked-prefill / Sarathi-Serve** | Agrawal et al. — OSDI 2024 | Chunked prefill piggybacking decode | Implemented: `SPLASH_PREFILL_TOKEN_BUDGET=2048`, `decodeShare` interleave (Scheduler.hpp, Engine.hpp:44) |
| **Hydragen** — shared-prefix attention | Juravsky, Brown, Ehrlich, Fu, Ré, Mirhoseini — ES-FoMo-II/ICML 2024 (arXiv 2402.05099) | Batch queries across sequences sharing prefix KV | **Gap**: pages are shared physically but each lane reads them independently; noted in PREFILL_OPTIMIZATION_PLAN.md §5 as ChunkAttention (arXiv 2402.15220) |
| **SpecInfer tree verification / Medusa-style trees** | Miao et al. — ASPLOS 2024 | Verify a token tree, not a chain | Implemented: `TREE_VERIFY_DESIGN.md`, ≤16-node comb, ancestor bitmask in `splash_attention_page_softmax` (`row_masks`, paged_attention_tile.h:113–139), `verify_attention_tree_*` kernels |
| **LMCache / tiered KV offload** | Cheng et al. — HotStorage 2024 (arXiv 2310.07240) | KV to disk/CPU, cross-process reuse | Implemented: `KvPageTier` + `SlotFile`, `--max-cache-disk`, `--persistent-cache`, `Cache::adopt`/`persist`, `WriteBehind` |
| **Preble** — prefix-aware scheduling | Srivatsa et al. — SOSP 2024 | Schedule around shared prefixes | Partial: `waitForPrefix` + `sharedPrefillBoundary` junction states (Engine.cpp:601–669) |
| **Gated Delta Networks** | Yang, Kautz, Hatamizadeh — ICLR 2025 (arXiv 2412.06464) | Hybrid linear-attention state vs KV | Implemented: `QwenHybridLayout`, `prefill/gdn.metal`, `decode/gdn.metal`, `StateCache` composite states at KV block boundaries |

## C. Gaps and opportunities

1. **Token-level eviction / sparse retention (H2O, SnapKV, StreamingLLM-for-target).** Cache is exact and full-history; eviction is whole-page LRU at cache level only. Nothing drops tokens *within* a live request. Justified by byte-exactness goals, but SnapKV is flagged as "separate investigation" (TurboQuant_ANLYSIS.md:142) — a natural fit since Page32 blocks could be dropped from page tables without copies.
2. **Shared-prefix attention compute (Hydragen / ChunkAttention).** Pages are shared physically; bandwidth isn't. Batched queries over a shared prefix could collapse per-lane KV reads. Explicitly deferred in PREFILL_OPTIMIZATION_PLAN.md:123–127 — only matters under multi-tenant load.
3. **Block-sparse / query-aware KV selection (Quest, CompactAttention arXiv 2605.16839).** Noted at PREFILL_OPTIMIZATION_PLAN.md:118–122: "selection becomes a block-table filter, no KV copies" — the paged layout makes this cheap mechanically; gated on numerics policy.
4. **Sub-page prefix granularity.** RadixAttention edge-splitting equivalent doesn't exist: partial blocks (<32 tokens) are never shareable, and the last-token anchor rule (Cache.cpp:183) wastes up to 32 tokens per prompt.
5. **Per-layer quantization budgets (CakeKV, DynamicKV).** `kv::Layout` applies one format to all attention layers; fp8-per-layer or layer-wise bit budgets unexplored. Relevant given the analysis showed V-sensitivity may vary by layer depth.
6. **fp8 KV quality ceiling.** ISSUES.md:46–50 — E4M3 caps at ~0.96–0.98 cosine; fix needs higher-precision K/V (e.g., mxfp8-E4M3 with per-vector scales, or asymmetric fp8-K + int8-V).
7. **Cross-request semantic dedup (MiniCache).** Dedup requires exact token+image identity (`blockImageIdentity`); no approximate/fuzzy block merging.
8. **MLA-style latent KV.** Only relevant if a served model adopts it; no infrastructure barrier — a latent KV head would map to `kvHeads=1` with a different `headDimension`, which `kv::Layout` already parameterizes.
9. **Deliberately rejected (documented dead ends):** TurboQuant, asymmetric K/V bit splits, KIVI recency windows, mxfp4/nvfp4/mxfp8 KV — all measured in `dev/turboquant_experiment.py` and rejected in `TurboQuant_ANLYSIS.md` with reproduction instructions.
# DFlash2 Decode Regression — RESOLVED

**Root cause: the benchmark's prompt remap, not an engine bug.**

`dd540e33` added `promptVocabulary = min(capabilities.vocabularySize,
200100)` + a remap `token >= promptVocabulary → 100 + token % (vocab-100)`
to `backend_benchmark.mm`. Four prompt tokens ≥200100 changed
(248046→48146, 248045→48145, 248068→48168, 248069→48169) — post-dd540e33
benchmarks ran a **different prompt** than the 87% era.

With the verbatim prompt (`DBG_NOREMAP=1` guard added to the remap),
the `33ea5cd9` build produces:

- **87.3% acceptance** (vs 42.9% on the remapped prompt)
- **103.6 tok/s B1** (vs ~50)
- **`output_token_hash: 8845173885570541823` — byte-identical to the
  good era's output**

The engine at `dd540e33`/`33ea5cd9` was never broken. Every anomaly
— the +1 ctxcommit offset, shifted proposals, first-token divergence
(16 vs 13), completely different Hidden0/logits — traced to measuring
different prompt content, not engine behavior. `prompt_sha256` is a
hardcoded constant, not computed — it never caught the change.

**Implications:**

- `docs/performance.md`'s "draft acceptance is down from 87%" note
  was also measured on the remapped prompt.
- **HEAD on the verbatim prompt: 88.7% acceptance, ~94 tok/s B1,
  identical output hash** — the "second regression" (18.4%) was the
  same benchmark artifact plus the 16-row denominator inflation
  (15 offered slots vs 7 live proposals). HEAD's engine is healthy.
- The 8→16 verify geometry change is the only real throughput cost:
  ~94 vs ~104 tok/s B1 (verify reads ~10% more bandwidth per step
  for no acceptance gain — 88.7% vs 87.3%).
- The WIP's parameterized 8-row verify path (~0% acceptance) is a
  separate, real bug — unrelated to the benchmark artifact. Fixing it
  restores the ~104 tok/s geometry.
- The GPU-predraft crash found during this bisect is fixed:
  `settleGpuPredraft()` now drains the deferred command on every
  state/extent-freeing path before `evictAllCache` can run.

## Remaining work (ordered)

1. **8-row verify bug** (`DFlashDraft.hpp` FIXME): the draft's
   `verifyRows()`/`queryRows()` is forced to `targetVerifyRows` (16)
   because block-8→8 rows gives ~0 acceptance — "proposals stop
   matching after the first committed step". Likely: the draft packs
   8 rows but a downstream consumer (retained-path gather, ring
   stride, captured-hidden index) still strides 16. Fix → verify
   ~87% + ~104 tok/s, then restore `verifyRows() =
   trained block rows` per the comment.
2. **Benchmark integrity** — `prompt_sha256` is hardcoded; compute it
   (or assert the built prompt hashes to it) so a remap can't
   silently change the measured content again.
3. **DFlashTwo split** — baseline confirmed green (~104 tok/s): new
   `DFlashTwoDraft.{hpp,cpp}` + `DraftKind::DFlashTwo`, migrate V2-only
   loader/conv/selector/tree code out of `DFlashDraft`, keep legacy
   DFlash/V1/DSpark/plain paths compiling.

## Prior findings (all bisects measured on the remapped prompt)

Measured with `backend-benchmark build/<name>.metallib
install/models/mlx-community/Qwen3.8-27B-4bit --scenario decode --samples 1`,
`--kv-format int8`, Qwen3.8-27B, B1, Apple M5 Pro.

## Bisect table

| Commit | Date | B1 tok/s | Draft acceptance | tok/batch | Verdict |
|---|---|---|---|---|---|
| 540a760a | Oct 2 | 92.6 | 87.3% | 7.11 | good |
| fbecdcb0 | Oct 2 | 96.0 | 87.3% | 7.11 | good |
| 1b1925d6 | Oct 3 | 92.5 | 87.3% | 7.11 | good |
| 6df52cd6 | Oct 5 | 97.1 | 87.3% | 7.11 | **last good** |
| **dd540e33** | Oct 5 | **50.6** | **42.9%** | **4.00** | **first bad** |
| 33ea5cd9 | Oct 6 | 55.6 | 42.9% | 4.00 | bad |
| 05a9e36c (HEAD) | Oct 12 | ~49 | 18.4%* | 3.76 | bad |
| WIP + forced-16 verify | — | ~47 | 34% | 3.37 | bad |
| WIP as-found (verifyRows=8) | — | ~21.6 | ~0% | ~1.1 | worst |

\* 18.4% is partly a denominator artifact: the 16-row geometry offers
15 proposal slots while the block-8 draft only fills 7; the honest loss
is 4.00 → 3.76 tokens/batch.

## Three stacked regressions

### 1. `dd540e33` — "Rename the engine to richengine and reorganize the runtime tree" (first bad)

Real acceptance halved: 7.11 → 4.00 tok/batch. Not a rename artifact.

**Ruled out by audit or A/B:**

- All kernel diffs are mechanical `SPLASH_*`→`RICHENGINE_*` renames
  (verified with name-normalized diffs): `sampling.metal`,
  `draft.metal`, `draft_select_*.metal`, `DraftAttention.h`,
  `DraftAttention.cpp`, `DraftSelector.cpp`, `DraftContextPlan.cpp`.
- `ExecutionGeometry.h` / `ExecutionLimits` / `Tuning.hpp` — identical
  values (8/7/8 geometry in both eras).
- `RICHENGINE_HEAD_FUSED_OFF=1` on the current build: identical
  ~47 tok/s @ 34% — the fused-head argmax fork is NOT the bug.
- `causalLayers` — the `sliding_attention → causalLayers |= 1<<layer`
  mapping exists at BOTH 6df52cd6 and dd540e33 (`git log -S` shows it
  landed in `1df9780c`, which benchmarks GOOD at 92.5 @ 87.3%).
- `linearScratch` threading — assigned for all three draft buffer
  phases (prefill/context/decode) from `decodeArena->linearScratch()`.
- Packed-manifest dispatch — this package is `mlx-affine`; the
  `applyManifestDraftKind` changes are on a different path.
- Makefile/build — file-path moves only, no flag changes.

**Debug signature** (`RICHENGINE_DRAFT_DEBUG=1` on the dd540e33 build,
same prompt):

- pos=1: `props=[561 635 9 198 220 16 13]` → 0/7 accepted, EVERY lane,
  EVERY sample — the first block after prefill is consistently dead.
- pos=39-96: healthy streaks (4, 7, 6, 5, 4, 5, 6 accepted).
- Scattered dead blocks mid-decode (pos=59, 63, 72, 81, 83: 0/7).

Intermittent alignment failure, not wholesale conditioning break.
The pos=1 props look position-shifted — 198 sits at index 3.

**Reframed (exact-append instrumentation):**

The target itself diverges — the prefill's first emitted token differs
between eras (`tok=[16]` good vs `tok=[13]` bad on the same prompt).
The draft is downstream of whatever the target produces; a subtly
wrong target logit path halves acceptance while positions stay correct.

Append streams (`DBG_POS` on both worktree builds):

- good: `kind=0 n=1 [16]` → `kind=1 n=1` → `n=8` per step (7.9 avg)
- bad: `kind=0 n=1 [13]` → `kind=1 n=2` → erratic 1–7 per step (~4.3)

**Kernel-sequence diff (DBG_OPS, both worktrees):**

Per-step dispatch lists are IDENTICAL — same kernels, same order,
layer by layer. Only differences: `int4`→`q8` KV format (A/B'd —
same 42.9%), `decode_head_argmax_q4` fused-head path (A/B'd OFF —
same 42.9%), `ane_ffn_*` split (`--ane-ffn-share 0` — same 42.9%).
So the divergence is inside a kernel's buffers/params, not the op set.

**Also ruled out this round:**

- `modelLayoutSha256` diff — it's the package manifest digest, not
  the resolved layout (noise).
- `KernelLayout` variants — target is `head_dim 256/kvHeads 4 →
  Kv4Group6`, unchanged path; the new `k8q4d128` variants are for
  128-dim attention (draft/LFM2).
- `score_scale` — staged via struct init with `plan.scoreScale`;
  0 = compiled default (identical to the old constexpr).
- Weight store — only `language_model.` prefix fallback + config
  validation added.
- `RichKvPage` — same `uint64_t` typedef as `SplashKvPage`.
- Arena `put()` diffs — additive only (I8Codes, HeadArgmax, Null
  guard); `exactTokens`/position accounting identical.
- Prompt generator — `min(vocab,200100)` → same `% 200000` range.
- `requireLayerTypes`/`fullAttentionPeriod` — identical in both eras;
  rope tables are uniform (no per-layer alt-rope at this era).

**What remains genuinely unexamined:**

The first emitted token differs (16 vs 13) on an identical prompt
with identical kernels in identical order — the difference must be
in a value (a param byte) or an input buffer binding. The decisive
remaining experiment is a file-group revert: copy good-era
`runtime/model/{Runtime*,QwenTarget*,TargetModel*}` sources into the
dd540e33 worktree, `sed` namespaces (`splash::→richengine::`,
`SPLASH_→RICHENGINE_`), rebuild — if acceptance recovers, bisect
within that group. Alternatively instrument a numeric dump (first
hidden row after prefill) and bisect ops by first-divergent value.

### 2. `33ea5cd9` → HEAD — second regression (unbisected)

42.9% → 18.4% measured; largely the 8→16-row geometry change
(`RICHENGINE_DRAFT_QUERY_ROWS`/`PROPOSAL_TOKENS`/`TARGET_VERIFY_ROWS`
went 8/7/8 → 16/15/16 somewhere in the Oct 6–12 pushes). Verify row
count roughly doubled per step; the honest loss is smaller than the
rate suggests (4.00 → 3.76 tok/batch). Needs its own bisect.

### 3. Verify geometry — resolved via a compiled 8/8/7 variant

Runtime-striding a 16-compiled build to 8 verify rows
(`RICHENGINE_VERIFY_NARROW=1`) is NOT output-identical: it emits a
different trajectory at ~27% acceptance (the fused-head m8 tile argmax
and the unfused path agree with each other but not with the fused m16
tile — the residual is a tile-variant numerics divergence, not stale
state). The correct path is a compiled 8/8/7 geometry: `make verify8`
produces `build8/richengine` + `build8/richengine.metallib`, verified
oracle-identical at every width (hash 8845173885570541823, 88.7%
acceptance, all eight transcripts token-identical). The server selects
it automatically for drafts whose `block_size <= 8`
(server/model_host.py `_engine_binary`).

B4 decode on the 27B: 121.7 → 249.9 tok/s (upstream splash: 334.3;
residual is kernel efficiency, not geometry).

## Reproducer

`/tmp/bisect-step.sh <commit>` — checks out into `/tmp/splash-bisect`,
builds `backend-benchmark` + metallib, runs the decode scenario, prints
B1 tok/s + acceptance. ~10 min per step.

```bash
git worktree add /tmp/splash-<tag> <commit>
ln -sfn <repo>/.venv /tmp/splash-<tag>/.venv
ln -sfn <repo>/install/models /tmp/splash-<tag>/install/models
cd /tmp/splash-<tag> && make -j8 build/engine-tests/backend-benchmark
./build/engine-tests/backend-benchmark build/*.metallib \
  "install/models/mlx-community/Qwen3.8-27B-4bit" \
  --scenario decode --samples 1
```

Pre-rename commits produce `splash.metallib` + `libsplash.a`; post-rename
`richengine.metallib` + `librichengine.a`.

## Other diagnostics

- Server-level debug: `RICHENGINE_DRAFT_DEBUG=1` prints per-step
  `props` vs `out` + accepted/retained per lane.
- `RICHENGINE_DECODE_TIMING=1` prints encode/finalize/wall ms per step.
- Decode step at B1 ≈ 68 ms ≈ the 14 GB weight pass at ~215 GB/s —
  bandwidth-bound; acceptance × proposals is the throughput lever.
- `norm_rms` in `RICHENGINE_OP_TIMINGS` output is a baked-span label,
  not a real 24 ms norm kernel.
- GPU predraft (`RICHENGINE_GPU_PREDRAFT`, default on): correct
  (adopts every step in steady state), neutral at B1 on the 27B.
- `RICHENGINE_VERIFY_TREE=1`: measured −17% on this model; tree comb
  leaves cost more than they rescue at ~40% acceptance.

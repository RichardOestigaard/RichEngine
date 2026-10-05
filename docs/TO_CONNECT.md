# Upstream commits to connect

Divergence: 56 commits on `upstream/main` since merge-base `f43509a`.
Merge commits are listed once under their PR; their constituent commits carry
the real diffs.

Local context that matters for conflicts: the working tree has the adaptive
proposal change (Sampling/GDN ABI + kernels, QwenTarget, Runtime), the
Runtime.mm split into `RuntimeImpl.hpp` + `Runtime{Encode,Ane,Ngram}.mm`, the
Env.hpp env-flag centralization, and pre-existing dirty work (DraftSelector,
LfmConv, MoE, fused-reduce in QwenTarget).

---

## Tier 1 — take: real fixes and features

### Mac sleep / lifecycle (issue #275, #292)

- **d94f6cd — AwakeClock for all engine timing.** `steady_clock` counts Mac
  sleep, so the 120 s watchdog, mask/resource waits, cache probation,
  keep-alives and request deadlines all fired instantly on wake, killing the
  engine mid-request. New `runtime/AwakeClock.hpp` (CLOCK_UPTIME_RAW) + 19
  files rewired. Reproduced and verified upstream. **Conflicts:** touches
  `Runtime.mm` (now split — the hunk is inside Impl methods, lands in the
  hpp or RuntimeEncode.mm), `MetalBackend.mm`, `Cache.cpp`, `NativeRuntime`,
  `KvPool`.
- **c72312d — prevent idle sleep while requests run.** IOPMAssertion
  (`caffeinate -i` equivalent) held while requests are in flight;
  `--allow-idle-sleep` opts out. 9 files, main.mm owns the assertion.
- **3c60bc6 — awake-clock rule docs.** `millisecondsSince` moves into
  AwakeClock.hpp; comments + arch-check updated. Pairs with d94f6cd.
- **466455c — `--idle-release DURATION|off`.** Unwires buffers/frees weights
  after a configurable idle period (default 10m, `off` never); adds
  `idle_release_seconds`/`released`/`restores` to `/status`. 22 files.
  **Conflicts:** `RuntimeResources.mm`, `MetalBackend.mm`, `TestConfig.hpp`
  (we touched TestConfig.h? no — but MetalBackend.mm has our envFlag edits;
  mechanical conflicts only).

### Server/API correctness

- **d4066a9 — keep tool-call arguments as the model writes them** (#293,
  #294, huge: 24 files, ±2.3k). Constrains arguments grammars only for strict
  tools; parses calls as the template lays them out; typed value conversion;
  drops schema validation of calls entirely (client reports rejections).
  Fixes renamed/dropped fields and `"12:00"`→`12` coercions. Highest-value
  server change.
- **8bfbdbf — read `enable_thinking` once.** `null`/non-bool kwargs bug:
  `{"enable_thinking": null}` now follows reasoning effort; non-bool → 400.
- **4133d89 — reasoning_effort `none` to chat templates.** Nex-N2.5-mini
  opened a think block on effort none → 400s on every Anthropic request
  without thinking. Verified byte-identical on 23,520 renders.
- **6836087 — invalid-JSON only for unreadable bodies; schema depth cap.**
  Server faults no longer surface as client 400s; 64-level schema cap; a
  token-limit-cut tool call is omitted instead of re-raised.
- **73c163a — Retry-After on every retryable error** incl. /v1/systemone's
  529s.
- **b6eb680 — answer-slot check on prompt tail only + stop on disconnect.**
  256K-token × 255-option question: 76 s → 0.3 s prepare.
- **d5d6fbc — cached PDF doesn't queue behind another PDF's render.** Lock
  narrowed to the render itself.
- **bb9db4c — check tool schemas without building validators.** Tools no
  longer evict response-format validators from the 256-entry cache; accepts
  some formerly-refused valid schemas.
- **b087de7 — `--request-timeout` takes durations** (`30m`, `2h`), same
  format as `--idle-release`.
- **2fc1b43, 59d9dea, 407d7e8 — small error/diagnostic fixes** (proper error
  codes, traceback line in console errors, limit comments).
- **2f02059 — serve lock error names the upgrade**, not a phantom second
  server.

### Engine robustness

- **bac0856 — refuse GGUF with repeated metadata key.** Defensive parse fix,
  2 files, zero cost.
- **3934cdb — legacy package manifests only checked on draft geometry.**
  Published packages can't change; checking batch width/page size against
  them blocks retuning. Small, safe.
- **4d9ab20 — validate each model config once, one rule.** Removes
  double-validation drift (`true` accepted as 1, capture-layer `5.9` as 5,
  1 MiB config cap).
- **bc44366 — image rows via RowCopy.** Deletes `Vision::inject` + its
  kernel, reuses the checked copy path. Net −86 lines.
- **25e4e8d — refuse Q4-sum norm for F32 weights / non-64-multiple widths.**
  Up-front invalid_argument instead of missing-pipeline failure.

---

## Tier 2 — take for divergence hygiene: upstream cleanups

These are upstream's own cleanup PRs (#306 model, #307 tooling, #308
ops-kernels, #309 server). They overlap our cleanup pass. Skipping them means
every future upstream sync gets harder; taking them costs conflict resolution
now. Their own diffs are well-scoped and self-verified (IR-identical claims).

- **0ebcda7 — shader entry points from macros** (#308): 12 kernel files, the
  duplicated verify/prefill store/reduce/gate entries become entry macros;
  common/attention_gate.h + lane_bindings.h added. Kernel names + IR
  unchanged except 3 verify gates (nuw/nsw flag only). **Conflicts:** touches
  `decode/gdn.metal` entry region — our live_rows edits are inside function
  bodies, should merge cleanly.
- **bcf3ebb — check every dispatch buffer's extent** (#308): GDN, Sampling,
  DraftSelector, RoPE, norms, PagedAttention, Embedding now validate buffer
  extents before encoding (GDN decode was a real OOB-by-short-buffer path).
  **Conflicts:** highest — touches GDN.cpp, Sampling.cpp, PagedAttention.cpp
  where our adaptive-proposal signatures changed. Resolve by keeping both:
  new params inside structs + their extent checks.
- **a61f5df — one BufferExtent::requireBytes helper** (#308): consolidates
  three ad-hoc buffer checks; names each buffer+extent in errors.
- **c19b194 — gpuFamilyClass()** (#308): families 9/10/11 plan as before;
  below-9 now refuses at startup instead of getting a broken tile mix.
  Touches Linear.cpp/LinearGguf.cpp/MoE.hpp (envFlag edits — trivial
  conflicts).
- **88cc762 — RICHENGINE_TARGET_ROPE_PAIRS/DRAFT_ROPE_PAIRS/MOE_EXPERT_SLOTS** in
  ABI headers; kernels stop re-spelling 32/64/256. **Conflicts:** user edits
  to moe.metal overlap textually; both mechanical.
- **824c6f1 — draft attention geometry once in abi/DraftAttention.h**:
  removes ten restatements of 32q/8kv/128hd/4096/6144. IR unchanged.
- **997276a — PQ2_0 decode shared between gathers** in
  common/gguf_embedding_formats.h. IR unchanged but gguf_embed_rotated_pq20
  (operand order only).
- **38732f8 — QwenTargetDimensions**: three hand-copied geometry views → one
  struct; fixes real drift (`isFullAttentionLayer` zero-period guard,
  denseIntermediateSize naming). **Conflicts:** QwenTarget.cpp/hpp,
  RuntimeArenas.mm — overlaps user's fused-reduce work; do after that's
  committed.
- **4619053 — per-target draft layout constants** (kQwen3_8DraftLayout,
  kQwen3_6MoeDraftLayout); kills the dangerous `DFlashDraftLayout{}` default.
- **2066102 — remove test-only interfaces**: drops ModelTelemetry
  lastDecodeWidth, requires WeightImages identity, un-exports vision loader
  internals. **Conflicts:** touches Model.hpp/Runtime.mm — our split moves
  the same region; merge by applying to Impl/hpp.
- **103578f — GgufTensorDescriptor self-encodes** at fixed offsets; removes
  memcpy'd padded struct. Image bytes unchanged.
- **08668fd — HTTP ingress bounds → server/connections.py**; pure move.
- **5a48a1d — POST routing table + per-API error dialects**; replaces the
  205-line do_POST. Server-side, no runtime overlap.
- **aa0aec9 — anthropic_block() shared** between stream and complete
  response. Small.
- **d1faee9 — kernel policy comments**: restates rules instead of histories;
  fixes three wrong comments. Cheap; some hunks touch gdn.metal/moe.metal
  comments — trivial conflicts.
- **e1911c8 — weight-image error wording** (names GGUF/affine images, not
  "packed file").

### Tooling/CI (#307)

- **0d68010 — `make check-native-build`**: builds every named test/benchmark/
  tool in hosted CI. Would have caught the broken `gdn-metal` test in this
  tree. **Conflicts:** Makefile line near our RuntimeEncode.mm additions —
  trivial.
- **ec011c7 — installed-client tests (OpenCode/Codex/Pi) in agent preflight.**
- **84c0398 — drop assembly check for packed drafts** (never shipped).
- **5eb065e — model-catalog workflow drops HF_TOKEN requirement.**

### Tests

- **b619a9d — gdn-decode verifies the decode gate bit-reproduces the prefill
  gate.** Directly guards the file we just changed (`live_rows` edits) — take
  it and re-run.
- **dbc0c5d — persistent-cache restart smoke test** (two clean
  serve/restart cycles, ≥90% reuse assertion).

---

## Tier 3 — skip (unimportant to this branch)

- **4c11cbf** — README wording.
- Merge commits themselves (93ba8ea, dafffe9, 1f46504, 7120840, 60c107d,
  1a13b75, 2d5a727, 1083612, c3a6de6, 4de3f1f, 35c828c) — containers only.

Nothing upstream is actually *unwanted* — the list is all strict
improvements. "Skip" means "adds nothing a merge won't bring anyway."

---

## Suggested order

1. Commit/stash current work (adaptive proposals + cleanup + user edits are
   all uncommitted — merging onto this tree will be messy).
2. Merge `upstream/main` wholesale rather than cherry-picking: the tree is
   fast-forwardable in intent and the cleanups interlock. Conflicts land
   mostly in `Runtime.mm`→Impl split, `Sampling.cpp`, `GDN.cpp`,
   `QwenTarget.cpp`, `moe.metal`, `Makefile`.
3. Verify: `make`, then gdn-decode / target-sampling / dflash-batch-control,
   then `make check-native-build` (new) plus `b619a9d`'s GDN gate test which
   covers the same file adaptive proposals touched.

# Subagent verification findings

Cross-check of `DS_FINDIGNS2.md` (audit, 2026-10-26) and
`DEEPSEEK_V4_1_FLASH_DIVE.md` (Metal 4 audit) against the current working
tree. Verified by four read-only subagents. Verdicts: STILL VALID unless
marked otherwise.

## Stop-the-line (ship-blocking)

- [ ] **#1 Architecture gate red.** `dev/tools/check_architecture.py` reports
  ~29 violations on the working tree (5× `ops → model/ModelPolicy.hpp` deps:
  `GDN.cpp:3`, `MoE.cpp:4`, `Linear.cpp:3`, `ExecutionPlans.cpp:4`,
  `DraftAttention.cpp:4`; 1× `metal/MetalEncode.mm:8`; 17× `graph.add(` in
  `model/` TUs; 6× sleep-counting clocks in `Runtime.mm:465,792,796,810`,
  `GgufPreparation.cpp:188`, `RuntimeAne.mm:161`). `dev/Makefile:99`
  `check-source: architecture-check` unchanged; `ci.yml:38` still runs
  `make check-source check-native-cpu`. Still no test running `check()`
  against the real root (`test_architecture.py` only mocks ROOT; `:173-174`
  calls `stale_policy_names()` unmocked).
- [ ] **#2 Release packaging omits modules.** `dev/tools/package.py:53-91`
  `SERVER_FILES` lacks `server/decisions.py` and `server/dfa.py`; both are
  live imports (`frontend.py:16-17`, `constraints.py:21`). Staged-import
  gate (`package.py:247-258`, `import server.server`) still present — `make
  package` fails its own verification. (`richengine8.metallib` staged at
  `package.py:107` — new.)
- [ ] **#3 Web UI unwired.** No pnpm/node job in CI; `dev/Makefile:6`
  `package: install-environment all` lacks the `webui` dep (root Makefile
  `webui` target exists at :197-199); `server/webui/` gitignored
  (`.gitignore:20-21`); fallback to `chat.html` intact
  (`server.py:80-86,507-508`).
- [ ] **#4 139 unwired tests.** `dev/Makefile:73-82` runs only
  `dev/tests/engine` + `test_server.py` + five named modules. Unwired:
  `test_dfa` 18, `test_pack` 7, `test_tuning` 82, `test_local` 7,
  `test_webfetch` 15, `test_websearch` 10 — incl. SSRF cases.

## High-value bugs — all still valid

- [ ] **#5 Model-load race.** `server/model_host.py:471-516`: first lock
  checks `loading`, `_hold()` outside lock, second section never re-checks
  `loading` before `_unload_locked` + `state="loading"`;
  `_publish_locked` (:386-401) overwrites the stack without closing.
- [ ] **#6 Semaphore double-release.** `server/runtime.py:727-734`
  `submit_batch` releases all acquired slots; `_write_bytes` failure →
  `_begin_generation_failure` (:1084) already released them (:1449,
  :1486-1488) → `BoundedSemaphore` `ValueError` masks `EngineUnhealthy`.
  Single-request path hardened (:661-665), batch not.
- [ ] **#7 GGUF traversal + unvalidatable metadata.** `install/gguf.py:369`
  unvalidated `names[0] + "_template.jinja"`; `install/assembly.py:310-311`
  `_write_durably(stage / name)` without `is_safe_path` (`..` escapes
  stage); `:327` `set(hashes) != set(gguf.DERIVED_FILES)` can never pass
  for a named-only template.
- [ ] **#8 Non-atomic pack install.** `install/pack.py:903-905` `rmtree`
  → `rename`; link swap :922, verify only at :923 — no stage verify, no
  backup, no rollback. (Assembly's `_reuse_or_write` :82-105 has the safe
  pattern; pack doesn't use it.)
- [ ] **#9 SearXNG SSRF.** `server.py:1014` passes `body["instance"]`
  straight to `websearch.search`; `websearch.py:143-160` accepts any
  http(s) host (private-address refusal explicitly N/A). Correction vs
  audit: error echo truncates to 200 chars (`_open` :38-44), not 4KB.

## Maintainability — all still valid

- [ ] **#10 God header.** `runtime/model/RuntimeImpl.hpp` = 3074 lines,
  still included by 7 TUs (`Runtime.mm:46`, `RuntimeEncodeVerify.mm:1`,
  `RuntimeEncode.mm:1`, `RuntimeEncodeDraft.mm:1`, `RuntimeNgram.mm:1`,
  `RuntimeDiffusion.mm:20`, `RuntimeAne.mm:1`).
- [ ] **#11 Debug strides/dumps.** `RuntimeImpl.hpp:1990-2196`: hardcoded
  2048/1024 strides (:2107,:2148,:2158,:2159,:2170), `/tmp/engine_*.bin`
  writes (:2172,:2190), `RICHENGINE_VERIFY_DUMP` getenv in loop (:1990-1993).
  `RICHENGINE_DRAFT_DEBUG` now raw-getenv in `model/DSparkDraft.cpp:100`.
- [ ] **#12 Static policy caches.** `TargetModel.cpp:610`,
  `TargetModelGemma.cpp:169,:227`, `Linear.cpp:717`, `MoE.cpp:449`;
  mutable `policySlot()` `ModelPolicy.hpp:184-187`, installed
  `:199-201`. Races across runtimes.
- [ ] **#13 Env-knob sprawl.** ~16 unregistered raw `getenv("RICHENGINE…")`
  sites: `WeightStore.cpp:160-234` (AF4_*), `ModelPolicy.hpp:146-173`,
  `Scheduler.cpp:386,410`, `Engine.cpp:1288,1983`, `Bootstrap.mm:194`,
  `GgufPreparation.cpp:190`, `RuntimeImpl.hpp:677,714,1993`,
  `TargetLoader.hpp:48`, `DFlashDraft.hpp:33`, `DSparkDraft.cpp:100`,
  `dispatch.mm:82`, `PagedAttention.cpp:450` (`VERIFY_M8_OFF`, unregistered
  `envFlag`). `Env.hpp` still `atoi`/`atof` (:27,:32). Tuning.hpp registers
  86 names.
- [ ] **#14 `laneEntry` underflow.** `RuntimeBindings.hpp:283-288`
  `entries.size()-1` on empty span → SIZE_MAX; `encodeBatchTreeKvCompact`
  `RuntimeEncodeVerify.mm:426-436` calls `bindPageTables` over fixed
  `kLaneCount` without a lanes guard.
- [ ] **#15 `runtime/qwen4` unwired, grew.** Now ~113 files (was 22);
  zero build/CI references; still swept into `PRODUCTION_ENGINE_INPUTS`
  build-identity hash (Makefile:266-269). New unwired tests under
  `dev/tests/engine/qwen4/`.
- [ ] **#16 Duplicated family loaders.** `Qwen3_8.cpp`/`Ornith9B.cpp` (79
  lines each) byte-identical except include/name; `Qwen3_6Moe.cpp:12-45` /
  `MiniCpmV47.cpp:14-45` `readFfn` verbatim; Lfm2/Lfm2Moe same skeleton.

## Low severity — all still present

- [ ] Whole-map evictions `MetalEncode.mm:991` (gpuSpans>256), `:1044-1047`
  (bakedSpans>256), `:1410-1413` (preparedCache>64).
- [ ] Safetensors offsets: `pack.py:226-244` checks size math only, no
  `0 <= start <= end <= len(raw)-base` bounds.
- [ ] `/ready` public route → `backend.is_ready()` RPC (`server.py:248-259`,
  `:518-537`).
- [ ] `HostPool.unload` TOCTOU (`model_host.py:829-848`): id resolved under
  lock, `unload()` after release without re-check.
- [ ] Packed installs skip `assembly.collect_garbage()` (`upstream.py:571-579`
  vs `:620`).
- [ ] No free-space preflight on installs (`hub.py:357-363`, `pack.py:888`).
- [ ] Per-connection `threading.Timer` (`server.py:176-179`).
- [ ] Installer `cancel()` doesn't kill process group (`installer.py:150-158`).
- [ ] Markdown renderer duplicated (`web/src/pages/chat/markdown.ts` vs
  `server/chat.html`).
- [ ] `MEMORY_KEY` not in `RESET_KEYS` (`web/.../Settings.tsx:64-75`).
- [ ] Oversized UI components; missing `aria-describedby`; timing-sensitive
  tests (`test_server.py:4142,4184,:7639`).

## Metal 4 audit — current state (unchanged conclusions)

- [ ] `simdgroup_barrier`: 20/22 sites still default `seq_cst`
  (`moe.metal`×9, `gguf_staged_tile.h`×4, `linear_gguf_sgmatrix`×2,
  `gguf_score`×2, `paged_store_row{,_fp8}`×3); only `decode/gdn.metal:743,
  :1067` scoped.
- [ ] `split_reduce.h` wait/notify dead-gated
  (`RICHENGINE_SPLIT_WAIT_NOTIFY` + `__HAVE_ATOMIC_WAIT_NOTIFY__`; family-10
  rejects `SIMDGroupParallelForwardProgress`; no Makefile define).
- [ ] Q5_K/Q6_K still staged composites, no packed operands
  (`LinearGguf.cpp:62-72`); packed formats gated `RICHENGINE_GGUF_PACKED_ON`
  (measured 2.5–4× slower — off).
- [ ] GGUF-under-ICB suspension still live (`Linear.cpp:852-861`,
  `MoE.cpp:659-674`); `GPU_SPANS` opt-in + now suppressed under
  `MTL_SHADER_VALIDATION` (`MetalEncode.mm:761-775`).
- [ ] `MTL4_EVENTS` remains opt-in (unresolved `cold_output_match` history);
  `SPAN_BARRIERS` opt-in; `SPAN_BANK` automatic default-on
  (`SPAN_BANK_OFF` hatch).
- [ ] Stale comment: `MetalBackendImpl.hpp:997` still calls MTL4 opt-in —
  it is default-on since (`Tuning.hpp:232`).

## Intersection with in-flight work

- `make package` cannot ship the unified dual-geometry binary until #2 is
  fixed (staged `import server.server` fails on missing `decisions.py`/`dfa.py`).
- New knobs from this workstream (`AF4_*`, `VERIFY_GEOMETRY`,
  `VERIFY_M8_OFF`) belong to finding #13 — register them in `Tuning.hpp`
  rather than raw `getenv`.
- 27B residual gap (~6% at B4): confirmed architectural (upstream rewrote
  decode pipeline: `sparse_attention`, `block_indexer`, rewritten
  gdn/sampling/paged_attention/moe); GDN commit already folded
  (`gdnTape_` default-on), host encode proven dead end.

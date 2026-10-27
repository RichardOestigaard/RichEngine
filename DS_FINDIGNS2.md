# RichEngine Codebase Findings

Date: 2026-10-26
Scope: `/Users/richardgoestigaard/Documents/GitHub/splash` (branch `dev`, HEAD `07a733f5`)
Method: read-only review of `server/`, `install/`, `runtime/`, `web/src/`, tests, build files, and CI. `dev/tools/check_architecture.py` was run against both the working tree and a clean `HEAD` worktree. No code was changed. A run of the six unexecuted test modules was started but aborted, so test results below are structural, not execution results.

## Stop-the-line

### 1. CI is red on `HEAD`

`dev/tools/check_architecture.py` exits 1 with **19 violations** on a clean `HEAD` worktree as well as on the current working tree, so this is not caused by uncommitted work.

Examples:

- `runtime/ops/Linear.cpp`, `MoE.cpp`, `GDN.cpp`, `DraftAttention.cpp`, `ExecutionPlans.cpp` include `model/ModelPolicy.hpp`, which the rule rejects as an operator depending on an upper layer.
- `runtime/model/RuntimeDiffusion.mm`, `RuntimeEncodeDraft.mm`, `RuntimeEncodeVerify.mm`, `TargetModel.cpp`, `TargetModelGemma.cpp` call `graph.add(...)`, which the rule rejects as the model dispatching a Metal pipeline directly.
- `runtime/model/Runtime.mm`, `RuntimeAne.mm`, `RuntimeDiffusion.mm`, `GgufPreparation.cpp` use sleep-counting clock names the rule rejects.
- `runtime/metal/MetalEncode.mm` includes `model/ModelPolicy.hpp`.
- `install/launcher.py` uses four absolute `install` imports instead of relative imports.

`dev/Makefile:94-97` makes `check-source` depend on `architecture-check`, and `.github/workflows/ci.yml` runs `make check-source check-native-cpu`. Either the rules or the code needs to move before a pull request from `dev` can pass.

Also missing: there is no test that runs `check_architecture.check()` against the real repository root; `dev/tests/engine/test_architecture.py` only checks synthetic temporary trees.

### 2. Release packaging omits imported server modules

`dev/tools/package.py:53` defines `SERVER_FILES` by hand. It omits both `decisions.py` and `dfa.py`, although `server/frontend.py:16-17` imports them at startup. The packaging gate at `package.py:254` runs `import server.server` from the staged tree, so `make package` should fail at its own verification step. If that gate were bypassed, the packaged server would fail with `ModuleNotFoundError`.

The same manual list is a repeated maintenance hazard: a script comparing `SERVER_FILES` against `server/*.py` reports exactly `decisions.py` and `dfa.py` missing.

### 3. The SolidJS UI is neither built nor tested in CI, and is absent from release packages

- `.github/workflows/ci.yml` has no `pnpm` job.
- `make package` depends on `install-environment all` (`dev/Makefile:6`), not on `webui`.
- `server/webui/` is gitignored (`.gitignore:20-21`), and `SERVER_FILES` contains no webui assets.
- The server falls back to the legacy `server/chat.html` when `server/webui/index.html` is absent (`server/server.py:507-508`).

Result: releases ship the legacy chat page, and TypeScript or build breakage in `web/` is invisible to CI. `web/package.json` has `typecheck` and `build` scripts but no test script.

### 4. 139 tests never run in CI

`dev/Makefile:73-82` runs the engine discovery suite plus `test_server.py` and five named top-level modules. It omits:

| Module | Test methods |
| --- | ---: |
| `dev/tests/test_dfa.py` | 18 |
| `dev/tests/test_pack.py` | 7 |
| `dev/tests/test_tuning.py` | 82 |
| `dev/tests/test_local.py` | 7 |
| `dev/tests/test_webfetch.py` | 15 |
| `dev/tests/test_websearch.py` | 10 |

That is 139 test methods, including the private-address, credential-in-URL, and redirect checks in `test_webfetch.py`. The comment at `dev/Makefile:71-72` says discovery keeps the gate complete, but discovery is scoped to `dev/tests/engine`.

## High-value bugs

### 5. Concurrent model loads can start two engines and leak one

`server/model_host.py:471-475` checks `state == "loading"` under the lock, then `_hold()` runs outside the lock at `:478-484`. The second critical section at `:485-492` checks `closed` and idleness but never re-checks `loading`, then calls `_unload_locked()` and sets `state = "loading"`.

Two overlapping loads can both pass the first check and both build a stack. `_publish_locked()` at `:500-501` overwrites `self.runtime`, `self.backend`, `self.app`, and `self.record` without closing the previous stack, leaking an engine process and GPU memory.

Fix: reserve the load under the first lock (set `loading` there and reset on failure), or add a dedicated load lock; re-check state and `closed` under the second lock and immediately before publish; refuse to overwrite a live stack.

### 6. Batch write failure double-releases admission slots

`server/runtime.py:727-733` unconditionally releases every slot acquired by `submit_batch()` in its exception path. But `_write_bytes()` calls `_begin_generation_failure()` on a transport failure (`runtime.py:1084-1088`), and that function clears `_pending` and releases the admission slot for every pending call at `runtime.py:1486-1488`.

Because `_admission_slots` is a `threading.BoundedSemaphore` (`runtime.py:470`), the second release raises `ValueError("Semaphore released too many times")`, masking the real `EngineUnhealthy` failure.

Fix: release only calls actually removed from `_pending`, matching `submit()` at `runtime.py:653-666`; or have `_begin_generation_failure()` return the released IDs and skip them here.

### 7. GGUF install has a path-traversal and metadata-validation defect

`install/gguf.py:363-369` derives a template filename from `tokenizer.chat_templates[0]`:

```python
template_file = str(names[0]) + "_template.jinja"
```

The value is not validated. `install/assembly.py:311` then writes `stage / name` through `_write_durably()` without `models.is_safe_path()`. A crafted GGUF can put `../` in the template name and write outside the metadata staging directory.

Separately, `_check_metadata` at `install/assembly.py:324-327` requires `set(hashes) == set(gguf.DERIVED_FILES)`. `DERIVED_FILES` has only the fixed `tokenizer/chat_template.jinja`; a legitimate named-only template produces an extra key and can never validate, so the install repeats or fails.

Fix: validate the template name with `models.is_safe_path` and restrict it to a safe stem; either add the derived name to the metadata record or reject named-only templates explicitly.

### 8. Packed install is not crash-safe and verifies after publishing

`install/pack.py:903-905` does:

```python
if destination.exists():
    shutil.rmtree(destination)
stage.rename(destination)
```

It then links the selection at `pack.py:922` and only afterward calls `verify(selection.link)` at `:923`.

A crash between `rmtree` and `rename` leaves the old package deleted and the selection dangling. A verification failure after linking cannot roll back because the previous working package is already gone.

Fix: verify the completed stage before touching `destination`; use a backup/rename/verify/delete sequence with rollback on failure.

### 9. Caller-controlled SearXNG instance is an SSRF

`server/server.py:1005-1017` passes `body.get("instance")` directly into `websearch.search()`. `server/websearch.py:143-162` accepts any `http(s)` host and builds the request base from it; the comment explicitly says the private-address refusal does not apply. `websearch.py:32-44` can also include up to 4 KB of an HTTP-error response body in the returned error.

A client or model-driven tool can use the API as a proxy to loopback, LAN, or cloud-metadata services. The endpoint is localhost-bound without authentication by default, which limits but does not remove the risk.

Fix: make the instance an operator-configured allowlist, or validate resolved addresses except for explicitly configured local instances. Do not echo provider response bodies in errors.

## Medium maintainability

### 10. `RuntimeImpl.hpp` is a 3,072-line god header

`RuntimeImpl.hpp:1876-2565` defines `finalizeDecode` inline, including a ~500-line debug block at `:1933-2440`. The header is included by seven translation units (`Runtime.mm`, `RuntimeDiffusion.mm`, `RuntimeAne.mm`, `RuntimeEncodeDraft.mm`, `RuntimeEncodeVerify.mm`, `RuntimeEncode.mm`, `RuntimeNgram.mm`).

Every decode-finalization change recompiles most of the model runtime, and scheduling, Metal binding, sampling, and debug I/O share one abstraction level. Move `finalizeDecode` and related inline bodies into `RuntimeDecode.mm`/`RuntimeFinalize.mm`; split debug instrumentation into its own translation unit.

### 11. Debug finalization uses hardcoded strides and unconditional `/tmp` writes

`RuntimeImpl.hpp:2105` reads 2,048 draft-hidden elements regardless of `geometry.draft.hiddenSize`; `:2119` uses a hardcoded 64-element ring stride; `:2146` uses `slot * 2048` for `ContextProjected`; `:2156-2157` use `slot * 1024` and `slot * 2048`. `:2170` and `:2188` write `/tmp/engine_*.bin` whenever `RICHENGINE_DRAFT_DEBUG` is enabled, and `:1991` calls `getenv("RICHENGINE_VERIFY_DUMP")` inside a per-lane loop.

On a model with different hidden/context/ring dimensions, the debug path reads the wrong rows and can read past the final row. Replace the constants with geometry accessors, gate dump paths behind an explicit output-directory knob, and read the environment once in the constructor.

### 12. Model policy is process-global but cached in function-local statics

`ModelPolicy.hpp:184-200` stores one mutable policy in a function-local static, and `Runtime.mm:54-56` overwrites it for every `Runtime`. Several call sites cache the value in `static const` locals:

- `runtime/model/TargetModel.cpp:610`
- `runtime/model/TargetModelGemma.cpp:169`, `:227`
- `runtime/ops/Linear.cpp:717`
- `runtime/ops/MoE.cpp:449`

A second runtime with a different maker inherits the first model's `noFusedGate`, `gegluOff`, `prefillFastInt8`, and `moeTopK`; concurrent construction also races on the unsynchronized slot. This contradicts `MetalEncode.mm:98-102`, which deliberately reads the policy per call.

Fix: make the resolved policy an immutable member of `Runtime`/`TargetModel` and pass it into plans; at minimum, remove every static policy cache and synchronize installation/reset.

### 13. 163 environment knobs, 86 registered, invalid values silently accepted

There are 163 unique `RICHENGINE_*` names in runtime `.cpp/.hpp/.mm` files (excluding `runtime/qwen4`), but only 86 appear in `Tuning.hpp`. Raw reads remain in `WeightStore.cpp:160-234` (`AF4_*`), `ModelPolicy.hpp:146-173`, `Scheduler.cpp:386`/`:410`, `Engine.cpp:1288`, and `RuntimeImpl.hpp:1991`.

`runtime/Env.hpp:25-32` uses `atoi`/`atof`: negative values wrap into huge `uint32_t` values, and non-numeric strings silently become zero. A typo such as `RICHENGINE_GDN_CHUNKED=-1` changes behavior with no diagnostic.

Fix: route every knob through `Tuning`/`ModelPolicy`, use `std::from_chars`/`strtod` with range and `isfinite` checks, and delete or explicitly mark debug-only knobs.

### 14. `laneEntry` underflows on an empty span

`runtime/model/RuntimeBindings.hpp:283-284` computes `entries.size() - 1` without checking `entries.empty()`. For an empty span this becomes `SIZE_MAX`, so `std::min(lane, SIZE_MAX)` returns `lane` and indexes out of bounds.

`encodeBatchTreeKvCompact` calls `bindPageTables(entries, ...)` at `runtime/model/RuntimeEncodeVerify.mm:436` without the empty/size guard used by `encodeTargetVerifyBatchForward` at `:10-13`.

Fix: throw `std::invalid_argument` at the top of `laneEntry` when `entries.empty()`, and validate the batch shape in `encodeBatchTreeKvCompact`.

### 15. `runtime/qwen4/` is unwired duplicated code

`runtime/qwen4/` contains 22 files whose names duplicate `runtime/model/` and ~13.4k lines total. It is tracked in git, but no `Makefile`, `dev/Makefile`, `dev/native.mk`, or CI workflow references it, and no test target builds `dev/tests/engine/qwen4/*`.

This is a fork/drift hazard: fixes to `runtime/model` and `runtime/ops` are not reflected in `runtime/qwen4`, and the copy is invisible to builds. Either wire it into a dedicated experimental target with tests or move it out of the production tree.

### 16. Model family loaders are near-identical copies

`runtime/model/Qwen3_8.cpp` and `Ornith9B.cpp` are both 94 lines and differ essentially only in include/function names and the registered arch (`Qwen3_8.cpp:55-90` vs `Ornith9B.cpp:55-90`).

`Lfm2.cpp:19-61`, `Lfm2Moe.cpp:20-63`, and `Lfm2Moe24B.cpp` repeat the same mixer readers; `Qwen3_6Moe.cpp:11-48` and `MiniCpmV47.cpp:11-48` share the same FFN reader. Every layout fix must be replicated across copies.

Fix: extract a shared loader parameterized by family hooks, keeping only registration and constants in the small files.

## Low severity

- **Cache overflow clears everything** — `runtime/metal/MetalEncode.mm:991` and `:1044` call `gpuSpans.clear()`/`bakedSpans.clear()` above 256 keys, and `:1410-1413` clears the whole `preparedCache` above 64 keys. One new graph shape discards unrelated hot spans and causes re-bake latency spikes. Use bounded global LRU eviction.
- **Packed installs bypass garbage collection** — `install/upstream.py:571-579` returns after `pack.install(...)`; `assembly.collect_garbage()` is only called at `:620` for the assembly path. Superseded packages and staging directories accumulate.
- **No free-space preflight** — `install/hub.py:357-363` prints download bytes but never compares them to `shutil.disk_usage`; `install/pack.py:888` stages a full copy. Large models can exhaust disk mid-install.
- **Safetensors offsets are not range-checked** — `install/pack.py:222-244` checks declared size against `end - start` but never checks `0 <= start <= end <= len(raw) - base`. Malformed shards produce opaque errors or read header bytes as tensor data.
- **Per-connection OS timer** — `server/server.py:176-179` starts a `threading.Timer` per accepted connection and cancels it in `finish()`, even though sockets already have `HTTP_IO_TIMEOUT` and responses always close the connection. Use one shared reaper.
- **Public `/ready` probes the engine per request** — `server/server.py:252-259` routes `/ready` as public; the handler at `:518-537` calls `backend.is_ready()`, which can perform a native RPC with a multi-second timeout. Cache/coalesce readiness or require auth off-loopback.
- **`HostPool.unload()` can unload the wrong model** — `server/model_host.py:831-848` selects a host by `model_id`, releases the lock, then calls `host.unload(force=force)` without the ID. A concurrent load can swap the resident; re-validate the ID under the host lock.
- **Installer cancellation does not signal the process tree** — `server/installer.py:91-101` starts with `start_new_session=True`, but `cancel()` at `:150-158` only calls `terminate()`. `server/tuner.py:489-501` uses `os.killpg`; do the same with a fallback.
- **Markdown renderer duplicated** — `web/src/pages/chat/markdown.ts:1-2` says it is a verbatim port of `server/chat.html`, which is still served at `/classic` and as the `/` fallback. Fixes must be applied twice and can diverge.
- **Model-written memories are not reset** — `web/src/pages/chat/tools.ts:24` defines `MEMORY_KEY`; `web/src/pages/Settings.tsx:64-75` omits it from `RESET_KEYS`, and `clearChats()` only removes `STORE_KEY`. Up to 200 model-written facts persist through “Reset all settings.”
- **Oversized UI components** — `web/src/pages/Models.tsx` is 1,844 lines with 24 `createSignal` calls; `Chat.tsx` is 1,446 lines with 23; `Playground.tsx` has 31 signals. Extract polling/install/tune/chat-session hooks.
- **Accessibility gap** — `Playground.tsx:372-373` and `Judge.tsx:335-336` set `aria-invalid` but the field-error `<p>` has no `id`, `aria-describedby`, or live region, so errors are not announced.
- **Timing-sensitive tests** — e.g. `dev/tests/test_server.py:4142,4184` assert under 0.5 seconds; `:7639-7640` sleeps 0.05 seconds then asserts call counts. Replace with event/condition waits.

## Repository hygiene

- Stray 84-byte binary named `8` at the repository root, untracked but committable.
- `CHANGELOG_UNPUSHED.md` says `server/webui/` is a “committed bundle,” but it is gitignored and generated by `make webui`.
- Core refactor files are untracked: `runtime/model/ModelRegistry.cpp`, `ModelRegistry.hpp`, `TargetLayout.hpp`, `TargetWeights.hpp`.
- `runtime/model/families/FamilyMakers.hpp` is staged as deleted.
- The working tree has 91 changed files (+3,084/−3,057) on top of HEAD, with `CHANGELOG_UNPUSHED.md` itself noting its file counts are stale.
- `runtime/model/ModelRegistry.hpp` still contains a comment saying the header “intentionally does not parse” until Phase B, which does not match its current inclusion by `ModelFactory.cpp` and every family header. Update or remove it.

## Recommended order

1. Resolve the `check_architecture.py` gate (rules or code) and add a test that runs it against the real root.
2. Fix release packaging: derive `SERVER_FILES` from `server/*.py`, add the missing modules, and assert full coverage in a test.
3. Wire the six omitted test modules into `check-python-engine`.
4. Add a CI web job (`pnpm install --frozen-lockfile`, `typecheck`, `build`, tests) and make packaging depend on `webui`.
5. Fix the server concurrency bugs: model-load reservation, batch slot ownership, and `HostPool.unload` identity.
6. Fix GGUF template validation and make packed installs atomic with rollback and GC.
7. Schedule the runtime maintainability work: split `RuntimeImpl.hpp`, remove static policy caches, centralize env parsing, and retire or wire `runtime/qwen4`.

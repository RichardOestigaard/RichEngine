# Structural Findings

Author: Claude Haiku 5.5 High
Date: 2026-10-07
Scope: 610 tracked files, about 210k lines. The working tree is mid-change (40+ modified and 40+ untracked files), so findings reflect current disk state.

## Top Structural Issues

### 1. HTTP handler is a god class (highest impact)

`server/server.py` — `FrontendHandler` (line 166)

- 1,404 lines, 47 methods.
- `do_POST` alone: 213 lines. Route lists are hardcoded per method.
- Streaming writers for three dialects (OpenAI, Anthropic, Responses) live inside the handler.

Fix: move per-dialect streams into their own modules. Replace route `if` chains with one route table. Keep the handler to HTTP plumbing.

### 2. Naming collision: "Frontend"

- `server/frontend.py` `Frontend`: 1,073 lines, 36 methods. It prepares prompts; it is not HTTP.
- `server.py` `FrontendHandler` is the HTTP layer.
- `server/runtime.py` `MultiplexedRuntime` is the native client. The top-level `runtime/` directory is C++.

Fix: rename by role, e.g. `PromptPreparer` and `NativeClient`.

### 3. Two web UIs, one build target

- `server/chat.html`: 1,070-line inline page, served at `/classic`.
- `web/` (SolidJS): builds to `server/webui` (`web/vite.config.ts`).
- `server/webui/` is not in `.gitignore`. `.gitignore` ignores `web/dist/`, which the build does not write to.

Fix: pick one primary UI. Ignore `server/webui/`, or stop tracking build output.

### 4. C++ engine methods too long

`runtime/engine/Engine.cpp`

- `Engine::retireCheckpoint`: about 237 lines (1040-1277).
- `Engine::reclaimMemory`: about 222 lines (1544-1766).
- `Engine::admit`: about 174 lines (676-850).
- `Engine::tick`: about 148 lines (177-325).

Fix: extract phases into named private methods, e.g. the restore, reclaim, and state-boundary steps.

### 5. `model/` mixes architectures with infrastructure

- Architectures: `Qwen3_6Moe`, `Qwen3_8`, `Ornith9B`, `Lfm2`, `Granite`, `Gemma4Moe`, `DiffusionGemma`.
- Infrastructure: `GgufFile`, `WeightStore`, `SlotFile`, `SafetensorsCheckpoint`, `Runtime*`.
- `runtime/model/RuntimeImpl.hpp` (1,837 lines) is included by 7 translation units. A change there rebuilds all seven.

Fix: split into `model/arch/`, `model/weights/`, `model/gguf/`, `model/draft/`. Turn `RuntimeImpl.hpp` into a narrow internal header.

### 6. Metal header layers overlap

- `runtime/metal/abi/` (19 headers) and `runtime/metal/kernels/common/` (30 headers) both define quant formats.
- Include chain: `abi/QuantFormat.h` -> `common/quant_formats.h` -> `abi/QuantTables.h`.
- Six `gguf_*` headers in `common/`.

Fix: verify ownership. Keep one source of truth per concept.

### 7. Protocol constants duplicated across languages

- `PROTOCOL_VERSION = 7` and `STATUS_SCHEMA_VERSION = 6` exist in both `server/protocol.py` and `runtime/engine/wire/Protocol.hpp`.
- Mitigated: `dev/tests/engine/test_protocol_python.py` checks parity.

Fix (lower priority): single schema source, generated for both sides.

### 8. Test layout

- `dev/tests/test_server.py`: 9,354 lines, one file.
- `dev/tests/*.py` holds real-model harnesses (`agent_real.py`, `smoke_real.py`, `tool_output.py`). Tests in `dev/tests/engine/` import them.

Fix: move harnesses to `dev/harness/`. Split `test_server.py` by endpoint family.

### 9. Build layering

- Root `Makefile` ends with `-include dev/Makefile`, which includes `dev/native.mk` (990 lines).
- `native.mk` has dozens of hand-written per-test rules.

Fix: pattern rules per test family. Replace the implicit include with explicit target groups.

### 10. Install package is flat, with one large module

- `install/launcher.py`: 1,712 lines. `_build_parser` alone: 193 lines.
- `install/gguf.py` `model_config`: 249 lines.
- Ten flat modules with overlapping "model" vocabulary: `models`, `families`, `catalog`, `upstream`, `assembly`, `hub`, `legacy`, `pack`.

Fix: group into a subpackage with per-concern modules. Split `launcher.py` into parser and command modules.

### 11. Repo hygiene

- Root holds planning and notes files: `GPT_6_LUNA_MAX_FINDINGS.md`, `diffusion_issue.md`, `ISSUES.md`, `TODO.md`.
- `docs/` has overlapping pairs: `GEMMA_DIFFUSION_OPTIMIZATION_PLAN.md` and `GEMMA_DIFFUSION_OPTIMIZE.md`.

Fix: consolidate docs. Move notes out of the root.

## What Already Works

- `dev/tools/check_architecture.py` enforces dependency boundaries. `server/backend.py` cannot import `frontend` or `api_shapes`.
- Module docstrings state ownership (e.g. `install/*.py`).
- Protocol parity is tested.

## Suggested Order

1. Split `FrontendHandler` (1) and rename `Frontend` (2). Largest gain, low risk.
2. Resolve the two web UIs (3). Quick decision; unblocks cleanup.
3. Split `RuntimeImpl.hpp` and `model/` (5). Largest C++ build-time win.
4. Break up long `Engine` methods (4).
5. Test and build layout (8, 9).

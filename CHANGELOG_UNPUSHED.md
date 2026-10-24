# Unpushed Changes

> Note: written against an earlier tree. It says 125 modified
> files; the actual count has since changed. Treat counts as stale.


Analysis of the uncommitted working tree: 125 modified files, ~40 new files,
~8,473 additions. The change set adds the Gemma 4 model family end-to-end, a
diffusion-decoding mode for DiffusionGemma, a SolidJS web UI with model
management, and per-device kernel autotuning.

## 1. Gemma 4 model family

- `runtime/model/Gemma4Moe.{hpp,cpp}` — new target for Gemma4-26B-A4B MoE:
  128 experts top-8 plus a dense GeGLU shared expert, dual attention
  geometry (256-dim sliding local layers + 512-dim global layers every 6th,
  `k_eq_v` — globals have no V projection).
- `install/pack.py` (992 lines) — install-time weight packing: HF
  safetensors → the runtime's affine Q4 format, since no GGUF/MLX loader
  maps gemma4 tensors.
- `install/families.py` — `Gemma4-26B-A4B` family signature plus the DFlash
  draft pairing (`z-lab/gemma-4-26B-A4B-it-DFlash`).
- `install/gguf.py` — SentencePiece/unigram tokenizer support and gemma4
  GGUF metadata.
- `install/upstream.py` — packed-package install path with revision and
  repack handling.
- `QwenTarget` templated on layout — dual-geometry attention
  (`altAttentionMask`), `gemmaMoe` routing, canvas sequence support.
- Metal kernels: `moe_route_scores_gemma`, `moe_route_select_gemma`,
  `geglu_multiply`, fused GeGLU expert tile; new `*_gemma_hd512`
  paged-attention verify variants including `_m2` two-pass register-saving
  versions and SWA-windowed forms.
- Server side: `tool_schema.py` gemma4 `<|tool_call>` grammar, `output.py` `_Gemma4CallProjector`
  streaming parser, `chat_templates.py` context-dependent
  generation-prompt probing.
- `docs/GEMMA4_PLAN.md` research doc; `run-gemma.sh`.

## 2. DiffusionGemma — diffusion decoding

- `DiffusionGemma.{hpp,cpp}` — Gemma4 MoE trunk run denoising-style over a
  256-token canvas with bidirectional attention.
- `DiffusionSampler.{hpp,cpp}` — entropy-bounded accept mask and
  early-exit policy (CPU-only for testability).
- `RuntimeDiffusion.mm` (762 lines) — denoising loop: multi-lane canvases,
  commit prefills, speculative prefill.
- `ops/Canvas.{cpp,hpp}` + `decode/canvas.metal`, `prefill/canvas.metal`,
  `common/GemmaKernels.h` — canvas noise, row-stats, accept and
  soft-embed kernels.
- `RuntimeImpl.hpp` — canvas arenas plus env knobs:
  `RICHENGINE_CANVAS_STEPS_PER_CMD`, `_PREFIX_EXIT`, `_COMMIT_TAIL`,
  `_SPECULATIVE_PREFILL`, `_EXIT_STABLE`.
- `docs/GEMMA_DIFFUSION_OPTIMIZATION_PLAN.md` — bandwidth roofline
  analysis (~25-28 GB/step, projected 130-690 tok/s by M5 SKU).
- `diffusion_issue.md`.

## 3. SolidJS web UI

- `web/` — SolidJS + Vite + ResolvCharts app: Chat, Models, Metrics, Disk,
  Settings, Playground and Judge pages.
- Builds into `server/webui/` (committed bundle), served at `/` by
  `FrontendHandler`; `make webui` target.

## 4. Model-management HTTP API

- `server/model_host.py` — load/swap/unload lifecycle (a swap unloads
  first for RAM headroom).
- `server/disk.py`, `server/installer.py` — disk wipe/dedupe actions and
  background install jobs.
- New endpoints: `/v1/disk`, `/v1/disk/wipe`, `/v1/disk/dedupe`,
  `/v1/models/available`, `/v1/models/install` (+`/cancel`),
  `/v1/models/load`, `/v1/models/unload`.
- New flags: `--models-dir`, `--unload-idle`, `--moe-union`.

## 5. MoE route union cap (experimental)

- `moe_route_cap` kernel plus `--moe-union N` / `RICHENGINE_MOE_UNION`:
  caps distinct routed experts per decode step, dead-marking overflow
  routes.

## 6. DeviceTuning / DevicePolicy

- `ops/DeviceTuning.cpp`, `ops/DevicePolicy.hpp` — measured
  per-device-family kernel plan tables (split tiers, measured linear
  plans) replacing hardcoded shape-match overrides. `Linear.cpp` and
  `LinearGguf.cpp` shrank accordingly.

## 8. Metal 4.1 decode pathways

Opt-in Metal 4.1 surface for decode. All flags default off; all verified
byte-exact (identical `output_token_hash` and acceptance rates) in ABBA
decode runs on MiniCPM5-2B-MLX, Ornith-1.5-9B-MLX-4bit and
Qwen3.8-27B-4bit.

- `ComputeDispatch::indirectGrid` / `indirectRange`
  (`MetalBackend.hpp`, `CommandGraph::addIndirect`/`setIndirectRange`,
  `MetalEncode.mm` both encoders) — GPU-written
  `MTLDispatchThreadgroupsIndirectArguments` launch sizes and
  `MTLIndirectCommandBufferExecutionRange` span replay ranges.
  Substrate for device-driven decode; `decode_dfa_mask` is the first
  consumer under `RICHENGINE_GPU_DECODE_GRID` (`decode_dfa_mask_grid`
  kernel writes the grid, consuming `DfaForcedSpan` on-device).
- `commit4` handles `EventStep` commands — queue-level
  `signalEvent:`/`waitForEvent:` splits with per-commit
  `MTL4CommitFeedback` merged via `SplitFeedback4`; event-bearing commands
  no longer fall back to the Metal 3 encoder.
- GPU-encoded spans replay on the MTL4 path (`gpu_encode` +
  `executeCommandsInBuffer` through the argument table) — previously a
  silent scalar-encode no-op under `RICHENGINE_MTL4`.
- Pipeline-name grammar (`MetalEncode.mm` `newSpecializedPipeline`):
  `kernel[hook=impl:arity]` binds `[[stitchable]]` call sites via
  `MTLStitchedLibraryDescriptor` + `linkedFunctions.privateFunctions`
  (MTL3) / `MTL4Compiler` + `MTL4StaticLinkingDescriptor` (MTL4);
  `kernel|persistent,fp=weak|fp=simd,relief=none` sets
  `MTL4ComputePipelineDescriptor` options through `MTL4Compiler`.
- `gdn.metal` intra-simdgroup barriers narrowed to
  `memory_order_acq_rel` + `thread_scope_simdgroup`.
- `split_reduce.h` gained an MSL 4.1 atomic wait/notify merge (first
  arriver parks, last arriver notifies) — opt-in
  `-DRICHENGINE_SPLIT_WAIT_NOTIFY`; this device rejects
  SIMDGroupParallelForwardProgress at pipeline creation so it stays off.
- `decode_step_pack` kernel + `verify_input_tokens_dev`: pack kernel
  writes next-step `anchor`/`live_rows`/`position_base`/`cache_length`
  from `RetainedCount`/`AcceptedCount`/`OutputTokens`; verify-input
  tokens bind the pack buffer when provably identical to the staged
  payload. Opt-in `RICHENGINE_DEVICE_VERIFY_PARAMS`.
- `Engine.cpp` depth-2 decode submission ring behind
  `RICHENGINE_DECODE_RING2`: `trySubmitAheadDecode` plans and submits the
  next decode before the retiring plan's `apply()`, overlapping the host
  turnaround with the ahead command's GPU execution.
- GGUF Q2_K native packed tile (`common/q2k_mpp_tile.h`,
  `decode/linear_gguf_q2k_sgmatrix.metal`, `gguf_decode_q2km_*`): matmul2d
  on `uint2b_format` packed data instead of staged dequant; routed under
  `nativeFormats()` + `RICHENGINE_GGUF_PACKED_ON` like its siblings.
- `dev/benchmarks/mtl4_benchmark.mm` — three-graph (direct/icb/gpuspan)
  A/B harness; `--compare` re-execs both encoder paths and prints the
  delta table.
- Verified non-starters on toolchain 32023 / Apple family 10 (recorded in
  `UNTAPPED_POTENTIAL.md`): `bf16 × fp8` matmul2d operands (no intrinsic
  — fp8 right operands pair only with half or same-format fp8) and
  `resetCommandsInBuffer:` (the replay path never rewrites commands).

## 9. Misc

- `model/NgramIndex.hpp` — n-gram predraft logic extracted into a pure,
  testable header.
- `StderrLine.hpp` / `StartupLog.hpp` — ANSI-styled, sanitized log lines
  matching the launcher's palette.
- `sampling.metal` (+451) and `paged_attention_tile.h` (+256) reworks.
- Docs and review artifacts: `FINDINGS.md`, `GPT_6_LUNA_MAX_FINDINGS.md`,
  `RECOMMENDED_MODELS.md`, README benchmark tables,
  `TREE_VERIFY_DESIGN.md`.
- New tests: `gemma4_target`, `diffusion_gemma`, `canvas_kernels`,
  `hd512_attention`, `ngram_index`, `gelu_saturation`, `test_gemma4.py`,
  `test_diffusiongemma.py`, `test_pack.py`; large expansions to
  `draft_selector_metal_test` and `test_server.py`.

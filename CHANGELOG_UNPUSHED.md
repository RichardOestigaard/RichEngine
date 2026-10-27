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

## 7. Metal 4.1 decode pathways

Metal 4.1 surface for decode, measured A/B on MiniCPM5-2B-MLX,
Ornith-1.5-9B-MLX-4bit, Qwen3.8-27B-4bit and
LFM2.5-24B-A2B-GGUF:QAD-Q4_0 with byte-exact output (identical
`output_token_hash` and acceptance rates). The measured winners are now
default-on with `RICHENGINE_*_OFF` escape hatches; unproven or losing
paths stay opt-in.

- `ComputeDispatch::indirectGrid` / `indirectRange`
  (`MetalBackend.hpp`, `CommandGraph::addIndirect`/`setIndirectRange`,
  `MetalEncode.mm` both encoders) — GPU-written
  `MTLDispatchThreadgroupsIndirectArguments` launch sizes and
  `MTLIndirectCommandBufferExecutionRange` span replay ranges.
  Substrate for device-driven decode; `decode_dfa_mask` is the first
  consumer under `RICHENGINE_GPU_DECODE_GRID` (`decode_dfa_mask_grid`
  kernel writes the grid, consuming `DfaForcedSpan` on-device).
- Event-bearing commands stay on the Metal 3 encoder: the commit4
  `signalEvent:`/`waitForEvent:` split was implemented, benchmarked and
  reverted — it misattributed GPU timing and regressed prefix-hit
  prefill output (a `cold_output_match` failure) plus ~25–35% cold-prefill
  wall. `commit4` requires `events.empty()` again.
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
  payload. Default on; `RICHENGINE_DEVICE_VERIFY_PARAMS_OFF` disables.
- `Engine.cpp` depth-2 decode submission ring: `trySubmitAheadDecode`
  plans and submits the next decode before the retiring plan's `apply()`,
  overlapping the host turnaround with the ahead command's GPU execution.
  Default on; `RICHENGINE_DECODE_RING2_OFF` disables.
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

## 8. Constrained decode (DFA) fast path + conv fusion

Agentic-workload (tool-call / JSON-schema) constrained decode, plus the
LFM conv-layer fusion it led to. All verified with identical output
hashes; measured via the new `benchmark-dfa` harness
(`dev/benchmarks/dfa_pack.py` + `dfa_benchmark.mm` — drives real
constrained decode in-process with a C++ DfaMirror answering mask
requests; `make benchmark-dfa MODEL=...`).

- Live-row mask requests: `simulationTokens` sends anchor + live
  proposals only (was: all `kDraftProposalTokens`); `verifyMaskRows`
  tracks the response row count. ~65% smaller mask responses on span
  rounds (measured mask rows 218 → 174/step).
- `decode_dfa_span_apply` kernel + merged constrained step
  (`RICHENGINE_DFA_MERGE`, now default-on): a Full-mode batch whose
  lanes all carry GPU DFAs encodes draft → target → commit as ONE
  CommandGraph — zero host syncs per step (was: two). Default on;
  `RICHENGINE_DFA_MERGE_OFF` disables.
- `RICHENGINE_DFA_SPAN_SKIP_DRAFT` (default-on): skip the draft
  embedding+forward when every lane's pending GPU forced span covers its
  proposal budget; draft state is unaffected (state advances via
  `addContextCommit`, never via draft decode). +2% on full-mode decode
  wall in combination with the merge.
- `RICHENGINE_GPU_DECODE_GRID` (default-on): `decode_dfa_mask` launches
  via GPU-written indirect grid.
- server/dfa.py bounds: `MAX_DFA_STATES` 4096 → 32766 (one below the
  wire's 32767), `MAX_OPTIONAL_PROPERTIES` 4 → 64; `MAX_ARRAY_BOUND` and
  `MAX_SCHEMA_DEPTH` kept (semantics / minimization blowup). More real
  tool schemas now reach the DFA path.
- Host-side: `commit(..., forced=False)` skips discarded forced-token
  computation under GPU modes (`backend.py`); opt-in per-state mask-row
  LRU under `RICHENGINE_DFA_MASK_CACHE` with `mask_stats()` counters
  (off by default — pure-Python mask cost makes it a measurement hook).
- Result on MiniCPM5-2B-MLX, 694-state tool-call DFA, width 2:
  311.6 tok/s vs 297.2 host-mask baseline (+4.9%), and +14% over
  unconstrained decode.
- `RICHENGINE_DFA_MODE=full` stays opt-in: GPU-mask emission produced a
  slightly different token stream in one position (unresolved parity
  question); `forced` mode is parity-clean.

## 9. LFM conv fusion (now default-on)

- `verify_lfm_conv_v4_table16` (shared/lfm_conv.metal): the conv verify
  mixer also writes the out-projection's Table16 operand (values +
  seeds/sums32 in `write_input`'s exact xor-shuffle order), letting the
  register-tile out_proj skip its `decode_linear_gguf_prepare` dispatch —
  ~30 fewer dispatches/step at widths 3–4 on LFM2.5-24B.
- Bug found in review: the fused sums differed a few ULP from the
  reference — Metal fast-math reassociates the adds while
  `linear_gguf_sgmatrix.metal` compiles `#pragma clang fp
  reassociate(off)`. Fix: the same pragma scoped inside the `if
  constexpr (Table16)` block; all widths now bitwise identical
  (`/tmp/conv_table_test` harness).
- Measured decode GPU: −1.9% (B1), −1.7% (B2), −0.8% (B3), −0.2% (B4);
  byte-exact hash/acceptance. `RICHENGINE_CONV_FUSE_OFF` disables.
- Related findings on LFM2.5-24B-A2B-GGUF:QAD-Q4_0: the packed `q40m`
  native tile loses to staged dequant (−14…21% at B1–B2 — stays off);
  MTL4 submission is ~1% slower than MTL3 on this conv-heavy model
  (first MTL4 regression observed — noted for the device policy table).

## 10. Misc

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

## 11. Budget-aware constrained decode (byte termination bound)

B_bytes (DPS_V41F P0 #2's byte-level upper bound) shipped as the constraint
DFA's per-state cost table, gating GPU mask rows on the decode budget:

- `server/dfa.py` — `CompiledDfa.byte_costs()`: per state, the minimum
  byte count of a walk to acceptance over unmarked edges (32767 = none a
  regular token can finish). Label-correcting reverse relaxation over the
  table, cached per compiled grammar.
- `server/protocol.py` / `server/backend.py` — `ConstraintDfaFrame.costs`
  shipped as an optional trailing section (u32 length + n_states i16) in
  `CONSTRAINT_DFA`; empty when the host has no bound. Wire stays
  compatible both ways.
- `runtime/engine/wire/Protocol.{hpp,cpp}`,
  `runtime/engine/{Engine.{hpp,cpp},wire/NativeRuntime.cpp}`,
  `runtime/model/{Model.hpp,Runtime.hpp,RuntimeRequest.hpp,Runtime.mm}` —
  the frame's cost table plumbs to request-owned `dfa-costs` buffers; a
  frame without the section keeps an all-zero stand-in whose zeros gate
  nothing.
- `runtime/metal/abi/Dfa.h` — `tokens_left` per lane in `DfaMaskParams`
  (192→208 B) and `DfaWalkParams` (204→208 B); `static_assert`s bumped.
- `runtime/metal/kernels/decode/dfa.metal` — `decode_dfa_mask`'s budget
  rule: a row's token admits iff `1 + cost(landing)` fits under the row's
  remaining-budget share, falling back to the plain walk where it does
  not; `decode_dfa_forced_tokens` caps proposed span slots at the lane's
  budget. `decode_dfa_mask` reads its cost tables at buffers 18-21 and
  the params at 22.
- `runtime/model/{RuntimeImpl.hpp,RuntimeEncodeVerify.mm}` —
  `DfaBufferKind::Cost` bound at all three `decode_dfa_mask` sites,
  `tokens_left` filled from `maxNewTokens - generatedTokens` at every
  walk/mask site.
- `runtime/Tuning.hpp` — `RICHENGINE_DFA_BUDGET_OFF` (presence) disables
  the rule at runtime for A/B on one binary; costs are always shipped.
- Tests: `dev/tests/test_dfa.py` — costs round-trip, cost-less-frame
  compatibility, and `byte_costs()` verified against the walk fixpoint
  (this caught a missing `bitorder="little"` in the marker mask unpack);
  `dev/tests/engine/native_peer.py` — the python decoder mirrors the
  optional section.

# Changelog

## Unreleased — blocked weight restore, test repairs

- `runtime/engine/wire/NativeRuntime.cpp`: a weight restore refused on an
  allocation (`metal::MetalAllocationError` other than `DriverRejected`) no
  longer stops the engine — it waits for memory, logging once, and retried
  each tick. A request blocked thirty seconds fails with a retryable
  `resource_timeout` (HTTP 503 + `Retry-After`), and the engine stays up
  for the next request. New `testRestoreRefusalWaitsAndRetries` in
  `native_engine_loop_test.cpp`.
- `dev/tests/engine/execution_plans_test.cc`: constants stale after the
  verify-rows 8→16 and draft ring 2048→4096 geometry changes — decode
  workloads now use `RICHENGINE_TARGET_VERIFY_ROWS`, and the MoE/draft/gate
  workspace bounds doubled to match.
- `dev/tests/engine/model_runtime_oracle_test.mm`: `model.draft` became a
  weights-variant — use `model.draftActualAllocatedBytes()`.
- `runtime/model/ModelDescriptor.mm`: a package whose recorded execution
  geometry predates the runtime (e.g. draft proposals 7 vs 15) now fails
  with an actionable message pointing at MLX/GGUF checkpoints.
- `runtime/metal/kernels/shared/gguf_repack.metal`: the AF4 repack read the
  MLX source's nibbles pair-interleaved instead of adjacent
  (`qs[e/2]` nibble `e&1`); every AF4 preparation produced wrong plane0.
- `runtime/model/WeightStore.*`: block images carry the
  `kBlockWeightsImageType` flag and `readProjection` dispatches on it —
  draft readers (DFlash/DSpark/DFlashV1) consume either layout. Draft
  images stay affine for now: draft attention reads the fused QKV tile
  directly, so block drafts need block-aware attention kernels.
- Verified on hardware (Apple M5 Pro, MiniCPM5-2B-MLX): `--weight-layout
  block` loads, serves and produces token-identical output to the affine
  layout; `/status` reports `weights.layout: "block"`.
- Kernel tuning covers block workloads: `tune-kernels`/`collectTuningWorkloads`
  include Block32 projections (Q4_K and AF4 segments — `blockFormatTuned` is
  the opt-in table), candidates are the staged/register tiles at every
  split, and `RICHENGINE_LINEAR_PLANS` entries may carry a `block` marker
  (`runtime/ops/DeviceTuning.cpp`, `dev/tuning/`). `richengine tune` now
  vetoes a fitted spec that changes the emitted tokens (`_token_parity`
  via `decode-profile --dump-tokens`) — the rule DeviceTuning's measured
  tables apply.
- `dev/tests/engine/q8_flash_attention_metal_test.mm`: the prefill tile
  eight-row view no longer asserts equality with the verify rows (16).
- `dev/benchmarks/gguf_decode_sweep.mm`: AF4 rows for the dense MLX shapes
  measure the repacked decode against the affine kernels'.
- Default KV format is now `int8` (`runtime/engine/RuntimeResources.hpp`,
  `runtime/main.mm`): it matches int4's page-fetch shape but scores higher
  draft acceptance, and the fp8 kv4/g6 verify tile overflows the 32 KB
  threadgroup budget at 16 rows so only int8 is usable of the 8-bit tier.
- The kv4/g6 chain verify splits its 16-row tile's two fused-row passes
  across two threadgroups (`_r2` for q8/int4/bf16): the M=96 tile sat one
  group per core at ~25 KB scratch; two groups per core streams −17% to
  −33% faster on every history/lane ([Qwen 3.8 27B Findings.md](dev/benchmarks/Qwen%203.8%2027B%20Findings.md)).

## Unreleased — block weight layout for MLX targets

- `server/serve_options.py`: `--weight-layout {auto,block,affine}` chooses the
  kernel layout MLX and draft weights load as; `auto` keeps the engine's
  default and GGUF targets always load the `block` layout
  (`server/model_host.py`, `runtime/main.mm`). A `block` request falls back
  to `affine` on sections without a block form, with a log line
  (`runtime/model/ModelFactory.cpp`, `runtime/engine/RuntimeResources.mm`).
- `/status` reports the loaded model's effective weight layout
  (`weights.layout`, status schema 7; mirrored as top-level
  `weight_layout`) and `/metrics` exports `richengine_weight_layout_info`
  (`runtime/engine/wire/Status.cpp`, `server/server.py`,
  `server/metrics.py`).
- MLX dense Q4 targets can load their weights in the GGUF `block` kernel
  layout instead of the `affine` layout
  (`runtime/model/AffineToBlock.*`, `GGUF_FMT_AF4`).

## Unreleased — thermal state, tool_reference blocks, atomic pins

- `server/thermal.py`: `/status` reports `instance.thermal_state`
  (`nominal`/`fair`/`serious`/`critical` from NSProcessInfo), `/metrics`
  exports `richengine_thermal_state{state=...}`, and a monitor thread logs
  each change (`server/server.py`, `server/metrics.py`).
- `server/api_shapes.py`: `tool_result` content accepts `tool_reference`
  blocks (Claude Code tool search), rendered as a text marker naming the
  tool. `/v1/messages` and `/v1/messages/count_tokens` share the path.
- `install/hub.py`: snapshot pins publish by atomic rename (`os.replace`)
  instead of `os.link`, fixing installs onto filesystems such as ExFAT that
  lack hard links.

## Unreleased — fixed connection worker pool

- `server/server.py`: `ConnectionWorkers` serves admitted connections from a
  fixed pool sized to the connection capacity instead of a thread per
  connection. `FrontendServer.process_request` queues each admitted
  connection; refused connections still get the retryable 503 through
  `LingeringCloser`. `server_close` drains and stops the pool.

## Unreleased — lint, test and server robustness fixes

- Root `ruff.toml` pins the CI rule set (`E4`, `E7`, `E9`, `F`, `I`) so a bare
  `ruff check` on `server/` and `install/` matches what
  `check-python-engine` enforces instead of ruff 0.16's all-rules default.
- `server/backend.py`: a `request_logger` that raises is reported through
  `log_unexpected` instead of being dropped silently.
- `server/runtime.py`: a failing engine-failure listener is reported through
  `log_unexpected`; the failure still reaches the calls and the engine.
- `server/webfetch.py`: title/text extraction failures fall back to raw text
  only for parse errors; the suppress scope no longer covers extraction.
- `server/websearch.py`, `server/images.py`, `server/diagnostics.py`:
  intentional exception swallows now read as `contextlib.suppress`.
- `server/tool_schema.py`: dropped an unused `many` local.
- Tests: `test_http_body_budget.py` follows the post-`11357d8` contract
  (`_read_json_body` raises `ValueError`; callers answer the 400). New
  coverage for a failing `request_logger`, `log_unexpected`, and the
  webfetch parse-failure fallback.
- `server/webui/` build output is no longer tracked; `make webui`
  regenerates it. Root stray notes moved under `docs/`.

## Unreleased — working tree

- New families: Ornith-1.5-9B, Ornith-1.5-35B-A3B, MiniCPM5-2B, LFM2.5-2.6B, LFM2.5-8B-A1B (`install/families.py`, `install/completions/suggested-models.txt`)
- New target/draft loaders: `Ornith9B`, `Dense`, `Lfm2`, `Lfm2Moe`, `PlainDraft`, `DSparkDraft` (Markov + confidence heads), `AnePredictor` (CoreML); `ModelFactory` dispatches draft layout via `std::visit`
- LFM2 conv support: `runtime/ops/LfmConv.*`, `runtime/metal/abi/LfmConv.h`, `shared/lfm_conv.metal`
- `--kv-format` gains `int4` (now default), `fp8`, `fp8e4m3`; new FP8 attention/store kernels (decode + prefill), `paged_attention_fp8_tile.h`, `Fp8PageFormatReference` tests
- MXFP4 GGUF kernels: `gguf_mxfp4_tile.h`, `gguf_mxfp4p_tile.h`, `shared/gguf_mxfp4p.metal`
- Tree-verify attention splits (`verifyAttentionSplits` rows param), `gdn_chunked.metal` prefill kernel, `mpp_simdgroup_attention` prototype
- Toolchain floor raised to macOS 27, `-std=metal4.1`, CoreML link, `mtl4_benchmark` target
- Server: `TokenizerContract` propagated to backend/frontend/constraints (`think_end_id`)
- New design docs: `docs/ANE_DRAFTING.md`, `docs/SPEC_DECODE_BOOST.md`, `docs/TREE_VERIFY_DESIGN.md`, `docs/TurboQuant_ANLYSIS.md`, `docs/PREFILL_OPTIMIZATION_PLAN.md`

## Unreleased — shared-prefix state, fused argmax head, int8 prefill

- Shared-prefix junction: requests carry `sharedPrefixTokens` (native
  request frame 87→91 bytes — bump `kProtocolVersion`/`PROTOCOL_VERSION`
  together before release so a mixed server/engine pair refuses). The
  server probes each chat template once with a sentinel turn to find where
  the shared system prompt and tools end, and the engine keeps a reusable
  state — the shared-prefix junction — at the last whole page inside that
  prefix, so follow-up requests resume there.
  `--shared-prefix-state` / `--no-shared-prefix-state` (default on).
  Server-logged TTFT A/B (same head, new document suffix): 2.5 s resumed
  at the junction vs 4.1–4.2 s typical without it (8.6 s worst case when
  no resumable state existed); identical prompts still replay in 0.2 s and
  cold requests are unchanged at 8.1 s.
- Fused greedy vocabulary head: on all-greedy, unconstrained, unstopped
  chain-verify steps the head argmaxes in-kernel and writes per-(row, tile)
  partials instead of logits — `decode_head_argmax_q4` for affine heads
  (128-column tiles) and `gguf_decode_*_m*_amax` for single-segment GGUF
  block heads (64-column tiles), reduced by the new
  `decode_head_argmax_reduce_tiles[_gguf]` sampling kernels. Lane semantics
  in `HeadArgmaxParams`/`GgufHeadArgmaxParams`; `RICHENGINE_HEAD_FUSED_OFF`
  disables.
- `RICHENGINE_PREFILL_FAST_INT8`: new `prefill/linear_i8.metal` two-term uint8
  prefill projection path; `I8Codes`/`I8Params` prefill tensors and
  `LinearScratch` operand buffers are zero-sized unless the flag is set.
- Adaptive proposal budgets are now on by default
  (`RICHENGINE_ADAPTIVE_PROPOSALS=0` disables): lanes that keep rejecting pay for
  fewer live verify rows.
- Draft forwards fold residuals into the projections
  (`ops::LinearEpilogue::Residual`) instead of separate `addResidual` passes.
- `max_position_embeddings` checks accept a source value smaller than the
  runtime's (`requireAtMost`): exports disagree on the trained context
  (131072 vs 128000).
- Stateless targets (dense Llama-family models such as MiniCPM5-2B) no
  longer dereference GDN state they do not have: `bindGdnStates` is skipped
  when `stateLayout.layers` is zero, and `QwenStateStorage::acquire` still
  constructs the (zero-byte) cells a lane binds when admission short-circuits.
- `backend-benchmark` prompts clamp into the model's vocabulary (unchanged
  for vocabularies ≥ 200,100), so the small-vocabulary families run it;
  `gguf_decode_sweep` joins the dev benchmarks.
- Benchmarks, M5 Pro (20-core GPU, 48 GB), `test-performance-real`: 35B-A3B
  260 tok/s B1 / 703 B4, 27B 103/246, MiniCPM5-2B Q4_K_M 141/390,
  LFM2.5-2.6B MXFP4 139/319 (README.md, docs/performance.md).

## Unreleased — Apple M5 Pro performance pass

Measured on Apple M5 Pro (GPU family 10, 20 cores), model Qwen3.8-27B-RichEngine,
512-token prompt, fused single-command-buffer decode cycles.

### Decode throughput

Two policy changes landed from this pass so far
(`runtime/ops/Linear.cpp`): `kPaired256TilesPerCore` 8→3 for one-lane plain
projections, and a full resident N256 grid for one- and two-lane gate/up
projections up to four tiles per core.

decode-profile, 6 cycles per width, median fused GPU ms/cycle:

| Batch | Before (ms) | After (ms) | Tok/s before | Tok/s after | Gain |
|---|---|---|---|---|---|
| B1 | 83.36 | 75.24 | 85.7 | 96.4 | +12.5% |
| B2 | 101.6 | 88.43 | 151.7 | 169.9 | +12.0% |
| B3 | 136.4 | 128.31 | 172.2 | 174.0 | +1.0% |
| B4 | 147.4 | 139.37 | 211.4 | 214.1 | +1.3% |

Tok/s after measured by `backend-benchmark --scenario decode`
(wall-clock, draft acceptance 0.873 unchanged); ms/cycle from
`decode-profile`, 6 cycles per width.

The two applied changes alone match or beat the previously recorded
full-pass numbers at B1–B3. B4 remains ~1 ms short; the lane-4 N128
resident-wave rule targets it.

Draft acceptance unchanged at ~0.87 across all batch widths.

### Changes applied

| Change | File | Effect |
|---|---|---|
| One-lane plain projections switch to Paired256 at 3 tiles/core (was 8) | `runtime/ops/Linear.cpp` | `{16640,5120}` B1 ~+17% operator-level; B1 cycle 83.36 → 80.34 |
| Gate/up lanes ≤2 use full N256 grid up to 4 tiles/core | `runtime/ops/Linear.cpp` | `decode_linear_q4_n256_gate_up` 33.4 → 28.0 ms at B1; B1 cycle → 74.84 |
| kv4/g6 chain verify splits the 16-row tile's two fused-row passes across two threadgroups (`_r2`) | `runtime/ops/PagedAttention.cpp`, `runtime/metal/kernels/decode/paged_attention.metal` | attention-sweep verify −17% to −33% fused at every history and lane count; 131k/lane4 20.46 → 16.69 ms ([Qwen 3.8 27B Findings.md](dev/benchmarks/Qwen%203.8%2027B%20Findings.md)) |
| Dense-FFN gate/up emits its output as the down projection's Table64 plus paired half sums (`decode_linear_q4_sg_gate_up_t64` / `_residual_psum`) | `runtime/metal/kernels/decode/linear_q4_sgmatrix.metal`, `runtime/ops/Linear.cpp`, `runtime/model/TargetModel.cpp` | removes the `decode_linear_q4_prepare` pass + intermediate bf16 round trip per dense-FFN down projection |

### Changes identified, not yet applied

| Change | File | Expected |
|---|---|---|
| Lane-4 plain paths use N128 until 4 tiles/core; resident wave (4 groups/core) to 12 tiles/core | `runtime/ops/Linear.cpp` | `{16640}`/`{14336}` B4 +12% operator-level |
| Lane-2 plain paths use one resident wave in the 6–12 tiles/core band | `runtime/ops/Linear.cpp` | `{16640}` B2 +9% operator-level |
| Skip ~170 KB/lane constraint-mask arena fill for unconstrained lanes | `runtime/model/Runtime.mm` | removes per-step host fill; only `TokenMask` lanes and abandoned constrained lanes still fill |
| Staged scale/bias epilogue (batched m16/m24/m32 decode tiles) | `runtime/metal/kernels/common/q4_mpp_tiles.h` | −4.5% B2, −2.5% B3; code from the original pass was not preserved, needs reimplementation |
| Policy anchors and mirror updated for new family-10 rules | `dev/tests/engine/linear_plan_test.mm` | needed once the remaining policy rules land |

### Measured and rejected (from the original pass)

| Idea | Result |
|---|---|
| Split32/Split64 decode kernels as defaults | +10–54% operator-level, but documented speculative-acceptance regressions on Apple10 — kept as tuning candidates only |
| Four-simdgroup variants of m16/m32 decode tiles | −17% to −120% on every production shape; kernels and policy reverted |
| Paired256 at small one-lane widths | −49% to −110%; threshold tuned instead |
| Pipelined m32 tiles (two quant-group partials in flight) | −5% to −8×; doubled cooperative-tensor partials spill registers — reverted |
| Register-cached per-column scale/bias in m32 epilogue | +21% to +43% slower; runtime column values prevent compile-time folding — reverted |
| N64 MoE expert tiles | −24% to −12×; 64-column tiles starve the MPP unit — reverted |
| `-mcpu=apple-m5`, LTO, `-funroll-loops`, C++23 | all within ±3% noise; engine is GPU-bound ~99% of the cycle |

### Remaining opportunities

| Area | Evidence |
|---|---|
| Long-context verify attention | `_r2` recovered the M=96 residency loss (−17% to −33%); the tile still streams below the ~285 GB/s peak, tree verify keeps the single-group passes |
| m16/m24/m32 decode tiles | 100–170 GB/s vs 220–285 GB/s for m8; epilogue vectorization is the next lever |
| `norm_rms` dispatch batching | 141 dispatches ≈ 1.5 ms/cycle (~1%) |
| Constrained decode host round-trip | per-step ~170 KB mask fill + extra submissions; grammar workloads only |
| MoE model tuning | Qwen3.6-35B-A3B shapes need a dedicated tune-kernels run |

### Validation

| Check | Result |
|---|---|
| `decode-profile` B1–B4, 512 prompt, 6 cycles | B1 75.24, B2 88.43, B3 128.31, B4 139.37 ms — no regressions vs recorded baselines |

Note: `decode-profile` caps each lane at 256 new tokens, so runs above
~6 cycles per width abort early ("completed request was decoded").

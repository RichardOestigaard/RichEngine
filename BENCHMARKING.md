# Benchmarking

How the published numbers are produced and how to reproduce them.

## The native backend benchmark

Build the runtime and the benchmark, then run it against an installed
package:

```sh
make -j4
make build/engine-tests/backend-benchmark
./build/engine-tests/backend-benchmark build/richengine.metallib \
    "install/models/incoai/Qwen3.8-27B-RichEngine" \
    --scenario decode,partial,short --samples 3
```

`MODEL_ROOT` is the package directory (`install/models/<owner>/<name>` or
the installer's selection link). Scenarios:

- `decode`: a shared 39-token prompt decoding 64 tokens per lane at widths
  B1–B4; reports aggregate wall tokens/s and draft acceptance.
- `partial`: a 14,096-token request cold, seeded (10K prefix) and with a
  10K cache hit; reports TTFT.
- `short`: cold prefills of 481–2,017 tokens.
- `context` / `exact`: long-context and cache-exactness checks.

Options: `--samples N` (default 1), `--progress FILE` (JSONL progress),
`--max-context TOKENS` (as serve's), `--kv-format FORMAT` and
`--ane-ffn-share SHARE` / `--ane-ffn-minimum-rows ROWS`.

## Fixed conventions

- **`--kv-format` defaults to `int8` in the benchmark**, not serve's
  `int4`. The published decode numbers run INT8: INT4's quantization noise
  flips near-tie argmaxes and measurably lowers draft acceptance. Always
  compare against numbers taken at the same format.
- Draft acceptance depends on the prompt. The benchmark's prompt is
  synthetic; do not mix acceptance or decode figures across benchmark
  versions or prompt sets.
- The Neural Engine split calibrates once per Mac and model; the first run
  on a machine takes seconds to tens of seconds longer. Give
  `--ane-ffn-share 0` for the GPU-alone baseline; omit it for the
  calibrated default. Decode rows run on the GPU either way.
- `--kv-format bf16` currently fails warmup on memory-tight machines
  ("warmup KV page 0 is outside the startup runway") even with a reduced
  `--max-context`; use `int8` for the reference-precision comparison.

## Results

Raw results land in `build/release/bench/*.json` (or `--output-dir` under
`make test-performance-real`, which also supports `BASELINE=<checkout>`
for ABBA-order comparisons against another build and
`ANE_FFN_SHARE=<share>` to pin the split).

## DiffusionGemma microbenchmarks

The `decode` scenario does not apply to the diffusion model: a denoise
step produces a 256-token canvas, not one retained token, and the
committed canvas is what the wall number divides. Until a canvas-aware
scenario exists, the kernels are benchmarked directly:

```sh
make build/engine-tests/canvas-kernels
./build/engine-tests/canvas-kernels build/richengine.metallib
make build/engine-tests/hd512-attention
./build/engine-tests/hd512-attention build/richengine.metallib
```

`canvas-kernels` prints µs/dispatch (median of 20 GPU-timestamped
reps) for the logit-tail kernels over a real-size 256×262144 fp32
logits buffer — the old and fused paths side by side — and checks
argmax/entropy/sample parity against a CPU reference on three
adversarial row bands. `hd512-attention` does the same old-vs-m2
comparison for the head_dim-512 attention splits at prefixes
{0, 1024, 8192} in each KV format.

Conventions for quoting diffusion numbers:

- Cite the flags: the fused paths default on — `RICHENGINE_CANVAS_EMBED_HIST`,
  `RICHENGINE_CANVAS_STATS_FUSED`, `RICHENGINE_CANVAS_STEPS_PER_CMD` (2),
  `RICHENGINE_CANVAS_PREFIX_EXIT`, `RICHENGINE_CANVAS_COMMIT_TAIL`.
  Set any of them to `0`/`1` for the pre-optimization baselines.
- Committed throughput is `committed_tokens / wall`, not tokens/forward;
  report both canvas steps used and tokens committed per canvas.
- `canvas_soft_embed_topk` is the reference implementation kept for
  A/B only (`RICHENGINE_CANVAS_EMBED_HIST=0` selects it; `exact=true`
  selects the full-table exact kernel — never quote it for production).

# Performance

[Back to RichEngine](../README.md#performance)

## RichEngine benchmarks

Measured on an M5 Pro (20-core GPU, 48 GB), serving the Qwen3.8-27B and
Qwen3.6-35B-A3B RichEngine packages, with the native backend benchmark
(`make test-performance-real`): decode at widths B1–B4 on a 39-token prompt
producing 64 tokens per lane, and a 14,096-token partial-prefix request,
three samples each
(`build/release/incoai--<repo>-RichEngine/backend-benchmark.json`). The MLX 4-bit
models prepare to the packages' target weights, byte for byte but for the
27B's 48 per-layer GDN decay vectors, each within a float ULP, and decode
within 0.5% of them on this M5 Pro
([upstream loading](../dev/benchmarks/upstream-loading.md)). GGUF targets run
other kernels; [GGUF against llama.cpp](#gguf-against-llamacpp) compares them.

| Metric | Qwen3.6-35B-A3B | Ornith-1.5-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: | ---: |
| Decode · short prompt (B1) | 260 tok/s | 161 tok/s | 103 tok/s |
| Decode · aggregate, 2 lanes (B2) | 458 tok/s | 279 tok/s | 177 tok/s |
| Decode · aggregate, 3 lanes (B3) | 594 tok/s | 358 tok/s | 191 tok/s |
| Decode · aggregate, 4 lanes (B4) | 714 tok/s | 433 tok/s | 246 tok/s |
| Prefill · 14,096-token prompt, cold | 2,937 tok/s | 2,960 tok/s | 600 tok/s |
| Time to first token · 14K prompt, 10K cached | 1.6 s | 1.6 s | 7.4 s |

Draft acceptance on the short-prompt decode: 69% on the 35B, 40% on the
Ornith 35B-A3B, 43% on the 27B. Decode benchmarks run `--kv-format int8`
(INT4 measurably lowers acceptance). The 27B's prefill rows run its
calibrated Neural Engine split (share 0.41); with `--disable-ane` it
prefills at 495 tok/s and reaches the first token in 8.7 s. The Ornith
35B-A3B is MoE — its experts do not fit the dense split, so its prefill
runs on the GPU alone.

The 27B's draft acceptance is down from 87% in earlier measurements of the
same benchmark — the KV format does not explain it (INT4 reads 51% here),
so the drop looks like a draft-path regression under investigation.

The Neural Engine prefill split (`--disable-ane` keeps the FFN on the GPU)
runs part of each dense layer's FFN on the Apple Neural Engine during
prefill while the GPU runs the rest, calibrated once per Mac and model. On
the same M5 Pro with the Qwen3.8-27B package (split at share 0.41 of 34
units, chunks of 512 rows or more, 27.5 ms per 2048-row FFN layer against
40.0 ms on the GPU alone):

| Metric | GPU alone | With the split |
| --- | ---: | ---: |
| Prefill · 14,096-token prompt, cold | 28.4 s | 22.5 s (1.26×) |
| Prefill · 10,000-token prompt, cold | 19.7 s | 15.5 s (1.27×) |
| Time to first token · 14K prompt, 10K cached | 8.7 s | 7.0 s (1.24×) |

The split serves every dense target whose hidden size packs into whole
512-channel blocks — Ornith 1.5 9B's 4096 channels run two 2048-channel
segments (the 27B's 5120 run two of 2560). On the same Mac with the Ornith
1.5 9B assembly (split at share 0.38 of 24 units, chunks of 512 rows or
more, 14.7 ms per 2048-row FFN layer against 21.2 ms on the GPU alone,
medians of three samples):

| Metric | GPU alone | With the split |
| --- | ---: | ---: |
| Prefill · 14,096-token prompt, cold | 8.3 s | 6.7 s (1.24×) |
| Prefill · 10,000-token prompt, cold | 5.9 s | 5.5 s (1.07×) |
| Time to first token · 14K prompt, 10K cached | 2.7 s | 2.4 s (1.09×) |

Decode is unchanged: its kernels do not run on the Neural Engine, and the
split does not engage while other requests decode.

### Gemma 4 / DiffusionGemma

Both Gemma 4 targets are 26B-A4B MoE (25.2B total, 3.8B active, packed Q4
~14.5 GB) and install from safetensors into the packed
`richengine-packed-q4-*` formats. `Gemma4-26B-A4B` decodes
autoregressively with the z-lab DFlash block-diffusion draft (Plain
draft, 8-row blocks from a block-16 training); `DiffusionGemma-26B-A4B`
generates a 256-token canvas per block through bidirectional denoising
(≤48 steps, entropy-bound acceptance, ~13–17 steps typical upstream).

No end-to-end model benchmark has been run yet — the numbers below are
GPU-timestamped kernel microbenchmarks on this Mac
(`make test-canvas-kernels`, `make test-hd512`, medians over 20 reps)
plus the bandwidth roofline in
[GEMMA_DIFFUSION_OPTIMIZATION_PLAN.md](GEMMA_DIFFUSION_OPTIMIZATION_PLAN.md).

Measured kernel results (kept on by default):

| Kernel | Before | After |
| --- | ---: | ---: |
| `canvas_soft_embed_topk` → histogram | ~91–155 ms | ~7–16 ms |
| `canvas_row_stats` → fused single pass | 2.7 ms | 1.4 ms |
| softcap + temperature elementwise passes | ~4.0 ms/step | 0 (fused on load) |
| hd512 attention → M-split | baseline | 18–45% faster (q8/int4/bf16) |

Step-driver flags (all default on unless noted):
`RICHENGINE_CANVAS_STEPS_PER_CMD` (2), `RICHENGINE_CANVAS_PREFIX_EXIT`,
`RICHENGINE_CANVAS_COMMIT_TAIL`; `RICHENGINE_CANVAS_SPECULATIVE_PREFILL`
is off pending a real-model measurement.

Projected committed-throughput roofline after fixes (~16.3 GB/step,
13–17 steps/canvas + one re-prefill): ~130–175 tok/s M5, ~260–350
M5 Pro, ~380–520 M5 Max 32-core, ~510–690 M5 Max 40-core. Upstream
(vLLM on H100/H200 FP8) reports ~1,000–1,300 tok/s.

KV sizing for the Gemma 4 dual geometry (25 sliding + 5 global layers,
K-only globals): ~111.6 KB/token `int8`/`fp8e4m3`, ~56.6 KB `int4`,
~220 KB `bf16` — about 29 MB per committed 256-token canvas at `int8`.
Quantized KV extents on this layout round to 256 pages (~0.5–0.9 GB),
so small canvas scratch allocations ride the request's own extent.

Serving floor: 32 GB Macs (int4 KV, ~110–140K context), 48 GB
comfortable, 64 GB+ for the full 262K context at `int8`. The ANE
prefill split does not apply — the shared expert fails the shape gates
and the canvas step is DRAM-bound regardless.

### 2B-class models

The same benchmark on the same Mac, on the GGUF installs
(`openbmb/MiniCPM5-2B-GGUF:Q4_K_M`, `LiquidAI/LFM2.5-2.6B-GGUF:MXFP4`). Both
run DSpark drafts; their acceptance on this prompt is low (9–11%), so the
decode numbers are mostly target-model speed.

| Metric | LFM2.5-2.6B (`MXFP4`) | MiniCPM5-2B (`Q4_K_M`) |
| --- | ---: | ---: |
| Decode · short prompt (B1) | 139 tok/s | 141 tok/s |
| Decode · aggregate, 2 lanes (B2) | 248 tok/s | 265 tok/s |
| Decode · aggregate, 3 lanes (B3) | 239 tok/s | 295 tok/s |
| Decode · aggregate, 4 lanes (B4) | 319 tok/s | 390 tok/s |
| Prefill · 14,096-token prompt, cold | 4,024 tok/s | 3,463 tok/s |
| Time to first token · 14K prompt, 10K cached | 1.1 s | 1.6 s |

The LFM2.5 run's B3 aggregate (239 tok/s) sits just below its B2 (248
tok/s) — the model saturates by width 2, so the benchmark's monotonicity
gate reports a performance failure even though every check's numbers are
valid. The `Q4_K_M` LFM2.5 install does not complete the batched decode
scenario on this build.

### Granite 4.2

The same benchmark on the `Q4_K_M` GGUF installs. Granite ships no draft
model, so decode verifies n-gram proposals instead of a neural draft's;
acceptance on this prompt is 39% on the 3B and 27% on the 8B.

| Metric | Granite-4.2-3B | Granite-4.2-8B |
| --- | ---: | ---: |
| Decode · short prompt (B1) | 303 tok/s | 116 tok/s |
| Decode · aggregate, 2 lanes (B2) | 565 tok/s | 230 tok/s |
| Decode · aggregate, 3 lanes (B3) | 662 tok/s | 274 tok/s |
| Decode · aggregate, 4 lanes (B4) | 883 tok/s | 363 tok/s |
| Prefill · 14,096-token prompt, cold | 2,117 tok/s | 1,226 tok/s |
| Time to first token · 14K prompt, 10K cached | 2.6 s | 4.1 s |

The Neural Engine split does not apply to either Granite: the 3B's hidden
size overflows the rotation's block count and the 8B's is no whole
2,560-channel segment, so both prefill on the GPU alone.

For repeatable measurements on your Mac, see [local benchmarks](../DEVELOPMENT.md#local-benchmarks).

## GGUF against llama.cpp

We compared RichEngine with llama.cpp (e6ab7c1, Metal) on the same Unsloth
UD-Q4_K_M files. For accuracy, both read the same text, 16,384 positions of
prose, code and chat, and at each position we compared the tokens they rank
first:

| Same token ranked first | Qwen3.8-27B | Qwen3.6-35B-A3B |
| --- | ---: | ---: |
| RichEngine and llama.cpp | 99.30–99.45% | 97.83–98.14% |
| llama.cpp on the CPU and on Metal | 97.8% | 96.5–96.9% |
| llama.cpp one token at a time and batched | 99.65–99.75% | 97.95% |

The positions where they differ are near-ties: there, llama.cpp's two best
tokens are a median 0.03–0.10 nats apart, against 2.6–2.7 nats over all
positions. RichEngine's perplexity is 0.1–0.4% (27B) and 0.1–0.9% (35B) above
llama.cpp's; llama.cpp's CPU backend is 1.7–1.8% above its Metal on the 27B.
RichEngine's figures cover an M5 Pro and an M3 Max with `--kv-format bf16` and
`--disable-ane`; the INT8 cache gives 99.23–99.25% and 97.92–97.94% on the
M5 Pro (the default is INT4).

Speed uses the selected SPEED-Bench coding prompts described above, with
greedy sampling. llama-server runs with its default settings, which do not
speculate, and for the 27B also
with the MTP draft Unsloth ships:

| Decode tok/s | M5 Pro, 20-core GPU | M3 Max, 40-core GPU |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B · RichEngine | 175 | 209 |
| Qwen3.6-35B-A3B · llama.cpp | 69 | 66 |
| Qwen3.8-27B · RichEngine | 74 | 92 |
| Qwen3.8-27B · llama.cpp | 16 | 17 |
| Qwen3.8-27B · llama.cpp with MTP | 27 | 20 |

RichEngine decodes 2.5–3.2× as fast as llama.cpp on the 35B and 4.5–5.3× on
the 27B (2.7–4.6× against its MTP). Prefilling a 2,048-token chunk, it runs
at 559 tok/s against 374 on the 27B and 3,662 against 1,968 on the 35B on
the M5 Pro, and at 245 against 193 and 1,814 against 1,575 on the M3 Max.

## Smaller GGUFs on 24 GB Macs

Measured on a 24 GB M6 (12-core GPU), with each model's DFlash2 draft:

| Unsloth GGUF | Code decode | Advertised context limit |
| --- | ---: | ---: |
| `Qwen3.8-27B-GGUF:UD-IQ3_XXS` | 43.5 tok/s | 73,721 tokens (102,393 with `--language-only`) |
| `Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL` | 145 tok/s | 256K tokens |

The context column reports capacity, not the prompt length of the decode
measurement. Coding agents need about 100K tokens (Claude Code's own prompt is
about 33K, and at 64K it compacted repeatedly and stopped), so serve the 27B
with `--language-only` for them.

RichEngine grows its caches only while macOS has 3 GiB free and gives cached memory
back when it runs short, but a request in service still takes the memory it
needs within `--max-memory`, so these models leave other applications little
room. A request that cannot get memory waits for it, then fails with
`resource_timeout`: close memory-heavy applications, or serve with
`--language-only`. Critical memory pressure can suspend a long request. Startup
suggests `--max-cache-disk` when memory may not hold the advertised context.
The tier then writes whenever memory runs short: serving Ternary-Bonsai-2-27B
PQ2_0 to six clients' mixed traffic for 30 minutes, a 16 GiB tier on the M6
wrote 26 GB, about 50 GB an hour, and read 36 GB.

See [the low-bit GGUF measurements](https://github.com/incoai/richengine/pull/160)
for the workloads, memory pressure, SSD settings and limitations, and
[the Bonsai measurements](https://github.com/incoai/richengine/pull/166) for PQ2_0.

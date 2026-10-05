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

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt (B1) | 260 tok/s | 103 tok/s |
| Decode · aggregate, 2 lanes (B2) | 458 tok/s | 177 tok/s |
| Decode · aggregate, 3 lanes (B3) | 594 tok/s | 191 tok/s |
| Decode · aggregate, 4 lanes (B4) | 703 tok/s | 246 tok/s |
| Prefill · 14,096-token prompt, cold | 2,876 tok/s | 466 tok/s |
| Time to first token · 14K prompt, 10K cached | 1.6 s | 9.4 s |

Draft acceptance on the short-prompt decode: 69% on the 35B, 87% on the
27B.

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
RichEngine's figures cover an M5 Pro and an M3 Max with `--kv-format bf16`; the
INT8 cache gives 99.23–99.25% and 97.92–97.94% on the M5 Pro (the default is
INT4).

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

# Splash

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash runs coding agents and OpenAI or Anthropic compatible applications on
one Mac. It combines [DFlash 2](https://inco.ai/blog/dflash2/) speculative
decoding, specialized Metal kernels, and automatic memory planning, with
vision, tool calling, and a built-in chat page.
It reuses cached prefixes and batches concurrent requests automatically.

## Quick start

Apple M3 or newer, macOS 27 or later, and [Homebrew](https://brew.sh).
The 4-bit examples need at least 36 GB of unified memory (48 GB recommended);
24 GB Macs can use [smaller GGUF variants](#models).

```bash
brew install incoai/tap/splash
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

The first run downloads the model and its matching draft and starts serving
on `127.0.0.1:8000`. Later starts reuse the downloads; leave room on disk for
them ([storage requirements](DEVELOPMENT.md#model-storage)).

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes / splash pi
```

Press Ctrl+C in the server terminal to stop Splash.
For LM Studio Bionic, follow its [Splash setup guide](https://lmstudio.ai/blog/splash-engine).

## Use the API

OpenAI Chat Completions, Responses and Completions, and Anthropic Messages,
with streaming, tool calls, JSON Schema output, images, and inline PDFs:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

Reasoning follows the model default; `"reasoning_effort": "none"` turns it off.
[Reasoning settings](DEVELOPMENT.md#default-reasoning-effort) ·
[API details](DEVELOPMENT.md#code-and-api-boundaries)

## Models

Splash supports these model families, with a matching DFlash2 draft selected
automatically:

| Model | GGUF example | MLX 4-bit |
| --- | --- | --- |
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.8-27B-4bit` |
| Qwen3.6-35B-A3B | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.6-35B-A3B-4bit` |

Unsloth GGUF variants span **1–8 bits**, including mixed-precision UD formats;
`UD-Q8_K_XL` and BF16 targets are not supported.
[Prism ML Ternary Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
is also supported in PQ2_0 (7.2 GB), including vision. Pass `OWNER/REPO:VARIANT`
to `--model`, as in the quick start. Smaller variants run on
[24 GB Macs](docs/performance.md#smaller-ggufs-on-24-gb-macs).
[27B variants](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) ·
[35B variants](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main)

Vision and the tokenizer come from the target model's source.
[Model loading and compatibility](DEVELOPMENT.md#upstream-model-loading) ·
[Supported formats](DEVELOPMENT.md#gguf-targets)

## Settings

Memory and context are sized automatically, up to the model's native context
window. To set your own limits or cache options, add these to `splash serve`:

| Option | Purpose |
| --- | --- |
| `--max-memory 28G` | Cap Metal memory use. |
| `--idle-release off` | Keep the model in memory while idle. Default: freed after 10 minutes without a request; a duration such as `2h` sets another time. |
| `--max-context 100K` | Set the context limit. |
| `--language-only` | Skip vision; serve text only. |
| `--kv-format int8` | KV cache format: `int4` (default), `int8`, `bf16`, `fp8e4m3`. |
| `--max-cache-disk 16G` | Offload KV cache and GDN states to SSD as needed. Off by default. |
| `--persistent-cache` | Keep the SSD cache across restarts. Off by default. |

Use `--max-memory` to leave room for other applications.
The server listens on localhost without authentication by default. For LAN
access, authentication, browser apps on other origins, and other options, see
[server configuration](DEVELOPMENT.md#server-configuration) or
`splash serve --help`.
[KV precision](DEVELOPMENT.md#kv-cache-precision) ·
[SSD cache](DEVELOPMENT.md#disk-cache)

### KV cache formats

The target KV cache stores keys and values per 32-token page, with one fp32
scale per token and KV head under quantization. Four formats:

| Format | Bytes per element | Notes |
| --- | ---: | --- |
| `int4` (default) | 0.5 | Symmetric 4-bit, native packed INT4 matrix ops. Halves KV traffic. |
| `int8` | 1 | Symmetric 8-bit. The original format. |
| `bf16` | 2 | Unquantized. Reference quality, twice the memory. |
| `fp8e4m3` | 1 | E4M3 with fp16 query staging. Experimental; lowest precision tier. |

For the 27B (16 attention layers, 4 KV heads, 256 dimensions) that is ~33 KB
per token under `int8`/`fp8e4m3`, ~17 KB under `int4`, ~66 KB under `bf16` —
so INT4 reads roughly half the KV bytes per decode step and doubles the KV
capacity the memory budget admits.

The catch is speculative decode. Verify computes target logits over the
quantized KV; quantization noise flips near-tie argmaxes and rejects draft
tokens that would otherwise be accepted. Measured on an M5 Pro (20-core GPU,
temperature 0, ~67K context):

| Format | Decode | Draft acceptance |
| --- | ---: | ---: |
| `int8` | 62 tok/s | 63% |
| `int4` | 56 tok/s | 55% |

The bandwidth saving (~6% at 67K) does not cover the acceptance loss, and
below ~30K the saving is negligible while the acceptance tax remains. INT4
pays off only where KV traffic dominates — extreme histories toward the
context limit — or where the extra capacity matters. For interactive and
agentic coding workloads `int8` is measurably faster today.

## Performance

Measured on an M5 Pro (16-core GPU, 48 GB), using the Splash
packages and selected SPEED-Bench coding prompts over HTTP. Ratios compare
with the next-fastest engine measured in that benchmark.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

[Launch benchmarks](https://inco.ai/blog/splash/) ·
[Measurement details](docs/performance.md#splash-10-launch-benchmarks) ·
[Run benchmarks locally](DEVELOPMENT.md#local-benchmarks)

### GGUF against llama.cpp

Same Unsloth UD-Q4_K_M weights on Metal. Decode speed in tok/s:

| Model | Engine | M5 Pro | M3 Max |
| --- | --- | ---: | ---: |
| 27B | llama.cpp | 16 | 17 |
| | llama.cpp with MTP | 27 | 20 |
| | **Splash** | **74** | **92** |
| 35B-A3B | llama.cpp | 69 | 66 |
| | **Splash** | **175** | **209** |

That is **2.5–3.2×** as fast on the 35B and **4.5–5.3×** on the 27B
(**2.7–4.6×** against MTP).

**Closely matches llama.cpp's predictions.**

| Next-token agreement ↑ | 27B | 35B-A3B |
| --- | ---: | ---: |
| llama.cpp: CPU vs. GPU | 97.8% | 96.5–96.9% |
| llama.cpp: single-token vs. batched | 99.65–99.75% | 97.95% |
| **Splash vs. llama.cpp** | **99.30–99.45%** | **97.83–98.14%** |

Splash uses BF16 KV in this comparison.
[Benchmark details](docs/performance.md#gguf-against-llamacpp)

## Design

Each supported model pairs a trained DFlash2 draft with Metal kernels for its
shapes. The runtime, scheduler, cache, and API are shared. Weights are
converted to the kernels' layouts as they load, with no copy on disk; kernels
ship precompiled, with no Xcode or local tuning required.
[How Splash works](https://inco.ai/blog/splash/)

## More

- [Development](DEVELOPMENT.md): build from source, architecture, tests, and releases.
- [Issues and feedback](https://github.com/incoai/splash/issues)
- [Apache-2.0](LICENSE). GGUF kernels include MIT-licensed material from
  llama.cpp; see [third-party notices](THIRD_PARTY_NOTICES). Model weights keep their own licenses.

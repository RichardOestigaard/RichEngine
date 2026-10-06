# RichEngine

[![CI](https://github.com/incoai/richengine/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/richengine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

RichEngine runs coding agents and OpenAI or Anthropic compatible applications on
one Mac. It combines [DFlash 2](https://inco.ai/blog/dflash2/) speculative
decoding, specialized Metal kernels, and automatic memory planning, with
vision, tool calling, and a built-in chat page.
It reuses cached prefixes and batches concurrent requests automatically.

## Quick start

Apple M3 or newer, macOS 27 or later, and [Homebrew](https://brew.sh).
The 4-bit examples need at least 36 GB of unified memory (48 GB recommended);
24 GB Macs can use [smaller GGUF variants](#models).

```bash
brew install incoai/tap/richengine
richengine serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

The first run downloads the model and its matching draft and starts serving
on `127.0.0.1:8000`. Later starts reuse the downloads; leave room on disk for
them ([storage requirements](DEVELOPMENT.md#model-storage)).

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
richengine opencode    # or: richengine claude / richengine codex / richengine hermes / richengine pi
```

Press Ctrl+C in the server terminal to stop RichEngine.
For LM Studio Bionic, follow its [RichEngine setup guide](https://lmstudio.ai/blog/richengine-engine).

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
[Tool calls](DEVELOPMENT.md#tool-calls) ·
[API details](DEVELOPMENT.md#code-and-api-boundaries)

## Models

RichEngine supports these model families, with a matching draft selected
automatically (Granite has none and speculates with n-grams instead):

| Model | GGUF example | MLX 4-bit |
| --- | --- | --- |
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.8-27B-4bit` |
| Qwen3.6-35B-A3B | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| MiniCPM5-2B | `openbmb/MiniCPM5-2B-GGUF:Q4_K_M` | — |
| LFM2.5-2.6B | `LiquidAI/LFM2.5-2.6B-GGUF:MXFP4` | — |
| Granite-4.2-3B | `ibm-granite/granite-4.2-3b-GGUF:Q4_K_M` | — |
| Granite-4.2-8B | `ibm-granite/granite-4.2-8b-GGUF:Q4_K_M` | — |

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
window. To set your own limits or cache options, add these to `richengine serve`:

| Option | Purpose |
| --- | --- |
| `--max-memory 28G` | Cap Metal memory use. |
| `--idle-release off` | Keep the model in memory while idle. Default: freed after 10 minutes without a request; a duration such as `2h` sets another time. |
| `--max-context 100K` | Set the context limit. |
| `--language-only` | Skip vision; serve text only. |
| `--kv-format int8` | KV cache format: `int4` (default), `int8`, `bf16`, `fp8e4m3`. |
| `--disable-ane` | Prefill on the GPU alone. Default: a dense model's long prompts also use the Neural Engine when that is faster. |
| `--max-cache-disk 16G` | Offload KV cache and GDN states to SSD as needed. Off by default. |
| `--persistent-cache` | Keep the SSD cache across restarts. Off by default. |

Use `--max-memory` to leave room for other applications.
The server listens on localhost without authentication by default. For LAN
access, authentication, browser apps on other origins, and other options, see
[server configuration](DEVELOPMENT.md#server-configuration) or
`richengine serve --help`.
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

Sizing for the 27B at its 131K-token context, per resident request:

| Format | KV bytes | Same-context capacity vs `bf16` |
| --- | ---: | ---: |
| `int4` | ~2.2 GB | 4× |
| `int8` | ~4.3 GB | 2× |
| `bf16` | ~8.7 GB | 1× |

Quantization also halves or quarters what `--max-cache-disk` stages to SSD:
`int4` gives a 16 GiB tier roughly four times the token reach of `bf16`.
Quality-wise, `int8` costs about 0.1–0.2 points of next-token agreement with
`bf16` on the llama.cpp comparison (99.2% vs 99.3–99.4%); `fp8e4m3` is
experimental and does not yet pair with tree verify.

Rule of thumb: leave `int4` for long-context or concurrent workloads where
capacity is the constraint; pass `--kv-format int8` for short interactive
traffic where acceptance rate dominates latency; use `bf16` only when
measuring against a reference.

## Performance

Measured on an M5 Pro (20-core GPU, 48 GB) with the RichEngine packages,
using the native backend benchmark (`make test-performance-real`): a
39-token prompt decoding 64 tokens per lane, and a 14,096-token
partial-prefix request.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 259 tok/s | 103 tok/s |
| Prefill · 14K prompt | 2,937 tok/s | 600 tok/s |
| Time to first token · 14K prompt, 10K cached | 1.7 s | 7.4 s |
| Aggregate decode · 4 concurrent short prompts | 714 tok/s | 246 tok/s |

Decode numbers run `--kv-format int8`; the benchmark's synthetic prompt
reads 69% draft acceptance on the 35B and 43% on the 27B — the 27B's drop
from an earlier 87% is under investigation (see
[BENCHMARKING.md](BENCHMARKING.md)).

A dense model's prefill also runs its FFN on the Apple Neural Engine when
calibration finds that faster — on this Mac the 27B's split at share 0.41
cuts a cold 14K prefill from 28.4 s to 22.5 s (1.26×), decode unchanged.
`--disable-ane` keeps prefill on the GPU alone.

The same benchmark on the 2B-class GGUF installs on this Mac:

| Metric | LFM2.5-2.6B (`MXFP4`) | MiniCPM5-2B (`Q4_K_M`) |
| --- | ---: | ---: |
| Decode · short prompt | 139 tok/s | 141 tok/s |
| Prefill · 14K prompt | 4,024 tok/s | 3,463 tok/s |
| Time to first token · 14K prompt, 10K cached | 1.1 s | 1.6 s |
| Aggregate decode · 4 concurrent short prompts | 319 tok/s | 390 tok/s |

The Granite 4.2 GGUF installs, measured the same way. Granite ships no
draft model, so decode verifies n-gram proposals instead of a neural
draft's; acceptance on this prompt is 39% on the 3B and 27% on the 8B.
Neither hidden size fits the Neural Engine split, so their prefills run on
the GPU alone:

| Metric | Granite-4.2-3B | Granite-4.2-8B |
| --- | ---: | ---: |
| Decode · short prompt | 303 tok/s | 116 tok/s |
| Prefill · 14K prompt | 2,117 tok/s | 1,226 tok/s |
| Time to first token · 14K prompt, 10K cached | 2.6 s | 4.1 s |
| Aggregate decode · 4 concurrent short prompts | 883 tok/s | 363 tok/s |

Agent-style follow-ups that share a chat template's leading system prompt and
tools resume from a shared-prefix junction: 2.5 s to first token against
4–8.6 s without it (identical prompts 0.2 s, cold 8.1 s, unchanged). Greedy
verification argmaxes in the vocabulary head's kernel — no logits round-trip
(`RICHENGINE_HEAD_FUSED_OFF` disables) — and `RICHENGINE_PREFILL_FAST_INT8`
selects a two-term uint8 prefill path.

[Measurement details](docs/performance.md#richengine-benchmarks) ·
[Run benchmarks locally](DEVELOPMENT.md#local-benchmarks)

### GGUF against llama.cpp

Same Unsloth UD-Q4_K_M weights on Metal. Decode speed in tok/s:

| Model | Engine | M5 Pro | M3 Max |
| --- | --- | ---: | ---: |
| 27B | llama.cpp | 16 | 17 |
| | llama.cpp with MTP | 27 | 20 |
| | **RichEngine** | **74** | **92** |
| 35B-A3B | llama.cpp | 69 | 66 |
| | **RichEngine** | **175** | **209** |

That is **2.5–3.2×** as fast on the 35B and **4.5–5.3×** on the 27B
(**2.7–4.6×** against MTP).

**Closely matches llama.cpp's predictions.**

| Next-token agreement ↑ | 27B | 35B-A3B |
| --- | ---: | ---: |
| llama.cpp: CPU vs. GPU | 97.8% | 96.5–96.9% |
| llama.cpp: single-token vs. batched | 99.65–99.75% | 97.95% |
| **RichEngine vs. llama.cpp** | **99.30–99.45%** | **97.83–98.14%** |

RichEngine uses BF16 KV and `--disable-ane` in this comparison.
[Benchmark details](docs/performance.md#gguf-against-llamacpp)

## Design

Each supported model pairs a trained DFlash2 draft with Metal kernels for its
shapes. The runtime, scheduler, cache, and API are shared. Weights are
converted to the kernels' layouts as they load, with no copy on disk; kernels
ship precompiled, with no Xcode or local tuning required.

## More

- [Development](DEVELOPMENT.md): build from source, architecture, tests, and releases.
- [Issues and feedback](https://github.com/incoai/richengine/issues)
- [Apache-2.0](LICENSE). RichEngine descends from
  [Splash](https://github.com/incoai/splash) — see [NOTICE](NOTICE).
  GGUF kernels include MIT-licensed material from llama.cpp; see
  [third-party notices](THIRD_PARTY_NOTICES). Model weights keep their own
  licenses.

## Credits

- **[Splash](https://github.com/incoai/splash)** — the project RichEngine
  derives from: the engine, Metal kernels, scheduler, and server this
  codebase builds on, published under Apache-2.0.

RichEngine implements or adapts ideas from:

- **Speculative decoding** — Leviathan, Kalman, Matias (ICML 2023,
  [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)) and Chen et al.
  ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)): the lossless
  accept/reject sampling rule in `sampling.metal`.
- **SpecInfer** — Miao et al. (ASPLOS 2024,
  [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)): token-tree
  verification (`docs/TREE_VERIFY_DESIGN.md`, `verify_attention_tree_*`
  kernels).
- **DFlash / DFlash 2** — Chen, Liang, Liu (ICML 2026,
  [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)) and Inco AI: the block
  diffusion draft architecture (`DFlashDraft`, `draft.metal`).
- **EAGLE / EAGLE-2 / EAGLE-3** — Li et al.
  ([arXiv:2406.16858](https://arxiv.org/abs/2406.16858),
  [arXiv:2503.01840](https://arxiv.org/abs/2503.01840)): hidden-state-conditioned
  drafting and dynamic draft trees.
- **Medusa** — Cai et al. (ICML 2024,
  [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)): leaf-alternate tree
  proposals (`RICHENGINE_ANE_MEDUSA`).
- **SuffixDecoding** — Oliaro, Jia, Campos, Qiao (NeurIPS 2025,
  [arXiv:2411.04975](https://arxiv.org/abs/2411.04975)) and **AgSpec** — Lee,
  Cho, Lim, Kwon ([arXiv:2610.01108](https://arxiv.org/abs/2610.01108)):
  retrieval-style drafting for agentic workloads; `RICHENGINE_NGRAM_PREDRAFT`
  is a per-request 3-gram variant.
- **SpecDec++** (ICML 2024, [arXiv:2405.19715](https://arxiv.org/abs/2405.19715))
  and **AdaEAGLE** ([arXiv:2412.18910](https://arxiv.org/abs/2412.18910)):
  adaptive draft-length budgets — realized here as an EWMA heuristic
  (`RICHENGINE_ADAPTIVE_PROPOSALS`).
- **Gated DeltaNet** — Yang, Kautz, Hatamizadeh (ICLR 2025,
  [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)) and the WY-form
  chunkwise DeltaNet of Yang, Wang, Zhang et al. (NeurIPS 2024,
  [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)): the hybrid model's
  linear-attention layers and `gdn.metal`/`gdn_chunked.metal` kernels.
  **Kimi Linear / KDA** ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692))
  is the same family's reference.
- **PagedAttention / vLLM** — Kwon et al. (SOSP 2023,
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)): paged KV, page
  tables, shared blocks (`KvPool`, `KvCache`).
- **RadixAttention / SGLang** — Zheng et al. (NeurIPS 2024,
  [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)): prefix reuse, which
  the block tree and shared-prefix junction implement at page granularity.
- **Sarathi-Serve** — Agrawal et al. (OSDI 2024,
  [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)): chunked prefill
  interleaved with decode.
- **FlashAttention / Flash-Decoding** — Dao et al. (NeurIPS 2022; 2023):
  online-softmax attention and split-KV partial merges.
- **LMCache** — Cheng et al. (HotStorage 2024,
  [arXiv:2310.07240](https://arxiv.org/abs/2310.07240)): tiered KV offload to
  disk (`--max-cache-disk`, `WriteBehind`).
- **Preble** — Srivatsa et al. (SOSP 2024): prefix-aware scheduling
  (`sharedPrefillBoundary` junctions).
- **"LLM in a flash"** — Alizadeh et al. (Apple, ACL 2024,
  [arXiv:2312.11514](https://arxiv.org/abs/2312.11514)): the bandwidth-budget
  framing behind the SSD state/KV tier.
- **GQA** — Ainslie et al. (EMNLP 2023,
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) and
  **StreamingLLM** — Xiao et al. (ICLR 2024,
  [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)): exploited
  structurally (grouped-query KV layouts) and in the draft ring.

Evaluated and rejected, with measurements in `docs/`: **TurboQuant**
([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), **KIVI**
([arXiv:2402.02750](https://arxiv.org/abs/2402.02750)), **KVQuant**
([arXiv:2401.18079](https://arxiv.org/abs/2401.18079)), **Ouroboros**
([arXiv:2402.13720](https://arxiv.org/abs/2402.13720)), **PEARL**
([arXiv:2408.11850](https://arxiv.org/abs/2408.11850)).

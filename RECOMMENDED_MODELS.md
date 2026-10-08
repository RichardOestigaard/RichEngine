# Recommended models

Which installed package to run per device, from the native backend
benchmark (`BENCHMARKING.md`): decode at widths B1–B4 on a 39-token prompt
producing 64 tokens per lane, `--kv-format int8`, three samples each.
Numbers are decode wall tokens/s unless noted.

## MiniCPM5-2B

| Device | Recommended package | Decode B1 | Decode B4 | Prefill 2048 (GPU) | Draft acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| Apple M5 Pro (20-core) | `openbmb/MiniCPM5-2B-MLX` | 216 tok/s | 534 tok/s | 312 ms | 15.2% |
| | `openbmb/MiniCPM5-2B-GGUF:Q4_K_M` | 185 tok/s | 490 tok/s | 312 ms | 15.2% |
| | `openbmb/MiniCPM5-2B-GGUF:MXFP4` | 134 tok/s | 282 tok/s | 429 ms | 13.8% |

**MLX runs best on the M5 Pro** — ~16% ahead of GGUF Q4_K_M at B1 and ~9%
at B4, with identical prefill time and identical draft acceptance. MXFP4
is 40–60% behind MLX at every width and 27% slower on prefill; its native
multiplane decode tiles do not pay for the format's extra weight bytes on
this model's shapes.

### Why MLX wins here

- **Weight bytes per step.** The affine MLX head reads ~137 MB per draft
  pass against the Q6_K GGUF head's ~214 MB, and the DSpark draft re-reads
  the head every step.
- **Dispatch overhead.** GGUF decode linears suspend the baked command
  span per dispatch (an ICB driver workaround); the affine path keeps
  nearly the whole step baked, so encode stays ~0.1 ms/step.
- **Kernel tuning.** The affine decode table carries measured MiniCPM5
  rows (fused QKV at 4 lanes, draft context projection, and a Paired128
  vocabulary head that replaces the Paired256 low-lane pick, ~28% faster
  on that dispatch). GGUF split changes measured faster in isolation were
  rejected end-to-end: reassociating the staged reductions flipped
  near-tie draft proposals and cut acceptance from 15.2% to ~10%.

### Caveats

- Draft acceptance is prompt-dependent (15.2% here on the synthetic
  benchmark prompt). At this acceptance the draft still pays for itself —
  2.06 tokens per batch — but on free-form prompts where it drops further,
  `RICHENGINE_NGRAM_PREDRAFT=1` is worth an A/B.
- Q4_K_M is the fallback when MLX is unavailable (smaller on disk, same
  acceptance, 15% slower decode). MXFP4 is not recommended on M5 Pro.
- These rows are measured on the 20-core M5 Pro; other Apple GPUs have not
  been measured for this model.

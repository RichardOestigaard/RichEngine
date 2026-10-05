# TurboQuant Analysis

Evaluation of TurboQuant (Zandieh et al., Google Research, ICLR 2026,
arXiv:2504.19874) as a KV-cache quantization format for RichEngine — including
the cheaper asymmetric K/V bit-width alternative.

## What it is

Online, data-oblivious vector quantizer for high-dimensional vectors:

1. Randomly rotate the input vector — coordinates concentrate into a
   near-Beta distribution.
2. Apply optimal per-coordinate scalar (Lloyd-Max) quantizers. No codebook
   training, no calibration data.
3. For inner products (`prod` mode): a second stage applies a 1-bit
   Quantized JL transform to the residual, yielding **unbiased**
   inner-product estimates.

Provably near-optimal distortion (within ~2.7x of the Shannon lower bound)
at 1–4 bits/coordinate. Claims ~3.5 bits/channel quality-neutral for KV
cache, ~2.5 bits marginal. Reference impls: `turboquant-py` (NumPy), Rust
crate.

## Why it looked promising

- RichEngine decode is bandwidth-bound; KV bytes/token is the lever.
- The README documents int4's failure mode: quantization noise flips
  near-tie argmaxes during verify and rejects draft tokens (int8: 62 tok/s
  / 63% acceptance vs int4: 56 tok/s / 55% at ~67K ctx). TurboQuant's
  unbiased inner-product mode targets exactly this.
- fp8's NVIDIA advantage doesn't transfer: fp8 `matmul2d` is emulated on
  Apple silicon (see Rigel notes in TO_EXPLORE.md).
- Rotation + scalar quantize maps well to Metal `matmul2d` epilogue fusion.

## Experiment

`dev/turboquant_experiment.py`: prefill 8K tokens on
mlx-community/Qwen3.8-27B-4bit, capture the KV cache, dequantize-in-place
per scheme, then run a 256-token verify pass and compare per-position
argmax against a bf16 reference run (paired). Proxy for spec-decode
acceptance. Run-to-run noise ≈ ±0.5–1 pt.

### Symmetric + TurboQuant

| Scheme | Eff. bits K/V | Token match vs bf16 | Clean 4-token windows |
|---|---:|---:|---:|
| int8 | 8/8 | 98.0% | 94% |
| int4 (current default) | ~4.1/4.1* | 93.7% | 79% |
| TQ K3prod + V3mse | 4/3 | ~92–95% | ~73–83% |
| TQ K4prod + V3mse | 5/3 | ~92.5% | ~76% |
| TQ K2prod + V2mse | 3/2 | ~81% | ~46% (collapses) |

\* int4 + fp32 scale per token-head ≈ 4.13 bits/coord. TQ `prod` adds 1
QJL bit over the MSE bits, so K3prod ≈ 4 effective.

### Asymmetric scalar splits (K bits / V bits)

| Scheme | Avg bits/el | Token match | Clean-4win |
|---|---:|---:|---:|
| K6 / V8 | ~7.1 | 95.3% | 83% |
| K4 / V8 | ~6.1 | 94.5% | 83% |
| K8 / V4 | ~6.1 | ~93–94% | ~78–81% |
| K4 / V6 | ~5.1 | 92.9% | 76% |
| K8 / V2 | ~5.1 | 90.6% | 70% |
| K4 / V2 | ~3.1 | 88.6% | 67% |

## Findings

1. **V precision dominates on Qwen3.5-27B — opposite of the usual
   K-high/V-low folklore.** K4→K8 at V4 gains nothing (~93.7→94.1);
   V4→V8 at K4 is the only consistent gain (93.7→94.5); V at 2 bits
   costs 4–5 pts regardless of K. Plausible mechanism: only 4 KV heads
   and 256-dim head vectors — value noise has no redundancy to average
   over and flows straight into the residual stream producing
   verify logits.
2. **No worthwhile sweet spot exists.** The best asymmetric config
   (K6/V8) is ~0.88x int8 bytes for slightly-worse-than-int8 quality —
   the compression is traded away to preserve quality. The realistic
   compression bound on this model is ~6–7 bits/element average, ~15%
   smaller than int8. Marginal.
3. **TurboQuant does not separate from scalar int4 within noise** at
   equal bytes in this test. Its unbiased-K advantage did not materialize
   on a model where K precision is not the bottleneck.
4. Below ~3 effective bits, every scheme falls off a cliff; the paper's
   2.5-bit quality claim did not reproduce.
5. **The bf16 recency-window trick backfires here** — tested below;
   distant-token noise dominates in this regime.
6. **mxfp4/nvfp4 lose to int4 at equal bytes**, and mxfp8 loses at
   ~2x bytes — block-fp formats are coarser than per-token-head
   symmetric int4, and M5's native mxfp4 matmul doesn't help
   bandwidth-bound decode.

### Recency window (bf16 tail + quantized past)

KIVI's residual buffer and sglang's FP4-KV work both claim the recent
tokens are where quantization quality dies. Tested by leaving the last N
token positions in bf16:

| Scheme | Token match | Clean-4win |
|---|---:|---:|
| int4 uniform | 93.7% | 79% |
| int4 + bf16 last 128 | 91.0% | 71% |
| int4 + bf16 last 512 | 91.8% | 73% |
| int4 + bf16 last 2048 | 91.8% | 73% |
| int2 + bf16 last 512 | 83.9% | 54% |
| K4 old / V2 old + bf16 last 512 | 90.2% | 70% |

The recency window **hurt** — worse than uniform int4 at every size.
Distant-token V noise dominates here (uniform long prompt, attention
spread across the whole cache), the opposite of the MLA/chunked-prefill
regime where the published result was measured.

### Block-fp formats (mxfp4 / nvfp4 / mxfp8)

Tested via MLX's native `mx.quantize`/`mx.dequantize` (the format Metal
4 `matmul2d` accelerates on M5):

| Scheme | Eff. bits/el | Token match | Clean-4win |
|---|---:|---:|---:|
| int4 uniform | ~4.1 | 93.7% | 79% |
| mxfp4 g32 | ~4.25 | 91.8% | 73% |
| nvfp4 g16 | ~4.5 | 91.8% | 73% |
| mxfp8 g32 | ~8.5 | 92.5% | 75% |
| K mxfp4 / V int8 | ~6.1 | 93.7% | 78% |
| K int8 / V mxfp4 | ~6.1 | 93.3% | 76% |

mxfp4 loses to int4 at equal bytes (~2 pts). The E2M1 grid + E8M0
block scale is coarser than per-token-head symmetric int4 with an fp32
scale; even mxfp8 at ~8.5 bits lands below int4 (E4M3's 3 mantissa bits
< int8). The M5 hardware-matmul advantage doesn't help decode either:
decode is bandwidth-bound and mxfp4 reads the same ~0.53 B/elem as int4
— same bytes, worse quality.

## Alternatives landscape (researched, not all tested)

| Method | Idea | Fit |
|---|---|---|
| KIVI | Per-channel K, per-token V grouped quant + residual window | Per-channel K attacks K outliers, but K is not our bottleneck; recency window tested and failed |
| KVQuant | Pre-RoPE K quant, non-uniform, 1% fp16 outliers | Needs calibration; complicates paged layout |
| QuaRot | Model-wide Hadamard rotation (weights + activations + KV) | Invasive model-level change for the same rotation family |
| PolarQuant | Polar-coordinate quantization | Same family, same kernel cost |
| SnapKV / CaM | Token pruning / value merging, not quantization | Orthogonal; SnapKV won long-context throughput in an independent benchmark — separate investigation if ever needed |
| mxfp4 KV | Native E2M1 + E8M0 block scales | Tested: loses to int4 at equal bytes; M5 matmul advantage irrelevant to bandwidth-bound decode |
| fp8 KV | E4M3 storage | Emulated on Metal (0.94x fp16) — footprint only |

## Caveats

- 8K ctx vs the 67K regime where the acceptance tax was measured; more
  near-tie scores at long context could shift gaps either way.
- Single repetitive prompt; real coding prompts have more near-ties.
- Greedy self-consistency proxy, not actual DFlash acceptance.
- Rotation + dequant compute not modeled in the verify path.

## Recommendation

**Do not implement TurboQuant, an asymmetric K/V split, a recency
hybrid, or mxfp4/nvfp4 KV. All were tested; the existing symmetric
int4 + fp32-per-token-head format beat every alternative at equal
bytes, and int8 remains the right default.**

1. Asymmetric split: V must stay at 6–8 bits, so achievable compression
   (~15% under int8) isn't worth a format change.
2. TurboQuant: its unbiased-K edge targets a bottleneck this model
   doesn't have; no separation from scalar int4 within noise.
3. Recency window: actively worse than uniform int4 in this regime.
4. mxfp4/nvfp4: same bytes as int4, ~2 pts worse; the M5 hardware
   matmul advantage is irrelevant to bandwidth-bound decode.
5. The kernel cost is real in every case; vLLM's independent study
   (2026-05) similarly found TurboQuant loses the throughput/capacity
   Pareto in practice.

**Keep int8 as default; keep int4 for capacity-constrained cases only.**
KV quantization below int8 on Qwen3.5-27B is a dead end for decode
quality — V wants bits, not geometry, and ~6–7 bits/element is the
practical floor. Spend effort on the other bytes/token levers in
TO_EXPLORE.md instead (fp4/mxfp4 *weight* paths, kernel fusion).

### Revisit if

- A future model architecture changes the K/V sensitivity balance
  (e.g. more KV heads, smaller head_dim, standard attention instead of
  hybrid GDN) — re-run `dev/turboquant_experiment.py` then.
- Pushing toward 200K+ contexts where capacity forces the issue.

## References

- Paper: https://arxiv.org/abs/2504.19874
- vLLM study: https://vllm.ai/blog/2026-05-11-turboquant
- `turboquant-py`: https://pypi.org/project/turboquant-py
- Experiment: `dev/turboquant_experiment.py` (venv: `.venv-tq/`)
- KV formats: README.md "KV cache formats"; DEVELOPMENT.md

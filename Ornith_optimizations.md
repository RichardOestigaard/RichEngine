# Ornith 1.5 9B — decode optimization analysis (M5 Pro, 16-row verify)

## Current throughput

Measured on the current tree (16 verify rows, all fixes). Identical output
hash `8850415756689472832` at every width; draft acceptance 10.4% (156/1500).

| Width | tok/step | Decode tok/s | Fused GPU/cycle |
|-------|----------|--------------|-----------------|
| B1 | 2.56 | 82.5–86.0 | ~35.3 ms |
| B2 | 2.56 | 119–124 | ~52 ms |
| B3 | 2.56 | 112–116 | ~79 ms |
| B4 | 2.56 | 123–126 | ~88 ms |

8-row baseline: ~67.8 tok/s, 1.94 tok/step at B1 → **+22–26% wall** from the
16-row geometry. MiniCPM5 regression-checked: identical hash, no loss.

## Where a B1 cycle goes (decode-profile, 35.3 ms fused / 37.8 ms dispatched)

| Bucket | ms | Share | Bandwidth |
|--------|-----|-------|-----------|
| `n128_split` GEMMs (QKV/FFN) | ~26.0 | 69% | ~260 GB/s — near peak |
| `decode_head_argmax_q4` (fused target head) | 3.6 | 10% | ~170 GB/s — **below peak** |
| `n128_m16_f32` (draft head) | 2.3 | 6% | ~265 GB/s — at peak |
| `norm_rms` × 79 dispatches | 1.3 | 4% | latency-bound |
| GDN fused + commit | 1.4 | 4% | fine |
| Verify attention (qkv/split/reduce/store) | ~1.0 | 3% | fine |
| Draft attention / selector / commits | ~1.3 | 4% | small |
| Dispatch-boundary slack (fused vs parts) | ~2.5 | 7% | inherent |

Device ceiling measured by `q4-decode-profile`: ~270–287 GB/s effective.
B1 floor at peak ≈ 22 ms; we're at 35.3 ms → ~13 ms of recoverable time.

## What was measured and decided this session

| Path | Result |
|------|--------|
| 8→16 verify rows (draft block 16) | **KEPT** — +22–26% B1, B3 54→115, B4 55→124 |
| Proposal cap sweep (RICHENGINE_PROPOSAL_CAP=7) | Neutral — physical rows dominate cost |
| Fused head beyond 32 rows (banded) | **Regressed** (B3 111.6 vs 115.4) — reverted |
| m48/m64 fused-M + sg16-m64 tiles | **KEPT** — fixed the B3/B4 cliff |
| m48-sg12/sg16, n128-m48 variants | All regressed — removed |
| Tree comb (chain 8 + 8 leaves) | Correct (identical hash), +32% tok/step at B1 — but **flat/slower wall** (85.2 vs 85.7 B1; 89.7 vs 123.2 B2). Env-gated: `RICHENGINE_VERIFY_TREE=1` |
| MiniCPM5 tree verify | Slower (146.9 vs 170.3 B1) — env opt-in, not default |
| tune-kernels sweep (72 keys) | 2 wins at 64 rows: `{12544,4096}@64` → N128 (+9.6%), `{248320,4096}@64` → N128 (+4.6%) — wired into `kMeasuredDecode` |
| `{12288,4096}@16 GateUp` N256 (+28% isolated) | No end-to-end gain — reverted |

## Remaining kernel-level levers, ranked

### 1. Head GEMM at B4 — ~8 ms recoverable (+~9% B4)

`decode_linear_q4_n256_m64_sg16_f32` runs the vocab head at ~98 GB/s
(12.3 ms / 2 dispatches). The same 610 MB of weights stream at 287 GB/s on
the m8 tile. The m64 fused-M tile is still register/compute-bound even at 16
simdgroups.

Fix: a **K-split argmax head** — stream weights once with partial argmax per
K-tile, reduce with the existing `decode_head_argmax_reduce_tiles` kernel.
Expected ~2.5–3 ms total vs 12.3 ms today.

### 2. Fused head input re-read at B1/B2 — ~1 ms recoverable (+~3% B1)

`decode_head_argmax_q4` tiles the vocab in 128-column groups; each of the
1940 threadgroups re-reads the full 32×4096 input (~256 KB → ~500 MB total
re-reads vs 610 MB weights). That overhead is the 170 vs 287 GB/s gap.

Fix: `TN=256` tile variant at M=16 (`tg_tile` = 16 KB, under the 32 KB
threadgroup cap; 248320/256 = 970 tiles). Halves the input re-read.
M=32 (B2) needs TN=128 — 32×256×4 B exceeds the cap — so keep both variants.

### 3. norm_rms fusion — ~1.3 ms + gaps (~+4% B1)

79 tiny RMS dispatches per cycle serialize between the GEMMs. Fuse the norm
into the consuming GEMM's prologue (the linear kernels already load the
activation buffer) or the preceding residual epilogue. Mechanical but touches
every layer path.

### 4. m48/m64 Split128 counts — small B3/B4 tail

The decode sweep covered no m48/m64 split entries — their split counts were
copied from the m32 policy. `n128_split_residual_m48` is the single biggest
B3 bucket (21.9 ms). A measured sweep of split counts at 48/64 rows may
recover a few ms.

### 5. Dispatch-boundary packing — ~2.5 ms at B1

Sum-of-parts exceeds the fused command by ~2.5 ms. More span-baking or
merging the trivial prologue kernels (embedding + verify_input + rope =
3 dispatches, ~30 µs of work each) is the remaining handle.

## The real 2× levers — both asset-blocked

### Draft quality (the wall)

10.4% acceptance caps the whole design: tokens/step = 1 + Σaccept. The extra
tail rows (8–15) contribute ~0.35 tokens/step for 2× the verify footprint.
A draft with ~87% recall at 16 (the DFlash2/codebook claim — the
`ornith9DFlash2DraftLayout()` hook exists but there is no checkpoint) would
roughly double acceptance and is the only engine-visible path to ~2×.

### ANE predraft overlap (+~12% B1)

The draft forward (~4–5 ms, ~1.2 GB) is pure GPU today. The
`RICHENGINE_ANE_PREDRAFT` path exists; it needs a CoreML artifact for the
draft. If produced, the draft hides under the target verify.

### Prefill (separate axis)

512-row prefill = 299 ms, ~87% in `prefill_linear_q4_n128_sg4` tiles. The
N128-sg4 vs N256 choice per shape is tunable if TTFT matters.

## Dead ends (measured, do not retry blindly)

- Proposal caps: physical verify rows dominate; capping is free but buys ~0.
- Fused head beyond 32 rows: loses to unfused at 48 rows.
- Tree comb at 10.4% acceptance: wins tokens/step, loses wall time.
- m48/sg12, m48/sg16, n128-m48, m24/m32 fused-M tiles: all slower.
- Draft bypass at 10.4% acceptance: the draft pays for itself (13% step cost
  for +156% tokens); bypass is off correctly.

## Commands

```sh
# Per-kernel attribution
./build/engine-tests/decode-profile build/engine-tests/production-and-test.metallib \
  install/models/ornith-ai/Ornith-1.5-9B-MLX-4bit

# Kernel candidate sweep (writes winners; add to kMeasuredDecode)
./build/engine-tests/tune-kernels build/engine-tests/production-and-test.metallib \
  install/models/ornith-ai/Ornith-1.5-9B-MLX-4bit --seconds 15 --candidates

# End-to-end A/B
./build/engine-tests/backend-benchmark build/engine-tests/production-and-test.metallib \
  install/models/ornith-ai/Ornith-1.5-9B-MLX-4bit --scenario decode --samples 3
```

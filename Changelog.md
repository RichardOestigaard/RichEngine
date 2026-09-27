# Changelog

## Unreleased — Apple M5 Pro performance pass

Measured on Apple M5 Pro (GPU family 10, 20 cores), model Qwen3.8-27B-Splash,
512-token prompt, fused single-command-buffer decode cycles.

### Decode throughput

Two policy changes landed from this pass so far
(`runtime/ops/Linear.cpp`): `kPaired256TilesPerCore` 8→3 for one-lane plain
projections, and a full resident N256 grid for one- and two-lane gate/up
projections up to four tiles per core.

decode-profile, 6 cycles per width, median fused GPU ms/cycle:

| Batch | Before (stash notes) | Full pass (stash notes) | Now |
|---|---|---|---|
| B1 | 83.36 | 74.92 | 73.89 |
| B2 | 101.6 | 87.87 | 86.92 |
| B3 | 136.4 | 129.32 | 127.96 |
| B4 | 147.4 | 138.54 | 139.54 |

The two applied changes alone match or beat the previously recorded
full-pass numbers at B1–B3. B4 remains ~1 ms short; the lane-4 N128
resident-wave rule targets it.

Draft acceptance unchanged at ~0.87 across all batch widths.

### Changes applied

| Change | File | Effect |
|---|---|---|
| One-lane plain projections switch to Paired256 at 3 tiles/core (was 8) | `runtime/ops/Linear.cpp` | `{16640,5120}` B1 ~+17% operator-level; B1 cycle 83.36 → 80.34 |
| Gate/up lanes ≤2 use full N256 grid up to 4 tiles/core | `runtime/ops/Linear.cpp` | `decode_linear_q4_n256_gate_up` 33.4 → 28.0 ms at B1; B1 cycle → 74.84 |

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
| Long-context verify attention | ~155 GB/s vs ~285 GB/s peak at 131k history; fp32 split partials scale linearly |
| m16/m24/m32 decode tiles | 100–170 GB/s vs 220–285 GB/s for m8; epilogue vectorization is the next lever |
| `norm_rms` dispatch batching | 141 dispatches ≈ 1.5 ms/cycle (~1%) |
| Constrained decode host round-trip | per-step ~170 KB mask fill + extra submissions; grammar workloads only |
| MoE model tuning | Qwen3.6-35B-A3B shapes need a dedicated tune-kernels run |

### Validation

| Check | Result |
|---|---|
| `decode-profile` B1–B4, 512 prompt, 6 cycles | B1 73.89, B2 86.92, B3 127.96, B4 139.54 ms — no regressions vs recorded baselines |

Note: `decode-profile` caps each lane at 256 new tokens, so runs above
~6 cycles per width abort early ("completed request was decoded").

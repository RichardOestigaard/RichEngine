# Qwen 3.8 27B Findings

Cross-check of the two "remaining opportunities" lines of the M5 Pro
performance pass in `Changelog.md`, re-measured on the pass's own machine.
These measurements characterize the current kernels and policies; they change
whenever the kernels, the shape table of `q4_decode_profile` or the device do.

## Setup

Apple M5 Pro (GPU family 10, 20 cores, 48 GB), macOS 27.0.1, source `a907d1f`
(build ID `src-2b0bd2b373d1516b908da3b0a80c2a6a53c3b6f30242b6c6af3dc20824089198`),
2026-10-04.

- Decode tiles: `make benchmark-decode` (`q4_decode_profile`, median of nine
  samples per group count, per production shape).
- Verify attention: `make benchmark-attention-sweep
  ATTENTION_SWEEP_ARGS='--histories 131072 --shapes 27b --phases verify
  --lanes 1,4 --repeat 15'`.

## Decode tiles: measured GB/s per tile-row class

Best (fastest) group count per pipeline, GB/s of the streamed weight bytes:

| shape | m8 | m16 | m24 | m32 |
| --- | ---: | ---: | ---: | ---: |
| ffn_gate_up | 261.8 | 255.4 | 170.2 | 164.6 |
| lm_head | 276.8 | 265.8 | 177.3 | 176.1 |
| gdn_input | 223.2 | 208.5 | 147.1 | 153.7 |
| full_input | 205.3 | 190.0 | 144.4 | 138.3 |
| mixer_output | 162.5 | 150.9 | 97.2 | 102.4 |
| draft_context | 154.7 | 144.6 | 98.7 | 103.4 |
| ffn_down | 153.8 | 142.0 | 98.6 | 103.0 |

Findings:

- The changelog's "100–170 vs 220–285" is a shape-cherry-picked compression.
  The m8 class spans 154–287 (153–163 on the three small shapes); m16 is
  142–268.
- Per shape, m16 loses 3–8% to m8; **m24 and m32 lose 30–40%**. The real gap
  is in m24/m32, the tiles the policy runs at three and four lanes.
- "Epilogue vectorization is the next lever" is a hypothesis, not a
  measurement. The per-quant-group scale/bias epilogue does one scalar device
  load per element per group, so wider tiles do 4–8x that work per thread;
  but the row gap may instead be accumulator register pressure and the fewer,
  larger threadgroups. The register-cached scale/bias variant measured 21–43%
  slower and was reverted (Changelog, "measured and rejected").
- Cycle-level, the only recorded evidence for the lever is the discarded
  staged scale/bias epilogue: −4.5% B2, −2.5% B3. The landed policy changes
  of the same pass gave +12% at B1/B2, where m16 is already near m8. The
  changelog's "dwarfs the policy tweaks" does not survive the measurement;
  "matches the policy tweaks at B3/B4" is the defensible claim.

## Long-context verify attention: measured

27b geometry (24 query heads, 4 KV heads, d=256), int8 Page32 KV, one
attention layer of the model's store + attention graph:

| lanes | fused ms | KV bytes | effective GB/s | split kernel | reduce |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.829 | 272.6 MB | 149 | 1.701 ms | 0.117 ms |
| 4 | 6.781 | 1090.6 MB | 161 | 6.454 ms | 0.369 ms |

- The changelog's "~155 vs ~285 peak" reproduces: 149–161 GB/s, 52–56% of
  the 287 GB/s the m8 tiles sustain above.
- "fp32 split partials scale linearly in history" is only true below ~65k
  tokens: `kv::verifyAttentionSplits` takes one split per 16 pages, capped at
  128 (`RICHENGINE_VERIFY_ATTENTION_MAXIMUM_SPLITS`). At 131k the partials are a
  fixed ~25 MB per layer (~9% of the KV traffic); the linear cost at 131k is
  the KV read itself running at half bandwidth.
- The model has 16 full-attention layers (64 layers, full-attention period
  4), so 131k history spends ~29 ms per verify step in fused attention time
  at one lane, against a ~15 ms floor at peak bandwidth. It is the only
  decode cost that grows with history, and the "dominant cost for long
  conversations" claim holds.

## Verdict

- **Verify attention: worth implementing.** ~13 ms/step of headroom at 131k
  and the only history-scaling cost. The experiments rejected in
  [remaining-decode-optimizations.md](remaining-decode-optimizations.md)
  rule out those implementations, not the 52%-of-peak gap.
- **Tile epilogue: worth one attempt, modest expectations.** Reimplement the
  staged scale/bias epilogue (the register-cached form is known to lose);
  target m24/m32 at B3/B4, where the 30–40% operator gap lives.

## Follow-up: `_r2` chain verify, 2026-10-12

Since this pass, `RICHENGINE_TARGET_VERIFY_ROWS` went 8→16 for tree verify,
so the chain tile is M = 96 fused rows: ~25 KB of threadgroup scratch and a
96-fp32 PV accumulator per thread drop core residency to one group, and the
measured verify kernel slowed to ~54 GB/s at 131k (from the 149–161 above).
The bottleneck is per-group serial cost, not fetch latency: warming the next
page's lines through plain loads costs 5–8% at every tile size, so the fix
had to shrink per-group footprint, not overlap fetches.

The `_r2` splits run the tile's two fused-row passes on two threadgroups
(group.x packs KV head and pass), keeping the `_m2` halved scratch and the
unchanged partial slots. A four-way `_r4` variant loses: per-page fixed
costs (three barriers, two tensorop issues per 16 KB) dominate below
MP = 48. attention-sweep, 27b int8 Page32, fused ms:

| history | lanes | `_split` | `_split_r2` |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.028 | 0.023 |
| 0 | 4 | 0.064 | 0.050 |
| 8192 | 1 | 0.522 | 0.353 |
| 8192 | 4 | 1.726 | 1.301 |
| 16384 | 1 | 0.835 | 0.591 |
| 16384 | 4 | 2.867 | 2.339 |
| 65536 | 1 | 2.957 | 2.318 |
| 65536 | 4 | 11.25 | 8.96 |
| 131072 | 1 | 5.26 | 4.25 |
| 131072 | 4 | 20.46 | 16.69 |

−17% to −33% at every history and lane count, larger than this pass's
predicted ceiling because it also recovers the M=96 regression. Outputs are
bit-identical to the `_m2` two-pass tile on every measured case and differ
from the single-pass tile only by MPP accumulation order; the paged plan
test's CPU-oracle runs pass for all three KV formats. Tree verify is
unchanged and is the same shape's next candidate.

Rejected on measurement: a 512-thread (16-simdgroup) single-pass tile
(`_w512`, `execution_simdgroups<16>` in both matmuls). It halves the PV
accumulator per thread and keeps 1x KV traffic, but loses to the plain
256-thread tile on every measured case — +5% to +29% fused time at
8192/65536/131072 and both lane counts — because threadgroup scratch, not
registers, bounds residency (one group per core either way) while the wider
dispatch only adds sync and issue cost. Its outputs are bit-identical to
the plain tile's on every measured case. Per-group resources only pay when
they also shrink scratch and raise resident groups, which is the `_r2`
mechanism.

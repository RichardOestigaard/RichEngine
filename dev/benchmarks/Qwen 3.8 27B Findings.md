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

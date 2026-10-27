# DPS_V41F Findings

Analysis of Metal 4.1, the local RichEngine/Splash tree, upstream
[incoai/splash](https://github.com/incoai/splash), [Locals](https://github.com/Generated-Computers/Locals),
and [Strata](https://github.com/Niko1221/Strata).

Date: 2026-10-26

## Scope and method

- Local tree: `/Users/richardgoestigaard/Documents/GitHub/splash`
  (`dev`, HEAD `07a733f5`, upstream remote `incoai/splash`).
- Upstream compared remotely with `gh api`; no repository was cloned.
- Strata and Locals inspected remotely with the GitHub API.
- Metal 4.1 checked against the macOS 27 SDK, WWDC26 sessions, the MSL 4.1
  specification, and the Rigel reverse-engineering paper.
- Local hot paths audited read-only.
- Numbers below are either measured by the cited source or explicitly marked
  as estimates. Where a number comes from a fork/agent benchmark, treat it as
  directional until reproduced locally.

## Executive summary

The local fork is **substantially ahead of upstream Splash** on Metal 4.x
adoption, per-device autotuning, ANE offload, model coverage, and benchmark
tooling. Do not merge upstream wholesale: it still targets macOS 26.4 and
Metal 4.0, and lacks the local MTL4/ANE/autotune work.

The remaining value is concentrated in four areas:

1. **Cherry-pick a small set of upstream scheduler/kernel commits.** The
   mixed constrained/unconstrained decode lane change is the largest
   current-model win; several tile/split changes are small and bit-identical.
2. **Wire `runtime/qwen4/` into the build and installer.** The Flash-Next,
   streamed-expert, PLE-cache, and sparse-KV stack is already staged in the
   local tree but not built or served.
3. **Port two agent-serving ideas from Strata/Locals:** branch-aware prompt
   checkpoints (open), and token-budget-aware constrained decoding
   (**shipped** as the byte-level bound behind `RICHENGINE_DFA_BUDGET_OFF`;
   P1 section for status).
4. **Retune Apple10 (M5/M6) tiles and splits.** Metal 4.1's headline tensor
   APIs are already in use; the remaining Metal 4.1 work is mostly tuning,
   not new API adoption.

Rejected: cooperative-tensor FlashAttention. An M5 Pro microbenchmark measured
it **1.8–3.1x slower** than the existing 8-SG tile because cooperative-tensor
input forces one SIMD group and loses GQA/KV reuse.

## P0 — highest-value changes

| Priority | Change | Source | Evidence | Effort / risk |
|---|---|---|---|---|
| 1 | Mix constrained and unconstrained decode lanes in one command | upstream `f52d1ad5`; local blocks it at `runtime/model/Model.hpp:269` | +24% aggregate on 35B (349→457 tok/s), +34% on 27B (138→210); halves decode commands. **Verified not applicable: the semantics are already present at dev HEAD** — see the status note below the table | none — done at base HEAD |
| 2 | Split K up to 16 on narrow register-tile grids | upstream `6198321e`; local `LinearConfig::kMaximumSplits=8` | 11–17% less projection time | S / low |
| 3 | One-lane 16-row staged tile under 4.5 bits/weight | upstream `b701328b` | 3.3% faster 27B IQ3 cycle, 0.6% Flash-Next, 2–10% faster tile; outputs bit-identical | S / low |
| 4 | MoE expert column tile for 1–2 live rows | upstream `ed63ffb5` | 5–9% faster per MoE layer on 35B Q4_K, 5–25% on M6; one-lane step 27.7 vs 29.7 ms | M / low |
| 5 | Treat `min_p=1` and `top_k=1` as greedy | upstream `8d5e05b2`; `runtime/ops/Sampling.hpp` | small, free; default sampling unchanged | XS / low |
| 6 | Support pruned routed-expert counts | upstream `7b21db34`; `runtime/model/ModelDescriptor.mm` | enables 256/512-expert Flash-Next variants | S / low |
| 7 | Auto-select the 8-row verify build for block-8 drafts | local audit; `REGRESSION_FINDINGS.md` | 16-row verify costs ~10% bandwidth with no acceptance gain | S / low |

**Status (Oct 2026), verified against upstream `f52d1ad5` and this fork's dev
HEAD:**

1. **Mixed lanes is already realized here.** The static audit blocking this at
   `Model.hpp:269` read `BatchPlan::constrained` as a gate; it was a dead
   field. `Scheduler` has no constraint grouping, and `RuntimeImpl` already
   runs the per-lane constrained ticket with mixed `decodeAsync`. The
   +24–34% upstream attributed to this change should largely be already
   banked in this fork's decode batches; any residual gap above that is
   upstream tile work with the same commit, not the lane-mixing itself.
   flag's removal and an engine-level mask-overlap test
   (`testConstrainedLaneSharesDecodeCommand`), on the local branch
   `mix-constrained-lanes`.
2. **Re-run gate.** The ~85 kv-first replay-point/suspension failures,
   ragged-scheduler's "boundary armed again", native-engine-loop's score-prefill
   chunking assert, and verify-budget's cycle labeling all fail at pristine
   HEAD. Both conclusions above are static-code verified but the suites that
   would behaviorally confirm them abort on those pre-existing failures; treat
   either as needing one clean run of `test-real` after those are triaged.

Validation:

```bash
make -j4
make build/engine-tests/backend-benchmark
./build/engine-tests/backend-benchmark build/richengine.metallib \
  "install/models/incoai/Qwen3.8-27B-RichEngine" \
  --scenario decode,partial,short --samples 3 --kv-format int8
```

Also build and compare `make verify8` where the draft block size is 8.

## P0 — unlock the staged Qwen4 / Flash-Next stack

`runtime/qwen4/` contains most of upstream's newer architecture stack:

- `ExpertStore`, `SparseRowStore`, streamed experts
- sparse/SSD-backed KV and KV history streaming
- PLE row cache
- host steps between GPU work
- prefill pass/group scheduling
- hyper-connections, n-gram embedding, block indexer

However, this tree is **not wired into `Makefile` or `dev/native.mk`**, and has
no `install/families` integration, so it is not served. Upstream's PLE cache
(`18f9c69b`) measured **+16/+20/+35% decode on a 24 GB machine**. Wiring this
in is higher leverage than most remaining micro-kernel work.

Relevant upstream commits: `2faa0124`, `d6003397`, `99dcf73c`, `dc650319`,
`18f9c69b`, `51172500`, `37025c27`, `4a16aaa6`, `daf6755b`, `e3837ba3`.

## P1 — agent-serving latency

### Branch-aware prompt checkpoints (Strata)

Local checkpoint planning deliberately omits checkpoints within one prefill
chunk (2,048 tokens) of either end (`runtime/engine/Engine.hpp:63`). The replay
point is the start of the generation prompt (`runtime/engine/Engine.cpp:665`),
so editing the final user message misses the reusable state. The server's
shared-prefix probe only covers the leading system message/tools
(`server/frontend.py:1149-1291`).

Port the conservative Strata behavior: checkpoint the first completed chunk
boundary within one chunk of the prompt end, evict it first, and never let it
displace a periodic checkpoint. Strata reduced its first edited-record case
from **17.236 s to 11.527 s** and three edits from **41.266 s to 35.802 s**.
At Splash's measured ~515 prefill tok/s, avoiding a 2,048-token reread is
roughly 4 seconds per branch.

Caveat: a snapshot costs a command split and a state copy (Strata's is about
118 MiB). Measure the composite-state size and arm this only for requests that
actually branch.

### Budget-aware constrained decoding (Locals) — shipped

Locals precomputes, per DFA state, the minimum remaining tokens to reach
acceptance and the token cost a sampled transition must leave
(`locals/budget.py`, `locals/engine.py`, `locals/forced.py`). At each step with
`r` tokens left, it masks to transitions that can finish within `r`, plus EOS.
This prevents `max_tokens` from truncating JSON or a tool call mid-value.

Splash already had GPU DFA (`runtime/metal/kernels/decode/dfa.metal`,
`server/constraints.py`) and forced spans, but no token-budget table. **Shipped
as the byte-level upper bound (B_bytes)** on the local branch
`budget-aware-dfa`:

- `CompiledDfa.byte_costs()` (`server/dfa.py`) computes, per state, the
  minimum **byte** count of an acceptance-requiring walk over unmarked
  edges (32767 = none a regular token can finish). It ships as an optional
  trailing section (u32 length + n_states i16) on the `CONSTRAINT_DFA`
  frame; empty keeps the old plain-walk semantics, so the wire is
  compatible in both directions.
- The GPU mask kernel gates a row's token on `1 + cost(landing)` fitting
  under that lane's remaining decode budget (`tokens_left` in
  `DfaMaskParams`/`DfaWalkParams`, both now 208/220 B with the field);
  `decode_dfa_forced_tokens` caps proposal-span slots at the budget a
  round can actually spend.
- A/B at runtime on one binary: `RICHENGINE_DFA_BUDGET_OFF` (presence)
  disables the gate; costs always ship. The gate arms only when
  `RICHENGINE_DFA_MODE=full` (or a family declares it).

Byte counts never understate the token cost (single-byte tokens walk every
unmarked edge), so gating can only refuse continuations a token-budget
rule would also refuse. Locals measured **1.31x** on 18 parallel JSON jobs
and **1.60x** when only keys are prompted; on Splash the rule is mainly a
correctness and waste-reduction win — near budget exhaustion it refuses
grammar walks that cannot complete instead of verifying rows the budget
truncates anyway, and it stops proposal spans from overdrawing the row
budget.

**A/B result (Oct 27 2026, `dfa-benchmark` on Qwen3.8-27B-4bit,
`RICHENGINE_DFA_MODE=full`, tool-call JSON fixture):** gated and ungated
(`RICHENGINE_DFA_BUDGET_OFF=1`) are behaviorally identical at width 1 and
width 4 across --steps 12/24/40 (same token ids, same batch counts,
`invalid_commits=0`, zero degenerate rows, decode GPU time within noise).
The rule never bound: a JSON grammar's per-state bound is the remaining
**byte** count of its walk (max 61 over the fixture's 694 states), and the
gate condition `budget ≥ 1 + cost(start)` falls back to the plain walk
once the remaining token budget is below that counted in bytes — for any
realistic grammar tail the fallback dominates before refusals could
diverge. So the rule as shipped is provably safe (cannot produce invalid
commits, no measured overhead) but under this fallback policy it acts
free rather than bind — punishing runaway spans needs the strict variant
(read: drop the fallback and let exhausted budgets starve rows to the
model's stop, or compare the bound in tokens, not bytes). Keep the
opt-out knob for that A/B. Where the rule pays off most: small
`max_tokens` on tool-call/JSON traffic plus budget-bound forced spans.

### Overlap or mix prefill with decode (Locals, Strata)

Locals concatenates decode rows with a prefill chunk so projections and MLP
weights are read once (`locals/mixed.py`, `locals/hybrid_batch.py`). Strata's
piped concurrency keeps active streams moving during long prompts.

Reported effects:

- Locals: worst stall during an 8K prompt **9.7 s → 2.8 s**; stream throughput
  **104 vs 107 tok/s**; prompt TTFT **10.4 s vs 9.8 s**.
- Strata: longest stream stop **11.9 s → 0.77–0.84 s**; stream rate
  **289–291 → 1,411–1,679 tok/s**; max stream gap **9.8 s → 1.0 s**; join TTFT
  **3.8 s → 4.6 s**.

This is a fairness/latency trade, not a universal throughput win. Start with an
opt-in second prefill channel or remove only false dependencies between
prefill compute and decode. Keep cancellation, residency, and memory admission
correct before making it default.

### Raise the 2-request scheduler ceiling

`runtime/engine/Scheduler.cpp:244-266` refuses to start more work once
`active_.size() >= 2`. This matches the B2→B3/B4 plateau
(177 → 191 → 246 tok/s on 27B). Raising it is a cheap experiment, but the
submit-ahead path has real dependencies: `Engine.cpp:1300-1332` returns
`Prepared::Waiting` and cannot yield mid-prepare.

### Decouple plain decode width from verify depth (Locals)

Locals routes each `(N, K, M, dtype)` to the fastest small-row kernel and
processes ragged batch attention in one launch. At 8 requests it measured
**272.7 vs 86.2 tok/s** against mlx-lm; at 16 requests **341 vs 157**. Splash
caps batch width at 4 and ties lanes to 16 verify rows
(`docs/BATCH_RESEARCH.md`, `runtime/engine/Scheduler.cpp`).

A separate plain-decode mode at 8–16 one-row requests is worthwhile even if
speculative verify stays at 4 lanes. This requires M8/M16 tiles, ragged
attention, row packing, and admission for per-row recurrent state. High effort
and risk; do it after the 4-lane path is tuned.

### Exact prompt-token replay (Locals)

Keep the last N prompt+reply token streams and raw tool-call replies so a
client resending a conversation reuses exact token IDs and encodes only the
suffix. This prevents template re-rendering from silently breaking prefix-cache
hits. Splash has prefix cache, junctions, and rolling checkpoints, but not raw
tool-call replay. Low-risk server work.

## P1 — memory, KV, and attention

### Mixed K8/V4 KV cache (Strata)

Strata's `--kv k8v4` keeps keys at 8 bits and rotated values at 4 bits:
23% less KV memory than INT8, more accurate than `q4_0`, with prompts 2–5%
slower. Splash's Qwen4 path supports only INT8 and BF16 KV
(`runtime/qwen4/ops/PagedKv.hpp:17`).

Add K8/V4 to `KvSlotFormat`/`PagedKv`, with Hadamard-rotated V packing and
attention tiles that accept separate K and V formats. Preserve
snapshot/restore versioning. Gate on speculative acceptance and logit parity,
not perplexity alone: local notes already show int4 KV costing draft acceptance
at long context.

### QSA small-row register path (Strata)

`block_indexer_score` uses an MPP product even for one-to-four decode rows.
Strata's `STRATA_QSA_ROWS=1` register-caches queries and keys and replaces four
reductions with six shuffles:

- 47.2 → 36.2 µs at one query
- 101.2 → 54.6 µs at four queries
- ~+3.2% end-to-end

Specialize `runtime/qwen4/metal/kernels/shared/block_indexer.metal` for 1–4
rows and keep MPP for prefill. The selection budget already matches Strata's
(512 four-token blocks plus the current block).

### Prefill head memory alias (Strata, conditional)

Strata's `--prefill-head-alias` raised chunk size from 512 to 2,048 and prefill
from **292.5 to 667.1 tok/s** on an 8 GB card. On unified-memory Macs the win
is smaller, and Splash's 2,048-token budget already tested flat at 4,096.
Only pursue this if it unlocks a larger chunk on 24 GB machines under
expert/KV pressure. Prefer aliasing the head's pages through a Metal heap over
unloading and rereading it.

### Contiguous expert pack (Strata, benchmark first)

Strata's contiguous `.experts` file produced M4 32 GB results of
**7.37 → 8.62 tok/s** decode and **64.7 → 84.1 tok/s** on a 6K prompt, with SSD
wait **46.6 → 31.5 ms/token**. Splash already reads native rows directly,
aligns pages, predicts the next block's experts, and reuses slots. GGUF expert
slices are often already contiguous, so benchmark a read trace before adding a
large copy step.

### Remaining low-risk kernel work

- Stage Q4 scales/biases in threadgroup memory in prefill
  (`runtime/metal/kernels/prefill/linear_q4.metal`,
  `runtime/metal/kernels/common/q4_mpp_tiles.h`). Scales and biases are still
  read per element from device memory.
- Fuse sampling mass into vocabulary search. Both read the same logits and
  masks, so this removes one full-vocabulary read per sampled row
  (`runtime/ops/Sampling.cpp:249-290`).
- Emit `Table64` from the MoE down/combine epilogue so the next projection
  skips prepare (`runtime/ops/Linear.cpp:912-920`).
- Fold the shared-expert scalar gate into route-scores as a second output
  instead of running a separate select (`runtime/ops/MoE.cpp:522-529`).
- Make the verify/prefill split floor a function of visible pages instead of a
  fixed 8 (`runtime/ops/PagedAttention.hpp:49-76`).

## Metal 4.1 status

Verified environment: Xcode 27.1, macOS SDK 27.0, macOS 27.0.1, M5 Pro.
Splash's floor is macOS 27 and Apple GPU family 9.

Already used by the local tree:

- Metal 4 queues, allocators, command buffers, argument tables, compilers,
  counter heaps (`runtime/metal/MetalEncode.mm`,
  `runtime/metal/MetalBackendImpl.hpp`)
- residency sets (`runtime/metal/Residency.hpp`)
- MPP `matmul2d` (`q4_mpp_tiles.h`, `gguf_mxfp4_tile.h`, `q2k_mpp_tile.h`)
- multiplane `tensor_blockwise` MXFP4
- packed Q2_K `uint2b`
- FP8 E4M3 KV
- acquire/release atomics where needed

Not used and not worth adopting for the hot path:

- host `MTLTensor` objects and `MTL4MachineLearningCommandEncoder` (the ML
  encoder dispatches compiled networks, not hand-written kernels)
- cooperative-tensor inputs, `reduce_rows`, `map_iterator`
- placement new, `interleave`/`deinterleave`, new texture reads
- archives/binary functions (startup-only)
- bf16 x fp8/fp4 matmul (unsupported pair)
- FP8/FP4 for speed on M4 (emulated; Rigel measured ~0.94x fp16)
- `atomic_wait`/notify retry (fails forward-progress requirements on family 10)

Remaining worthwhile Metal 4.x work:

1. M5/Apple10 tile, split-K, N-tile, and core-count retuning: 1–5% estimated.
2. Validate/tune MXFP4 multiplane and Q2_K on M5 base/Pro/Max: 1–3%.
3. Fused gate/up and epilogue tuning; verify/draft split tuning: 1–3%.
4. Barrier/atomic scope audit: <1%.
5. Fix `chipName(10)` returning `"M4"` instead of `"M5"`
   (`runtime/metal/DeviceCapabilities.hpp:61`).
6. Fix `attention-sweep` dropping `AttentionFixturePlan::attentionCore`.

Hardware caveats:

- M3/M4 are Apple9 and have no neural accelerator. MPP is only ~1.05–1.21x
  over `simdgroup_matrix` on M4, and fp8 is emulated. Treat FP8/FP4/int2 as
  footprint features on Apple9, not speed features.
- M5/M5 Pro/Max are Apple10 and have a neural accelerator in every GPU core.
  It helps compute-bound prefill; decode remains bandwidth-bound.
- `tensor_blockwise`/multiplane are API-available from Apple7, but the local
  Apple10 gate for MXFP4 is reasonable until M4 performance is measured.

### Rejected: cooperative-tensor FlashAttention

Microbenchmark: one head, D=256, Page32, bf16 K/V, 2,048 query rows, 2K/8K/32K
history, M5 Pro, Metal 4.1.

| History | CT M=8 | CT M=16 | CT M=32 |
|---|---:|---:|---:|
| 2K | ~3.1x slower | ~1.8x slower | ~2.2x slower |
| 8K | ~3.2x slower | ~1.85x slower | ~2.8x slower |
| 32K | ~3.05x slower | ~1.95x slower | ~2.9x slower |

Outputs matched (max diff 5e-3 at 2K, 3e-4 at 32K), so this is a performance
result, not a correctness failure. Cooperative-tensor input requires one SIMD
group; the current 8-SG tile shares K/V across 64 rows, while a one-SG M=8
tile rereads K/V eight times. Do not implement this path.

## Do not port

- Strata CPU/GPU expert tiers, pinned/visible slots, PCIe zero-copy: unified
  memory makes these largely N/A.
- DeepSeek V4.1 5-of-6 expert scheduling: quality trade, and Strata does not
  run DeepSeek V4.1 on Mac.
- `moe_mm` grouped MPP: Splash already has the indirect grouped-row form.
- MTP-chain command merging: Splash already runs decode/draft in a command
  ring.
- QSA cell budget: already equivalent.
- KV prefetch/PLE read-ahead: local `KvHistoryStream` and `Qwen4Steps::readAhead`
  already cover these.
- 8-bit KV attention, fused GDN decode, projection fusion, MoE expert
  grouping: already present or covered by MPP tiles.
- Chunked GDN: local measurements show it slower despite the plan doc ranking
  it first.

## Documentation contradictions to fix

- MTL4 is **on by default** (`runtime/Tuning.hpp:39`,
  `runtime/metal/MetalEncode.mm:456-460`), but
  `runtime/metal/MetalBackendImpl.hpp:997-999`, `UNTAPPED_POTENTIAL.md`, and
  `docs/SWE_2_MTL_FINDINGS.md` still call it opt-in via `RICHENGINE_MTL4=1`.
- `docs/performance.md:177` says the default KV format is INT4, while
  benchmarks run `--kv-format int8`.
- The "acceptance down from 87%" note in `docs/performance.md:36-38` was a
  prompt-remap artifact; `REGRESSION_FINDINGS.md` reports 87.3–88.7% and
  ~94–104 tok/s B1 after the fix.
- `docs/DECODE_SPEED_PLAN.md` says submit-ahead is prefill-only; decode ring2
  is default-on (`runtime/engine/Engine.cpp:321`). Its prepare-fusion item is
  effectively shipped through table-norm.
- `docs/PREFILL_OPTIMIZATION_PLAN.md` ranks chunked GDN first even though its
  own measurements show it slower.
- `docs/PERF_REPORT.md:28,100` still says one pending ticket and submit-ahead
  unbuilt.

## Recommended execution order

1. ~~Cherry-pick `f52d1ad5`~~ — already realized at dev HEAD (see the P0
   status note). Cherry-pick `6198321e`, `b701328b`, `ed63ffb5`, `8d5e05b2`,
   `7b21db34`; add automatic verify8 selection.
2. Sweep verify/prefill split floors and the scheduler active-request ceiling;
   fuse sampling mass, emit MoE `Table64`, stage Q4 scales.
3. Wire `runtime/qwen4/` into `Makefile`, `dev/native.mk`, and the installer.
4. Branch-aware checkpoints; ~~budget-aware constrained decoding~~ — shipped on
   `budget-aware-dfa` (byte-level bound, `RICHENGINE_DFA_BUDGET_OFF` A/B);
   A/B it with `RICHENGINE_DFA_MODE=full` and merge.
5. Prototype mixed prefill+decode and K8/V4 KV; retune Apple10 tiles.
6. Reject cooperative-tensor FlashAttention and any Metal 4.1 feature that
   lacks a measured fast path.

## Sources

- [incoai/splash](https://github.com/incoai/splash)
- [Generated-Computers/Locals](https://github.com/Generated-Computers/Locals)
- [Niko1221/Strata](https://github.com/Niko1221/Strata)
- [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [WWDC26 session 330: Optimize custom machine learning operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)
- [WWDC26 session 359: Build real-time neural rendering pipelines with Metal](https://developer.apple.com/videos/play/wwdc2026/359/)
- [Tech Talk 111432: Accelerate your machine learning workloads with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/)
- [Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU](https://arxiv.org/html/2606.12765)

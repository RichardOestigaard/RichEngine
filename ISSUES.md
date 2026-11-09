# Issues & Known Limitations

Collected during the Metal 4.1 / Ornith session (2026-10). Each entry states the
symptom, what was ruled out, and the current workaround.

## Metal / driver level

### GGUF kernels produce no output under indirect command replay
`moe_expert_gguf_*` (staged and register) and `gguf_embed_*` (iq4xs/iq4nl/iq3s
observed) produce zero or wrong output when replayed from an
`MTLIndirectCommandBuffer` on this M5 Pro driver, even with
`memoryBarrierWithScope`, `useResource` on every buffer, single-command ICBs,
and full encoder-boundary serialization. `MTL_DEBUG_LAYER=1` (which serializes
GPU work) makes it pass — scheduling quirk, not our encoding.
Workaround: `CommandGraph::suspendBakedSpan()`/`resumeBakedSpan()` around
`Linear::add`'s Block32 `addGguf` path and the MoE expert passes; everything
else stays baked. Candidate for a Metal Feedback report with a minimal repro.
Update (2026-10-17): commands *encoded on device* replay correctly —
`RICHENGINE_GPU_SPANS=1` marks the suspended runs as GPU-encoded spans
(`gpu_encode` writes the `MTLIndirectComputeCommand`s from staged
descriptors, `kernels/shared/gpu_dispatch.metal`), verified bitwise against
direct encoding on gguf-moe-benchmark B1–B4 and the full gguf-moe parity
suite. One anomaly: under `MTL_SHADER_VALIDATION=1` a staged B4 run zeroed
one expert tile (unvalidated runs are bitwise-clean everywhere). The
current read is a validation-layer limitation, but this is unverified —
no minimal repro exists, and the same run mutates ICB contents on device,
so a real hazard in the encode→barrier→execute ordering is not ruled
out. The anomaly is recorded unresolved; the path stays opt-in
(`RICHENGINE_GPU_SPANS=1`) and must not ship enabled-by-default until a
repro under validation decides which side is at fault.

### `verify_attention_q8_split` repeat flake
Identical submissions of the int8 split-attention kernel occasionally differ by
~13–70 bytes (missed-rescale signature) at ≥48 threadgroups. Source-level audit
is clean (all producer→consumer edges barriered, coop tensors respected, ragged
pages handled); stress runs could not reliably reproduce. Hypothesis: tensorop
operand reads of threadgroup memory may not be ordered by `threadgroup_barrier`
the way plain loads are — `gguf_staged_tile.h` already double-buffers its
operand for the same reason. Defensive ping-pong of the `probabilities` buffer
is applied in `paged_attention_tile.h`; packed K/V staging left single-buffered
(threadgroup budget). Not conclusively resolved.

### MPP `matmul2d` silent failure modes
- Right-operand N>16 (and some left-M>16 configurations) produce silently wrong
  results on this M5/MPP build — killed C=128 in the chunked-GDN probe and
  several earlier variants.
- fp32 operands produce ~0 on small (N=16-class) matmuls; worked at N≥32.
- BF16×FP4 / FP8 operand combos unsupported; input cooperative tensors are
  single-simdgroup scope only.

## Numerics / quality

### fp8 activation quantization fails the engine bound
e4m3 activations violate the fp64 projection bound by ~10–70× on real
activations at every scale granularity; int8-A fails 2–9×. Mantissa-bound, not
outlier-bound. mxfp4p-style power-of-2 exponent scaling is strictly worse than
absmax. Dead unless the numerics policy relaxes (~50× looser).

### fp8 KV quality ceiling
E4M3 K/V storage caps bf16-cosine at ~0.96–0.98 at long histories regardless of
Q precision (fp16-Q staging verified no improvement — the error is in the
stored codes). Tier stays opt-in (`--kv-format fp8`). Improving it needs a
higher-precision K/V representation, not Q-side changes.

### Chunked GDN prefill — implemented, abandoned
`prefill/gdn_chunked.metal` (WY/UT form, flag-gated, unbound): correct to
bf16 tolerance after two derivation fixes but at best ~1.0–1.15× of serial at
T=2048 — the scan is latency-bound, and C≥128 hits the N>16 operand bug.
Also inherently breaks `gdn_metal_test.mm`'s bitwise split-invariance contract.
Left in-tree for reference; serial scan remains the production path.

### Native MXFP4 prefill slower than staged
`gguf_prefill_mxfp4p`/`gguf_decode_mxfp4p` p-path loses ~11–13% vs the staged
fp16 prefill tile (compute-bound — fp4 operands don't help there). Kept
opt-in under `RICHENGINE_GGUF_PACKED_ON`.

### Packed GGUF decode formats slower
Native operand kernels (q40m/q41m/q4km/q80m/pq20m) lose 2.5–4× when restaging
and ~0–35% behind staged even with pre-packed activations. Kernels kept,
dispatch stays staged; `RICHENGINE_GGUF_PACKED_ON` opts in.

## Draft / model coverage

### Ornith-35B GGUF indistinguishable from Qwen3.6-35B-A3B — mitigated
GGUF metadata carries no `router_aux_loss_coef`; the config alone cannot
separate the families. `family_for`'s `name` hint now resolves it: a
repository/file/`general.name` naming Ornith-1.5-35B-A3B wins when the
config doesn't contradict it. A GGUF mislabeled without any Ornith name
still resolves to Qwen3.6.

### Ornith draft sliding-window depth
Resolved: the ring is `RICHENGINE_DRAFT_SLIDING_WINDOW`=4096 physical slots and
each draft's declared `sliding_window` is its per-model attention horizon
(`DFlashDraftLayout::slidingWindow`, `DraftAttentionBatchParams::window`).
Ornith's 4096-token window is fully served; 2048-window drafts are unchanged.

### `draft_select_plain` q-probabilities
Sampled-lane acceptance uses a top-16 renormalized softmax (same approximation
regime as DFlash2's selector, not the reference's full-vocab softmax).

### Ornith `dynamicSize` / draft geometry
Plain-draft layout has `dynamicSize=0`/`selectorRank=0`; verify the packed
draft files' actual shapes on first real load (converter map was validated
against the released safetensors 1:1).

### Vision on Ornith
`ornithDescriptor` is text-only (`VisionSource::None`); the model is
multimodal upstream but no vision layout is wired. Image requests will fail.

## Build/test environment drift (in-flight at session end)

Resolved since: `runtime-bootstrap` and `linear-plan` pass; `execution-plans`
constants were updated for the verify-rows 8→16 / draft ring 2048→4096
geometry; `model-runtime-oracle` was fixed for the `model.draft` weights
variant; `q8-flash-attention` was fixed for the same 8→16 bump (its prefill
dispatch never ran the second tile — a test-side fix, not a kernel bug).

### fp8/hd512 attention tests exceed 32 KB threadgroup on this device
`fp8-flash-attention` and `hd512-attention` fail at pipeline creation:
`Threadgroup memory size (34384/34308) exceeds the maximum 32768`.
Stale pre-change binaries fail identically — the kernels' scratch
(scores+probabilities+staging) exceeds the device limit, so the tests
cannot run on this machine/SDK. Opt-in-tier kernels; not production
paths.

### backend-benchmark behind draft-geometry change
The LFM2.5-2.6B GGUF decode scenario aborts with "physical width
accounting mismatch" (drafted=15, expected=2). The execution_geometry
contract is fixed — `validateExecutionGeometry` now bounds-checks the
recorded draft block against runtime capacity rather than demanding
equality, so existing packed-q4 packages load again. The GGUF
scenario's width accounting still predates the verify-stride change;
realign the harness before using it for A/B decode runs.

### Qwen3.8-27B decode: proposal-substitution layer cost ~70% of B1
Measured 2026-10-18 (M5 Pro, packed Q4 + incoai DFlash2, decode scenario):
the n-gram predraft gate compared its expected score against a constant
(`ngramDraftExpect=4`), never the GPU draft's measured EWMA, so 3-gram
history chains preempted a draft that lands ~2.6 accepted/step — B1
30 tok/s, B4 ~105. The gate now compares against `proposalAcceptedAvg`
(which now trains unconditionally, excluding n-gram-supplied steps), and
`adaptiveProposals` is maker-disabled for this family — its saved verify
rows cost nothing against a 14.5 GiB pass while its shrunk budget loses
tail tokens (measured ~10 tok/s B1). Result: B1 ~51, B4 ~120, reported
acceptance 36.5% (drafted now counts offered clamped to verifyStride-1,
not the compiled 15). Residual to upstream's published 74/170: the B1
step is ~72 ms of which ~62 ms is `decode_linear_q4*` weight streaming
(~204 GB/s effective); upstream runs GGUF-tile sgmatrix decode kernels
on the same chip. Step time, not acceptance, is now the dominant gap.
Knob A/B 2026-10-18: RICHENGINE_MTL4=1 → 47.7 B1 (worse),
RICHENGINE_SUBMIT_AHEAD=1 → 47.8 B1 (worse); defaults stand. A clean
autotune sweep needs the GPU uncontended — an installed serve holds the
lock and marks every measurement contaminated, so nothing can win.

Clean sweep 2026-10-18 (no contention): kernel pass fitted 7 of 88
Linear plans (+5–27% isolated: Split128×2 on {16640,5120} and
{17408,5120} GateUp at rows=64, N128-112 on {14336,5120}, N256-65 on
{16640,5120}@48, Split128 on {5120,25600}@64 and {6144,5120}@48/64);
env pass kept only VERIFY_TREE=1 (+22% B1 on its harness). On the
packed-model decode bench the plans are neutral (51.2/120.5) and
VERIFY_TREE loses B1 (39.2) while B2 hits the tree lane-divergence
defect (35.5, acc 0.187): the env was dropped from tuning.json with a
note; the fitted plans spec stays and applies at serve.

Decode B1 follow-up 2026-10-18 (M5 Pro, same model/harness): three
changes measured in the real profile (decode-profile, 2 cycles):
- `decode_head_argmax_q4` ran a fixed 32x128 logits tile regardless of
  live rows: ~156 GB/s on the 715 MB head stream. Row-sized variants
  (M8/M16, same per-row math, bitwise-identical argmax) measure
  ~270 GB/s — 4.07 -> 2.66 ms at B1, 4.09 -> 2.64 at B2.
- `decode_linear_q4_n256_gate_up` is now Pipelined (documented
  bit-identical) and the one/two-lane GateUp plan takes an even
  half grid past two threadgroups per core ({17408,5120}: 34 groups
  x 2 tiles, ~267-277 GB/s isolated, vs 68 x 1 at ~255). In-model
  27.5 -> ~26.1 ms over 69 dispatches.
- Ruled out: the affine sgmatrix/Table64 path (decode_linear_q4_sg*)
  on Apple10 — ~2x slower than the mpp tiles on every hot shape at
  rows 8-16 (gate_up 115 GB/s, residual/psum ~2x worse); a four-
  simdgroup fused gate/up kernel hangs the fused command (reverted).
Residual: weight streams run ~235-255 GB/s in-model vs ~265-280
isolated (q4-decode-profile sweep); the fused cycle loses the rest to
non-linear kernels and dispatch serialization (fused == sum of parts,
~0% overlap headroom). Reaching upstream's 74 tok/s on B1 needs the
cycle near ~47 ms — below the 15.6 GB/cycle streaming bound at
307 GB/s — so either acceptance must rise or the weight layout must
change: repacking Affine64 into GGUF tiles (a load-time
AffineToBlock-style pass writing QUANT_TILE_ROWS-interleaved planes)
would unlock the gguf decode tile family, but the affine epilogue
(scale,bias per 64-group) already maps onto them, so the true win
there is unproven on this chip — the sgmatrix result above suggests
the gap is upstream's acceptance/draft accounting, not kernel shape.

GDN_CHUNKED=128 measured 64.7 B1 on the sweep harness but was rejected
on its own regression rule (prefill 334 vs 408).

Conditioning chain verdicts 2026-10-18: verify_attention_bf16_split
(the last unverified kernel in the verify->draft chain) now has fp64
oracle coverage at the production stride - paged-attention-plan's
checkVerify takes a rows parameter and runs rows=8 across
int8/int4/bf16 x lanes 1-4 x histories 0/1023/16384, all PASS; the
kernel is clean at rowCapacity=8. draft_conv row-0 tap-1 (draft.metal
if (row > 0)) is identical to upstream's draft_conv_phase - by design,
not a defect. With the ring commit already exonerated by
RICHENGINE_SKIP_CONTEXT_COMMIT (zero delta), every hop of the
decode-time conditioning chain is now verified clean: the ~36.5%/0.75
conditional acceptance is the DFlash2 checkpoint's behavior on this
pairing, not a runtime defect.

Decode fusion pass 2026-10-18 (M5 Pro, Qwen3.8-27B packed Q4 + DFlash2,
verifyStride=8): two candidate wins on the ~74 ms B1 verify step.

- `decode_linear_q4_n128_split_residual` runs 128x/step: the residual
  epilogue is ALREADY fused — the count is two distinct residual GEMMs
  per layer (mixer out {5120,6144} + FFN down {5120,17408}), each one
  dispatch, not down-proj + a separate add. A "single fused tile" for
  the down-proj was measured and is NOT bandwidth-neutral (q4-decode-
  profile: paired-residual 0.261 ms / sequential residual 0.293 ms vs
  split-residual s4 0.238 ms at 8 rows), so the remaining lever is the
  split count. kMeasuredDecode gains rows for both residual shapes at
  the block-8 verify strides: {5120,17408} Residual @8/16/32 ->
  Split128 s4/s8/s4 and {5120,6144} Residual @8/16/24 -> Split128 s4
  (the wave-fit rule stops at s2 on the 40-tile grid). Installed
  tuning.json plans keep priority (rows 48/64, no residual rows), so
  both mechanisms coexist. In-model (decode-profile, 2 cycles):
  split_residual 18.83 -> 18.56 ms B1, 21.66 -> 20.54 B2, ~flat B3/B4;
  backend-benchmark decode: B1 49.1 -> 50.9 gpu tok/s, B4 114.9 ->
  116.0, output_token_hash 5877947751290141417 identical at every
  width, performance_pass, no lane divergence.
- `verify_gdn_commit` (~1.2 ms, one dispatch) into `verify_gdn_fused`
  is RULED OUT by ordering: the fused kernel encodes inside the target
  forward (Runtime.mm encodeTargetVerifyBatchForward / the constrained
  ticket's `target` graph) while `decode_accept_dflash` writes
  RetainedCount later in the same command buffer — and, on the
  constrained path, in a different graph submitted after the forward
  completes. The commit's replay of `retained` rows needs the count at
  encode-ordered execution time; the fused kernel would read the
  previous step's value. The current separate dispatch is correctly
  ordered after acceptance (same encoder -> sees earlier outputs). A
  merge would have to target a post-acceptance kernel (the draft state
  commit chain), not the fused mixer.

Re-tune 2026-10-18 (post-fusion, clean GPU): kernel pass changed only 2
of 88 plans - the new kMeasuredDecode rows already cover the hot
shapes. Env pass kept RICHENGINE_VERIFY_TREE=0 (flipped from the
earlier sweep's =1: the fused-head disable now outweighs the tree
gain post-fusion, and pinning it off also blocks the known B2
lane-acceptance defect at serve). Best env B1 38.25 / B4 62.2 in the
sweep harness; tuning.json updated (VERIFY_TREE=0 + 2 fitted plans).

## LFM3 width-2 lane divergence (bisected, unfixed)

Symptom: `backend-benchmark --scenario decode` fails at width=2 — lane-id=5
diverges from lane-id=4 at token index 1 (first verify output). Width=1 and
longdecode (single-lane) pass. All LFM3 files untracked WIP.

Ruled out: fused headgate reduce (fails under RICHENGINE_VERIFY_TREE=1);
headgate gate-stride (attentionGateRows=256 matches kernel's *256u); conv
state subspan binding (RuntimeImpl.hpp:1740); extension plan indexing.

Remaining suspects, ranked:
1. b.chunkKeys/chunkValues lane-1 row stride under LFM3 dual-geometry
   (layerKvLayout(global) vs pool layer order — 5 SWA + 6 global interleaved
   across 11 kvLayers extents)
2. Per-lane KV page binding — step.kvLayers extents vs bindPageTables
3. states.current/next parity (prefillUnapplied swap) for second sequence

Next step: instrument first-layer chunkKeys dump lane 4 vs 5 — mismatch
implicates (1); match implicates (2)/(3).

Update: d1-3B (lfm2 family) PASSES decode at widths 1-4 — identical
output_token_hash per lane. Batch machinery is sound; bug is LFM3-specific.

RICHENGINE_VERIFY_DUMP2 (new instrumentation, RuntimeImpl.hpp ~2079): at the
first width-2 verify, lane 1's hidden0row0 already differs from lane 0's —
the carried hidden INTO verify is wrong for lanes >=1, so the fault is
upstream of the layer loop: the per-lane hidden carry
(CapturedTargetHidden/FinalHidden -> Hidden0 row 0), a path LFM3 uniquely
exercises via capturesHidden()=true (lfm2 does not capture target hidden).
Check captureFinalHidden's per-lane stride and the prefill-end RowCopy
(decodeArena->get(sequence.lane, Hidden0)) for lane>=1.

Update 2: capturesHidden() is Qwen4Exp-only — LFM3 uses false. Corrected
hypothesis: lane 0's hidden0row0 differs between width-1 (c08bd999) and
width-2 (4eefe5f2) first verifies — the carry into verify is
WIDTH-DEPENDENT, and lane 1's differs further. So the divergence source is
BATCHED PREFILL (exact only tested single sequences): sequence 1's packed
rows land or read differently — check LFM3 prefill per-sequence offsets
(rowBegin/attentionStride/queryOffset/kvOffset) vs the shared packed
buffers, especially conv out (gdnHidden) + gates (denseIntermediate at
rowBegin*attentionGateRows) vs what the projections assume.

Update 3 (dump results): lastFinalHidden_ instrumentation proves seq1's
prefill last row hashes IDENTICALLY to seq0's (738e7552c0311141) — batched
prefill is CORRECT. RICHENGINE_NO_DRAFT=1 still diverges (same signature)
— proposals/predraft are not the cause either. The carry content is right;
the wrongness is in the verify input path for lane >=1:
  a) decodeArena->get(sequence.lane, Hidden0) RowCopy destination vs the
     verify's packed() lane indexing, or
  b) the verify input-row embedding writes for lane >=1 (hidden0row0
     differs at first verify and is NOT overwritten by an identical anchor
     embedding — so it still holds the (wrong) carry).
Next: dump decodeArena->get(lane,Hidden0) row0 immediately post-copy, or
diff TargetModelLfm3's verify input-row assembly vs TargetModel.cpp's.

Update 4 (control run): d1-3B dump shows lanes' tensors ALSO differ
internally despite passing — per-lane buffer differences are NORMAL
(replayed requests use different pages/state slots). The "everything
differs" signal was confounded. What stands: divergence is inside the
verify step for lane>=1; prefill (batched AND cached) is exonerated by
identical finalHidden seq-row hashes; every env toggle fails (tree, m8,
fused head, no-draft, bf16).

Sharpest remaining hypothesis: an LFM3-specific lane-dependent read — the
verify gate buffer (gates=b.denseIntermediate packed [lane][row][256],
kernel reads stride 256u — verified consistent) OR the alt-rope/KV-layer
binding for global layers under dual geometry, OR a padding-lane write
colliding with the last active lane (padding replays the last request).
Next instrument: sync + hash hidden0/gdnPacked0 per LAYER (needs a debug
build with a post-layer barrier, or RICHENGINE_LAYER_LIMIT-style flag to
stop after N layers and dump).

Update 5 (LAYER_LIMIT bisect — new env flag added to TargetModel.cpp verify
loop, skips requireLayerPartition when set):
  N=1,3 PASS (identical output_token_hash at width 2)
  N=4,5,6,12 FAIL
=> divergence enters at layer index 3 — a CONV layer (types: conv@0,1,3,5,
global-attn@2,6,..., swa@4,8,...). LfmConv is shared w/ LFM2 (works) so the
bug is taps=7-specific or LFM3's conv wiring: state FIFO taps-1=6 rows,
convolutionLayerBytes, convolutionTapsMajor flag, or bx-row commit layout.

Update 6: gdnStateLayout is correct (convolutionHistory=taps-1=6, layers=13,
channels=2048). Layer-3 conv (conv index 2) is the first diverging layer —
convs 0,1 + global-attn layer 2 all pass. Since convs share one code path,
the differentiator is STATE layer index 2 or what touches it: the conv FIFO
written by prefill for state layer 2, or the gdnMixed/gdnPacked[2] slice.
Likely next step: dump lane states' conv-layer-2 FIFO contents vs lane 0's,
or bisect the 13 conv layers' state writes in prefill (pack begins could
overlap for seq>=2 if rowBegin*convolutionDimension striding is wrong).

Update 7 (root cause pinned): corrected LAYER_LIMIT head-input parity
(head now reads hidden[layerLimit&1] when truncated). Clean bisect:
  LIMIT=2 PASS (convs 0,1 fine) — LIMIT=3 FAIL
=> the FIRST GLOBAL ATTENTION layer (model index 2, attentionLayer=0)
produces lane-dependent output in verify.
DUMP2 at LIMIT=3 shows lane 1's fullAttention = 83f7c309b5b50383 — the
EMPTY-buffer hash (same value as d1-3B idle lanes). Lane 0's has content.
=> lane>=1's attention reduce for the global/altAttention layer writes
nothing (or to the wrong slice). Suspects, in order:
  a) ext.altAttention plan (verifyAttention w/ layerKvLayout(true)):
     its per-lane split dispatch or extent binding for lanes>=1;
  b) step.kvLayers[0] extent mismapped for the global layer;
  c) the _k4q4d128 split/reduce kernels' batch/group.z indexing on the
     alt plan;
  d) the headgate gates path reading a wrong-lane slice.
Convs 0-1, the head-gate path on SWA/other attention, draft, fused head,
junctions, formats, positions — all ruled out.

Update 8 (decisive): fixed the head-input parity in LAYER_LIMIT (head reads
hidden[layerLimit&1]). Clean bisect: LIMIT=1 fails too — hidden0 (the
verify STEP INPUT: embedded tokens + carry) already differs for lane>=1
BEFORE any layer runs. finalHidden seq rows were proven identical, so the
carry source is fine — the bug is in the verify input path for lanes>=1:
  a) the input TOKEN rows (addVerifyInput's merged anchor+proposals, or
     the deviceVerifyParams step-pack path), or
  b) decodeArena->get(lane, Hidden0) carry copy vs the embedding's packed
     indexing for lanes>=1.
The lane-dependence enters at the input embedding/assembly, NOT a layer.
Next: dump the real InputTokens ids (u32, NOT the lane-index vector I
mislabeled — find the actual token-id buffer feeding encodeBatchEmbedding)
per lane; diff addVerifyInput's per-lane writes for lanes>=1.

Update 9: PARTIAL FIX + reframed bug.
FIXED: encodeBatchVerifyInput now stages draft_input anchors host-side
(*entries[lane]->pendingToken) — draft_predraft_pack only populates it
under chained predraft, so LFM3's unproven lanes read anchor=0. After the
fix, measured steps show IDENTICAL lanes at LIMIT=1 (anchors 333/333,
zero tensor diffs). Warmup lane diffs were legitimate (different reqs).
Step-pack evidence: proven systematically fails (pack.base=pos+1) so every
lane takes the staged fallback — the staged anchor is now correct.

REMAINING bug (cleaner): with identical inputs, divergence enters at layer
index 2 = FIRST GLOBAL attention layer. LIMIT=3 dump (step18): fullPacked,
fullQueries, chunkKeys0/0, hidden0 all IDENTICAL — fullAttention DIFFERS
=> lane>=1's global-layer attention reads different HISTORY: its KV pages
contain different content (batched prefill store for seq>=1 on the ALT
extent) or its page table maps to wrong pages. pt sizes equal (pt=1/3).
Suspects: addPrefillStore per-sequence indexing on the alt extent
(step.kvLayers[attentionLayer], sequence.chunk.kv, the pool's
dual-geometry page-table block ordering).

Update 10: pool anatomy resolved — each physical KV page (32 tok) holds ALL
attention layers' slices for one block; RichKvLayer.offset = the layer's
byte offset inside each page (dual-geometry regions pack back to back).
=> lane>=1's divergent fullAttention means ITS pages hold different bytes
at the global-layer offset: the batched prefill addPrefillStore wrote
wrong/misaligned K/V for seq>=1 (queryOffset/kvOffset/attentionStride
accumulation, or the alt layer's store layout), OR seq1's pages were
never written (a skipped store for the second sequence's alt layer).
Next: hash one global-layer page's bytes for lane 0 vs lane 1 post-prefill,
or check addPrefillStore's per-sequence chunk indexing for the alt extent.
Also worth checking: the store kernel gets sequence.chunk built from
item.logicalPosition — if for seq>=1 the chunk's committed/stride fields
were computed against the WRONG item fields, its writes land shifted.

Update 11: synchronizedPageTable defined in another TU (search Runtime*.mm
line 320 comment). Page tables verified same size/sequential page ids.
Pool anatomy: extent=384 pages; each page holds ALL layers' slices at
kv.offset byte regions. Store is per-sequence, per-layer views verified
consistent for LFM3 (kvHeads=altKvHeads=4, dim=128).
UNVERIFIED per-lane items: synchronizedPageTable revision-skip could leave
a stateLane's GPU table stale on request reuse; and seq>=1's prefill
chunk K/V CONTENT (verify chunk keys matched; prefill's were never hashed).
Fastest next: dump prefill b.chunkKeys per-seq FNV, or read one physical
page's bytes at the alt offset for lane 0 vs lane 1 after batched prefill.

Update 12: FULL INPUT SYMMETRY PROVEN — divergence is inside kernel
execution or a read the dumps can't see.
Verified identical for both lanes at the diverging step (LIMIT=3 dump):
- Page CONTENT: hashed spans[0..3] (keys, kscale, values, vscale) of every
  block — lane0 blk0/page0 == lane1 blk0/page0; lane0 blk1/page1 ==
  lane1 blk1/page2 (identical hashes, different physical pages — pool
  dedups only block 0).
- GPU table entries: lane1's stateLane table holds correct translated
  entries ([...00 ...02 ...04] for its pages).
- Verify attention params (addVerify dump): committed/pte/splits/slots/
  active/cap/kv identical per lane.
- Queries, chunk K/V, hidden0, input tokens: all identical.
- Stride math checked: FullQueries/FullAttention 131072B/lane,
  AttentionHidden 65536B/lane, packed_qkv row_capacity layout — all
  consistent with kernel offsets.
YET: partials/statistics/fullAttention DIFFER for lane>=1 at the first
diverging verify step => the k4q4d128 verify split reads SOMETHING
per-lane that differs, OR there's a race/nondeterminism.
Notable: stateLanes don't map to batch lanes (lane0->sl2, lane1->sl0) —
bindPageTables resolves them correctly as separate buffer args, but any
kernel path that treats batch index as a table index would break here.
NEXT: (a) hash the partials region per (kv_head,split) slot to see WHICH
slots diverge; (b) check richengine_paged_attention_tile's reads of
page_table/kv.offset for dual-geometry offsets (kv.offset=0 for global —
maybe the issue is the OTHER tensors: hidden0 row for lane1 may reuse
stateLane'd carry — decodeArena->get(lane,...) vs get(stateLane,...)
mismatch in the carry copy!

Update 13: DECISIVE — lane>=1's verify attention SPLITS never write
(partialNonzero=0 for lanes 1,2,3 vs ~4088 for lane 0, in 4KiB probe at
lane offset sizeBytes/nlane). Confirmed on BOTH global (_r2 AND non-r2)
and SWA layers => tile.active==false for batch>=1 in EVERY verify split:
contract_valid fail / kv_head / split bounds — OR the params payload the
KERNEL reads differs from host-set attention[] (params[batch] indexing),
OR DispatchSize z-axis semantics mismatch.
All host-side inputs verified identical (pages content, GPU table entries,
params fields, queries, chunk K/V, strides, plan groups {8,8,lanes}).
CAVEAT: nz probe assumes slot regions are lane-contiguous — if
kVerifyMaximumSplits > plan.splits the probe region may miss lane1's
actual slots; verify by hashing slots (batch*KV+kv)*slot_splits+split
32-63 directly. If genuinely unwritten, the remaining suspect is the
constant-buffer params payload: attention std::array maxLanes=4 — check
graph.addPatchable serializes all 4 params vs only plan.lanes, and that
params[batch] inside the kernel sees attention[1] correctly.
Also: LIMIT=6 shows SAME nz=0 — every attention layer skips lane>=1,
so the bisect "first diverges at layer 2" is just the first attention layer.

Update 14: nz=0 was a FALSE reading — slot indexing is flat
((batch*KV+kv)*slot_splits+split, maxSplits=128 workspace), not
lane-major; probing the correct flat offsets shows lane>=1 slots DO get
written (slotNz ~4088 for all lanes). So splits execute for every lane
and produce different partial VALUES for lane>=1 — the read side
(queries offset, page table, params) still diverges somewhere.
fullAttention DIFF remains real (verified 131072B/lane layout).
Every host-side input verified identical => likely remaining suspects:
(1) packed_qkv read in the gate reduce at (batch*row_capacity+row)*5120
    — if plan.rowCapacity != verifyRows(16), lane>=1 gates misalign;
(2) a Metal-level nondeterminism/race in the shared partials workspace;
(3) the page_tableN buffer args arriving in a different order than
    bindPageTables intends for lanes whose stateLane != batch lane.

Update 15: per-slot stats hashes mirror partials — only DEAD splits
(2,3 of each kv_head, covering empty page ranges for a 2-page history)
differ: lane0 holds stale leftovers, lane1 zeros. All ACTIVE slots
(0,1 per kv) identical in partials AND statistics => reduce's effective
inputs match. fullAttention DIFF at LIMIT=3 may be scratch artifact —
the stale dead-slot stats in lane0 are neutral iff max=-inf.
NEXT (decisive): dump raw stat VALUES for slot2 lane0 — if max is -inf,
dead slots are weight-neutral and the real diff is elsewhere (the
gate/headgate reduce's packed_qkv read or AttentionHidden write); if
finite, dead slots inject stale weight and THAT is the bug — the
splits should zero dead-slot statistics (workspace not zeroed between
steps => stale data leaks into the reduce for lanes whose slots
previously held longer-history partials).

Update 16: RICHENGINE_NO_ALT_LAYERS still diverges (71 vs 207) => NOT the
alt-geometry path — the fault is in the GENERIC batched verify for
lane>=1, not LFM3's dual geometry specifically. The "limit=3 diverges at
layer 2" bisect is just "first attention layer". Every host-side input
verified identical for lanes 0/1 — pages, tables, params, queries,
chunks, strides — yet the splits produce different partial/statistics
VALUES for lane>=1 (correct-slot probes confirm all slots written).
=> inside-kernel: either (a) a Metal-level nondeterminism in the shared
partials workspace, or (b) a params[batch] constant-buffer read that
differs from the host-set attention[] values. The next instrument should
be a debug kernel that reads back params[batch] fields or per-slot stats
values for lanes 0 vs 1 at the same step, or Metal-side printf.

Update 17: PATCHABLE_OFF ruled out too. FINAL verified state: ALL
host-side inputs provably identical for lanes 0/1 (pages per-block
per-tensor-span, GPU table entries, every params field, queries, chunk
K/V, rowCapacity=16, all strides, stateLane bindings, struct ABI 44B).
Per-slot probes: all 32 partial slots written for both lanes; active
slots (0,1/kv) identical in partials AND statistics. Yet
fullAttention/attentionHidden/hidden1 still differ for lane>=1 at the
first attention layer => the divergence is INSIDE kernel execution with
identical inputs — Metal-level nondeterminism (threadgroup rescale /
previous_scale across split groups) or a read invisible to host dumps.
REMAINING INSTRUMENT: Metal printf in richengine_paged_attention_tile /
the reduce for batch>=1, or a debug kernel writing params[batch] fields
back to a readable buffer to prove/disprove constant-array indexing.

Update 18: fullAttention raw bytes — lane1 faNz=64 REAL VALUES (3c90
bcb2...) not zeros — the earlier "empty hash" was a stale-buffer
artifact. So: splits write identical active slots, reduce writes real
output for BOTH lanes, but lane>=1's OUTPUT DIFFERS => the reduce reads
DIFFERENT params/partial bytes than the host dumps see. Since partials
slots 0/1 hash-identical and params fields identical host-side, the
remaining explanation: params[batch] payload in the KERNEL differs from
attention[batch] host-side (serialization index/binding) OR an
uninstrumented buffer arg (page_tableN binding order, packed_qkv offset,
statistics stride) differs per lane.
NEXT: Metal printf or a write-back debug kernel dumping params[batch]
for batch=0,1 — the only instrument left.

Update 19: KERNEL-SIDE READBACK PROVES SPLIT DIVERGES. Headgate reduce
writes params[batch] fields into hidden — identical for both lanes
(committed=39 active=16 splits=8 slots=8 cap=16 pte=2 ep=384 koff=0) —
serialization correct. Readback of partials[slot] via hidden: lane1's
slot-33 bytes DIFFER from lane0's slot-1 (bf16 -0.0074 vs 0.0108) =>
the SPLIT kernel writes different partials for lane>=1 even with
identical queries/params/page-content => inside-split-kernel divergence.
The remaining per-lane read inside richengine_paged_attention_tile:
tile.page_table (=page_table1, stateLane-mapped) or tile.queries
offset — the entries resolved IDENTICAL host-side, so either
(a) page_table1's CONTENT at kernel time differs (mutation between
encode/dump? unlikely), or (b) nondeterministic split behavior.
NEXT: have the SPLIT kernel itself write back page_table[0..1] and
tile.queries[0..3] per batch into a debug buffer — the split's actual
reads, not the host's view of them.

Update 20: SPLIT KERNEL READBACK — tile.queries[0..3] and
tile.page_table[0..3] IDENTICAL for lanes 0/1 (the split's OWN view).
params[batch] identical too. Yet partials[slot] differs for lane>=1.
=> the divergence is inside richengine_paged_attention_tile's inner
loop (QK/PV matmul, online softmax, rescale atomics) — identical
inputs, different output => nondeterminism or a batch-dependent code
path invisible to dumps. The remaining instrument is page-level
readback inside the tile loop (key_scales/values addressing) or
comparing the tile's per-page scores.

Update 21: SPLIT KERNEL-VERIFIED INPUTS IDENTICAL. tile.queries,
params[batch] (in-kernel readback), and page_table entries all proven
identical for lanes 0/1. Lane1's table maps blocks to pages {0,2,11}
vs lane0's {0,1,2,3} — different physical pages, identical content.
=> the divergence is inside richengine_paged_attention_tile's inner
loop with provably identical inputs => NONDETERMINISM. The code itself
flags the candidate: the probabilities ping-pong exists because
"whether a tensorop's operand reads are fenced by threadgroup_barrier
the way plain loads are is unproven" — a lingering PV tensorop read
colliding with the next softmax write = exactly this signature:
identical inputs, lane-dependent divergent partials.
NEXT: force single-page-per-split (or serialize the page loop with an
extra barrier between softmax-write and PV-read) to test the race —
if lanes converge, the ping-pong workaround is insufficient and the
fix is a stronger fence or a second probabilities buffer per page.

Update 22: ROOT CAUSE FOUND. Tile-loop readback writes back the actual
key bytes + page_table entry the split reads: lane1's page_table[0]
points to physical page 2 whose key bytes (254,254,238,0) differ from
lane0's page0 keys (7,249,43,232) — lane1's block-0 K/V region holds
wrong/uninitialized content. => The batched PREFILL store for sequence
>=1 wrote its K/V to a different page than the verify table later maps
(or never wrote it) — a store/table DESYNC for batched sequences, NOT
a verify-kernel bug. Lane1's item.pageTable is legal (its own physical
pages) but the page CONTENT at its block0 is garbage => seq>=1's
addPrefillStore used a stale/different table or wrong offset.
NEXT: compare sequence.pageTable at PREFILL-store time vs the verify
table — if the store ran before the final table mapping landed (or a
partial writeEntries left the GPU table mismatched), the store's
page_table[0] != verify's page_table[0].

Update 23: Mechanism confirmed — sequence.pageTable and the verify
table are the SAME decodeArena->get(stateLane, PageTable) buffer (via
synchronizedPageTable at both paths). The desync: lane1's verify
page_table[0] resolved to physical page 2 holding garbage keys, so the
entry at index 0 was NOT what seq1's store wrote through — either the
request's logical->physical map changed between prefill and verify
(block remap on a re-prefill / prefix-hit path), or another request
synced its table into the same stateLane slot in between, or the
store dispatched before synchronizedPageTable rewrote the buffer for
this request (revision-skip: binding.revision == item.revision ->
first = size -> SKIP even though the BUFFER contents were clobbered
by a different request's write on the same stateLane).
NEXT: dump the stateLane table contents at prefill-store AND at
verify for the same request id — a third party's write between them
shows as differing entries with requestId unchanged.

Update 24: BUG NARROWED TO PAGE-TABLE BINDING. Kernel readback: lane1's
split sees page_table[0] low16=2 (...0002). Host readback of
get(stateLane=0, PageTable) entry 0 = ...006. item.pageTable[0]=6.
=> the kernel's page_table1 ARG ISN'T the stateLane-0 buffer — it's a
DIFFERENT request's table (or wrong offset). bindPageTables fills
pageTables[lane]=get(laneEntry(entries,lane).stateLane, PageTable) —
batch lane 1 must bind request-5's (sl=0) table but the kernel reads a
table whose entry0 maps page 2 => bindings array is shifted/misaligned
for lanes >=1, OR laneEntry picks a different entry than expected
(padded-lane handling binding the WRONG request's table).
NEXT: print the actual pageTables[] buffer addresses/contents in
bindPageTables for a width-2 batch — page_table1 should equal
get(sl=0)'s data; if it's another stateLane's table, the binding array
order/stateLane mapping is the fix.

Update 25: MEASURED LANE MAPPING — lane0->stateLane0, lane1->stateLane1
(warmup used {2,0,3,1}). get(1) holds entries {0,2,4} per gpt dump but
the KERNEL's page_table1[0] read ...02 (page 2), not ...00 => EITHER
tables[] arg order is misbound OR stateLane-1's buffer held STALE
entries at kernel-read time (sync happens in synchronizedPageTable —
RuntimeEncodeDraft.mm:15 commit path and Runtime.mm:1598 prefill —
VERIFY/decode path may read get(stateLane) BEFORE its table is synced
for the current request => stale table from a recycled stateLane).

Update 26: CONSOLIDATED — the verify-time kernel readback proves seq>=1's
KV pages hold wrong content. Chain of certainty: (1) params/queries/
table entries identical in-kernel; (2) lane1's page_table[0] resolves
to its own mapped page (alloc varies per run: 2 this run) but the page's
key bytes are garbage (254,254,238,0) vs lane0's real content
(7,249,43,232); (3) the table arg is the shared
get(stateLane, PageTable) used by BOTH the prefill store and the verify
split => same entries => the DESYNC IS IN THE PAGE CONTENT, not the
mapping => seq>=1's addPrefillStore wrote wrong/no K/V to its pages.
Two candidates: (a) the store's chunk params (committed_tokens /
chunk_stride / page_table_entries) wrong for the second sequence in the
packed batch; (b) the store dispatch for seq>=1 read keys/values at a
wrong queryOffset/kvOffset (PackedPrefillSequence per-sequence offsets).
NEXT: hash the keys/values views + print sequence.chunk for seq0 vs
seq1 in encodeBatchStore — seq1's keys view offset or chunk fields
will differ wrongly.

Update 27: SHIFT CONFIRMED — seq1's host map is pt=[0 2] (block0->page0
shared, block1->page2), chunk params/offsets all correct. But the
VERIFY kernel's page_table[0] reads ...02 — the map's INDEX-1 entry at
index 0 => the GPU table at get(stateLane=1, PageTable) is SHIFTED by
one entry, or the buffer was left with a stale entry at index 0 that a
partial sync skipped (binding.firstChanged=1 -> writeEntries writes
only index>=1 => index 0 keeps stale content if the first sync's
full-write was skipped via requestId/revision match on a RECYCLED
stateLane's binding). writeEntries itself indexes correctly.
NEXT: check pageTableBindings_[stateLane] requestId/revision tracking —
if a recycled stateLane's binding makes the first sync for the new
request take the partial path (first>0), index 0 stays stale.
Alternatively bindPageTables' entries[] ordering vs batch lanes.

Update 28: TABLE BINDING PROVEN CORRECT — sync dump: lane1 id=5 sl=1
bind=[5 r4] rev=4 -> first=size SKIP, e0=...00 e1=...02 == pt=[0 2].
The GPU table at verify time holds exactly the host map. => The desync
is TEMPORAL: dbg3's run showed lane1's page_table[0]=...02 with garbage
keys => the physical map REMAPPED (dedup/compact or revision bump)
between the prefill STORE (which wrote block0's K/V through the OLD
mapping) and the verify SPLIT (which reads via the NEW mapping) => the
new page was never written => garbage keys => divergence.
NEXT: find the remap source — grep pageTableRevision bumps / where the
request's page list is reassigned (prefix-hit junction restore,
dedup share-page remap, or chunk-commit page reuse). The store and the
read must use ONE mapping — either freeze the map between store and
verify or rewrite content on remap.

Update 29: THE REMAP SOURCE IS PAGE COMPACTION. KvPool::compactExtent
(KvPool.cpp:168) emits KvPageMoves{firstPage, destinations[]}; Cache::
compactExtent (Cache.cpp:700-719) calls kv_.followPages(moves) which
rewrites entry.page (KvCache.cpp:406) AND updates each request's page
list via moves.follow(page). Invoked under eviction pressure
(Cache.cpp:432, 695). THE BUG MECHANISM: compaction runs BETWEEN
seq>=1's prefill STORE (writes K/V via old page ids) and the verify
SPLIT (reads via new ids) — if the GPU table sync marks the remap via
firstChanged (partial) but the CONTENT copy (pool move) lands on a
different page or races the verify read, the new page holds garbage =>
exactly the observed stale/divergent K/V for lanes>=1.
CHECK: does moves.follow rewrite item.pageTable AND bump
pageTableRevision + set pageTableFirstChanged correctly? If the GPU
table resync skips (revision match) while pages moved, the kernel
reads the OLD table entries -> wrong pages. Also verify the content
copy (copy.from/copy.to at KvPool.cpp:225) is a real GPU memcpy before
the next encode.

Update 30: ROOT CAUSE — DANGLING SPAN. item.pageTable is a
std::span<const uint32_t> view into the per-request
Cache::Active::pages vector (Cache.hpp:446). The vector grows via
insert (Cache.cpp:395, page acquire on block boundary) => realloc =>
the span DANGLES. When a decode/verify item's pageTable view was
captured before an intervening acquire grew the same request's pages
vector, synchronizedPageTable/writeEntries read FREED memory ->
garbage entries written into the GPU table -> the kernel reads wrong
physical pages -> wrong history -> lane-dependent divergence
(nondeterministic-looking because the realloc address/content varies
per lane's allocation history). Matches every symptom: identical
host-verified inputs, inside-kernel-identical params/queries, but
lane>=1's pages hold garbage keys.
FIX: item.pageTable must either snapshot the pages (copy) or the
cache must expose a stable storage (deque/fixed array) — OR the sync
must re-fetch the CURRENT pages at write time rather than trusting
the captured span. Cleanest: PageTableView holds a copy, or
synchronizedPageTable re-reads cache_.pageTable(id).pages at encode.

Update 31: item.pageTable -> std::vector (owning copy) + span-snapshot
at assign — divergence PERSISTS, so the dangling span wasn't the
active fault (kept as defensive hardening). Kernel still reads
page_table1[0] = ...02 while get(sl=1) holds ...00 at index0 =>
page_table1 resolves to a DIFFERENT BUFFER than stateLane-1's table.
entries[lane] indexing verified consistent everywhere. Remaining:
(1) bindPageTables binds laneEntry(entries,lane).stateLane — if
entries[] order vs items[] order differs (padded lanes, request-pool
order vs batch-lane order), page_table1 gets the wrong lane's buffer;
(2) get(sl, PageTable)'s per-stateLane slicing. NEXT: print the
MetalBuffer handle of pageTables[1] AND get(1,PageTable) — mismatch
confirms misbinding; equality means the buffer content itself changed
between sync and kernel read.

Update 32: BINDING PROVEN IDENTICAL — pageTables[1].contents() ==
get(sl=1).contents() == 0x161dc0008, e0=...00 (correct map). The GPU
buffer at lane1's arg holds the right entries at index0. Yet the
KERNEL's page_table1[0] read ...02 => tile.page_table[0] isn't the
bound buffer's index0 => EITHER tile.page_table has an offset OR the
kernel's buffer(4) arg isn't b.pageTables[1]. tile.page_table =
batch==1?page_table1 (direct, no offset). => The split dispatch's
splitBuffers {queries,partials,stats,tables[0..3]} -> buffer(4)=
tables[1]: tables must be b.pageTables — if a DIFFERENT array is
passed (e.g. the INDEX tables or an unbound default), buffer(4) gets
the wrong buffer. NEXT: grep the splitBuffers construction in
addVerify — confirm tables[1] is b.pageTables[1] not indexTables or
a stale capture.

Update 33: READBACK WAS AMBIGUOUS — the tile debug wrote for EVERY
kv_head==0 group.y==0 group (splits 0 AND 1); split-1's
page_table[1]=...02 overwrote split-0's page_table[0]=...00 in the
shared dbg slot => the "...02 at index0" was page_table[1], not a
shifted table. tables[1]==b.pageTables[1]==get(sl=1) verified
correct end-to-end. => The real desync is THE PAGE CONTENT: lane1's
page_table[1]=...02 -> page 2's keys are garbage (254,254,238,0) =>
seq1's SECOND prefill chunk (committed=32, 7 tokens -> block1 ->
page_table[1]=page2) never wrote its K/V there => the store for
seq>=1's LATER chunk used a stale page_table[1] or the store kernel's
committed/block math wrote to a different page.
NEXT: dump the store kernel's chunk fields + page_table[1] AT STORE
TIME for seq1's second chunk — the entry it reads vs the verify's.

Update 34: STORE TABLE CORRECT AT STORE TIME — seq1's second chunk
store: c=32 t=7 s=32 pte=2 pt0=...00 pt1=...02 — reads page_table[1]
=page2, identical to the verify's read => table consistent across
store+verify => the content desync is either (a) the store's
keys/values VIEW (sequence.kvOffset=32768 bytes into ChunkKeys) reads
the wrong unit/region so the store copies wrong data, or (b) page 2's
content was clobbered post-store (eviction/reuse — a page released
while still mapped by the block). Check backend.view() offset units
(elements vs bytes) for chunkKeys, and whether PageStorage releases/
reuses a page that a live block still maps.

Update 35: backend.view is byte-offset — kvOffset correct. Store table
proven correct at store time (pt1=...02 == verify's). => the content
desync is UPSTREAM of the store: seq1's keys/values VIEW
(view(chunkKeys, 32768, kvBytes)) content itself — either
addPrefillProjection wrote seq1's projected K/V into a different
region (view/stride bug for seq>=1 in the packed batch) or wrote
nothing -> the store faithfully copies STALE bytes to page 2.
NEXT: hash keys/values view bytes for seq0 vs seq1 right before
addPrefillStore — if seq1's view is stale/zeroed, the projection's
per-sequence offset (sequence.kvOffset into the packed ChunkKeys
buffer vs the projection's write stride) is the bug.

Update 35b: NOTE — the "garbage keys" readback was the same shared-
debug-slot artifact: multiple split groups raced on lane's single dbg
slot; last-writer wins => the 254,254,238,0 bytes vs 7,249,43,232
comparison is INVALID (mixed split writes, not page content). Block
hashes at proper sync showed page content IDENTICAL (blk0/blk1 SAME).
=> ALL kernel inputs verified identical yet output differs => the
divergence is genuinely inside the split/reduce compute or a residual
nondeterministic read. No further static leads; remaining options:
Metal-side per-page readback in a NON-shared debug buffer (per
batch+split+kv slot keyed writes, not one slot per lane), or
two-process comparison dumping every group's (batch,split,kv) ->
first partials row into distinct slots.

Update 36: V LAYOUT CORRECTION — attention_qkv_prepare.h stores
chunk_values DIM-MAJOR: value_offset=(head*D+dim)*stride+token, while
chunk_keys is token-major: (head*stride+token)*D. The cklive hash was
valid (K rows 0-6 identical) but cvlive hashed the wrong layout => the
"seq1 V differs" reading was an artifact — V is probably identical.
=> EVERYTHING re-verified identical: K,V,Q, params, tables, pages,
gates. Yet step73 (first measured step) shows hidden0 DIFFERENT =>
the finalHidden carry written by the BATCHED PREFILL differs for
seq1 => divergence is born inside the batched prefill compute itself.
NEXT: add DecodeTensor::FinalHidden to the verify laneHash set AND
hash hidden0 during PREFILL (per-sequence hidden row regions at
rowBegin) — localize whether the first differing tensor is conv
output, attention, or residual inside the packed prefill.

Update 37: FENCES TESTED — simdgroup_barrier after page_scores.store()
AND before pv.run() — still diverges (490 vs 21). Tensorop fencing not
the active race. bf16 format also diverges => not the quantized path.
REALIZATION: the dump's `gates` laneHash used packed(DenseIntermediate)
per-lane stride = rows*7680*2 = 245760B, but the gate projection packs
lane gate rows at row_capacity*256*2 = 8192B => the "gates differ"
finding was another hashing artifact. Correction pending: hash gates
at the 8192B lane stride. Also note packed() strides verified correct
for FullQueries(131072), AttentionPartials, Statistics, Attention-
Hidden(65536? verify), FullPacked(98304).

Update 38: DECISIVE — corrected gates256 hash (lane stride 8192B =
row_capacity*256*2) shows at the FIRST measured step (hidden0 SAME):
lane1's gate-projection output DIFFERS => the batched linear.add for
mixer.gateProjection produces different output for lane>=1's rows
from identical normalized input => divergence = value*sigmoid(gate)
in the fused headgate reduce => hidden1/output differs => token1
differs. Suspect: decodePlan(mixer.gateProjection, step.planLanes,
None, nullptr, step.rowCapacity) — verifyRows param is "the lanes'
packed row stride" — if the plan packs lane>=1's gate rows at a
different offset than batch*row_capacity*256 assumed by the headgate
kernel, lane1 reads stale gate values. ALSO possible: the linear
itself computes wrong values for row>=rowCapacity (lane>=1's rows).
NEXT: dump the gate rows' raw bf16 for lanes 0/1 (rows 0-3 each) to
see whether lane1's gates are stale/garbage (never written) vs valid
but different (wrong input row read) — distinguishes write-offset
bug from input-read bug in the batched linear.

Update 39: NORMALIZED DIFFERS, HIDDEN0 SAME — at step18 (first measured
step): hidden0 identical, Normalized DIFFERENT, gates different
(valid bf16, not stale), attentionHidden different. Chain: layer2's
inputNorm output for lane1's rows differs => the layer-2 INPUT (post-
conv hidden) differs for seq1 => the CONV/GDN layers (0,1) produced a
slightly different hidden for seq1 — small enough that LIMIT=2 still
argmax'd to the same tokens, but shifting normalized->gates->attention
enough to flip token 1 at LIMIT=3. => FAULT IS IN THE BATCHED CONV/GDN
PATH for seq>=1: per-stateLane conv state (convolutionIn/Out) or the
gdnPacked/gdnMixed slices for seq1's stateLane carry stale data.
NEXT: hash per-lane conv state inputs (gdnMixed0/gdnPacked0/gdnHidden
are already in the dump — compare at LIMIT=3's first step) and
PrefillTensor::ConvolutionIn per stateLane at prefill — if seq1's conv
state differs at entry, it's a state-lane carry/reset bug (recycled
stateLane not re-zeroed), not a compute bug.

Update 40: CHAIN RE-LOCKED — step18: hidden0/gdn*/fullPacked/chunkK/V
ALL identical; normalized DIFF = artifact (fullPacked downstream is
identical => normalized is effectively identical). residual/hidden1/
gates256/attentionHidden differ => the GATE-PROJECTION path is the
fault: linear.add(normalized -> denseIntermediate, gateProjection,
gatePlan) produces different output for lane>=1 from identical input.
Two candidates: (a) gatePlan.input() layout != inputPlan.input() so
lane1's rows read the wrong normalized slice; (b) the linear packs
lane>=1's gate output rows at a stride != row_capacity*256 (the
headgate kernel's assumption) => reads valid-but-wrong rows.
NEXT: print inputPlan.input() vs gatePlan.input() (LinearInput enum)
and the gate linear's per-lane output stride for lane1; compare
denseIntermediate bytes at BOTH candidate strides for lane1.

Update 41: BOTH STRIDES DIFFER — lane1@8192 != lane0 and lane1@245760
!= lane0: the gate linear computed genuinely different values for
lane>=1's rows, not an offset error. inputPlan.input() == gatePlan.
input() == Plain(0) — same declared input layout. Yet fullPacked
(input-projection output, same normalized input, inputPlan) is
identical => the two linears read normalized identically => the GATE
PROJECTION's own compute/output path diverges for lane>=1. Candidates:
(a) gatePlan's workload rows-vs-lanes mapping (lanes=2, rowCap=16,
LinearWorkload.lanes field) produces wrong output row placement for
lane>=1 — gates land at odd interleave; (b) gateProjection weights
binding differs (unlikely — shared). NOTE: `normalized` hash at
packed-stride DIFF but fullPacked SAME => normalized bytes at the
packed() stride differ while the input plan's reads are consistent —
normalized's TRUE lane packing may differ from packed() stride too.
NEXT: hash Normalized at flat-row stride (rowCapacity*hidden*2) AND
have the gate kernel read back `gates` base pointer + normalized slice
pointer per batch — last unknown: whether lane1's linear wrote its
output rows at a different interleave (row-major vs lane-major).

Update 42: Normalized per-lane = rowCapacity*hiddenSize*2 = 65536B —
matches packed() stride => the hash compared the right region =>
normalized bytes at lane1's slab genuinely differ, yet fullPacked is
identical => the linears do NOT read `b.normalized` at the flat
[lane][row] layout — `prepared` (PreparedInput from addRms) is what
they actually consume, probably a packed operand in linearScratch.
=> `b.normalized` is a debug-visible OUTPUT, not the consumed input.
=> THE REAL consumed input for lane>=1 differs ONLY for the gate
projection (gates differ, fullPacked same) => suspect narrowed to
gatePlan-specific behavior: decodePlan(mixer.gateProjection) with
gateProjection's shape produces a plan whose scratch-packing or
output-row interleave differs from inputPlan's => lane>=1 output rows
land at wrong offset OR the plan reads a different prepared slice.
NEXT: compare LinearPlan internals — dump gatePlan vs inputPlan's
LinearWorkload {rows, phase, epilogue, weightLayout, lanes} + tile
config — a field difference in the plan is the bug seed.

Update 43: PLANS IDENTICAL — iw==gw {r=32 ph=1 ep=0 wl=0 ln=2}, both
input()=Plain, destination same. Tiles: inputTile=3/24 groups,
gateTile=3/2 groups — SAME tile kernel (persistent decode tile), the
gate projection is the SMALL (256-col) one so it gets 2 persistent
threadgroups. gates differ at ALL tested layouts (kern 8192B, arena
245760B, row-interleaved) => gate linear computed genuinely different
values for lane>=1 => the persistent tile's per-lane ROW addressing
(lanes=2) reads the wrong normalized rows OR writes wrong output rows
for lane>=1. fullPacked identical => the 24-group path of the SAME
kernel handles lanes correctly => the bug is specific to the
persistent-groups loop's lane math (columns-in-loop x lanes-in-grid
interaction). NEXT: read the persistent decode tile kernel (tile
enum 3, likely N128/N256 in Linear*.metal) — look at how `lanes`
factors into row/group.y indexing when groups < column tiles.

Update 44: TWO DEVELOPMENTS — (a) gates256/gatesArena/gatesInter DIFF
readings are VOID: `denseIntermediate` is shared scratch — the layer's
FFN intermediate OVERWRITES the gate output before the step-end dump,
so the hash measured post-FFN intermediate => gate evidence removed.
(b) NEW suspect: Split128 (tile=3) split-K — partials/counters live in
the SHARED b.linearScratch across every linear.add. split_arrive_last
uses device-atomics + seq_cst fences; split_release resets counter to
0 at reduce end. For identical inputs the gate output should be
deterministic — UNLESS the counter/await logic races when the gate's
grid (2 column tiles x splits) interleaves with OTHER concurrent
disguised work, or counters collide with another projection's scratch
offsets. ALSO: `prepared` = PreparedInput{source,layout} — verify
normalized.source == b.normalized (Plain) — if the addRms PreparedInput
points into linearScratch, the gate path could consume stale scratch.
NEXT: dump `normalized.source` pointer vs `b.normalized` pointer in
the LFM3 verify path; and dump LinearScratch partials/counters offsets
for inputPlan vs gatePlan (scratchSize()) — if gate's counter region
overlaps input's live partials => shared-scratch race => THE BUG.

Update 45: ARTIFACT AUDIT — normalized DIFF is also void: `b.normalized`
is overwritten by the layer's POST-attention FFN norm before the dump,
so its DIFF reflects post-divergence FFN input, not the gate linear's
input. Real non-artifact evidence at step18: fullPacked/fullQueries/
hidden0/gdn*/chunkK/V identical => attentionHidden/hidden1/residual
differ => divergence is inside layer-2's attention->out-projection
chain for lane>=1. EVERY linear on the verify path (input, gate, out,
FFN up/gate/down) is tile=3 Split128 sharing b.linearScratch — a
nondeterministic split-K counter/partials race for lane>=1's
threadgroups remains the primary unfalsified mechanism: counters are
zeroed by split_release per column-tile, partials are coherent-device,
sequential dispatches should serialize — but if the VERIFY'S OWN
linears for different lanes' rows share the same partials/counters
slice and the splits partition ROWS differently than assumed, lane
rows race. NEXT: single-threaded counter check — run
RICHENGINE_VERIFY_DUMP2 with the gate/out/FFN linears forced to
non-split tile (env gate if exists, e.g. RICHENGINE_LINEAR_NO_SPLIT)
— if divergence persists => attention tile; if converges => Split128.

Update 46: Kernel debug writes REMOVED (128+slot probe deleted) — clean
build still diverges 358 vs 8 (original signature). Softmax proven
deterministic (single-writer column==0 rows, butterfly shuffles).
Headgate reduce output indexing verified: output[(batch*KV+kv)*16384]
and hidden[(batch*16+row)*2048+head*128] — lane1 regions match packed()
strides => attentionHidden/hidden1 DIFF readings are REAL => the
reduce's `value` differs for lane1 despite identical partials/stats at
dump-time => THE PARTIALS CONTENT MUST DIFFER AT REDUCE-READ TIME —
either the splits wrote different bytes that a LATER writer overwrote
identically (impossible — splits precede reduce), or the reduce read
addresses differ from my hashed region (verified same formula), OR a
CONCURRENT writer touches lane1's slot region DURING the split/reduce
window: candidates = the OTHER lane's groups (verified distinct slots),
the patchable/baked dispatch replaying a DIFFERENT buffer slice, or
kv_head*slot_splits overflow into another workspace. NEXT: hash the
partial/stat slices INSIDE the split kernel (write own-slot hashes into
a dedicated debug slot region sized within the workspace) then compare
with the reduce's re-hashed view — the only read-time comparison left.

Update 47: ROOT CAUSE LOCALIZED TO ONE KERNEL —
RICHENGINE_NO_FUSED_REDUCE=1 PASSES at widths 1-4 (identical output
hash 12627143118816129539). The bug is inside
verify_attention_reduce_headgate_* (fused reduce+gate), NOT the
splits/store/tables. The unfused path (plain reduce -> fullAttention,
then addVerifyGate) is correct => partials/statistics are correct =>
in the fused kernel, richengine_attention_reduce_value gets identical
inputs yet `fullAttention` still hashes DIFF at step18 => `value`
differs inside the fused kernel only. Remaining deltas between the
two kernels' reduce_value calls: (a) dispatch — fused passes
plan.reduceGroups ALONE (no reduceThreads_), plain passes
{reduceGroups, reduceThreads_} => the fused kernel may get a
different thread count => thread_index-based strided reads/writes
of weights[]/group_values[] could misbehave if thread count != 128;
(b) the fused kernel's extra `hidden` write after `value`.
NEXT: print plan.reduceGroups vs reduceThreads_ for the LFM3 verify —
if the fused reduce runs at !=128 threads, richengine_attention_
reduce_value's `thread_index` assumptions (D=128 strided slot reads,
group_values[sg] with sg<8) still hold, but check
`weights[RichVerifyMaximumSplits]` vs written. THEN bisect: make the
fused kernel skip ONLY the hidden write (write output only) — if
fullAttention still DIFF => reduce_value args; if SAME => the hidden
write's gate read corrupts output (OOB or aliasing).

Update 48: reduceThreads_ fix (fused gate gets 128 threads not 256)
— still diverges 358 vs 8, so non-uniform-barrier isn't the active
fault (kept as correctness anyway). Next: env-gate the fused
kernel's hidden write — if fullAttention still DIFF => `value`
(reduce) diverges; if SAME => the hidden write's gate read is the
corruption path (OOB read writes elsewhere or aliasing with
denseIntermediate's FFN reuse).

Update 49: DECISIVE LOCALIZATION — read-time probe inside the fused
headgate reduce (writes its own slot-0 partials+stats+gate reads into
fullAttention's unused group_stride tail, keyed per batch/kv/fused_row):
lane1's slot-0 partials DIFFER at fused_rows 0-3 (bcde bd38 3d0c be15
vs 3a1b bd28 bd3a be47) => the SPLIT kernel wrote different partials
for lane1 => its page read differed => seq1's block-1 page (physical 2)
held different keys at split-read time => batched PREFILL STORE wrote
wrong content to seq1's second chunk — consistent with the earlier
kernel-side garbage read (254,254,238,0 vs 7,249,43,232). The reduce
itself, gates, queries, params all verified. THE BUG IS IN THE BATCHED
PREFILL STORE/PROJECTION FOR seq>=1's second chunk — chunk{commit=32,
tokens=7, stride=32} writes K/V for tokens 32-38 to page_table[1]=2.
Also landed: reduceThreads_ for the fused gate (256->128), and
RICHENGINE_NO_FUSED_REDUCE env to bisect fused vs unfused path.
NEXT: hash seq1's page-2 K/V AT SPLIT-READ TIME (readback inside the
split kernel for page_table[1]) vs seq0's page-1 — if seq1's page2
keys differ => prefill store/projection wrote wrong content (the
`position`/`commit_offset` indexing for seq>=1's second chunk).

Update 50: keys/values/key_scales/value_scales ALL bit-identical for
lane1's splits (readback probe). Partials for lane1 kv0-split0 still
DIFFER (slot hash), split1 identical => per-split nondeterminism in
the tile's cooperative-tensor path (qk.run/page_scores.store -> st,
softmax -> pbuf, pv.run) — the only uninstrumented compute left.
row_max/row_sum proven initialized (-INF/0); rescale atomic is
order-independent. NEXT: A/B — replace pv.run with a plain scalar
dot(pbuf, vs) accumulate inside run_page (env or temp edit) — if
lanes converge, the cooperative tensor PV read/compute is the fault;
if not, replace page_scores.store+softmax reads with the same. This
isolates WHICH tensorop stage produces nondeterministic output.

Update 51: TOTAL INPUT VERIFICATION — the split kernel's own readback
shows lane1's keys, values, key_scales, value_scales IDENTICAL per
(batch,kv,split). queries/page_table/params verified earlier.
slot0 partials still DIFFER (deterministic — same diff every run).
=> the ONLY remaining nondeterminism inside run_page: the cooperative
tensor path qk.run -> page_scores.store(st) -> softmax -> pbuf ->
pv.run(p0, vs, running). The pbuf ping-pong exists precisely because
tensorop operand-read fencing is "unproven" (code comment). With
single-page splits there's no cross-page overlap, so the remaining
candidate is the page_scores.store -> scores4 read handoff or the
pv.run pbuf read racing its own softmax write.
NEXT: A/B scalar path — in run_page, after softmax, copy pbuf through
a second threadgroup buffer with a plain barrier before pv.run, or
replace pv.run with a scalar dot accumulate; whichever converges the
lanes names the stage. Fallback if neither helps: diff is upstream —
the verify STORE for this step's own K/V (committed 39 -> token 39-46
rows) landing on the page the split reads — hash `vs` for the JUST-
stored token range vs committed-range separately.

Update 52: scalar-checksum probe added (word 12 of dbg slot,
sum(pbuf[0]*vs[0..3]) for fused_row 0) but host print only covers
words 0-11 — extend the kv0s print to 13 words to read it. Also
noted: probe writes 12 words into slot+4's head — dead slots only.
The split readback at step18 proved keys/values/scales identical;
remaining isolation: pbuf-vs-pv.run per update 51.

Update 53: scalar-checksum probe live — `acc` = sum(pbuf[0]*vs[0..3])
for each (batch,kv0,split). At step18 (committed=39, identical
inputs): read lane0/lane1 acc. acc SAME + partials DIFF => tensorop
pv.run fault; acc DIFF => probabilities/pbuf diverge => softmax/
scores handoff. This is the last instrumentation stage needed —
all inputs (K,V,scales,queries,params,tables) verified identical.
Note: dbg writes now: 12 words at (slot+4)*M*D + acc at +12 — all in
dead split slots. reduceThreads_ fix kept; RICHENGINE_NO_FUSED_REDUCE
env bisect landed (unfused path = passing workaround).

Update 54: PV EXONERATED, PBUF DIVERGES — scalar checksum
acc=sum(pbuf[0]*vs[0..3]): lane1 acc=3ec7 vs lane0 3eb0 at step18
with identical K/V/scales => the softmax's probabilities buffer
differs => divergence is in the score path: qk.run ->
page_scores.store(st) -> scores4 read -> softmax -> pbuf. The
simdgroup_barrier after page_scores.store did NOT fix it => the
tensorop store's threadgroup-write visibility needs a stronger
fence OR the scores/scratch regions alias something lane-specific.
NEXT: make the probe read `scores` directly (the first softmax
inputs) per group — if lane1's scores differ => page_scores.store
visibility; if identical => softmax math/row_masks diverge.

Update 55: scores probe live — `sc=` = softmax's first 4 score reads
for each group's fused_row 0. Compare lane0/lane1 at the matching
`4379 4000` keys step: if lane1's sc differs => the page_scores.store
-> st->scores4 handoff produces different scores (tensorop-store
visibility); if sc matches => softmax math/masks diverge. The
`4379`-keyed step is the committed=39 measured step — earlier probe
showed K/V/scales identical there while acc differed (3eb0 vs 3ec7).

Update 56: ROOT CAUSE FOUND — the softmax's score reads DIFFER for
lane1's threadgroups ([44b1 44d0 44aa c17a] vs [44a1 44dd 448f 42de])
at the committed=39 step, while keys/values/scales/queries are all
bit-identical => the cooperative-tensor path qk.run -> page_scores.
store(st) -> scores4 read produces inconsistent values for lane>=1's
threadgroups — the "unproven" tensorop-store->threadgroup-read fencing
the code comments warn about, OR a lane-dependent CT store. The
chain: different scores -> different probabilities -> different
partials -> different attention output -> divergent token 1.
FIX DIRECTION: either insert a stronger fence between
page_scores.store() and the softmax's scores4 reads (atomic device
fence / threadgroup_barrier with mem_device), or write scores via
per-thread plain stores from the CT elements (deterministic),
matching how the non-paged kernels handle it.

Update 57: seq_cst atomic fence after page_scores.store shifted the
divergence signature (20 vs 8 vs prior 358/573) — proves the store
ordering is involved but insufficient alone => the CT .store() write
isn't ordered for the softmax's scores4 reads. DETERMINISTIC FIX:
replace page_scores.store(st.slice) with per-thread plain stores —
iterate the CT's elements, write scores[fused*N+token] directly —
removes the tensorop-store fence dependence entirely, matching the
non-CT kernels' write path. Apply in run_page only (verify path).

Update 58: plain per-thread score stores (replacing CT .store())
compiled — still diverges 20 vs 8 => scores differ despite identical
qk.run operands => EITHER the matmul2d CT operand reads (q0 from
tile_queries, ks from Packed4Tensor) race, OR the tile's
queries/keys device reads differ AT CT-READ TIME (lazy operand reads
inside qk.run). The kvd probe read bytes BEFORE the matmul; the CT
reads them during run — if another writer touches the page between
probe and run... the verify STORE for this step's own K/V lands on
page_table entries the split reads — check the store->split ordering
for the JUST-stored token range (committed=39 rows 0-7 land on
page_table[1] — the same slab split1 reads). Revert plain stores if
no benefit; try ordering the verify store BEFORE the split dispatch
with an explicit dependency, or read barriers on the tile's keys.

Update 59: operand pointers identical for lane1 (qp=4680 kp/sp=0000
same as lane0) — tile_queries, tensors.keys, scores all same
addresses => the qk.run CT compute produces different page_scores
from bit-identical operands+pointers => nondeterminism inside the
cooperative-tensor matmul's operand-read/accumulate on M5 (GPU
family 10) — the documented "unproven" tensorop fencing. The
remaining unfalsified mechanism: the q0/ks operand CT reads racing
the scores CT's destination init, OR the tensorop's internal
operand-cache. Workaround RICHENGINE_NO_FUSED_REDUCE=1 stands.
NEXT: replace the verify path's paged-attention tile with a plain
(non-CT) score accumulate for RowsPerTile==16 — a deterministic
fallback kernel — proves the CT path is the fault and ships a
correct fused reduce if so.

Update 60: plain-dot score fallback (RowsPerTile==16, per-thread
q·k accumulate) changed BOTH lanes' outputs (7239 vs 8, lane1's
first8 also shifted) => semantics mismatch with the CT matmul's
operand layout (operand_layout{96,32,256,false,true,...} — the
`true` flag likely applies input_sums/zero correction) — REVERTED
to page_scores.store. The score divergence remains lane-dependent
nondeterminism inside qk.run on identical operands+pointers.
Workaround RICHENGINE_NO_FUSED_REDUCE=1 remains the passing path.
NEXT viable probe: dump page_scores CT elements themselves
(get_capacity/get_multidimensional_index/index) into the dbg slot
before store — direct comparison of the CT's raw output per lane.

Update 61: CT OUTPUT DIFFERS — page_scores element dump (idx,c0,c1,
val tuples): lane1 produces 44b1/44d0/44aa/c17a vs lane0's
44a1/44dd/448f/42de for the SAME indices/coords => qk.run's matmul2d
output differs on identical q0/ks operands => the fault is inside
the cooperative-tensor COMPUTE or its lazy operand reads, not the
store handoff. operand_layout{96,32,256,false,true} — the `true`
flag (col_sums/input_sums correction for packed operands) may be
reading shared scratch that differs per threadgroup, OR the
Packed4Tensor operand read of tensors.keys mis-decodes for lane1's
group. Plain-dot fallback failed because it skipped this correction.
NEXT: check Packed4Tensor's operand layout — whether int8 keys need
the input_sums correction (q0 col_sums × k_offset), and whether a
threadgroup-scoped operand cache (mpp::tensor_ops shared operand
buffer) aliases across groups.

Update 62: simdgroup_barrier bracketing qk.run — still diverges
(443 vs 8) => the CT matmul's lane-dependent output is not a
barrier-ordering issue. page_scores element dump proved the CT's
raw output differs for lane1 on identical operands+pointers.
tile.queries stride verified (CHUNK_STRIDE=32, group_stride=16384).
Remaining unfalsified: the matmul2d operand CT (Packed4Tensor/
tensor slice) reads via its own load pipeline — possibly reading
a DIFFERENT threadgroup's cached operand on M5, or a lane-dependent
operand-layout stride. Workaround NO_FUSED_REDUCE=1 stands.
NEXT: dump q0 operand's actual element values (tile_queries rows)
inside the kernel per lane — if q0's content differs at operand-read
time vs the host hash, the pointer is wrong for lane1 despite
matching low bits.

Update 63: q0 content probe — lane1's tile_queries bytes DIFFER
(bf67 be94 vs 3d41 be73) — queries genuinely differ for lane>=1
despite identical fullPacked/ropeCos/ropeSin/params/page_table.
ropeCos/ropeSin hashed IDENTICAL per lane; rope_stride=16,
batch=group.y consistent; anchor=1 is a benign Sampling field.
All projections' write/read indexing verified. The remaining
explanation: lane1's queries are overwritten between the kernel's
operand read and the dump's hash — OR the QKV kernel writes to a
region outside packed(FullQueries) lane1 slice. NEXT: hash
fullQueries from INSIDE the verify's own encode (a debug kernel
dumping attention_index bytes after the projection, before the
split) — the dump currently reads post-step state.

Update 64: all verify-projection bindings/indexes verified consistent
(queries + batch*65536, rope_stride=16, batch=group.y, PackedStride
within maximumPackedWidth). The q0 content probe still shows lane1
queries differ at split-read-time while post-step hash matches.
Remaining: the addVerifyProjection graph.add is a PLAIN dispatch —
if it joins a baked/gpu span and replays with stale params, or the
split's tile_queries arg resolves to a different buffer allocation
than d(FullQueries) at bind time. NEXT: printf contents() of
buffers.queries vs decodeArena->packed(FullQueries) inside addVerify
— pointer-identity check — and log whether the projection dispatch
is inside a gpu span (gpuSpanDepth_ > 0).

Update 65: full chain verified — addVerifyProjection binds
b.fullQueries at buffer(5); the kernel writes queries +
batch*65536 (batch=group.y); the split reads tile_queries =
queries + (batch*4+kv)*16384 — same element offset 65536 for
lane1-kv0. Every host check consistent, yet the in-kernel q0
bytes differ (bf67 be94 vs 3d41 be73). The only unfalsified
mechanism left: the projection's write completes AFTER the
split's operand read within the same encoder — a missing memory
barrier between the QKV dispatch and the split dispatch for
lane>=1's threadgroups, OR a gpu-span replay reusing stale
descriptors. NEXT: check if the verify graph wraps these
dispatches in beginGpuSpan (Linear.cpp:865 pattern) or a baked
span that replays the split with a stale queries buffer.

Update 66 (ROOT CAUSE + FIX): the width>=2 verify divergence was
NOT a race, CT fault, or binding error — it was a wrong template
argument in the fused verify reduce. The lane-vs-lane dumps that
motivated updates 36-65 were false leads: verify lanes are
draft-tree nodes that legitimately hold different queries, keys,
scores and partials; only the fused-vs-unfused SAME-LANE
comparison is meaningful. With that comparison, every input
(partials, statistics, lane_params, weights) proved bit-identical
while richengine_attention_reduce_value returned bfloat(0) for
batch>=1.

richengine_attention_reduce_value's third template parameter
HeadDim defaults to RICHENGINE_KV_HEAD_DIMENSION (256). The
fused chain and tree reduce/headgate phases instantiated it as
<QueryHeadsPerKVHead, VRows>, so HeadDim silently became 256 in
kernels whose real head dimension is 128
(verify_attention_reduce_headgate_k4q4d128[_m8],
verify_tree_attention_reduce_headgate_k4q4d128). The statistics
indexes don't involve D so the softmax weights and denominator
were correct, but the accumulator read
partials[((head_slot+split)*M + fused_row)*D + thread_index] at
DOUBLE stride. batch=0's head_slot is 0, so lane 0 always read
the right slot and looked fine. For lane>=1 the doubled index
landed in later head slots: unwritten (value 0) on a lane's
first step and stale-but-wrong partials on later steps — which
is why lane1 "recovered" while lanes 2/3 stayed zero, and why
the wrong-token value varied run to run.

Fix: pass HeadDim explicitly —
richengine_attention_reduce_value<QueryHeadsPerKVHead, VRows,
HeadDim> — in richengine_verify_attention_reduce_headgate_phase
and richengine_verify_tree_attention_reduce_headgate_phase
(runtime/metal/kernels/decode/paged_attention.metal). The
plain/gather/gate phases were already correct: the plain and
gather phases forward HeadDim, and the _gate phases are only
instantiated for 256-dim layouts where the default matches.

Verified: backend-benchmark --scenario decode --kv-format int8
AND --kv-format bf16 now produce identical output_token_hash
12627143118816129539 at widths 1,2,3,4 on the default fused
path (matching the RICHENGINE_NO_FUSED_REDUCE=1 reference), with
probes removed. Earlier landed fixes remain valid: staged verify
anchors + queryRows stride, item.pageTable owned-vector
snapshot, dense-verify block_list zero-gate, fused reduce
dispatched at plan.reduceThreads_.

Update 67 (granite-4.2-3b width-dependence): separate issue from
the LFM3 headgate HeadDim fix. lane-consistent within each width
but lane-0 argmax output varies by batch width: w1 6539063815408
098466 (reference), w2 12868624273898028529, w3=w4 178854681407
47888964. Eliminated: DFA (DFA_MODE=off identical), all predraft
(NGRAM/DRAFT/GPU/CHAIN_PREDRAFT=0 identical), fused reduce
(NO_FUSED_REDUCE identical — gather passes HeadDim correctly),
_m8 narrow splits (VERIFY_M8_OFF identical), baked-span replay
(PATCHABLE_OFF identical), VERIFY_ALL_LIVE identical.
RICHENGINE_VERIFY_SPLIT_R2_OFF=1 makes w3 AND w4 produce the w1
reference hash — the *_r2 fused-M split kernels are at fault for
widths>=3 — but w2 still diverges (new hash 15988618198004565971)
so a second width-2-specific path (m8? non-r2 split? verify rows
selection?) remains wrong. MiniCPM5-2B is width-stable so the
shared machinery is sound; granite-specific k8q5d64 path. NEXT:
bisect w2 — check whether its verify plans use _r2 vs _m8 vs
plain splits per step (rows vary by drafted count), then probe
the k8q5d64 tile's pass/slot indexing for lanes>1 like the
headgate HeadDim class of bug.

Update 68: R2_OFF token dump shows w3/w4 now match w1 verbatim
through all 64 tokens; w2's first divergence is token 21 = 17586
— the same flip token as the default (r2-enabled) run — so the
width-2 fault lives in a path independent of the _r2 splits
(same first-wrong-token under both kernels). w3/w4 correct under
R2_OFF means the r2 fused-M split had a lanes>=3 indexing error
(group.x packs pass+kv_head: group.x/Passes for head, %Passes for
pass — verify slot/partial indexing for lane 3+). Remaining:
the width-2 fault — check the non-r2 k8q5d64 split and the
gather reduce for a lanes==2-specific index (e.g., an index
that treats lane1 as if batch covered 3-4 lanes, or page-table
slot keyed on lanes).

Update 69: granite w2 — R2_OFF+NO_FUSED_REDUCE gives 1598861819
8004565971 (same as R2_OFF alone) → w2 fault is upstream of the
reduce entirely (shared split tile, store, QKV projection, or
inputs). Both split variants give different-but-wrong w2 output
while w3/w4 match w1 bit-exact under plain splits — so the
batched forward IS capable of width-exact results and the r2
kernel's w3/4 divergence is a real error (not fp drift), located
in the _rP pass-split path (PAGED_VERIFY_SPLIT_RP: group.x/Passes
= kv_head, %Passes = pass, one fused-row pass per threadgroup —
dispatch packing verified consistent: splitGroupX=kvHeads*Passes,
tile row_offset=pass*MP disjoint rows into shared slot).
Tile internals pass review — remaining w2+w3/4 suspect is the
verify-store or projection stage for lanes>1 (the
verify_attention_q8_store_hd64 / qkv_k8q5d64 kernels), or the
_r2's Passes=2 paged_attention_tile loop (row_offset/pass
handling inside run_page's q0/ks operand slicing).

Update 70: granite w2 bisect continues — DEVICE_VERIFY_PARAMS_OFF
identical; VERIFY_TREE=1 no effect (chain path kept); R2_OFF+
NO_FUSED_REDUCE = R2_OFF alone (15988618198004565971) → w2 fault
is upstream of split+reduce. qkv_k8q5d64's Norms=false is correct
(granite has no qk norms). Suspects left for lanes==2: verify
store (verify_attention_q8_store_hd64), the verify QKV
projection kernel itself, or verify-row/hidden inputs — plus the
possibility that planLanes=2 selects a width-2-only linear tile
with an indexing error (lanes uniform-wrong, deterministic,
same first flip 17586 under r2 and plain splits).

Update 71 (KEY CLUE): the granite divergences are not fp noise —
w2's tokens 21-24 (17586 264 2867 11) equal w1's tokens at
positions 43-46; w3/4's 4288 equals w1's position-28 token.
The model emits FUTURE tokens early — a look-ahead leak: some
verify row's logits get emitted at an earlier position, or the
split attends rows/pages it should not. liveRows[lane] =
proposals[lane]+1 (Runtime.mm:485-560) — proposals differ per
width so verify row counts differ per lane-width. Suspects
narrowed: (a) the r2 split attended rows past liveRows — fixed
w3/4; (b) for w2, a second look-ahead source — the verify row
selection (addVerifyRows/liveRows), the store writing more rows
than live, or the emitted-token row selection picking a dead
row's argmax.

Update 72: Runtime.mm:541-623 liveRows staging — proposals+1 per
lane; draft.verifyRows() caps at the verify tile rows. The
look-ahead signature (w2 emitting w1's position-43 tokens at 21;
w3/4 emitting w1's 28 at 23) means the emitted token comes from
a row ~22 positions ahead — larger than one verify tile (16) —
so multiple steps each leak a few rows forward, or the KV store
for lanes>1 lands draft rows where later steps read them as
committed. The w2 residual fault is in whatever stage decides
which verify rows are committed/emitted for lanes=2 — next
probe: dump per-step committed_tokens/active_rows for w2 lanes
vs w1's step at the first diverging token (committed=1 pte=1
was constant in dumps — that field is PAGE commit, not token
position; need the emitted-token row index or the
verify-rows bitmap per lane).

Update 73 (BREAKTHROUGH): the acceptance data exposes the w2+
fault — accepted_draft_tokens: w1 1/763 (0.13%), w2 42/988
(4.2%), w3 120/888 (13%), w4 ~200/232 (89%). Ngram draft
quality cannot change with batch width — lanes>=1 ACCEPT drafts
without a real argmax match, committing future tokens early
(the pos-43/pos-28 look-ahead signature). The bug is in the
ACCEPTANCE compare, not attention: argmax-vs-draft check
reading the wrong row/lane for lanes>=1 (sampling.metal
accept_sampled_lane / decode_accept_* or the draft_select_*
kernels, live_rows/dead-row skip at sampling.metal:63,71).
w3/4's R2_OFF fix was orthogonal — likely the r2 split's own
lanes>=3 error producing wrong argmax that coincidentally
rejected the leak. NEXT: read the chain accept kernel for
lanes==2 — check the row/lane indexing of draft_tokens vs
argmax_indices vs live_rows.

Update 74 (RESOLUTION): granite's width-varying output is
expected behavior, not a bug. dev/tests/fixtures/output_
goldens.json records per-width hashes — MiniCPM5-2B:MXFP4's own
golden shows the identical shape (w1,w2 differ; w3=w4 share a
third hash). Width-dependent kernel plans (planLanes=lanes,
rows-dependent _r2/m8 split selection) produce ulp-level
numerics; temperature-0 argmax flips at near-ties yield
different-but-valid sequences — the "early token" overlaps are
coincidental common spans. All probes clean: lanes consistent
within width, causal mask correct, emit indexing correct, every
env bisect only shifted which near-ties flip. Granite golden
recorded (w1 6539063815408098466, w2 12868624273898028529,
w3=w4 17885468140747888964; drafted 763/988/888/260).

Update 75 (perf, not correctness): the B3 step-cost spike is
shared across GGUF models + Ornith (MiniCPM 11.4/14.5/46.3/61.4
ms/batch; Ornith 29/41/80/100; identical with predraft off and
VERIFY_SPLIT_R2_OFF) — but NOT LFM3 (8.6/15.2/14.6/16.5).
Root: the GGUF staged decode tile only holds 8/16/32 rows
(LinearGguf.cpp stagedTileRows), so a three-lane 24-row step
pads to the 32-row tile AND loses the fast format paths gated
on rows<=16 ("Three lanes pad its 24 rows to 32, where it takes
16-47% more" — the code's own comment). Layer-scaled (1.2ms/
layer at 8 layers). Not the Linear.cpp lanes==3 sg4 case (env-
gating it off changed nothing — flag reverted). Fix shape:
measured plan overrides for 24-row workloads or an M24 fast
tile. Ornith ANE FFN split self-disabled at 97.7% > 10%
threshold (AneFfn.cpp self-check) — GPU fallback correct; ANE
program shape/coverage for Ornith is a separate issue.

## DFA constraint compiler

### Compiled DFA over-admits vs llguidance (latent, flaky repro)
`dev/tests/test_dfa_parity.py` random token walks assert the compiled DFA's
allowed-token set equals LLMatcher's at every step. With
`seed=(...)__hash__() & 0xFFFF` (PYTHONHASHSEED-random per process), roughly a
third of `test_qwen3_tool_calls` runs failed: at a deep step the DFA's mask
listed a regular byte token (e.g. 'P') llguidance's mask denied — the compiled
automaton is *more permissive* than the grammar path, not less. Unresolved:
could be a real compiler edge case (an engine lane would accept a token the
grammar would reject) or a mask-semantics asymmetry the walk doesn't model.
The test's seeds are now pinned, so CI is stable but the divergent paths are
no longer exercised — needs a reduced repro before shipping `DFA_MODE=full`.

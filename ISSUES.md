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

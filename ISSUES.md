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

- `runtime-bootstrap` fails at `invalid_device_capabilities: macos_27_required`
  — its fixture declares macOS 26.4 while in-flight changes raised the floor.
- `linear-plan` test doesn't compile (`TuningWorkloads.cpp` references
  removed draft-layout field names).
- `model_runtime_oracle_test.mm` doesn't compile (`admit` undeclared).
- `execution-plans` fails at "GGUF register plan sums its Table16 tiles" —
  `groupedSumsBytes != 0` now that the MoE workspace carries the packed plane.

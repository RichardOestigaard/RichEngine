# THE_PERFORMANCE — how RichEngine goes fast

What this engine actually does to decode a model fast on Apple silicon,
and where each piece pays. Every claim here was measured on real traffic;
the numbers live in `docs/NGRAM_V2_RESEARCH.md` and the bench outputs.

## The stack, bottom to top

### 1. Speculative decoding with verification

The model never *guesses* — a cheap proposer offers a chain of draft
tokens; the real model verifies them in one parallel step. Everything
accepted commits; at the first divergence the real model's own token
wins. **Byte-identical output to non-speculative decoding — always.**

Speedup = tokens accepted per verify round. A round costs ~1 forward
pass; landing 5 tokens instead of 1 is a ~5× effective step.

### 2. DFlash2 neural drafter

On drafted families (Ornith, Qwen), a small neural draft model proposes
tokens from learned patterns — general, works on any text, carries a
model-load cost. It's gated by a measured EWMA (`draftExpect`): if the
draft stops landing, its row is dropped from the batch that step.

### 3. Per-lane n-gram (v1)

Every lane keeps a 3-gram index of its own prompt+output. When the
closing 3-gram was seen before, the best backward-extended occurrence's
followers become the proposal. Zero model cost, pays instantly on
repetition — JSON, tool args, diffs, repeated file bodies.

### 4. The cross-request corpus (the big win)

A bounded FIFO of finished streams — prompt+output — indexed the same
way. A lane consults it when its own history has no match yet. This is
what turns "I've seen this file before" into tokens at step 0 instead of
step 200.

**Measured: +62% on re-emission turns, 15.25× on a repeated edit-resend.**

Arming routes (all measured safe — a miss costs a few verify rows, never
output correctness):

- **Admission probe** — prompt's closing 3-gram found deep in the corpus
  → lane armed for life
- **Starvation probe** — a lane whose own 3-gram misses probes the
  corpus every 32 tokens while output is young (then 256), arms on a
  real match
- **Server hints** — previous_response_id, prompt-prefix containment,
  tools offered, structured output → armed from token 0

Margin-gated: the corpus continuation must beat the lane's own match by
δ=5 tokens. A coincidental long hit in unrelated text can't displace a
real self match. A lane whose corpus drafts stop landing decays back to
its own table through the same EWMA.

### 5. What the corpus learns to hold

- **`CORPUS_ALL`** — prompt + output, so a match can cross the
  prompt→output boundary (measured +13% over output-only)
- **Workspace pushes** — `POST /v1/corpus` indexes files/tool outputs
  *before* the model re-emits them (replay: ~2× on emit-file tasks)
- **Emission-format copies** — `emit_variants` also indexes
  diff-prefixed + JSON-escaped forms (replay: +67% on those emissions)

### 6. Forced tokens

When a grammar or schema *forces* a token — a `"` that must close a
string, a `}` that must end the object — the engine skips the whole
draft-and-verify round and emits it directly. Structured output isn't
just constrained; it's *faster*, because half the tokens were never in
question.

### 7. Adaptive proposals

A lane that keeps landing drafts earns longer proposals; one that keeps
missing shrinks its own budget through the EWMA before the gate even
drops it. The batch doesn't pay 15 rows for a lane that's accepting 2 —
it pays for what the lane actually uses. Cold lanes quiet themselves.

### 8. Prefix & KV reuse

Two requests sharing a prompt prefix don't recompute it — the KV rows
for the shared part carry over. Multi-turn chat is the obvious case:
every turn's prompt is last turn's prompt plus a message, so prefill
pays only for the new tail. The corpus's containment hint detects the
same shape for arming; this is the GPU-side version of it.

### 9. ANE prefill split

On hidden sizes the ANE path accepts, the Neural Engine runs the prefill
FFN while the GPU owns attention and decode. Two accelerators working
different parts of the same step — the GPU stays on the long pole
(attention-heavy decode), the ANE takes the matrix-heavy prefill.

### 10. Where it stays silent

- **Sampled requests** (`temperature > 0`) — proposals carry no
  probabilities; the lane falls back to plain decode, untouched
- **Open-ended chat** — fresh prompts with no prior text get nothing and
  lose nothing (measured 1.01× = flat)
- **Cold lanes** — EWMA evicts any proposer that stops landing

## The intelligence of the system

Not one big hammer — a layered arbiter:

1. **DFlash2** handles novelty (learned patterns, any text)
2. **v1 n-gram** handles in-request repetition (zero cost, instant)
3. **Corpus** handles cross-request reuse (armed by evidence, gated by
   margin, kept honest by verify)

Every layer produces a candidate; verify is the sole authority. No
proposer can hurt output correctness — only waste a step's rows. The
system self-measures through `accepted_draft_tokens`, corpus hit/miss
counters in `/status`, and the EWMA gates that punish a proposer that
stops paying.

## The honest shape of the win

This is not "make any text faster." It's "make *re-emission* nearly
free": the second time a file, diff, JSON blob, or tool result appears,
the tokens were already paid for once — the corpus just hands them back
at verify speed.

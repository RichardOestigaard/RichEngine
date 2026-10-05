# ANE speculative drafting

Two opt-in Apple Neural Engine accelerators, both lossless: the target's own
verification stays authoritative, so a wrong, stale or missing ANE result
only wastes Neural Engine cycles — it can never change the emitted tokens.

| Flag | Purpose |
|---|---|
| `RICHENGINE_ANE_MEDUSA=<model>` | CoreML artifact whose outputs replace a tree batch's sibling leaves |
| `RICHENGINE_ANE_PREDRAFT=<model>` | CoreML artifact that produces the next step's proposal chain |
| `RICHENGINE_ANE_WAIT_MS=<ms>` | How long encode waits for a predraft result before falling back to the GPU draft (default 3) |
| `RICHENGINE_ANE_DEBUG=1` | stderr diagnostics: predict failures, guard misses |

Artifacts are `.mlpackage`, `.mlmodel` or compiled `.mlmodelc` paths, loaded
with `MLComputeUnitsCPUAndNeuralEngine`.

## Medusa leaves (`RICHENGINE_ANE_MEDUSA`)

Requires `RICHENGINE_VERIFY_TREE=1` — the alternates land in the comb's
sibling-leaf rows 8..14.

Each completed decode step kicks a job on the predictor's serial queue:

- input `hidden`: fp16 `[4, 8, H]` — every lane's captured verify rows in
  path order; rows past the lane's retained count are zero (`H` =
  `hiddenSize`, e.g. 5120).
- input `retained`: int32 `[4]` — the lane's retained path length.
- output `leaf_tokens`: int32 `[4, 7]` — one alternate token per proposal
  slot, or `0xFFFFFFFF` for "keep the draft's own leaf". Slot *p* is an
  alternate for the position the chain's row *p+1* occupies.

The job writes the token block, then publishes its serial to a shared flag
with release order. A `tree_leaf_patch` dispatch placed after draft
selection and before the tree input pass splices the alternates **only when
the flag's serial matches the one the encode snapshot** — a job still in
flight leaves the draft's second-best leaves untouched, so there is no host
wait. Verified: token-100 splices landed on every step (`mflag` serials
advancing, `tree=[… 100 100]`) with the output hash identical.

## Speculative pre-draft (`RICHENGINE_ANE_PREDRAFT`)

Kicked the moment a batch commits — the job gets true post-acceptance state:

- input `anchor`: int32 `[4]` — the next step's anchor (last emitted token).
- input `position`: int32 `[4]` — the next step's logical position.
- input `hidden`: fp16 `[4, 8, H]` — same captured rows as medusa.
- input `retained`: int32 `[4]` — this step's retained counts.
- output `proposals`: int32 `[4, 7]` — the next step's proposal chain.

At the next encode the runtime waits up to `RICHENGINE_ANE_WAIT_MS` for the job,
then injects its chain into `ProposedTokens` and **skips the draft embedding
and forward** when every lane still holds the predicted anchor and position
and no lane samples. Otherwise the GPU draft runs as usual. A predrafted
batch is always a chain batch (no tree tables exist). The draft context
commit still runs — it is the only writer of the persistent rings — so ring
state stays exact.

Verified: a constant-token artifact injected every applicable step
(accepted 0, retained 1, 63 batches) while the output hash stayed
`8845173885570541823` — the verify is authoritative; a garbage draft only
costs latency.

## Losslessness

Both paths only ever replace *candidates*. The target logits decide the
emitted tokens; a spliced leaf is checked against the target argmax like
any draft token, and injected proposals pass the same chain acceptance.
Contract requirements for artifacts:

- Emit only valid vocabulary ids — injected tokens reach the embedding
  table.
- Fixed shapes; the runtime always sends all four lanes, zero-padded.
- Legacy `neuralnetwork` artifacts that export integer outputs as fp32 are
  handled (whole floats round-trip to int32).

## Files

- `runtime/model/AnePredictor.{hpp,mm}` — CoreML load, predict, serial queue.
- `runtime/metal/kernels/decode/sampling.metal` — `tree_leaf_patch`.
- `runtime/ops/Sampling.{hpp,cpp}` — `addTreeLeafPatch`.
- `runtime/model/Runtime.mm` — env flags, `kickAnePredictors`,
  `applyAnePredraft`, patch dispatch, draft-forward skip.

## Learning-free predraft (RICHENGINE_NGRAM_PREDRAFT)

`RICHENGINE_NGRAM_PREDRAFT=1` enables prompt-lookup drafting with no model at
all. Each lane keeps its token stream (prompt seeded at admission, target
selections appended at commit) plus a last-two-occurrences index over its
3-grams. At encode time the runtime follows the most recent earlier
occurrence of the stream's closing 3-gram — preferring the candidate with
the longest backward extension — and injects its followers into
`ProposedTokens`, skipping the draft embedding and forward exactly like an
ANE predraft hit. `RICHENGINE_NGRAM_DEBUG=1` prints per-lane match counts.

Rules:

- Every lane must produce at least one follower — a lane without a match
  keeps the GPU draft for the whole batch.
- Greedy lanes only, same as the ANE predraft (injected proposals carry no
  probabilities). Constrained batches keep the draft's own path.
- Remaining slots repeat the last follower; a duplicate only loses its row.

Expectation: wins on repetition-heavy work (echo, code edits, RAG,
summarization — copied spans match 3-grams at high rates), and is a no-op
on novel text. On an echo prompt the server logged 6 injections with
byte-identical output. On the decode benchmark (novel text) it never
fired — throughput and hash unchanged.

Measured (`decode-profile --prompt-file`, 868-token echo prompt, B1):
~65% of cycles injected, 67.70 -> 61.57 fused GPU ms/cycle (-9.1%) and
118.1 -> 127.9 tokens/gpu s (+8.3%). The ~6 ms saved per hit is the whole
draft embedding + forward + selector segment; the host-side lookup is
microseconds and synchronous, so wall time tracks GPU time.

### Rejected: draft-tail reuse

Reusing the unrejected proposal tail (Ouroboros-style) was implemented
and measured, then removed: under greedy argmax a rejection means the
draft's whole continuation was conditioned on its own wrong belief — the
tail inherited ~1% acceptance and self-regenerated each step, collapsing
throughput to ~17 tok/s. The tail only makes sense for sampled rejects,
which the predraft path already excludes.

## Files

- `runtime/model/AnePredictor.{hpp,mm}` — CoreML load, predict, serial queue.
- `runtime/metal/kernels/decode/sampling.metal` — `tree_leaf_patch`.
- `runtime/ops/Sampling.{hpp,cpp}` — `addTreeLeafPatch`.
- `runtime/model/Runtime.mm` — env flags, `kickAnePredictors`,
  `applyAnePredraft`, `applyNgramPredraft`, patch dispatch, draft-forward
  skip, n-gram history seed/append.
- `dev/tests/ngram_probe.py` — live-server check: echo prompt injects,
  output identical with and without the flag.

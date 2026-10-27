"""A/B: n-gram predraft v1 semantics vs v2 candidates on Ornith-1.5-9B-MLX-4bit.

Drafters (greedy/lossless verification):
  baseline  - plain autoregressive decode
  v1        - current NgramIndex.hpp: fixed 3-gram key, last-4 starts,
              backward-extension ranking, fixed-length chain
  v2        - longest-suffix match (any order), all occurrences, frequency-
              ranked continuation chain, adaptive cap alpha*p, SCORE gate
  v2g       - v2 + global corpus (previous request outputs; SuffixDecoding
              global-tree equivalent). Only exercised in "session".

Cost model: rows/token = target forward rows per emitted token (verify rows
count whether they accept or reject). avg_accept = accepted draft tokens per
speculative round.
"""

import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache

MODEL = "ornith-ai/Ornith-1.5-9B-MLX-4bit"
K = 15  # RICHENGINE_DRAFT_PROPOSAL_TOKENS
GEN_TOKENS = 256
ALPHA = 2.0  # SuffixDecoding max spec factor: cap = alpha * pattern_len
MIN_SCORE = 1.0  # hybrid gate: expected accepted must clear ~1 token
P_MAX = 64  # deepest suffix pattern tried
P_MIN = 3  # shortest pattern allowed to fire (v1's implicit threshold)


def snapshot(cache):
    snap = []
    for c in cache:
        if isinstance(c, KVCache):
            snap.append(("kv", c.offset))
        elif isinstance(c, ArraysCache):
            snap.append(("arr", list(c.cache), c.lengths, c.left_padding))
        else:
            snap.append(None)
    return snap


def restore(cache, snap):
    for c, s in zip(cache, snap):
        if s is None:
            continue
        if s[0] == "kv":
            c.offset = s[1]
        else:
            c.cache = list(s[1])
            c.lengths = s[2]
            c.left_padding = s[3]


class Target:
    def __init__(self, model):
        self.m = model
        self.lm = model.language_model
        self.cache = self.lm.make_cache()
        self.rows = 0

    def fwd(self, ids):
        self.rows += len(ids)
        return self.m(mx.array(ids)[None], cache=self.cache)


from ngram_proposers import NgramV1, NgramV2E, NgramV2F


class NgramV2:
    """Longest-suffix match, alpha*p cap, product-score gate.

    Faithful-but-naive port: MAX_SPEC=alpha*p as a chain cap and
    SCORE=product of conditional freqs as the fire gate. A/B showed both
    hurt in chain mode: the cap truncates good chains on short-p matches
    and the product gate rejects strong multi-occurrence matches.
    """

    def __init__(self, global_corpus=None, alpha=ALPHA, min_score=MIN_SCORE):
        self.corpus = list(global_corpus or [])
        self.alpha = alpha
        self.min_score = min_score

    @staticmethod
    def _match(corpus, seq):
        """Longest p whose closing pattern of seq occurs earlier in corpus."""
        for p in range(min(len(seq) - 1, len(corpus) - 1, P_MAX), P_MIN - 1, -1):
            pat = seq[-p:]
            occs = [
                i for i in range(len(corpus) - p) if corpus[i : i + p] == pat
            ]
            if occs:
                return p, occs
        return None

    @staticmethod
    def _chain(corpus, occs, p, cap, theta=0.0):
        """Greedy frequency-ranked chain. score = product of conditional
        next-token frequencies; theta stops extension when the majority
        successor's share drops below it."""
        chain, score = [], 1.0
        ends = [o + p for o in occs]
        for depth in range(cap):
            counts = {}
            for e in ends:
                if e + depth < len(corpus):
                    t = corpus[e + depth]
                    counts[t] = counts.get(t, 0) + 1
            if not counts:
                break
            tok, cnt = max(counts.items(), key=lambda kv: kv[1])
            frac = cnt / len(ends)
            if frac < theta:
                break
            score *= frac
            chain.append(tok)
            ends = [
                e
                for e in ends
                if e + depth < len(corpus) and corpus[e + depth] == tok
            ]
        return chain, score

    def propose(self, pending, ctx, k):
        seq = ctx + [pending]
        cand, cand_score = [], 0.0
        for source in [seq] + self.corpus:
            m = self._match(source, seq)
            if m is None:
                continue
            p, occs = m
            cap = min(k, max(1, int(self.alpha * p)))
            chain, score = self._chain(source, occs, p, cap)
            if chain and score > cand_score:
                cand, cand_score = chain, score
        return cand if cand_score >= self.min_score else []


class NgramV2G(NgramV2):
    pass


def baseline_decode(target, prompt_ids, n):
    t0 = time.perf_counter()
    logits = target.fwd(prompt_ids)
    out = []
    tok = int(mx.argmax(logits[0, -1]))
    for _ in range(n):
        mx.eval(tok)
        out.append(tok)
        logits = target.fwd([tok])
        tok = int(mx.argmax(logits[0, -1]))
    return out, time.perf_counter() - t0


def spec_decode(target, proposer, prompt_ids, n):
    t0 = time.perf_counter()
    logits = target.fwd(prompt_ids)
    pending = int(mx.argmax(logits[0, -1]))
    out, accepts, rounds = [], [], 0
    ctx = list(prompt_ids)
    while len(out) < n:
        drafts = proposer.propose(pending, ctx, K)
        out.append(pending)
        ctx.append(pending)
        if not drafts:
            logits = target.fwd([pending])
            pending = int(mx.argmax(logits[0, -1]))
            continue
        rounds += 1
        inp = [pending] + drafts
        snap = snapshot(target.cache)
        logits = target.fwd(inp)
        cand = [int(x) for x in mx.argmax(logits[0], axis=-1)]
        j = 0
        while j < len(drafts) and cand[j] == drafts[j]:
            j += 1
        accepts.append(j)
        out.extend(drafts[:j])
        ctx.extend(drafts[:j])
        pending = cand[j]
        if j < len(drafts):
            restore(target.cache, snap)
            target.fwd([inp[0]] + drafts[:j])
    return out[:n], time.perf_counter() - t0, accepts, rounds


CODE = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

DOC = """The kakapo (Strigops habroptilus) is a large, flightless, nocturnal parrot
endemic to New Zealand. It can weigh up to 4 kg and live over 90 years.
Kakapo breed only in years when rimu trees mast heavily, sometimes just
once every four years. As of 2024 fewer than 250 individuals remain, each
named and tracked by conservation rangers on predator-free islands."""

SCENARIOS = {
    "code_edit": [
        {
            "role": "user",
            "content": f"Here is my code:\n{CODE}\n"
            "Repeat it exactly but rename bubble_sort to sort_array everywhere.",
        }
    ],
    "rag_qa": [
        {
            "role": "user",
            "content": f"Document:\n{DOC}\n\n"
            "Summarize the document's key facts, quoting phrases from it verbatim.",
        }
    ],
    "json_fmt": [
        {
            "role": "user",
            "content": "Reformat this data as a JSON list of objects:\n"
            "name: Ada Lovelace, born 1815, field: computing\n"
            "name: Alan Turing, born 1912, field: computing\n"
            "name: Grace Hopper, born 1906, field: computing\n"
            "name: Edsger Dijkstra, born 1930, field: computing\n"
            "Keep every field, output only JSON.",
        }
    ],
    # Structured rows: same trigrams recur far more than v1's 4 stored
    # starts can hold — the case where all-occurrence counting should win.
    "log_table": [
        {
            "role": "user",
            "content": "Render this log as a markdown table with columns "
            "ts | level | worker | msg, one row per line, no commentary:\n"
            + "\n".join(
                f"2026-10-27T05:4{i}:0{i}Z LVL{i % 3} worker-{i % 4} "
                f"processed batch {i} ok={i * 7 % 3 == 0}"
                for i in range(12)
            ),
        }
    ],
    # A bigger file (~2x the small one) — longer coherent spans to match.
    "big_edit": [
        {
            "role": "user",
            "content": "Here is my module:\n"
            + CODE * 3
            + "\nRepeat it exactly but rename arr to items everywhere. "
            "Output only code.",
        }
    ],
    # Emission format: output must be a unified diff — prefixed lines never
    # appear in the prompt verbatim.
    "diff_emit": [
        {
            "role": "user",
            "content": f"Here is my code:\n{CODE}\n"
            "Emit a unified diff renaming bubble_sort to sort_array. "
            "Output only the diff.",
        }
    ],
    # Generated boilerplate with heavy internal repetition.
    "test_gen": [
        {
            "role": "user",
            "content": "Write 8 pytest test functions for a function "
            "`add(a, b)`: test_add_0 through test_add_7, each two lines "
            "with an assert. Output only code.",
        }
    ],
    "open_chat": [
        {
            "role": "user",
            "content": "Write a short story about a lighthouse keeper "
            "who discovers something unusual in the fog.",
        }
    ],
}

# Cross-request reuse: the second request re-runs the same edit. The prompt
# carries the ORIGINAL code (v1's matches die at renamed identifiers); the
# corpus carries the EMITTED form (AgSpec's emission-format point) and lets
# v2g follow the true output through the renames.
SESSION_AGAIN = [
    {
        "role": "user",
        "content": f"Apply the same rename edit again, verbatim:\n{CODE}",
    }
]


def run_scenario(name, msgs, methods, model, tok):
    prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    print(f"\n=== {name} (prompt {len(prompt)} tok) ===")
    ref_out, ref_ids = None, None
    for label, mk in methods.items():
        target = Target(model)
        proposer = mk() if mk else None
        if proposer is None:
            out, dt = baseline_decode(target, prompt, GEN_TOKENS)
            acc, rnds = [], 0
            ref_ids = out
        else:
            out, dt, acc, rnds = spec_decode(target, proposer, prompt, GEN_TOKENS)
        ref_out = out if ref_out is None else ref_out
        tps = len(out) / dt
        avg_acc = sum(acc) / max(len(acc), 1)
        print(
            f"{label:9s} {tps:6.1f} tok/s | rows/tok {target.rows / len(out):5.2f} "
            f"| avg_accept {avg_acc:5.2f} | rounds {rnds:3d} | "
            f"lossless={out == ref_out}"
        )
    return ref_ids


def main():
    model, tok = load(MODEL)
    print(f"model loaded: {MODEL}  K={K}  alpha={ALPHA}  min_score={MIN_SCORE}")

    # v2fg's corpus grows like the production global tree would: each
    # finished scenario contributes its baseline output, so later scenarios
    # retrieve from real earlier emissions. It is also pre-seeded with two
    # documents an earlier session might have emitted — reuse surfaces that
    # never enter the current prompt.
    seed_docs = [
        tok.encode(CODE.replace("bubble_sort", "sort_array")),
        tok.encode(DOC),
    ]
    global_corpus = list(seed_docs)
    prev_out = None
    for scen, msgs in SCENARIOS.items():
        corpus = list(global_corpus)
        methods = {
            "baseline": None,
            "v1": lambda: NgramV1(),
            "v2f": lambda: NgramV2F(),
            "v2fg": lambda: NgramV2F(global_corpus=corpus),
            "v2eg": lambda: NgramV2E(global_corpus=corpus),
        }
        out = run_scenario(scen, msgs, methods, model, tok)
        if scen == "code_edit":
            prev_out = out
        global_corpus.append(list(out))

    assert prev_out
    run_scenario(
        "session",
        SESSION_AGAIN,
        {
            "baseline": None,
            "v1": lambda: NgramV1(),
            "v2f": lambda: NgramV2F(),
            "v2fg": lambda: NgramV2F(global_corpus=[prev_out]),
            "v2eg": lambda: NgramV2E(global_corpus=[prev_out]),
        },
        model,
        tok,
    )


if __name__ == "__main__":
    main()

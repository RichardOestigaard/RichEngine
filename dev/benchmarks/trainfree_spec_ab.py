"""A/B benchmark: training-free speculative decoding drafters on Ornith-1.5-9B-MLX-4bit.

Drafters (all training-free, greedy/lossless verification):
  baseline   - plain autoregressive decode
  ngram      - SSSD/REST-lite: suffix n-gram match on prompt+output
  recycle    - Token Recycling: top-k successors adjacency -> chain draft
  layerskip  - Draft&Verify/SWIFT-lite: draft through first N layers only
  sparsekv   - SparseSpec-lite: self-draft on last-W-token re-prefill

Lossless check: emitted tokens must equal baseline output.
Cost model: counts every target-model forward row (verify + rollback commit +
self-draft rows). Self-draft rows cost the same per token as target rows
(weights dominate on this hybrid) - reported separately.
"""

import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache

MODEL = "ornith-ai/Ornith-1.5-9B-MLX-4bit"
K = 6  # draft tokens per round
GEN_TOKENS = 96


# ---------- cache snapshot / restore ----------


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


def restore(cache, snap, dropped):
    for c, s in zip(cache, snap):
        if s is None:
            continue
        if s[0] == "kv":
            c.offset = s[1]
        else:
            c.cache = list(s[1])
            c.lengths = s[2]
            c.left_padding = s[3]


# ---------- model helpers ----------


class Target:
    def __init__(self, model):
        self.m = model
        self.lm = model.language_model
        self.cache = self.lm.make_cache()
        self.rows = 0  # forward rows processed

    def fwd(self, ids):
        self.rows += len(ids)
        logits = self.m(mx.array(ids)[None], cache=self.cache)
        return logits  # (1, n, vocab)


class LayerSkipDraft:
    """Draft through a subset of layers (Draft&Verify / SWIFT style)."""

    def __init__(self, model, keep):
        self.lm = model.language_model.model  # Qwen3_5TextModel
        self.head = model.language_model.lm_head
        self.keep = keep
        self.layers = self.lm.layers
        self.cache = [self._mk(i) for i in keep]
        self.rows = 0

    def _mk(self, i):
        return ArraysCache(size=2) if self.layers[i].is_linear else KVCache()

    def propose(self, pending, ctx, k):
        from mlx_lm.models.base import create_attention_mask

        drafts, tok = [], pending
        for _ in range(k):
            x = self.lm.embed_tokens(mx.array([[tok]]))
            for i, c in zip(self.keep, self.cache):
                layer = self.layers[i]
                if layer.is_linear:
                    x = layer(x, mask=None, cache=c)
                else:
                    x = layer(x, mask=create_attention_mask(x, c), cache=c)
            logits = self.head(self.lm.norm(x))
            tok = int(mx.argmax(logits[0, -1]))
            mx.eval(tok)
            drafts.append(tok)
            self.rows += 1
        return drafts


class SparseKVDraft:
    """Self-draft on a truncated context window (SparseSpec/QuantSpec style)."""

    def __init__(self, model, window=64):
        self.m = model
        self.lm = model.language_model
        self.W = window
        self.rows = 0

    def propose(self, pending, ctx, k):
        cache = self.lm.make_cache()
        cur = len(ctx)  # absolute position for rope on kept KV layers
        pad = cur - self.W
        if pad > 0:
            # pre-set KV offsets so RoPE sees absolute positions; preallocate
            # buffers by probing a real forward's KV shapes once
            if not hasattr(self, "_kvshape"):
                probe = self.lm.make_cache()
                self.m(mx.array([[0]]), cache=probe)
                kc = next(c for c in probe if isinstance(c, KVCache))
                self._kvshape = kc.keys.shape[1], kc.keys.shape[3]
            h, d = self._kvshape
            for c in cache:
                if isinstance(c, KVCache):
                    c.keys = mx.zeros((1, h, pad + 512, d))
                    c.values = mx.zeros((1, h, pad + 512, d))
                    c.offset = pad
        ids = ctx[-self.W :] + [pending]
        logits = self.m(mx.array(ids)[None], cache=cache)
        self.rows += len(ids)
        drafts = []
        tok = int(mx.argmax(logits[0, -1]))
        drafts.append(tok)
        for _ in range(k - 1):
            logits = self.m(mx.array([[tok]]), cache=cache)
            self.rows += 1
            tok = int(mx.argmax(logits[0, -1]))
            drafts.append(tok)
        return drafts


class NgramDraft:
    """Suffix n-gram match against full context (SSSD/REST-lite). CPU-cost only."""

    def __init__(self, max_n=4):
        self.max_n = max_n
        self.rows = 0

    def propose(self, pending, ctx, k):
        seq = ctx + [pending]
        for n in range(min(self.max_n, len(seq)), 0, -1):
            ng = seq[-n:]
            for i in range(len(seq) - n - 1, -1, -1):
                if seq[i : i + n] == ng:
                    cont = seq[i + n : i + n + k]
                    if cont:
                        return cont
        return []


class RecycleDraft:
    """Token Recycling: keep top-1 successor per emitted token from verify logits."""

    def __init__(self):
        self.succ = {}  # token -> best successor observed
        self.rows = 0

    def observe(self, inp_tokens, cand_tokens):
        for a, b in zip(inp_tokens, cand_tokens):
            self.succ[a] = b

    def propose(self, pending, ctx, k):
        drafts, t = [], pending
        for _ in range(k):
            t = self.succ.get(t)
            if t is None:
                break
            drafts.append(t)
        return drafts


# ---------- decoders ----------


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
            nxt = int(mx.argmax(logits[0, -1]))
            if hasattr(proposer, "observe"):
                proposer.observe([pending], [nxt])
            pending = nxt
            continue
        rounds += 1
        inp = [pending] + drafts
        snap = snapshot(target.cache)
        logits = target.fwd(inp)
        cand = [int(x) for x in mx.argmax(logits[0], axis=-1)]
        if hasattr(proposer, "observe"):
            proposer.observe(inp, cand)
        j = 0
        while j < len(drafts) and cand[j] == drafts[j]:
            j += 1
        accepts.append(j)
        emitted = drafts[:j]
        out.extend(emitted)
        ctx.extend(emitted)
        pending = cand[j]  # chosen, unemitted, unconsumed
        if j < len(drafts):
            restore(target.cache, snap, len(drafts) - j)
            target.fwd([inp[0]] + drafts[:j])  # re-commit accepted state
    return out[:n], time.perf_counter() - t0, accepts, rounds


# ---------- scenarios ----------

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
    "open_chat": [
        {
            "role": "user",
            "content": "Write a short story about a lighthouse keeper "
            "who discovers something unusual in the fog.",
        }
    ],
}


def main():
    model, tok = load(MODEL)
    print(f"model loaded: {MODEL}")

    methods = {
        "baseline": None,
        "ngram": lambda: NgramDraft(),
        "recycle": lambda: RecycleDraft(),
        "layerskip": lambda: LayerSkipDraft(model, keep=list(range(8))),
        "sparsekv": lambda: SparseKVDraft(model, window=64),
    }

    for scen, msgs in SCENARIOS.items():
        prompt = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )
        print(f"\n=== {scen} (prompt {len(prompt)} tok) ===")
        ref_out = None
        for name, mk in methods.items():
            target = Target(model)
            proposer = mk() if mk else None
            if proposer is None:
                out, dt = baseline_decode(target, prompt, GEN_TOKENS)
                acc, rnds = [], 0
                draft_rows = 0
            else:
                out, dt, acc, rnds = spec_decode(target, proposer, prompt, GEN_TOKENS)
                draft_rows = getattr(proposer, "rows", 0)
            ref_out = out if ref_out is None else ref_out
            ok = out == ref_out
            tps = len(out) / dt
            avg_acc = sum(acc) / max(len(acc), 1)
            print(
                f"{name:10s} {tps:6.1f} tok/s | rows/token {target.rows / len(out):5.2f} "
                f"| draft_rows {draft_rows:4d} | avg_accept {avg_acc:4.2f} "
                f"| rounds {rnds:3d} | lossless={ok}"
            )


if __name__ == "__main__":
    main()

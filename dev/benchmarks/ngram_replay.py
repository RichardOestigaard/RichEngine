"""Replay recorded token traces against n-gram proposer variants — no model.

Reads JSONL written by the server under RICHENGINE_TRACE_TOKENS (see
server/backend.py::trace_tokens): one record per completed request with the
full prompt ids and emitted output ids.

For each request the replay walks the recorded output the way speculative
decode would: at every committed position the proposer sees ctx = prompt +
output[:i] and pending = output[i]; its chain accepts the longest prefix
matching the recorded continuation, and the step commits 1 + j tokens. On a
greedy (temperature 0) trace this is exact — the recorded output IS what the
target would re-emit. Sampled requests are an upper bound.

Variants:
  v1   - per-request fixed 3-gram (production NgramIndex.hpp semantics)
  v2f  - longest-suffix match, all occurrences, majority-extension chain
  v2fg - v2f + global corpus = every earlier request's output
         (SuffixDecoding global tree)

Usage: python dev/benchmarks/ngram_replay.py TRACE.jsonl [--k 15]
"""

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ngram_proposers import NgramV1, NgramV2F, NgramV2P


def simulate(proposer, prompt, output, k):
    """Spec-decode walk over the recorded stream. Returns
    (steps, accepted, fired, p_histogram)."""
    ctx = list(prompt)
    i, steps, accepted, fired = 0, 0, 0, 0
    p_hist = Counter()
    while i < len(output) - 1:
        pending = output[i]
        drafts = proposer.propose(pending, ctx, k)
        ctx.append(pending)
        i += 1
        if drafts:
            fired += 1
            if hasattr(proposer, "last_match_p"):
                p_hist[proposer.last_match_p] += 1
            j = 0
            while (
                j < len(drafts)
                and i + j < len(output)
                and drafts[j] == output[i + j]
            ):
                j += 1
            ctx.extend(drafts[:j])
            i += j
            accepted += j
        steps += 1
    return steps, accepted, fired, p_hist


def session_of(record, earlier):
    """A request continues a session when an earlier request's whole prompt
    is a prefix of its prompt (append-only chat history), or it names one
    via previous_id."""
    prompt = record["prompt"]
    if record.get("previous_id"):
        for e in earlier:
            if e["id"] == record["previous_id"]:
                return e.get("_session", e["id"])
    for e in reversed(earlier):
        ep = e["prompt"]
        if len(ep) <= len(prompt) and prompt[: len(ep)] == ep:
            return e.get("_session", e["id"])
    return record["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--min-out", type=int, default=4)
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--arm-min", type=int, default=8)
    ap.add_argument("--corpus-all", action="store_true",
                    help="corpus = prompt+output per request")
    ap.add_argument("--variant", action="append", default=None)
    args = ap.parse_args()

    records = []
    with open(args.trace) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("created_at", 0))
    records = [r for r in records if len(r.get("output", [])) >= args.min_out]
    if not records:
        print("no usable records")
        return

    def bare(corpus):
        """Session-tagged corpus docs stripped to plain token lists."""
        return [c[0] if isinstance(c, tuple) else c for c in corpus]

    variants = {
        "v1": lambda corpus, hinted, sess: NgramV1(),
        "v2f": lambda corpus, hinted, sess: NgramV2F(),
        "v2fg": lambda corpus, hinted, sess: NgramV2F(
            global_corpus=bare(corpus)
        ),
        # Opt-in prototype: hinted requests get the corpus matcher; the
        # rest keep production v1. Signals: session continuation
        # (prompt-prefix containment / previous_id) or a suffix probe hit —
        # the prompt's closing 16 tokens already present in the corpus.
        "v2opt": lambda corpus, hinted, sess: (
            NgramV2F(global_corpus=bare(corpus)) if hinted else NgramV1()
        ),
        # Production-faithful corpus proposer (NgramIndex.hpp semantics):
        # fixed-3 + extension, margin, self-arming probes.
        "v2p": lambda corpus, hinted, sess: NgramV2P(
            global_corpus=corpus, margin=args.margin,
            arm_min=args.arm_min, mode=2,
        ),
        # Tier-1 lever matrix (all on corpus docs, margin 5, mode 2):
        "v2p+occ": lambda corpus, hinted, sess: NgramV2P(
            global_corpus=corpus, margin=args.margin,
            arm_min=args.arm_min, mode=2, occ_all=True,
        ),
        "v2p+grp": lambda corpus, hinted, sess: NgramV2P(
            global_corpus=corpus, margin=args.margin,
            arm_min=args.arm_min, mode=2, grouped=True,
        ),
        "v2p+ses": lambda corpus, hinted, sess: NgramV2P(
            global_corpus=corpus, margin=args.margin,
            arm_min=args.arm_min, mode=2, session=sess,
        ),
        "v2p+all": lambda corpus, hinted, sess: NgramV2P(
            global_corpus=corpus, margin=args.margin,
            arm_min=args.arm_min, mode=2, occ_all=True, grouped=True,
            session=sess,
        ),
    }
    if args.variant:
        variants = {k: v for k, v in variants.items() if k in args.variant}

    def suffix_probe_hit(prompt, corpus, n=16):
        tail = prompt[-n:]
        return any(
            any(c[i : i + len(tail)] == tail for i in range(len(c) - n))
            for c in bare(corpus)
        ) if len(prompt) >= n else False

    totals = {v: dict(steps=0, accepted=0, fired=0, emitted=0) for v in variants}
    p_hist_all = Counter()
    global_corpus = []
    earlier = []

    print(f"{'req':<10} {'sess':<10} {'out':>5} {'temp':>4}  " +
          "  ".join(f"{v:>16}" for v in variants))
    print("     (tok/s columns show accepted/step · fired%)")
    for r in records:
        r["_session"] = session_of(r, earlier)
        hinted = r["_session"] != r["id"] or suffix_probe_hit(
            r["prompt"], global_corpus
        )
        r["_hint"] = hinted
        row = []
        for name, mk in variants.items():
            prop = mk(global_corpus, hinted, r["_session"])
            steps, accepted, fired, p_hist = simulate(
                prop, r["prompt"], r["output"], args.k
            )
            t = totals[name]
            t["steps"] += steps
            t["accepted"] += accepted
            t["fired"] += fired
            t["emitted"] += len(r["output"])
            p_hist_all.update(p_hist)
            speedup = len(r["output"]) / max(steps, 1)
            row.append(
                f"{speedup:5.2f}x {accepted / max(fired, 1):4.1f}a "
                f"{100 * fired / max(steps, 1):3.0f}%"
            )
        print(
            f"{r['id'][:9]:<10} {r['_session'][:9]:<10} "
            f"{len(r['output']):>5} {r.get('temperature') or 0:>4}  "
            + "  ".join(row)
            + ("   <== hint" if hinted else "")
        )
        earlier.append(r)
        doc = list(r["output"])
        if args.corpus_all:
            doc = list(r["prompt"]) + doc
        global_corpus.append((doc, r["_session"]))

    print("\n=== totals ===")
    for name, t in totals.items():
        steps = max(t["steps"], 1)
        print(
            f"{name:<5} speedup {t['emitted'] / steps:5.2f}x  "
            f"accepted/step {t['accepted'] / steps:5.2f}  "
            f"fired {100 * t['fired'] / steps:3.0f}%  "
            f"avg/round {t['accepted'] / max(t['fired'], 1):5.2f}"
        )
    if p_hist_all:
        print("\nmatched pattern length p (v2f/v2fg):")
        for p in sorted(p_hist_all):
            print(f"  p={p:3d}  {p_hist_all[p]:4d}")


if __name__ == "__main__":
    main()

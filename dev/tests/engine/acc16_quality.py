#!/usr/bin/env python3
"""RICHENGINE_ATTENTION_ACC16 model-quality validation.

Drives ``richengine serve-native`` through server.runtime.MultiplexedRuntime
directly (no HTTP server) and runs an identical request stream under two
engine legs: baseline (fp32 PV accumulator) and ACC16 (bfloat accumulator,
RICHENGINE_ATTENTION_ACC16=1). Both legs share this process's environment
minus/plus the flag, so the only difference is the kernel selection.

Phases:

  score   Teacher-forced logit deltas. Score-only wire requests over a fixed
          corpus: per position the engine returns raw logits for the true
          next token plus a fixed ~250-token probe pool. Metrics: |delta
          logit| on the true token and pool, candidate-set argmax flips,
          and decision-margin sign changes. Exercises the prefill attention
          plan's _a16 kernels.

  greedy  Greedy-decode parity. Fixed prompts at temperature 0, N tokens.
          Metrics: first-divergence index, token agreement, exact match.
          Exercises the verify/decode attention plan's _a16 kernels.

  needle  Retrieval under long context. A unique fact is planted at several
          depths of 8K/32K-token fillers; the model must repeat the code.
          Metrics: answer correctness per leg. Stress case for accumulated
          bf16 error, which grows with context length.

Usage:

  .venv/bin/python dev/tests/engine/acc16_quality.py \
      --model-dir install/models/mlx-community/Qwen3.8-27B-4bit \
      --binary build/richengine --out /tmp/acc16_report.json

  --legs base,a16        restrict to one leg (default both)
  --phase score,greedy   restrict phases (default all)
  --stride N             score-position stride (default 3)
  --max-positions N      cap scored positions per text (default 220)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from server import protocol as wire  # noqa: E402
from server import runtime as engine_runtime  # noqa: E402

# ---------------------------------------------------------------- corpus --

CORPUS = [
    # prose / knowledge
    "The history of the printing press is usually told through Gutenberg's "
    "forty-two-line Bible, but movable type had circulated in Korean workshops "
    "for decades before metal type reached Mainz. What changed in Europe was "
    "not the idea of printing but the economics of it: a punch-cut matrix and "
    "a hand mould let one workshop produce identical letters at a rate that "
    "scribes could never match, and within fifty years more books existed in "
    "Europe than in the previous millennium. Scholars estimate that by 1500 "
    "roughly twenty million volumes had been printed, an output that took "
    "monasteries centuries to approach by hand.",
    # code explanation
    "def merge_sort(items):\n    if len(items) <= 1:\n        return items\n"
    "    mid = len(items) // 2\n    left = merge_sort(items[:mid])\n"
    "    right = merge_sort(items[mid:])\n    return merge(left, right)\n\n"
    "def merge(left, right):\n    out = []\n    i = j = 0\n"
    "    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n            out.append(left[i])\n"
    "            i += 1\n        else:\n            out.append(right[j])\n"
    "            j += 1\n    return out + left[i:] + right[j:]\n\n"
    "Merge sort runs in O(n log n) time because each merge level does linear "
    "work and the recursion tree has log n levels. It is stable, which makes "
    "it the sort of choice when equal keys must keep their input order.",
    # math
    "To integrate x squared times e to the x, begin with integration by "
    "parts and set u equal to x squared and dv equal to e to the x dx. Then "
    "du is two x dx and v is e to the x. The integral becomes x squared e to "
    "the x minus the integral of two x e to the x dx. Apply parts again with "
    "u equal to two x and dv equal to e to the x dx, giving two x e to the x "
    "minus two e to the x. Combining terms yields e to the x times x squared "
    "minus two x plus two, plus a constant of integration.",
    # instructions / how-to
    "Sourdough bread begins with a starter: flour and water left to ferment "
    "until wild yeasts colonize the mixture. Feed the starter equal parts "
    "flour and water twice a day until it doubles reliably within four "
    "hours. For the dough, mix four hundred grams of starter with five "
    "hundred grams of bread flour, three hundred sixty grams of water, and "
    "ten grams of salt. Rest the shaggy mixture for thirty minutes, then "
    "perform four sets of stretch and folds spaced thirty minutes apart. "
    "Bulk ferment until the dough has risen fifty percent, shape it tightly, "
    "and proof overnight in the refrigerator before baking in a covered pot "
    "at two hundred thirty degrees Celsius.",
    # dialogue-ish / QA
    "Q: Why does the sky appear blue during the day?\n"
    "A: Sunlight contains every color of the visible spectrum, and air "
    "molecules scatter short wavelengths far more strongly than long ones. "
    "Blue light scatters roughly four times more than red, so the diffuse "
    "sky we see is dominated by blue. At sunrise and sunset the light "
    "travels through more atmosphere, scattering away most of the blue and "
    "leaving the reds and oranges that paint the horizon.\n"
    "Q: What is the difference between weather and climate?\n"
    "A: Weather is the state of the atmosphere at a particular place and "
    "time, while climate is the long-term average of weather patterns over "
    "decades. A single cold day does not disprove a warming trend, just as "
    "one hot afternoon does not establish one.",
    # scientific
    "Photosynthesis converts light energy into chemical energy through two "
    "linked stages. In the light-dependent reactions, chlorophyll absorbs "
    "photons and splits water molecules, releasing oxygen and generating "
    "ATP and NADPH. The Calvin cycle then fixes carbon dioxide into "
    "glyceraldehyde-3-phosphate using that energy. C4 plants spatially "
    "separate carbon fixation from the Calvin cycle to reduce photorespiry, "
    "while CAM plants separate them in time, opening their stomata only at "
    "night to conserve water in arid environments.",
    # multilingual
    "Laphotosynthese convertit l'energie lumineuse en energie chimique par "
    "deux etapes liees. Dans les reactions dependantes de la lumiere, la "
    "chlorophylle absorbe les photons et scinde les molecules d'eau, "
    "liberant de l'oxygene et produisant de l'ATP et du NADPH. Le cycle de "
    "Calvin fixe ensuite le dioxyde de carbone en glyceraldehyde-3-phosphate "
    "grace a cette energie. Les plantes C4 separent spatialement la fixation "
    "du carbone du cycle de Calvin pour reduire la photorespiration.",
    # structured / data
    "{\n  \"order_id\": \"A-1042\",\n  \"customer\": \"M. Okafor\",\n"
    "  \"items\": [\n    {\"sku\": \"KB-87\", \"qty\": 2, \"unit_price\": 49.99},\n"
    "    {\"sku\": \"MS-203\", \"qty\": 1, \"unit_price\": 129.50}\n  ],\n"
    "  \"shipping\": {\"method\": \"express\", \"eta_days\": 2},\n"
    "  \"total\": 229.48,\n  \"currency\": \"USD\"\n}\n\n"
    "The order total equals the sum of quantity times unit price across "
    "line items: two keyboards at forty-nine ninety-nine plus one mouse at "
    "one hundred twenty-nine fifty gives two hundred twenty-nine forty-eight "
    "dollars before tax and shipping surcharges are applied at checkout.",
]

# Short greedy-parity prompts. Long-context cases are generated below.
GREEDY_PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome, "
    "ignoring case and punctuation. Include a docstring and two examples.",
    "Explain the difference between TCP and UDP to a junior engineer in "
    "about five sentences.",
    "List three practical ways to reduce the energy consumption of a data "
    "center, with a one-line justification for each.",
    "Write the first paragraph of a short story about a lighthouse keeper "
    "who discovers something unusual in the logbook.",
    "What are the main arguments for and against four-day work weeks? "
    "Structure your answer as two balanced paragraphs.",
    "Translate the following sentence into formal French: 'The meeting has "
    "been moved to Thursday at three o'clock.' Then explain any choices "
    "you made.",
    "A train leaves Station A at 9:00 traveling 60 km/h and another leaves "
    "Station B, 300 km away, at 9:30 traveling 90 km/h toward A. When do "
    "they meet? Show your reasoning step by step.",
    "Describe the water cycle in exactly six sentences, each under "
    "twenty words.",
]

NEEDLE_FACT = "The secret verification code is ZX-417."
NEEDLE_QUERY = (
    "Read the passage carefully. At the end, answer this question with "
    "only the code itself: what is the secret verification code?"
)

FILLER_SENTENCE = (
    "Routine maintenance log entry {index}: inspected the north corridor "
    "lighting, replaced two ballast units, verified the fire panel shows "
    "no faults, and restocked the supply cabinet with filters and tape. "
)


def filler_tokens(tokenizer, count: int, offset: int = 0) -> list[int]:
    """Deterministic filler text tokenized to roughly `count` tokens."""
    out: list[int] = []
    i = offset
    while len(out) < count:
        text = " ".join(FILLER_SENTENCE.format(index=j) for j in range(i, i + 8))
        out.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        i += 8
    return out[:count]


def _sp(name):
    """Special-token literal without the angle-bracket source form."""
    return "<" + "|" + name + "|" + ">"


def chat_wrap(tokenizer, body_ids, thinking=False):
    """Chat-template wrapper around raw body token ids (enable_thinking
    shape): the body sits inside the user turn."""
    pre = tokenizer(_sp("im_start") + "user\n",
                    add_special_tokens=False)["input_ids"]
    think_open = "<" + "think" + ">"
    think_close = "<" + "/think" + ">"
    if thinking:
        post = tokenizer(
            _sp("im_end") + "\n" + _sp("im_start") + "assistant\n"
            + think_open + "\n",
            add_special_tokens=False,
        )["input_ids"]
    else:
        post = tokenizer(
            _sp("im_end") + "\n" + _sp("im_start") + "assistant\n"
            + think_open + "\n\n" + think_close + "\n\n",
            add_special_tokens=False,
        )["input_ids"]
    return pre + list(body_ids) + post


def template_ids(tokenizer, messages, **kwargs):
    from collections.abc import Mapping

    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **kwargs
    )
    ids = out["input_ids"] if isinstance(out, Mapping) else out
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


# ----------------------------------------------------------------- engine --


def native_command(binary: Path, model_dir: Path, kv_format: str) -> list[str]:
    command = [
        str(binary),
        "serve-native",
        str(model_dir),
        "auto",
        "auto",
    ]
    if kv_format != "int4":
        command.extend(("--kv-format", kv_format))
    return command


def leg_env(acc16: bool) -> dict:
    env = dict(os.environ)
    env.pop("RICHENGINE_ATTENTION_ACC16", None)
    if acc16:
        env["RICHENGINE_ATTENTION_ACC16"] = "1"
    return env


def start_leg(binary, model_dir, kv_format, acc16, startup_timeout):
    runtime = engine_runtime.MultiplexedRuntime(
        native_command(binary, model_dir, kv_format),
        startup_timeout=startup_timeout,
        pending_limit=64,
        env=leg_env(acc16),
    )
    t0 = time.monotonic()
    runtime.wait_ready(timeout=startup_timeout)
    return runtime, time.monotonic() - t0


def score_request(prompt: tuple[int, ...], candidates: tuple[int, ...],
                  prefix: tuple[int, ...], deadline: float):
    frame = wire.RequestFrame(
        request_id=0,
        priority=wire.RequestPriority.NORMAL,
        absolute_deadline_unix_micros=0,
        remaining_deadline_micros=0,
        logical_max_output_tokens=0,
        prompt_tokens=tuple(prompt),
        prefix_tokens=tuple(prefix),
        sampling=wire.SamplingParameters(),
        seed=0,
        constraint=wire.ConstraintMode.NONE,
        image_spans=(),
        image_pixels=b"",
        audio_spans=(),
        audio_mel=b"",
        return_progress=False,
        score_tokens=tuple(candidates),
        generation_prompt_tokens=0,
        flags=wire.RequestFlag(0),
    )
    return engine_runtime.GenerationRequest(frame, deadline, None, None)


def generation_prompt_tokens(tokenizer, thinking: bool) -> int:
    """Trailing generation-prompt token count, matching what the frontend
    computes for the chat template's generation suffix."""
    text = _sp("im_start") + "assistant\n" + "<" + "think" + ">\n"
    if not thinking:
        text += "\n<" + "/think" + ">\n\n"
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def generation_request(prompt: tuple[int, ...], max_tokens: int,
                       deadline: float, gen_prompt_tokens: int = 0):
    frame = wire.RequestFrame(
        request_id=0,
        priority=wire.RequestPriority.NORMAL,
        absolute_deadline_unix_micros=0,
        remaining_deadline_micros=0,
        logical_max_output_tokens=max_tokens,
        prompt_tokens=tuple(prompt),
        sampling=wire.SamplingParameters(),
        seed=0,
        constraint=wire.ConstraintMode.NONE,
        image_spans=(),
        image_pixels=b"",
        audio_spans=(),
        audio_mel=b"",
        return_progress=False,
        score_tokens=(),
        generation_prompt_tokens=gen_prompt_tokens,
        flags=wire.RequestFlag(0),
    )
    return engine_runtime.GenerationRequest(frame, deadline, None, None)


# ------------------------------------------------------------- phase: score --


def run_score_phase(runtime, tokenizer, args):
    """Teacher-forced candidate logits over the corpus. Returns
    {text_index: {position: {token: logit}}}."""
    texts = []
    for raw in CORPUS:
        ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
        texts.append(ids)
    if args.score_long_tokens:
        # One long repetitive text: mid-range context coverage plus a
        # worst-case-ish attention profile (many similar keys).
        texts.append(filler_tokens(tokenizer, args.score_long_tokens))

    # Probe pool: frequent corpus tokens plus seeded random vocabulary ids.
    freq = Counter(t for ids in texts for t in ids)
    vocab_size = len(tokenizer)
    rng = random.Random(20260714)
    pool = [t for t, _ in freq.most_common(args.pool_size)]
    seen = set(pool)
    while len(pool) < args.pool_size + args.random_pool:
        t = rng.randrange(vocab_size)
        if t not in seen:
            seen.add(t)
            pool.append(t)

    results = {}
    deadline = time.monotonic() + args.score_deadline
    batch_size = args.batch_size
    for ti, ids in enumerate(texts):
        positions = list(range(args.score_start, len(ids), args.stride))
        if args.max_positions and len(positions) > args.max_positions:
            step = len(positions) / args.max_positions
            positions = [
                positions[min(len(positions) - 1, int(i * step))]
                for i in range(args.max_positions)
            ]
        if not positions:
            continue
        # Shared prefix: everything before the first scored position's
        # prompt start. Item i's full prompt is ids[:i]; with prefix
        # ids[:first-1]... simpler: prefix = ids[:positions[0]] minus one
        # token so every suffix stays non-empty, and the first item covers
        # position positions[0].
        first = positions[0]
        prefix_len = max(1, first - 1)
        prefix = tuple(ids[:prefix_len])
        items = []
        meta = []
        for pos in positions:
            # prompt = ids[:pos] -> prefix ids[:prefix_len] + suffix
            # ids[prefix_len:pos]; scored token is ids[pos].
            suffix = tuple(ids[prefix_len:pos])
            if not suffix:
                continue
            true_tok = ids[pos]
            candidates = list(pool)
            if true_tok not in candidates:
                candidates.append(true_tok)
            candidates = candidates[: wire.MAX_SCORE_TOKENS]
            # protocol requires distinct candidates
            candidates = list(dict.fromkeys(candidates))
            items.append(
                score_request(suffix, tuple(candidates), prefix, deadline)
            )
            meta.append((pos, true_tok, tuple(candidates)))
        text_results = {}
        for chunk_start in range(0, len(items), batch_size):
            chunk_items = items[chunk_start : chunk_start + batch_size]
            chunk_meta = meta[chunk_start : chunk_start + batch_size]
            calls = runtime.submit_batch(
                chunk_items, [None] * len(chunk_items),
                [None] * len(chunk_items),
            )
            for call, (pos, true_tok, candidates) in zip(calls, chunk_meta):
                done = call.result(timeout=args.request_timeout)
                logits = done.option_logits
                if len(logits) != len(candidates):
                    raise RuntimeError(
                        f"score returned {len(logits)} logits for "
                        f"{len(candidates)} candidates"
                    )
                text_results[pos] = {
                    "true": true_tok,
                    "candidates": candidates,
                    "logits": logits,
                }
        results[ti] = text_results
        print(
            f"  [score] text {ti}: {len(text_results)} positions "
            f"({time.monotonic():.0f})",
            flush=True,
        )
    return results


# ---------------------------------------------------------- phase: greedy --


def collect_generation(runtime, prompt_ids, max_tokens, deadline,
                       request_timeout, gen_prompt_tokens=0):
    events: list[wire.TokensEvent] = []

    def on_event(call, event):
        if isinstance(event, wire.TokensEvent):
            events.append(event)

    call = runtime.submit(
        generation_request(
            tuple(prompt_ids), max_tokens, deadline, gen_prompt_tokens
        ),
        on_event=on_event,
    )
    done = call.result(timeout=request_timeout)
    tokens = []
    for ev in events:
        tokens.extend(ev.tokens)
    return tokens[: done.completion_tokens], done


def greedy_prompts(tokenizer, args):
    prompts = []
    for text in GREEDY_PROMPTS:
        prompts.append(
            template_ids(
                tokenizer, [{"role": "user", "content": text}]
            )
        )
    # Long-context cases: filler + question inside the user turn.
    for ctx in args.greedy_contexts:
        filler = filler_tokens(tokenizer, ctx)
        q_ids = tokenizer(
            NEEDLE_QUERY, add_special_tokens=False
        )["input_ids"]
        prompts.append(chat_wrap(tokenizer, filler + q_ids, thinking=True))
    return prompts


def run_greedy_phase(runtime, tokenizer, args):
    prompts = greedy_prompts(tokenizer, args)
    gpt = generation_prompt_tokens(tokenizer, True)
    out = []
    for i, ids in enumerate(prompts):
        deadline = time.monotonic() + args.request_timeout
        tokens, done = collect_generation(
            runtime, ids, args.greedy_tokens, deadline,
            args.request_timeout, gpt,
        )
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        out.append(
            {
                "index": i,
                "prompt_tokens": len(ids),
                "generated": tokens,
                "text": text,
                "finish": int(done.reason),
                "wall_ms": done.wall_micros / 1000.0,
            }
        )
        print(
            f"  [greedy] prompt {i}: {len(tokens)} tokens "
            f"({done.wall_micros / 1e6:.1f}s)",
            flush=True,
        )
    return out


# ---------------------------------------------------------- phase: needle --


def run_needle_phase(runtime, tokenizer, args):
    results = []
    gpt = generation_prompt_tokens(tokenizer, False)
    for ctx in args.needle_contexts:
        for depth in args.needle_depths:
            pos = int(ctx * depth)
            head = filler_tokens(tokenizer, pos)
            fact = tokenizer(NEEDLE_FACT, add_special_tokens=False)[
                "input_ids"
            ]
            tail_count = ctx - pos - len(fact)
            tail = filler_tokens(tokenizer, max(0, tail_count), offset=999)
            query = tokenizer(NEEDLE_QUERY, add_special_tokens=False)[
                "input_ids"
            ]
            body = head + fact + tail + query
            prompt = chat_wrap(tokenizer, body, thinking=False)
            deadline = time.monotonic() + args.request_timeout * 4
            tokens, done = collect_generation(
                runtime, prompt, args.needle_tokens, deadline,
                args.request_timeout * 4, gpt,
            )
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            hit = "ZX-417" in text or "417" in text
            results.append(
                {
                    "ctx": ctx,
                    "depth": depth,
                    "prompt_tokens": len(prompt),
                    "hit": hit,
                    "text": text[:200],
                }
            )
            print(
                f"  [needle] ctx={ctx} depth={depth}: "
                f"{'HIT' if hit else 'MISS'} — {text[:60]!r}",
                flush=True,
            )
    return results


# ----------------------------------------------------------------- compare --


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def compare_score(base, a16):
    deltas_true, deltas_pool = [], []
    flips = 0
    margin_sign_flips = 0
    positions = 0
    texts = sorted(set(base) | set(a16))
    for ti in texts:
        bpos, apos = base.get(ti, {}), a16.get(ti, {})
        for pos in sorted(set(bpos) & set(apos)):
            b, a = bpos[pos], apos[pos]
            if b["true"] != a["true"] or b["candidates"] != a["candidates"]:
                continue
            positions += 1
            logits_b = dict(zip(b["candidates"], b["logits"]))
            logits_a = dict(zip(a["candidates"], a["logits"]))
            tb = b["true"]
            deltas_true.append(logits_a[tb] - logits_b[tb])
            for tok in b["candidates"]:
                deltas_pool.append(abs(logits_a[tok] - logits_b[tok]))
            # argmax over the candidate pool under each leg
            am_b = max(b["candidates"], key=lambda t: logits_b[t])
            am_a = max(a["candidates"], key=lambda t: logits_a[t])
            if am_b != am_a:
                flips += 1
            # margin of the true token vs the best alternative
            def margin(logits):
                others = [logits[t] for t in logits if t != tb]
                return logits[tb] - max(others) if others else float("inf")
            if (margin(logits_b) > 0) != (margin(logits_a) > 0):
                margin_sign_flips += 1
    abs_true = [abs(d) for d in deltas_true]
    return {
        "positions": positions,
        "true_logit_delta": {
            "mean": sum(deltas_true) / len(deltas_true) if deltas_true else None,
            "mean_abs": sum(abs_true) / len(abs_true) if abs_true else None,
            "p50_abs": pct(abs_true, 0.5),
            "p95_abs": pct(abs_true, 0.95),
            "max_abs": max(abs_true) if abs_true else None,
        },
        "pool_logit_delta_abs": {
            "mean": sum(deltas_pool) / len(deltas_pool) if deltas_pool else None,
            "p95": pct(deltas_pool, 0.95),
            "max": max(deltas_pool) if deltas_pool else None,
        },
        "candidate_argmax_flips": flips,
        "argmax_flip_rate": flips / positions if positions else None,
        "true_token_margin_sign_flips": margin_sign_flips,
        "margin_sign_flip_rate": (
            margin_sign_flips / positions if positions else None
        ),
    }


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare_greedy(base, a16):
    pairs = []
    for b, a in zip(base, a16):
        bt, at = b["generated"], a["generated"]
        agree = sum(1 for x, y in zip(bt, at) if x == y)
        pairs.append(
            {
                "index": b["index"],
                "prompt_tokens": b["prompt_tokens"],
                "base_tokens": len(bt),
                "a16_tokens": len(at),
                "first_divergence": lcp(bt, at),
                "agreement": agree / max(1, min(len(bt), len(at))),
                "exact": bt == at,
            }
        )
    lcps = [p["first_divergence"] for p in pairs]
    return {
        "prompts": pairs,
        "median_first_divergence": pct(lcps, 0.5),
        "min_first_divergence": min(lcps) if lcps else None,
        "exact_matches": sum(1 for p in pairs if p["exact"]),
        "mean_agreement": (
            sum(p["agreement"] for p in pairs) / len(pairs) if pairs else None
        ),
    }


def compare_needle(base, a16):
    rows = []
    for b, a in zip(base, a16):
        rows.append(
            {
                "ctx": b["ctx"],
                "depth": b["depth"],
                "base_hit": b["hit"],
                "a16_hit": a["hit"],
                "agree": b["hit"] == a["hit"],
            }
        )
    return {
        "cases": rows,
        "base_hits": sum(r["base_hit"] for r in rows),
        "a16_hits": sum(r["a16_hit"] for r in rows),
        "disagreements": [r for r in rows if not r["agree"]],
    }


# -------------------------------------------------------------------- main --


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="install/models/mlx-community/Qwen3.8-27B-4bit")
    ap.add_argument("--binary", default="build/richengine")
    ap.add_argument("--kv-format", default="int8")
    ap.add_argument("--out", default="")
    ap.add_argument("--legs", default="base,a16")
    ap.add_argument("--phase", default="score,greedy,needle")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-positions", type=int, default=220)
    ap.add_argument("--score-start", type=int, default=8)
    ap.add_argument("--score-long-tokens", type=int, default=3000)
    ap.add_argument("--pool-size", type=int, default=200)
    ap.add_argument("--random-pool", type=int, default=54)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--greedy-tokens", type=int, default=256)
    ap.add_argument("--needle-tokens", type=int, default=48)
    ap.add_argument("--needle-contexts", default="8192,32768")
    ap.add_argument("--needle-depths", default="0.1,0.5,0.9")
    ap.add_argument("--greedy-contexts", default="4096,16384")
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--score-deadline", type=float, default=7200.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir)
    binary = Path(args.binary)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))

    legs = [s.strip() for s in args.legs.split(",") if s.strip()]
    phases = [s.strip() for s in args.phase.split(",") if s.strip()]
    args.needle_contexts = [int(x) for x in args.needle_contexts.split(",")]
    args.needle_depths = [float(x) for x in args.needle_depths.split(",")]
    args.greedy_contexts = [int(x) for x in args.greedy_contexts.split(",")]

    report = {"model_dir": str(model_dir), "kv_format": args.kv_format,
              "legs": {}, "comparison": {}}
    leg_data = {}

    for leg in legs:
        acc16 = leg == "a16"
        print(f"=== leg {leg} (acc16={acc16}) ===", flush=True)
        runtime, load_s = start_leg(
            binary, model_dir, args.kv_format, acc16, args.startup_timeout
        )
        print(f"  engine ready in {load_s:.1f}s", flush=True)
        data = {"load_s": load_s}
        try:
            if "score" in phases:
                print("  phase: score", flush=True)
                data["score"] = run_score_phase(runtime, tokenizer, args)
            if "greedy" in phases:
                print("  phase: greedy", flush=True)
                data["greedy"] = run_greedy_phase(runtime, tokenizer, args)
            if "needle" in phases:
                print("  phase: needle", flush=True)
                data["needle"] = run_needle_phase(runtime, tokenizer, args)
        finally:
            runtime.close()
        leg_data[leg] = data
        report["legs"][leg] = {k: v for k, v in data.items() if k != "score"}
        # score raw logits are huge; keep them in the JSON only on request
        if args.out:
            report["legs"][leg]["score"] = data.get("score")

    if "base" in leg_data and "a16" in leg_data:
        b, a = leg_data["base"], leg_data["a16"]
        if "score" in b and "score" in a:
            report["comparison"]["score"] = compare_score(b["score"], a["score"])
        if "greedy" in b and "greedy" in a:
            report["comparison"]["greedy"] = compare_greedy(
                b["greedy"], a["greedy"]
            )
        if "needle" in b and "needle" in a:
            report["comparison"]["needle"] = compare_needle(
                b["needle"], a["needle"]
            )

    # Console summary
    comp = report["comparison"]
    print("\n=== comparison ===")
    if "score" in comp:
        s = comp["score"]
        t = s["true_logit_delta"]
        print(
            f"score: {s['positions']} positions | "
            f"true-token |Δlogit| mean={t['mean_abs']:.4f} "
            f"p95={t['p95_abs']:.4f} max={t['max_abs']:.4f} | "
            f"pool |Δ| p95={s['pool_logit_delta_abs']['p95']:.4f} | "
            f"argmax flips={s['candidate_argmax_flips']} "
            f"({s['argmax_flip_rate']:.2%}) | "
            f"margin sign flips={s['true_token_margin_sign_flips']} "
            f"({s['margin_sign_flip_rate']:.2%})"
        )
    if "greedy" in comp:
        g = comp["greedy"]
        print(
            f"greedy: {len(g['prompts'])} prompts | "
            f"median first-divergence={g['median_first_divergence']} tok | "
            f"min={g['min_first_divergence']} | "
            f"exact={g['exact_matches']} | "
            f"mean agreement={g['mean_agreement']:.3f}"
        )
        for p in g["prompts"]:
            print(
                f"  prompt {p['index']} (ctx {p['prompt_tokens']}): "
                f"div@{p['first_divergence']} "
                f"agree={p['agreement']:.3f} exact={p['exact']}"
            )
    if "needle" in comp:
        n = comp["needle"]
        print(
            f"needle: base {n['base_hits']} hits, a16 {n['a16_hits']} hits, "
            f"{len(n['disagreements'])} disagreements"
        )
        for r in n["cases"]:
            mark = "" if r["agree"] else "  <-- DISAGREE"
            print(
                f"  ctx={r['ctx']} depth={r['depth']}: "
                f"base={'HIT' if r['base_hit'] else 'MISS'} "
                f"a16={'HIT' if r['a16_hit'] else 'MISS'}{mark}"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nreport: {args.out}")


if __name__ == "__main__":
    main()

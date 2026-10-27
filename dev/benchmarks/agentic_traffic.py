"""Drive realistic agentic traffic at a running server so the
RICHENGINE_TRACE_TOKENS capture has real multi-turn token streams to
replay (dev/benchmarks/ngram_replay.py).

Usage:
  RICHENGINE_TRACE_TOKENS=/tmp/trace.jsonl ./richengine serve --model M &
  python dev/benchmarks/agentic_traffic.py [--base http://127.0.0.1:8123]
                                          [--model MODEL_ID]

Sessions cover the shapes n-gram drafting should exploit: edit loops that
re-emit files, a fresh session repeating the same task (cross-request
corpus reuse), tool-argument JSON loops, self-refinement fix loops,
verbatim quoting, structured-log rendering, and an open-chat control.
"""

import argparse
import json
import urllib.request

CODE = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def quick_check(items):
    total = 0
    for it in items:
        total += it.price * it.qty
    return total
"""

LOG = "\n".join(
    f"2026-10-27T05:4{i}:0{i}Z LVL{i % 3} worker-{i % 4} "
    f"processed batch {i} ok={i * 7 % 3 == 0}"
    for i in range(12)
)

DOC = """The kakapo (Strigops habroptilus) is a large, flightless, nocturnal parrot
endemic to New Zealand. It can weigh up to 4 kg and live over 90 years.
Kakapo breed only in years when rimu trees mast heavily, sometimes just
once every four years. As of 2024 fewer than 250 individuals remain, each
named and tracked by conservation rangers on predator-free islands."""


class Traffic:
    def __init__(self, base, model):
        self.url = base.rstrip("/") + "/v1/chat/completions"
        self.model = model

    def call(self, messages, max_tokens=512):
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        self.last = data
        return data["choices"][0]["message"]["content"]

    def session(self, name, turns, max_tokens=512):
        msgs = []
        for i, text in enumerate(turns):
            msgs.append({"role": "user", "content": text})
            reply = self.call(msgs, max_tokens) or ""
            msgs.append({"role": "assistant", "content": reply})
            # Server-side decode metrics: the production A/B reads these.
            decode = (self.last.get("metrics") or {}).get("decode") or {}
            tps = (
                (self.last.get("timings") or {}).get("predicted_per_second")
            )
            print(
                f"{name} t{i}: {len(reply)} chars | "
                f"{tps or 0:.1f} tok/s | "
                f"draft {decode.get('accepted_draft_tokens', 0)}/"
                f"{decode.get('drafted_tokens', 0)}",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    t = Traffic(args.base, args.model)

    # Edit loop: rename, diff, re-emit, second rename — the model's own
    # output re-enters the prompt each turn.
    t.session(
        "edit_loop",
        [
            f"Here is my code:\n{CODE}\nRepeat it exactly but rename "
            "bubble_sort to sort_array everywhere. Output only code.",
            "Now emit it as a unified diff against the original. "
            "Output only the diff.",
            "Repeat the renamed file once more, verbatim.",
            "Also rename quick_check to fast_check and emit the whole "
            "file again, verbatim.",
        ],
    )

    # Fresh session, same task — reuses session A's emissions only via a
    # global corpus; the prompt alone can't supply the emitted form.
    t.session(
        "edit_again",
        [
            f"Here is my code:\n{CODE}\nRepeat it exactly but rename "
            "bubble_sort to sort_array everywhere. Output only code.",
            "Emit that renamed file once more, verbatim.",
        ],
    )

    # Tool-argument loop: JSON bodies repeating a file's contents with a
    # changing path — the emission format differs from the raw file.
    t.session(
        "toolargs",
        [
            'Call write_file with JSON: {"path": "src/sort.py", '
            '"content": "<the renamed file contents>"}. Output only JSON.',
            "Same call again, path src/sort_v2.py. Output only JSON.",
            "Same call again, path src/sort_v3.py. Output only JSON.",
        ],
    )

    # Self-refinement fix loop: each turn re-emits most of the previous
    # function — the classic agent retry pattern.
    t.session(
        "fix_loop",
        [
            "Write a Python function `parse_log_line(line)` returning "
            "(ts, level, worker, msg) from a log line. Output only code.",
            "It raised ValueError on empty lines — fix it and emit the "
            "full function again. Output only code.",
            "Still fails on lines with missing fields — handle that and "
            "emit the full function again. Output only code.",
        ],
    )

    # Structured rendering: repeated row structure with >4 identical
    # trigram contexts — stresses v1's last-4-starts cap.
    t.session(
        "log_table",
        [
            f"Render this log as a markdown table with columns "
            f"ts | level | worker | msg, one row per line:\n{LOG}",
            "Same table again but only the WARN and ERROR rows.",
        ],
    )

    # Verbatim quoting.
    t.session(
        "rag",
        [
            f"Document:\n{DOC}\n\nQuote the two sentences containing "
            "weight and lifespan, verbatim.",
            "Quote them again, same order.",
        ],
    )

    # Control: open-ended generation.
    t.session(
        "chat",
        [
            "Write a short story about a lighthouse keeper who finds a "
            "strange fog bank."
        ],
    )
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agentic-workflow A/B for generated-history junctions + SuffixReplay.

Simulates a tool-use agent: each step the model's reply is appended plus a
synthetic tool result, so the prompt grows through generated tokens each
round — the exact case junctions over generated history target. Mid-run,
two parallel sibling continuations fork off the shared history.

Metric per step: prompt_n (tokens actually prefilled), prompt_ms (TTFT),
cached_tokens (matched prefix). Compares RICHENGINE_NO_HISTORY_JUNCTIONS
off (=new junctions) vs on (=old prompt-only behavior).

Usage: .venv/bin/python dev/benchmarks/agentic_workflow_ab.py [BINARY] [MODEL]
"""
import json, os, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ornith-ai/Ornith-1.5-9B-MLX-4bit"
TASK = ("You are a research agent. First, outline 5 search queries about "
        "migratory birds, one per line.")
TOOL_RESULTS = [
    "TOOL search: found 12 papers on arctic tern routes; top hit doi:10.1/x42",
    "TOOL read: abstract — terns cover 70k km/yr; key figure 3 maps routes",
    "TOOL search: godwit nonstop flight records; 11 days, 12000 km noted",
    "TOOL compare: frigatebird stays aloft months via dynamic soaring",
]

class Args:
    def __init__(self, package, binary):
        self.package, self.binary = package, binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context, self.max_memory = 16384, None
        self.max_cache_disk = self.max_image_pixels = None
        self.kv_format = "int8"


def chat(port, messages, comp, timeout=300):
    s, d = smoke_real.request(port, "POST", "/v1/chat/completions",
                              {"model": MODEL, "temperature": 0,
                               "max_completion_tokens": comp,
                               "messages": messages}, timeout=timeout)
    if s != 200:
        return None
    u, t = d.get("usage", {}), d.get("timings", {})
    return {"reply": smoke_real.answer_text(d),
            "prompt_n": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "tok_s": round(t.get("predicted_per_second", 0), 1)}


def run(binary, off):
    pkg = ROOT / "install/models" / MODEL
    env = dict(os.environ)
    if off:
        env["RICHENGINE_NO_HISTORY_JUNCTIONS"] = "1"
    server = smoke_real.RealServer(Args(pkg, binary), env)
    steps, fork = [], {}
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        p = server.port
        messages = [{"role": "user", "content": TASK}]
        # Sequential agent rounds: each prompt embeds all prior generated
        # text plus a fresh tool result.
        for tool in TOOL_RESULTS:
            r = chat(p, messages, 160)
            if not r:
                return {"ok": False, "error": "step failed", "steps": steps}
            steps.append({k: v for k, v in r.items() if k != "reply"})
            messages += [{"role": "assistant", "content": r["reply"]},
                         {"role": "user", "content": tool}]
        # Parallel fork: two continuations off the shared generated history.
        results = {}
        def sibling(tag, q):
            r = chat(p, messages + [{"role": "user", "content": q}], 32)
            results[tag] = ({k: v for k, v in r.items() if k != "reply"}
                            if r else {"error": True})
        ts = [threading.Thread(target=sibling, args=(t, q))
              for t, q in (("a", "Summarize in one sentence."),
                           ("b", "Which bird flies farthest?"))]
        for t in ts: t.start()
        for t in ts: t.join()
        return {"ok": True, "variant": "junctions_off" if off else "junctions_on",
                "steps": steps, "fork": results}
    except Exception as e:
        return {"ok": False, "error": str(e).replace("\n", " ")[:140],
                "steps": steps, "fork": fork}
    finally:
        server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    for off in (False, True):
        print(json.dumps(run(binary, off)), flush=True)


if __name__ == "__main__":
    main()

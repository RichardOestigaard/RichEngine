#!/usr/bin/env python3
"""SuffixReplay multi-round A/B: force every turn's end onto a KV page
boundary (prompt+completion == 0 mod 32) so the finish() publication can
fire each round, then watch turn N prompt cost with the flag on vs off.

Usage: .venv/bin/python dev/benchmarks/suffix_replay_rounds.py [BINARY] [MODEL]
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ornith-ai/Ornith-1.5-9B-MLX-4bit"
USERS = ["List the first 40 prime numbers, one per line.",
         "Now list their squares, one per line.",
         "Now sum them in running order, one total per line.",
         "In one line, say how many numbers you wrote overall."]

class Args:
    def __init__(self, package, binary):
        self.package, self.binary = package, binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context, self.max_memory = 8192, None
        self.max_cache_disk = self.max_image_pixels = None
        self.kv_format = "int8"


def chat(port, messages, comp):
    body = {"model": MODEL, "temperature": 0, "max_completion_tokens": comp,
            "messages": messages}
    return smoke_real.request(port, "POST", "/v1/chat/completions", body,
                              timeout=300)


def run(binary, on):
    pkg = ROOT / "install/models" / MODEL
    env = dict(os.environ)
    if on:
        env["RICHENGINE_SUFFIX_REPLAY"] = "1"
    server = smoke_real.RealServer(Args(pkg, binary), env)
    rounds = []
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        p = server.port
        messages = []
        for user in USERS:
            messages.append({"role": "user", "content": user})
            # probe: learn this turn's rendered prompt length
            s, d = chat(p, messages, 1)
            if s != 200:
                return {"ok": False, "error": str(d)[:120], "rounds": rounds}
            pn = d.get("usage", {}).get("prompt_tokens") or 0
            gap = (32 - pn % 32) % 32
            comp = 32 * 10 + gap          # enough text + boundary alignment
            s, d = chat(p, messages, comp)
            if s != 200:
                return {"ok": False, "error": str(d)[:120], "rounds": rounds}
            u = d.get("usage", {})
            t = d.get("timings", {})
            total = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
            rounds.append({"prompt_n": u.get("prompt_tokens"),
                           "prompt_ms": round(t.get("prompt_ms", 0), 1),
                           "total": total, "aligned": total % 32 == 0})
            messages.append({"role": "assistant",
                             "content": smoke_real.answer_text(d)})
        return {"ok": True, "variant": "on" if on else "off", "rounds": rounds}
    except Exception as e:
        return {"ok": False, "variant": "on" if on else "off",
                "error": str(e).replace("\n", " ")[:140], "rounds": rounds}
    finally:
        server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    for on in (True, False):
        print(json.dumps(run(binary, on)), flush=True)


if __name__ == "__main__":
    main()

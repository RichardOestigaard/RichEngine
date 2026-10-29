#!/usr/bin/env python3
"""SuffixReplay forced-fire probe: end turn-1 exactly on a KV page boundary
(32 tokens) so the finish() publication fires, then measure how much
recurrent-layer replay turn-2 skips (richengine_cache_resource_replay_tokens).

Usage: .venv/bin/python dev/benchmarks/suffix_replay_fire.py [BINARY] [MODEL]
"""
import json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ornith-ai/Ornith-1.5-9B-MLX-4bit"
PROMPT = "Count from 1 to 200, one number per line."

class Args:
    def __init__(self, package, binary):
        self.package, self.binary = package, binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context, self.max_memory = 8192, None
        self.max_cache_disk = self.max_image_pixels = None
        self.kv_format = "int8"


def _find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            got = _find(v, key)
            if got is not None:
                return got
    if isinstance(node, list):
        for v in node:
            got = _find(v, key)
            if got is not None:
                return got
    return None


def replay_tokens(port):
    status, doc = smoke_real.request(port, "GET", "/status", None)
    if status != 200 or not isinstance(doc, dict):
        return None
    return _find(doc, "resource_replay_tokens")


def run(binary, off):
    pkg = ROOT / "install/models" / MODEL
    env = dict(os.environ)
    if not off:
        env["RICHENGINE_SUFFIX_REPLAY"] = "1"
    server = smoke_real.RealServer(Args(pkg, binary), env)
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        p = server.port
        # Step 1: measure the rendered prompt length with a tiny request.
        body = {"model": MODEL, "max_completion_tokens": 1, "temperature": 0,
                "messages": [{"role": "user", "content": PROMPT}]}
        s, d = smoke_real.request(p, "POST", "/v1/chat/completions", body, timeout=120)
        prompt_n = d.get("usage", {}).get("prompt_tokens") if s == 200 else None
        if not prompt_n:
            return {"ok": False, "error": f"prompt probe failed {s}"}
        # Step 2: turn-1 generation sized so prompt+completion lands on a
        # page boundary (32): the finish() publication then fires.
        gap = (32 - prompt_n % 32) % 32
        comp = 32 * 12 + gap          # ~1 page of padding + alignment
        body["max_completion_tokens"] = comp
        s, d1 = smoke_real.request(p, "POST", "/v1/chat/completions", body, timeout=300)
        u1 = d1.get("usage", {})
        total = (u1.get("prompt_tokens") or 0) + (u1.get("completion_tokens") or 0)
        r0 = replay_tokens(p)
        # Step 3: turn-2 reuses prompt+generation; follow-up resumes from
        # the deepest published state.
        text = smoke_real.answer_text(d1) if s == 200 else ""
        body["messages"] = [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "What was the last number you wrote?"}]
        body["max_completion_tokens"] = 16
        s, d2 = smoke_real.request(p, "POST", "/v1/chat/completions", body, timeout=120)
        r1 = replay_tokens(p)
        t2 = d2.get("timings", {}) if s == 200 else {}
        return {"ok": s == 200, "variant": "off" if off else "on",
                "prompt_n": prompt_n, "comp": comp,
                "turn1_total": total, "aligned": total % 32 == 0,
                "replay_delta": (r1 - r0) if r0 is not None and r1 is not None else None,
                "turn2_prompt_n": t2.get("prompt_n"),
                "turn2_prompt_ms": round(t2.get("prompt_ms", 0), 1)}
    except Exception as e:
        return {"ok": False, "variant": "off" if off else "on",
                "error": str(e).replace("\n", " ")[:140]}
    finally:
        server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    for off in (False, True):
        print(json.dumps(run(binary, off)), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SuffixReplay A/B: two-turn chat per model, turn-2 prefix reuse.

Usage: .venv/bin/python dev/benchmarks/suffix_replay_ab.py [BINARY] [off|on|both]
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    "ornith-ai/Ornith-1.5-9B-MLX-4bit",       # hybrid GDN (qwen3_5)
    "LiquidAI/LFM2.5-2.6B-GGUF:QAD-Q4_0",     # lfm2 hybrid conv+attn
    "openbmb/MiniCPM5-2B-MLX",                # dense llama control
]

class Args:
    def __init__(self, package, binary):
        self.package = package
        self.binary = binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context = 8192
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def probe(binary, name, off):
    pkg = ROOT / "install/models" / name
    env = dict(os.environ)
    if off:
        env["RICHENGINE_SUFFIX_REPLAY_OFF"] = "1"
    server = smoke_real.RealServer(Args(pkg, binary), env)
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        p = server.port
        # turn 1: force a generation of a few hundred tokens
        body = {"model": name,
                "messages": [{"role": "user",
                              "content": "Write a detailed 400-word story about a lighthouse keeper."}],
                "max_completion_tokens": 400, "temperature": 0}
        s, d1 = smoke_real.request(p, "POST", "/v1/chat/completions", body, timeout=300)
        if s != 200:
            return {"model": name, "ok": False, "error": str(d1)[:120]}
        text = smoke_real.answer_text(d1)
        # turn 2: same conversation + follow-up -> prefix reuse through generation
        body["messages"] = [
            {"role": "user", "content": "Write a detailed 400-word story about a lighthouse keeper."},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Summarize it in one sentence."}]
        body["max_completion_tokens"] = 32
        s, d2 = smoke_real.request(p, "POST", "/v1/chat/completions", body, timeout=300)
        t1 = d1.get("timings", {})
        t2 = d2.get("timings", {})
        u2 = d2.get("usage", {})
        return {"model": name, "ok": True,
                "turn1_completion": u2 and d1.get("usage", {}).get("completion_tokens"),
                "turn2_prompt_tokens": u2.get("prompt_tokens"),
                "turn2_cache_n": t2.get("cache_n"),
                "turn2_prompt_ms": t2.get("prompt_ms"),
                "turn2_tok_s": round(t2.get("predicted_per_second", 0), 1),
                "timings2": {k: v for k, v in t2.items() if k != "ms"}} \
            if s == 200 else {"model": name, "ok": False, "error": str(d2)[:120]}
    except Exception as e:
        return {"model": name, "ok": False, "error": str(e).replace("\n", " ")[:140]}
    finally:
        server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"
    for name in MODELS:
        for off in ([False, True] if mode == "both" else [mode == "off"]):
            r = probe(binary, name, off)
            r["variant"] = "off" if off else "on"
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()

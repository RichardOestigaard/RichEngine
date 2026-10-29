#!/usr/bin/env python3
"""Opt-in flag A/B: parity (greedy output equality) + decode tok/s.

Variants: baseline / each experimental flag. Long prompt (>32K chars)
so sparse selection paths actually engage on QSA arches; on non-qwen4exp
models the QSA flags must be inert (output must match exactly).

Usage: .venv/bin/python dev/benchmarks/qsa_flags_ab.py [BINARY] [models...]
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    "ornith-ai/Ornith-1.5-9B-MLX-4bit",
    "openbmb/MiniCPM5-2B-MLX",
]
VARIANTS = [
    ("baseline", {}),
    ("suffix_replay", {"RICHENGINE_SUFFIX_REPLAY": "1"}),
    ("index_share_4", {"RICHENGINE_QSA_INDEX_SHARE": "4"}),
    ("fused_topk", {"RICHENGINE_QSA_FUSED_TOPK": "1"}),
    ("window_256_8", {"RICHENGINE_QSA_DECODE_WINDOW": "256",
                      "RICHENGINE_QSA_DECODE_WINDOW_SINK": "8"}),
    ("locks", {"RICHENGINE_QSA_LOCKS": "1"}),
]
# ~36K chars of prompt text -> thousands of tokens; sparse paths engage.
LONG_PROMPT = ("Explain step by step how a suspension bridge distributes "
               "load. " * 900)
LONG_PROMPT += ("\nNow: in one paragraph, summarize the key idea about the "
                "towers.")

class Args:
    def __init__(self, package, binary):
        self.package = package
        self.binary = binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context = 65536
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def run_model(binary, name):
    pkg = ROOT / "install/models" / name
    baseline_text = None
    for variant, flags in VARIANTS:
        env = dict(os.environ)
        env.update(flags)
        server = smoke_real.RealServer(Args(pkg, binary), env)
        try:
            smoke_real.validate_status(server.wait_ready(300), "int8")
            body = {"model": name,
                    "messages": [{"role": "user", "content": LONG_PROMPT}],
                    "max_completion_tokens": 96, "temperature": 0}
            s, doc = smoke_real.request(server.port, "POST",
                                        "/v1/chat/completions", body,
                                        timeout=600)
            if s != 200:
                out = {"model": name, "variant": variant, "ok": False,
                       "error": str(doc)[:120]}
            else:
                t = doc.get("timings", {})
                text = smoke_real.answer_text(doc)
                if variant == "baseline":
                    baseline_text = text
                out = {"model": name, "variant": variant, "ok": True,
                       "prompt_n": t.get("prompt_n"),
                       "prompt_ms": round(t.get("prompt_ms", 0), 1),
                       "tok_s": round(t.get("predicted_per_second", 0), 1),
                       "parity": (text == baseline_text)
                       if baseline_text is not None else None}
            print(json.dumps(out), flush=True)
        except Exception as e:
            print(json.dumps({"model": name, "variant": variant, "ok": False,
                              "error": str(e).replace("\n", " ")[:140]}),
                  flush=True)
        finally:
            server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
        else ROOT / "build/richengine"
    models = sys.argv[2:] or MODELS
    for name in models:
        run_model(binary, name)


if __name__ == "__main__":
    main()

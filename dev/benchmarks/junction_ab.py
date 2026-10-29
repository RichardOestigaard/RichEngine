#!/usr/bin/env python3
"""Generated-history junction A/B: a producer generates; then two siblings
share [prompt + generation] as their prompt prefix and run concurrently —
the leader arms a junction inside the shared generated region, the follower
resumes from it. Compare follower's prompt work new-binary vs old.

Usage: .venv/bin/python dev/benchmarks/junction_ab.py [BINARY]
"""
import json, os, sys, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ornith-ai/Ornith-1.5-9B-MLX-4bit"

class Args:
    def __init__(self, package, binary):
        self.package, self.binary = package, binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context, self.max_memory = 8192, None
        self.max_cache_disk = self.max_image_pixels = None
        self.kv_format = "int8"


def chat(port, messages, comp=64):
    return smoke_real.request(port, "POST", "/v1/chat/completions",
                              {"model": MODEL, "temperature": 0,
                               "max_completion_tokens": comp,
                               "messages": messages}, timeout=300)


def run(binary):
    pkg = ROOT / "install/models" / MODEL
    server = smoke_real.RealServer(Args(pkg, binary), dict(os.environ))
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        p = server.port
        s, d = chat(p, [{"role": "user", "content":
                         "List 80 interesting facts about octopuses, "
                         "one per line."}], 400)
        if s != 200:
            return {"ok": False, "error": str(d)[:120]}
        reply = smoke_real.answer_text(d)
        base = [{"role": "user", "content":
                 "List 80 interesting facts about octopuses, one per line."},
                {"role": "assistant", "content": reply}]
        results = {}
        def sibling(tag, question):
            s2, d2 = chat(p, base + [{"role": "user", "content": question}], 24)
            t = d2.get("timings", {}) if s2 == 200 else {}
            u = d2.get("usage", {}) if s2 == 200 else {}
            results[tag] = {"prompt_n": u.get("prompt_tokens"),
                            "prompt_ms": round(t.get("prompt_ms", 0), 1),
                            "cached": (u.get("prompt_tokens_details") or {})
                                .get("cached_tokens")}
        # Concurrent siblings: leader prefills + arms the deep junction while
        # the follower waits behind it.
        t1 = threading.Thread(target=sibling, args=("a", "Which fact is the oddest?"))
        t2 = threading.Thread(target=sibling, args=("b", "Which fact surprised you most?"))
        t1.start(); t2.start(); t1.join(); t2.join()
        return {"ok": True, "siblings": results}
    except Exception as e:
        return {"ok": False, "error": str(e).replace("\n", " ")[:140]}
    finally:
        server.close()


print(json.dumps(run(Path(sys.argv[1]).resolve() if len(sys.argv) > 1
                     else ROOT / "build/richengine")))

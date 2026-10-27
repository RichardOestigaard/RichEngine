#!/usr/bin/env python3
"""One-shot smoke matrix: for each installed model package, boot the server,
send one 64-token chat, report decode throughput or the startup error.

Usage: .venv/bin/python dev/benchmarks/model_matrix.py [BINARY]
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = sorted(
    f"{ns.name}/{p.name}"
    for ns in (ROOT / "install/models").iterdir()
    if ns.is_dir() and not ns.name.startswith(".")
    for p in ns.iterdir()
    if p.is_dir()
)


class Args:
    def __init__(self, package, binary):
        self.package = package
        self.binary = binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context = 4096
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def probe(binary, name):
    pkg = ROOT / "install/models" / name
    server = smoke_real.RealServer(Args(pkg, binary))
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        body = {
            "model": name,
            "messages": [{"role": "user", "content": "Count to ten."}],
            "max_completion_tokens": 64,
        }
        s, doc = smoke_real.request(
            server.port, "POST", "/v1/chat/completions", body, timeout=120)
        if s != 200:
            err = (doc.get("error") or {}).get("message", "?")[:90] if isinstance(doc, dict) else "?"
            return {"model": name, "ok": False, "error": err}
        t = doc.get("timings", {})
        return {
            "model": name, "ok": True,
            "tok_s": round(t.get("predicted_per_second", 0), 1),
            "cached": t.get("cache_n", 0),
        }
    except Exception as e:
        tail = str(e).replace("\n", " ")[:140]
        return {"model": name, "ok": False, "error": tail}
    finally:
        server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    for name in MODELS:
        print(json.dumps(probe(binary, name)), flush=True)


if __name__ == "__main__":
    main()

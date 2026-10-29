#!/usr/bin/env python3
"""Token-ID-stable A/B for generated-history junctions.

/v1/completions accepts `prompt` as a list of token ids — no re-encoding,
so a generated prefix matches the KV chain exactly. Two siblings share
[prompt + generation] as prompt ids and run back-to-back; with junctions
on, the leader arms a state boundary inside the shared generated region
and the follower resumes from it (smaller prompt_n, shallower GDN replay).

Counters: usage.prompt_tokens, prompt_tokens_details.cached_tokens,
and /status resources.stateCache.junctionMaterializations.

Usage: .venv/bin/python dev/benchmarks/token_stable_ab.py [BINARY] [MODEL]
"""
import json, os, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "ornith-ai/Ornith-1.5-9B-MLX-4bit"
SEED_TEXT = ("Write a numbered list of 120 detailed facts about deep-sea "
             "hydrothermal vent ecosystems.")

class Args:
    def __init__(self, package, binary):
        self.package, self.binary = package, binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context, self.max_memory = 16384, None
        self.max_cache_disk = self.max_image_pixels = None
        self.kv_format = "int8"


def completions(port, prompt, comp, timeout=300):
    s, d = smoke_real.request(port, "POST", "/v1/completions",
                              {"model": MODEL, "prompt": prompt,
                               "max_tokens": comp, "temperature": 0},
                              timeout=timeout)
    if s != 200:
        return None
    u, t = d.get("usage", {}), d.get("timings", {})
    return {"prompt_n": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "tok_s": round(t.get("predicted_per_second", 0), 1),
            "text": d.get("choices", [{}])[0].get("text", "")}


def tokenize(port, text):
    s, d = smoke_real.request(port, "POST", "/tokenize",
                              {"model": MODEL, "prompt": text}, timeout=60)
    if s != 200:
        return None
    return d.get("tokens") or d.get("input_ids")


def junctions(port):
    s, d = smoke_real.request(port, "GET", "/status", None)
    if s != 200 or not isinstance(d, dict):
        return None
    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for v in node.values():
                r = find(v, key)
                if r is not None:
                    return r
        if isinstance(node, list):
            for v in node:
                r = find(v, key)
                if r is not None:
                    return r
        return None
    return find(d, "junctionMaterializations") or find(d, "junction_materializations")


def run(binary, off):
    pkg = ROOT / "install/models" / MODEL
    trace = tempfile.NamedTemporaryFile(
        prefix="trace_", suffix=".jsonl", delete=False).name
    env = dict(os.environ)
    env["RICHENGINE_TRACE_TOKENS"] = trace
    if off:
        env["RICHENGINE_NO_HISTORY_JUNCTIONS"] = "1"
    server = smoke_real.RealServer(Args(pkg, binary), env)
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        p = server.port
        # Producer: text prompt -> trace captures exact prompt+output ids.
        prod = completions(p, SEED_TEXT, 400)
        if not prod:
            return {"ok": False, "error": "producer failed"}
        j0 = junctions(p)
        # Recover exact ids from the trace.
        records = [json.loads(l) for l in open(trace) if l.strip()]
        last = records[-1]
        prefix = last["prompt"] + last["output"]
        # Two siblings share prompt+generation as raw ids; different tails.
        tail_a = tokenize(p, "\nFact 1 was: ") or [1, 2, 3]
        tail_b = tokenize(p, "\nFact 7 says: ") or [4, 5, 6]
        leader = completions(p, prefix + tail_a, 24)
        # Follower lands while the leader's prefill may still be armed;
        # either way its admission sees the published junction.
        follower = completions(p, prefix + tail_b, 24)
        # A second pair, concurrent this time, for the in-flight planning path.
        results = {}
        def sib(tag, tail):
            results[tag] = completions(p, prefix + tail, 24)
        t1 = threading.Thread(target=sib, args=("c", tail_a + [999]))
        t2 = threading.Thread(target=sib, args=("d", tail_b + [998]))
        t1.start(); t2.start(); t1.join(); t2.join()
        j1 = junctions(p)
        return {"ok": True,
                "variant": "junctions_off" if off else "junctions_on",
                "producer_prompt_n": prod["prompt_n"],
                "prefix_tokens": len(prefix),
                "leader": {k: v for k, v in leader.items() if k != "text"},
                "follower": {k: v for k, v in follower.items() if k != "text"},
                "concurrent": {t: ({k: v for k, v in r.items() if k != "text"}
                                  if r else {"error": True})
                               for t, r in results.items()},
                "junctions_before": j0, "junctions_after": j1}
    except Exception as e:
        return {"ok": False, "variant": "off" if off else "on",
                "error": str(e).replace("\n", " ")[:160]}
    finally:
        server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    for off in (False, True):
        print(json.dumps(run(binary, off)), flush=True)


if __name__ == "__main__":
    main()

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
import hashlib, json, os, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"
# Enough stable leading context to cross the checkpoint minimum and several
# KV pages; short chats deliberately do not pay for branch snapshots.
CONTEXT = ("Migration evidence: arctic terns cross hemispheres, godwits fly "
           "nonstop, and frigatebirds soar for months. ") * 180
TASK = ("You are a research agent. First, outline 5 search queries about "
        "migratory birds, one per line.")
# Tool output is deliberately long enough that an edited turn has real
# generated/agent history to replay after the shared system prefix.
def _tool_result(head, detail):
    return head + " " + (detail + " ") * 90


TOOL_RESULTS = [
    _tool_result("TOOL search: found 12 papers on arctic tern routes; top hit doi:10.1/x42.",
                 "Route tables compare colony, season, distance, stopover, wind, and arrival mass."),
    _tool_result("TOOL read: abstract — terns cover 70k km/yr; key figure 3 maps routes.",
                 "The abstract reports tracked birds, timing, corridor width, altitude, and uncertainty."),
    _tool_result("TOOL search: godwit nonstop flight records; 11 days, 12000 km noted.",
                 "Record entries note departure, landing, duration, telemetry quality, and weather."),
    _tool_result("TOOL compare: frigatebird stays aloft months via dynamic soaring.",
                 "Comparison rows cover soaring mode, energy cost, sleep, range, and evidence quality."),
]
# The benchmark compares engine reuse, not sampling. History therefore uses
# one fixed, substantial assistant answer in both variants rather than
# whatever slightly different text each process happened to decode.
CANONICAL_REPLY = (
    "1. Arctic terns migrate farthest. 2. Godwits hold the nonstop record. "
    "3. Frigatebirds stay aloft longest. The evidence compares distance, "
    "duration, telemetry quality, wind assistance, and uncertainty. "
) * 42
FINAL_INSTRUCTION = (
    "Using every tool result above, write the final comparison. Name the "
    "winner in each category, distinguish total migration distance from "
    "nonstop flight distance, mention evidence quality, and keep the answer "
    "under two hundred words. "
) * 4

class Args:
    def __init__(self, package, binary):
        self.package, self.binary = package, binary
        self.model = f"{package.parent.name}/{package.name}"
        self.max_context, self.max_memory = 32768, None
        self.max_cache_disk = self.max_image_pixels = None
        self.kv_format = "int8"
        self.branch_checkpoint_state = False


def chat(port, messages, comp, timeout=300):
    body = {"model": MODEL, "temperature": 0,
            "max_completion_tokens": comp,
            "messages": messages}
    # MiniCPM's template supports disabling its think block. LFM2.5's does
    # not, so the option stays model-dependent rather than breaking either.
    if "MiniCPM" in MODEL:
        body["reasoning_effort"] = "none"
    s, d = smoke_real.request(port, "POST", "/v1/chat/completions",
                              body, timeout=timeout)
    if s != 200:
        return None
    u, t = d.get("usage", {}), d.get("timings", {})
    return {"reply": smoke_real.answer_text(d),
            "prompt_n": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prompt_ms": round(t.get("prompt_ms", 0), 1),
            "tok_s": round(t.get("predicted_per_second", 0), 1)}


def run(binary, branch):
    pkg = ROOT / "install/models" / MODEL
    env = dict(os.environ)
    args = Args(pkg, binary)
    args.branch_checkpoint_state = branch
    server = smoke_real.RealServer(args, env)
    steps, fork = [], {}
    try:
        smoke_real.validate_status(server.wait_ready(300), "int8")
        p = server.port
        # Exclude one-time kernel and tokenizer warm-up from every timing.
        if not chat(p, [{"role": "user", "content": "warm up"}], 8):
            return {"ok": False, "error": "warmup failed", "steps": steps}
        messages = [{"role": "system", "content": CONTEXT},
                    {"role": "user", "content": TASK}]
        # Sequential agent rounds: each prompt embeds all prior generated
        # text plus a fresh tool result. The resent assistant text is fixed,
        # so branch-off and branch-on measure byte-identical conversations.
        for tool in TOOL_RESULTS:
            r = chat(p, messages, 64)
            if not r:
                return {"ok": False, "error": "step failed", "steps": steps}
            steps.append({**{k: v for k, v in r.items() if k != "reply"},
                          "reply_sha256": hashlib.sha256(r["reply"].encode()).hexdigest()[:12]})
            messages += [{"role": "assistant", "content": CANONICAL_REPLY},
                         {"role": "user", "content": tool}]
        messages.append({"role": "assistant", "content": CANONICAL_REPLY})
        base = list(messages)

        # The producer establishes the branch point before this final user
        # instruction. That instruction is long enough that the lane's own
        # page-aligned replay point can lie inside it, so an edit cannot use
        # the producer's end-of-prompt state without reprocessing.
        final_messages = base + [{"role": "user", "content": FINAL_INSTRUCTION}]
        r = chat(p, final_messages, 64)
        if not r:
            return {"ok": False, "error": "producer failed", "steps": steps}
        steps.append({**{k: v for k, v in r.items() if k != "reply"},
                      "reply_sha256": hashlib.sha256(r["reply"].encode()).hexdigest()[:12]})

        # Edit the final user instruction: everything before that content is
        # unchanged, the exact reuse a branch checkpoint is meant to serve.
        edited_messages = base + [
            {"role": "user", "content": FINAL_INSTRUCTION.replace("Using every tool result", "Using all tool evidence", 1)}
        ]
        r = chat(p, edited_messages, 768)
        edited = ({**{k: v for k, v in r.items() if k != "reply"},
                   "reply_sha256": hashlib.sha256(r["reply"].encode()).hexdigest()[:12]}
                  if r else {"error": True})
        # Parallel fork: two replacements for the same final user turn.
        results = {}
        def sibling(tag, q):
            r = chat(p, base + [{"role": "user", "content": q}], 64)
            results[tag] = ({**{k: v for k, v in r.items() if k != "reply"},
                             "reply_sha256": hashlib.sha256(r["reply"].encode()).hexdigest()[:12]}
                            if r else {"error": True})
        ts = [threading.Thread(target=sibling, args=(t, q))
              for t, q in (("a", FINAL_INSTRUCTION.replace("write the final comparison", "summarize the final comparison")),
                           ("b", FINAL_INSTRUCTION.replace("write the final comparison", "rank the birds by evidence quality")))]
        for t in ts: t.start()
        for t in ts: t.join()
        code, status = smoke_real.request(p, "GET", "/status")
        state = status.get("state", {}) if code == 200 else {}
        memory = {key: state.get(key) for key in
                  ("entries", "bytes", "checkpoint_entries", "checkpoint_bytes",
                   "checkpoint_evictions", "checkpoint_retirements")}
        return {"ok": True, "variant": "branch_on" if branch else "branch_off",
                "steps": steps, "edited": edited, "fork": results,
                "memory": memory}
    except Exception as e:
        return {"ok": False, "error": str(e).replace("\n", " ")[:140],
                "steps": steps, "fork": fork}
    finally:
        server.close()


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    results = [run(binary, branch) for branch in (False, True)]
    for result in results:
        print(json.dumps(result), flush=True)
    if all(result.get("ok") for result in results):
        print(json.dumps({"comparison": {
            "edited_ttft_ms": [result["edited"].get("prompt_ms") for result in results],
            "fork_ttft_ms": [[fork.get("prompt_ms") for fork in result["fork"].values()]
                             for result in results],
            "edited_reply_match": len({step.get("reply_sha256") for result in results
                                       for step in [result["edited"]]}) == 1,
            "edited_reply_nonempty": [bool(step.get("reply_sha256")) and step.get("reply_sha256") != "e3b0c44298fc"
                                      for result in results for step in [result["edited"]]],
            "memory": [result["memory"] for result in results],
        }}), flush=True)


if __name__ == "__main__":
    main()

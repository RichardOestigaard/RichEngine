import json, subprocess, sys, urllib.request

def metrics(port):
    out = {}
    for line in urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode().splitlines():
        p = line.split()
        if len(p) == 2:
            out[p[0]] = float(p[1])
    return out

def run(port, model, n=3, tokens=400):
    body = json.dumps({"model": model,
        "prompt": "Write a long detailed essay about the history of computing, covering mechanical calculators, vacuum tubes, transistors, integrated circuits, and microprocessors.",
        "max_tokens": tokens, "temperature": 0, "stream": False}).encode()
    b = metrics(port)
    for _ in range(n):
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions", data=body,
            headers={"Content-Type": "application/json"})).read()
    a = metrics(port)
    steps = a["richengine_scheduler_decode_batches_total"] - b["richengine_scheduler_decode_batches_total"]
    wall = (a["richengine_decode_wall_milliseconds_total"] - b["richengine_decode_wall_milliseconds_total"]) / 1000
    toks = a["richengine_decode_output_tokens_total"] - b["richengine_decode_output_tokens_total"]
    print(f"{port} {model}: {steps/wall:.1f} steps/s, {toks/wall:.1f} tok/s", flush=True)

pairs = [(int(p), m) for p, m in (x.split(" ", 1) for x in sys.argv[1:])]
for _ in range(2):
    for port, model in pairs:
        run(port, model)

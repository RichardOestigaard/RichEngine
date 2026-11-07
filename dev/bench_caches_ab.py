"""A/B benchmark: cache proposals 1-3 under simulated agentic flow.

Simulates K concurrent agent sessions. Each step every session appends a
turn (user / assistant+tool_calls / tool result) and resubmits the whole
conversation — the real agentic pattern. Times template render, prompt
tokenizer encode, and grammar/spec builds.
"""
import random
import sys
import time

sys.path.insert(0, ".")

import orjson
from transformers import AutoTokenizer

from server import dfa as dfa_compiler
from server import tool_schema
from server.tokenization import PromptTokenizer

TOK_DIR = "install/models/test/Ornith-9B-Distilled/tokenizer"
SESSIONS = 8
STEPS = 30

TOK = AutoTokenizer.from_pretrained(TOK_DIR)
MARKER = TOK.convert_ids_to_tokens(TOK.eos_token_id)
print("marker:", MARKER, "| eos:", TOK.eos_token)

# ---- realistic tool set (OpenAI-style, ~8 tools) ----
def _schema(props, required):
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": _schema({
            "path": {"type": "string", "description": "File path"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        }, ["path"]),
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": _schema({
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, ["path", "content"]),
    }},
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Search the codebase for a pattern.",
        "parameters": _schema({
            "pattern": {"type": "string"},
            "glob": {"type": "string"},
            "max_results": {"type": "integer"},
        }, ["pattern"]),
    }},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command.",
        "parameters": _schema({
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        }, ["command"]),
    }},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List directory contents.",
        "parameters": _schema({"path": {"type": "string"}}, []),
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": _schema({
            "query": {"type": "string"},
            "num_results": {"type": "integer"},
        }, ["query"]),
    }},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch a URL.",
        "parameters": _schema({"url": {"type": "string"}}, ["url"]),
    }},
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Update the task list.",
        "parameters": _schema({
            "todos": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {"type": "string"},
                },
            }},
        }, ["todos"]),
    }},
]

SYSTEM = (
    "You are a coding agent. You have access to tools for reading files, "
    "running commands, searching code, and managing tasks. Always think "
    "step by step, use tools when needed, and verify your work before "
    "declaring completion. " * 12
)

USER_TURNS = [
    "Fix the failing test in server/protocol.py.",
    "The test test_token_words_must_be_exact_uint32_ints fails on bool input.",
    "Can you also add a regression test for IntEnum rejection?",
    "Run the full protocol test suite and show me the results.",
    "Now check whether _words handles the 1M-token cap efficiently.",
    "Profile the change and tell me if the type scan is now cheaper.",
    "Update the docs comment in protocol.py to match the new behavior.",
    "Commit the change with a clear message.",
]

ASSISTANT_TURNS = [
    "I'll look at the failing test first, then trace _words.",
    "Found it — the bool slips through __index__. Patching _words now.",
    "Done. The early-exit scan preserves the exact-int contract.",
    "Tests pass: 37 protocol tests, 236 subtests, zero failures.",
    "The remaining scan is O(n) but single-pass; array does range checks in C.",
    "Profile shows ~0.5ms on a 50k-token prompt, down from ~3ms.",
    "Comment updated. The wire format is unchanged.",
    "Committed as perf: protocol word packing.",
]

TOOL_OUTPUT = (
    "test_token_words_must_be_exact_uint32_ints ... ok\n"
    "test_request_validation_runs_once ... ok\n"
    "test_score_request_rejects_generation_combinations ... ok\n"
    + "\n".join(
        f"  subtest case_{i}: passed in 0.001s" for i in range(120)
    )
    + "\n\n" + ("lorem ipsum dolor sit amet " * 60)
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "tests_run": {"type": "integer"},
        "confidence": {"type": "number"},
        "details": {
            "type": "object",
            "properties": {
                "before_ms": {"type": "number"},
                "after_ms": {"type": "number"},
                "speedup": {"type": "number"},
            },
            "required": ["before_ms", "after_ms", "speedup"],
        },
    },
    "required": ["summary", "files_changed", "tests_run"],
}


def make_session(seed):
    rng = random.Random(seed)
    messages = [{"role": "system", "content": SYSTEM}]
    return {"messages": messages, "rng": rng, "turn": 0}


def grow_session(sess):
    """Append one agentic turn; returns nothing (mutates)."""
    rng = sess["rng"]
    t = sess["turn"] % len(USER_TURNS)
    msgs = sess["messages"]
    msgs.append({"role": "user", "content": USER_TURNS[t]})
    msgs.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{sess['turn']}_{i}",
            "type": "function",
            "function": {
                "name": rng.choice(["read_file", "run_command", "search_code"]),
                "arguments": {"path": f"src/file_{i}.py", "command": "pytest -x"},
            },
        } for i in range(rng.choice([1, 1, 2]))],
    })
    calls = msgs[-1]["tool_calls"]
    for i in range(len(calls)):
        msgs.append({
            "role": "tool",
            "tool_call_id": calls[i]["id"],
            "content": TOOL_OUTPUT[: rng.choice([800, 2000, 4000])],
        })
    msgs.append({"role": "assistant", "content": ASSISTANT_TURNS[t]})
    sess["turn"] += 1


def render(sess):
    return TOK.apply_chat_template(
        sess["messages"],
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
    )


# ---------- proposal 1: PromptTokenizer CAPACITY ----------
def run_encode_bench(capacity):
    pt = PromptTokenizer(TOK, MARKER)
    pt.CAPACITY = capacity
    assert pt.enabled, "reply/prefix memory not enabled for this tokenizer"
    sessions = [make_session(i) for i in range(SESSIONS)]
    encode_ms = []
    render_ms = []
    for step in range(STEPS):
        for si, sess in enumerate(sessions):
            grow_session(sess)
            t0 = time.perf_counter()
            text = render(sess)
            render_ms.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            pt.encode(text)
            encode_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "encode_total_ms": sum(encode_ms),
        "encode_mean_ms": sum(encode_ms) / len(encode_ms),
        "encode_p95_ms": sorted(encode_ms)[int(len(encode_ms) * 0.95)],
        "hits": pt.hits,
        "misses": pt.misses if hasattr(pt, "misses") else "?",
        "entries": len(pt.entries),
        "render_mean_ms": sum(render_ms) / len(render_ms),
    }


# ---------- proposal 2: grammar/spec build cache ----------
def build_policy():
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
    return tool_schema.ToolPolicy(
        schemas=schemas,
        required=False,
        parallel=True,
        strict=frozenset({"read_file", "run_command"}),
        constrained=True,
    )


def bench_grammar(n=200):
    # warm cached_property once is NOT representative: real flow builds a
    # fresh ToolPolicy per request, so rebuild it each iteration.
    t_tool_grammar = t_tool_spec = t_json_grammar = t_json_spec = 0.0
    for _ in range(n):
        p = build_policy()
        t0 = time.perf_counter()
        tool_schema.tool_grammar(p, False)
        t_tool_grammar += time.perf_counter() - t0
        p = build_policy()
        t0 = time.perf_counter()
        dfa_compiler.tool_spec(p, False)
        t_tool_spec += time.perf_counter() - t0
        t0 = time.perf_counter()
        tool_schema.json_grammar(RESPONSE_SCHEMA, False)
        t_json_grammar += time.perf_counter() - t0
        t0 = time.perf_counter()
        dfa_compiler.json_spec(RESPONSE_SCHEMA, False)
        t_json_spec += time.perf_counter() - t0
    # cached path: canonical key + dict hit
    key_src = orjson.dumps({
        "s": {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS},
        "st": sorted(["read_file", "run_command"]),
        "rs": RESPONSE_SCHEMA,
    }, option=orjson.OPT_SORT_KEYS)
    t0 = time.perf_counter()
    for _ in range(n):
        k = orjson.dumps({
            "s": {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS},
            "st": sorted(["read_file", "run_command"]),
            "rs": RESPONSE_SCHEMA,
        }, option=orjson.OPT_SORT_KEYS)
        _ = {k: 1}.get(k)
    t_key = (time.perf_counter() - t0) / n * 1000
    return {
        "tool_grammar_ms": t_tool_grammar / n * 1000,
        "tool_spec_ms": t_tool_spec / n * 1000,
        "json_grammar_ms": t_json_grammar / n * 1000,
        "json_spec_ms": t_json_spec / n * 1000,
        "cache_key_ms": t_key,
        "key_bytes": len(key_src),
    }


# ---------- proposal 3: render memo ----------
def bench_render_memo(dup_rate=0.0):
    sessions = [make_session(100 + i) for i in range(SESSIONS)]
    rng = random.Random(7)
    memo = {}
    canon_ms = []
    render_ms = []
    hits = misses = 0
    for step in range(STEPS):
        for sess in sessions:
            grow_session(sess)
            msgs = sess["messages"]
            # occasional exact-duplicate resubmission (retry/regen)
            if dup_rate and rng.random() < dup_rate:
                pass  # same messages object -> identical render input
            t0 = time.perf_counter()
            key = orjson.dumps([msgs, {"add_generation_prompt": True, "tools": TOOLS}],
                               option=orjson.OPT_SORT_KEYS)
            canon_ms.append((time.perf_counter() - t0) * 1000)
            if key in memo:
                hits += 1
                continue
            misses += 1
            t0 = time.perf_counter()
            memo[key] = render(sess)
            render_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "canon_mean_ms": sum(canon_ms) / len(canon_ms),
        "render_mean_ms": (sum(render_ms) / len(render_ms)) if render_ms else 0,
        "hits": hits, "misses": misses,
        "memo_entries": len(memo),
    }


print("\n=== growing conversation sizes (last step, session 0) ===")
s = make_session(0)
for _ in range(STEPS):
    grow_session(s)
text = render(s)
print(f"messages={len(s['messages'])} chars={len(text)} tokens={len(TOK(text, add_special_tokens=False)['input_ids'])}")

print("\n=== proposal 1: encode, CAPACITY=4 vs 32 ===")
for cap in (4, 32):
    r = run_encode_bench(cap)
    print(f"cap={cap}: encode total={r['encode_total_ms']:.0f}ms "
          f"mean={r['encode_mean_ms']:.2f}ms p95={r['encode_p95_ms']:.2f}ms "
          f"hits={r['hits']} entries={r['entries']} render_mean={r['render_mean_ms']:.1f}ms")

print("\n=== proposal 2: grammar/spec build ===")
g = bench_grammar()
for k, v in g.items():
    print(f"  {k}: {v:.3f}" + (" ms" if "ms" in k else ""))
print(f"  per-request total (tool_grammar+tool_spec): {g['tool_grammar_ms']+g['tool_spec_ms']:.2f} ms")

print("\n=== proposal 3: render memo, pure agentic (0% dup) ===")
r3 = bench_render_memo(0.0)
print(r3)
print("=== proposal 3: render memo, 5% duplicate resubmits ===")
r3b = bench_render_memo(0.05)
print(r3b)

#!/usr/bin/env python3
"""Drive real tool-calling traffic against a serving instance.

Unlike tool-shaped prompt simulations, this benchmark executes the model's
filesystem tool calls in an isolated temporary workspace. It exercises the
full OpenAI tool protocol: assistant tool_calls, client-side execution,
tool results, and continued generation. Validation is compile-only; model
code is never executed.

Usage:
  .venv/bin/python dev/benchmarks/agentic_tool_execution.py [BINARY] [MODEL]
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file in the isolated workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the isolated workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List workspace files.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_python",
            "description": "Compile-check a Python file without executing it.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_tests",
            "description": "Run unittest discovery below a workspace directory with a ten-second timeout.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


class Args:
    def __init__(self, binary: Path, model: str, max_memory: str | None = "30G"):
        self.package = ROOT / "install/models" / model
        self.binary = binary
        self.model = model
        self.max_context = 16384
        self.max_memory = max_memory
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def safe_path(root: Path, value: str) -> Path:
    root = root.resolve()
    # Tool agents often write container-style absolute paths (/workspace/x or
    # /src/x). Treat the workspace as the filesystem root: every absolute
    # path is interpreted inside it, never on the host.
    virtual = value
    if virtual.startswith("/workspace/"):
        virtual = virtual[len("/workspace/"):]
    elif virtual.startswith("/"):
        virtual = virtual.lstrip("/")
    path = (root / virtual).resolve()
    if path != root and root not in path.parents:
        raise ValueError("path escapes workspace")
    return path


def execute_tool(root: Path, name: str, arguments: str) -> str:
    root = root.resolve()
    try:
        args = json.loads(arguments or "{}")
        if name == "write_file":
            path = safe_path(root, args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            return json.dumps({"ok": True, "path": args["path"], "bytes": len(args["content"])})
        if name == "read_file":
            path = safe_path(root, args["path"])
            return json.dumps({"ok": True, "path": args["path"], "content": path.read_text(encoding="utf-8")})
        if name == "list_files":
            files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
            return json.dumps({"ok": True, "files": files})
        if name == "validate_python":
            path = safe_path(root, args["path"])
            try:
                py_compile.compile(str(path), cfile=str(root / ".validation.pyc"), doraise=True)
                return json.dumps({"ok": True, "path": args["path"], "valid": True})
            except py_compile.PyCompileError as error:
                return json.dumps({"ok": True, "path": args["path"], "valid": False, "error": str(error)})
        if name == "run_python_tests":
            path = safe_path(root, args.get("path", "."))
            env = dict(os.environ)
            env.update({
                "PYTHONPATH": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": str(root),
            })
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(path), "-t", str(root), "-p", "test*.py"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = (proc.stdout + proc.stderr)[-3000:]
            return json.dumps({
                "ok": True,
                "path": args.get("path", "."),
                "passed": proc.returncode == 0,
                "returncode": proc.returncode,
                "output": output,
            })
        return json.dumps({"ok": False, "error": f"unknown tool {name}"})
    except Exception as error:  # tool errors are observations, not harness failures
        return json.dumps({"ok": False, "error": str(error)[:500]})


def chat(port: int, model: str, messages: list, tokens: int = 384, include_tools: bool = True):
    body = smoke_real.chat_body(
        model,
        "",
        temperature=0,
        max_completion_tokens=tokens,
        tools=TOOLS if include_tools else None,
    )
    if not include_tools:
        body.pop("tools", None)
    body["messages"] = messages
    status, doc = smoke_real.request(
        port, "POST", "/v1/chat/completions", body, timeout=600
    )
    if status != 200:
        raise RuntimeError(f"chat failed {status}: {doc}")
    return doc


def metrics(doc: dict) -> dict:
    timings = doc.get("timings") or {}
    decode = (doc.get("metrics") or {}).get("decode") or {}
    return {
        "tok_s": round(timings.get("predicted_per_second") or 0, 2),
        "completion_tokens": (doc.get("usage") or {}).get("completion_tokens", 0),
        "accepted": decode.get("accepted_draft_tokens", 0),
        "offered": decode.get("drafted_tokens", 0),
    }


def status_ngram(port: int):
    status, doc = smoke_real.request(port, "GET", "/status", timeout=10)
    return doc.get("ngram") if status == 200 else None


def run(binary: Path, model: str, max_memory: str | None = "30G") -> dict:
    server = smoke_real.RealServer(Args(binary, model, max_memory), dict(os.environ))
    executed = []
    turns = []
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        with tempfile.TemporaryDirectory(prefix="richengine-tools-") as tmp:
            workspace = Path(tmp)
            messages = [
                {
                    "role": "system",
                    "content": "You are a coding agent. Use the provided filesystem tools. "
                    "Do not claim a file exists before writing it. Keep final answers concise.",
                },
                {
                    "role": "user",
                    "content": f"The isolated workspace root is {workspace}. Prefer relative paths. "
                    "Create src/sort_tools.py containing a bubble_sort function and a "
                    "quick_check function. Then read it back, compile-check it, fix any problem "
                    "you find, and finish with a one-sentence summary.",
                },
            ]
            start = time.monotonic()
            final_text = ""
            for turn in range(8):
                doc = chat(server.port, model, messages)
                message = doc["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                content = message.get("content") or ""
                turns.append({"turn": turn, "tool_calls": len(calls), **metrics(doc)})
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        **({"tool_calls": calls} if calls else {}),
                    }
                )
                if not calls:
                    final_text = content
                    break
                validated = False
                for call in calls:
                    function = call.get("function") or {}
                    name = function.get("name", "")
                    result = execute_tool(workspace, name, function.get("arguments", "{}"))
                    parsed = json.loads(result)
                    validated |= name == "validate_python" and bool(parsed.get("valid"))
                    executed.append({"turn": turn, "name": name, "result": parsed})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"call_{turn}_{len(executed)}"),
                            "content": result,
                        }
                    )
                names_so_far = [item["name"] for item in executed]
                if validated and "write_file" in names_so_far and "read_file" in names_so_far:
                    messages.append({"role": "user", "content": "Validation succeeded. Finish with one concise sentence; no more tools."})
                    doc = chat(server.port, model, messages, include_tools=False)
                    message = doc["choices"][0]["message"]
                    final_text = message.get("content") or ""
                    turns.append({"turn": turn + 1, "tool_calls": 0, **metrics(doc)})
                    break
            names = [item["name"] for item in executed]
            target = workspace / "src/sort_tools.py"
            write_ok = any(item["name"] == "write_file" and item["result"].get("ok") for item in executed)
            read_ok = any(item["name"] == "read_file" and item["result"].get("ok") for item in executed)
            result = {
                "ok": bool(write_ok and read_ok),
                "task_file_exists": target.exists(),
                "agent_finished": bool(final_text),
                "turns": turns,
                "executed_tools": executed,
                "files": sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file()),
                "final_text": final_text[:300],
                "wall_s": round(time.monotonic() - start, 2),
                "ngram": status_ngram(server.port),
            }
            if not result["ok"]:
                raise RuntimeError("required write/read tool sequence did not execute: " + json.dumps(executed)[:1600])
            return result
    finally:
        server.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build/richengine"))
    parser.add_argument("model", nargs="?", default=MODEL)
    parser.add_argument("--max-memory", default="30G", help="server memory cap; use 0 for none")
    args = parser.parse_args()
    memory = None if args.max_memory == "0" else args.max_memory
    print(json.dumps(run(Path(args.binary).resolve(), args.model, memory), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Closed-loop coding-agent task suite.

Runs independent tool-using agents against isolated workspace fixtures. Each
task contains a buggy module and real unittest coverage; the model must inspect,
rewrite, test, and finish. Unlike transcript replay, tool results feed back into
subsequent generations and task success is checked independently by the harness.

The runner executes the task list with bounded concurrency and records wall,
prefill, decode, tool, acceptance, GPU, memory-pressure, and n-gram telemetry.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import agentic_tool_execution as toolbench  # noqa: E402
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"


def task(task_id: str, target: str, files: dict[str, str], prompt: str) -> dict:
    return {"id": task_id, "target": target, "files": files, "prompt": prompt}


def unittest_source(module: str, cases: str) -> str:
    return f"""import unittest

from src.{module} import *


class GeneratedBehaviorTests(unittest.TestCase):
{cases}


if __name__ == "__main__":
    unittest.main()
"""


def fixture_tasks() -> list[dict]:
    tasks = []

    bubble_impl = '''def bubble_sort(values):
    result = values
    for end in range(len(result) - 1, 0, -1):
        for index in range(end):
            if result[index] < result[index + 1]:
                result[index], result[index + 1] = result[index + 1], result[index]
    return result


def quick_check(values):
    return bubble_sort(list(values)) == sorted(values)
'''
    bubble_test = unittest_source("sort_tools", '''    def test_bubble_sort_orders_values(self):
        self.assertEqual(bubble_sort([4, 1, 3, 2]), [1, 2, 3, 4])

    def test_bubble_sort_keeps_input(self):
        values = [3, 1, 2]
        bubble_sort(values)
        self.assertEqual(values, [3, 1, 2])

    def test_quick_check(self):
        self.assertTrue(quick_check([5, 2, 4]))''')
    tasks.append(task(
        "sort-tools", "src/sort_tools.py",
        {"src/sort_tools.py": bubble_impl, "tests/test_sort_tools.py": bubble_test},
        "Inspect the workspace, fix src/sort_tools.py so its tests pass, run the Python tests, "
        "and finish with one concise sentence. Use relative paths.",
    ))

    median_impl = '''def clamp(value, low, high):
    return min(max(value, high), low)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return ordered[middle]
'''
    median_test = unittest_source("number_tools", '''    def test_clamp(self):
        self.assertEqual(clamp(9, 0, 5), 5)
        self.assertEqual(clamp(-2, 0, 5), 0)

    def test_odd_median(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_even_median(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)''')
    tasks.append(task(
        "number-tools", "src/number_tools.py",
        {"src/number_tools.py": median_impl, "tests/test_number_tools.py": median_test},
        "Inspect the workspace, repair src/number_tools.py until the real unit tests pass, "
        "run them with the tool, then finish concisely. Use relative paths.",
    ))

    words_impl = '''def reverse_words(text):
    return " ".join(reversed(text.split(" ")))


def count_vowels(text):
    return sum(text.count(vowel) for vowel in "aeiou")
'''
    words_test = unittest_source("text_tools", '''    def test_reverse_words(self):
        self.assertEqual(reverse_words("one two three"), "three two one")

    def test_reverse_words_collapses_space(self):
        self.assertEqual(reverse_words("one  two\\nthree"), "three two one")

    def test_count_vowels_case_insensitive(self):
        self.assertEqual(count_vowels("Alpha Echo"), 5)''')
    tasks.append(task(
        "text-tools", "src/text_tools.py",
        {"src/text_tools.py": words_impl, "tests/test_text_tools.py": words_test},
        "Inspect the workspace, fix src/text_tools.py, run the Python tests, repair any failure, "
        "and finish with one concise sentence. Use relative paths.",
    ))

    csv_impl = '''def parse_row(row):
    return row.split(",")


def escape_cell(value):
    text = str(value)
    if "," in text or '"' in text or "\\n" in text:
        return '"' + text + '"'
    return text
'''
    csv_test = unittest_source("csv_tools", '''    def test_parse_plain_row(self):
        self.assertEqual(parse_row("a,b,c"), ["a", "b", "c"])

    def test_parse_quoted_commas(self):
        self.assertEqual(parse_row('a,"b,c",d'), ["a", "b,c", "d"])

    def test_escape_quotes(self):
        self.assertEqual(escape_cell('say "hi", now'), '"say ""hi"", now"')''')
    tasks.append(task(
        "csv-tools", "src/csv_tools.py",
        {"src/csv_tools.py": csv_impl, "tests/test_csv_tools.py": csv_test},
        "Inspect the workspace, correct src/csv_tools.py, execute its unit tests, fix failures, "
        "and finish concisely. Use relative paths.",
    ))

    json_impl = '''def deep_get(mapping, path, default=None):
    current = mapping
    for part in path.split("."):
        current = current[part]
    return current


def flatten_keys(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten_keys(value, name + "."))
        else:
            result[name] = value
    return result
'''
    json_test = unittest_source("json_tools", '''    def test_deep_get_nested(self):
        self.assertEqual(deep_get({"a": {"b": 3}}, "a.b"), 3)

    def test_deep_get_default(self):
        self.assertEqual(deep_get({"a": {}}, "a.b", 9), 9)

    def test_flatten_keys(self):
        self.assertEqual(flatten_keys({"a": {"b": 1}, "c": 2}), {"a.b": 1, "c": 2})''')
    tasks.append(task(
        "json-tools", "src/json_tools.py",
        {"src/json_tools.py": json_impl, "tests/test_json_tools.py": json_test},
        "Inspect the workspace, fix src/json_tools.py until tests pass, run the tests with the tool, "
        "and finish with one concise sentence. Use relative paths.",
    ))

    chunk_impl = '''def chunks(values, size):
    return [values[index:index + size] for index in range(0, len(values) - size, size)]


def flatten(groups):
    result = []
    for group in groups:
        result += group
    return result
'''
    chunk_test = unittest_source("list_tools", '''    def test_chunks_include_tail(self):
        self.assertEqual(chunks([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_chunks_exact(self):
        self.assertEqual(chunks([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_flatten(self):
        self.assertEqual(flatten([[1], [2, 3]]), [1, 2, 3])''')
    tasks.append(task(
        "list-tools", "src/list_tools.py",
        {"src/list_tools.py": chunk_impl, "tests/test_list_tools.py": chunk_test},
        "Inspect the workspace, repair src/list_tools.py, run its Python tests, iterate if needed, "
        "and finish concisely. Use relative paths.",
    ))

    token_impl = '''def tokenize(text):
    return text.lower().split()


def token_counts(tokens):
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0)
    return counts
'''
    token_test = unittest_source("token_tools", '''    def test_tokenize_punctuation(self):
        self.assertEqual(tokenize("Hi, hi!"), ["hi", "hi"])

    def test_token_counts(self):
        self.assertEqual(token_counts(["a", "b", "a"]), {"a": 2, "b": 1})''')
    tasks.append(task(
        "token-tools", "src/token_tools.py",
        {"src/token_tools.py": token_impl, "tests/test_token_tools.py": token_test},
        "Inspect the workspace, fix src/token_tools.py, run the real Python tests, repair failures, "
        "and finish with one concise sentence. Use relative paths.",
    ))

    table_impl = '''def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\\n".join(lines)


def column(rows, index):
    return rows[index]
'''
    table_test = unittest_source("table_tools", '''    def test_markdown_table(self):
        self.assertEqual(markdown_table(["a"], [[1]]), "| a |\\n| --- |\\n| 1 |")

    def test_column_selects_column(self):
        self.assertEqual(column([[1, 2], [3, 4]], 1), [2, 4])''')
    tasks.append(task(
        "table-tools", "src/table_tools.py",
        {"src/table_tools.py": table_impl, "tests/test_table_tools.py": table_test},
        "Inspect the workspace, correct src/table_tools.py, run its unit tests, fix any failure, "
        "and finish concisely. Use relative paths.",
    ))


    semver_impl = '''def parse_version(text):
    return tuple(int(part) for part in text.split("."))


def compare_versions(left, right):
    if left == right:
        return 0
    return -1 if left < right else 1


def is_compatible(version, minimum):
    return compare_versions(version, minimum) >= 0
'''
    semver_test = unittest_source("semver_tools", '''    def test_parse_padded_version(self):
        self.assertEqual(parse_version("1.2"), (1, 2, 0))

    def test_compare_numeric_lengths(self):
        self.assertEqual(compare_versions((1, 10, 0), (1, 9, 9)), 1)
        self.assertEqual(compare_versions((1, 2, 0), (1, 2)), 0)

    def test_compatible(self):
        self.assertTrue(is_compatible((2, 0, 0), (1, 9, 0)))''')
    tasks.append(task(
        "semver-tools", "src/semver_tools.py",
        {"src/semver_tools.py": semver_impl, "tests/test_semver_tools.py": semver_test},
        "Inspect the workspace, fix src/semver_tools.py so version ordering and defaults pass tests, "
        "run the Python tests, and finish concisely. Use relative paths.",
    ))

    config_impl = '''def parse_config(text):
    result = {}
    for line in text.splitlines():
        if not line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def typed_value(value):
    if value == "true":
        return True
    if value == "false":
        return False
    return value
'''
    config_test = unittest_source("config_tools", '''    def test_parse_ignores_comments_and_blanks(self):
        self.assertEqual(parse_config("# hi\\n\\nport=8000\\n"), {"port": "8000"})

    def test_typed_numbers_and_bools(self):
        self.assertEqual(typed_value("42"), 42)
        self.assertEqual(typed_value("3.5"), 3.5)
        self.assertIs(typed_value("true"), True)

    def test_typed_plain_string(self):
        self.assertEqual(typed_value("alpha"), "alpha")''')
    tasks.append(task(
        "config-tools", "src/config_tools.py",
        {"src/config_tools.py": config_impl, "tests/test_config_tools.py": config_test},
        "Inspect the workspace, repair src/config_tools.py, run the real Python tests, iterate on failures, "
        "and finish concisely. Use relative paths.",
    ))

    inventory_impl = '''def total_quantity(items):
    return sum(item["price"] for item in items)


def quantities_by_sku(items):
    result = {}
    for item in items:
        result[item["sku"]] = item["quantity"]
    return result


def restock_needed(items, minimum):
    return [item["sku"] for item in items if item["quantity"] <= minimum]
'''
    inventory_test = unittest_source("inventory_tools", '''    def test_total_quantity(self):
        items = [{"quantity": 2, "price": 5}, {"quantity": 3, "price": 7}]
        self.assertEqual(total_quantity(items), 5)

    def test_quantities_accumulate(self):
        items = [{"sku": "a", "quantity": 2}, {"sku": "a", "quantity": 3}]
        self.assertEqual(quantities_by_sku(items), {"a": 5})

    def test_restock_is_strictly_below(self):
        items = [{"sku": "a", "quantity": 2}, {"sku": "b", "quantity": 1}]
        self.assertEqual(restock_needed(items, 2), ["b"])''')
    tasks.append(task(
        "inventory-tools", "src/inventory_tools.py",
        {"src/inventory_tools.py": inventory_impl, "tests/test_inventory_tools.py": inventory_test},
        "Inspect the workspace, fix src/inventory_tools.py until its tests pass, run them, "
        "and finish with one concise sentence. Use relative paths.",
    ))

    slug_impl = '''def slugify(text):
    return text.lower().replace(" ", "-")


def unique_slug(slug, existing):
    if slug not in existing:
        return slug
    return slug + "-2"
'''
    slug_test = unittest_source("slug_tools", '''    def test_slugify_punctuation_and_space(self):
        self.assertEqual(slugify("Hello,  World!"), "hello-world")

    def test_slugify_strips_dashes(self):
        self.assertEqual(slugify("--Hello--"), "hello")

    def test_unique_slug_increments(self):
        self.assertEqual(unique_slug("post", {"post", "post-2"}), "post-3")''')
    tasks.append(task(
        "slug-tools", "src/slug_tools.py",
        {"src/slug_tools.py": slug_impl, "tests/test_slug_tools.py": slug_test},
        "Inspect the workspace, correct src/slug_tools.py, execute its tests, repair failures, "
        "and finish concisely. Use relative paths.",
    ))

    xml_impl = '''def escape_xml(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unescape_xml(text):
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def attribute_escape(text):
    return escape_xml(text)
'''
    xml_test = unittest_source("xml_tools", '''    def test_escape_xml(self):
        self.assertEqual(escape_xml("a<&>"), "a&lt;&amp;&gt;")

    def test_unescape_xml(self):
        self.assertEqual(unescape_xml("a&lt;&amp;&gt;"), "a<&>")

    def test_attribute_quotes(self):
        self.assertEqual(attribute_escape('say "hi"'), "say &quot;hi&quot;")''')
    tasks.append(task(
        "xml-tools", "src/xml_tools.py",
        {"src/xml_tools.py": xml_impl, "tests/test_xml_tools.py": xml_test},
        "Inspect the workspace, fix src/xml_tools.py, run the unit tests, repair failures, "
        "and finish with one concise sentence. Use relative paths.",
    ))

    window_impl = '''def moving_average(values, width):
    return [sum(values[index:index + width]) / width for index in range(len(values) - width)]


def maximum_window(values, width):
    return [max(values[index:index + width]) for index in range(len(values) - width)]
'''
    window_test = unittest_source("window_tools", '''    def test_moving_average(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])

    def test_maximum_window(self):
        self.assertEqual(maximum_window([1, 5, 2, 4], 2), [5, 5, 4])

    def test_full_width(self):
        self.assertEqual(moving_average([2, 4], 2), [3.0])''')
    tasks.append(task(
        "window-tools", "src/window_tools.py",
        {"src/window_tools.py": window_impl, "tests/test_window_tools.py": window_test},
        "Inspect the workspace, correct src/window_tools.py, run its tests, iterate if necessary, "
        "and finish concisely. Use relative paths.",
    ))

    score_impl = '''def password_score(password):
    score = len(password) >= 8
    score += any(char.isdigit() for char in password)
    return score


def strength_label(score):
    if score >= 4:
        return "strong"
    if score >= 2:
        return "medium"
    return "weak"
'''
    score_test = unittest_source("password_tools", '''    def test_score_features(self):
        self.assertEqual(password_score("Abcdef1!"), 4)

    def test_score_short_password(self):
        self.assertEqual(password_score("abc"), 1)

    def test_strength_label(self):
        self.assertEqual(strength_label(4), "strong")
        self.assertEqual(strength_label(2), "medium")''')
    tasks.append(task(
        "password-tools", "src/password_tools.py",
        {"src/password_tools.py": score_impl, "tests/test_password_tools.py": score_test},
        "Inspect the workspace, repair src/password_tools.py, run the real unit tests, "
        "and finish with one concise sentence. Use relative paths.",
    ))

    roman_impl = '''def roman_value(numeral):
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    return sum(values[char] for char in numeral)


def is_valid_roman(numeral):
    return bool(numeral)
'''
    roman_test = unittest_source("roman_tools", '''    def test_additive_roman(self):
        self.assertEqual(roman_value("VIII"), 8)

    def test_subtractive_roman(self):
        self.assertEqual(roman_value("IX"), 9)
        self.assertEqual(roman_value("XL"), 40)

    def test_invalid_characters(self):
        self.assertFalse(is_valid_roman("ABC"))''')
    tasks.append(task(
        "roman-tools", "src/roman_tools.py",
        {"src/roman_tools.py": roman_impl, "tests/test_roman_tools.py": roman_test},
        "Inspect the workspace, fix src/roman_tools.py, execute its Python tests, iterate on failures, "
        "and finish concisely. Use relative paths.",
    ))

    return tasks


def prepare_workspace(root: Path, spec: dict) -> None:
    for package in ("src", "tests"):
        path = root / package / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for relative, content in spec["files"].items():
        path = toolbench.safe_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def response_metrics(doc: dict) -> dict:
    timings = doc.get("timings") or {}
    usage = doc.get("usage") or {}
    decode = (doc.get("metrics") or {}).get("decode") or {}
    return {
        "prompt_ms": timings.get("prompt_ms") or 0.0,
        "decode_ms": timings.get("predicted_ms") or 0.0,
        "prompt_tokens": timings.get("prompt_n") or usage.get("prompt_tokens") or 0,
        "completion_tokens": timings.get("predicted_n") or usage.get("completion_tokens") or 0,
        "accepted": decode.get("accepted_draft_tokens", 0),
        "offered": decode.get("drafted_tokens", 0),
    }


def get_status(port: int) -> dict:
    status, doc = smoke_real.request(port, "GET", "/status", timeout=10)
    return doc if status == 200 else {}


def run_tests_direct(workspace: Path) -> dict:
    result = toolbench.execute_tool(workspace, "run_python_tests", json.dumps({"path": "tests"}))
    return json.loads(result)


def run_task(port: int, model: str, spec: dict, max_turns: int, max_tokens: int, started: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="richengine-task-") as tmp:
        workspace = Path(tmp)
        prepare_workspace(workspace, spec)
        messages = [
            {
                "role": "system",
                "content": "You are a coding agent operating through filesystem and test tools. "
                "Inspect before editing. Use the run_python_tests tool for validation. "
                "Do not claim success before tests pass. Keep final answers concise.",
            },
            {
                "role": "user",
                "content": f"The isolated workspace root is /workspace. {spec['prompt']}",
            },
        ]
        task_started = time.monotonic()
        turns = []
        executed = []
        final_text = ""
        failure = ""
        tool_ms = 0.0
        tests_passed = False
        completion_prompted = False
        for turn_index in range(max_turns):
            try:
                doc = toolbench.chat(port, model, messages, tokens=max_tokens)
            except Exception as error:
                failure = str(error)
                break
            message = doc["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            metrics = response_metrics(doc)
            turns.append({"turn": turn_index, "tool_calls": len(calls), **metrics})
            messages.append({
                "role": "assistant",
                "content": content,
                **({"tool_calls": calls} if calls else {}),
            })
            if not calls:
                final_text = content
                break
            for call in calls:
                function = call.get("function") or {}
                tool_started = time.monotonic()
                result = toolbench.execute_tool(
                    workspace, function.get("name", ""), function.get("arguments", "{}")
                )
                tool_ms += (time.monotonic() - tool_started) * 1000.0
                parsed = json.loads(result)
                executed.append({
                    "turn": turn_index,
                    "name": function.get("name", ""),
                    "ok": parsed.get("ok", False),
                    "passed": parsed.get("passed"),
                })
                if function.get("name") == "run_python_tests" and parsed.get("passed"):
                    tests_passed = True
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call_{turn_index}_{len(executed)}"),
                    "content": result,
                })
            wrote_any = any(item["name"] == "write_file" and item.get("ok") for item in executed)
            if tests_passed and wrote_any and not completion_prompted:
                completion_prompted = True
                messages.append({
                    "role": "user",
                    "content": "The tests now pass. Finish with one concise sentence; do not use tools.",
                })
                try:
                    doc = toolbench.chat(port, model, messages, tokens=96, include_tools=False)
                    message = doc["choices"][0]["message"]
                    final_text = message.get("content") or ""
                    turns.append({"turn": turn_index + 1, "tool_calls": 0, **response_metrics(doc)})
                except Exception as error:
                    failure = str(error)
                break
        else:
            failure = "turn limit reached"
        test_result = run_tests_direct(workspace)
        target_exists = toolbench.safe_path(workspace, spec["target"]).exists()
        wrote_target = any(
            item["name"] == "write_file" and item.get("ok") for item in executed
        )
        ended = time.monotonic()
        prompt_ms = sum(turn["prompt_ms"] for turn in turns)
        decode_ms = sum(turn["decode_ms"] for turn in turns)
        completion = sum(turn["completion_tokens"] for turn in turns)
        prompt_tokens = sum(turn["prompt_tokens"] for turn in turns)
        accepted = sum(turn["accepted"] for turn in turns)
        offered = sum(turn["offered"] for turn in turns)
        success = bool(final_text and test_result.get("passed") and wrote_target and target_exists)
        return {
            "task_id": spec["id"],
            "success": success,
            "failure": failure,
            "start_s": round(task_started - started, 3),
            "end_s": round(ended - started, 3),
            "wall_s": round(ended - task_started, 3),
            "prompt_ms": round(prompt_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "tool_ms": round(tool_ms, 2),
            "agent_ms": round((ended - task_started) * 1000.0 - tool_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "accepted": accepted,
            "offered": offered,
            "turns": turns,
            "tool_counts": {name: sum(item["name"] == name for item in executed) for name in sorted({item["name"] for item in executed})},
            "test_passed": bool(test_result.get("passed")),
            "test_output_tail": (test_result.get("output") or "")[-500:],
            "final_text": final_text[:240],
        }


def nested_counter(status: dict, section: str, key: str) -> float:
    return ((status.get(section) or {}).get(key) or 0) if status else 0


def counter_delta(before: dict, after: dict, section: str, key: str) -> float:
    return nested_counter(after, section, key) - nested_counter(before, section, key)


def summarize(results: list[dict], suite_wall: float, before: dict, after: dict) -> dict:
    completion = sum(item["completion_tokens"] for item in results)
    successful = [item for item in results if item["success"]]
    engine_failures = sum(
        "engine is recovering" in item.get("failure", "")
        or "inference engine stopped" in item.get("failure", "")
        for item in results
    )
    decode_after = (((after.get("model_timing") or {}).get("decode") or {}).get("total_gpu_ms") or 0)
    decode_before = (((before.get("model_timing") or {}).get("decode") or {}).get("total_gpu_ms") or 0)
    wall_after = (((after.get("model_timing") or {}).get("decode") or {}).get("total_wall_ms") or 0)
    wall_before = (((before.get("model_timing") or {}).get("decode") or {}).get("total_wall_ms") or 0)
    return {
        "tasks": len(results),
        "successful": len(successful),
        "engine_failures": engine_failures,
        "valid": engine_failures == 0,
        "invalid_reason": "engine restart" if engine_failures else "",
        "success_rate": round(len(successful) / len(results), 4) if results else 0,
        "suite_wall_s": round(suite_wall, 3),
        "completion_tokens": completion,
        "wall_tok_s": round(completion / suite_wall, 2) if suite_wall else 0,
        "task_wall_mean_s": round(statistics.mean(item["wall_s"] for item in results), 3) if results else 0,
        "task_wall_median_s": round(statistics.median(item["wall_s"] for item in results), 3) if results else 0,
        "decode_ms_total": round(sum(item["decode_ms"] for item in results), 2),
        "prompt_ms_total": round(sum(item["prompt_ms"] for item in results), 2),
        "tool_ms_total": round(sum(item["tool_ms"] for item in results), 2),
        "accepted_draft_tokens": sum(item["accepted"] for item in results),
        "offered_draft_tokens": sum(item["offered"] for item in results),
        "decode_gpu_ms_delta": round(decode_after - decode_before, 3),
        "decode_wall_ms_delta": round(wall_after - wall_before, 3),
        "decode_gpu_ms_per_token": round((decode_after - decode_before) / completion, 4) if completion else 0,
        "decode_wall_ms_per_token": round((wall_after - wall_before) / completion, 4) if completion else 0,
        "ngram_mixed_lanes_delta": round(counter_delta(before, after, "ngram", "mixed_lanes"), 3),
        "ngram_corpus_lanes_delta": round(counter_delta(before, after, "ngram", "mixed_corpus_lanes"), 3),
        "ngram_chain_tokens_delta": round(counter_delta(before, after, "ngram", "chain_tokens"), 3),
        "ngram_host_us_delta": round(counter_delta(before, after, "ngram", "host_microseconds"), 3),
        "verify_rows_delta": round(counter_delta(before, after, "ngram", "verify_rows_total"), 3),
        "verify_rows_per_token": round(counter_delta(before, after, "ngram", "verify_rows_total") / completion, 4) if completion else 0,
        "corpus_hits_delta": round(counter_delta(before, after, "corpus", "hits"), 3),
    }


class Args:
    def __init__(self, binary: Path, model: str, ctx: int, max_memory: str | None):
        self.package = ROOT / "install/models" / model
        self.binary = binary
        self.model = model
        self.max_context = ctx
        self.max_memory = max_memory
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def run_suite(binary: Path, model: str, args) -> dict:
    overrides = dict(item.split("=", 1) for item in args.env)
    memory = None if args.max_memory == "0" else args.max_memory
    server = smoke_real.RealServer(
        Args(binary, model, args.ctx, memory), {**os.environ, **overrides}
    )
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        tasks = fixture_tasks()[: args.tasks]
        before = get_status(server.port)
        started = time.monotonic()
        results: list[dict] = []
        lock = threading.Lock()
        def one(spec):
            result = run_task(server.port, model, spec, args.max_turns, args.max_tokens, started)
            with lock:
                results.append(result)
                print(f"{spec['id']}: {'ok' if result['success'] else 'fail'} {result['wall_s']}s", flush=True)
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.width) as pool:
            list(pool.map(one, tasks))
        suite_wall = time.monotonic() - started
        after = get_status(server.port)
        results.sort(key=lambda item: item["task_id"])
        return {
            "model": model,
            "width": args.width,
            "task_count": len(tasks),
            "env": overrides,
            "summary": summarize(results, suite_wall, before, after),
            "tasks": results,
            "status_before": {"ngram": before.get("ngram"), "decode": (before.get("model_timing") or {}).get("decode"), "corpus": before.get("corpus")},
            "status_after": {"ngram": after.get("ngram"), "decode": (after.get("model_timing") or {}).get("decode"), "corpus": after.get("corpus")},
            "server_log_tail": server.tail(),
        }
    finally:
        server.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build/richengine"))
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--max-memory", default="30G")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = run_suite(Path(args.binary).resolve(), args.model, args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))
    print(json.dumps({"summary": payload["summary"], "ngram": payload["status_after"]["ngram"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

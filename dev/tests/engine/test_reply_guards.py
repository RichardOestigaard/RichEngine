"""Reply-level loop and repeated-call guards.

The pure functions cover the detection itself; the collect tests drive
FrontendHandler._collect with canned job events, as test_server.py does,
to cover the steering loop: a reply repeating the previous turn's call is
decoded again with the steering prefix, up to the attempt cap, and ends
with a message when it still repeats.
"""

import time
import unittest
from itertools import count
from types import SimpleNamespace
from unittest import mock

from server import protocol as wire
from server import reply_guards, tool_schema
from server import server as api
from server.backend import Job, NativeResult


def result(reason="stop", completion_tokens=10):
    return NativeResult(reason, 5, completion_tokens, 1, 1, 1)


def call(name, *arguments):
    """A call's text in the default dialect, as the model writes it."""
    return (
        f"<tool_call>\n<function={name}>\n"
        + "".join(
            f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments
        )
        + "</function>\n</tool_call>"
    )


def policy(*names):
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]
    return tool_schema.normalize_tools(tools, "auto", True)[1]


def job_for(events, **fields):
    job = Job(
        request_id=next(ids),
        prompt_tokens=[1, 2],
        max_new_tokens=64,
        seed=0,
        sampling=wire.SamplingParameters(),
        deadline=time.monotonic() + 60,
        public_id=f"job-{next(ids)}",
        **fields,
    )
    for event in events:
        job.events.put(event)
    return job


ids = count(1)


def handler_for(app):
    handler = object.__new__(api.FrontendHandler)
    handler._client_disconnected = lambda: False
    handler.server = SimpleNamespace(app=app)
    return handler


def app_for(submitted=()):
    """A stand-in Frontend with the pieces steering reaches for: an ids
    counter, a backend that queues each retry's canned events, a tokenizer
    and a constraint compiler."""
    backend = SimpleNamespace(active={}, cancel=mock.Mock())
    submissions = iter(submitted)
    backend.submit = mock.Mock(
        side_effect=lambda job: [job.events.put(event) for event in next(submissions)]
    )
    return SimpleNamespace(
        backend=backend,
        ids=count(1000),
        _tokenize=lambda text, **_: {"input_ids": [9, 9]},
        _compile_constraint=mock.Mock(return_value=None),
    )


class LoopingTests(unittest.TestCase):
    def test_constant_stream_is_a_loop(self):
        self.assertTrue(reply_guards.looping([7] * 200))
        self.assertTrue(reply_guards.looping([7] * 128))

    def test_block_repeats_detect_by_smallest_period(self):
        tokens = [1, 2, 3] * 50
        self.assertTrue(reply_guards.looping(tokens))
        # A different token at the end breaks the suffix's periodicity.
        self.assertFalse(reply_guards.looping(tokens + [9]))

    def test_short_blocks_must_span(self):
        # A block under 8 tokens must repeat over at least short_span:
        # data repeats a few tokens on purpose.
        self.assertFalse(reply_guards.looping([1, 2] * 60))
        self.assertTrue(reply_guards.looping([1, 2] * 64))
        self.assertTrue(reply_guards.looping([1, 2] * 200))

    def test_long_blocks_repeat_three_times(self):
        self.assertTrue(reply_guards.looping(list(range(10)) * 3))
        self.assertFalse(reply_guards.looping(list(range(10)) * 2))

    def test_too_short_output_never_loops(self):
        self.assertFalse(reply_guards.looping([7] * 23))

    def test_in_call_thresholds(self):
        # Inside a call only 16 repeats over 256 tokens count: a file of
        # identical lines repeats on purpose.
        tokens = [7] * 300
        self.assertTrue(
            reply_guards.looping(tokens, reps=16, min_span=256, short_span=256)
        )
        self.assertFalse(
            reply_guards.looping([7] * 200, reps=16, min_span=256, short_span=256)
        )
        # A 40-token block needs only three repeats by default but eight
        # inside a call's data.
        block = list(range(40))
        self.assertTrue(reply_guards.looping(block * 8))
        self.assertFalse(
            reply_guards.looping(block * 4, reps=16, min_span=256, short_span=256)
        )

    def test_free_text_thresholds(self):
        # Without tools a repetition that may be wanted is only scanned
        # for, at the looser thresholds.
        self.assertTrue(
            reply_guards.looping([7] * 300, reps=12, min_span=96, short_span=256)
        )


class LineLoopingTests(unittest.TestCase):
    def test_rephrased_same_intent(self):
        lines = [
            "I will use the read tool now",
            "Let me read the file next",
            "I'll proceed with the read call",
            "Reading the file as planned",
        ]
        text = "\n".join(lines[index % len(lines)] for index in range(31))
        self.assertTrue(reply_guards.line_looping(text))

    def test_distinct_lines_do_not_loop(self):
        text = "\n".join(f"line number {index} of output" for index in range(31))
        self.assertFalse(reply_guards.line_looping(text))

    def test_fewer_lines_than_the_window(self):
        text = "\n".join("same line repeated" for _ in range(29))
        self.assertFalse(reply_guards.line_looping(text))

    def test_short_lines_do_not_count(self):
        text = "\n".join("short" for _ in range(60))
        self.assertFalse(reply_guards.line_looping(text))


class PastGraceTests(unittest.TestCase):
    def test_without_state_passes_flag_through(self):
        self.assertTrue(reply_guards.past_grace(True, 10, None))
        self.assertFalse(reply_guards.past_grace(False, 10, None))

    def test_grace_window_counts_from_first_detection(self):
        state = SimpleNamespace(loop_at=None)
        self.assertFalse(reply_guards.past_grace(True, 10, state))
        self.assertFalse(
            reply_guards.past_grace(True, 10 + reply_guards.LOOP_GRACE - 1, state)
        )
        self.assertTrue(
            reply_guards.past_grace(True, 10 + reply_guards.LOOP_GRACE, state)
        )

    def test_stopping_the_repetition_resets(self):
        state = SimpleNamespace(loop_at=None)
        reply_guards.past_grace(True, 10, state)
        self.assertFalse(reply_guards.past_grace(False, 20, state))
        self.assertIsNone(state.loop_at)


class CallKeyTests(unittest.TestCase):
    def test_arguments_compare_by_value_not_text(self):
        first = {"function": {"name": "f", "arguments": '{"a": 1, "b": 2}'}}
        second = {"function": {"name": "f", "arguments": {"b": 2, "a": 1}}}
        self.assertEqual(reply_guards.call_key(first), reply_guards.call_key(second))

    def test_unparseable_arguments_compare_as_text(self):
        call = {"function": {"name": "f", "arguments": '{"a":'}}
        self.assertEqual(reply_guards.call_key(call), ("f", '{"a":'))

    def test_different_calls_differ(self):
        first = {"function": {"name": "f", "arguments": "{}"}}
        second = {"function": {"name": "g", "arguments": "{}"}}
        self.assertNotEqual(reply_guards.call_key(first), reply_guards.call_key(second))


class PreviousCallsTests(unittest.TestCase):
    def test_last_assistant_turns_calls_with_results(self):
        messages = [
            {"role": "user", "content": "roll it"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "roll_cut",
                            "arguments": {"item": "s1"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "rolled s1"},
        ]
        prev = reply_guards.previous_calls(messages)
        self.assertEqual(len(prev), 1)
        (name, arguments), result = next(iter(prev.items()))
        self.assertEqual(name, "roll_cut")
        self.assertEqual(result, "rolled s1")
        repeat = {
            "function": {
                "name": "roll_cut",
                "arguments": '{"item": "s1"}',
            }
        }
        self.assertIn(reply_guards.call_key(repeat), prev)

    def test_only_the_last_assistant_turn_compares(self):
        earlier = {
            "id": "c0",
            "function": {"name": "old_tool", "arguments": {}},
        }
        later = {
            "id": "c1",
            "function": {"name": "roll_cut", "arguments": {"item": "s1"}},
        }
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "tool_calls": [earlier]},
            {"role": "tool", "tool_call_id": "c0", "content": "r0"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "tool_calls": [later]},
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        ]
        prev = reply_guards.previous_calls(messages)
        self.assertEqual(len(prev), 1)
        self.assertNotIn(("old_tool", "{}"), prev)

    def test_a_user_message_since_means_nothing_to_repeat(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": {}}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "r"},
            {"role": "user", "content": "and now?"},
        ]
        self.assertEqual(reply_guards.previous_calls(messages), {})

    def test_calls_without_result_get_empty_text(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": {}}}
                ],
            },
        ]
        self.assertEqual(list(reply_guards.previous_calls(messages).values()), [""])


class DetectTests(unittest.TestCase):
    def test_max_calls(self):
        calls = [
            {"id": str(i), "function": {"name": "f", "arguments": f'{{"i": {i}}}'}}
            for i in range(reply_guards.MAX_CALLS)
        ]
        self.assertEqual(reply_guards.detect([], "", closed_calls=calls), "max_calls")

    def test_duplicate_call(self):
        calls = [
            {"id": "a", "function": {"name": "f", "arguments": '{"x": 1}'}},
            {"id": "b", "function": {"name": "f", "arguments": '{"x": 1}'}},
        ]
        self.assertEqual(
            reply_guards.detect([], "", closed_calls=calls), "duplicate_call"
        )
        # The same tool twice with different arguments is not a duplicate.
        calls[1]["function"]["arguments"] = '{"x": 2}'
        self.assertIsNone(reply_guards.detect([], "", closed_calls=calls, tools=True))

    def test_loop_without_tools_uses_grace(self):
        state = SimpleNamespace(loop_at=None)
        tokens = [7] * 300
        self.assertIsNone(reply_guards.detect(tokens, "", state=state))
        self.assertEqual(
            reply_guards.detect(
                tokens + [7] * reply_guards.LOOP_GRACE, "", state=state
            ),
            "loop",
        )

    def test_loop_with_tools_cuts_at_once(self):
        self.assertEqual(
            reply_guards.detect([7] * 200, "", tools=True, state=None), "loop"
        )

    def test_loop_in_call_waits_for_grace(self):
        state = SimpleNamespace(loop_at=None)
        tokens = [7] * 300
        self.assertIsNone(reply_guards.detect(tokens, "", in_call=True, state=state))
        self.assertEqual(
            reply_guards.detect(
                tokens + [7] * reply_guards.LOOP_GRACE,
                "",
                in_call=True,
                state=state,
            ),
            "loop_in_call",
        )

    def test_line_loop_with_tools(self):
        lines = [
            "I will use the read tool now",
            "Let me read the file next",
        ]
        text = "\n".join(lines[index % 2] for index in range(31))
        self.assertEqual(reply_guards.detect([], text, tools=True), "loop")


class TextsTests(unittest.TestCase):
    def test_steer_text_names_the_call_and_result(self):
        prev = {("roll_cut", '{"item":"s1"}'): "rolled   the  cut"}
        calls = [
            {
                "function": {
                    "name": "roll_cut",
                    "arguments": '{"item": "s1"}',
                }
            }
        ]
        text = reply_guards.steer_text(calls, prev)
        self.assertIn("roll_cut", text)
        self.assertIn("rolled the cut", text)
        self.assertIn("do something different", text)

    def test_steer_text_empty_without_a_repeat(self):
        calls = [{"function": {"name": "other", "arguments": "{}"}}]
        self.assertEqual(reply_guards.steer_text(calls, {}), "")

    def test_stop_text_asks_the_user(self):
        calls = [
            {
                "function": {
                    "name": "roll_cut",
                    "arguments": '{"item": "s1"}',
                }
            }
        ]
        text = reply_guards.stop_text(calls)
        self.assertIn("I am stuck", text)
        self.assertIn("roll_cut", text)


class CollectGuardTests(unittest.TestCase):
    def collect(self, app, job, **callbacks):
        return handler_for(app)._collect(job, **callbacks)

    def test_unguarded_collect_unchanged(self):
        app = app_for()
        job = job_for(
            [("text", "answer"), ("done", result())],
            tool_policy=None,
            thinking=False,
            response_validator=None,
        )
        job.cohort = [job]
        collected = self.collect(app, job)
        self.assertEqual(collected.content, "answer")
        app.backend.cancel.assert_not_called()

    def test_first_call_repeating_previous_steers(self):
        prev = reply_guards.previous_calls(
            [
                {"role": "user", "content": "roll"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "roll_cut",
                                "arguments": {"item": "s1"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "rolled"},
            ]
        )
        repeated = call("roll_cut", ("item", "s1"))
        different = call("other", ("item", "x"))
        app = app_for(
            submitted=[
                [("text", different), ("done", result())],
            ]
        )
        job = job_for(
            [("text", repeated), ("done", result("cancelled"))],
            tool_policy=policy("roll_cut", "other"),
            thinking=False,
            response_validator=None,
        )
        job.prev_calls = prev
        seen = []
        collected = self.collect(
            app, job, on_text=lambda field, text: seen.append((field, text))
        )
        app.backend.cancel.assert_called_once_with(job)
        app.backend.submit.assert_called_once()
        retry = app.backend.submit.call_args[0][0]
        self.assertIsNot(retry, job)
        self.assertEqual(retry.prompt_tokens[:2], [1, 2])
        # The steering text went out as content, then the retry's call.
        self.assertIn("do something different", "".join(t for _, t in seen))
        self.assertEqual(len(collected.tool_calls), 1)
        self.assertEqual(collected.tool_calls[0]["function"]["name"], "other")

    def test_still_repeating_ends_with_the_stop_text(self):
        prev = reply_guards.previous_calls(
            [
                {"role": "user", "content": "roll"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "roll_cut",
                                "arguments": {"item": "s1"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "rolled"},
            ]
        )
        repeated = call("roll_cut", ("item", "s1"))
        # The sole tool is banned after the first steering, so the second
        # repeat ends the turn with a message.
        app = app_for(
            submitted=[
                [("text", repeated), ("done", result("cancelled"))],
            ]
        )
        job = job_for(
            [("text", repeated), ("done", result("cancelled"))],
            tool_policy=policy("roll_cut"),
            thinking=False,
            response_validator=None,
        )
        job.prev_calls = prev
        seen = []
        collected = self.collect(
            app, job, on_text=lambda field, text: seen.append((field, text))
        )
        app.backend.submit.assert_called_once()
        self.assertEqual(collected.tool_calls, [])
        self.assertIn("I am stuck", collected.content)
        self.assertEqual(collected.result.reason, "stop")

    def test_steering_stops_after_the_attempt_cap(self):
        prev = reply_guards.previous_calls(
            [
                {"role": "user", "content": "roll"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "roll_cut",
                                "arguments": {"item": "s1"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "rolled"},
            ]
        )
        repeated = call("roll_cut", ("item", "s1"))
        # Three steered decodes repeat too (a second tool keeps a ban from
        # emptying the grammar), then the turn ends with the stop text.
        app = app_for(
            submitted=[[("text", repeated), ("done", result("cancelled"))]]
            * reply_guards.STEER_ATTEMPTS
        )
        job = job_for(
            [("text", repeated), ("done", result("cancelled"))],
            tool_policy=policy("roll_cut", "other"),
            thinking=False,
            response_validator=None,
        )
        job.prev_calls = prev
        collected = self.collect(app, job)
        self.assertEqual(app.backend.submit.call_count, reply_guards.STEER_ATTEMPTS)
        self.assertEqual(collected.tool_calls, [])
        self.assertIn("I am stuck", collected.content)
        self.assertEqual(collected.result.reason, "stop")

    def test_max_calls_trims_the_reply(self):
        text = "".join(
            call("roll_cut", ("item", f"s{i}"))
            for i in range(reply_guards.MAX_CALLS + 2)
        )
        app = app_for()
        job = job_for(
            [("text", text), ("done", result("cancelled"))],
            tool_policy=policy("roll_cut"),
            thinking=False,
            response_validator=None,
        )
        job.prev_calls = {}
        collected = self.collect(app, job)
        app.backend.cancel.assert_called_once_with(job)
        self.assertEqual(len(collected.tool_calls), reply_guards.MAX_CALLS)
        self.assertEqual(collected.result.reason, "stop")

    def test_duplicate_call_is_dropped(self):
        text = call("roll_cut", ("item", "s1")) + call("roll_cut", ("item", "s1"))
        app = app_for()
        job = job_for(
            [("text", text), ("done", result("cancelled"))],
            tool_policy=policy("roll_cut"),
            thinking=False,
            response_validator=None,
        )
        job.prev_calls = {}
        collected = self.collect(app, job)
        app.backend.cancel.assert_called_once_with(job)
        self.assertEqual(len(collected.tool_calls), 1)
        self.assertEqual(collected.result.reason, "stop")

    def test_free_text_loop_reports_length(self):
        app = app_for()
        job = job_for(
            [
                ("text", "same"),
                ("text", "same again"),
                ("done", result("cancelled")),
            ],
            tool_policy=None,
            thinking=False,
            response_validator=None,
        )
        # Without tools a repetition that may be wanted goes on LOOP_GRACE
        # tokens past looking like a loop before the guard cuts it: the
        # first check starts the grace count, the second ends it.
        sizes = iter((300, 300 + reply_guards.LOOP_GRACE))

        class Streamer:
            @property
            def token_ids(self):
                return [7] * next(sizes, 0)

        app.backend.active[job.request_id] = SimpleNamespace(streamer=Streamer())
        collected = self.collect(app, job)
        app.backend.cancel.assert_called_once_with(job)
        self.assertEqual(collected.result.reason, "length")

    def test_loop_with_tools_reports_stop(self):
        app = app_for()
        job = job_for(
            [("text", "same"), ("done", result("cancelled"))],
            tool_policy=policy("roll_cut"),
            thinking=False,
            response_validator=None,
        )
        app.backend.active[job.request_id] = SimpleNamespace(
            streamer=SimpleNamespace(token_ids=[7] * 200)
        )
        collected = self.collect(app, job)
        self.assertEqual(collected.result.reason, "stop")

    def test_repeat_after_sent_calls_keeps_them(self):
        prev = reply_guards.previous_calls(
            [
                {"role": "user", "content": "roll"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "roll_cut",
                                "arguments": {"item": "s1"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "rolled"},
            ]
        )
        text = call("other", ("item", "x")) + call("roll_cut", ("item", "s1"))
        app = app_for()
        job = job_for(
            [("text", text), ("done", result("cancelled"))],
            tool_policy=policy("roll_cut", "other"),
            thinking=False,
            response_validator=None,
        )
        job.prev_calls = prev
        deltas = []
        collected = self.collect(
            app,
            job,
            on_text=lambda field, text: None,
            on_tool_delta=deltas.append,
        )
        app.backend.cancel.assert_called_once_with(job)
        app.backend.submit.assert_not_called()
        # Calls already sent cannot be steered away from: the reply keeps
        # them, up to the first repeat.
        self.assertEqual(len(collected.tool_calls), 1)
        self.assertEqual(collected.tool_calls[0]["function"]["name"], "other")
        self.assertEqual(collected.result.reason, "stop")
        # The repeated call's deltas were held and never went out.
        names = [
            delta["function"]["name"]
            for delta in deltas
            if "name" in delta.get("function", {})
        ]
        self.assertEqual(names, ["other"])

    def test_required_choice_keeps_the_repeat(self):
        prev = reply_guards.previous_calls(
            [
                {"role": "user", "content": "roll"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "roll_cut",
                                "arguments": {"item": "s1"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "rolled"},
            ]
        )
        _, required = tool_schema.normalize_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "roll_cut",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "required",
            True,
        )
        app = app_for()
        job = job_for(
            [("text", call("roll_cut", ("item", "s1"))), ("done", result())],
            tool_policy=required,
            thinking=False,
            response_validator=None,
        )
        job.prev_calls = prev
        collected = self.collect(app, job)
        app.backend.submit.assert_not_called()
        self.assertEqual(len(collected.tool_calls), 1)


if __name__ == "__main__":
    unittest.main()

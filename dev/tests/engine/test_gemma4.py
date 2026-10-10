"""Gemma 4's chat template, tool calls and thinking channel, server-side.

The fixture is the unmodified upstream chat template of
google/gemma-4-26B-A4B-it. Its tokenizer contract resolves the channel
and call tokens, generation_config.json adds its tool-response stop, the
gemma4 dialect reads `call:name{...}` calls between the call tokens, and
the reasoning splitter reads its thought channel.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transformers import PreTrainedTokenizerFast

from dev.tests import test_server as fixtures
from dev.tests.tool_output import project, streamed_arguments
from server import chat_templates, constraints, output, tool_schema
from server import frontend as request_frontend

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/chat_templates/gemma4.jinja"
TEMPLATE = FIXTURE.read_text()

Q = '<|"|>'
CHANNEL_OPEN = "<|channel>thought\n"
CHANNEL_CLOSE = "<channel|>"
CALL_OPEN = "<|tool_call>call:"
CALL_CLOSE = "}<tool_call|>"


def gcall(name, arguments=""):
    """A call as the gemma4 template writes it."""
    return CALL_OPEN + name + "{" + arguments + CALL_CLOSE


def gstr(text):
    return Q + text + Q


def tokenizer(template):
    result = PreTrainedTokenizerFast(
        tokenizer_object=fixtures._byte_backend({0: "hello"})
    )
    result.chat_template = template
    return result


def policy(schemas=None, **overrides):
    fields = {
        "schemas": schemas or {"lookup": {"type": "object"}},
        "required": False,
        "parallel": True,
        "strict": frozenset(),
        "constrained": False,
        "dialect": tool_schema.GEMMA4,
    }
    return tool_schema.ToolPolicy(**{**fields, **overrides})


class Gemma4ChatTemplateTests(unittest.TestCase):
    def test_template_is_native_and_its_dialect_detected(self):
        chosen = chat_templates.ChatTemplates(tokenizer(TEMPLATE)).select(None)
        self.assertEqual(chosen.later_system, chat_templates.NATIVE)
        self.assertIs(chosen.dialect, tool_schema.GEMMA4)

    def test_turns_and_tool_framing_render(self):
        upstream = tokenizer(TEMPLATE)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a key.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "Leading instructions"},
            {"role": "user", "content": "First question"},
        ]
        rendered = upstream.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )
        self.assertIn("<|turn>system\nLeading instructions", rendered)
        self.assertIn("<|tool>declaration:lookup{", rendered)
        self.assertIn("<|turn>user\nFirst question<turn|>", rendered)
        # Thinking disabled: the generation prompt carries an empty closed
        # thought channel.
        self.assertTrue(
            rendered.endswith("<|turn>model\n" + CHANNEL_OPEN + CHANNEL_CLOSE),
            rendered,
        )
        # enable_thinking opens the system turn with the think token and
        # leaves the model's turn for it to write the channel itself.
        rendered = upstream.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        self.assertIn("<|turn>system\n<|think|>\nLeading instructions", rendered)
        self.assertTrue(rendered.endswith("<|turn>model\n"), rendered)

    def test_calls_and_responses_render_in_model_turns(self):
        upstream = tokenizer(TEMPLATE)
        call = {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Look it up",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"key": "alpha", "n": 42},
                    },
                }
            ],
        }
        result = {"role": "tool", "tool_call_id": "call_1", "content": "beta"}
        messages = [
            {"role": "system", "content": "Leading instructions"},
            {"role": "user", "content": "First question"},
            call,
            result,
            # A later system message renders in place as a system turn.
            {"role": "system", "content": "Later"},
            {"role": "user", "content": "Next question"},
        ]
        rendered = upstream.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Thoughts are stripped from history on a later turn...
        self.assertNotIn("Look it up", rendered)
        preserved = upstream.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            preserve_thinking=True,
        )
        # ...and kept where preserve_thinking asks, gated to tool calls.
        self.assertIn(
            CHANNEL_OPEN
            + "Look it up\n"
            + CHANNEL_CLOSE
            + gcall("lookup", "key:" + gstr("alpha") + ",n:42"),
            preserved,
        )
        self.assertIn(
            "<|tool_response>response:lookup{value:"
            + gstr("beta")
            + "}<tool_response|>",
            rendered,
        )
        self.assertIn("<|turn>system\nLater<turn|>", rendered)

    def test_generation_prompt_marks_thinking_by_its_flag(self):
        contract = constraints.TokenizerContract(
            262144,
            (1, 50, 106),
            "<turn|>",
            101,
            48,
            100,
            "<|channel>thought\n",
            "<channel|>",
            "<|think|>",
        )
        app = fixtures.make_frontend(
            tokenizer(TEMPLATE),
            None,
            "test-model",
            4096,
            10,
            2,
            vision=False,
            contract=contract,
        )
        for kwargs, thinking in (
            ({"enable_thinking": True}, True),
            ({"enable_thinking": False}, False),
            ({}, False),
        ):
            with self.subTest(kwargs=kwargs):
                job = app.prepare(
                    {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "chat_template_kwargs": kwargs,
                    },
                    deadline=fixtures.FOREVER,
                )
                self.assertEqual(job.thinking, thinking)


class Gemma4ToolCallReadingTests(unittest.TestCase):
    def read(self, text, schemas=None, **kwargs):
        return project(text, policy(schemas), **kwargs)

    def test_calls_read_chunked_and_whole(self):
        text = (
            "Let me look. "
            + gcall(
                "lookup",
                "key:" + gstr("alpha") + ",n:42,ok:true,"
                "vals:[1," + gstr("x") + ",{y:2}]",
            )
            + " done. "
            + gcall("lookup", "key:" + gstr("beta"))
        )
        schemas = {
            "lookup": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "n": {"type": "integer"},
                    "ok": {"type": "boolean"},
                    "vals": {"type": "array"},
                },
            }
        }
        for size in (None, 1, 5):
            with self.subTest(size=size):
                content, calls, _ = self.read(text, schemas, size=size)
                self.assertEqual(content.strip(), "Let me look.  done.")
                self.assertEqual(
                    [json.loads(call["function"]["arguments"]) for call in calls],
                    [
                        {
                            "key": "alpha",
                            "n": 42,
                            "ok": True,
                            "vals": [1, "x", {"y": 2}],
                        },
                        {"key": "beta"},
                    ],
                )
                self.assertTrue(
                    all(call["function"]["name"] == "lookup" for call in calls)
                )

    def test_strings_hold_commas_braces_and_markup(self):
        content, calls, _ = self.read(gcall("f", "text:" + gstr("a,b} {c:1} ,x")))
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"text": "a,b} {c:1} ,x"},
        )

    def test_call_cut_at_the_limit_keeps_its_arguments(self):
        cut = CALL_OPEN + "lookup{key:" + gstr("alp")
        _, calls, events = project(cut, policy(), incomplete=True)
        self.assertEqual(
            [(c["function"]["name"], c["function"]["arguments"]) for c in calls],
            [("lookup", '{"key":"alp"')],
        )
        self.assertEqual(streamed_arguments(events), '{"key":"alp"')

    def test_unclosed_call_still_reads(self):
        _, calls, _ = self.read(CALL_OPEN + "lookup{key:" + gstr("alpha"))
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"key": "alpha"}
        )

    def test_declared_string_streams_its_characters(self):
        schemas = {
            "lookup": {"type": "object", "properties": {"key": {"type": "string"}}}
        }
        _, calls, events = project(
            gcall("lookup", "key:" + gstr("alpha")),
            policy(schemas),
        )
        self.assertEqual(streamed_arguments(events), '{"key":"alpha"}')
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"key": "alpha"}
        )

    def test_channel_close_in_content_is_dropped(self):
        content, calls, _ = self.read("Sure." + CHANNEL_CLOSE + " Here.")
        self.assertEqual((content, calls), ("Sure. Here.", []))


class Gemma4ReasoningTests(unittest.TestCase):
    def splitter(self, thinking, **kwargs):
        return output.ReasoningSplitter(
            thinking, think_open=CHANNEL_OPEN, think_end=CHANNEL_CLOSE, **kwargs
        )

    def feed(self, splitter, text, size=None):
        events = []
        for offset in range(0, len(text), size or len(text)):
            events += splitter.put(text[offset : offset + (size or len(text))])
        return events + splitter.finish()

    def test_thought_channel_splits_from_the_answer(self):
        text = CHANNEL_OPEN + "think it through\n" + CHANNEL_CLOSE + "The answer."
        for size in (None, 1, 7):
            with self.subTest(size=size):
                events = self.feed(self.splitter(True), text, size)
                merged = {}
                for kind, part in events:
                    merged[kind] = merged.get(kind, "") + part
                self.assertEqual(
                    merged,
                    {
                        "reasoning_content": "think it through\n",
                        "content": "The answer.",
                    },
                )

    def test_model_writes_the_channel_itself(self):
        # A hybrid prompt that ends at the role tag: the output opens the
        # channel without the template marking it.
        text = CHANNEL_OPEN + "why not\n" + CHANNEL_CLOSE + "Done."
        for size in (None, 3):
            with self.subTest(size=size):
                events = self.feed(self.splitter(False), text, size)
                merged = {}
                for kind, part in events:
                    merged[kind] = merged.get(kind, "") + part
                self.assertEqual(
                    merged,
                    {"reasoning_content": "why not\n", "content": "Done."},
                )

    def test_disabled_prompt_answers_directly(self):
        events = self.feed(self.splitter(False), "Just the answer.")
        self.assertEqual(events, [("content", "Just the answer.")])


class Gemma4ContractTests(unittest.TestCase):
    VOCABULARY = {
        "text": 0,
        "<eos>": 1,
        "<turn|>": 106,
        "<|tool_response>": 50,
        "<tool_call|>": 49,
        "<|tool_call>": 48,
        '<|"|>': 52,
        "<|think|>": 98,
        "<|channel>": 100,
        "<channel|>": 101,
    }

    def tokenizer(self):
        return mock.Mock(
            get_vocab=lambda: dict(self.VOCABULARY),
            encode=lambda text, **_: [self.VOCABULARY[text]],
            eos_token_id=1,
        )

    CONFIG = {
        "eos_token_id": [1, 106],
        "text_config": {"vocab_size": 262144, "eos_token_id": [1, 106]},
    }

    def test_channel_framing_and_call_tokens_resolve(self):
        contract = constraints.validate_tokenizer(self.tokenizer(), self.CONFIG)
        self.assertEqual(contract.eos_tokens, (1, 106))
        self.assertEqual(contract.think_end_id, 101)
        self.assertEqual(contract.think_open_id, 100)
        self.assertEqual(contract.think_open, CHANNEL_OPEN)
        self.assertEqual(contract.think_end, CHANNEL_CLOSE)
        self.assertEqual(contract.think_flag, "<|think|>")
        self.assertEqual(contract.tool_call_open_id, 48)

    def test_generation_config_stop_tokens_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(self.CONFIG))
            (root / "generation_config.json").write_text(
                json.dumps({"eos_token_id": [1, 106, 50]})
            )
            contract = constraints.validate_tokenizer(self.tokenizer(), root)
            self.assertEqual(contract.eos_tokens, (1, 50, 106))

    def test_generation_prompt_detects_thinking_by_flag(self):
        text = "<|turn>model\n"
        rendered = (
            "<|turn>system\n<|think|>\nLeading<turn|>\n<|turn>user\nHi<turn|>\n" + text
        )
        thinking, count = request_frontend._generation_prompt(
            (text, (105,)), rendered, [1, 105], think_flag="<|think|>"
        )
        self.assertTrue(thinking)
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()

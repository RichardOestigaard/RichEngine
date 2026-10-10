"""DiffusionGemma's chat template variant and canvas-burst delivery.

The fixture is the unmodified upstream chat template of
google/diffusiongemma-26B-A4B-it. It is the same Gemma 4 family as
gemma4.jinja except that the non-thinking generation prompt is a bare
model-turn opening: upstream dropped the empty closed thought channel
that google/gemma-4-26B-A4B-it still writes. Both spellings are in the
wild, so both are covered.

The diffusion runtime commits output in canvas chunks (up to 256 tokens
at once) rather than per token. The server-side delivery path already
takes token batches: these tests cover a full 256-token burst through
the incremental detokenizer, the reasoning splitter (whose gemma4
thought-channel markers may land mid-canvas) and the SSE stream.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import test_server as fixtures
from dev.tests.engine.test_gemma4 import (
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    gcall,
    gstr,
    tokenizer,
)
from dev.tests.test_server import ByteLevelTestTokenizer
from server import backend as backend_api
from server import chat_templates, constraints, output, tool_schema

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/chat_templates/diffusiongemma.jinja"
)
TEMPLATE = FIXTURE.read_text()
GEMMA4_TEMPLATE = (FIXTURE.parent / "gemma4.jinja").read_text()

USER = {"role": "user", "content": "Hi"}
NL = chr(10)
TURN_MODEL = "<|turn>model\n"
SYSTEM_THINK = "<|turn>system\n<|think|>\n"
THINK_FLAG = "<|think|>"
AGAIN_TURN = "<|turn>user\nAgain<turn|>"


def render(messages, template=TEMPLATE, **kwargs):
    return tokenizer(template).apply_chat_template(messages, tokenize=False, **kwargs)


class DiffusionGemmaTemplateTests(unittest.TestCase):
    def test_dialect_and_later_system_detect(self):
        chosen = chat_templates.ChatTemplates(tokenizer(TEMPLATE)).select(None)
        self.assertEqual(chosen.later_system, chat_templates.NATIVE)
        self.assertIs(chosen.dialect, tool_schema.GEMMA4)

    def test_non_thinking_generation_prompt_is_bare(self):
        rendered = render([USER], add_generation_prompt=True)
        self.assertTrue(rendered.endswith(TURN_MODEL), rendered)
        self.assertNotIn(CHANNEL_OPEN + CHANNEL_CLOSE, rendered[-64:])
        # google/gemma-4-26B-A4B-it still closes an empty thought channel
        # here: both generation-prompt spellings are in the wild.
        other = render([USER], template=GEMMA4_TEMPLATE, add_generation_prompt=True)
        self.assertTrue(
            other.endswith(TURN_MODEL + CHANNEL_OPEN + CHANNEL_CLOSE),
            other,
        )

    def test_enable_thinking_flag_opens_the_system_turn(self):
        rendered = render([USER], add_generation_prompt=True, enable_thinking=True)
        self.assertIn(SYSTEM_THINK, rendered)
        self.assertTrue(rendered.endswith(TURN_MODEL), rendered)

    def _call_conversation(self):
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
                        "arguments": {"key": "alpha"},
                    },
                }
            ],
        }
        result = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "beta",
        }
        return [
            {"role": "system", "content": "Leading instructions"},
            {"role": "user", "content": "First question"},
            call,
            result,
            {"role": "user", "content": "Next question"},
        ]

    def test_thinking_gate_and_preserve_thinking(self):
        messages = self._call_conversation()
        # Closed history: the later user turn gates the call's reasoning
        # out unless preserve_thinking keeps it with the calls.
        stripped = render(messages, add_generation_prompt=True)
        self.assertNotIn("Look it up", stripped)
        kept = render(messages, add_generation_prompt=True, preserve_thinking=True)
        self.assertIn(
            CHANNEL_OPEN
            + "Look it up\n"
            + CHANNEL_CLOSE
            + gcall("lookup", "key:" + gstr("alpha")),
            kept,
        )

    def test_ongoing_turn_keeps_its_thinking(self):
        # Reasoning after the last user message is the live thought: the
        # thinking_gate keeps it without preserve_thinking.
        rendered = render(self._call_conversation()[:3], add_generation_prompt=True)
        self.assertIn(CHANNEL_OPEN + "Look it up\n" + CHANNEL_CLOSE, rendered)

    def test_tool_call_arguments_must_be_a_mapping(self):
        bad = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "x",
                    },
                }
            ],
        }
        with self.assertRaises(Exception):
            render([USER, bad], add_generation_prompt=False)

    def test_none_content_renders(self):
        # Upstream's null handling: an assistant turn without content.
        rendered = render(
            [
                USER,
                {"role": "assistant", "content": None},
                {"role": "user", "content": "Again"},
            ],
            add_generation_prompt=True,
        )
        self.assertIn(AGAIN_TURN, rendered)

    def test_thinking_detection_uses_the_flag(self):
        contract = constraints.TokenizerContract(
            262144,
            (1, 50, 106),
            "<turn|>",
            101,
            48,
            100,
            CHANNEL_OPEN,
            CHANNEL_CLOSE,
            THINK_FLAG,
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
                        "messages": [dict(USER)],
                        "chat_template_kwargs": kwargs,
                    },
                    deadline=fixtures.FOREVER,
                )
                self.assertEqual(job.thinking, thinking)


class DiffusionGemmaContractTests(unittest.TestCase):
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

    def test_generation_config_stop_tokens_resolve(self):
        # generation_config.json states eos_token_id [1, 106, 50].
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "eos_token_id": [1, 106],
                        "text_config": {
                            "vocab_size": 262144,
                            "eos_token_id": 1,
                        },
                    }
                )
            )
            (root / "generation_config.json").write_text(
                json.dumps({"eos_token_id": [1, 106, 50]})
            )
            contract = constraints.validate_tokenizer(self.tokenizer(), root)
        self.assertEqual(contract.eos_tokens, (1, 50, 106))
        self.assertEqual(contract.think_flag, THINK_FLAG)
        self.assertEqual(
            (contract.think_open, contract.think_end),
            (CHANNEL_OPEN, CHANNEL_CLOSE),
        )
        self.assertEqual(contract.tool_call_open_id, 48)


class CanvasBurstTests(unittest.TestCase):
    # A diffusion canvas commits up to 256 tokens per TokensEvent; the
    # streamer decodes a burst exactly as it decodes the same tokens
    # delivered one at a time.

    def test_a_full_canvas_decodes_like_per_token_delivery(self):
        raw = [b"alpha ", b"beta "] * 100
        raw += [bytes([v]) for v in "中文中".encode()]
        raw += [bytes([v]) for v in "👨‍👩‍👧‍👦".encode()]
        raw += [b" tail ", b".", b"!"]
        raw += [b"x"] * (256 - len(raw))
        tokenizer = ByteLevelTestTokenizer(raw)
        self.assertEqual(len(tokenizer.token_ids), 256)
        expected = tokenizer.decode(tokenizer.token_ids)
        burst, steps = [], []
        streamer = backend_api.CallbackStreamer(tokenizer, burst.append)
        streamer.put_tokens(tokenizer.token_ids)
        streamer.end()
        streamer = backend_api.CallbackStreamer(tokenizer, steps.append)
        for token_id in tokenizer.token_ids:
            streamer.put_tokens([token_id])
        streamer.end()
        self.assertEqual("".join(burst), expected)
        self.assertEqual("".join(steps), expected)

    def test_stop_sequence_mid_canvas_truncates_once(self):
        raw = [b"keep ", b"<ST", b"OP>"]
        raw += [b"drop ", b"more "] * 126
        raw += [b"end"]
        tokenizer = ByteLevelTestTokenizer(raw)
        self.assertEqual(len(tokenizer.token_ids), 256)
        chunks, stopped = [], []
        streamer = backend_api.CallbackStreamer(
            tokenizer,
            chunks.append,
            ("<STOP>",),
            lambda: stopped.append(True),
        )
        streamer.put_tokens(tokenizer.token_ids)
        streamer.put_tokens([tokenizer.token_ids[-1]])
        streamer.end()
        self.assertEqual("".join(chunks), "keep ")
        self.assertEqual(stopped, [True])
        self.assertEqual(streamer.stop_sequence, "<STOP>")

    def test_thought_channel_splits_out_of_a_canvas(self):
        raw = [
            b"<|channel>",
            b"thought\n",
            b"plan ",
            b"it\n",
            b"<channel|>",
        ]
        raw += [b"The ", b"answer", b" is ", b"42."] * 62
        raw += [b"tail"] * (256 - len(raw))
        tokenizer = ByteLevelTestTokenizer(raw)
        self.assertEqual(len(tokenizer.token_ids), 256)
        chunks = []
        streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
        streamer.put_tokens(tokenizer.token_ids)
        streamer.end()
        splitter = output.ReasoningSplitter(
            True, think_open=CHANNEL_OPEN, think_end=CHANNEL_CLOSE
        )
        merged = {}
        for chunk in chunks:
            for kind, text in splitter.put(chunk):
                merged[kind] = merged.get(kind, "") + text
        for kind, text in splitter.finish():
            merged[kind] = merged.get(kind, "") + text
        self.assertEqual(merged["reasoning_content"], "plan it\n")
        self.assertEqual(merged["content"], "The answer is 42." * 62 + "tail" * 3)

    def test_canvas_burst_streams_over_sse(self):
        batch = [1, 2, 3] + [14] * 126 + [15] * 127
        self.assertEqual(len(batch), 256)
        runtime = fixtures.FakeRuntime(fixtures.Plan([batch]))
        harness = fixtures.Harness(runtime)
        try:
            status, _, payload = harness.request(
                "POST",
                "/v1/chat/completions",
                fixtures.ServerTest.body(stream=True),
            )
            self.assertEqual(status, 200, payload)
            events = [
                line[6:]
                for line in payload.decode().splitlines()
                if line.startswith("data: ")
            ]
            self.assertEqual(events[-1], "[DONE]")
            chunks = [json.loads(event) for event in events[:-1]]
            deltas = [
                chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]
            ]
            reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
            content = "".join(delta.get("content", "") for delta in deltas)
            self.assertEqual(reasoning, "because ")
            self.assertEqual(
                content,
                "answer\n" + "first " * 126 + "second\n" * 127,
            )
        finally:
            harness.close()


if __name__ == "__main__":
    unittest.main()

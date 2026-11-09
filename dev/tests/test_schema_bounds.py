"""Integer minimum/maximum bounds in structured output: is the grammar the
engine generates bound-aware? The DFA cannot compare magnitudes — bounded
schemas fall outside compile_dfa's subset — so every bounded request runs
the llguidance grammar path. These tests feed candidate documents token by
token through the exact grammar tool_schema.json_grammar / tool_grammar
build and assert out-of-range numbers die in the matcher. If a future
llguidance release drops minimum/maximum handling in its %json schemas, the
reject cases go green-to-red here, before it ships as silent out-of-range
generations."""

import unittest

from llguidance import LLMatcher

from dev.tests.test_dfa import DfaCase, strict_tool
from server import dfa, tool_schema
from server.errors import APIError
from server.output import validate_response_content
from server.schema_validation import build_validator

BOUNDED = {
    "type": "object",
    "properties": {"n": {"type": "integer", "minimum": 5, "maximum": 9}},
    "required": ["n"],
    "additionalProperties": False,
}


class SchemaBoundsTest(DfaCase, unittest.TestCase):
    """Token-level acceptance of the structured-output grammar."""

    def ids_for(self, text, specials=()):
        """`text` as token ids: each named special string as its token id,
        all other text as single-byte tokens — the encoding a real served
        tokenizer gives, where dialect literals like `<function=` are plain
        text rather than the extra special tokens this fixture's vocab
        defines for other dialects."""
        byte_to_id = {
            piece[0]: index
            for index, piece in enumerate(self.vocab_tokens)
            if len(piece) == 1
        }
        ids = []
        rest = text
        while rest:
            hits = [
                (rest.find(mark), mark) for mark in specials if rest.find(mark) >= 0
            ]
            if not hits:
                ids += [byte_to_id[b] for b in rest.encode()]
                break
            at, mark = min(hits)
            ids += [byte_to_id[b] for b in rest[:at].encode()]
            ids.append(self.tokenizer.token_to_id(mark))
            rest = rest[at + len(mark) :]
        return ids

    def feeds(self, grammar, text, specials=()):
        """Whether `text` is a complete accepted document of `grammar`:
        every token consumed, then the grammar takes the stop token."""
        matcher = LLMatcher(self.guidance, grammar, log_level=0)
        self.assertFalse(matcher.is_error(), matcher.get_error())
        for token_id in self.ids_for(text, specials):
            if not matcher.consume_token(token_id):
                return False
        return matcher.is_stopped() or matcher.consume_token(self.eos[0])

    def check(self, schema, accept, reject):
        grammar = self.json_grammar(schema, False)
        for text in accept:
            self.assertTrue(self.feeds(grammar, text), f"should accept {text!r}")
        for text in reject:
            self.assertFalse(self.feeds(grammar, text), f"should reject {text!r}")

    def test_integer_minimum_maximum(self):
        self.check(
            {"type": "integer", "minimum": 5, "maximum": 9},
            accept=["5", "7", "9"],
            reject=["4", "10", "0", "99", "-7"],
        )

    def test_integer_exclusive_bounds(self):
        self.check(
            {"type": "integer", "exclusiveMinimum": 5, "exclusiveMaximum": 9},
            accept=["6", "8"],
            reject=["5", "9", "4", "10"],
        )

    def test_integer_negative_range(self):
        self.check(
            {"type": "integer", "minimum": -9, "maximum": -5},
            accept=["-7", "-5", "-9"],
            reject=["-10", "-4", "5"],
        )

    def test_number_bounds(self):
        self.check(
            {"type": "number", "minimum": 5, "maximum": 9},
            accept=["7", "5.5", "9"],
            reject=["4.9", "10", "3"],
        )

    def test_bounded_property_inside_object(self):
        self.check(
            BOUNDED,
            accept=['{"n":5}', '{"n":7}', '{"n":9}'],
            reject=['{"n":3}', '{"n":12}', '{"n":-5}'],
        )

    def test_bounds_exclude_schema_from_dfa_subset(self):
        """A bounded numeric node is not regular: compile_dfa returns None
        and the request takes the llguidance grammar path — which the tests
        above prove enforces the bounds. Pin the fallback so a future DFA
        that compiles bounds *wrong* fails loudly instead of silently
        shadowing the grammar path."""
        self.assertIsNone(
            dfa.compile_dfa(self.json_spec(BOUNDED, False), self.markers)
        )
        # The same property without bounds stays in the subset.
        unbounded = {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
        }
        self.assertIsNotNone(
            dfa.compile_dfa(self.json_spec(unbounded, False), self.markers)
        )

    def test_tool_grammar_bounded_argument(self):
        """A strict tool's bounded integer parameter is generated through
        %json in the call grammar: out-of-range arguments die in the matcher."""
        tools = [
            strict_tool(
                "lookup",
                {"k": {"type": "integer", "minimum": 5, "maximum": 9}},
                ["k"],
            )
        ]
        policy = tool_schema.normalize_tools(
            tools, "required", True, dialect=tool_schema.QWEN3_XML
        )[1]
        grammar = self.tool_grammar(policy, False, None)
        specials = ("<tool_call>", "</tool_call>")
        for value, ok in (("7", True), ("3", False), ("12", False)):
            text = (
                "<tool_call>\n<function=lookup>\n<parameter=k>\n"
                + value
                + "\n</parameter>\n</function>\n</tool_call>"
            )
            self.assertEqual(
                self.feeds(grammar, text, specials), ok, f"argument {value!r}"
            )

    def test_post_validation_backstop(self):
        """Even where a grammar path ignored bounds, the response_format
        validator built by normalize_response_format refuses an out-of-range
        document — validate_response_content raises."""
        validator = build_validator(BOUNDED)
        validate_response_content('{"n":7}', validator)
        with self.assertRaises(APIError):
            validate_response_content('{"n":3}', validator)


if __name__ == "__main__":
    unittest.main()

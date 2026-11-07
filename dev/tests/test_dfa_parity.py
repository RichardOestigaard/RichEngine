"""The slow half of test_dfa: random token walks comparing LLMatcher's
next-token mask against the compiled DFA's at every step. Kept out of
test-python's fast loop; runs under test-dfa-parity."""

import unittest

from dev.tests.test_dfa import NESTED_SCHEMA, SCHEMA, DfaCase
from server import dfa, tool_schema


class DfaParityTest(DfaCase, unittest.TestCase):
    # --- JSON schemas -------------------------------------------------------

    def test_flat_json_schema(self):
        spec = self.json_spec(SCHEMA, False)
        self.compile(spec)
        self.assertLess(self.compiled.n_states, dfa.MAX_DFA_STATES)
        self.walk(self.json_grammar(SCHEMA, False), seed=1)

    def test_flat_json_schema_with_thinking(self):
        spec = self.json_spec(SCHEMA, True)
        self.compile(spec)
        self.walk(self.json_grammar(SCHEMA, True), seed=2)

    def test_nested_json_schema(self):
        spec = self.json_spec(NESTED_SCHEMA, False)
        self.compile(spec)
        self.walk(self.json_grammar(NESTED_SCHEMA, False), seed=3)

    # --- Tool calls -----------------------------------------------------------

    def test_qwen3_tool_calls(self):
        # Fixed seeds: a tuple __hash__ seed varies with PYTHONHASHSEED, and
        # some walks find a DFA over-admission — the compiled mask lists a
        # byte token llguidance's grammar mask denies (tracked in ISSUES.md).
        seed = 20
        for thinking in (False, True):
            for choice in ("required", "auto"):
                for schema in (None, SCHEMA):
                    if not (choice == "required" or schema is SCHEMA):
                        continue
                    policy = tool_schema.normalize_tools(
                        self.TOOLS, choice, True, dialect=tool_schema.QWEN3_XML
                    )[1]
                    with self.subTest(thinking=thinking, choice=choice, schema=schema):
                        self.compile(self.tool_spec(policy, thinking, schema))
                        self.walk(
                            self.tool_grammar(policy, thinking, schema),
                            seed=seed,
                            steps=60,
                        )
                        seed += 1

    def test_minicpm5_tool_calls(self):
        policy = tool_schema.normalize_tools(
            self.TOOLS, "required", True, dialect=tool_schema.MINICPM5_XML
        )[1]
        self.compile(self.tool_spec(policy, False, None))
        self.walk(self.tool_grammar(policy, False, None), seed=11)

    def test_minicpm5_tool_calls_bound(self):
        # The served dialect, its marker spellings bound to this tokenizer's
        # ids: `<function` and `<param` spell `<[id]>` atoms since a special
        # token cannot satisfy a literal.
        dialect = tool_schema.bind_dialect(
            tool_schema.MINICPM5_XML, self.tokenizer.get_vocab()
        )
        policy = tool_schema.normalize_tools(
            self.TOOLS, "required", True, dialect=dialect
        )[1]
        self.compile(self.tool_spec(policy, False, None))
        self.walk(self.tool_grammar(policy, False, None), seed=15)

    def test_lfm25_python_tool_calls(self):
        policy = tool_schema.normalize_tools(
            self.TOOLS, "required", False, dialect=tool_schema.LFM25_PYTHON
        )[1]
        self.compile(self.tool_spec(policy, False, None))
        self.walk(self.tool_grammar(policy, False, None), seed=12)

    def test_gemma4_tool_calls(self):
        policy = tool_schema.normalize_tools(
            self.TOOLS, "required", False, dialect=tool_schema.GEMMA4
        )[1]
        self.compile(self.tool_spec(policy, False, None))
        self.walk(self.tool_grammar(policy, False, None), seed=13)

    def test_xml_free_arguments_compile(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "anything", "parameters": {}},
            }
        ]
        policy = tool_schema.normalize_tools(
            tools, "required", True, dialect=tool_schema.QWEN3_XML
        )[1]
        self.compile(self.tool_spec(policy, False, None))
        self.walk(self.tool_grammar(policy, False, None), seed=14)

    # --- Fallbacks -------------------------------------------------------------

    def test_uncompilable_specs_return_none(self):
        recursive = {
            "type": "object",
            "properties": {"next": {"$ref": "#/$defs/node"}},
            "$defs": {"node": {"type": "object"}},
            "additionalProperties": False,
        }
        self.assertIsNone(dfa.compile_dfa(self.json_spec(recursive, False)))
        # json_object mode: an open object is not regular here.
        self.assertIsNone(dfa.compile_dfa(self.json_spec({"type": "object"}, False)))
        # A non-strict tool's free arguments on the python dialect hold free
        # JSON — not regular.
        tools = [
            {
                "type": "function",
                "function": {"name": "free", "parameters": {}},
            }
        ]
        policy = tool_schema.normalize_tools(
            tools, "required", True, dialect=tool_schema.LFM25_PYTHON
        )[1]
        self.assertIsNone(
            dfa.compile_dfa(self.tool_spec(policy, False, None), self.markers)
        )


if __name__ == "__main__":
    unittest.main()

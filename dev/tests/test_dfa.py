"""DFA-compiled constraints produce the same masks as the llguidance
grammar path — random token walks compare LLMatcher's next-token mask
against the compiled DFA's at every step."""

import array
import os
import random
import threading
import unittest
from collections import OrderedDict
from unittest import mock

from llguidance import LLExecutor, LLMatcher, LLTokenizer
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask_par
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from dev.tests.test_server import _byte_alphabet
from server import constraints, dfa, tool_schema

# Every tool dialect's structural tokens plus the standard specials, so
# canonical tokenization protects their spellings exactly as in production.
SPECIALS = [
    "<tool_call>",
    "</tool_call>",
    "</think>",
    "<eos>",
    "<|tool_call>",
    "<tool_call|>",
    '<|"|>',
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<function",
    "</function>",
    "<param",
    "</param>",
    "<tool_sep>",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "count": {"type": "integer"},
        "ok": {"type": "boolean"},
        "mode": {"enum": ["fast", "slow", 2, None]},
    },
    "required": ["answer", "count", "ok", "mode"],
    "additionalProperties": False,
}

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "point": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        },
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "opt1": {"type": "integer"},
        "opt2": {"type": "null"},
        "note": {"type": "string"},
    },
    "required": ["point", "tags"],
    "additionalProperties": False,
}


def strict_tool(name, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


class DfaParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocabulary = {char: token for token, char in _byte_alphabet().items()}
        vocabulary["[UNK]"] = len(vocabulary)
        vocabulary["<eos>"] = len(vocabulary)
        cls.tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        cls.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                pre_tokenizers.Split(Regex(""), behavior="isolated"),
            ]
        )
        cls.tokenizer.decoder = decoders.ByteLevel()
        cls.tokenizer.add_special_tokens(SPECIALS)
        cls.guidance = LLTokenizer(
            cls.tokenizer.to_str(), eos_token=cls.tokenizer.token_to_id("<eos>")
        )
        cls.executor = LLExecutor()
        cls.eos = (cls.tokenizer.token_to_id("<eos>"),)
        cls.vocab_tokens = constraints._vocab_token_bytes(
            cls.tokenizer, cls.guidance.vocab_size
        )
        cls.specials = frozenset(
            token_id
            for token_id in range(cls.guidance.vocab_size)
            if cls.guidance.is_special_token(token_id)
        )
        cls.markers = tuple(
            cls.vocab_tokens[t] for t in cls.specials if cls.vocab_tokens[t]
        )
        cls.think_end_id = cls.tokenizer.token_to_id("</think>")

    def mask_ids(self, matcher):
        bitmask = allocate_token_bitmask(1, self.guidance.vocab_size)
        fill_next_token_bitmask_par(self.executor, [(matcher, 0)], bitmask)
        words = array.array("I", bitmask.tobytes())
        return {
            index * 32 + bit
            for index, word in enumerate(words)
            for bit in range(32)
            if word >> bit & 1
        }

    def advance(self, state, token_id):
        return self.compiled.advance_token(
            state, token_id, self.vocab_tokens, self.eos, self.specials
        )

    def walk(self, grammar, steps=80, seed=0):
        """Random token walks: identical allowed sets at every step."""
        matcher = LLMatcher(self.guidance, grammar, log_level=0)
        self.assertFalse(matcher.is_error(), matcher.get_error())
        rng = random.Random(seed)
        state = 0
        seen = []
        for step in range(steps):
            expected = self.mask_ids(matcher)
            got = set(
                self.compiled.masks_for_state(
                    state, self.vocab_tokens, self.eos, self.specials
                )
            )
            self.assertEqual(expected, got, f"step {step}")
            if not expected:
                return
            # A token outside the mask must die in the DFA.
            dead = [
                tid
                for tid in rng.sample(range(self.guidance.vocab_size), 8)
                if tid not in expected and tid not in self.eos
            ]
            for tid in dead:
                self.assertLess(
                    self.advance(state, tid), 0, f"step {step} token {tid}"
                )
            # llguidance's mask can list a special token its consumer then
            # refuses; skip such picks — the walk continues with the rest.
            candidates = list(sorted(expected))
            token_id = None
            while candidates and token_id is None:
                pick = rng.choice(candidates)
                candidates.remove(pick)
                if pick in self.eos:
                    token_id = pick
                    break
                probe = LLMatcher(self.guidance, grammar, log_level=0)
                for prev in seen:
                    probe.consume_token(prev)
                if probe.consume_token(pick):
                    token_id = pick
            if token_id is None:
                return
            if token_id in self.eos:
                self.assertTrue(self.compiled.is_accepting(state))
                matcher.consume_token(token_id)
                return
            self.assertTrue(matcher.consume_token(token_id), f"step {step}")
            seen.append(token_id)
            state = self.advance(state, token_id)
            self.assertGreaterEqual(state, 0, f"step {step}")

    def compile(self, spec):
        self.compiled = dfa.compile_dfa(spec, self.markers)
        self.assertIsNotNone(self.compiled)
        return self.compiled

    def tool_grammar(self, policy, thinking, schema):
        return tool_schema.tool_grammar(
            policy, thinking, schema, think_end_id=self.think_end_id
        )

    def tool_spec(self, policy, thinking, schema):
        return dfa.tool_spec(policy, thinking, schema, think_end="</think>")

    def json_grammar(self, schema, thinking):
        return tool_schema.json_grammar(
            schema, thinking, think_end_id=self.think_end_id, think_end_text="</think>"
        )

    def json_spec(self, schema, thinking):
        return dfa.json_spec(schema, thinking, think_end="</think>")

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

    TOOLS = [
        strict_tool(
            "lookup",
            {"query": {"type": "string"}, "k": {"type": "integer"}},
            ["query", "k"],
        ),
        strict_tool("finish", {}, []),
    ]

    def test_qwen3_tool_calls(self):
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
                            seed=(thinking, choice, schema is not None).__hash__() & 0xFFFF,
                            steps=60,
                        )

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
        self.assertIsNone(
            dfa.compile_dfa(self.json_spec({"type": "object"}, False))
        )
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


class DfaConstraintTest(DfaParityTest):
    """The constraint object's commit/mask/finish contract."""

    def constraint(self):
        compiled = dfa.compile_dfa(self.json_spec(SCHEMA, False), self.markers)
        contract = constraints.TokenizerContract(
            self.guidance.vocab_size, self.eos, "<eos>", self.think_end_id
        )
        return constraints.DfaConstraint(
            compiled,
            self.vocab_tokens,
            self.specials,
            contract,
            encode=self.tokenizer.encode,
        )

    def rows(self, payload):
        words = array.array("I", payload)
        width = (self.guidance.vocab_size + 31) // 32
        return [
            [
                index * 32 + bit
                for index, word in enumerate(words[start : start + width])
                for bit in range(32)
                if word >> bit & 1
            ]
            for start in range(0, len(words), width)
        ]

    def test_masks_commit_finish(self):
        c = self.constraint()
        first = self.rows(c.masks(()))
        self.assertEqual(len(first), 1)
        self.assertIn(ord("{"), first[0])
        self.assertNotIn(self.eos[0], first[0])
        # Commit the whole document token by token.
        for text, accepting in (('{"answer":"x","count":3,"ok":true,"mode":"fast"}', True),):
            c = self.constraint()
            for tid in self.tokenizer.encode(text).ids:
                c.commit([tid])
            c.finish()
        # An unaccepted prefix fails finish.
        c = self.constraint()
        for tid in self.tokenizer.encode('{"answer"').ids:
            c.commit([tid])
        with self.assertRaises(constraints.ConstraintError):
            c.finish()
        # A disallowed token fails commit-time validation on consume.
        c = self.constraint()
        for tid in self.tokenizer.encode('{"answer":"x"').ids:
            c.commit([tid])
        z = self.tokenizer.token_to_id("z")
        c.commit([z])
        with self.assertRaises(constraints.ConstraintError):
            c.masks(())

    def test_forced_tokens(self):
        c = self.constraint()
        # `{` and whitespace are both live at the start: nothing is forced.
        self.assertEqual(c.forced_tokens(), ())
        # After the call opener token, the fixed literal is forced.
        policy = tool_schema.normalize_tools(
            DfaParityTest.TOOLS, "required", True, dialect=tool_schema.QWEN3_XML
        )[1]
        compiled = dfa.compile_dfa(self.tool_spec(policy, False, None), self.markers)
        contract = constraints.TokenizerContract(
            self.guidance.vocab_size, self.eos, "<eos>", self.think_end_id
        )
        c = constraints.DfaConstraint(
            compiled, self.vocab_tokens, self.specials, contract,
            encode=self.tokenizer.encode,
        )
        forced = c.commit([self.tokenizer.token_to_id("<tool_call>")])
        self.assertTrue(forced)
        text = self.tokenizer.decode(list(forced), skip_special_tokens=False)
        self.assertTrue("\n<function=lookup>\n<parameter=query>\n".startswith(text))


class DfaWireTest(DfaParityTest):
    """The CONSTRAINT_DFA/VOCABULARY frame payload the engine walks."""

    def round_trip(self, message):
        from dev.tests.engine import native_peer
        from server import protocol as wire

        frame = bytes(wire.serialize_message(message))
        messages = native_peer.ClientFrameReader().feed(frame)
        self.assertEqual(len(messages), 1)
        return messages[0][0]

    def test_constraint_dfa_frame_round_trip(self):
        """The grammar's eos set and marker mask survive the wire."""
        from server import protocol as wire

        spec = self.json_spec(SCHEMA, False)
        compiled = self.compile(spec)
        eos_states = bytes(
            acc | ok for acc, ok in zip(compiled.accepting, compiled.eos_ok)
        )
        marker = b"".join(m.to_bytes(32, "little") for m in compiled.marker)
        message = wire.ConstraintDfaFrame(
            91,
            compiled.n_states,
            compiled.table.tobytes(),
            eos_states,
            self.eos,
            marker,
        )
        decoded = self.round_trip(message)
        self.assertIsInstance(decoded, wire.ConstraintDfaFrame)
        self.assertEqual(decoded.request_id, 91)
        self.assertEqual(decoded.n_states, compiled.n_states)
        self.assertEqual(decoded.eos_ids, self.eos)
        self.assertEqual(decoded.accepting, eos_states)
        self.assertEqual(decoded.marker, marker)

    def test_vocabulary_frame_round_trip(self):
        """The special-token bitset decodes to the id set it was packed from."""
        from server import protocol as wire

        tokens = tuple(b"t" * ((i % 3) + 1) for i in range(20))
        special = (1, 7, 19)
        decoded = self.round_trip(wire.VocabularyFrame(tokens, special))
        self.assertIsInstance(decoded, wire.VocabularyFrame)
        self.assertEqual(decoded.special_tokens, special)

    def test_marker_semantics(self):
        """A regular token dies on a marked edge; a special must take one.

        A hand-built DFA keeps the check independent of the compiler: state 1
        accepting, edge (0, 'a') marked, edge (0, 'b') unmarked."""
        n_states = 2
        table = array.array("h", [-1] * (n_states * 256))
        table[0 * 256 + ord("a")] = 1
        table[0 * 256 + ord("b")] = 1
        accepting = dfa._bitset([False, True])
        live = dfa._bitset([True, True])
        marker = [1 << ord("a"), 0]
        eos_ok = dfa._bitset([False, False])
        compiled = dfa.CompiledDfa(n_states, table, accepting, live, marker, eos_ok)
        vocab = (b"a", b"b", b"<a>")
        special = frozenset({2})
        eos = (9,)
        # Regular token on the marked edge dies; on the unmarked one lives.
        self.assertEqual(compiled.advance_token(0, 1, vocab, eos, special), 1)
        self.assertLess(compiled.advance_token(0, 0, vocab, eos, special), 0)
        # A special token must take a marked edge: 'a' is marked, so it
        # walks; on 'b' it dies.
        vocab3 = (b"a", b"b", b"a")
        self.assertEqual(compiled.advance_token(0, 2, vocab3, eos, special), 1)
        self.assertLess(compiled.advance_token(0, 0, vocab, eos, special), 0)
        # A special token whose bytes are the unmarked edge dies.
        self.assertLess(compiled.advance_token(0, 2, (b"a", b"b", b"b"), eos, special), 0)
        # Grammar eos stops only at an EOS-admitting state.
        self.assertEqual(compiled.advance_token(1, 9, vocab, eos, special), n_states)
        self.assertLess(compiled.advance_token(0, 9, vocab, eos, special), 0)

    def test_foreign_stop_maps_to_stopped(self):
        """An engine stop token the grammar's eos set does not name — always
        a special token — stops the mirror at an accepting state."""
        spec = self.json_spec(SCHEMA, False)
        compiled = self.compile(spec)
        contract = constraints.TokenizerContract(
            self.guidance.vocab_size, self.eos, "<eos>", self.think_end_id
        )
        c = constraints.DfaConstraint(
            compiled,
            self.vocab_tokens,
            self.specials,
            contract,
            encode=self.tokenizer.encode,
        )
        text = '{"answer":"x","count":3,"ok":true,"mode":"fast"}'
        for tid in self.tokenizer.encode(text).ids:
            c.commit([tid])
        # A special token the grammar does not know at an accepting state:
        # the mirror treats it as the stop, not an invalid token.
        foreign = self.tokenizer.token_to_id("<tool_call>")
        self.assertIn(foreign, self.specials)
        c.commit([foreign])
        c.finish()
        # A regular token still errors there.
        c = constraints.DfaConstraint(
            compiled, self.vocab_tokens, self.specials, contract,
            encode=self.tokenizer.encode,
        )
        for tid in self.tokenizer.encode(text).ids:
            c.commit([tid])
        c.commit([self.tokenizer.token_to_id("z")])
        with self.assertRaises(constraints.ConstraintError):
            c.masks(())


class DfaFactoryTest(DfaParityTest):
    """ConstraintFactory.create_dfa: the DFA ships to the engine while a
    TokenConstraint delegate on the same grammar serves masks, commits and
    the finishing check — identical output to the grammar path."""

    def factory(self):
        # A factory on the class tokenizer without the hf wrapper: its
        # fields are what __init__ sets, minus the from_tokenizer conversion
        # the class fixture already did.
        factory = constraints.ConstraintFactory.__new__(
            constraints.ConstraintFactory
        )
        factory.contract = constraints.TokenizerContract(
            self.guidance.vocab_size, self.eos, "<eos>", self.think_end_id
        )
        factory.hf_tokenizer = self.tokenizer
        factory.tokenizer = self.guidance
        factory.executor = self.executor
        factory.source_bytes = 0
        factory.cache = OrderedDict()
        factory._dfa_cache = OrderedDict()
        factory.lock = threading.Lock()
        factory.pending = {}
        factory.hits = factory.misses = 0
        factory._vocab = None
        return factory

    def test_forced_mode_masks_delegate_to_matcher(self):
        factory = self.factory()
        spec = self.json_spec(SCHEMA, False)
        grammar = self.json_grammar(SCHEMA, False)
        with mock.patch.dict(os.environ, {"RICHENGINE_DFA_MODE": "forced"}):
            constraint = factory.create_dfa(spec, grammar)
        self.assertIsInstance(constraint, constraints.DfaConstraint)
        reference = factory.create(grammar)
        # The delegate's cached start-state mask answers the initial request.
        self.assertEqual(constraint.masks(()), reference.masks(()))
        for tid in self.tokenizer.encode('{"answer":"x","count":3').ids:
            constraint.commit([tid])
            reference.commit([tid])
            self.assertEqual(constraint.masks(()), reference.masks(()))
        for tid in self.tokenizer.encode(',"ok":true,"mode":"fast"}').ids:
            constraint.commit([tid])
            reference.commit([tid])
        constraint.finish()
        reference.finish()
        self.assertEqual(constraint.masks(()), reference.masks(()))

    def test_off_mode_returns_none(self):
        factory = self.factory()
        with mock.patch.dict(os.environ, {"RICHENGINE_DFA_MODE": "off"}):
            self.assertIsNone(
                factory.create_dfa(
                    self.json_spec(SCHEMA, False), self.json_grammar(SCHEMA, False)
                )
            )


if __name__ == "__main__":
    unittest.main()

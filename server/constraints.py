"""Tokenizer contract and cached token-level output constraints."""

import array
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, wait
from dataclasses import dataclass
from pathlib import Path

from llguidance import LLExecutor, LLMatcher, LLTokenizer
from llguidance.hf import from_tokenizer as guidance_tokenizer
from llguidance.numpy import (
    allocate_token_bitmask,
    fill_next_token_bitmask_par,
    fill_next_token_bitmask_par_with_draft_tokens,
)

from . import dfa as dfa_compiler
from . import runtime as engine_runtime
from .errors import APIError, ConstraintError
from .tool_schema import (
    THINK_END,
    THINK_OPEN,
    TOOL_CALL_OPEN,
    TOOL_CALL_OPEN_ALT,
    dialect_named,
)


def dfa_mode():
    """The engine's RICHENGINE_DFA_MODE tri-state, read the same way there:
    "full" computes a DFA lane's masks and forced spans on the GPU,
    "forced" keeps the host mask round trip and only derives forced spans
    and the automaton state on-device, "off" leaves the DFA unused.
    Anything unset or unrecognized is "off"."""
    mode = os.environ.get("RICHENGINE_DFA_MODE", "off").strip().lower()
    return mode if mode in ("full", "forced", "off") else "off"


# The think-block spellings the served families use, in preference order:
# (close text, open text, leading special token of the open, system-turn
# flag that marks a thinking-enabled prompt).
THINK_FRAMINGS = (
    (THINK_END, THINK_OPEN, THINK_OPEN, None),
    ("<channel|>", "<|channel>thought\n", "<|channel>", "<|think|>"),
)


@dataclass(frozen=True)
class TokenizerContract:
    """The ids the loaded model's configuration and tokenizer state.

    Everything grammar enforcement bounds or matches on comes from the served
    model rather than a model family pinned in code, so a checkpoint whose
    configuration states the same contract serves without a code change."""

    # The model's token count: bitmasks cover it and generated ids stay below.
    vocabulary: int
    # Every token the model's configuration declares a stop token.
    eos_tokens: tuple
    # The token that ends a chat turn; prompts split there for reuse.
    marker: str
    # The id the chat template's think-close text encodes to.
    think_end_id: int
    # The id the chat template's tool-call-open text encodes to, or None for
    # a tokenizer that spells tool calls differently (a model without one
    # cannot emit a call the grammars write anyway).
    tool_call_open_id: int | None = None
    # The think framing's open spelling and the special token that begins it
    # (decoded despite skip_special_tokens so the output parser sees it),
    # plus think_end's spelling, and the system-turn flag that marks a
    # thinking-enabled prompt where the family has one (gemma4's think
    # token).
    think_open_id: int | None = None
    think_open: str = "<think>"
    think_end: str = THINK_END
    think_flag: str | None = None


class TokenConstraint:
    # A mask request simulates at most the engine's target verify rows
    # (ExecutionLimits::targetVerifyRows, 16: the pending anchor and the
    # proposal table's 15 slots) and takes a mask before and after each.
    MAX_ROWS = 17

    def __init__(self, matcher, executor, contract, initial_mask=None, encode=None):
        self.matcher = matcher
        self.executor = executor
        self._encode = encode
        self.vocabulary = contract.vocabulary
        self.eos_tokens = contract.eos_tokens
        self.bitmask = allocate_token_bitmask(self.MAX_ROWS, self.vocabulary)
        # The grammar start state's mask, compiled once per grammar by the
        # factory. Serving it from cache keeps the only fully serialized mask
        # request — the one a finished prompt waits on before its first
        # token — off the grammar executor.
        self._initial_mask = initial_mask
        # Generated batches the grammar has not consumed yet. The reader
        # thread commits them; the mask thread consumes them before the next
        # mask, so reading the native stream never waits for the grammar.
        self._lock = threading.Lock()
        self._committed: list[tuple[int, ...]] = []
        # Whether generated tokens have advanced the grammar past its start
        # state; only then is a cached start-state mask no longer valid.
        self._advanced = False
        # A second matcher confined to the reader thread, advanced eagerly by
        # commit(). Its fast-forward tokens — the tokens the grammar forces
        # next, anchored at the position the engine's pending token occupies —
        # are pushed to the engine as guaranteed draft proposals.
        self._ff_matcher = None if matcher is None else matcher.deep_copy()
        self._ff_failed = False

    def forced_tokens(self):
        """The tokens the grammar forces at the reader thread's position, or
        empty. tokens[0] occupies the pending anchor's slot.

        llguidance's compute_ff_tokens only fires for canonical tokenizers;
        when it is empty the forced byte span still applies — encode it and
        validate the tokenization on a matcher copy so a span that cannot be
        consumed is never pushed."""
        if (
            self._ff_matcher is None
            or self._ff_failed
            or not self._encode
            or os.environ.get("RICHENGINE_DISABLE_FORCED_TOKENS")
        ):
            return ()
        try:
            tokens = tuple(self._ff_matcher.compute_ff_tokens())
            if tokens:
                return tokens
            forced = self._ff_matcher.compute_ff_bytes()
            if not forced:
                return ()
            try:
                text = forced.decode("utf-8")
            except UnicodeDecodeError:
                return ()
            candidate = tuple(self._encode(text, add_special_tokens=False))
            if len(candidate) < 2 or not candidate:
                return ()
            copy = self._ff_matcher.deep_copy()
            if not copy.consume_tokens(list(candidate)):
                return ()
            return candidate
        except Exception:
            self._ff_failed = True
            return ()

    def commit(self, token_ids, *, forced=True):
        """Consume generated tokens and return the span the grammar forces
        next. `forced=False` keeps every bookkeeping effect — the pending
        batch and the fast-forward matcher's position — but skips computing
        a span the caller is going to discard."""
        if any(not 0 <= token < self.vocabulary for token in token_ids):
            raise ConstraintError("generated token is out of range")
        with self._lock:
            self._committed.append(tuple(token_ids))
            if self._ff_matcher is not None and not self._ff_failed:
                try:
                    self._ff_matcher.consume_tokens(list(token_ids))
                except Exception:
                    self._ff_failed = True
            return self.forced_tokens() if forced else ()

    def finish(self):
        """Checks the tokens generated after the last mask."""
        self._consume_committed()

    def masks(self, simulation_tokens):
        self._consume_committed()
        if (
            not simulation_tokens
            and not self._advanced
            and self._initial_mask is not None
        ):
            return self._initial_mask
        if len(simulation_tokens) >= self.MAX_ROWS:
            raise ConstraintError("too many simulation tokens")
        in_range = next(
            (
                index
                for index, token in enumerate(simulation_tokens)
                if not 0 <= token < self.vocabulary
            ),
            len(simulation_tokens),
        )
        valid_count = self.matcher.validate_tokens(list(simulation_tokens[:in_range]))
        valid_tokens = simulation_tokens[:valid_count]
        if valid_tokens:
            fill_next_token_bitmask_par_with_draft_tokens(
                self.executor,
                [(self.matcher, 0, list(valid_tokens))],
                self.bitmask,
            )
        else:
            fill_next_token_bitmask_par(
                self.executor, [(self.matcher, 0)], self.bitmask
            )
        rows = len(simulation_tokens) + 1
        valid_rows = valid_count + 1
        if valid_rows < rows:
            self.bitmask[valid_rows:rows] = self.bitmask[valid_rows - 1]
        if not self.bitmask[:valid_rows].any(axis=1).all():
            raise ConstraintError("output grammar has no valid token")
        return self.bitmask[:rows].tobytes()

    def _consume_committed(self):
        with self._lock:
            batches, self._committed = self._committed, []
        if batches:
            self._advanced = True
        for token_ids in batches:
            # LLGuidance's bulk API rejects EOS after a NoExtension stop.
            stopped_eos = (
                len(token_ids) == 1
                and token_ids[0] in self.eos_tokens
                and not self.matcher.is_error()
                and self.matcher.is_stopped()
                and self.matcher.is_accepting()
            )
            valid = (
                self.matcher.consume_token(token_ids[0])
                if stopped_eos
                else self.matcher.consume_tokens(token_ids)
            )
            if not valid:
                # The lines after the first dump the parser state, output
                # included.
                error = self.matcher.get_error()
                raise ConstraintError(
                    error.splitlines()[0] if error else "invalid token"
                )


def _byte_char_inverse():
    """The byte-level pretokenizer's codepoint-per-byte alphabet, inverted:
    token text maps back to raw bytes."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAD))
        + list(range(0xAE, 0x100))
    )
    mapping = {}
    extra = 0
    for value in range(256):
        codepoint = value if value in printable else 256 + extra
        extra += value not in printable
        mapping[chr(codepoint)] = value
    return mapping


_BYTE_CHAR_INVERSE = _byte_char_inverse()


def _vocab_token_bytes(tokenizer, vocabulary):
    """Every token's raw bytes in token-id order, for a byte-level
    tokenizer; ids whose spelling is unknown get empty (dead) bytes."""
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    id_to_token = getattr(backend, "id_to_token", None)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    decode = getattr(tokenizer, "decode", None) or getattr(backend, "decode", None)

    def text_of(token_id):
        if id_to_token is not None:
            try:
                return id_to_token(token_id)
            except Exception:
                pass
        if convert is not None:
            try:
                return convert(token_id)
            except Exception:
                pass
        return None

    def decoded_bytes(token_id):
        if decode is None:
            return b""
        for kwargs in ({"skip_special_tokens": False}, {}):
            try:
                return decode([token_id], **kwargs).encode("utf-8")
            except (TypeError, Exception):
                continue
        return b""

    out = []
    for token_id in range(vocabulary):
        text = text_of(token_id)
        if text is not None:
            try:
                out.append(bytes(_BYTE_CHAR_INVERSE[char] for char in text))
                continue
            except KeyError:
                pass
        out.append(decoded_bytes(token_id))
    return tuple(out)


# RICHENGINE_DFA_MASK_CACHE opts DfaConstraint.masks into serving rows a
# per-state LRU instead of the delegate matcher. The rows come from
# CompiledDfa.masks_for_state — a pure-Python walk of the whole vocabulary
# that costs far more per miss than the matcher's mask (the reason the
# no-delegate path is documented as too slow to serve traffic) — so the
# flag is a measurement path, not a serving one: the constraint's
# mask_stats() records what hits save and what misses cost.
_MASK_ROW_CACHE_SIZE = 512
# (id(dfa), state) -> (dfa, allowed ids). The stored reference pins the
# compiled DFA, so an id freed and reused can never alias a stale row.
_mask_row_cache = OrderedDict()
_mask_row_cache_lock = threading.Lock()


def _mask_cache_enabled():
    return bool(os.environ.get("RICHENGINE_DFA_MASK_CACHE"))


class DfaConstraint:
    """A TokenConstraint-shaped constraint backed by a compiled byte DFA.

    The mirror state tracks generated tokens for fast-forward pushes exactly
    as TokenConstraint's fast-forward matcher does. `inner`, a
    TokenConstraint over the same grammar, serves masks(), commit
    validation and finish(); without one the constraint walks the DFA's
    token bytes itself — adequate for tests, too slow to serve traffic."""

    MAX_ROWS = TokenConstraint.MAX_ROWS

    def __init__(
        self, dfa, vocab_tokens, special_tokens, contract, encode=None, inner=None
    ):
        self.dfa = dfa
        self._inner = inner
        self.vocab_tokens = vocab_tokens
        self.special_tokens = frozenset(special_tokens)
        self.vocabulary = contract.vocabulary
        self.eos_tokens = contract.eos_tokens
        self._eos = frozenset(contract.eos_tokens)
        self._encode = encode
        self._words = (self.vocabulary + 31) // 32
        # The constraint's consuming state and the reader thread's mirror.
        self._state = 0
        self._mirror = 0
        self._stopped = False
        self._mirror_stopped = False
        self._ff_failed = False
        self._advanced = False
        self._initial_mask = None
        self._lock = threading.Lock()
        self._committed: list[tuple[int, ...]] = []
        # RICHENGINE_DFA_MASK_CACHE counters; see _mask_row_cache above.
        self._mask_timing = {
            "cached_rows": 0,
            "computed_rows": 0,
            "compute_ms_total": 0.0,
            "delegate_fallbacks": 0,
        }

    def _advance(self, state, token_id):
        return self.dfa.advance_token(
            state, token_id, self.vocab_tokens, self._eos, self.special_tokens
        )

    def forced_tokens(self):
        """The tokens the DFA forces at the mirror position, anchored like
        TokenConstraint.forced_tokens: tokens[0] fills the pending slot. A
        span the byte walk does not derive falls back to the delegate
        matcher's fast-forward tokens."""
        tokens = ()
        if (
            not self._ff_failed
            and not self._mirror_stopped
            and self._encode
            and not os.environ.get("RICHENGINE_DISABLE_FORCED_TOKENS")
        ):
            try:
                tokens = self.dfa.forced_tokens_after(
                    self._mirror,
                    self.vocab_tokens,
                    self._encode,
                    self._eos,
                    self.special_tokens,
                )
            except Exception:
                self._ff_failed = True
        if not tokens and self._inner is not None:
            return self._inner.forced_tokens()
        return tokens

    def commit(self, token_ids, *, forced=True):
        """Consume generated tokens and return the forced span at the mirror
        position, like TokenConstraint.commit. `forced=False` — a GPU-DFA
        mode whose engine derives the span itself — skips the mirror's
        encode-and-validate answer while the delegate and the mirror still
        advance."""
        if any(not 0 <= token < self.vocabulary for token in token_ids):
            raise ConstraintError("generated token is out of range")
        # The delegate consumes the same tokens for its mask bookkeeping;
        # its own forced-token answer is superseded by the DFA mirror's.
        if self._inner is not None:
            self._inner.commit(token_ids, forced=forced)
        with self._lock:
            if self._inner is None or _mask_cache_enabled():
                self._committed.append(tuple(token_ids))
            if not self._ff_failed:
                try:
                    self._advance_mirror(token_ids)
                except Exception:
                    self._ff_failed = True
            return self.forced_tokens() if forced else ()

    def _foreign_stop(self, state, token_id):
        """A special token the DFA cannot consume at an EOS-admitting state:
        the engine's own stop set can drift from the grammar's eos list
        (engine stop spellings are always special tokens), so the mirror
        maps the token to the stop state rather than erroring."""
        if (
            token_id in self.special_tokens
            and state < self.dfa.n_states
            and (self.dfa.is_accepting(state) or self.dfa._eos_ok(state))
        ):
            return self.dfa.n_states
        return -1

    def _advance_mirror(self, token_ids):
        for token_id in token_ids:
            if self._mirror_stopped:
                raise ConstraintError("token after end of output")
            state = self._advance(self._mirror, token_id)
            if state < 0:
                state = self._foreign_stop(self._mirror, token_id)
            if state < 0:
                raise ConstraintError("invalid token")
            if state == self.dfa.n_states:
                self._mirror_stopped = True
            else:
                self._mirror = state

    def finish(self):
        """Checks the tokens generated after the last mask reached an
        accepting state."""
        if self._inner is not None:
            self._inner.finish()
            return
        self._consume_committed()
        if not self._stopped and not self.dfa.is_accepting(self._state):
            raise ConstraintError("output grammar did not reach an accepting state")

    def masks(self, simulation_tokens):
        if self._inner is not None:
            if _mask_cache_enabled():
                return self._cached_masks(simulation_tokens)
            return self._inner.masks(simulation_tokens)
        self._consume_committed()
        if (
            not simulation_tokens
            and not self._advanced
            and self._initial_mask is not None
        ):
            return self._initial_mask
        if len(simulation_tokens) >= self.MAX_ROWS:
            raise ConstraintError("too many simulation tokens")
        rows = len(simulation_tokens) + 1
        words = self._words
        mask = array.array("I", [0] * (rows * words))
        # Rows before and after each simulated token; simulation stops at
        # the first invalid token and its row repeats for the rest.
        states = [self._state]
        cursor = self._state
        for token_id in simulation_tokens:
            if not 0 <= token_id < self.vocabulary:
                break
            cursor = self._advance(cursor, token_id)
            if cursor < 0:
                break
            states.append(cursor)
        for row in range(len(states)):
            state = states[row]
            allowed = (
                self.dfa.masks_for_state(
                    state, self.vocab_tokens, self._eos, self.special_tokens
                )
                if state < self.dfa.n_states
                else []
            )
            if not allowed:
                raise ConstraintError("output grammar has no valid token")
            base = row * words
            for token_id in allowed:
                mask[base + (token_id >> 5)] |= 1 << (token_id & 31)
        if len(states) < rows:
            mask[len(states) * words :] = mask[(len(states) - 1) * words : len(states) * words]
        return mask.tobytes()

    def _cached_masks(self, simulation_tokens):
        """The RICHENGINE_DFA_MASK_CACHE path: the same rows the no-delegate
        branch computes, through the per-state LRU, with the delegate's mask
        as the answer for a state the DFA walk cannot cover."""
        if len(simulation_tokens) >= self.MAX_ROWS:
            raise ConstraintError("too many simulation tokens")
        self._consume_committed()
        rows = len(simulation_tokens) + 1
        words = self._words
        mask = array.array("I", [0] * (rows * words))
        states = [self._state]
        cursor = self._state
        for token_id in simulation_tokens:
            if not 0 <= token_id < self.vocabulary:
                break
            cursor = self._advance(cursor, token_id)
            if cursor < 0:
                break
            states.append(cursor)
        for row in range(len(states)):
            allowed = self._mask_row(states[row])
            if not allowed:
                # A state the DFA has no row for — a stopped request, or an
                # empty row — is the delegate's question to answer anyway.
                self._mask_timing["delegate_fallbacks"] += 1
                return self._inner.masks(simulation_tokens)
            base = row * words
            for token_id in allowed:
                mask[base + (token_id >> 5)] |= 1 << (token_id & 31)
        if len(states) < rows:
            mask[len(states) * words :] = mask[(len(states) - 1) * words : len(states) * words]
        return mask.tobytes()

    def _mask_row(self, state):
        """One state's sorted allowed-token ids, LRU-cached per compiled
        DFA; None for the stopped state the table has no row for."""
        if state >= self.dfa.n_states:
            return None
        key = (id(self.dfa), state)
        with _mask_row_cache_lock:
            entry = _mask_row_cache.get(key)
            if entry is not None:
                _mask_row_cache.move_to_end(key)
        if entry is not None:
            self._mask_timing["cached_rows"] += 1
            return entry[1]
        started = time.perf_counter()
        allowed = tuple(
            self.dfa.masks_for_state(
                state, self.vocab_tokens, self._eos, self.special_tokens
            )
        )
        timing = self._mask_timing
        timing["computed_rows"] += 1
        timing["compute_ms_total"] += (time.perf_counter() - started) * 1000.0
        with _mask_row_cache_lock:
            _mask_row_cache[key] = (self.dfa, allowed)
            _mask_row_cache.move_to_end(key)
            while len(_mask_row_cache) > _MASK_ROW_CACHE_SIZE:
                _mask_row_cache.popitem(last=False)
        return allowed

    def mask_stats(self):
        """A copy of the RICHENGINE_DFA_MASK_CACHE counters."""
        return dict(self._mask_timing)

    def _consume_committed(self):
        with self._lock:
            batches, self._committed = self._committed, []
        if batches:
            self._advanced = True
        for token_ids in batches:
            for token_id in token_ids:
                if self._stopped:
                    raise ConstraintError("token after end of output")
                state = self._advance(self._state, token_id)
                if state < 0:
                    state = self._foreign_stop(self._state, token_id)
                if state < 0:
                    raise ConstraintError("invalid token")
                if state == self.dfa.n_states:
                    self._stopped = True
                else:
                    self._state = state


def validate_tokenizer(tokenizer, config, serving=None):
    """Checks the loaded tokenizer against the loaded model's configuration
    and returns the ids generation is bound to, a TokenizerContract.

    config is the configuration mapping itself, or the directory holding it:
    the tokenizer files an assembly links sit beside a copy of the model's
    config.json. serving is the model's ServingContract: its think framing
    and tool dialect name the spellings to bind; without one (an old
    assembly) the spellings are probed, as they always were."""
    generation_config = None
    if not isinstance(config, dict):
        config_dir = config
        try:
            config = json.loads(Path(config_dir, "config.json").read_bytes())
        except (OSError, ValueError, TypeError) as error:
            raise engine_runtime.EngineUnhealthy(
                f"model configuration is unreadable: {error}"
            ) from None
        if not isinstance(config, dict):
            raise engine_runtime.EngineUnhealthy("model configuration is not an object")
        # The generation configuration's own stop ids — a family whose
        # serving configuration stops on more than the model card's, as
        # Gemma 4 does on its tool-response token — count too.
        try:
            generation_config = json.loads(
                Path(config_dir, "generation_config.json").read_bytes()
            )
        except (OSError, ValueError, TypeError):
            generation_config = None
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = config
    vocabulary_size = text_config.get("vocab_size")
    if type(vocabulary_size) is not int or vocabulary_size <= 0:
        raise engine_runtime.EngineUnhealthy(
            "model configuration states no vocabulary size"
        )
    # Stop ids live in both the top-level (generation) configuration and the
    # text configuration; every one the model states stops generation.
    eos_ids = set()
    for source in (config, text_config, generation_config or {}):
        eos = source.get("eos_token_id")
        if type(eos) is list:
            eos_ids.update(eos)
        elif eos is not None:
            eos_ids.add(eos)
    if not eos_ids or any(
        type(token_id) is not int or not 0 <= token_id < vocabulary_size
        for token_id in eos_ids
    ):
        raise engine_runtime.EngineUnhealthy(
            "model configuration states no valid EOS token"
        )
    eos_tokens = tuple(sorted(eos_ids))
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or any(
        type(token_id) is not int or not 0 <= token_id < vocabulary_size
        for token_id in vocabulary.values()
    ):
        raise engine_runtime.EngineUnhealthy(
            "tokenizer vocabulary does not fit the native model"
        )
    by_id = {token_id: token for token, token_id in vocabulary.items()}
    if len(by_id) != len(vocabulary):
        raise engine_runtime.EngineUnhealthy(
            "tokenizer assigns a native token id to several tokens"
        )
    # The think framing is the contract's where the model declares one;
    # otherwise the family's whose close token the vocabulary has. The
    # open's leading special token decodes for the output parser.
    if serving is not None and vocabulary.get(serving.think.end) is not None:
        framing = (
            serving.think.end,
            serving.think.open,
            serving.think.open_token,
            serving.think.flag,
        )
    else:
        framing = next(
            (item for item in THINK_FRAMINGS if vocabulary.get(item[0]) is not None),
            None,
        )
    think_end, think_open, open_token, think_flag = (
        framing if framing is not None else (THINK_END, THINK_OPEN, None, None)
    )
    think_end_id = vocabulary.get(think_end)
    think_open_id = vocabulary.get(open_token) if open_token is not None else None
    # The call-open token is the contract dialect's marker first, then the
    # other spellings served families use.
    declared = None
    if serving is not None:
        dialect = dialect_named(serving.tool_dialect)
        if dialect is not None:
            # The call-open's leading special token where the dialect marks
            # one; the bare spelling otherwise.
            declared = (
                dialect.structural[0] if dialect.structural else dialect.call_open
            )
    call_tokens = tuple(
        dict.fromkeys(
            token
            for token in (declared, TOOL_CALL_OPEN, TOOL_CALL_OPEN_ALT)
            if token is not None
        )
    )
    tool_call_open = next(
        (
            token
            for token in call_tokens
            if vocabulary.get(token) is not None
        ),
        None,
    )
    tool_call_open_id = (
        vocabulary.get(tool_call_open) if tool_call_open is not None else None
    )
    expected = [(by_id.get(token_id), token_id) for token_id in eos_tokens]
    expected.append((think_end, think_end_id))
    if think_open_id is not None:
        expected.append((open_token, think_open_id))
    if tool_call_open is not None:
        expected.append((tool_call_open, tool_call_open_id))
    for token, token_id in expected:
        if token is None or token_id is None:
            raise engine_runtime.EngineUnhealthy(
                f"tokenizer has no token for native token {token_id}"
            )
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise engine_runtime.EngineUnhealthy(
                f"tokenizer must encode {token!r} as native token {token_id}"
            )
    if tokenizer.eos_token_id not in eos_tokens:
        raise engine_runtime.EngineUnhealthy(
            "tokenizer EOS token does not match the native model"
        )
    marker = by_id.get(tokenizer.eos_token_id)
    if marker is None:
        raise engine_runtime.EngineUnhealthy(
            "tokenizer EOS token is not in its vocabulary"
        )
    return TokenizerContract(
        vocabulary_size,
        eos_tokens,
        marker,
        think_end_id,
        tool_call_open_id,
        think_open_id,
        think_open,
        think_end,
        think_flag,
    )


def _grammar_error(error):
    # A compiler panic carries a backtrace rather than a reason, and the lines
    # after the first echo the grammar source with every schema it holds.
    if error.startswith("panic"):
        return APIError(400, "tool or output schema is too large to compile")
    return APIError(400, f"unsupported output schema: {error.splitlines()[0]}")


class ConstraintFactory:
    CACHE_SIZE = 32
    CACHE_SOURCE_BYTES = 8 * 1024 * 1024

    def __init__(self, tokenizer, contract):
        self.contract = contract
        self.hf_tokenizer = tokenizer
        self.tokenizer = guidance_tokenizer(
            tokenizer,
            n_vocab=contract.vocabulary,
            eos_token=list(contract.eos_tokens),
            slices=LLTokenizer.json_slices(),
        )
        self.executor = LLExecutor()
        self.source_bytes = 0
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.pending = {}
        self.hits = 0
        self.misses = 0
        # Compiled DFAs keyed by the grammar string (a deterministic product
        # of the spec); False marks a spec outside the DFA-compilable subset
        # so fallback requests do not pay the compiler each time.
        self._dfa_cache = OrderedDict()
        # The vocabulary's token bytes and special-token ids, for DFA
        # constraints; computed once, on first use.
        self._vocab = None

    def _vocabulary(self):
        if self._vocab is None:
            vocab_tokens = _vocab_token_bytes(
                self.hf_tokenizer, self.contract.vocabulary
            )
            special = frozenset(
                token_id
                for token_id in range(self.contract.vocabulary)
                if self.tokenizer.is_special_token(token_id)
            )
            self._vocab = vocab_tokens, special
        return self._vocab

    def create_dfa(self, spec, grammar, *, timeout=None, prefixes=None):
        """A DFA constraint for `spec`, or None where the spec is outside the
        DFA-compilable subset — the caller then takes the grammar path.

        The compiled DFA rides to the engine as a CONSTRAINT_DFA sidecar
        for on-device forced spans (and, under RICHENGINE_DFA_MODE=full,
        masks); the constraint itself serves masks(), commits and the
        finishing check through a TokenConstraint on `grammar`, the same
        grammar the llguidance path would compile."""
        if os.environ.get("RICHENGINE_DISABLE_DFA") or dfa_mode() == "off":
            return None
        with self.lock:
            cached = self._dfa_cache.get(grammar)
            if cached is not None:
                self._dfa_cache.move_to_end(grammar)
        if cached is None:
            vocab_tokens, special = self._vocabulary()
            markers = tuple(
                bytes
                for token_id, bytes in enumerate(vocab_tokens)
                if token_id in special and bytes
            )
            compiled = dfa_compiler.compile_dfa(spec, markers)
            with self.lock:
                existing = self._dfa_cache.setdefault(grammar, compiled or False)
                self._dfa_cache.move_to_end(grammar)
                while len(self._dfa_cache) > self.CACHE_SIZE:
                    self._dfa_cache.popitem(last=False)
            cached = existing
        if cached is False:
            return None
        compiled = cached
        vocab_tokens, special = self._vocabulary()
        try:
            matcher, initial_mask = self._matcher(grammar, timeout, prefixes)
        except Exception:
            # No DFA ships without its delegate: the grammar path the
            # caller falls back to compiles it and reports the real error.
            return None
        for tokens, _message in prefixes() if prefixes else ():
            state = 0
            for token_id in tokens:
                state = compiled.advance_token(
                    state, token_id, vocab_tokens, self.contract.eos_tokens, special
                )
                if state < 0:
                    break
            if state < 0:
                # The delegate grammar accepted this opening, so the DFA is
                # the stricter read; the caller's grammar fallback validates
                # the real capability rather than refusing on a mismatch.
                return None
        encode = getattr(self.hf_tokenizer, "encode", None)
        return DfaConstraint(
            compiled,
            vocab_tokens,
            special,
            self.contract,
            encode=encode,
            inner=TokenConstraint(
                matcher.deep_copy(),
                self.executor,
                self.contract,
                initial_mask,
                encode=encode,
            ),
        )

    def create(self, grammar, *, timeout=None, prefixes=None):
        """`prefixes`, called when the grammar is compiled, returns pairs of
        tokens the output must be able to begin with and the error for a
        grammar that cannot. A grammar can compile and still exceed the
        parser's limits where generation reaches a construct, after the whole
        prompt has been processed."""
        matcher, initial_mask = self._matcher(grammar, timeout, prefixes)
        return TokenConstraint(
            matcher.deep_copy(),
            self.executor,
            self.contract,
            initial_mask,
            encode=getattr(self.hf_tokenizer, "encode", None),
        )

    def _compile_initial_mask(self, matcher):
        """The start-state next-token mask, identical for every request a
        grammar serves: computed once, at compile time and off the serving
        path, where the engine's initial mask request used to wait for it."""
        if not self.contract.vocabulary:
            return None
        bitmask = allocate_token_bitmask(1, self.contract.vocabulary)
        fill_next_token_bitmask_par(
            self.executor, [(matcher.deep_copy(), 0)], bitmask
        )
        return bitmask[:1].tobytes()

    def _matcher(self, grammar, timeout, prefixes):
        # Compilation uses the frontend's bounded preparation slots. Share
        # identical misses without blocking unrelated immutable templates.
        with self.lock:
            cached = self.cache.get(grammar)
            if cached is not None:
                self.cache.move_to_end(grammar)
                self.hits += 1
                return cached[0], cached[2]
            pending = self.pending.get(grammar)
            owner = pending is None
            if owner:
                pending = self.pending[grammar] = Future()
        if not owner:
            if not wait((pending,), timeout=timeout).done:
                raise APIError(504, "request timed out", "request_timeout")
            matcher, initial_mask = pending.result()
            with self.lock:
                if grammar in self.cache:
                    self.cache.move_to_end(grammar)
                self.hits += 1
            return matcher, initial_mask
        try:
            error = LLMatcher.validate_grammar(grammar, self.tokenizer)
            if error:
                raise _grammar_error(error)
            matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
            if matcher.is_error():
                raise _grammar_error(matcher.get_error())
            for tokens, message in prefixes() if prefixes else ():
                if matcher.deep_copy().consume_tokens(tokens):
                    continue
                # Name the token the grammar refuses, so the caller's
                # generic message says where consumption stopped.
                probe = matcher.deep_copy()
                bad = next(
                    (
                        token_id
                        for token_id in tokens
                        if not probe.consume_tokens([token_id])
                    ),
                    None,
                )
                detail = ""
                if bad is not None:
                    spelling = self._vocabulary()[0][bad]
                    detail = (
                        f" at token {bad} {spelling!r}"
                        if spelling
                        else f" at token {bad}"
                    )
                raise APIError(400, f"{message}{detail}")
            initial_mask = self._compile_initial_mask(matcher)
            size = len(grammar.encode())
            # Oversized grammars remain usable without displacing the cache.
            # This bounds source bytes; LLGuidance bounds compiler complexity.
            with self.lock:
                self.misses += 1
                if size <= self.CACHE_SOURCE_BYTES:
                    self.cache[grammar] = (matcher, size, initial_mask)
                    self.source_bytes += size
                    while (
                        len(self.cache) > self.CACHE_SIZE
                        or self.source_bytes > self.CACHE_SOURCE_BYTES
                    ):
                        _, (_, evicted_size, _) = self.cache.popitem(last=False)
                        self.source_bytes -= evicted_size
            pending.set_result((matcher, initial_mask))
            return matcher, initial_mask
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self.lock:
                del self.pending[grammar]

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.cache),
                "capacity": self.CACHE_SIZE,
                "source_bytes": self.source_bytes,
                "source_budget_bytes": self.CACHE_SOURCE_BYTES,
                "hits": self.hits,
                "misses": self.misses,
            }

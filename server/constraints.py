"""Tokenizer contract and cached token-level output constraints."""

import json
import threading
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

from . import runtime as engine_runtime
from .errors import APIError, ConstraintError
from .tool_schema import THINK_END, TOOL_CALL_OPEN


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


class TokenConstraint:
    # A mask request simulates at most the engine's target verify rows
    # (ExecutionLimits::targetVerifyRows, 8: the pending anchor and seven
    # draft proposals) and takes a mask before and after each.
    MAX_ROWS = 9

    def __init__(self, matcher, executor, contract):
        self.matcher = matcher
        self.executor = executor
        self.vocabulary = contract.vocabulary
        self.eos_tokens = contract.eos_tokens
        self.bitmask = allocate_token_bitmask(self.MAX_ROWS, self.vocabulary)
        # Generated batches the grammar has not consumed yet. The reader
        # thread commits them; the mask thread consumes them before the next
        # mask, so reading the native stream never waits for the grammar.
        self._lock = threading.Lock()
        self._committed: list[tuple[int, ...]] = []

    def commit(self, token_ids):
        if any(not 0 <= token < self.vocabulary for token in token_ids):
            raise ConstraintError("generated token is out of range")
        with self._lock:
            self._committed.append(tuple(token_ids))

    def finish(self):
        """Checks the tokens generated after the last mask."""
        self._consume_committed()

    def masks(self, simulation_tokens):
        self._consume_committed()
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


def validate_tokenizer(tokenizer, config):
    """Checks the loaded tokenizer against the loaded model's configuration
    and returns the ids generation is bound to, a TokenizerContract.

    config is the configuration mapping itself, or the directory holding it:
    the tokenizer files an assembly links sit beside a copy of the model's
    config.json."""
    if not isinstance(config, dict):
        try:
            config = json.loads(Path(config, "config.json").read_bytes())
        except (OSError, ValueError, TypeError) as error:
            raise engine_runtime.EngineUnhealthy(
                f"model configuration is unreadable: {error}"
            ) from None
        if not isinstance(config, dict):
            raise engine_runtime.EngineUnhealthy(
                "model configuration is not an object"
            )
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
    for source in (config, text_config):
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
    think_end_id = vocabulary.get(THINK_END)
    tool_call_open_id = vocabulary.get(TOOL_CALL_OPEN)
    expected = [(by_id.get(token_id), token_id) for token_id in eos_tokens]
    expected.append((THINK_END, think_end_id))
    if tool_call_open_id is not None:
        expected.append((TOOL_CALL_OPEN, tool_call_open_id))
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
        vocabulary_size, eos_tokens, marker, think_end_id, tool_call_open_id
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

    def create(self, grammar, *, timeout=None, prefixes=None):
        """`prefixes`, called when the grammar is compiled, returns pairs of
        tokens the output must be able to begin with and the error for a
        grammar that cannot. A grammar can compile and still exceed the
        parser's limits where generation reaches a construct, after the whole
        prompt has been processed."""
        matcher = self._matcher(grammar, timeout, prefixes)
        return TokenConstraint(matcher.deep_copy(), self.executor, self.contract)

    def _matcher(self, grammar, timeout, prefixes):
        # Compilation uses the frontend's bounded preparation slots. Share
        # identical misses without blocking unrelated immutable templates.
        with self.lock:
            cached = self.cache.get(grammar)
            if cached is not None:
                self.cache.move_to_end(grammar)
                self.hits += 1
                return cached[0]
            pending = self.pending.get(grammar)
            owner = pending is None
            if owner:
                pending = self.pending[grammar] = Future()
        if not owner:
            if not wait((pending,), timeout=timeout).done:
                raise APIError(504, "request timed out", "request_timeout")
            matcher = pending.result()
            with self.lock:
                if grammar in self.cache:
                    self.cache.move_to_end(grammar)
                self.hits += 1
            return matcher
        try:
            error = LLMatcher.validate_grammar(grammar, self.tokenizer)
            if error:
                raise _grammar_error(error)
            matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
            if matcher.is_error():
                raise _grammar_error(matcher.get_error())
            for tokens, message in prefixes() if prefixes else ():
                if not matcher.deep_copy().consume_tokens(tokens):
                    raise APIError(400, message)
            size = len(grammar.encode())
            # Oversized grammars remain usable without displacing the cache.
            # This bounds source bytes; LLGuidance bounds compiler complexity.
            with self.lock:
                self.misses += 1
                if size <= self.CACHE_SOURCE_BYTES:
                    self.cache[grammar] = (matcher, size)
                    self.source_bytes += size
                    while (
                        len(self.cache) > self.CACHE_SIZE
                        or self.source_bytes > self.CACHE_SOURCE_BYTES
                    ):
                        _, (_, evicted_size) = self.cache.popitem(last=False)
                        self.source_bytes -= evicted_size
            pending.set_result(matcher)
            return matcher
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

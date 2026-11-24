"""Prompt tokenization: a rendered prompt's tokens, in which only the text
the chat template wrote reads as a token the tokenizer adds, and bounded token
reuse at literal chat-message boundaries."""

import json
import os
import re
import secrets
import sys
import threading
from array import array
from collections import OrderedDict

from tokenizers import Tokenizer

# Emitted-id reply memory (RICHENGINE_NO_REPLY_ID_MEMORY=1 disables): a reply
# the model emitted re-encodes differently when a client resends it as
# history, which silently misses the prefix cache. Remembering the emitted
# ids lets prompt tokenization splice them back instead.
_REPLY_MEMORY_DISABLED = "RICHENGINE_NO_REPLY_ID_MEMORY"


def _reply_memory_disabled():
    return os.environ.get(_REPLY_MEMORY_DISABLED) == "1"


# Reply texts index on their last _REPLY_SUFFIX chars; _reply_at probes at
# most that many suffix slices per turn boundary.
_REPLY_SUFFIX = 24

# A surrogate code point. Request text keeps a lone one that JSON spells, such
# as "\ud800"; it encodes no token, so each reads as U+FFFD first.
_SURROGATE = re.compile(r"[\ud800-\udfff]")


def tokenizable(text):
    """`text` with each lone surrogate as U+FFFD, which it encodes as."""
    return _SURROGATE.sub("\ufffd", text)


# The mark PromptTokenizer.quote sets after the first character of an added
# token that begins in what a client sent: a text no request can hold, as its
# 128 random bits are drawn when the server starts, and of characters that a
# template's JSON or trimming of the text leaves as they are.
QUOTE = f"__richengine_quoted_{secrets.token_hex(16)}__"


def unquoted(text):
    """`text` without its quote marks: the text the model reads."""
    return text.replace(QUOTE, "")


def added_tokens(tokenizer):
    """The tokens `tokenizer` adds beside its model's vocabulary, by their
    text: a chat template's turn and role tokens, its tags, an image's pad.
    It splits its input at each before anything else reads the text, and
    PromptTokenizer keeps that split from the text a client sent. A
    ValueError names what it cannot keep it from: a tokenizer without a fast
    backend, an added token that takes the spaces beside it or stands only
    as a word, which its literal text does not find, or that QUOTE cannot
    go inside, and a pipeline where the text between two added tokens does
    not encode on its own. A normalized one, as add_tokens adds them, is
    found in the normalized text, its literal text among others
    (PromptTokenizer.literal). A tokenizer that adds none splits nothing."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("tokenizer must be a fast tokenizer")
    tokens = {}
    for token_id, token in backend.get_added_tokens_decoder().items():
        text = token.content
        if token.lstrip or token.rstrip or token.single_word:
            raise ValueError(
                f"tokenizer must find the added token {text!r} as its literal text"
            )
        # The mark follows the token's first character, and its own text
        # must not complete a token with the characters around it.
        if (
            len(text) < 2
            or text in QUOTE
            or any(
                QUOTE.startswith(text[-length:]) or QUOTE.endswith(text[:length])
                for length in range(1, len(text))
            )
        ):
            raise ValueError(f"tokenizer adds a token that cannot be quoted: {text!r}")
        tokens[text] = token_id
    if tokens and not _encodes_segments_alone(backend):
        raise ValueError(
            "tokenizer must encode the text between two added tokens on its own"
        )
    return tokens


def _encodes_segments_alone(backend):
    """Whether the text between two literal added tokens encodes as it does
    alone: byte-level BPE after regex splits, Qwen's one or Flint's three,
    each of which reads a segment on its own, and no normalizer or NFC.
    Another pipeline may not: for example, Metaspace's prepend_scheme="first"
    depends on whether a segment starts the complete input, not just a
    split. A post-processor has no part in it: prompts encode without the
    tokenizer's own special tokens, and it adds no other."""
    if type(backend.model).__name__ != "BPE" or backend.model.dropout:
        return False
    if backend.pre_tokenizer is None:
        return False
    pre = json.loads(backend.pre_tokenizer.__getstate__())
    parts = pre["pretokenizers"] if pre.get("type") == "Sequence" else [pre]
    if [part.get("type") for part in parts] != ["Split"] * (len(parts) - 1) + [
        "ByteLevel"
    ] or parts[-1].get("add_prefix_space"):
        return False
    return (
        backend.normalizer is None
        or json.loads(backend.normalizer.__getstate__()).get("type") == "NFC"
    )


def _rest_pattern(rests):
    """A pattern for what follows an added token's first character where the
    token begins in a text: the rest of one, each of `rests`, or as much of
    a rest as the text holds where it ends first, which what follows the
    text in a prompt could complete. It reads the characters that rests
    share once: 1 MB of markup takes 1.7 ms with Flint's 144 tokens, where
    trying each rest in turn took 23 ms."""
    shared = {}
    for rest in rests:
        node = shared
        for character in rest:
            node = node.setdefault(character, {})
        # A token ends here: nothing more to read.
        node[""] = {}

    def pattern(node):
        if "" in node:
            return ""
        following = "|".join(
            re.escape(character) + pattern(after) for character, after in node.items()
        )
        return rf"(?:\Z|{following})"

    return pattern(shared)


class PromptTokenizer:
    """Encodes rendered prompts so that only the chat template's own text
    reads as a token the tokenizer adds (added_tokens). A tokenizer finds an
    added token wherever the text spells it, so one that a message only
    quotes, as a tool result showing a parser's `tool_call_start =
    "<tool_call>"` does, would reach the model as the control token, and a
    `<|im_end|>` in a document as the end of its turn. `quote` marks each
    where it begins in what a client sent, before the template renders it;
    `encode` reads the template's text between two of the tokens the template
    wrote as ordinary text, a marked token as its characters.

    It reuses a cached prompt's tokens up to a message end: a `marker`, the
    chat turn's end token, where the tokenizer splits its input. A prompt
    that extends a cached prompt reuses it whole; any other prompt reuses
    the longest run of whole messages it shares with a cached prompt, which
    keeps where each of its markers ends. So a branch of a conversation, or
    a return to an earlier turn, reuses the messages they share. It keeps up
    to CAPACITY prompts, in storage for BUDGET_BYTES, and splices the
    remembered ids of replies the prompt resends."""

    MIN_PREFIX_CHARS = 4096
    BUDGET_BYTES = 8 * 1024 * 1024
    # One entry per live conversation prefix, and each branch of one its own;
    # BUDGET_BYTES is the real memory bound, so this only needs to outnumber
    # concurrent sessions.
    CAPACITY = 32
    REPLY_CAPACITY = 256

    def __init__(self, tokenizer, marker=None):
        self.tokenizer = tokenizer
        # The token that ends a chat turn; the tokenizer's EOS by default, the
        # validated contract's turn-end token where one is supplied.
        self.marker = (
            getattr(tokenizer, "eos_token", None) if marker is None else marker
        )
        self.enabled = isinstance(self.marker, str) and self._supports_boundaries(
            tokenizer
        )
        self.marker_id = (
            tokenizer.backend_tokenizer.token_to_id(self.marker)
            if self.enabled
            else None
        )
        # The quote machinery. A tokenizer added_tokens cannot serve keeps
        # every text literal: validate_tokenizer refuses it before serving,
        # so an empty map here only stands where there is no contract to
        # hold (tests, probes).
        try:
            self.added = added_tokens(tokenizer)
        except ValueError:
            self.added = {}
        backend = getattr(tokenizer, "backend_tokenizer", None)
        # Whether the tokenizer finds its added tokens as their literal text
        # alone. A normalized one, as add_tokens adds them, it finds in the
        # normalized text, where NFC makes another spelling of it the token
        # too, as U+212A for a K, which no mark breaks: such a tokenizer's
        # prompts all encode as the text between the template's tokens.
        self.literal = backend is not None and (
            backend.normalizer is None
            or not any(
                token.normalized
                for token in backend.get_added_tokens_decoder().values()
            )
        )
        # Where an added token begins in a text, to mark it there: for each
        # first character of one, that character before the rest of any, or
        # before the text's end where that comes first (_rest_pattern).
        rests = {}
        for text in self.added:
            rests.setdefault(text[0], []).append(text[1:])
        self._begins = [
            (first, re.compile(f"{re.escape(first)}(?={_rest_pattern(rest)})"))
            for first, rest in rests.items()
        ]
        if self.added:
            # Each whole one, the longest first as the tokenizer finds them.
            self._tokens = re.compile(
                "("
                + "|".join(map(re.escape, sorted(self.added, key=len, reverse=True)))
                + ")"
            )
            # The tokenizer's own normalizer, pre-tokenizer and model, which
            # it shares, without its added tokens: how it encodes ordinary
            # text.
            self._plain = Tokenizer(backend.model)
            if backend.normalizer is not None:
                self._plain.normalizer = backend.normalizer
            self._plain.pre_tokenizer = backend.pre_tokenizer
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = self.reused_tokens = 0
        # Reply text -> the ids the model emitted for it, a small LRU.
        self.replies = OrderedDict()
        # Reply text suffix -> the reply texts carrying it, so _reply_at
        # checks a handful of buckets per boundary instead of every reply.
        self.reply_buckets = {}
        self.reply_hits = 0
        self.lock = threading.Lock()

    def _supports_boundaries(self, tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not getattr(tokenizer, "is_fast", False) or backend is None:
            return False
        # Literal, non-stripping added tokens split BPE input before ordinary
        # normalization/pretokenization. Do not assume this for other pipelines.
        if type(backend.model).__name__ != "BPE" or backend.model.dropout:
            return False
        if backend.pre_tokenizer is None:
            return False
        pre = json.loads(backend.pre_tokenizer.__getstate__())
        parts = pre["pretokenizers"] if pre.get("type") == "Sequence" else [pre]
        if [part.get("type") for part in parts] not in (
            ["ByteLevel"],
            ["Split", "ByteLevel"],
        ) or parts[-1].get("add_prefix_space"):
            # For example, Metaspace's prepend_scheme="first" depends on
            # whether a segment starts the complete input, not just a split.
            return False
        for component, allowed in (
            (backend.normalizer, {"NFC"}),
            (backend.post_processor, {"ByteLevel"}),
        ):
            if (
                component is not None
                and json.loads(component.__getstate__()).get("type") not in allowed
            ):
                return False
        markers = []
        for token in backend.get_added_tokens_decoder().values():
            if self.marker not in token.content:
                continue
            if token.content != self.marker:
                return False
            markers.append(token)
        return len(markers) == 1 and not any(
            (
                markers[0].normalized,
                markers[0].lstrip,
                markers[0].rstrip,
                markers[0].single_word,
            )
        )

    @staticmethod
    def _marked(begun):
        """The first character of a token that begins, with the mark after
        it. A function, as `sub` looks a replacement text up by a hash of
        its whole pattern at every call: 2 to 4 microseconds for these."""
        return begun[0] + QUOTE

    def quote(self, value):
        """`value`, what a request gives its chat template (messages, tool
        definitions, template variables), with QUOTE inside every added
        token that a string or a key of it begins: one it spells whole, and
        one its end cuts off, which what the template writes next could
        complete, as the `>` Qwen's templates write after a call's name
        would, or the next of a message's text parts. A U+0000 is U+FFFD
        first: a text that holds one inside a token spells no token to mark,
        and a template that deletes the character, as Flint's does, would
        write the token whole. The template renders the marked text as it
        renders any other, and no longer finds a tag there itself."""
        if isinstance(value, str):
            if "\x00" in value:
                value = value.replace("\x00", "\ufffd")
            for first, begins in self._begins:
                # Few strings hold the character at all: looking for it
                # first quotes a 66K-token conversation of 2,600 strings in
                # 0.24 ms, where searching each took 0.40 ms.
                if first in value:
                    value = begins.sub(self._marked, value)
            return value
        if isinstance(value, dict):
            return {self.quote(key): self.quote(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.quote(item) for item in value]
        return value

    def tokens(self, text):
        """The tokens of `text`, a rendered prompt or a part of one that
        encodes on its own. Without a mark it encodes as the tokenizer
        encodes it, where that finds its added tokens as their literal text
        alone (literal). Otherwise each added token it still spells whole is
        the template's, and the text between two of them, its marks taken
        out, encodes as ordinary text: what the tokenizer does with a text
        it splits at those tokens alone."""
        text = tokenizable(text)
        if not self.added or (QUOTE not in text and self.literal):
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        parts = self._tokens.split(text)
        plain = self._plain.encode_batch(
            [unquoted(part) for part in parts[::2]], add_special_tokens=False
        )
        tokens = list(plain[0].ids)
        for token, following in zip(parts[1::2], plain[1:]):
            tokens.append(self.added[token])
            tokens.extend(following.ids)
        return tokens

    def record_reply(self, text, token_ids):
        """Remember `token_ids` as the ids the model emitted for the reply
        text `text`, which a later conversation resends inside an assistant
        turn. The stored text is marked as `quote` marks the resent one, so
        the match lands on the rendered prompt's quoted spelling. The
        turn-end token the emission ended on is not part of the reply, so
        one trailing marker id is dropped."""
        if (
            not self.enabled
            or _reply_memory_disabled()
            or not isinstance(text, str)
            or not text
            or self.marker in text
        ):
            return
        ids = [int(token) for token in token_ids]
        if ids and ids[-1] == self.marker_id:
            ids.pop()
        if not ids:
            return
        text = self.quote(text)
        with self.lock:
            self._unindex_reply(text)
            self.replies.pop(text, None)
            self.replies[text] = ids
            self._index_reply(text)
            while len(self.replies) > self.REPLY_CAPACITY:
                evicted, _ = self.replies.popitem(last=False)
                self._unindex_reply(evicted)

    @staticmethod
    def _reply_key(text):
        return text[-_REPLY_SUFFIX:]

    def _index_reply(self, text):
        self.reply_buckets.setdefault(self._reply_key(text), set()).add(text)

    def _unindex_reply(self, text):
        key = self._reply_key(text)
        bucket = self.reply_buckets.get(key)
        if bucket is None:
            return
        bucket.discard(text)
        if not bucket:
            del self.reply_buckets[key]

    def _replies_active(self):
        if _reply_memory_disabled():
            return False
        with self.lock:
            return bool(self.replies)

    def _reply_at(self, text, boundary):
        """The remembered reply whose text ends at `boundary` in `text`, as
        (ids, start) with the longest text winning; None without a match."""
        with self.lock:
            # A reply ending at `boundary` keys on its last _REPLY_SUFFIX
            # chars — or its whole text when shorter — so the slices ending
            # at `boundary` name every bucket that can hold a match.
            found = None
            for length in range(1, min(_REPLY_SUFFIX, boundary) + 1):
                bucket = self.reply_buckets.get(text[boundary - length : boundary])
                if not bucket:
                    continue
                for candidate in bucket:
                    if (
                        len(candidate) <= boundary
                        and text.endswith(candidate, 0, boundary)
                        and (found is None or len(candidate) > len(found))
                    ):
                        found = candidate
            if found is None:
                return None
            ids = self.replies[found]
            self.replies.move_to_end(found)
            self.reply_hits += 1
        return ids, boundary - len(found)

    def _encode_middle(self, text):
        """Tokens for `text`, which ends just past a turn marker, splicing
        the remembered ids of replies it resends. Pieces split only at turn
        boundaries, so without a match the result is the one-call encoding;
        None when a literal marker is not a tokenizer boundary."""
        pieces = []
        pos = len(text)
        while pos:
            boundary = text.rfind(self.marker, 0, pos)
            if boundary < 0:
                pieces.append(self.tokens(text[:pos]))
                break
            tail = self.tokens(text[boundary:pos])
            # As in encode, another added token can consume part of the
            # marker; this literal occurrence is then no boundary.
            if not tail or tail[0] != self.marker_id:
                return None
            pieces.append(tail)
            pos = boundary
            hit = self._reply_at(text, boundary)
            if hit is not None:
                ids, pos = hit
                pieces.append(list(ids))
        tokens = []
        for piece in reversed(pieces):
            tokens.extend(piece)
        return tokens

    def encode(self, text):
        """The tokens of the rendered prompt `text` (tokens), reusing a
        cached prompt's and a remembered reply's."""
        if not self.enabled:
            return self.tokens(text)
        boundary = text.rfind(self.marker)
        if boundary < 0:
            return self.tokens(text)
        boundary += len(self.marker)
        prefix = text[:boundary]
        remembered = self._replies_active()
        if boundary < self.MIN_PREFIX_CHARS:
            if remembered:
                middle = self._encode_middle(prefix)
                if middle is not None:
                    return middle + self.tokens(text[boundary:])
            return self.tokens(text)
        with self.lock:
            # An extension, an agent's ordinary request, reuses the longest
            # cached prompt it extends, whole: one comparison a prompt, which
            # stops at the first character that differs. Only a prompt that
            # extends none, which would encode from its start, compares
            # message by message. A shorter cached prompt the prompt extends
            # is taken even where another shares more of its messages: that
            # needs a stale entry beside a longer branch, and costs encode
            # time, never a token.
            key = max(
                (kept for kept in self.entries if prefix.startswith(kept)),
                key=len,
                default=None,
            )
            # How many of its messages: all ([:None]) of one the prompt extends.
            shared = None
            reused_chars = 0 if key is None else len(key)
            if key is None:
                for kept, (_, ends, _) in self.entries.items():
                    count = start = 0
                    for end in ends:
                        if prefix[start:end] != kept[start:end]:
                            break
                        count, start = count + 1, end
                    if start > reused_chars:
                        key, shared, reused_chars = kept, count, start
            if key is None:
                self.misses += 1
                reused, char_ends, token_ends = array("I"), array("Q"), array("Q")
            else:
                cached_tokens, char_ends, token_ends = self.entries[key]
                self.entries.move_to_end(key)
                char_ends, token_ends = char_ends[:shared], token_ends[:shared]
                reused = cached_tokens[: token_ends[-1]]
                self.hits += 1
                self.reused_tokens += len(reused)
        tokens = reused.tolist()
        if reused_chars < boundary:
            span = prefix[reused_chars:]
            middle = self._encode_middle(span) if remembered else None
            # The rest of the prefix encodes in one call: one call per message
            # took a cold 800-message, 15.9K-token prompt from 11.9 to 17.1 ms.
            # Each marker in it is the marker's token, which no other text
            # encodes to (enabled) and a remembered reply never holds, so the
            # n-th marker's token ends the n-th message. A different added
            # token can consume part of a marker, which is then no tokenizer
            # boundary: fewer tokens than markers are left, the last marker
            # finds none, and the prompt encodes whole.
            index = len(tokens) - 1
            tokens.extend(middle if middle is not None else self.tokens(span))
            end = reused_chars
            while end < boundary:
                end = prefix.find(self.marker, end) + len(self.marker)
                try:
                    index = tokens.index(self.marker_id, index + 1)
                except ValueError:
                    return self.tokens(text)
                char_ends.append(end)
                token_ends.append(index + 1)
            # A cached prompt: its tokens, and where each of its messages
            # ends, in characters and in tokens.
            entry = (array("I", tokens), char_ends, token_ends)
            size = sys.getsizeof(prefix) + sys.getsizeof(entry)
            size += sum(map(sys.getsizeof, entry))
            if size <= self.BUDGET_BYTES:
                with self.lock:
                    # An extension replaces the prompt it extends, whose
                    # message ends it holds; a branch keeps its own.
                    for old in {key, prefix} - {None}:
                        if old == key and not prefix.startswith(key):
                            continue
                        previous = self.entries.pop(old, None)
                        if previous is not None:
                            self.bytes -= sys.getsizeof(old) + sys.getsizeof(
                                previous
                            ) + sum(map(sys.getsizeof, previous))
                    self.entries[prefix] = entry
                    self.bytes += size
                    while (
                        self.bytes > self.BUDGET_BYTES
                        or len(self.entries) > self.CAPACITY
                    ):
                        old, previous = self.entries.popitem(last=False)
                        self.bytes -= sys.getsizeof(old) + sys.getsizeof(
                            previous
                        ) + sum(map(sys.getsizeof, previous))
                        self.evictions += 1
        return tokens + self.tokens(text[boundary:])

    def split(self, text):
        """`text` cut just past its last marker where the tokenizer splits it
        there: the tokens before the cut, and the text after it, which
        encodes on its own (tokens). No tokens and all of `text` where it
        does not."""
        boundary = text.rfind(self.marker) if self.marker is not None else -1
        if self.enabled and boundary >= 0:
            boundary += len(self.marker)
            head = self.tokens(text[:boundary])
            # As in encode, another added token may consume part of the
            # marker, which is then no boundary.
            if head and head[-1] == self.marker_id:
                return head, text[boundary:]
        return [], text

    def stats(self):
        with self.lock:
            return {
                "enabled": self.enabled,
                "entries": len(self.entries),
                "bytes": self.bytes,
                "budget_bytes": self.BUDGET_BYTES,
                "capacity": self.CAPACITY,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "reused_tokens": self.reused_tokens,
                "reply_entries": len(self.replies),
                "reply_hits": self.reply_hits,
            }

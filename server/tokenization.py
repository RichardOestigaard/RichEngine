"""Bounded token reuse at literal chat-message boundaries."""

import json
import os
import sys
import threading
from array import array
from collections import OrderedDict

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


class PromptTokenizer:
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
        self.entries = OrderedDict()
        self.bytes = self.hits = self.reused_tokens = 0
        # Reply text -> the ids the model emitted for it, a small LRU.
        self.replies = OrderedDict()
        # Reply text suffix -> the reply texts carrying it, so _reply_at
        # checks a handful of buckets per boundary instead of every reply.
        self.reply_buckets = {}
        self.reply_hits = 0
        self.lock = threading.Lock()

    MIN_PREFIX_CHARS = 4096
    BUDGET_BYTES = 8 * 1024 * 1024
    # One entry per live conversation prefix; BUDGET_BYTES is the real
    # memory bound, so this only needs to outnumber concurrent sessions.
    CAPACITY = 32
    REPLY_CAPACITY = 256

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

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def record_reply(self, text, token_ids):
        """Remember `token_ids` as the ids the model emitted for the reply
        text `text`, which a later conversation resends inside an assistant
        turn. The turn-end token the emission ended on is not part of the
        reply, so one trailing marker id is dropped."""
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
                pieces.append(self._encode(text[:pos]))
                break
            tail = self._encode(text[boundary:pos])
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
        if not self.enabled:
            return self._encode(text)
        boundary = text.rfind(self.marker)
        if boundary < 0:
            return self._encode(text)
        boundary += len(self.marker)
        prefix = text[:boundary]
        remembered = self._replies_active()
        if boundary < self.MIN_PREFIX_CHARS:
            if remembered:
                middle = self._encode_middle(prefix)
                if middle is not None:
                    return middle + self._encode(text[boundary:])
            return self._encode(text)
        with self.lock:
            key = max(
                (key for key in self.entries if prefix.startswith(key)),
                key=len,
                default="",
            )
            cached = self.entries.get(key)
            if cached is not None:
                self.entries.move_to_end(key)
                self.hits += 1
                self.reused_tokens += len(cached) // array("I").itemsize
        tokens = array("I", cached).tolist() if cached is not None else []
        if key != prefix:
            span = prefix[len(key) :]
            middle = self._encode_middle(span) if remembered else None
            tokens.extend(middle if middle is not None else self._encode(span))
            # A different added token can consume part of the marker. In that
            # case this literal occurrence is not a tokenizer boundary.
            if not tokens or tokens[-1] != self.marker_id:
                return self._encode(text)
            packed = array("I", tokens).tobytes()
            size = sys.getsizeof(prefix) + sys.getsizeof(packed)
            if size <= self.BUDGET_BYTES:
                with self.lock:
                    # An extension replaces its earlier prefix; unrelated
                    # concurrent conversations retain their own LRU entries.
                    for old in {key, prefix}:
                        previous = self.entries.pop(old, None)
                        if previous is not None:
                            self.bytes -= sys.getsizeof(old) + sys.getsizeof(previous)
                    self.entries[prefix] = packed
                    self.bytes += size
                    while (
                        self.bytes > self.BUDGET_BYTES
                        or len(self.entries) > self.CAPACITY
                    ):
                        old, previous = self.entries.popitem(last=False)
                        self.bytes -= sys.getsizeof(old) + sys.getsizeof(previous)
        return tokens + self._encode(text[boundary:])

    def split(self, text):
        """`text` cut just past its last marker where the tokenizer splits it
        there: the tokens before the cut, and the text after it, which
        encodes on its own. No tokens and all of `text` where it does not."""
        boundary = text.rfind(self.marker) if self.marker is not None else -1
        if self.enabled and boundary >= 0:
            boundary += len(self.marker)
            head = self._encode(text[:boundary])
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
                "reused_tokens": self.reused_tokens,
                "reply_entries": len(self.replies),
                "reply_hits": self.reply_hits,
            }

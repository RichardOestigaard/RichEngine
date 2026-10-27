"""Byte-level DFA compilation of constrained output.

Compiles the regular subset of what the llguidance grammars produce — flat
or nested closed JSON object schemas and the tool-call dialect envelopes —
to a minimized byte-level DFA the native engine walks directly. Anything
outside the subset returns None from compile_dfa and callers keep the
llguidance path.

Marker edges mark byte spans only a special token may spell (grammar token
references, and literal text that canonical tokenization maps to one): a
non-special token's byte walk may never take a marker-only edge, and a
special token must take at least one."""

from __future__ import annotations

import array
import json
import re
from dataclasses import dataclass

from . import tool_schema
from .errors import APIError
from .tool_schema import MAX_NAME_LENGTH, NAME_SPACE, QWEN3_XML, THINK_END

# Compiled DFAs stay small enough to ship in a CONSTRAINT_DFA frame, whose
# wire limit is 32767 states (Protocol.cpp kMaximumDfaStates). Staying one
# below keeps every state id — including the stop state at n_states — inside
# the int16 domain the transition table speaks.
MAX_DFA_STATES = 32766
# Object properties a schema leaves optional compile to state-sharing
# branches; measured growth is linear (~85 states per string-valued
# property), so this is a compile-time guard, not a semantic one — the state
# cap above stays the real bound.
MAX_OPTIONAL_PROPERTIES = 64
# Array item bounds above this compile unbounded or fall back. This must
# match tool_schema.MAX_GRAMMAR_BOUND exactly: the grammar path drops
# minItems/maxItems above its bound, so a larger DFA bound would enforce a
# constraint the fallback grammar does not — a semantic split, not a subset
# widening.
MAX_ARRAY_BOUND = 64
# Keep at 16: deep value chains make _minimize's Moore refinement run a
# round per distinguishable level — O(states² · 256) — so a deeper bound
# would blow up compile time long before the state cap bites.
MAX_SCHEMA_DEPTH = 16
MAX_FORCED_BYTES = 4096
# Determinization bail-out; the minimized DFA may still end smaller.
_MAX_SUBSETS = 65536

_MAX_CP = 0x10FFFF
_WS_BYTES = frozenset(b" \t\n\r")

_JSON_STRING = r'"([^"\\\x00-\x1f\x7f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
_JSON_INT = r"-?(0|[1-9][0-9]*)"
_JSON_NUM = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
_PYTHON_STRING = r"'([^'\\\n\r]|\\[\\'nr])*'"
_NUM = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"

# Schema keywords a DFA value may use; anything else falls back.
_ANNOTATIONS = tool_schema.SCHEMA_ANNOTATIONS | {"$defs", "definitions"}
_VALUE_KEYS = _ANNOTATIONS | {
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "anyOf",
    "oneOf",
}


class _Unsupported(Exception):
    """Outside the DFA-compilable subset; callers take the grammar path."""


class _Nfa:
    """Thompson NFA over bytes. sym edges carry (byteset, target, marker)."""

    def __init__(self):
        self.eps = []
        self.sym = []
        self.acc = set()
        # States inside a `.`/`.*` lexeme, where EOS is allowed even
        # mid-way: llguidance lets the model stop inside free text.
        self.textual = set()

    def state(self):
        self.eps.append([])
        self.sym.append([])
        return len(self.eps) - 1


def _lit(nfa, data, marker=False):
    start = nfa.state()
    s = start
    for byte in data:
        nxt = nfa.state()
        nfa.sym[s].append((1 << byte, nxt, marker))
        s = nxt
    return start, s


def _cat(nfa, frags):
    if not frags:
        q = nfa.state()
        return q, q
    for (_, e), (s2, _) in zip(frags, frags[1:]):
        nfa.eps[e].append(s2)
    return frags[0][0], frags[-1][1]


def _alt(nfa, frags):
    if len(frags) == 1:
        return frags[0]
    s, e = nfa.state(), nfa.state()
    for s2, e2 in frags:
        nfa.eps[s].append(s2)
        nfa.eps[e2].append(e)
    return s, e


def _opt(nfa, frag):
    s, e = nfa.state(), nfa.state()
    s2, e2 = frag
    nfa.eps[s] += [e, s2]
    nfa.eps[e2].append(e)
    return s, e


def _star(nfa, frag):
    s, e = nfa.state(), nfa.state()
    s2, e2 = frag
    nfa.eps[s] += [e, s2]
    nfa.eps[e2] += [e, s2]
    return s, e


def _rep(nfa, build, lo, hi):
    """The fragment `build()` produces, repeated lo..hi times (hi=None for
    unbounded)."""
    parts = [build() for _ in range(lo)]
    if hi is None:
        parts.append(_star(nfa, build()))
    else:
        s, e = nfa.state(), nfa.state()
        cur = s
        for _ in range(hi - lo):
            s2, e2 = build()
            nfa.eps[cur].append(s2)
            nfa.eps[cur].append(e)
            cur = e2
        nfa.eps[cur].append(e)
        parts.append((s, e))
    return _cat(nfa, parts)


def _byteset(lo, hi):
    out = 0
    for b in range(lo, hi + 1):
        out |= 1 << b
    return out


def _edge(nfa, byteset):
    s, e = nfa.state(), nfa.state()
    nfa.sym[s].append((byteset, e, False))
    return s, e


# --- UTF-8 range expansion -------------------------------------------------


def _encode_cp(cp, n):
    if n == 1:
        return [cp]
    lead = (cp >> (6 * (n - 1))) | {2: 0xC0, 3: 0xE0, 4: 0xF0}[n]
    return [lead] + [0x80 | (cp >> (6 * i)) & 0x3F for i in range(n - 2, -1, -1)]


def _decode_cp(bs):
    if len(bs) == 1:
        return bs[0]
    cp = bs[0] & {2: 0x1F, 3: 0x0F, 4: 0x07}[len(bs)]
    for b in bs[1:]:
        cp = cp << 6 | (b & 0x3F)
    return cp


def _utf8n(lo, hi, n):
    """Byte-range sequences covering the range [lo,hi] of n-byte codepoints.

    Split the codepoint range at the least significant byte position where
    the encoded endpoints differ: a low part padded with 0xBF continuations,
    a mid part whose differing byte spans the gap, and a high part padded
    with 0x80. Each part recurses until its last byte is the only variable."""
    if n == 1:
        return [[(lo, hi)]]
    lo_b, hi_b = _encode_cp(lo, n), _encode_cp(hi, n)
    i = 0
    while lo_b[i] == hi_b[i]:
        i += 1
    if i == n - 1:
        return [[(lo_b[j], hi_b[j]) for j in range(n)]]
    suffix = n - 1 - i
    lo_end = _decode_cp(lo_b[: i + 1] + [0xBF] * suffix)
    hi_start = _decode_cp(hi_b[: i + 1] + [0x80] * suffix)
    out = _utf8n(lo, lo_end, n)
    if lo_b[i] + 1 <= hi_b[i] - 1:
        out.append(
            [(b, b) for b in lo_b[:i]]
            + [(lo_b[i] + 1, hi_b[i] - 1)]
            + [(0x80, 0xBF)] * suffix
        )
    out += _utf8n(hi_start, hi, n)
    return out


def _utf8seqs(lo, hi):
    out = []
    for cplo, cphi, n in (
        (0x00, 0x7F, 1),
        (0x80, 0x7FF, 2),
        (0x800, 0xD7FF, 3),
        (0xE000, 0xFFFF, 3),
        (0x10000, 0x10FFFF, 4),
    ):
        a, b = max(lo, cplo), min(hi, cphi)
        if a <= b:
            out.extend(_utf8n(a, b, n))
    return out


def _to_ranges(cps):
    """A set of codepoints as sorted disjoint [lo,hi] ranges."""
    ranges = []
    for cp in sorted(cps):
        if ranges and ranges[-1][1] == cp - 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])
    return ranges


def _merge_ranges(items):
    """(lo,hi) ranges as sorted disjoint merged ranges."""
    ranges = []
    for lo, hi in sorted(items):
        if lo > hi:
            continue
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1][1] = max(ranges[-1][1], hi)
        else:
            ranges.append([lo, hi])
    return ranges


def _complement_ranges(ranges):
    out = []
    nxt = 0
    for lo, hi in sorted(ranges):
        if lo > nxt:
            out.append((nxt, lo - 1))
        nxt = hi + 1
    if nxt <= _MAX_CP:
        out.append((nxt, _MAX_CP))
    return out


def _class_frag(nfa, ranges, negated=False):
    """One character of `ranges` (cp [lo,hi] pairs) as a byte-level fragment."""
    if negated:
        ranges = _complement_ranges(ranges)
    byte_set = 0
    parts = []
    for lo, hi in ranges:
        if lo < 0x80:
            byte_set |= _byteset(lo, min(hi, 0x7F))
            lo = 0x80
        if lo > hi:
            continue
        for seq in _utf8seqs(lo, hi):
            start = nfa.state()
            s = start
            for blo, bhi in seq:
                t = nfa.state()
                nfa.sym[s].append((_byteset(blo, bhi), t, False))
                s = t
            parts.append((start, s))
    if byte_set:
        parts.append(_edge(nfa, byte_set))
    return _alt(nfa, parts) if parts else _lit(nfa, b"\x00")


# --- Regex -----------------------------------------------------------------

_CLASS_ESCAPES = {
    "d": [(0x30, 0x39)],
    "w": [(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)],
    "s": [(0x09, 0x0D), (0x20, 0x20)],
}
_CHAR_ESCAPES = {"n": 0x0A, "r": 0x0D, "t": 0x09, "f": 0x0C, "v": 0x0B, "0": 0}


class _Rx:
    """A compact regex parser over characters; /slash/ literals are the
    caller's problem. Unsupported constructs raise _Unsupported."""

    def __init__(self, pattern):
        self.s = pattern
        self.i = 0
        self.dotall = False

    def _peek(self):
        return self.s[self.i] if self.i < len(self.s) else ""

    def parse(self):
        node = self._alternation()
        if self.i != len(self.s):
            raise _Unsupported("trailing regex input")
        return node

    def _alternation(self):
        parts = [self._concat()]
        while self._peek() == "|":
            self.i += 1
            parts.append(self._concat())
        return ("alt", parts) if len(parts) > 1 else parts[0]

    def _concat(self):
        parts = []
        while self.i < len(self.s) and self.s[self.i] not in "|)":
            parts.append(self._repeat())
        return ("cat", parts)

    def _repeat(self):
        node = self._atom()
        while True:
            c = self._peek()
            if c == "*":
                lo, hi = 0, None
                self.i += 1
            elif c == "+":
                lo, hi = 1, None
                self.i += 1
            elif c == "?":
                lo, hi = 0, 1
                self.i += 1
            elif c == "{":
                m = re.match(r"\{(\d+)(?:,(\d*))?\}", self.s[self.i :])
                if not m:
                    break
                lo = int(m.group(1))
                hi = (
                    lo
                    if m.group(2) is None
                    else (int(m.group(2)) if m.group(2) else None)
                )
                self.i += m.end()
            else:
                break
            node = ("rep", node, lo, hi)
            if self._peek() == "?":
                self.i += 1  # lazy quantifier: same language
        return node

    def _escape(self):
        """After a backslash: a codepoint or class-ranges escape."""
        self.i += 1
        if self.i >= len(self.s):
            raise _Unsupported("trailing backslash")
        c = self.s[self.i]
        self.i += 1
        if c in _CLASS_ESCAPES:
            return _CLASS_ESCAPES[c]
        if c in "DWS":
            return _complement_ranges(_CLASS_ESCAPES[c.lower()])
        if c in _CHAR_ESCAPES:
            cp = _CHAR_ESCAPES[c]
            return [(cp, cp)]
        if c == "x":
            cp = int(self.s[self.i : self.i + 2], 16)
            self.i += 2
            return [(cp, cp)]
        if c == "u":
            cp = int(self.s[self.i : self.i + 4], 16)
            self.i += 4
            return [(cp, cp)]
        # A bare escape of any other character is the character.
        return [(ord(c), ord(c))]

    def _atom(self):
        c = self._peek()
        if c == "(":
            self.i += 1
            dotall = self.dotall
            if self.s.startswith("?:", self.i):
                self.i += 2
            elif self.s.startswith("?s:", self.i):
                self.i += 3
                self.dotall = True
            elif self._peek() == "?":
                raise _Unsupported("unsupported group syntax")
            node = self._alternation()
            if self._peek() != ")":
                raise _Unsupported("unclosed group")
            self.i += 1
            self.dotall = dotall
            return node
        if c == "[":
            return ("set", *self._class())
        if c == ".":
            self.i += 1
            ranges = [(0, _MAX_CP)] if self.dotall else [(0, 9), (0x0B, _MAX_CP)]
            return ("set", ranges, False)
        if c == "\\":
            ranges = self._escape()
            if len(ranges) == 1 and ranges[0][0] == ranges[0][1]:
                return ("lit", chr(ranges[0][0]))
            return ("set", ranges, False)
        if c in "^$":
            raise _Unsupported("anchors are not supported")
        self.i += 1
        return ("lit", c)

    def _class(self):
        self.i += 1  # '['
        negated = self._peek() == "^"
        if negated:
            self.i += 1
        cps = []
        first = True
        while True:
            if self.i >= len(self.s):
                raise _Unsupported("unclosed character class")
            if self.s[self.i] == "]" and not first:
                self.i += 1
                break
            first = False
            if self.s[self.i] == "\\":
                items = self._escape()
                if len(items) == 1 and items[0][0] == items[0][1]:
                    lo = items[0][0]
                    if self._peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                        self.i += 1
                        hi_cp = self._class_char()
                        cps.append((lo, hi_cp))
                        continue
                    cps.append((lo, lo))
                else:
                    cps.extend(items)
                continue
            lo = self._class_char()
            if self._peek() == "-" and self.i + 1 < len(self.s) and self.s[self.i + 1] != "]":
                self.i += 1
                hi = self._class_char()
                cps.append((lo, hi))
            else:
                cps.append((lo, lo))
        return _merge_ranges(cps), negated

    def _class_char(self):
        c = self.s[self.i]
        if c == "\\":
            self.i += 1
            c = self.s[self.i]
            if c in _CHAR_ESCAPES:
                cp = _CHAR_ESCAPES[c]
            elif c == "x":
                cp = int(self.s[self.i + 1 : self.i + 3], 16)
                self.i += 2
            elif c == "u":
                cp = int(self.s[self.i + 1 : self.i + 5], 16)
                self.i += 4
            else:
                cp = ord(c)
            self.i += 1
            return cp
        self.i += 1
        return ord(c)


def _rx(nfa, node):
    kind = node[0]
    if kind == "lit":
        return _lit(nfa, node[1].encode("utf-8"))
    if kind == "set":
        return _class_frag(nfa, node[1], node[2])
    if kind == "cat":
        return _cat(nfa, [_rx(nfa, c) for c in node[1]])
    if kind == "alt":
        return _alt(nfa, [_rx(nfa, c) for c in node[1]])
    if kind == "rep":
        return _rep(nfa, lambda: _rx(nfa, node[1]), node[2], node[3])
    raise _Unsupported(f"unknown regex node {kind}")


def _regex(nfa, pattern):
    return _rx(nfa, _Rx(pattern).parse())


# --- DFA algorithms ---------------------------------------------------------


def _eps_closure(nfa, states):
    out = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in nfa.eps[s]:
            if t not in out:
                out.add(t)
                stack.append(t)
    return frozenset(out)


def _determinize(nfa, start):
    """Subset construction. Rows are 256-wide; marker bit per byte edge = all
    contributing NFA edges are marker edges."""
    rows = []
    accepting = []
    markers = []
    textual = []
    seed = _eps_closure(nfa, [start])
    ids = {seed: 0}
    work = [seed]
    i = 0
    while i < len(work):
        subset = work[i]
        accepting.append(any(s in nfa.acc for s in subset))
        textual.append(any(s in nfa.textual for s in subset))
        row = [-1] * 256
        mark = 0
        per_byte = {}
        for s in subset:
            for bs, dst, mk in nfa.sym[s]:
                m = bs
                while m:
                    low = m & -m
                    b = low.bit_length() - 1
                    m ^= low
                    targets, all_marker = per_byte.get(b, (set(), True))
                    if targets is None:
                        targets, all_marker = set(), True
                    targets.add(dst)
                    all_marker &= mk
                    per_byte[b] = (targets, all_marker)
        for b, (targets, all_marker) in per_byte.items():
            sub = _eps_closure(nfa, targets)
            j = ids.get(sub)
            if j is None:
                j = len(work)
                ids[sub] = j
                work.append(sub)
                if len(work) > _MAX_SUBSETS:
                    raise _Unsupported("determinized automaton is too large")
            row[b] = j
            if all_marker:
                mark |= 1 << b
        rows.append(row)
        markers.append(mark)
        i += 1
    return rows, accepting, markers, textual


def _minimize(rows, accepting, markers, textual):
    """Moore refinement on (accepting, textual, per-byte (target block,
    marker bit))."""
    n = len(rows)
    block = [a * 2 + t for a, t in zip(accepting, textual)]
    while True:
        signatures = {}
        new = [0] * n
        for s in range(n):
            mark = markers[s]
            row = rows[s]
            sig = (
                block[s],
                tuple(
                    (block[t], (mark >> b) & 1) if t >= 0 else (-1, 0)
                    for b, t in enumerate(row)
                ),
            )
            new[s] = signatures.setdefault(sig, len(signatures))
        if new == block:
            break
        block = new
    # Rebuild in BFS order so the start block is state 0.
    rep = {}
    for s, b in enumerate(block):
        rep.setdefault(b, s)
    order = {}
    queue = [block[0]]
    order[block[0]] = 0
    while queue:
        b = queue.pop(0)
        s = rep[b]
        for t in rows[s]:
            if t >= 0:
                tb = block[t]
                if tb not in order:
                    order[tb] = len(order)
                    queue.append(tb)
    n2 = len(order)
    new_rows = [[-1] * 256 for _ in range(n2)]
    new_acc = [False] * n2
    new_marks = [0] * n2
    new_textual = [False] * n2
    for b, j in order.items():
        s = rep[b]
        new_acc[j] = accepting[s]
        new_marks[j] = markers[s]
        new_textual[j] = textual[s]
        for byte, t in enumerate(rows[s]):
            if t >= 0:
                new_rows[j][byte] = order[block[t]]
    return new_rows, new_acc, new_marks, new_textual


def _complete(rows, accepting):
    """A full transition table: the dead state is an extra last row."""
    dead = len(rows)
    out = [dead if t < 0 else t for row in rows for t in row]
    rows2 = [out[i * 256 : (i + 1) * 256] for i in range(len(rows))]
    rows2.append([dead] * 256)
    return rows2, list(accepting) + [False]


def _complement(rows, accepting):
    rows2, acc2 = _complete(rows, accepting)
    return rows2, [not a for a in acc2], [0] * len(rows2)


def _product(rows1, acc1, rows2, acc2):
    """Intersection of two partial DFAs; dead edges never pair."""
    start = (0, 0)
    ids = {start: 0}
    work = [start]
    rows, acc, marks = [], [], []
    i = 0
    while i < len(work):
        a, b = work[i]
        acc.append(acc1[a] and acc2[b])
        row = [-1] * 256
        mark = 0
        for byte in range(256):
            ta, tb = rows1[a][byte], rows2[b][byte]
            if ta < 0 or tb < 0:
                continue
            pair = (ta, tb)
            j = ids.get(pair)
            if j is None:
                j = len(work)
                ids[pair] = j
                work.append(pair)
                if len(work) > _MAX_SUBSETS:
                    raise _Unsupported("product automaton is too large")
            row[byte] = j
        rows.append(row)
        marks.append(mark)
        i += 1
    return rows, acc, marks


def _anychar_dfa():
    """The `.*` character stream as a DFA: any valid UTF-8 character string."""
    nfa = _Nfa()
    s, e = _class_frag(nfa, [(0, _MAX_CP)])
    nfa.eps[e].append(s)
    nfa.acc.add(s)
    return _determinize(nfa, s)[:3]


def _contains_dfa(needles):
    """`.*(n1|n2|...).*` over valid characters: accepts strings containing a
    needle."""
    nfa = _Nfa()
    s0 = nfa.state()
    sa = nfa.state()
    for q in (s0, sa):
        s, e = _class_frag(nfa, [(0, _MAX_CP)])
        nfa.eps[q].append(s)
        nfa.eps[e].append(q)
    nfa.acc.add(sa)
    for needle in needles:
        if not needle:
            continue
        s, e = _lit(nfa, needle)
        nfa.eps[s0].append(s)
        nfa.eps[e].append(sa)
    return _determinize(nfa, s0)[:3]


def _not_contains(needles):
    """The TEXT automaton: valid UTF-8 strings containing none of the byte
    needles — lark's `/(?s:.*)/ & ~/(?s:.*)needle(?s:.*)/`."""
    rows, acc, marks = _contains_dfa(needles)
    rows2, acc2, _ = _complement(rows, acc)
    v_rows, v_acc, _ = _anychar_dfa()
    return _product(v_rows, v_acc, rows2, acc2)


def _embed_dfa(nfa, dfa, textual=False):
    """Replant a DFA as an NFA fragment: any accepting state may end it.
    Edges into states that can never accept are pruned first — otherwise a
    dying branch (say a barrier match inside free text) merges with a
    marker edge and erases its marker. `textual` marks the replanted states
    as free text where EOS may sit."""
    rows, acc, marks = dfa
    live = _live_states(rows, acc)
    ids = [nfa.state() for _ in rows]
    for i, row in enumerate(rows):
        for b, t in enumerate(row):
            if t >= 0 and live[t >> 3] >> (t & 7) & 1:
                nfa.sym[ids[i]].append((1 << b, ids[t], bool(marks[i] >> b & 1)))
        if textual:
            nfa.textual.add(ids[i])
    end = nfa.state()
    for i, a in enumerate(acc):
        if a:
            nfa.eps[ids[i]].append(end)
    return ids[0], end


def _live_states(rows, accepting):
    """Bitset of states that can reach an accepting state."""
    n = len(rows)
    rev = [[] for _ in range(n)]
    for s, row in enumerate(rows):
        for t in row:
            if t >= 0:
                rev[t].append(s)
    live = bytearray((n + 7) // 8)
    stack = []
    for s, a in enumerate(accepting):
        if a:
            live[s >> 3] |= 1 << (s & 7)
            stack.append(s)
    while stack:
        s = stack.pop()
        for p in rev[s]:
            if not live[p >> 3] >> (p & 7) & 1:
                live[p >> 3] |= 1 << (p & 7)
                stack.append(p)
    return live


def _bitset(flags):
    out = bytearray((len(flags) + 7) // 8)
    for i, f in enumerate(flags):
        if f:
            out[i >> 3] |= 1 << (i & 7)
    return out


# --- Compiled DFA ------------------------------------------------------------


class CompiledDfa:
    """A minimized byte-level DFA: state 0 is the start; table rows carry
    next-state or -1 per byte; accepting and live are bitmasks; marker is a
    per-state 256-bit mask of edges only special tokens may traverse."""

    __slots__ = (
        "n_states", "table", "accepting", "live", "marker", "eos_ok", "_costs"
    )

    def __init__(self, n_states, table, accepting, live, marker, eos_ok):
        self.n_states = n_states
        self.table = table  # array.array('h'), n_states * 256
        self.accepting = accepting  # bytearray bitmask
        self.live = live  # bytearray bitmask of states reaching acceptance
        self.marker = marker  # list[int], 256-bit per state
        self.eos_ok = eos_ok  # bytearray bitmask of free-text states
        self._costs = None  # termination_cost()'s cache

    def is_accepting(self, state):
        return bool(self.accepting[state >> 3] >> (state & 7) & 1)

    def _eos_ok(self, state):
        return bool(self.eos_ok[state >> 3] >> (state & 7) & 1)

    def _is_live(self, state):
        return bool(self.live[state >> 3] >> (state & 7) & 1)

    def _walk(self, state, data, special):
        """Byte walk honoring marker rules; -1 for a dead or non-live end."""
        marked = False
        for b in data:
            base = state * 256
            nxt = self.table[base + b]
            if nxt < 0:
                return -1
            edge_marked = self.marker[state] >> b & 1
            if edge_marked:
                if not special:
                    return -1
                marked = True
            state = nxt
        if special and not marked:
            return -1
        return state if self._is_live(state) else -1

    def advance_token(self, state, token_id, vocab_tokens, eos_tokens, special_tokens):
        """The state after `token_id`, or -1 when the token is not allowed.
        `n_states` is the 'stopped' state after a legal EOS."""
        if token_id in eos_tokens:
            if self.is_accepting(state) or self._eos_ok(state):
                return self.n_states
            return -1
        if state >= self.n_states:
            return -1
        data = vocab_tokens[token_id]
        if not data:
            return -1
        return self._walk(state, data, token_id in special_tokens)

    def masks_for_state(
        self, state, vocab_tokens, eos_tokens=(), special_tokens=frozenset()
    ):
        """Token ids the state allows: byte-walked tokens that stay live, EOS
        at accepting states."""
        out = []
        if self.is_accepting(state) or self._eos_ok(state):
            out.extend(eos_tokens)
        for tid, data in enumerate(vocab_tokens):
            if tid in eos_tokens or not data:
                continue
            if self._walk(state, data, tid in special_tokens) >= 0:
                out.append(tid)
        out.sort()
        return out

    def _forced_bytes(self, state):
        """The forced byte chain: bytes of states with exactly one live
        out-edge."""
        out = bytearray()
        while len(out) < MAX_FORCED_BYTES and state < self.n_states:
            nxt = -1
            byte = 0
            base = state * 256
            for b in range(256):
                t = self.table[base + b]
                if t >= 0 and self._is_live(t):
                    if nxt >= 0:
                        return bytes(out)
                    nxt, byte = t, b
            if nxt < 0:
                break
            out.append(byte)
            state = nxt
        return bytes(out)

    def forced_tokens_after(
        self, state, vocab_tokens, encode, eos_tokens=(), special_tokens=frozenset()
    ):
        """The token ids canonically spelling the forced byte chain, like
        TokenConstraint.forced_tokens: encode, require an anchor plus at
        least one more token, and validate on a state copy."""
        chain = self._forced_bytes(state)
        if not chain or encode is None:
            return ()
        try:
            text = chain.decode("utf-8")
        except UnicodeDecodeError:
            return ()
        try:
            encoded = encode(text, add_special_tokens=False)
            candidate = tuple(getattr(encoded, "ids", encoded))
        except Exception:
            return ()
        if len(candidate) < 2:
            return ()
        q = state
        for tid in candidate:
            q = self.advance_token(q, tid, vocab_tokens, eos_tokens, special_tokens)
            if q < 0 or q >= self.n_states:
                return ()
        return candidate

    def byte_costs(self):
        """Per state, the minimum byte count of a walk to acceptance over
        edges a regular token may take (never marker-only), the bound the
        engine's decode budget rule gates with: every unmarked edge is
        walkable by a single-byte token, so no grammar path finishes in
        fewer tokens than the bound read. Marker-only paths (only special
        tokens may walk them) and dead states get the 32767 sentinel — the
        engine's budget rule keeps such rows on the plain grammar walk
        rather than trusting a bound it cannot see. Cached: a grammar's
        DFA repeats across its requests."""
        if self._costs is not None:
            return self._costs
        try:
            import numpy as np
        except Exception:
            self._costs = ()
            return self._costs
        n = self.n_states
        sentinel = 32767
        table = (
            np.frombuffer(bytes(self.table), dtype=np.int16)
            .astype(np.int32)
            .reshape(n, 256)
        )
        marked = np.unpackbits(
            np.frombuffer(
                b"".join(m.to_bytes(32, "little") for m in self.marker),
                np.uint8,
            ).reshape(n, 32),
            axis=1,
            bitorder="little",
        )
        accept = np.unpackbits(
            np.frombuffer(bytes(self.accepting), np.uint8), bitorder="little"
        )[:n].astype(bool)
        dist = np.full(n, sentinel, np.int32)
        dist[accept] = 0
        unmarked = marked == 0
        # Label-correcting rounds to fixpoint: a state's bound shrinks only
        # through an edge into a state whose bound just shrank, so each
        # round propagates at least one distance and the loop ends at the
        # fixpoint the shortest path needs.
        for _ in range(n):
            cand = np.where((table >= 0) & unmarked, dist[table] + 1, sentinel)
            update = np.minimum(dist, cand.min(axis=1))
            if np.array_equal(update, dist):
                break
            dist = update
        self._costs = array.array("h", dist.tolist())
        return self._costs


# --- Spec input --------------------------------------------------------------


@dataclass(frozen=True)
class DfaSpec:
    """The structural description of one constrained output, mirroring the
    arguments tool_grammar/json_grammar take."""

    kind: str  # "json" | "tools"
    thinking: bool = False
    # The think-close token's byte spelling: marker and default TEXT barrier.
    think_end: str = THINK_END
    policy: object = None  # ToolPolicy for "tools"
    response_schema: object = None


def tool_spec(policy, thinking, response_schema=None, *, think_end=THINK_END):
    return DfaSpec("tools", thinking, think_end, policy, response_schema)


def json_spec(schema, thinking, *, think_end=THINK_END):
    return DfaSpec("json", thinking, think_end, None, schema)


def _barrier_literals(pattern):
    """A dialect's barrier regex as byte literals: alternatives joined by
    top-level `|`, each a plain literal once unescaped."""
    alts = []
    cur = []
    esc = False
    for ch in pattern:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == "|":
            alts.append("".join(cur))
            cur = []
            continue
        if ch in ".[](){}*+?^$":
            raise _Unsupported("barrier is not a literal alternation")
        cur.append(ch)
    alts.append("".join(cur))
    out = [a.encode("utf-8") for a in alts if a]
    if not out:
        raise _Unsupported("empty barrier")
    return out


def _split_literal(nfa, data, markers, canonical=True):
    """Literal text as fragments.

    When canonical, spans spelling a special token become marker edges —
    the parser forces canonical tokenization for a literal that is the
    forced continuation (not an alternative head, and not after an element
    that can continue). When not, the literal is plain bytes."""
    if not canonical:
        return _lit(nfa, data)
    parts = []
    pos = 0
    while pos < len(data):
        best = None
        for m in markers:
            i = data.find(m, pos)
            if i < 0:
                continue
            if best is None or i < best[0] or (i == best[0] and len(m) > len(best[1])):
                best = (i, m)
        if best is None:
            break
        i, m = best
        if i > pos:
            parts.append(_lit(nfa, data[pos:i]))
        parts.append(_lit(nfa, m, marker=True))
        pos = i + len(m)
    if pos < len(data):
        parts.append(_lit(nfa, data[pos:]))
    return _cat(nfa, parts)


def _spelling_frag(nfa, dialect, text, markers, head_canonical=True):
    """Dialect text like ToolDialect.spelling: spans matching the dialect's
    token markers are token references — always marker edges, at any
    position. Other literal spans go through _split_literal: special token
    spellings inside a literal are protected only when the literal is a
    forced continuation — spans past the head are, the head span is only
    when head_canonical."""
    special = dialect.token_markers()
    parts = []
    head = True
    while text:
        marker = next((m for m in special if text.startswith(m)), None)
        if marker is not None:
            parts.append(_lit(nfa, marker.encode(), marker=True))
            text = text[len(marker) :]
            head = False
            continue
        ends = [text.find(m) for m in special]
        end = min((i for i in ends if i >= 0), default=len(text))
        parts.append(
            _split_literal(
                nfa,
                text[:end].encode(),
                markers,
                canonical=head_canonical or not head,
            )
        )
        head = False
        text = text[end:]
    return _cat(nfa, parts)


def _ws(nfa):
    ws_set = sum(1 << b for b in _WS_BYTES)
    return _rep(nfa, lambda: _edge(nfa, ws_set), 0, tool_schema.MAX_WHITESPACE)


# --- JSON values --------------------------------------------------------------


def _json_value(nfa, schema, depth):
    if depth > MAX_SCHEMA_DEPTH:
        raise _Unsupported("schema nests too deep")
    if not isinstance(schema, dict):
        raise _Unsupported("non-object schema node")
    keys = set(schema) - _VALUE_KEYS
    if keys:
        raise _Unsupported(f"schema keyword {sorted(keys)[0]}")
    values = schema.get("enum", [schema["const"]] if "const" in schema else None)
    if values is not None:
        if not isinstance(values, list) or any(
            v is not None and not isinstance(v, (str, bool, int)) for v in values
        ):
            raise _Unsupported("unsupported enum value")
        parts = [
            _lit(nfa, json.dumps(v, separators=(",", ":")).encode()) for v in values
        ]
        if not parts:
            raise _Unsupported("empty enum")
        return _alt(nfa, parts)
    union = schema.get("anyOf", schema.get("oneOf"))
    if isinstance(union, list):
        if not union:
            raise _Unsupported("empty union")
        return _alt(
            nfa, [_json_value(nfa, member, depth + 1) for member in union]
        )
    kind = schema.get("type")
    if isinstance(kind, list):
        return _alt(
            nfa,
            [
                _json_value(nfa, {**schema, "type": k}, depth + 1)
                for k in kind
            ],
        )
    if kind == "string":
        return _regex(nfa, _JSON_STRING)
    if kind == "integer":
        return _regex(nfa, _JSON_INT)
    if kind == "number":
        return _regex(nfa, _JSON_NUM)
    if kind == "boolean":
        return _regex(nfa, "true|false")
    if kind == "null":
        return _lit(nfa, b"null")
    if kind == "array" or "items" in schema or "minItems" in schema or "maxItems" in schema:
        if kind not in ("array", None):
            raise _Unsupported("non-array items")
        return _json_array(nfa, schema, depth)
    if kind == "object" or (kind is None and "properties" in schema):
        return _json_object(nfa, schema, depth)
    raise _Unsupported(f"schema type {kind!r}")


def _json_array(nfa, schema, depth):
    items = schema.get("items", True)
    if isinstance(items, list) or "prefixItems" in schema:
        raise _Unsupported("tuple items")
    lo = schema.get("minItems", 0)
    hi = schema.get("maxItems")
    if type(lo) is not int or lo < 0 or lo > MAX_ARRAY_BOUND:
        raise _Unsupported("minItems")
    if hi is not None and (type(hi) is not int or hi < lo or hi > MAX_ARRAY_BOUND):
        raise _Unsupported("maxItems")

    def item():
        return _cat(nfa, [_ws(nfa), _json_value(nfa, items, depth + 1)])

    start, end = nfa.state(), nfa.state()
    if lo == 0:
        nfa.eps[start].append(end)
    # First item (mandatory when lo >= 1): ws v (ws , ws v){lo-1,hi-1}
    first_s, first_e = item()
    nfa.eps[start].append(first_s)
    tail = _rep(
        nfa,
        lambda: _cat(nfa, [_ws(nfa), _lit(nfa, b","), _ws(nfa), _json_value(nfa, items, depth + 1)]),
        max(0, lo - 1),
        (hi - 1) if hi is not None else None,
    )
    nfa.eps[first_e].append(tail[0])
    nfa.eps[tail[1]].append(end)
    close_s, close_e = _cat(nfa, [_ws(nfa), _lit(nfa, b"]")])
    nfa.eps[end].append(close_s)
    head_s, head_e = _lit(nfa, b"[")
    nfa.eps[head_e].append(start)
    return head_s, close_e


def _json_object(nfa, schema, depth):
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise _Unsupported("properties")
    if schema.get("additionalProperties", True) is not False:
        raise _Unsupported("open object")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(
        name not in props for name in required
    ):
        raise _Unsupported("undeclared required property")
    optional = [name for name in props if name not in required]
    if len(optional) > MAX_OPTIONAL_PROPERTIES:
        raise _Unsupported("too many optional properties")
    names = list(props)

    # Chain states (index, emitted): optional props may be skipped, emitted
    # pairs after the first carry a leading comma. Tokens are ws-separated.
    nodes = {}
    for i in range(len(names) + 1):
        for emitted in (False, True):
            nodes[(i, emitted)] = nfa.state()
    end = nfa.state()
    for i, name in enumerate(names):
        value = props[name]
        req = name in required
        for emitted in (False, True):
            src = nodes[(i, emitted)]
            if not req:
                nfa.eps[src].append(nodes[(i + 1, emitted)])
            frags = []
            if emitted:
                frags += [_ws(nfa), _lit(nfa, b",")]
            frags += [
                _ws(nfa),
                _lit(nfa, json.dumps(name).encode()),
                _ws(nfa),
                _lit(nfa, b":"),
                _ws(nfa),
                _json_value(nfa, value, depth + 1),
            ]
            s, e = _cat(nfa, frags)
            nfa.eps[src].append(s)
            nfa.eps[e].append(nodes[(i + 1, True)])
    for emitted in (False, True):
        s, e = _cat(nfa, [_ws(nfa), _lit(nfa, b"}")])
        nfa.eps[nodes[(len(names), emitted)]].append(s)
        nfa.eps[e].append(end)
    head_s, head_e = _lit(nfa, b"{")
    nfa.eps[head_e].append(nodes[(0, False)])
    return head_s, end


# --- Tool-call dialects --------------------------------------------------------


def _ordered_params(nfa, params):
    """params: [(builder(emitted) -> frag, required)] emitted in order with
    optional ones skippable; the builder adds the separator once another
    parameter precedes it."""
    n = len(params)
    nodes = {}
    for i in range(n + 1):
        for emitted in (False, True):
            nodes[(i, emitted)] = nfa.state()
    start, end = nodes[(0, False)], nfa.state()
    for i, (build, required) in enumerate(params):
        for emitted in (False, True):
            src = nodes[(i, emitted)]
            if not required:
                nfa.eps[src].append(nodes[(i + 1, emitted)])
            s, e = build(emitted)
            nfa.eps[src].append(s)
            nfa.eps[e].append(nodes[(i + 1, True)])
    nfa.eps[nodes[(n, False)]].append(end)
    nfa.eps[nodes[(n, True)]].append(end)
    return start, end


def _xml_param(nfa, dialect, name, vschema, markers):
    prefix = (dialect.param_open + name + dialect.param_name_close).encode()
    close = dialect.param_close.encode()
    choices = []
    for member in tool_schema._union_members(vschema):
        texts = tool_schema._literal_texts(member, vschema)
        if texts is None:
            types = tool_schema.schema_types(member, vschema)
            if types is None or "string" in types:
                # Any text: a lazy value up to the closing markup.
                return _cat(
                    nfa,
                    [
                        _spelling_frag(nfa, dialect, prefix.decode(), markers),
                        _embed_dfa(nfa, _not_contains([close])),
                        _spelling_frag(nfa, dialect, close.decode(), markers),
                    ],
                )
            choices.append(_json_value(nfa, member, 0))
            continue
        for text in texts:
            if dialect.value_barrier is not None and dialect.value_barrier in text:
                raise APIError(400, "tool parameter enum contains XML framing")
            # Enum alternatives are alternative heads: byte-level.
            choices.append(
                _split_literal(nfa, text.encode(), markers, canonical=False)
                if text
                else _lit(nfa, b"")
            )
    if not choices:
        raise _Unsupported(f"tool parameter {name} has no value")
    return _cat(
        nfa,
        [
            _spelling_frag(nfa, dialect, prefix.decode(), markers),
            _alt(nfa, choices),
            _split_literal(nfa, close, markers),
        ],
    )


def _xml_args(nfa, dialect, schema, markers):
    if schema["additionalProperties"] is not False:
        raise _Unsupported("undeclared parameters")
    required = schema["required"]
    params = []
    for name, vschema in schema["properties"].items():
        if vschema is False:
            if name in required:
                raise APIError(
                    400, f"required tool parameter cannot have a value: {name}"
                )
            continue
        if (
            not isinstance(name, str)
            or not name
            or len(name) > MAX_NAME_LENGTH
            or name != name.strip(NAME_SPACE)
            or any(c in name for c in dialect.param_name_forbidden)
        ):
            raise APIError(400, "invalid tool parameter name")
        params.append(
            (
                lambda emitted, name=name, vschema=vschema: _xml_param(
                    nfa, dialect, name, vschema, markers
                ),
                name in required,
            )
        )
    return _ordered_params(nfa, params)


def _python_value(nfa, vschema):
    string_schema = tool_schema.raw_string_schema(vschema, None)
    if string_schema is not None:
        if string_schema[0] == "raw":
            return _regex(nfa, _PYTHON_STRING)
        return _alt(
            nfa,
            [
                _lit(nfa, tool_schema._python_string_literal(v).encode())
                for v in string_schema[1]
            ],
        )
    schema = vschema
    union = schema.get("anyOf", schema.get("oneOf")) if isinstance(schema, dict) else None
    if union:
        return _alt(nfa, [_python_value(nfa, option) for option in union])
    if "enum" in schema or "const" in schema:
        values = schema["enum"] if "enum" in schema else [schema["const"]]
        if not all(v is None or isinstance(v, (str, bool, int, float)) for v in values):
            raise _Unsupported("unsupported enum value")
        return _alt(
            nfa, [_lit(nfa, tool_schema._python_scalar(v).encode()) for v in values]
        )
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if kinds == ["boolean"]:
        return _regex(nfa, "True|False")
    if kinds == ["null"]:
        return _lit(nfa, b"None")
    if kinds and all(k in ("integer", "number") for k in kinds):
        return _regex(nfa, _NUM)
    raise _Unsupported("parameter value is not a DFA scalar")


def _python_args(nfa, dialect, schema, markers):
    if schema["additionalProperties"] is not False:
        raise _Unsupported("undeclared parameters")
    required = schema["required"]
    params = []
    for name, vschema in schema["properties"].items():
        if vschema is False:
            if name in required:
                raise APIError(
                    400, f"required tool parameter cannot have a value: {name}"
                )
            continue
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or any(c in name for c in "=(),'\"\\ \t\n\r")
        ):
            raise APIError(400, "invalid tool parameter name")

        def build(emitted, name=name, vschema=vschema):
            parts = [_lit(nfa, b", ")] if emitted else []
            parts += [
                _split_literal(nfa, (name + "=").encode(), markers),
                _python_value(nfa, vschema),
            ]
            return _cat(nfa, parts)

        params.append((build, name in required))
    return _ordered_params(nfa, params)


def _gemma4_value(nfa, schema, root):
    schema = tool_schema._resolve_tool_schema(schema, root)
    if schema is False or not isinstance(schema, dict):
        raise _Unsupported("parameter value")
    union = schema.get("anyOf", schema.get("oneOf"))
    if union:
        return _alt(
            nfa, [_gemma4_value(nfa, option, root) for option in union]
        )
    values = tool_schema._enumeration(schema)
    if values is not None:
        if not all(v is None or isinstance(v, (str, bool, int, float)) for v in values):
            raise _Unsupported("unsupported enum value")
        if any(isinstance(v, str) and '<|"|>' in v for v in values):
            raise APIError(400, "string tool parameter enum contains framing")
        return _alt(
            nfa, [_lit(nfa, tool_schema._gemma4_literal(v).encode()) for v in values]
        )
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    string_schema = tool_schema.raw_string_schema(schema, '<|"|>')
    if string_schema is not None and (kind is None or "string" in (kinds or ())):
        quote = '<|"|>'.encode()
        if string_schema[0] == "raw":
            return _cat(
                nfa,
                [
                    _lit(nfa, quote, marker=True),
                    _embed_dfa(nfa, _not_contains([quote])),
                    _lit(nfa, quote, marker=True),
                ],
            )
        return _alt(
            nfa,
            [
                _lit(nfa, tool_schema._gemma4_literal(v).encode())
                for v in string_schema[1]
            ],
        )
    if kinds == ["boolean"]:
        return _regex(nfa, "true|false")
    if kinds == ["null"]:
        return _lit(nfa, b"null")
    if kinds and all(k in ("integer", "number") for k in kinds):
        return _regex(nfa, _NUM)
    raise _Unsupported("parameter value is not a DFA scalar")


def _gemma4_args(nfa, dialect, schema, markers):
    if schema["additionalProperties"] is not False:
        raise _Unsupported("undeclared parameters")
    required = schema["required"]
    params = []
    for name, vschema in schema["properties"].items():
        if vschema is False:
            if name in required:
                raise APIError(
                    400, f"required tool parameter cannot have a value: {name}"
                )
            continue
        if (
            not isinstance(name, str)
            or not name
            or len(name) > MAX_NAME_LENGTH
            or name != name.strip(NAME_SPACE)
            or any(c in name for c in dialect.param_name_forbidden)
        ):
            raise APIError(400, "invalid tool parameter name")

        def build(emitted, name=name, vschema=vschema):
            parts = [_lit(nfa, b",")] if emitted else []
            parts += [
                _split_literal(nfa, (name + ":").encode(), markers),
                _gemma4_value(nfa, vschema, schema),
            ]
            return _cat(nfa, parts)

        params.append((build, name in required))
    return _ordered_params(nfa, params)


def _name_class_ranges(characters):
    return _complement_ranges(_to_ranges({ord(c) for c in characters}))


def _free_args(nfa, dialect, markers):
    """An unconstrained tool's arguments: any parameters in the dialect's
    XML framing, each with raw text. Non-XML dialects' free arguments hold
    free JSON and are not regular."""
    if dialect.kind != "xml":
        raise _Unsupported("free arguments need the XML framing")
    forbidden = set(dialect.param_name_forbidden)
    edge = forbidden | {" ", "\t"}
    name = _cat(
        nfa,
        [
            _class_frag(nfa, _name_class_ranges(edge)),
            _opt(
                nfa,
                _cat(
                    nfa,
                    [
                        _rep(
                            nfa,
                            lambda: _class_frag(nfa, _name_class_ranges(forbidden)),
                            0,
                            MAX_NAME_LENGTH - 2,
                        ),
                        _class_frag(nfa, _name_class_ranges(edge)),
                    ],
                ),
            ),
        ],
    )
    close = dialect.param_close.encode()
    # extra* is a star: each parameter's head sits at a choice boundary.
    param = _cat(
        nfa,
        [
            _spelling_frag(nfa, dialect, dialect.param_open, markers, head_canonical=False),
            name,
            _spelling_frag(nfa, dialect, dialect.param_name_close, markers),
            _embed_dfa(nfa, _not_contains([close])),
            _spelling_frag(nfa, dialect, dialect.param_close, markers),
        ],
    )
    return _star(nfa, param)


def _strict_args(nfa, dialect, schema, markers):
    if dialect.kind == "python":
        return _python_args(nfa, dialect, schema, markers)
    if dialect.kind == "gemma4":
        return _gemma4_args(nfa, dialect, schema, markers)
    return _xml_args(nfa, dialect, schema, markers)


def _call_frag(nfa, dialect, name, policy, markers):
    if name in policy.strict:
        args = _strict_args(nfa, dialect, policy.argument_schemas[name], markers)
    else:
        args = _free_args(nfa, dialect, markers)
    if dialect.kind == "python":
        # `name(` heads a call alternative: byte-level, special spellings
        # inside it are not forced.
        return _cat(
            nfa,
            [
                _split_literal(
                    nfa, (name + "(").encode(), markers, canonical=False
                ),
                args,
                _lit(nfa, b")"),
            ],
        )
    parts = []
    if dialect.call_separator:
        # `sep?` expands to alternatives: after the separator the opening is
        # the forced continuation (canonical), after the skip it is the
        # alternative's head (byte-level).
        sep = dialect.call_separator
        opening = dialect.call_opening(name).encode()
        parts.append(
            _alt(
                nfa,
                [
                    _cat(
                        nfa,
                        [
                            _spelling_frag(
                                nfa, dialect, sep, markers, head_canonical=False
                            ),
                            _spelling_frag(nfa, dialect, opening.decode(), markers),
                        ],
                    ),
                    _spelling_frag(nfa, dialect, opening.decode(), markers, head_canonical=False),
                ],
            )
        )
    else:
        parts.append(
            _spelling_frag(
                nfa, dialect, dialect.call_opening(name), markers, head_canonical=False
            )
        )
    parts += [
        args,
        _spelling_frag(nfa, dialect, dialect.call_close, markers),
    ]
    return _cat(nfa, parts)


def _answer(nfa, spec):
    return _cat(
        nfa,
        [_ws(nfa), _json_value(nfa, spec.response_schema, 0), _ws(nfa)],
    )


def _tools_body(nfa, spec, policy, dialect, markers):
    barriers = _barrier_literals(dialect.barrier) + [dialect.think_end.encode()]

    def text():
        return _embed_dfa(nfa, _not_contains(barriers))

    def ws():
        return _ws(nfa)

    builders = [
        (lambda name=name: _call_frag(nfa, dialect, name, policy, markers))
        for name in policy.schemas
    ]

    def choice():
        return _alt(nfa, [build() for build in builders])

    def call():
        choice_frag = choice()
        if dialect.kind != "python":
            return choice_frag
        inner = choice_frag
        if policy.parallel:
            inner = _cat(
                nfa,
                [
                    choice_frag,
                    _star(nfa, _cat(nfa, [_lit(nfa, b", "), choice()])),
                ],
            )
        return _cat(
            nfa,
            [
                _split_literal(nfa, dialect.call_open.encode(), markers),
                inner,
                _split_literal(nfa, dialect.call_close.encode(), markers),
            ],
        )

    if not builders:
        # tool_choice none: free text, or a JSON answer when one is set.
        return text() if spec.response_schema is None else _answer(nfa, spec)
    if policy.required or spec.response_schema is not None:
        if policy.parallel:
            tail = _cat(
                nfa,
                [_star(nfa, _cat(nfa, [text(), call()])), text()],
            )
        else:
            tail = ws()
        body = _cat(nfa, [ws(), call(), tail])
        if not policy.required:
            body = _alt(nfa, [body, _answer(nfa, spec)])
        return body
    if policy.parallel:
        return _cat(nfa, [_star(nfa, _cat(nfa, [text(), call()])), text()])
    return _cat(nfa, [text(), _opt(nfa, _cat(nfa, [call(), ws()]))])


def _spec_frag(nfa, spec, markers):
    if spec.kind == "json":
        parts = []
        if spec.thinking:
            parts += [
                _embed_dfa(nfa, _not_contains([spec.think_end.encode()])),
                _lit(nfa, spec.think_end.encode(), marker=True),
            ]
        parts += [_ws(nfa), _json_value(nfa, spec.response_schema, 0)]
        return _cat(nfa, parts)
    if spec.kind == "tools":
        policy = spec.policy
        if policy is None:
            raise _Unsupported("tools spec without a policy")
        dialect = policy.dialect or QWEN3_XML
        parts = []
        if spec.thinking:
            barriers = _barrier_literals(dialect.barrier) + [
                dialect.think_end.encode()
            ]
            parts += [
                _embed_dfa(nfa, _not_contains(barriers)),
                _lit(nfa, spec.think_end.encode(), marker=True),
            ]
        parts.append(_tools_body(nfa, spec, policy, dialect, markers))
        return _cat(nfa, parts)
    raise _Unsupported(f"spec kind {spec.kind!r}")


def compile_dfa(spec, special_texts=()):
    """A minimized byte-level CompiledDfa for spec, or None where the spec
    needs the llguidance grammar path.

    `special_texts` are the vocabulary's special-token byte spellings:
    literal text spelling one is protected by marker edges, since canonical
    tokenization lets only that token produce it."""
    markers = list(special_texts)
    for text in (spec.think_end,):
        data = text.encode()
        if data not in markers:
            markers.append(data)
    nfa = _Nfa()
    try:
        start, end = _spec_frag(nfa, spec, markers)
    except _Unsupported:
        return None
    nfa.acc.add(end)
    try:
        rows, acc, marks, textual = _determinize(nfa, start)
        rows, acc, marks, textual = _minimize(rows, acc, marks, textual)
    except _Unsupported:
        return None
    if len(rows) > MAX_DFA_STATES:
        return None
    accepting = _bitset(acc)
    live = _live_states(rows, acc)
    table = array.array("h")
    for row in rows:
        table.extend(row)
    # Edges into a state no walk can accept from read as dead everywhere —
    # _walk's end-state live check, _forced_bytes' live-edge count — so they
    # are pruned once here. The CONSTRAINT_DFA frame ships this table, and
    # the GPU walk's `>= 0` test then matches the host's live check.
    for i, target in enumerate(table):
        if target >= 0 and not live[target >> 3] >> (target & 7) & 1:
            table[i] = -1
    return CompiledDfa(len(rows), table, accepting, live, marks, _bitset(textual))

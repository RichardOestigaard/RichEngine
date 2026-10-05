"""Incremental model-output parsing, output blocks and final answer
validation."""

import re
from dataclasses import dataclass

from jsonschema.exceptions import ValidationError
from referencing.exceptions import Unresolvable

from . import json_codec
from .errors import APIError
from .schema_validation import SchemaEvaluationError
from .tool_schema import (
    CALL_OPEN,
    JSON_TYPES,
    MAX_NAME_LENGTH,
    NAME_SPACE,
    QWEN3_XML,
    THINK_END,
    json_value,
)

TOOL_ARGUMENT_DELTA_CHARS = 16 * 1024
# A value read as JSON nests at most this deep, so that encoding it back never
# exhausts the stack.
MAX_JSON_NESTING = 256
_SURROGATE = re.compile("[\ud800-\udfff]")


def hold_partial(text, *markers):
    """`text` split before its longest end that begins one of `markers`, which
    more text may complete."""
    longest = max(len(marker) for marker in markers)
    for length in range(min(len(text), longest - 1), 0, -1):
        if any(
            length < len(marker) and text.endswith(marker[:length])
            for marker in markers
        ):
            return text[:-length], text[-length:]
    return text, ""


class ReasoningSplitter:
    def __init__(self, thinking, tool_calls=False):
        self.reasoning = thinking
        self.pending = ""
        # Whether the newlines after </think>, which set the answer apart in
        # the chat template's layout of a turn, are still to be dropped.
        self.separator = False
        # Where a call may follow, a call's opening also ends the reasoning
        # and begins the answer.
        self.ends = (THINK_END, CALL_OPEN) if tool_calls else (THINK_END,)

    def put(self, text):
        if not self.reasoning:
            return self._content(text)
        self.pending += text
        ends = [
            (index, marker)
            for marker in self.ends
            if (index := self.pending.find(marker)) >= 0
        ]
        if ends:
            end, marker = min(ends)
            reasoning = self.pending[:end]
            content = self.pending[end:]
            if marker == THINK_END:
                content = content[len(THINK_END) :]
            self.pending = ""
            self.reasoning = False
            self.separator = marker == THINK_END
            output = [("reasoning_content", reasoning)] if reasoning else []
            return output + self._content(content)
        ready, self.pending = hold_partial(self.pending, *self.ends)
        return [("reasoning_content", ready)] if ready else []

    def _content(self, text):
        if self.separator:
            text = text.lstrip("\n")
            if not text:
                return []
            self.separator = False
        return [("content", text)]

    def finish(self):
        if not self.pending:
            return []
        kind = "reasoning_content" if self.reasoning else "content"
        text, self.pending = self.pending, ""
        return [(kind, text)]


_NOT_JSON = object()
# A template that writes a value through Jinja's string filter, as Nex's and
# some of Qwen's do, spells a boolean or null as Python does.
_PYTHON_LITERALS = {"True": True, "False": False, "None": None}


def _json_value(text):
    """The JSON value `text` spells, if it can be written back as JSON: its
    numbers finite, its strings Unicode and its containers nested at most
    MAX_JSON_NESTING deep. Otherwise _NOT_JSON."""
    try:
        value = json_codec.loads(text)
    except (ValueError, RecursionError):
        return _NOT_JSON
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, str):
            if _SURROGATE.search(item):
                return _NOT_JSON
        elif isinstance(item, (dict, list)):
            if depth == MAX_JSON_NESTING:
                return _NOT_JSON
            children = [*item, *item.values()] if isinstance(item, dict) else item
            pending.extend((child, depth + 1) for child in children)
    return value


def convert_value(text, types):
    """A parameter's text as the JSON types its schema declares read it, any
    type where `types` is None: a string as the text, and another value as
    the JSON the text spells when that is of a declared type or when no
    declared type is a string. A boolean or null declared without a string
    may also be spelled as Python spells it. Otherwise the text."""
    value = _json_value(text)
    if value is _NOT_JSON:
        value = _PYTHON_LITERALS.get(text.strip(), _NOT_JSON)
        declared = (
            value is not _NOT_JSON
            and types is not None
            and "string" not in types
            and JSON_TYPES[type(value)] in types
        )
        return value if declared else text
    if types is None or "string" in types:
        kind = JSON_TYPES[type(value)]
        declared = (
            types is None or kind in types or (kind == "integer" and "number" in types)
        )
        return value if declared and kind != "string" else text
    return value


def StreamingToolCallProjector(policy, request_id, structured=False):
    """The output projector for the policy's tool-call dialect."""
    dialect = getattr(policy, "dialect", None) or QWEN3_XML
    if dialect.kind == "python":
        return _PythonCallProjector(policy, request_id, structured, dialect)
    return _XmlToolCallProjector(policy, request_id, structured, dialect)


class _ProjectorBase:
    """Parse a model's tool-call framing as it arrives into OpenAI JSON
    argument deltas, for streamed and complete responses alike.

    Calls read as the chat template lays them out. Names lose the space
    around them; a call that names no function is dropped, other text inside
    a call is dropped, and a repeated parameter keeps its first value, which
    may have streamed. A value converts by the types its tool declares for
    it (convert_value), and one that may only be a string streams as it is
    written. Text outside calls streams as it arrives, after a call as
    before one, and a </think> there is dropped.
    """

    # The state the text after a call's opening enters, and the state a
    # finished call leaves for its close.
    _after_open = "name"
    _after_call = "closing"

    def __init__(self, policy, request_id, structured, dialect):
        self.policy = policy
        self.dialect = dialect
        # The text that opens a call and begins its name: the call's open
        # marker and the prefix its name follows.
        self.call_marker = dialect.call_open + dialect.name_prefix
        # Text outside calls: a call's opening opens one, and a </think>,
        # which a model that called a tool from its reasoning may still
        # write, is dropped.
        self.text_tags = (self.call_marker, THINK_END)
        self.request_id = request_id
        self.pending = ""
        self.state = "output" if structured else "content"
        self.call_index = 0
        # The open call, None while the text names none or names no function.
        self.call_id = None
        self.function_name = None
        # The types by which its declared parameters and any others convert.
        self.parameter_types = {}
        self.other_types = None
        self.parameter_names = set()
        self.argument_fragments = []
        self.parameter_count = 0
        # The open parameter, None while its value is dropped.
        self.parameter_name = None
        self.value_types = None
        self.value_streams = False
        self.value_started = False
        self.value_parts = []
        self.content_fragments = []
        # How many content fragments the stream has published (the rest are
        # whitespace it holds), and whether the text since the start of the
        # output or the last call has shown a visible character yet.
        self.streamed_count = 0
        self.text_visible = False
        self.closed_calls = []
        # Whether the open parameter's value arrived as a CDATA section.
        self.cdata_open = None

    @staticmethod
    def _malformed():
        raise APIError(
            500, "model returned malformed tool output", "invalid_model_output"
        )

    def _literal(self, value):
        if self.pending.startswith(value):
            self.pending = self.pending[len(value) :]
            return True
        if value.startswith(self.pending):
            return False
        self._malformed()

    def _emit_content(self, value, events):
        if not value:
            return
        self.content_fragments.append(value)
        # The chat template sets calls apart from text with whitespace. Hold
        # whitespace that starts the output or follows a call until visible
        # text arrives. Before the first text or after the last, it only
        # frames the calls and is dropped; between two texts it separates
        # them and streams with the later one.
        if not self.text_visible:
            if not value.strip():
                return
            self.text_visible = True
        unsent = self.content_fragments[self.streamed_count :]
        self.streamed_count = len(self.content_fragments)
        events.append(("content", "".join(unsent)))

    def _streamed_content(self):
        return "".join(self.content_fragments[: self.streamed_count])

    def _emit_argument(self, fragment, events):
        if self.call_id is None:
            return
        self.argument_fragments.append(fragment)
        for chunk in argument_deltas(fragment):
            events.append(
                (
                    "tool",
                    {
                        "index": self.call_index,
                        "function": {"arguments": chunk},
                    },
                )
            )

    def _begin_call(self, name, events):
        self.argument_fragments = []
        self.parameter_count = 0
        self.parameter_names = set()
        name = name.strip(NAME_SPACE)
        if not name:
            return
        if not self.streamed_count:
            # Whitespace before the first text only framed the calls.
            self.content_fragments.clear()
        self.function_name = name
        self.call_id = f"call_{self.request_id}_{self.call_index}"
        self.parameter_types, self.other_types = self.policy.parameter_types(name)
        events.append(
            (
                "tool",
                {
                    "index": self.call_index,
                    "id": self.call_id,
                    "type": "function",
                    "function": {"name": name},
                },
            )
        )
        self._emit_argument("{", events)

    def _key(self, name):
        separator = "," if self.parameter_count else ""
        self.parameter_count += 1
        return f"{separator}{json_codec.dumps(name)}:"

    def _begin_parameter(self, name, events):
        name = name.strip(NAME_SPACE)
        # A repeated parameter keeps its first value, which may have streamed;
        # a parameter without a name, or of a call that names no function,
        # keeps none.
        kept = name and name not in self.parameter_names and self.call_id is not None
        self.parameter_names.add(name)
        self.parameter_name = name if kept else None
        self.value_types = self.parameter_types.get(name, self.other_types)
        # A value that may only be a string streams as it is written; any
        # other waits for its end to convert.
        self.value_streams = self.value_types == {"string"}
        self.value_started = False
        self.value_parts = []
        if self.value_streams and self.parameter_name is not None:
            self._emit_argument(self._key(name) + '"', events)
        self.state = "value"

    def _put_value(self, text, events):
        if self.parameter_name is None:
            return
        if not self.value_started:
            if not text:
                return
            self.value_started = True
            text = text.removeprefix("\n")
        if not text:
            return
        if self.value_streams:
            self._emit_argument(json_codec.dumps(text)[1:-1], events)
        else:
            self.value_parts.append(text)

    def _end_parameter(self, text, events):
        if self.parameter_name is not None:
            if not self.value_started:
                self.value_started = True
                text = text.removeprefix("\n")
            self._put_value(text, events)
            if self.value_streams:
                self._emit_argument('"', events)
            else:
                value = convert_value("".join(self.value_parts), self.value_types)
                self._emit_argument(
                    self._key(self.parameter_name) + json_codec.dumps(value), events
                )
        self.parameter_name = None
        self.value_parts = []
        self.state = "arguments"

    def _finish_call(self, events):
        self._emit_argument("}", events)
        if self.call_id is not None:
            self.closed_calls.append(
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": self.function_name,
                        "arguments": "".join(self.argument_fragments),
                    },
                }
            )
            self.call_index += 1
        self.call_id = None
        self.function_name = None
        self.argument_fragments = []
        self.text_visible = False
        self.state = self._after_call

    def _next_tag(self, *tags):
        """The tag of `tags` that the pending text spells first, and where."""
        found = [(index, tag) for tag in tags if (index := self.pending.find(tag)) >= 0]
        return min(found) if found else (-1, None)

    def _name(self, delimiter):
        """The name the pending text spells up to `delimiter`, taken off the
        text; "" for a longer one, and None while it may yet end."""
        end = self.pending.find(delimiter, 0, MAX_NAME_LENGTH + 1)
        if end < 0:
            return None if len(self.pending) <= MAX_NAME_LENGTH else ""
        name, self.pending = self.pending[:end], self.pending[end + 1 :]
        return name

    def _output_state(self, events):
        """The "output" and "json" states of a structured answer, which is
        one JSON value or tool calls: once JSON starts, call spellings
        inside its strings are just data. True to keep reading, False while
        more text must arrive, None in any other state."""
        if self.state == "output":
            first = self.pending.lstrip()
            if not first:
                return False
            self.state = "content" if first.startswith("<") else "json"
        if self.state == "json":
            self._emit_content(self.pending, events)
            self.pending = ""
            return False
        return None

    def _content_state(self, events):
        """The "content" state every dialect shares: emit text up to the
        next call's opening, skip a call separator, drop a </think>. False
        while only a marker's partial prefix remains."""
        separator = self.dialect.call_separator
        if separator and self.pending.startswith(separator):
            self.pending = self.pending[len(separator) :]
            return True
        start, tag = self._next_tag(*self.text_tags)
        if tag is None:
            markers = self.text_tags + ((separator,) if separator else ())
            ready, self.pending = hold_partial(self.pending, *markers)
            self._emit_content(ready, events)
            return False
        self._emit_content(self.pending[:start], events)
        self.pending = self.pending[start + len(tag) :]
        if tag == self.call_marker:
            self.state = self._after_open
        return True

    def _drain(self, events):
        """Close whatever complete output ended inside."""

    def finish(self, incomplete):
        """The content and calls of the output, and the events the stream
        still owes for what put() held back.

        Output cut at the token limit keeps an open call with the arguments
        it has, and no call whose name it cut; other output that ends inside
        a call closes the call, and what it held back after a value could
        only begin the value's close. Cut output with a call has the content
        the stream published, which leaves out whitespace that no visible
        text has followed since the start or the last call. Otherwise the
        content is the text outside calls without whitespace that only frames
        them and, when cut, a trailing partial tag."""
        events = []
        if not incomplete:
            self._drain(events)
        if self.state in ("closing", "call_sep"):
            self.pending = ""
            self.state = "content"
        if (
            self.state in ("content", "output", "json")
            and self.pending
            and not (
                incomplete
                and any(tag.startswith(self.pending) for tag in self.text_tags)
            )
        ):
            self.content_fragments.append(self.pending)
        elif self.closed_calls:
            # Whitespace held after the last text only framed the calls.
            del self.content_fragments[self.streamed_count :]
        self.pending = ""
        calls = list(self.closed_calls)
        if self.call_id is not None:
            calls.append(
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": self.function_name,
                        "arguments": "".join(self.argument_fragments),
                    },
                }
            )
        streamed = self._streamed_content()
        content = streamed if calls and incomplete else "".join(self.content_fragments)
        if unsent := content[len(streamed) :]:
            events.append(("content", unsent))
        return content, calls, events


_CDATA_OPEN = "<![CDATA["


class _XmlToolCallProjector(_ProjectorBase):
    """The projector of an XML dialect: calls are elements whose parameters
    are elements wrapping a value that ends at the parameter's close where
    the next tag follows."""

    def __init__(self, policy, request_id, structured, dialect):
        super().__init__(policy, request_id, structured, dialect)
        # Where a value ends: the close the template writes after it, then
        # the next parameter, the function's close, or the call's close a
        # model may write without the function's. A value may hold the tags
        # in any other order.
        follows = (dialect.param_open, dialect.body_close) + (
            (dialect.block_close,) if dialect.block_close else ()
        )
        self.value_ends = tuple(dialect.param_close + tag for tag in follows)
        self.argument_tags = follows

    def _drain(self, events):
        if self.state == "name":
            name = self.pending if len(self.pending) <= MAX_NAME_LENGTH else ""
            self.pending = ""
            self._begin_call(name, events)
            self.state = "arguments"
        if self.state == "value":
            self.pending = ""
            self._end_parameter("", events)
        if self.state in ("arguments", "parameter"):
            self.pending = ""
            self._finish_call(events)

    def put(self, text):
        self.pending += text
        events = []
        while self.pending:
            handled = self._output_state(events)
            if handled is not None:
                if not handled:
                    break
                continue
            if self.state == "content":
                if not self._content_state(events):
                    break
                continue
            if self.state == "name":
                if (name := self._name(self.dialect.name_close[0])) is None:
                    break
                self._begin_call(name, events)
                self.state = "arguments"
                continue
            if self.state == "arguments":
                # Text between parameters is dropped, but for a tag it may
                # begin.
                start, tag = self._next_tag(*self.argument_tags)
                if tag is None:
                    self.pending = hold_partial(self.pending, *self.argument_tags)[1]
                    break
                self.pending = self.pending[start + len(tag) :]
                if tag == self.dialect.param_open:
                    self.state = "parameter"
                else:
                    self._finish_call(events)
                    if (
                        not self.dialect.block_close
                        or tag == self.dialect.block_close
                    ):
                        self.state = "content"
                continue
            if self.state == "parameter":
                if (name := self._name(self.dialect.param_name_close[0])) is None:
                    break
                self._begin_parameter(name, events)
                continue
            if self.state == "value":
                ends = self.value_ends
                skip = len(self.dialect.param_close)
                if (
                    self.dialect.cdata
                    and not self.value_started
                    and not self.value_parts
                ):
                    # A value may open as a CDATA section; wait until that
                    # much text has arrived or ruled it out.
                    if self.cdata_open is None:
                        if self.pending.startswith(_CDATA_OPEN):
                            self.pending = self.pending[len(_CDATA_OPEN) :]
                            self.cdata_open = True
                        elif _CDATA_OPEN.startswith(self.pending):
                            break
                        else:
                            self.cdata_open = False
                    if self.cdata_open:
                        ends = tuple("]]>" + end for end in ends)
                        skip += 3
                start, tag = self._next_tag(*ends)
                if tag is None:
                    held = hold_partial(self.pending, *ends)[1]
                    self._put_value(
                        self.pending[: len(self.pending) - len(held)], events
                    )
                    self.pending = held
                    break
                self._end_parameter(self.pending[:start], events)
                # The tag after the close stays for the arguments to read.
                self.pending = self.pending[start + skip :]
                self.cdata_open = None
                continue
            if self.state == "closing":
                # After the function's close, the template closes the call's
                # block.
                rest = self.pending.lstrip(NAME_SPACE)
                if rest.startswith(self.dialect.block_close):
                    self.pending = rest[len(self.dialect.block_close) :]
                elif self.dialect.block_close.startswith(rest):
                    break
                self.state = "content"
        return events


class _PythonCallProjector(_ProjectorBase):
    """The projector of a python-call dialect: ``[name(arg='v', ...)]``
    between the call-list tokens. Bare values read to the next ``, `` or
    ``)`` at bracket depth zero; a quoted string's escapes decode as it
    reads."""

    _after_open = "call_started"
    _after_call = "call_sep"

    def _begin_parameter(self, name, events):
        name = name.strip(NAME_SPACE)
        kept = name and name not in self.parameter_names and self.call_id is not None
        self.parameter_names.add(name)
        self.parameter_name = name if kept else None
        self.value_started = True
        self.value_parts = []

    def _end_parameter_value(self, value, events):
        if self.parameter_name is not None:
            self._emit_argument(
                self._key(self.parameter_name) + json_codec.dumps(value), events
            )
        self.parameter_name = None
        self.value_parts = []

    def _begin_argument(self, events):
        # The name ends at "="; "(", ")" or "," first is malformed.
        equals = self.pending.find("=")
        earlier = [
            found
            for character in "(),"
            if (found := self.pending.find(character)) >= 0
        ]
        if equals < 0:
            if earlier:
                self._malformed()
            return False
        if earlier and min(earlier) < equals:
            self._malformed()
        name = self.pending[:equals]
        self.pending = self.pending[equals + 1 :]
        self._begin_parameter(name, events)
        self.state = "argument_value"
        return True

    _ESCAPES = {"\\": "\\", "'": "'", "n": "\n", "r": "\r"}

    def _finish_string(self, events):
        """Read a quoted string's characters, decoding its escapes, to its
        close quote."""
        index = 0
        while index < len(self.pending):
            character = self.pending[index]
            if character == "'":
                value = "".join(self.value_parts)
                self.pending = self.pending[index + 1 :]
                self._end_parameter_value(value, events)
                self.state = "argument_sep"
                return True
            if character == "\\":
                if index + 1 >= len(self.pending):
                    break
                escaped = self._ESCAPES.get(self.pending[index + 1])
                if escaped is None:
                    self._malformed()
                self.value_parts.append(escaped)
                index += 2
                continue
            self.value_parts.append(character)
            index += 1
        self.pending = self.pending[index:]
        return False

    def _finish_value(self, events):
        """Read a bare value to its ``, `` or ``)`` delimiter at depth zero,
        honoring JSON strings and containers, then emit it."""
        depth = 0
        quote = None
        index = 0
        while index < len(self.pending):
            character = self.pending[index]
            if quote is not None:
                if character == "\\":
                    index += 1
                elif character == quote:
                    quote = None
            elif character in "'\"":
                quote = character
            elif character in "[{":
                depth += 1
            elif character in "]}":
                depth -= 1
            elif character in ",)" and depth == 0:
                break
            index += 1
        else:
            return False
        token = self.pending[:index]
        self.pending = self.pending[index:]
        self._end_parameter_value(_python_call_value(token), events)
        self.state = "argument_sep"
        return True

    def _drain(self, events):
        if self.state == "call_started" and "(" not in self.pending:
            # The cut name never began a call's arguments.
            self.pending = ""
        if self.state == "argument_string":
            # A quoted string that never closed keeps the text it decoded.
            self._end_parameter_value("".join(self.value_parts), events)
            self.pending = ""
        elif self.state == "argument_value":
            token, self.pending = self.pending, ""
            self._end_parameter_value(_python_call_value(token), events)
        if self.state in ("argument_head", "argument_sep") and self.call_id is not None:
            self._finish_call(events)

    def put(self, text):
        self.pending += text
        events = []
        while self.pending:
            handled = self._output_state(events)
            if handled is not None:
                if not handled:
                    break
                continue
            if self.state == "content":
                if not self._content_state(events):
                    break
                continue
            if self.state == "call_started":
                # ", " joins calls; the space may arrive after the comma.
                stripped = self.pending.lstrip(" ")
                if stripped != self.pending:
                    self.pending = stripped
                    continue
                if (name := self._name("(")) is None:
                    break
                self._begin_call(name, events)
                self.state = "argument_head"
                continue
            if self.state == "argument_head":
                stripped = self.pending.lstrip(" ")
                if stripped != self.pending:
                    self.pending = stripped
                    continue
                if self.pending.startswith(")"):
                    self.pending = self.pending[1:]
                    self._finish_call(events)
                    continue
                if not self._begin_argument(events):
                    break
                continue
            if self.state == "argument_value":
                # A quote opens a string; anything else is a bare literal or
                # JSON container read to its delimiter.
                if self.pending.startswith("'"):
                    self.pending = self.pending[1:]
                    self.state = "argument_string"
                    continue
                if not self._finish_value(events):
                    break
                continue
            if self.state == "argument_string":
                if not self._finish_string(events):
                    break
                continue
            if self.state == "argument_sep":
                if self.pending.startswith(","):
                    self.pending = self.pending[1:]
                    self.state = "argument_head"
                    continue
                if self.pending.startswith(")"):
                    self.pending = self.pending[1:]
                    self._finish_call(events)
                    continue
                if ",)".find(self.pending[0]) < 0:
                    self._malformed()
                break
            if self.state == "call_sep":
                if self.pending.startswith(","):
                    self.pending = self.pending[1:]
                    self.state = "call_started"
                    continue
                if self._literal(self.dialect.call_close):
                    self.state = "content"
                    continue
                break
        return events


def _python_call_value(token):
    """A bare argument value of a python-call dialect: the template's scalar
    spellings, or JSON for containers and quoted strings."""
    token = token.strip()
    if token == "True":
        return True
    if token == "False":
        return False
    if token == "None":
        return None
    return json_value(token)


def argument_deltas(arguments):
    # Keep individual SSE frames bounded even when a tool has a large string
    # argument. Callers preserve fragment order.
    for offset in range(0, len(arguments), TOOL_ARGUMENT_DELTA_CHARS):
        yield arguments[offset : offset + TOOL_ARGUMENT_DELTA_CHARS]


@dataclass(slots=True)
class Block:
    """One block of output: the reasoning, a run of text or a tool call."""

    kind: str  # "reasoning", "text" or "tool"
    # The text streamed into the block, or a call's argument fragments.
    parts: list[str]
    call_id: str | None = None
    name: str | None = None
    # "completed" or "incomplete" once the block closes.
    status: str = "in_progress"

    @property
    def text(self):
        return "".join(self.parts)


class BlockSequencer:
    """Output in order as blocks, one open at a time: the reasoning, each run
    of text, each tool call. A new kind or a tool header closes the open
    block. Messages and Responses render the same sequence, streamed or not;
    each callback receives a block with its position."""

    def __init__(self, on_open=None, on_delta=None, on_close=None):
        self.on_open = on_open
        self.on_delta = on_delta
        self.on_close = on_close
        self.blocks = []
        self.open = None

    def _start(self, block):
        self._close("completed")
        self.blocks.append(block)
        self.open = block
        if self.on_open is not None:
            self.on_open(len(self.blocks) - 1, block)

    def _append(self, text):
        self.open.parts.append(text)
        if self.on_delta is not None:
            self.on_delta(len(self.blocks) - 1, self.open, text)

    def _close(self, status):
        block, self.open = self.open, None
        if block is None:
            return
        block.status = status
        if self.on_close is not None:
            self.on_close(len(self.blocks) - 1, block)

    def text(self, field, text):
        """Collected text, in field "reasoning_content" or "content"."""
        kind = "reasoning" if field == "reasoning_content" else "text"
        if self.open is None or self.open.kind != kind:
            self._start(Block(kind, []))
        self._append(text)

    def tool(self, delta):
        """A projected tool delta: a call's header or its arguments."""
        function = delta["function"]
        if "name" in function:
            self._start(Block("tool", [], call_id=delta["id"], name=function["name"]))
        if arguments := function.get("arguments"):
            self._append(arguments)

    def finish(self, incomplete, reasoning_open):
        """Close the open block, cut if the output was cut inside it, and end
        with an empty text block when there is neither text nor a call."""
        if self.open is not None:
            cut = incomplete and (self.open.kind != "reasoning" or reasoning_open)
            self._close("incomplete" if cut else "completed")
        if all(block.kind == "reasoning" for block in self.blocks):
            self._start(Block("text", []))
            self._close("incomplete" if incomplete else "completed")
        return self.blocks


def _validate(validator, value):
    try:
        validator.validate(value)
    except AttributeError as error:
        # referencing's draft 3 crawls the keys of an extends object as schemas
        # whenever a reference lookup scans the document for identifiers.
        raise SchemaEvaluationError(
            "schema reference could not be evaluated"
        ) from error


def validate_response_content(content, validator):
    if validator is None:
        return
    try:
        value = json_codec.loads(content)
        _validate(validator, value)
    except SchemaEvaluationError as error:
        raise APIError(500, str(error), "output_validation_failed") from error
    except (ValueError, ValidationError, Unresolvable, RecursionError) as error:
        raise APIError(
            500,
            f"model returned invalid structured output: {error}",
            "invalid_model_output",
        ) from error

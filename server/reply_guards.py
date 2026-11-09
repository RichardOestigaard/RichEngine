"""Reply-level guards: token loops, repeated calls and the steering a
repeated call is re-decoded with.

A small model can repeat a token block forever or call the same tool with
the same arguments turn after turn; a grammar cannot see either — the loop
is valid output and each call parses on its own. The server watches the
committed token and text stream with these pure functions, stops a reply
that trips them early, and re-decodes a reply that repeats the previous
turn's call with a steering prefix and a banned-tool grammar, up to
STEER_ATTEMPTS times, before ending the turn with a message instead of
running to the token limit.

Ported from Locals (Server._guard, _looping, _line_looping, _past_grace
and the steering loop): the thresholds and the texts are the same; the
call data comes from the output projector rather than re-parsed text."""

from . import json_codec

# Tokens a repetition that may be wanted goes on after it looks like a
# loop (detect): a loop goes on to the token limit, a repetition asked
# for ends. Cut at once, "the line 50 times" had 13 lines, and a write
# call of 40 identical lines streamed arguments that never closed.
LOOP_GRACE = 1024

# A reply decoded again after it repeated the previous call gets this much
# output; the steered reply should be a different call, not an essay (Gemma
# 4 wrote 4,096 tokens of prose after the steering text).
STEERED_MAX_TOKENS = 2048

# Re-decodes a turn gets after its first call repeated the previous turn's:
# the first steering without bans, then up to two more banning the repeated
# tools. Still repeating after them, the turn ends with the stop text.
STEER_ATTEMPTS = 3

# The tool calls one reply may make. Locals' default; enough for a turn's
# work, small enough to catch a call loop early.
MAX_CALLS = 6


def looping(tokens, reps=3, max_period=256, min_span=24, short_span=128):
    """True if the reply ends in one block of tokens repeated `reps` times
    (any block length up to `max_period`; the repeats spanning at least
    `min_span` tokens): a small model saying the same sentences over and
    over, which no grammar can see. Gemma 4 E4B looped on a 45-token cycle
    of "I will execute the edit." lines until the token limit. A block
    shorter than 8 tokens must repeat over at least `short_span` tokens:
    data repeats a few tokens on purpose ("[0, 0, 0, ...]", a grid of
    dots), a loop goes on until the limit."""
    n = len(tokens)
    if n < min_span:
        return False
    # A short block repeated is also periodic at its multiples: decide by
    # the smallest period.
    for period in range(1, 8):
        if n >= 32 and tokens[n - 32 + period :] == tokens[n - 32 : n - period]:
            span = max(reps * period, min_span, short_span)
            return (
                span <= n
                and tokens[n - span + period :] == tokens[n - span : n - period]
            )
    last = tokens[-1]
    for period in range(8, min(max_period, n // reps) + 1):
        if tokens[n - 1 - period] != last:
            # A period must bring the last token back.
            continue
        span = max(reps * period, min_span, short_span if period < 8 else 0)
        if span <= n and tokens[n - span + period :] == tokens[n - span : n - period]:
            # The last `span` tokens have period `period`.
            return True
    return False


def line_looping(text, window=30, distinct=6, min_len=12):
    """True if the last `window` sentence-length lines of a reply say at
    most `distinct` different things: a model rephrasing the same intention
    without acting (Gemma 4, thinking: "I'll use the read tool. / Let's
    read the file. / I'll proceed with the read call." for 4,000 tokens),
    which an exact repeat check cannot see because the order of the lines
    changes."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= min_len]
    return len(lines) >= window and len(set(lines[-window:])) <= distinct


def past_grace(flag, n, state):
    """Whether a reply that looks like a loop has gone on looking like one
    for LOOP_GRACE tokens (`n` tokens so far; `state.loop_at` where it
    began, cleared when the repetition stops)."""
    if state is None:
        return flag
    if not flag:
        state.loop_at = None
        return False
    if getattr(state, "loop_at", None) is None:
        state.loop_at = n
    return n - state.loop_at >= LOOP_GRACE


def _text_of(content):
    """A message's content as one text: the string itself, or the text
    parts of a content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def arguments_key(arguments):
    """One call's arguments as the key the previous turn's calls and this
    turn's compare by: a parsed value canonically respelled (its object
    keys sorted), or the text itself where it spells no JSON."""
    if isinstance(arguments, str):
        try:
            arguments = json_codec.loads(arguments)
        except ValueError:
            return arguments
    if isinstance(arguments, dict):
        return json_codec.encode_compact(arguments, sort_keys=True).decode()
    try:
        return json_codec.dumps(arguments)
    except (TypeError, ValueError, RecursionError):
        return str(arguments)


def call_key(call):
    """(name, arguments) an OpenAI-shaped tool call compares by."""
    function = call.get("function") or {}
    return function.get("name"), arguments_key(function.get("arguments"))


def previous_calls(messages):
    """(name, arguments) of the tool calls in the last assistant turn ->
    the text of their results. The turn before a fresh user message made
    no calls this turn could repeat, so nothing is compared then."""
    results = {}
    out = {}
    for message in reversed(messages or []):
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = _text_of(message.get("content"))
        elif message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                out[call_key(call)] = results.get(call.get("id"), "")
            return out
        elif message.get("role") == "user":
            return out
    return out


def first_repeat(calls, prev):
    """The index of the first call identical to one of the previous turn's
    (name and arguments), or None."""
    for index, call in enumerate(calls):
        if call_key(call) in prev:
            return index
    return None


def dedupe(calls):
    """`calls` without later calls identical to an earlier one (name and
    argument text), order kept: a reply may name the same tool twice on
    purpose; the same call again is a loop."""
    seen, out = set(), []
    for call in calls:
        function = call.get("function") or {}
        key = (function.get("name"), function.get("arguments"))
        if key not in seen:
            seen.add(key)
            out.append(call)
    return out


def detect(
    tokens,
    text,
    *,
    closed_calls=(),
    in_call=False,
    tools=False,
    max_calls=MAX_CALLS,
    state=None,
):
    """Why a reply should stop early: enough tool calls, a call repeated
    within the turn, or a token loop. `tokens` are the ids generated so far
    (None where they cannot be read); `text` the generated content text
    when a line-loop scan is due, else None. `closed_calls` are the calls
    the output projector has finished, `in_call` whether output is inside
    one now, and `state` the per-decode grace bookkeeping (a ReplyGuard, or
    anything with a loop_at attribute; None checks without grace).

    A reply may also repeat on purpose: without tools ("write it ten
    times") and inside a call's arguments (a file of identical lines), so
    there it takes many more repeats to count as a loop, and then
    LOOP_GRACE tokens more of it. A file written through a call repeats
    lines on purpose (a grid, a table, test data): 16 repeats over 256
    tokens or more (8 over 128 cut an 8 x 8 board out of a write call, and
    the call was dropped)."""
    if len(closed_calls) >= max_calls:
        return "max_calls"
    keys = [call_key(call) for call in closed_calls]
    if len(keys) != len(set(keys)):
        return "duplicate_call"
    if in_call:
        if tokens is None:
            return None
        flag = looping(tokens, reps=16, min_span=256, short_span=256)
        return "loop_in_call" if past_grace(flag, len(tokens), state) else None
    if tools:
        if tokens is not None and looping(tokens):
            return "loop"
        if text is not None and line_looping(text):
            return "loop"
        return None
    if tokens is None:
        return None
    flag = looping(tokens, reps=12, min_span=96, short_span=256)
    return "loop" if past_grace(flag, len(tokens), state) else None


def steer_text(calls, prev):
    """The text a turn re-decoded after repeating the previous call starts
    with: what the call was and that its result is already above."""
    for call in calls:
        key = call_key(call)
        if key in prev:
            function = call["function"]
            result = " ".join(prev[key].split())[:160]
            return (
                f"The {function['name']} call above was already made and "
                f"returned: {result!r}. Repeating it would give the same "
                "result, so I will do something different now:\n\n"
            )
    return ""


def stop_text(calls):
    """The message a turn ends with when it still repeats the previous
    call and no steering is left: a question for the user rather than a
    loop (once a headless agent took "I am stuck" for the end of its task,
    the turn decodes more steering first — this is the last resort)."""
    function = calls[0]["function"]
    return (
        f"I am stuck: I keep wanting to repeat the same {function['name']} "
        f"call ({function['arguments'][:200]}), whose result is already "
        "above. Please tell me how to proceed (for example, which command "
        "or file to use)."
    )


class ReplyGuard:
    """Per-request state the loop and repeated-call guards keep across the
    decodes of one turn: the previous turn's calls to compare against, the
    tools a steered reply may no longer name, where a repetition that may
    be wanted began, and the tool deltas held while a call that could
    repeat the previous turn's is still open."""

    def __init__(self, job):
        # (name, arguments) -> result text of the previous assistant turn's
        # calls; the job's prepared prompt carries them.
        self.prev = getattr(job, "prev_calls", None) or {}
        self.prev_names = {name for name, _arguments in self.prev}
        # The response format a steered reply may answer under instead:
        # its schema, or its regex pattern (guided_regex), whichever the
        # request constrained output with.
        self.schema = getattr(job, "reply_schema", None)
        self.regex = getattr(job, "reply_regex", None)
        # Tools a steered reply may not call: its call to them repeated the
        # previous turn's.
        self.banned = set()
        # Steered re-decodes so far this turn.
        self.attempts = 0
        # The steering text the current decode was started with, and
        # whether it sits inside the prompt's open think block.
        self.prefix = None
        self.prefix_reasoning = False
        self.begin_attempt()

    def begin_attempt(self):
        """Reset the per-decode state; the ban and the attempt count carry
        over between decodes of the same turn."""
        # Where a repetition that may be wanted began (past_grace).
        self.loop_at = None
        # The generated content text this decode (line_looping's input).
        self.text = ""
        # How far the line-loop scan has run; it rescans at most every 512
        # new characters, not per event.
        self.line_scan_at = 0
        # Closed calls already compared with the previous turn's.
        self.vetted = 0
        # The index of the call whose deltas wait on vetting, and those
        # deltas: a call naming a tool the previous turn called could be a
        # repeat, so it is held until it closes and compares (calls already
        # sent cannot be steered away from).
        self.hold = None
        self.buffered = []
        # The first call identical to a previous-turn call, once found.
        self.repeat_at = None

"""Prepare API requests for generation and manage Responses history."""

import hashlib
import json
import os
import re
import secrets
import threading
from collections import OrderedDict
from contextlib import contextmanager
import queue
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import count
from pathlib import Path

from . import constraints, decisions, json_codec, judgments
from . import dfa as dfa_compiler
from . import images as image_input
from . import protocol as wire
from .api_shapes import (
    IMAGE_PAD_TOKEN,
    canonical_responses_input,
    normalize_messages,
    responses_to_chat_body,
    template_messages,
)
from .backend import CacheInfo, Job, remaining_request_time
from .chat_templates import (
    LATER_SYSTEM_UNSUPPORTED,
    RESERVED_TEMPLATE_KWARGS,
    render_chat_template,
    template_options,
)
from .diagnostics import print_status
from .errors import APIError, ContextLengthError
from .latency import LatencyMetrics
from .metrics import is_finite_number
from .serve_options import REASONING_EFFORTS, parse_served_model_name
from .tokenization import PromptTokenizer
from .tool_schema import (
    QWEN3_XML,
    THINK_END,
    THINK_END_TOKEN_ID,
    ToolPolicy,
    bind_dialect,
    dialect_named,
    json_grammar,
    normalize_response_format,
    normalize_tools,
    tool_grammar,
)

PREPARATION_WAIT_SECONDS = 30.0

# The priority a request names, by its lowercase wire name.
_PRIORITIES = {priority.name.lower(): priority for priority in wire.RequestPriority}


MIN_FLOAT32_SUBNORMAL = float.fromhex("0x1p-149")
FLOAT32_MAX = float.fromhex("0x1.fffffep127")
# vLLM raises a nonzero temperature below this to it (_MAX_TEMP in
# vllm/sampling_params.py); RichEngine does the same.
MIN_SAMPLING_TEMPERATURE = 0.01

# The sampling numbers a request may set: each with its default (Qwen's
# generation config for temperature and top_p, and otherwise the value that
# leaves the logits unchanged, as in vLLM), the values the engine accepts and
# how a 400 names them. A nonzero top_p or repetition_penalty must not round
# to zero in the engine's float32, which would make the value invalid.
SAMPLING_NUMBERS = {
    "temperature": (
        1.0,
        lambda value: 0 <= value <= 2,
        "a number in [0, 2]",
    ),
    "top_p": (
        0.95,
        lambda value: MIN_FLOAT32_SUBNORMAL <= value <= 1,
        "a number in (0, 1]",
    ),
    "presence_penalty": (0.0, lambda value: -2 <= value <= 2, "a number in [-2, 2]"),
    "frequency_penalty": (0.0, lambda value: -2 <= value <= 2, "a number in [-2, 2]"),
    "repetition_penalty": (
        1.0,
        lambda value: MIN_FLOAT32_SUBNORMAL <= value <= FLOAT32_MAX,
        "a positive number",
    ),
    "min_p": (0.0, lambda value: 0 <= value <= 1, "a number in [0, 1]"),
}
# top_k's default, from Qwen's generation config as temperature's and top_p's
# are. With it, the table covers every sampling option the frame carries.
TOP_K_DEFAULT = 20
assert set(SAMPLING_NUMBERS) | {"top_k"} == set(wire.SAMPLING_FIELDS)


def _preparation_checkpoint(deadline, disconnected):
    """A check that a request's preparation may go on: its deadline has not
    passed, and its client has not left, where `disconnected` can tell."""

    def checkpoint():
        remaining_request_time(deadline)
        if disconnected is not None and disconnected():
            raise ConnectionResetError("client disconnected during preparation")

    return checkpoint


def _drop_nulls(body, extras):
    """The body without the null values of the sampling options and of the
    endpoint's other nullable fields, which then take their defaults."""
    nullable = {*wire.SAMPLING_FIELDS, *extras}
    return {
        key: value
        for key, value in body.items()
        if value is not None or key not in nullable
    }


def _request_metadata(body):
    """The request's metadata map for the response object to echo: at most
    16 short string pairs, as the OpenAI APIs bound it."""
    metadata = body.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or len(metadata) > 16
        or any(
            not isinstance(key, str)
            or len(key) > 64
            or not isinstance(value, str)
            or len(value) > 512
            for key, value in metadata.items()
        )
    ):
        raise APIError(
            400, "metadata must be an object of up to 16 short string pairs"
        )
    return metadata


# Text completions' output budget when max_tokens is omitted: OpenAI's
# default for the endpoint, which vLLM and SGLang also use.
COMPLETION_DEFAULT_MAX_TOKENS = 16


# Volatile-field prompt normalization: agent harnesses put session-volatile
# values (working directory, dates, environment details) early in the system
# prompt, which ends the shared prefix the KV cache can reuse there. Moving
# the detected lines and tagged blocks to the end of the system message keeps
# the stable prefix contiguous. RICHENGINE_NO_PROMPT_NORMALIZE=1 restores the
# request's original order.
_VOLATILE_FIELD_MARKERS = (
    "working directory",
    "current directory",
    "cwd",
    "today's date",
    "current date",
    "date:",
    "timestamp",
    "session",
    "environment",
    "os:",
    "platform:",
    "runtime",
    "model:",
)

# A word marker is a field only where punctuation, space or the line's end
# follows it; the ":" markers spell their own boundary.
_VOLATILE_FIELD_BOUNDARY = frozenset(" \t:=(-")

# XML-ish blocks agents wrap volatile context in.
_VOLATILE_TAGS = ("env", "environment", "context", "session", "runtime")
_VOLATILE_TAG_OPEN = re.compile(r"<([a-z_]+)[\s>/]")

# Fenced blocks are volatile only when their info string says they hold
# volatile values; any other fence is stable as a whole, so a volatile-looking
# line inside quoted code never moves.
_VOLATILE_FENCE_INFO = frozenset((*_VOLATILE_TAGS, "dotenv"))


def _volatile_label(stripped):
    """Whether a stripped, lowercased line opens a volatile field, allowing a
    markdown header's leading '#'s."""
    text = stripped.lstrip("#").lstrip()
    for marker in _VOLATILE_FIELD_MARKERS:
        if text.startswith(marker) and (
            marker[-1] in ":"
            or len(text) == len(marker)
            or text[len(marker)] in _VOLATILE_FIELD_BOUNDARY
        ):
            return True
    return False


def _volatile_tag(stripped):
    """The tag whose block `stripped` opens, when it is a volatile one."""
    match = _VOLATILE_TAG_OPEN.match(stripped)
    return match.group(1) if match and match.group(1) in _VOLATILE_TAGS else None


def _fence(stripped):
    return stripped[:3] if stripped[:3] in ("```", "~~~") else None


def _volatile_continuation(line):
    """Whether `line` continues a volatile field's value: an indented or
    bulleted line."""
    stripped = line.strip()
    return bool(stripped) and (
        line[0] in " \t" or stripped.startswith(("-", "*", "•"))
    )


def _absorb_block(lines, index, out, close):
    """Append the tagged block opening at `index` to `out`, through its close
    or the text's end."""
    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1
        if close in line.lower():
            break
    return index


def _absorb_fence(lines, index, out, fence):
    """Append the fenced block opening at `index` to `out`, through its
    closing fence or the text's end."""
    out.append(lines[index])
    index += 1
    while index < len(lines):
        line = lines[index]
        out.append(line)
        index += 1
        if _fence(line.strip()) == fence:
            break
    return index


def _move_volatile_lines(content):
    """`content` with its volatile lines and blocks moved to the end, order
    preserved; `content` itself when nothing matches or nothing stays."""
    if not content.strip():
        return content
    trailing = content.endswith("\n")
    lines = content.split("\n")
    if trailing:
        lines.pop()
    stable, volatile = [], []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip().lower()
        tag = _volatile_tag(stripped)
        if tag is not None:
            index = _absorb_block(lines, index, volatile, f"</{tag}>")
            continue
        fence = _fence(stripped)
        if fence is not None:
            info = stripped[3:].strip()
            index = _absorb_fence(
                lines,
                index,
                volatile
                if info and info.split(maxsplit=1)[0] in _VOLATILE_FENCE_INFO
                else stable,
                fence,
            )
            continue
        if _volatile_label(stripped):
            volatile.append(line)
            index += 1
            while index < len(lines) and _volatile_continuation(lines[index]):
                volatile.append(lines[index])
                index += 1
            continue
        stable.append(line)
        index += 1
    if not stable or not volatile:
        return content
    result = "\n".join(stable + volatile)
    return result + "\n" if trailing else result


def move_volatile_system_fields(messages):
    """`messages` with detected session-volatile blocks moved to the end of
    each system message, so the stable prompt prefix stays contiguous for the
    KV cache. The same list when nothing matches or normalization is off."""
    if os.environ.get("RICHENGINE_NO_PROMPT_NORMALIZE") == "1":
        return messages
    moved = None
    for index, message in enumerate(messages):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        normalized = _move_volatile_lines(content)
        if normalized != content:
            if moved is None:
                moved = list(messages)
            moved[index] = {**message, "content": normalized}
    return messages if moved is None else moved


# What follows the leading system prompt and tools in the probe that finds
# their shared prefix: a first turn no request's own is likely to begin with.
SHARED_PREFIX_PROBE = {"role": "user", "content": "⁣"}


# A stable marker lets repeated image requests reuse the compiled template.
IMAGE_RENDER_MARKER = f"__richengine_image_{secrets.token_hex(16)}__"


def _generation_prompt(
    probed,
    rendered,
    tokens,
    *,
    think_open="<think>",
    think_end=THINK_END,
    think_flag=None,
):
    """Whether generation opens a think block, and how many of the prompt's
    last tokens the generation prompt is (zero where they differ from its
    tokens). `probed` is its text and tokens, as the startup probe found
    them for the request's template options, and the rendered prompt must
    end with that text.

    Generation thinks where the prompt ends inside an open think block, or
    where the model writes the block's open itself and the template marks
    the prompt with the family's thinking flag (Gemma 4's think token in
    its system turn)."""
    text, ids = probed
    if not text or not rendered.endswith(text):
        raise APIError(
            400, "chat template must end with an assistant generation prefix"
        )
    count = len(ids)
    thinking = text.rfind(think_open) > text.rfind(think_end) or (
        think_flag is not None and think_flag in rendered
    )
    return (
        thinking,
        count if count < len(tokens) and tuple(tokens[-count:]) == ids else 0,
    )


@dataclass(frozen=True, slots=True)
class StoredResponse:
    response_json: bytes
    history_json: bytes

    @property
    def size(self):
        return len(self.response_json) + len(self.history_json)

    @property
    def response(self):
        return json_codec.loads_compact(self.response_json)


class ResponseStore:
    """Process-local Responses state with one strict byte-budgeted LRU."""

    BUDGET_BYTES = 64 * 1024 * 1024

    def __init__(self):
        self.records = OrderedDict()
        self.bytes = 0
        self.evictions = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, response_id):
        with self.lock:
            record = self.records.pop(response_id, None)
            if record is None:
                self.misses += 1
                return None
            self.records[response_id] = record
            self.hits += 1
        return record

    def put(self, response, history_items):
        record = StoredResponse(
            json_codec.encode(response), json_codec.encode(history_items)
        )
        if record.size > self.BUDGET_BYTES:
            return False
        response_id = response["id"]
        with self.lock:
            previous = self.records.pop(response_id, None)
            if previous is not None:
                self.bytes -= previous.size
            self.records[response_id] = record
            self.bytes += record.size
            while self.bytes > self.BUDGET_BYTES:
                _, evicted = self.records.popitem(last=False)
                self.bytes -= evicted.size
                self.evictions += 1
        return True

    def delete(self, response_id):
        with self.lock:
            record = self.records.pop(response_id, None)
            if record is None:
                return False
            self.bytes -= record.size
            return True

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.records),
                "bytes": self.bytes,
                "budget_bytes": self.BUDGET_BYTES,
                "evictions": self.evictions,
                "hits": self.hits,
                "misses": self.misses,
            }


@dataclass
class Prompt:
    messages: list
    tools: list | None
    tool_policy: ToolPolicy | None
    reasoning_effort: str | None
    response_schema: dict | bool | None = None
    response_validator: object = None
    preserve_thinking: bool | None = None
    # The request's own chat_template_kwargs enable_thinking, which outranks
    # its reasoning effort (template_options); None where it sets none.
    enable_thinking: bool | None = None
    # Its other template variables, which outrank RichEngine's own.
    template_kwargs: dict = field(default_factory=dict)


@dataclass
class RenderedPrompt:
    text: str
    tokens: list[int]
    images: list
    image_positions: list[int]
    thinking: bool
    # Images precede the generation prompt; expanding them keeps this count.
    generation_prompt_tokens: int
    # Leading tokens other requests with the same system prompt and tools
    # share; zero unless requested, or when an image lies within them.
    shared_prefix_tokens: int = 0


@dataclass(frozen=True)
class GenerationOptions:
    """Sampling, stop and scheduling options, validated alike by every
    generation API."""

    sampling: wire.SamplingParameters
    stop_sequences: tuple[str, ...]
    ignore_eos: bool
    seed: int
    priority: wire.RequestPriority


class Frontend:
    # Class-level fallbacks for harnesses that bypass __init__ (tests): the
    # served model's contract overrides each at construction.
    image_pad_token = IMAGE_PAD_TOKEN
    image_geometry = image_input.DEFAULT_GEOMETRY
    sampling_defaults: dict = {}
    serving_dialect = None
    encoder = False
    decision_temperatures: dict = {}

    def __init__(
        self,
        tokenizer,
        backend,
        model,
        max_context,
        request_timeout,
        preparation_capacity,
        *,
        constraint_factory,
        chat_templates,
        thinking_codec,
        vision,
        max_image_pixels=image_input.MAX_PIXELS,
        served_model_names=(),
        announce_served_name=False,
        default_reasoning_effort=None,
        contract=None,
        serving=None,
        shared_prefix_states=False,
    ):
        if not isinstance(preparation_capacity, int) or preparation_capacity <= 0:
            raise ValueError("frontend preparation capacity must be positive")
        # Announced by the engine in its Ready event. Without it, message
        # normalization rejects image and PDF input before any decoding.
        self.vision = vision
        self.latencies = LatencyMetrics()
        self.tokenizer = tokenizer
        # Probed at startup; requests choose among these, never the
        # tokenizer's own.
        self.chat_templates = chat_templates
        # The validated tokenizer contract's ids; the module defaults stand
        # in only where no contract was validated (the test harness).
        self.think_end_id = (
            THINK_END_TOKEN_ID if contract is None else contract.think_end_id
        )
        # The think block's spellings; the defaults are Qwen's.
        self.think_open = "<think>" if contract is None else contract.think_open
        self.think_end = THINK_END if contract is None else contract.think_end
        self.think_flag = None if contract is None else contract.think_flag
        # The serving contract's spellings and defaults. Where the model
        # declares none (old assemblies) every field keeps the probe/default
        # the server has always used.
        self.serving = serving
        # The contract's sampling defaults are validated here, at load: a
        # bad value is the model file's fault, not the first request's, and
        # must not surface as a 400.
        self.sampling_defaults = {}
        for name, value in (serving.sampling if serving else {}).items():
            if name == "top_k":
                if not isinstance(value, int) or isinstance(value, bool) or value < -1:
                    raise ValueError(
                        "serving contract's sampling default top_k must be "
                        "0 or -1 (disabled) or a positive integer"
                    )
            elif name in SAMPLING_NUMBERS:
                _, accepts, requirement = SAMPLING_NUMBERS[name]
                if not is_finite_number(value) or not accepts(value):
                    raise ValueError(
                        f"serving contract's sampling default {name} must be {requirement}"
                    )
            self.sampling_defaults[name] = value
        self.image_pad_token = (
            serving.vision.pad_token
            if serving is not None and serving.vision is not None
            else IMAGE_PAD_TOKEN
        )
        self.image_geometry = (
            image_input.ImageGeometry(
                patch=serving.vision.patch_size,
                merge=serving.vision.merge_size,
                min_pixels=serving.vision.min_pixels,
            )
            if serving is not None and serving.vision is not None
            else image_input.DEFAULT_GEOMETRY
        )
        # The contract's tool dialect fills in where the template probe
        # spells no known framing; a detected one still wins — the template
        # decides how calls render.
        self.serving_dialect = (
            dialect_named(serving.tool_dialect) if serving is not None else None
        )
        # Bound tool dialects by name, resolved on first use: marker spellings
        # get their token ids from this model's vocabulary (bind_dialect).
        self._dialect_bindings = {}
        self.prompt_tokenizer = PromptTokenizer(
            tokenizer, None if contract is None else contract.marker
        )
        self.backend = backend
        # The package id the engine loaded. /status reports it, so an alias
        # can never hide what served a request (#81).
        self.model = model
        served = tuple(parse_served_model_name(name) for name in served_model_names)
        # The contract's aliases join the command line's.
        if serving is not None:
            served += tuple(
                parse_served_model_name(name) for name in serving.served_names
            )
        if announce_served_name and not served:
            raise ValueError("announce_served_name needs a served model name")
        # The name generation and scoring responses report.
        self.response_model = served[0] if announce_served_name else model
        # /v1/models order: the name responses report first, so a client
        # configured from data[0] (richengine <client>) requests the name it gets
        # back. Every name stays accepted.
        self.model_names = tuple(dict.fromkeys((self.response_model, model, *served)))
        if (
            default_reasoning_effort is not None
            and default_reasoning_effort not in REASONING_EFFORTS
        ):
            raise ValueError("invalid default_reasoning_effort")
        # The contract's effort applies where the flag stated none.
        if default_reasoning_effort is None and serving is not None:
            effort = serving.reasoning_effort
            if effort is not None and effort not in REASONING_EFFORTS:
                raise ValueError("invalid serving reasoning_effort")
            default_reasoning_effort = effort
        self.default_reasoning_effort = default_reasoning_effort
        self.max_context = max_context
        self.request_timeout = request_timeout
        self.constraint_factory = constraint_factory
        self.max_image_pixels = max_image_pixels
        self.images = image_input.ImageCache(self.image_geometry)
        self.ids = count(1)
        self.preparation_capacity = preparation_capacity
        # RICHENGINE_SYSTEMONE_WINDOW caps /v1/systemone's in-flight
        # questions below preparation_capacity; 0 or unset keeps the bound.
        self.systemone_window = max(
            0, int(os.environ.get("RICHENGINE_SYSTEMONE_WINDOW", "0") or 0)
        )
        self.preparation_slots = threading.BoundedSemaphore(preparation_capacity)
        self.preparation_lock = threading.Lock()
        self.preparation_active = 0
        self.preparation_waiting = 0
        self.response_store = ResponseStore()
        self.thinking_codec = thinking_codec
        self.shared_prefix_states = shared_prefix_states
        # A d1-family checkpoint answers /v1/systemone through its own
        # prompt and grouped readout; every other model keeps the SemIf
        # letter-slot pathway. Nothing else changes: same score-only jobs.
        # The contract's encoder flag and calibrated readout temperatures:
        # the former refuses generation outright; the decision pathway's
        # questions carry the latter into their softmax.
        self.encoder = getattr(serving, "encoder", False)
        self.decision_temperatures = (
            getattr(serving, "decision_temperatures", None) or {}
        )
        # The score pathway decides how /v1/judgments and /v1/systemone
        # prepare and read back: SemIf letter slots for generative models,
        # the decision module's own markup and readout for d1.
        decision = decisions.is_decision_model(model) or (
            self.serving is not None and self.serving.decision
        )
        self.pathway = (
            decisions.DecisionPathway(self.decision_temperatures)
            if decision
            else judgments.SemIfPathway()
        )
        # Agents repeat a few system prompts and tool sets.
        self._shared_prefix_probe = lru_cache(maxsize=8)(
            self._render_shared_prefix_probe
        )

    @property
    def decision_pathway(self):
        """Whether the resident's score pathway is the d1 decision one;
        /v1/status reports it and the pool prefers it for score requests."""
        return isinstance(self.pathway, decisions.DecisionPathway)

    @staticmethod
    def _deadline_checkpoint(deadline, disconnected):
        """A pathway's preparation checkpoint callable."""
        return _preparation_checkpoint(deadline, disconnected)

    def accepts_model(self, model):
        return isinstance(model, str) and model in self.model_names

    @property
    def input_modalities(self):
        """The one list /status, /v1/models and client setup report."""
        return ["text", "image", "pdf"] if self.vision else ["text"]

    def status(self):
        status = self.backend.status()
        status["vision"] = self.vision
        status["input_modalities"] = self.input_modalities
        status["chat_template"] = self.chat_templates.status()
        with self.preparation_lock:
            status["frontend"] = {
                "preparation_capacity": self.preparation_capacity,
                "active": self.preparation_active,
                "waiting": self.preparation_waiting,
            }
        status["grammar_cache"] = self.constraint_factory.stats()
        status["dfa_mode"] = getattr(
            self.constraint_factory, "resolved_dfa_mode", constraints.dfa_mode
        )()
        status["response_store"] = self.response_store.stats()
        status["image_cache"] = self.images.stats()
        status["tokenizer_cache"] = self.prompt_tokenizer.stats()
        status["latency"] = self.latencies.snapshot()
        return status

    def _prepare_images(self, messages, deadline, *, check_context=True):
        """Prepared images in template render order: content parts in message
        order, images in document order."""
        parts = [
            part
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        limit = wire.MAX_IMAGE_SPANS
        if len(parts) > limit:
            raise APIError(400, f"requests support at most {limit} images")
        prepared = self.images.request_batch()
        tokens = pixel_bytes = 0
        for part in parts:
            # An image can take a few hundred milliseconds to decode; an
            # expired request starts no more of them.
            remaining_request_time(deadline)
            try:
                payload = image_input.decode_data_url(part["image_url"]["url"])
                image = self.images.prepare(payload, self.max_image_pixels)
                tokens += image.tokens
                pixel_bytes += len(image.pixels)
                self._check_image_request_size(
                    tokens,
                    len(prepared) + 1,
                    pixel_bytes,
                    check_context=check_context,
                    image_tokens_only=True,
                )
                prepared.append(image)
            except image_input.ImageCapacityError as error:
                raise APIError(503, str(error), "frontend_overloaded") from error
            except image_input.ImageError as error:
                raise APIError(400, str(error)) from error
        return prepared

    def _check_image_request_size(
        self,
        tokens,
        image_count,
        pixel_bytes,
        *,
        check_context=True,
        image_tokens_only=False,
    ):
        if check_context and tokens >= self.max_context:
            raise ContextLengthError(
                tokens, self.max_context - 1, image_tokens_only=image_tokens_only
            )
        frame_bytes = (
            wire.REQUEST_FIXED_BYTES
            + 4 * tokens
            + wire.IMAGE_SPAN_BYTES * image_count
            + pixel_bytes
        )
        if frame_bytes > wire.MAX_FRAME_PAYLOAD_BYTES:
            raise APIError(400, "images exceed the request size limit")

    def _render_image_tokens(self, messages, template):
        """Track placeholders emitted by the template, not quoted in input text.

        A temporary render marker is removed before tokenization, so the pinned
        template's final text and token IDs remain unchanged. Token offsets tie
        each real image to its placeholder even when a coding agent has read
        documentation or source containing literal vision tokens.
        """
        rendered = self._apply_chat_template(
            messages,
            {
                **template,
                "tokenize": False,
                "chat_template": template["chat_template"].replace(
                    self.image_pad_token, IMAGE_RENDER_MARKER
                ),
            },
        )
        parts = rendered.split(IMAGE_RENDER_MARKER)
        image_offsets = set()
        offset = 0
        for part in parts[:-1]:
            offset += len(part)
            image_offsets.add((offset, offset + len(self.image_pad_token)))
            offset += len(self.image_pad_token)
        rendered = self.image_pad_token.join(parts)
        encoded = self._tokenize(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        positions = [
            index
            for index, span in enumerate(encoded["offset_mapping"])
            if tuple(span) in image_offsets
        ]
        return list(encoded["input_ids"]), positions, rendered

    def _image_token_count(self, prompt_tokens, prepared, positions):
        pad_id = self.tokenizer.convert_tokens_to_ids(self.image_pad_token)
        if len(positions) != len(prepared) or any(
            prompt_tokens[position] != pad_id for position in positions
        ):
            raise APIError(400, "image count does not match the rendered template")
        return len(prompt_tokens) + sum(image.tokens - 1 for image in prepared)

    def _expand_image_pads(self, prompt_tokens, prepared, positions):
        """Widens the template's single placeholder per image to the image's
        merged token count and returns the spans the engine injects into."""
        token_count = self._image_token_count(prompt_tokens, prepared, positions)
        # Validate lengths before expanding tokens or copying repeated pixels.
        # HTTP body and image-cache limits do not bound decoded request size.
        self._check_image_request_size(
            token_count,
            len(prepared),
            sum(len(image.pixels) for image in prepared),
        )
        expanded, spans, cursor = [], [], 0
        pad_id = self.tokenizer.convert_tokens_to_ids(self.image_pad_token)
        for position, image in zip(positions, prepared):
            expanded.extend(prompt_tokens[cursor:position])
            spans.append(
                wire.ImageSpan(
                    len(expanded),
                    image.tokens,
                    image.grid_height,
                    image.grid_width,
                    image.digest_lo,
                    image.digest_hi,
                )
            )
            expanded.extend([pad_id] * image.tokens)
            cursor = position + 1
        expanded.extend(prompt_tokens[cursor:])
        pixels = b"".join(image.pixels for image in prepared)
        return expanded, tuple(spans), pixels

    def request_deadline(self, body, started_at):
        timeout = body.get("timeout")
        if timeout is None:
            timeout = self.request_timeout
        elif not is_finite_number(timeout) or timeout <= 0:
            raise APIError(400, "timeout must be positive")
        return started_at + min(timeout, self.request_timeout)

    def prepare(
        self,
        body,
        *,
        deadline,
        output_field=None,
        clamp_output_budget=False,
        thinking_display="summarized",
    ):
        """A Chat request, or a Messages request converted to Chat.
        Output limit errors name output_field, that API's own field; with
        clamp_output_budget, a limit larger than what the context leaves is
        lowered to it instead of refused. thinking_display "omitted" hides
        Messages reasoning behind a signature."""
        with self._preparation(deadline):
            return self._prepare(
                body,
                None,
                deadline,
                output_field=output_field,
                clamp_output_budget=clamp_output_budget,
                thinking_display=thinking_display,
            )

    def prepare_completion(self, body, *, deadline):
        """A text completion: the prompt generates as given, with no chat
        template, reasoning split, tools or images."""
        if self.encoder or not self.pathway.generates:
            raise APIError(
                422,
                "this model is a decision "
                + ("encoder" if self.encoder else "model")
                + " and does not generate",
                "encoder_no_generate" if self.encoder else "decision_no_generate",
            )
        with self._preparation(deadline):
            body = _drop_nulls(
                body, ("n", "best_of", "max_tokens", "suffix", "echo", "logprobs")
            )
            if not self.accepts_model(body.get("model", self.model)):
                raise APIError(
                    404, f"model {body['model']} not found", "model_not_found"
                )
            # A request generates one text and returns only that text.
            for field in ("suffix", "logprobs"):
                if field in body:
                    raise APIError(400, f"{field} is not supported")
            if body.get("echo", False) is not False:
                raise APIError(400, "echo is not supported")
            for field in ("best_of", "n"):
                value = body.get(field, 1)
                if not isinstance(value, int) or isinstance(value, bool) or value != 1:
                    raise APIError(400, f"{field} must be 1")
            options = self._generation_options(body)
            prompt_tokens = self._completion_prompt(body.get("prompt"))
            remaining_request_time(deadline)
            # The default is a ceiling: a prompt that leaves less context
            # generates up to the rest.
            requested = body.get("max_tokens")
            max_new = self._output_budget(
                COMPLETION_DEFAULT_MAX_TOKENS if requested is None else requested,
                prompt_tokens,
                "max_tokens",
                clamp=requested is None,
            )
            return self._generation_job(options, prompt_tokens, max_new, deadline)

    def _completion_prompt(self, prompt):
        """A string encodes as a raw prompt, with the tokenizer's own special
        tokens such as a BOS; token ids must be in its vocabulary."""
        if isinstance(prompt, str):
            try:
                tokens = self._tokenize(prompt, add_special_tokens=True)["input_ids"]
            except Exception as error:
                raise APIError(400, "prompt could not be tokenized") from error
        elif isinstance(prompt, list) and all(type(token) is int for token in prompt):
            vocabulary = len(self.tokenizer)
            if any(not 0 <= token < vocabulary for token in prompt):
                raise APIError(400, "prompt token ids must be in the vocabulary")
            tokens = prompt
        else:
            raise APIError(400, "prompt must be one string or one array of token ids")
        if not tokens:
            raise APIError(400, "prompt must not be empty")
        return list(tokens)

    def count_tokens(self, body, *, deadline):
        with self._preparation(deadline):
            prompt = self._prepare_prompt(body, None, deadline=deadline)
            rendered = self._render_prompt(prompt, deadline, check_context=False)
            return self._image_token_count(
                rendered.tokens, rendered.images, rendered.image_positions
            )

    def tokenize(self, body, *, deadline):
        content = body.get("content")
        if not isinstance(content, str):
            raise APIError(400, "content must be a string")
        add_special = body.get("add_special", False)
        if not isinstance(add_special, bool):
            raise APIError(400, "add_special must be a boolean")
        for option, supported in (("parse_special", True),):
            if body.get(option, supported) is not supported:
                raise APIError(
                    400, f"only {option}={str(supported).lower()} is supported"
                )
        with_pieces = body.get("with_pieces", False)
        if not isinstance(with_pieces, bool):
            raise APIError(400, "with_pieces must be a boolean")
        with self._preparation(deadline):
            try:
                encoded = self._tokenize(
                    content,
                    add_special_tokens=add_special,
                    **({"return_offsets_mapping": True} if with_pieces else {}),
                )
            except NotImplementedError as error:
                raise APIError(
                    400, "this tokenizer cannot report token offsets"
                ) from error
            except Exception as error:
                raise APIError(400, "content could not be tokenized") from error
            remaining_request_time(deadline)
            response = {"tokens": encoded["input_ids"]}
            if with_pieces:
                # Char spans into `content` for each id; specials report (0, 0).
                offsets = encoded.get("offset_mapping") or []
                response["offsets"] = [[int(start), int(end)] for start, end in offsets]
            return response

    def _priority(self, body):
        priority_name = body.get("priority", "normal")
        if not isinstance(priority_name, str) or priority_name not in _PRIORITIES:
            raise APIError(400, "priority must be foreground, normal, or background")
        return _PRIORITIES[priority_name]

    def _score_job(
        self,
        prompt_tokens,
        slot_ids,
        deadline,
        priority,
        prompt_sha256=None,
        shared_prefix_tokens=0,
        prefix_tokens=(),
    ):
        return Job(
            request_id=next(self.ids),
            prompt_tokens=prompt_tokens,
            prefix_tokens=prefix_tokens,
            max_new_tokens=0,
            seed=0,
            sampling=wire.SamplingParameters(),
            deadline=deadline,
            priority=priority,
            score_tokens=tuple(slot_ids),
            public_id=secrets.token_hex(16),
            prompt_sha256=prompt_sha256,
            shared_prefix_tokens=shared_prefix_tokens,
        )

    def _encode_score_prompt(self, messages, labels, admit, checkpoint, what):
        """The tokens, answer-slot token ids and text of a scoring prompt. The
        request's input drives the render, so a failure is its error."""
        try:
            return judgments.encode_prompt(
                self.prompt_tokenizer,
                self.chat_templates.select(None).source,
                messages,
                labels,
                admit=admit,
                checkpoint=checkpoint,
            )
        except judgments.ScoringUnsupported as error:
            raise APIError(500, str(error), "scoring_unsupported") from error
        except (APIError, judgments.SystemOneError, ConnectionResetError):
            # The request's own limits, deadline and client end preparation.
            raise
        except Exception as error:
            raise APIError(400, f"{what} prompt could not be rendered") from error

    def prepare_judgment(self, body, *, deadline, disconnected=None):
        """Prepare a judgment row through the resident's score pathway."""
        return self.pathway.prepare_judgment(
            self, body, deadline=deadline, disconnected=disconnected
        )

    def judgment_response(self, row, job, result):
        """The judgment response for one prepared row. Entries carrying their
        own responder (the d1 pathway's Judgment) use it; SemIf rows keep the
        letter-slot readout."""
        responder = getattr(row, "judgment_response", None)
        if responder is not None:
            return responder(self.response_model, job, result)
        return judgments.judgment_response(self.response_model, row, job, result)

    def systemone_metadata(self):
        """The prompt/readout identity /v1/systemone reports."""
        return self.pathway.systemone_metadata()

    def prepare_systemone(self, body, *, deadline, disconnected=None):
        """Prepare systemone questions through the resident's score
        pathway."""
        return self.pathway.prepare_systemone(
            self, body, deadline=deadline, disconnected=disconnected
        )

    def systemone_answer(self, spec, logits):
        """The typed answer for one prepared question. Prepared entries that
        carry their own readout (the d1 pathway's Question) use it; SemIf
        specs keep letter-slot softmax. `None` is the no-inference answer."""
        answer = getattr(spec, "answer", None)
        if answer is not None:
            return answer(logits)
        if logits is None:
            return judgments.deterministic_answer(spec)
        return judgments.systemone_answer(spec, judgments.softmax(list(logits)))

    def apply_template(self, body, *, deadline):
        add_generation_prompt = body.get("add_generation_prompt", True)
        if not isinstance(add_generation_prompt, bool):
            raise APIError(400, "add_generation_prompt must be a boolean")
        with self._preparation(deadline):
            prompt = self._prepare_prompt(body, None, deadline=deadline)
            return self._render_prompt(
                prompt,
                deadline,
                check_context=False,
                add_generation_prompt=add_generation_prompt,
            ).text

    @contextmanager
    def _preparation(self, deadline):
        remaining = remaining_request_time(deadline)
        with self.preparation_lock:
            self.preparation_waiting += 1
        with self.latencies.measure("preparation_queue"):
            acquired = self.preparation_slots.acquire(
                timeout=min(remaining, PREPARATION_WAIT_SECONDS)
            )
        with self.preparation_lock:
            self.preparation_waiting -= 1
            if acquired:
                self.preparation_active += 1
        if not acquired:
            remaining_request_time(deadline)
            raise APIError(
                503,
                "frontend preparation capacity is exhausted",
                "frontend_overloaded",
            )
        try:
            remaining_request_time(deadline)
            with self.latencies.measure("preparation"):
                yield
        finally:
            with self.preparation_lock:
                self.preparation_active -= 1
            self.preparation_slots.release()

    def _prepare_prompt(self, body, tool_namespaces, *, deadline):
        if not self.accepts_model(body.get("model", self.model)):
            raise APIError(404, f"model {body['model']} not found", "model_not_found")
        reasoning_effort = body.get("reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = self.default_reasoning_effort
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str)
            or reasoning_effort not in REASONING_EFFORTS
        ):
            raise APIError(400, "invalid reasoning_effort")
        preserve_thinking = body.get("preserve_thinking")
        if preserve_thinking is not None and not isinstance(preserve_thinking, bool):
            raise APIError(400, "preserve_thinking must be a boolean")
        template_kwargs = body.get("chat_template_kwargs")
        if template_kwargs is None:
            template_kwargs = {}
        elif not isinstance(template_kwargs, dict):
            raise APIError(400, "chat_template_kwargs must be an object")
        elif reserved := sorted(RESERVED_TEMPLATE_KWARGS & template_kwargs.keys()):
            raise APIError(400, f"chat_template_kwargs cannot set {reserved[0]}")
        template_kwargs = dict(template_kwargs)
        enable_thinking = template_kwargs.pop("enable_thinking", None)
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise APIError(
                400, "chat_template_kwargs enable_thinking must be a boolean"
            )
        messages = move_volatile_system_fields(
            template_messages(
                normalize_messages(
                    body.get("messages"), vision=self.vision, deadline=deadline
                )
            )
        )
        tools, tool_policy = normalize_tools(
            body.get("tools"),
            body.get("tool_choice"),
            body.get("parallel_tool_calls", True),
            tool_namespaces,
            self._tool_dialect(
                self.chat_templates.select(body.get("tools")).dialect
            ),
        )
        response_schema, response_validator = normalize_response_format(
            body.get("response_format")
        )
        return Prompt(
            messages,
            tools,
            tool_policy,
            reasoning_effort,
            response_schema,
            response_validator,
            preserve_thinking,
            enable_thinking,
            template_kwargs,
        )

    def _tokenize(self, text, **options):
        with self.latencies.measure("tokenization"):
            return self.tokenizer(text, **options)

    def corpus_push(self, body):
        """POST /v1/corpus: tokenize text and push it into the engine's
        cross-request n-gram corpus — workspace files, tool outputs, and
        (emit_variants=true) the diff-prefixed and JSON-escaped copies
        whose emission forms differ from the raw text."""
        text = body.get("text")
        if not isinstance(text, str) or not text:
            raise APIError(400, '"text" must be a non-empty string')
        session = body.get("session", 0)
        if isinstance(session, str):
            # Client-side conversation ids arrive as strings (opencode
            # sessionID); fold to a nonzero u64 session key.
            session = int.from_bytes(
                hashlib.blake2b(session.encode(), digest_size=8).digest()
            ) | 1
        if not isinstance(session, int) or session < 0:
            raise APIError(400, '"session" must be a non-negative integer')
        docs = [text]
        if body.get("emit_variants"):
            docs.append("".join("+" + l for l in text.splitlines(keepends=True)))
            docs.append(json.dumps(text))
        pushed = []
        for doc in docs:
            ids = self._tokenize(doc, add_special_tokens=False)["input_ids"]
            if ids:
                self.backend.runtime.corpus_append(ids, session)
                pushed.append(len(ids))
        return {"pushed": pushed}

    def _apply_chat_template(self, messages, template):
        with self.latencies.measure("template"):
            return render_chat_template(self.tokenizer, messages, template)

    def _render_shared_prefix_probe(self, key):
        """The tokens of a request's leading system message and template
        options followed by the probe turn instead of its own; empty where
        the template cannot render it."""
        head, template = json.loads(key)
        try:
            text = self._apply_chat_template([*head, SHARED_PREFIX_PROBE], template)
        except Exception:
            return ()
        return tuple(self._tokenize(text, add_special_tokens=False)["input_ids"])

    def _shared_prefix_probe_tokens(self, messages, template):
        """The tokens of the prompt's leading system message, tools and
        template options followed by a probe turn; empty without a system
        message or tools."""
        head = messages[:1] if messages and messages[0]["role"] == "system" else []
        if head and not isinstance(head[0]["content"], str):
            return ()
        if not head and "tools" not in template:
            return ()
        # Key order is kept: templates render tools in the request's order.
        return self._shared_prefix_probe(
            json_codec.encode_compact([head, template])
        )

    def _render_prompt(
        self,
        prompt,
        deadline,
        *,
        check_context=True,
        add_generation_prompt=True,
        shared_prefix=False,
    ):
        chat_template = self.chat_templates.select(prompt.tools)
        if not chat_template.accepts(prompt.messages):
            raise APIError(400, LATER_SYSTEM_UNSUPPORTED)
        template = {
            "tokenize": False,
            "return_dict": False,
            "chat_template": chat_template.source,
            **template_options(
                reasoning_effort=prompt.reasoning_effort,
                preserve_thinking=prompt.preserve_thinking,
                tools=prompt.tools,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=prompt.enable_thinking,
            ),
            **prompt.template_kwargs,
        }
        # Rendered first, so the request's own rendering stays the latest.
        shared_prefix_probe = (
            self._shared_prefix_probe_tokens(prompt.messages, template)
            if shared_prefix
            else None
        )
        with self.latencies.measure("images"):
            images = self._prepare_images(
                prompt.messages, deadline, check_context=check_context
            )
        remaining_request_time(deadline)
        if images and self.tokenizer.convert_tokens_to_ids(self.image_pad_token) is None:
            raise APIError(400, "the tokenizer does not define the image pad token")
        positions = []
        try:
            if images:
                tokens, positions, rendered = self._render_image_tokens(
                    prompt.messages, template
                )
            else:
                rendered = self._apply_chat_template(prompt.messages, template)
                with self.latencies.measure("tokenization"):
                    tokens = self.prompt_tokenizer.encode(rendered)
        except APIError:
            raise
        except Exception as error:
            frame = error.__traceback__
            template_frame = None
            while True:
                if frame.tb_frame.f_code.co_filename == "<template>":
                    template_frame = frame
                if frame.tb_next is None:
                    break
                frame = frame.tb_next
            frame = template_frame or frame
            location = Path(frame.tb_frame.f_code.co_filename).name
            print_status(
                f"Template error · {type(error).__name__} · {location}:{frame.tb_lineno}",
                error=True,
            )
            raise APIError(400, "messages could not be rendered") from error
        remaining_request_time(deadline)
        thinking, generation_prompt_tokens = False, 0
        if add_generation_prompt:
            probed = chat_template.generation_prompt(template)
            if not probed[0] and chat_template.prompt_varies(template):
                # The startup probe found no generation prompt common to
                # its conversations: this template appends different text
                # depending on the last turn (Gemma 4 appends nothing after
                # a tool response). The request's own is the suffix over
                # the same conversation rendered without it.
                probed = self._request_generation_prompt(
                    prompt.messages, template, rendered, tokens
                )
            if probed[0] or not chat_template.prompt_varies(template):
                thinking, generation_prompt_tokens = _generation_prompt(
                    probed,
                    rendered,
                    tokens,
                    think_open=self.think_open,
                    think_end=self.think_end,
                    think_flag=self.think_flag,
                )
        requested = template.get("enable_thinking")
        if (
            add_generation_prompt
            and requested is not None
            and thinking != bool(requested)
            and thinking
        ):
            raise APIError(
                400, "chat template does not support the requested thinking mode"
            )
        # The leading tokens that depend only on the system message, tools
        # and template options, which an agent repeats across requests: the
        # prompt's common prefix with the probe. The engine keeps a reusable
        # state there, so the next such request resumes after the head
        # whatever follows it. No template structure is assumed.
        shared_prefix_tokens = 0
        for ours, probed in zip(tokens, shared_prefix_probe or ()):
            if ours != probed:
                break
            shared_prefix_tokens += 1
        # Expanding an image before it would move it.
        if positions and positions[0] < shared_prefix_tokens:
            shared_prefix_tokens = 0
        return RenderedPrompt(
            rendered,
            tokens,
            images,
            positions,
            thinking,
            generation_prompt_tokens,
            shared_prefix_tokens,
        )

    def _request_generation_prompt(self, messages, template, rendered, tokens):
        """This request's generation prompt as a (text, tokens) pair, when
        no probe conversation's prompt could stand for it: the suffix the
        same render gains from add_generation_prompt, with its token count
        where the prompt's tokens are a suffix of the prompt's."""
        history = self._apply_chat_template(
            messages, {**template, "add_generation_prompt": False}
        )
        if not rendered.startswith(history):
            return "", ()
        suffix = rendered[len(history) :]
        if not suffix:
            return "", ()
        history_tokens = self.prompt_tokenizer.encode(history)
        count = len(history_tokens)
        ids = tokens[count:] if tokens[:count] == history_tokens else ()
        return suffix, tuple(ids)

    def _prepare(
        self,
        body,
        tool_namespaces,
        deadline,
        *,
        output_field,
        clamp_output_budget,
        thinking_display,
    ):
        # A decision model scores only; every generation entry point
        # (Chat, Messages, Responses) funnels here.
        if self.encoder or not self.pathway.generates:
            raise APIError(
                422,
                "this model is a decision "
                + ("encoder" if self.encoder else "model")
                + " and does not generate",
                "encoder_no_generate" if self.encoder else "decision_no_generate",
            )
        body = _drop_nulls(
            body,
            (
                "n",
                "max_tokens",
                "max_completion_tokens",
                "stream",
                "parallel_tool_calls",
            ),
        )
        # Fields checked without the prompt fail before it is prepared, which
        # can render documents and build validators.
        options = self._generation_options(body)
        n = body.get("n", 1)
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 4:
            raise APIError(400, "n must be an integer from 1 to 4")
        # TokensEvent carries token ids only, so per-token logprobs would
        # need engine support.
        logprobs = body.get("logprobs")
        if logprobs is not None and (not isinstance(logprobs, bool) or logprobs):
            raise APIError(400, "logprobs is not supported")
        metadata = _request_metadata(body)
        prompt = self._prepare_prompt(body, tool_namespaces, deadline=deadline)
        tools, tool_policy = prompt.tools, prompt.tool_policy
        response_schema, response_validator = (
            prompt.response_schema,
            prompt.response_validator,
        )
        # A stop sequence could cut a tool call or a structured result short;
        # under tool_choice none the tools are only described, never called.
        if options.stop_sequences and (
            (tools and tool_policy.schemas) or response_schema is not None
        ):
            raise APIError(
                400, "stop cannot be combined with tools or structured output"
            )
        # Constrained tool calls and structured output generate under a
        # grammar, which decides where the output ends; tools beside a
        # response schema share one.
        tool_constrained = tool_policy is not None and (
            tool_policy.constrained or response_schema is not None
        )
        constrained = tool_constrained or response_schema is not None
        if n > 1 and constrained:
            # One compiled constraint carries per-request grammar state; it
            # cannot feed sibling lanes.
            raise APIError(
                400,
                "n values above 1 cannot be combined with tools, tool_choice "
                "none or structured output",
            )
        if options.ignore_eos and constrained:
            raise APIError(
                400,
                "ignore_eos cannot be combined with constrained tool calls, "
                "tool_choice none or structured output",
            )
        rendered = self._render_prompt(
            prompt, deadline, shared_prefix=self.shared_prefix_states
        )
        prompt_tokens, prepared_images = rendered.tokens, rendered.images
        image_positions, thinking = rendered.image_positions, rendered.thinking
        constraint = None
        remaining_request_time(deadline)
        if constrained:

            def build_constraint():
                with self.latencies.measure("grammar"):
                    return self._compile_constraint(
                        tool_constrained,
                        tool_policy,
                        thinking,
                        response_schema,
                        deadline,
                    )

            # Compile on the factory's pool: the request's own admission,
            # tokenization and prefill overlap the compile. The first mask
            # or forced-span access joins it.
            defer = getattr(self.constraint_factory, "defer", None)
            constraint = defer(build_constraint) if defer else build_constraint()
        remaining_request_time(deadline)
        tools_signature = None
        if tools:
            digest = hashlib.sha1(
                json_codec.encode_compact(tools, sort_keys=True)
            ).hexdigest()[:8]
            tools_signature = (len(tools), digest)
        image_spans, image_pixels = (), b""
        if prepared_images:
            prompt_tokens, image_spans, image_pixels = self._expand_image_pads(
                prompt_tokens, prepared_images, image_positions
            )
        remaining_request_time(deadline)
        requested = body.get("max_completion_tokens", body.get("max_tokens"))
        if output_field is None:
            # Chat takes either field; errors name the one the client sent.
            output_field = (
                "max_completion_tokens"
                if "max_completion_tokens" in body
                else "max_tokens"
            )
        max_new = self._output_budget(
            requested, prompt_tokens, output_field, clamp_output_budget
        )
        return self._generation_job(
            options,
            prompt_tokens,
            max_new,
            deadline,
            thinking=thinking,
            thinking_display=thinking_display,
            tool_policy=tool_policy,
            response_validator=response_validator,
            response_format=body.get("response_format"),
            response_metadata=metadata,
            constraint=constraint,
            image_spans=image_spans,
            image_pixels=image_pixels,
            image_owner=prepared_images if prepared_images else None,
            tools_signature=tools_signature,
            reply_memory=self.prompt_tokenizer,
            generation_prompt_tokens=rendered.generation_prompt_tokens,
            shared_prefix_tokens=rendered.shared_prefix_tokens,
            output_clamped_to_context=requested is not None and max_new < requested,
            n_samples=n,
        )

    def expand(self, job):
        """Sibling jobs for an n > 1 request: the same prepared prompt on
        one lane each with a distinct seed, submitted together so the
        scheduler packs the lanes into a single decode batch and later lanes'
        prompt resolves from the first's prefix. Returns the cohort with the
        primary job first; single-completion requests return [job]."""
        n = job.n_samples
        if n <= 1:
            return [job]
        cohort = [job] + [
            replace(
                job,
                request_id=next(self.ids),
                public_id=secrets.token_hex(16),
                seed=(job.seed + index) % 2**64,
                events=queue.Queue(),
                cache=CacheInfo(),
                latency=None,
                reply_memory=None,
                n_samples=1,
                cohort=[],
            )
            for index in range(1, n)
        ]
        for sibling in cohort:
            sibling.cohort = cohort
            # The cohort shares the whole prompt: mark it so the producer
            # keeps its state at that junction and followers — this cohort
            # and later same-prefix requests — resume from it instead of
            # recomputing the prefill.
            sibling.shared_prefix_tokens = (
                len(sibling.prefix_tokens) + len(sibling.prompt_tokens)
            )
        return cohort

    def _tool_dialect(self, detected):
        """The request's tool dialect — the template probe's, else the
        serving contract's — with its marker spellings' token ids bound, so
        a marker with no `<name>` atom form (minicpm5's `<function`) writes
        as `<[id]>`, the special token the template spells it as."""
        dialect = detected or self.serving_dialect
        if dialect is None:
            return None
        bindings = getattr(self, "_dialect_bindings", None)
        if bindings is None:
            return bind_dialect(dialect, self.tokenizer.get_vocab())
        bound = bindings.get(dialect.name)
        if bound is None:
            bound = bind_dialect(dialect, self.tokenizer.get_vocab())
            bindings[dialect.name] = bound
        return bound

    def _compile_constraint(
        self, tool_constrained, tool_policy, thinking, response_schema, deadline
    ):
        """Compile the request's constraint. Runs on the factory's compile
        pool so request prep overlaps it."""
        create_dfa = getattr(self.constraint_factory, "create_dfa", None)
        if tool_constrained:
            # The grammar is built either way: the DFA path keeps a
            # llguidance delegate on it for masks and checks, and a spec
            # outside the DFA subset compiles it outright.
            grammar = tool_grammar(
                tool_policy,
                thinking,
                response_schema,
                think_end_id=self.think_end_id,
            )
            constraint = create_dfa and create_dfa(
                dfa_compiler.tool_spec(
                    tool_policy,
                    thinking,
                    response_schema,
                    think_end=self.think_end,
                ),
                grammar,
                timeout=remaining_request_time(deadline),
                prefixes=lambda: self._call_openings(tool_policy, thinking),
            )
            if constraint is not None:
                return constraint
            return self.constraint_factory.create(
                grammar,
                timeout=remaining_request_time(deadline),
                prefixes=lambda: self._call_openings(tool_policy, thinking),
            )
        grammar = json_grammar(
            response_schema,
            thinking,
            think_end_id=self.think_end_id,
            think_end_text=self.think_end,
        )
        constraint = create_dfa and create_dfa(
            dfa_compiler.json_spec(
                response_schema, thinking, think_end=self.think_end
            ),
            grammar,
            timeout=remaining_request_time(deadline),
        )
        if constraint is not None:
            return constraint
        return self.constraint_factory.create(
            grammar,
            timeout=remaining_request_time(deadline),
        )

    def _call_openings(self, policy, thinking):
        """The tokens that begin each strict tool's call. Its parameter names
        may come next, so a tool with more of them than the parser admits
        fails there."""
        reasoning = [self.think_end_id] if thinking else []
        dialect = policy.dialect or self.serving_dialect or QWEN3_XML
        return [
            (
                reasoning
                + self._tokenize(dialect.call_opening(name), add_special_tokens=False)[
                    "input_ids"
                ],
                f"tool {name} has too many parameters to constrain",
            )
            for name in policy.schemas
            if name in policy.strict
        ]

    def _generation_options(self, body):
        numbers = {}
        for name, (default, accepts, requirement) in SAMPLING_NUMBERS.items():
            value = body.get(name, self.sampling_defaults.get(name, default))
            if not is_finite_number(value) or not accepts(value):
                raise APIError(400, f"{name} must be {requirement}")
            numbers[name] = float(value)
        if 0 < numbers["temperature"] < MIN_SAMPLING_TEMPERATURE:
            numbers["temperature"] = MIN_SAMPLING_TEMPERATURE
        # A top_k of 0 or -1 keeps every token, which the frame says with 0. A
        # top_k past the vocabulary keeps every token too, so one past the
        # frame's u32 is sent as its largest value.
        top_k = body.get("top_k", self.sampling_defaults.get("top_k", TOP_K_DEFAULT))
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < -1:
            raise APIError(
                400, "top_k must be 0 or -1 (disabled) or a positive integer"
            )
        top_k = min(max(top_k, 0), 0xFFFFFFFF)
        stop = body.get("stop")
        if stop in (None, []):
            stop_sequences = ()
        elif isinstance(stop, str) and stop:
            stop_sequences = (stop,)
        elif (
            isinstance(stop, list)
            and 1 <= len(stop) <= 4
            and all(isinstance(value, str) and value for value in stop)
        ):
            stop_sequences = tuple(stop)
        else:
            raise APIError(400, "stop must be a string or up to four strings")
        # RichEngine does not implement logit_bias, so only an empty one (null or
        # {}) is accepted; a non-empty one is refused rather than ignored.
        if body.get("logit_bias") not in (None, {}):
            raise APIError(400, "logit_bias is not supported")
        ignore_eos = body.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise APIError(400, "ignore_eos must be a boolean")
        seed = body.get("seed")
        if seed is None:
            seed = secrets.randbits(64)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise APIError(400, "seed must be an unsigned 64-bit integer")
        return GenerationOptions(
            wire.SamplingParameters(top_k=top_k, **numbers),
            stop_sequences,
            ignore_eos,
            seed,
            self._priority(body),
        )

    def _output_budget(self, requested, prompt_tokens, field, clamp=False):
        """The output token budget requested under the API's field name,
        within the context window the prompt leaves. A request that names
        none may use all of that window, as in vLLM and SGLang."""
        if len(prompt_tokens) >= self.max_context:
            raise ContextLengthError(len(prompt_tokens), self.max_context - 1)
        remaining = self.max_context - len(prompt_tokens)
        max_new = remaining if requested is None else requested
        if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new <= 0:
            raise APIError(400, f"{field} must be a positive integer")
        if max_new > remaining:
            if not clamp:
                raise APIError(
                    400,
                    f"prompt and {field} exceed the context window: "
                    f"{len(prompt_tokens)} + {max_new} > {self.max_context} tokens",
                    "context_length_exceeded",
                )
            # This API treats the output budget as a ceiling. Generate up to
            # the remaining context and report the length stop if it is reached.
            max_new = remaining
        return max_new

    def _generation_job(self, options, prompt_tokens, max_new, deadline, **fields):
        """The job for a prepared prompt, with the endpoint's own fields."""
        return Job(
            request_id=next(self.ids),
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new,
            seed=options.seed,
            sampling=options.sampling,
            deadline=deadline,
            priority=options.priority,
            stop_sequences=options.stop_sequences,
            flags=(
                wire.RequestFlag.IGNORE_END_OF_SEQUENCE
                if options.ignore_eos
                else wire.RequestFlag(0)
            ),
            think_open=self.think_open,
            think_end=self.think_end,
            public_id=secrets.token_hex(16),
            **fields,
        )

    def prepare_responses(
        self, body, *, deadline, reserve_input=None, find_response=None
    ):
        store = body.get("store")
        if store is not None and not isinstance(store, bool):
            raise APIError(400, "store must be a boolean")
        store = True if store is None else store
        metadata = _request_metadata(body)
        previous_id = body.get("previous_response_id")
        if previous_id is not None and (
            not isinstance(previous_id, str) or not previous_id
        ):
            raise APIError(400, "previous_response_id must be a non-empty string")
        if find_response is None:
            find_response = self.response_store.get
        with self._preparation(deadline):
            previous_items = []
            if previous_id is not None:
                previous = find_response(previous_id)
                if previous is None:
                    # Clients key on this code to resend the full history.
                    raise APIError(
                        404,
                        "previous response not found",
                        "previous_response_not_found",
                    )
                # The immutable record remains valid if the store evicts it.
                # Reserve its input bytes before materializing the history.
                if reserve_input is not None:
                    reserve_input(len(previous.history_json))
                previous_items = json_codec.loads_compact(previous.history_json)
            items = [*previous_items, *canonical_responses_input(body.get("input"))]
            chat, namespaces = responses_to_chat_body(body, items)
            job = self._prepare(
                chat,
                namespaces,
                deadline,
                output_field="max_output_tokens",
                clamp_output_budget=False,
                thinking_display="summarized",
            )
            job.response_store = store
            job.response_previous_id = previous_id
            # The fields the response object echoes from the request.
            job.response_metadata = metadata
            job.response_tools = body.get("tools")
            job.response_tool_choice = body.get("tool_choice") or "auto"
            job.response_instructions = body.get("instructions")
            job.response_reasoning = body.get("reasoning")
            job.response_max_output_tokens = body.get("max_output_tokens")
            job.response_top_logprobs = body.get("top_logprobs")
            if store:
                job.response_history_items = items
            return job

    def persist_response(self, job, response, output):
        if not job.response_store:
            return
        # The store encodes the history at once, so later changes to the
        # request's items or the output cannot reach it.
        history = [*job.response_history_items, *output]
        if not self.response_store.put(response, history):
            response["store"] = False
            job.response_store = False

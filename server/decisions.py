"""Native decision-model pathway for /v1/systemone (Liquid d1 family).

A decision model answers each typed question in a single forward pass: the
prompt is the model's own System One markup and the answer is a softmax over
the answer slot's logits, max-pooled across each option's surface forms.
This mirrors the public System One prompt/readout contract (LiquidAI/d1-3B
`prompt.py`/`runner.py`).

The module is deliberately self-contained: it shares only the endpoint's
wire contract with judgments.py (request validation, error details, answer
shaping, batch caps) and none of the SemIf direct-options prompt or
letter-slot logic. No runtime or model-family code is involved — the engine
serves each question through the ordinary score-only request shape.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass

try:
    import orjson as _orjson
except ImportError:
    _orjson = None

from . import judgments
from .errors import APIError, ContextLengthError

IM_START = "<" + "|im_start|" + ">"
IM_END = "<" + "|im_end|" + ">"

PROMPT_VERSION = "d1-system-one-v1"
READOUT = (
    "native full-vocabulary last-position logits max-pooled over each "
    "option's surface forms"
)
# "Calibrated" is a property of the weights (d1's post-training), not a
# locally measured guarantee: the readout reports the model's own option
# distribution, vs SemIf's conditional letter score.
PROBABILITY_STATUS = "calibrated decision distribution over declared options"

# The wire protocol carries at most this many distinct score tokens per
# request; the flattened option groups must fit inside it.
MAX_SCORE_TOKENS = 255

YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")

# Candidate answer codes once letters run out, ordered so multi-character
# forms are only reached by tokenizers that split digits.
_FALLBACK_POOL = (
    [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    + [f"{index:02d}" for index in range(100)]
    + [chr(code) for code in range(ord("a"), ord("z") + 1)]
    + [f"#{index}" for index in range(200)]
    + [
        chr(a) + chr(b)
        for a in range(ord("A"), ord("Z") + 1)
        for b in range(ord("A"), ord("Z") + 1)
    ]
)


class DecisionUnsupported(RuntimeError):
    """The served tokenizer cannot express the decision readout."""


_MODEL_OVERRIDE = "RICHENGINE_DECISION_MODEL"
_MODEL_NAME = re.compile(r"(?:^|[/_\-.])d1(?:[-_.]|$)", re.IGNORECASE)


def is_decision_model(name):
    """Whether `name` (package id or path) is a d1-family decision model.

    `RICHENGINE_DECISION_MODEL=1|0` overrides detection outright — to serve a
    renamed checkpoint natively, or to keep a d1-named model on the generic
    SemIf pathway.
    """
    flag = os.environ.get(_MODEL_OVERRIDE)
    if flag == "1":
        return True
    if flag == "0":
        return False
    return bool(name) and bool(_MODEL_NAME.search(str(name)))


def option_codes(labels):
    """The answer codes for a criteria list: the labels themselves when they
    already are single letters, else A..Z, else 00..99 (Liquid's rule)."""
    labs = [str(label).strip() for label in labels]
    if labs and all(len(label) == 1 and label.isalpha() for label in labs):
        return labs
    if len(labs) <= 26:
        return [chr(ord("A") + index) for index in range(len(labs))]
    return [f"{index:02d}" for index in range(len(labs))]


@functools.lru_cache(maxsize=128)
def _single_ids(tokenizer, texts):
    """_single_token_ids memoized per tokenizer: the same answer forms are
    probed for every question of a kind, so encode each once."""
    out, seen = [], set()
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in seen:
            out.append(encoded[0])
            seen.add(encoded[0])
    return tuple(out)


def _single_token_ids(tokenizer, texts):
    """The texts that encode as exactly one token id, deduplicated."""
    return list(_single_ids(tokenizer, tuple(texts)))


@functools.lru_cache(maxsize=128)
def _aliased(tokenizer, codes):
    return tuple(_alias_pairs(tokenizer, codes))


def _alias_pairs(tokenizer, codes):
    """Assign every code a distinct single-token id, falling back through
    the pool when the preferred spelling is not one token."""
    used = set()
    out = []

    def take(raw):
        encoded = tokenizer.encode(raw, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] in used:
            return False
        out.append((raw, encoded[0]))
        used.add(encoded[0])
        return True

    for code in codes:
        if take(code):
            continue
        if not any(take(raw) for raw in _FALLBACK_POOL):
            raise DecisionUnsupported(
                f"no single-token answer code left for {len(codes)} options"
            )
    return out


def _aliases(tokenizer, codes):
    """_alias_pairs memoized: the code assignment for a label set is stable
    across requests on one tokenizer."""
    return list(_aliased(tokenizer, tuple(codes)))


def _text(value):
    """An instruction or description as prompt text; JSON values serialize."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _option_line(code, label, description):
    if description is None or description == "":
        text = label.replace("_", " ")
    elif isinstance(description, str):
        text = description
    else:
        text = json.dumps(description, ensure_ascii=False)
    return f"{code} {text}"


# The readout temperature keys a calibrated decision model's serving
# contract states (the GGUF's lfm2.decision.temperature.* table): noul by
# side count, choice bucketed by option count, score by label count.
def _temperature_keys(kind, count):
    if kind == "noul":
        return (f"noul.{count}", "noul")
    if kind == "score":
        bucket = (
            "score.3_5" if 3 <= count <= 5
            else "score.6_10" if 6 <= count <= 10
            else "score"
        )
        return (bucket, "score")
    bucket = (
        "choice.2" if count == 2
        else "choice.3_5" if count <= 5
        else "choice.6_10" if count <= 10
        else "choice.11"
    )
    return (bucket, "choice")


@dataclass(frozen=True)
class Question:
    """A validated question plus its decision readout: each option's surface
    forms as indexes into `score_tokens` (one group per option, in label
    order), their flattened distinct ids in score-token request order, and
    the serving contract's calibrated readout temperatures when the model
    declares them."""

    spec: object
    groups: tuple
    score_tokens: tuple
    codes: tuple = ()
    temperatures: object = None

    @property
    def deterministic(self):
        return self.spec.deterministic

    def temperature(self):
        """The temperature this question's group scores divide by before
        softmax; 1.0 where the contract states none."""
        if not self.temperatures:
            return 1.0
        for key in _temperature_keys(self.spec.kind, len(self.groups)):
            value = self.temperatures.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
            ):
                return float(value)
        return 1.0

    def pooled_scores(self, logits):
        """Each option's max surface-form logit, in label order."""
        get = logits.__getitem__
        return [float(max(map(get, group))) for group in self.groups]

    def distribution(self, scores):
        """The option distribution of pooled scores, scaled by the
        contract's temperature for this spec kind, then softmaxed."""
        temperature = self.temperature()
        return judgments.softmax([score / temperature for score in scores])

    def probabilities(self, logits):
        """The option distribution over the readout's pooled scores."""
        return self.distribution(self.pooled_scores(logits))

    def answer(self, logits):
        """The typed answer. `None` is the no-inference static answer."""
        if logits is None:
            return judgments.systemone_answer(self.spec, [1.0])
        return judgments.systemone_answer(
            self.spec, self.distribution(self.pooled_scores(logits))
        )


def _readout_impl(tokenizer, kind, labels):
    """The index groups, flattened score tokens and answer codes for one
    (kind, labels) readout, per tokenizer. The surface-form token set a
    question scores depends only on its type and labels, so every question
    of a kind shares the probe's result."""
    if kind == "noul":
        token_groups = (
            tuple(_single_token_ids(tokenizer, YES_FORMS)),
            tuple(_single_token_ids(tokenizer, NO_FORMS)),
        )
        if not token_groups[0] or not token_groups[1]:
            raise DecisionUnsupported(
                "the served tokenizer has no single-token yes/no forms"
            )
        codes = ()
    elif kind == "score":
        token_groups = tuple(
            tuple(_single_token_ids(tokenizer, [label])) for label in labels
        )
        if any(not group for group in token_groups):
            raise DecisionUnsupported(
                "score needs single-token digit forms on this tokenizer"
            )
        codes = ()
    else:
        aliases = _aliases(tokenizer, option_codes(labels))
        token_groups = tuple(
            tuple(dict.fromkeys((token, *_single_token_ids(tokenizer, [f" {code}"]))))
            for (code, token) in aliases
        )
        codes = tuple(code for code, _ in aliases)
    flat, seen = [], set()
    for group in token_groups:
        for token in group:
            if token not in seen:
                seen.add(token)
                flat.append(token)
    if len(flat) < len(token_groups) or len(flat) < 2:
        raise DecisionUnsupported("answer slots collapse to one token id")
    if len(flat) > MAX_SCORE_TOKENS:
        raise DecisionUnsupported(
            f"{len(token_groups)} options need {len(flat)} score tokens; "
            f"at most {MAX_SCORE_TOKENS} are supported"
        )
    position = {token: index for index, token in enumerate(flat)}
    groups = tuple(tuple(position[token] for token in group) for group in token_groups)
    return groups, tuple(flat), codes


@functools.lru_cache(maxsize=128)
def _readout(tokenizer, kind, labels):
    """The index groups, flattened score tokens and answer codes for one
    (kind, labels) readout, per tokenizer — or the DecisionUnsupported a
    tokenizer without the surface forms raises. Exceptions memoize as
    cached values: an unsupported tokenizer fails once, not per question."""
    try:
        return _readout_impl(tokenizer, kind, labels)
    except DecisionUnsupported as error:
        return error


def prepare_question(tokenizer, spec, temperatures=None):
    """The Question for a validated spec, or DecisionUnsupported.
    `temperatures` is the serving contract's decision_temperatures map."""
    if spec.deterministic:
        return Question(spec, (), (), temperatures=temperatures)
    readout = _readout(tokenizer, spec.kind, spec.labels)
    if isinstance(readout, DecisionUnsupported):
        raise readout
    groups, score_tokens, codes = readout
    return Question(spec, groups, score_tokens, codes, temperatures)


def question_block(question):
    """The QUESTION: body for one question, per its type's contract."""
    spec = question.spec
    instructions = _text(spec.instructions)
    if spec.kind == "noul":
        extra = ""
        if spec.descriptions != spec.labels:
            extra = (
                f"\nYes: {_text(spec.descriptions[0])}"
                f"\nNo: {_text(spec.descriptions[1])}"
            )
        return f"{instructions}{extra}\n\nReply with yes or no only."
    if spec.kind == "score":
        legend = "\n".join(
            f"{index} {_text(description)}"
            for index, description in enumerate(spec.descriptions)
        )
        return (
            f"{instructions}\n\n{legend}\n\n"
            f"Reply with a single digit 0-{len(spec.labels) - 1} only."
        )
    lines = "\n".join(
        _option_line(code, label, description)
        for code, label, description in zip(
            question.codes, spec.labels, spec.descriptions
        )
    )
    return (
        f"{instructions}\n\nOptions:\n{lines}\n\n"
        "Reply with the option code only."
    )


def state_block(state):
    """The state's block before QUESTION: (the json_only style). orjson's
    OPT_INDENT_2 emits the identical bytes json.dumps does here."""
    if isinstance(state, str):
        return f"{state}\n\n"
    if _orjson is not None:
        return _orjson.dumps(state, option=_orjson.OPT_INDENT_2).decode() + "\n\n"
    return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"


@dataclass(frozen=True)
class Judgment:
    """A /v1/judgments row prepared for the decision pathway: the row's
    options rendered as one choice question. `judgment_response` emits the
    same response shape the SemIf pathway reports, with the decision
    readout's version and pooled per-option scores as `option_logits`."""

    row: object
    question: Question

    def judgment_response(self, model, job, result):
        scores = self.question.pooled_scores(list(result.option_logits))
        return {
            "id": self.row["id"],
            "option_ids": [option["id"] for option in self.row["options"]],
            # The pooled scores are reported raw; the contract's calibrated
            # temperature scales them only for the softmax.
            "probabilities": self.question.distribution(scores),
            "option_logits": scores,
            "input_tokens": result.prompt_tokens,
            "answer_token_ids": list(job.score_tokens),
            "prompt_sha256": job.prompt_sha256,
            "prompt_version": PROMPT_VERSION,
            "model": {"id": model},
            "readout": READOUT,
            "probability_status": PROBABILITY_STATUS,
            "forward_seconds": result.start_to_first_token_ms / 1000.0,
            "total_seconds": result.request_wall_ms / 1000.0,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": result.prompt_tokens,
            },
        }


def prompt_prefix(state, bos=""):
    """The prompt up to the question block: identical for every question
    over the same state, so a request encodes it once."""
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return f"{bos}{IM_START}user\n{body}"


def prompt_suffix(question):
    """The question block and the closing markup."""
    return f"{question_block(question)}{IM_END}\n{IM_START}assistant\n"


def prompt_text(state, question, bos=""):
    """The full System One prompt for one question over a state."""
    return prompt_prefix(state, bos) + prompt_suffix(question)


@functools.lru_cache(maxsize=64)
def _prefix_tokens(tokenizer, prefix_text):
    """The shared state prefix's tokens, memoized per tokenizer across
    requests: repeated-state workloads (eval sweeps, judgment batches over
    one state) encode it once."""
    return tuple(tokenizer.encode(prefix_text, add_special_tokens=False))


@functools.lru_cache(maxsize=512)
def _suffix_tokens(tokenizer, suffix_text):
    """A question suffix's tokens, memoized per tokenizer: sweeps that
    repeat questions over different states re-encode nothing."""
    return tuple(tokenizer.encode(suffix_text, add_special_tokens=False))


@functools.lru_cache(maxsize=64)
def _prefix_tail_text(tokenizer, prefix_text):
    """The text of the shared prefix's last tokens; decoding once per state
    lets the seam check run on a short boundary string."""
    tail = _prefix_tokens(tokenizer, prefix_text)[-8:]
    return tokenizer.decode(list(tail))


def _split_prompt_tokens(tokenizer, prefix_text, suffix_text):
    """The canonical encode of prefix + suffix, as the shared prefix's
    cached tokens spliced with the suffix's — checked on a short boundary
    encode because a tokenizer that merges across the seam falls back to
    the whole-prompt encode."""
    prefix = _prefix_tokens(tokenizer, prefix_text)
    suffix = _suffix_tokens(tokenizer, suffix_text)
    boundary = _prefix_tail_text(tokenizer, prefix_text) + suffix_text
    if tuple(tokenizer.encode(boundary, add_special_tokens=False)) != (
        prefix[-8:] + suffix
    ):
        return list(
            tokenizer.encode(prefix_text + suffix_text, add_special_tokens=False)
        )
    return list(prefix) + list(suffix)


@functools.lru_cache(maxsize=64)
def _prefix_digest(prefix_text):
    """A sha256 hasher primed with the shared prefix's bytes, memoized per
    state text. Callers must copy() it before feeding a suffix, so the
    cached hasher is never consumed."""
    return hashlib.sha256(prefix_text.encode())


def _prompt_digest(prefix_text, suffix_text):
    """The whole prompt's sha256 through the cached prefix digest."""
    hasher = _prefix_digest(prefix_text).copy()
    hasher.update(suffix_text.encode())
    return hasher.hexdigest()


class DecisionPathway:
    """The d1 pathway object a frontend holds: Liquid System One prompts,
    grouped surface-form readout, score-batch submission. `temperatures`
    is the serving contract's decision_temperatures map.

    `generates` is False — decision models score only — and `batches` tells
    the endpoint loop to submit each request's questions in one shared-
    prefix frame. The prepare methods run on a host frontend exposing the
    scoring services: tokenizer, max_context, accepts_model, _priority,
    _preparation, _score_job, shared_prefix_states and
    _deadline_checkpoint."""

    generates = False
    batches = True

    def __init__(self, temperatures=None):
        self.temperatures = temperatures or {}

    def systemone_metadata(self):
        """The prompt/readout identity /v1/systemone reports."""
        return {
            "prompt_version": PROMPT_VERSION,
            "readout": READOUT,
            "probability_status": PROBABILITY_STATUS,
        }

    def prepare_judgment(self, host, body, *, deadline, disconnected=None):
        """The d1 pathway's judgment prepare pass: the row's options become
        one choice question in the decision prompt's own markup."""
        unknown = sorted(
            set(body)
            - {"id", "state", "question", "options", "model", "timeout", "priority"}
        )
        if unknown:
            raise APIError(400, f"unsupported fields: {', '.join(unknown)}")
        if not host.accepts_model(body.get("model", host.model)):
            raise APIError(404, f"model {body['model']} not found", "model_not_found")
        try:
            judgments.validate_row(body)
        except ValueError as error:
            raise APIError(400, str(error)) from error
        priority = host._priority(body)
        checkpoint = host._deadline_checkpoint(deadline, disconnected)
        with host._preparation(deadline):
            spec = judgments.SystemOneQuestion(
                kind="choice",
                instructions=body["question"],
                labels=tuple(option["id"] for option in body["options"]),
                descriptions=tuple(
                    option["description"] for option in body["options"]
                ),
                legend=None,
                deterministic=False,
            )
            try:
                question = prepare_question(
                    host.tokenizer, spec, self.temperatures
                )
            except DecisionUnsupported as error:
                raise APIError(
                    500, str(error), "decision_unsupported"
                ) from error
            bos = getattr(host.tokenizer, "bos_token", None)
            bos = bos if isinstance(bos, str) else ""
            try:
                prefix_text = prompt_prefix(body["state"], bos)
                suffix_text = prompt_suffix(question)
                prefix_tokens = _prefix_tokens(host.tokenizer, prefix_text)
                tokens = _split_prompt_tokens(
                    host.tokenizer, prefix_text, suffix_text
                )
            except Exception as error:
                raise APIError(
                    400, "judgment prompt could not be rendered"
                ) from error
            checkpoint()
            if not tokens:
                raise APIError(400, "judgment prompt could not be rendered")
            if len(tokens) > host.max_context:
                raise ContextLengthError(len(tokens), host.max_context)
            job = host._score_job(
                tokens,
                question.score_tokens,
                deadline,
                priority,
                _prompt_digest(prefix_text, suffix_text),
                # The state prefix repeats across judgments over the same
                # state: named a shared prefix, the engine keeps its
                # junction state and the next request resumes past it.
                shared_prefix_tokens=(
                    len(prefix_tokens) if host.shared_prefix_states else 0
                ),
            )
        return job, Judgment(body, question)

    def prepare_systemone(self, host, body, *, deadline, disconnected=None):
        """The d1 pathway's prepare pass: same validation, caps and job
        shape as the SemIf one; the prompt markup and the grouped readout
        are the decision model's own."""
        details = []
        model = body.get("model")
        if not isinstance(model, str) or not model:
            details.append(judgments.detail(["model"], "field required", "missing"))
        elif not host.accepts_model(model):
            details.append(
                judgments.detail(
                    ["model"], f"model {model} is not served by this endpoint"
                )
            )
        state, specs, question_details = judgments.validate_systemone(body)
        details.extend(question_details)
        try:
            priority = host._priority(body)
        except APIError as error:
            details.append(judgments.detail(["priority"], error.message))
        if details:
            raise judgments.SystemOneError(details)
        bos = getattr(host.tokenizer, "bos_token", None)
        bos = bos if isinstance(bos, str) else ""
        jobs = []
        total_tokens = 0
        checkpoint = host._deadline_checkpoint(deadline, disconnected)
        # Every question shares the state's prefix; encode it once and join
        # the question suffix, verified on the first question — a tokenizer
        # that merges across the boundary gets whole-prompt encodes. The
        # budget charges the prefix once plus each question's suffix: the
        # engine prefills the shared prefix once too.
        prefix_text = prefix_tokens = None
        split_encode = None
        with host._preparation(deadline):
            for qid, spec in specs:
                try:
                    question = prepare_question(
                        host.tokenizer, spec, self.temperatures
                    )
                except DecisionUnsupported as error:
                    raise APIError(500, str(error), "decision_unsupported") from error
                if question.deterministic:
                    jobs.append((qid, question, None))
                    continue
                try:
                    if prefix_tokens is None:
                        prefix_text = prompt_prefix(state, bos)
                        prefix_tokens = _prefix_tokens(
                            host.tokenizer, prefix_text
                        )
                        total_tokens += len(prefix_tokens)
                    suffix_text = prompt_suffix(question)
                    # The job carries only its own suffix; the shared prefix
                    # travels by reference, so a batch never copies it per
                    # question.
                    suffix_tokens = _suffix_tokens(
                        host.tokenizer, suffix_text
                    )
                    if split_encode is not True:
                        whole = list(
                            host.tokenizer.encode(
                                prefix_text + suffix_text,
                                add_special_tokens=False,
                            )
                        )
                        if split_encode is None:
                            split_encode = (
                                list(prefix_tokens) + list(suffix_tokens)
                                == whole
                            )
                        if not split_encode:
                            # A tokenizer that merges across the seam gets a
                            # plain full-prompt job that submits on its own.
                            tokens = whole
                            job_prefix = ()
                    if split_encode:
                        tokens = suffix_tokens
                        job_prefix = prefix_tokens
                except Exception as error:
                    raise APIError(
                        400, "question prompt could not be rendered"
                    ) from error
                checkpoint()
                if not tokens:
                    raise APIError(
                        400, "question prompt could not be rendered"
                    )
                if len(job_prefix) + len(tokens) > host.max_context:
                    raise ContextLengthError(
                        len(job_prefix) + len(tokens), host.max_context
                    )
                suffix_count = len(tokens)
                if (
                    total_tokens + suffix_count
                    > judgments.MAX_SYSTEMONE_TOTAL_TOKENS
                ):
                    raise judgments.SystemOneError(
                        [
                            judgments.detail(
                                ["questions", qid],
                                "total prepared question tokens exceed "
                                f"{judgments.MAX_SYSTEMONE_TOTAL_TOKENS}",
                            )
                        ]
                    )
                total_tokens += suffix_count
                jobs.append(
                    (
                        qid,
                        question,
                        host._score_job(
                            tokens,
                            question.score_tokens,
                            deadline,
                            priority,
                            # The state prefix repeats across requests: named
                            # a shared prefix, the engine keeps its junction
                            # state and the next request resumes past it.
                            shared_prefix_tokens=(
                                len(prefix_tokens)
                                if host.shared_prefix_states and job_prefix
                                else 0
                            ),
                            prefix_tokens=job_prefix,
                        ),
                    )
                )
        return jobs

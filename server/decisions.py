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

import json
import os
import re
from dataclasses import dataclass

from . import judgments

IM_START = "<" + "|im_start|" + ">"
IM_END = "<" + "|im_end|" + ">"

PROMPT_VERSION = "d1-system-one-v1"
READOUT = (
    "native full-vocabulary last-position logits max-pooled over each "
    "option's surface forms"
)
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


def _single_token_ids(tokenizer, texts):
    """The texts that encode as exactly one token id, deduplicated."""
    out, seen = [], set()
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in seen:
            out.append(encoded[0])
            seen.add(encoded[0])
    return out


def _aliases(tokenizer, codes):
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


@dataclass(frozen=True)
class Question:
    """A validated question plus its decision readout: the surface-form token
    groups (one per option, in label order) and their flattened distinct ids
    in score-token request order."""

    spec: object
    groups: tuple
    score_tokens: tuple
    codes: tuple = ()

    @property
    def deterministic(self):
        return self.spec.deterministic

    def answer(self, logits):
        """The typed answer. `None` is the no-inference static answer."""
        if logits is None:
            return judgments.systemone_answer(self.spec, [1.0])
        position = {token: index for index, token in enumerate(self.score_tokens)}
        scores = [
            max(float(logits[position[token]]) for token in group)
            for group in self.groups
        ]
        return judgments.systemone_answer(self.spec, judgments.softmax(scores))


def prepare_question(tokenizer, spec):
    """The Question for a validated spec, or DecisionUnsupported."""
    if spec.deterministic:
        return Question(spec, (), ())
    if spec.kind == "noul":
        groups = (
            tuple(_single_token_ids(tokenizer, YES_FORMS)),
            tuple(_single_token_ids(tokenizer, NO_FORMS)),
        )
        if not groups[0] or not groups[1]:
            raise DecisionUnsupported(
                "the served tokenizer has no single-token yes/no forms"
            )
        codes = ()
    elif spec.kind == "score":
        groups = tuple(
            tuple(_single_token_ids(tokenizer, [label])) for label in spec.labels
        )
        if any(not group for group in groups):
            raise DecisionUnsupported(
                "score needs single-token digit forms on this tokenizer"
            )
        codes = ()
    else:
        aliases = _aliases(tokenizer, option_codes(spec.labels))
        groups = tuple(
            (token, *_single_token_ids(tokenizer, [f" {code}"]))
            for (code, token) in aliases
        )
        groups = tuple(
            tuple(token for token in dict.fromkeys(group)) for group in groups
        )
        codes = tuple(code for code, _ in aliases)
    flat = []
    for group in groups:
        for token in group:
            if token not in flat:
                flat.append(token)
    if len(flat) < len(groups) or len(flat) < 2:
        raise DecisionUnsupported("answer slots collapse to one token id")
    if len(flat) > MAX_SCORE_TOKENS:
        raise DecisionUnsupported(
            f"{len(groups)} options need {len(flat)} score tokens; "
            f"at most {MAX_SCORE_TOKENS} are supported"
        )
    return Question(spec, groups, tuple(flat), codes)


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
    """The state's block before QUESTION: (the json_only style)."""
    if isinstance(state, str):
        return f"{state}\n\n"
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
        logits = list(result.option_logits)
        position = {
            token: index
            for index, token in enumerate(self.question.score_tokens)
        }
        scores = [
            max(float(logits[position[token]]) for token in group)
            for group in self.question.groups
        ]
        return {
            "id": self.row["id"],
            "option_ids": [option["id"] for option in self.row["options"]],
            "probabilities": judgments.softmax(scores),
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


def prompt_text(state, question, bos=""):
    """The full System One prompt for one question over a state."""
    body = "" if state is None else f"{state_block(state)}\nQUESTION:\n"
    return (
        f"{bos}{IM_START}user\n{body}{question_block(question)}"
        f"{IM_END}\n{IM_START}assistant\n"
    )

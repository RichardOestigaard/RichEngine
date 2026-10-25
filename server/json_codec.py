"""Strict JSON and compact UTF-8 encoding for API payloads and retained state."""

import json
import math
import re

import orjson


class JSONEncodingError(RuntimeError):
    """A server-side value could not be serialized as JSON."""


_ENCODER = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
# Escape lone surrogates and separators recognized by text-based SSE readers.
_ESCAPED = re.compile("[\ud800-\udfff\u0085\u2028\u2029]")


def _escape(text):
    return _ESCAPED.sub(lambda match: f"\\u{ord(match[0]):04x}", text)


def _reject_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number is outside the supported range")
    return number


def loads(value):
    return json.loads(
        value, parse_constant=_reject_json_constant, parse_float=_finite_json_float
    )


def dumps(value):
    try:
        return _escape(_ENCODER.encode(value))
    except (TypeError, ValueError, RecursionError) as error:
        raise JSONEncodingError("value is not JSON serializable") from error


def encode(value):
    return dumps(value).encode("utf-8")


def loads_compact(value):
    """`loads` for trusted producers where a fast decode suffices.

    orjson rejects what the strict reader accepts — escaped lone surrogates,
    integers past u64 — so those documents take the strict reader instead."""
    try:
        return orjson.loads(value)
    except ValueError:
        return loads(value)


def encode_compact(value, *, sort_keys=False):
    """Compact UTF-8 bytes for in-process keys, digests and retained bytes.

    Unlike `dumps`, the output never crosses a text-based stream boundary, so
    the separator and surrogate escapes need not apply: orjson encodes what it
    can, and the values it cannot — lone surrogates, integers past u64,
    non-string keys — take the strict encoder instead. orjson writes
    non-finite floats as null, so callers pass only values read through
    `loads`, where such floats cannot occur."""
    try:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS if sort_keys else 0)
    except (TypeError, ValueError, RecursionError):
        return dumps(value).encode("utf-8")


def encoded_size(value):
    """Count encoded bytes without materializing the complete document."""
    try:
        return sum(
            len(_escape(part).encode("utf-8")) for part in _ENCODER.iterencode(value)
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise JSONEncodingError("value is not JSON serializable") from error

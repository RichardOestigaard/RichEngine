"""Tool and response-format schema normalization and llguidance grammars."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from urllib.parse import unquote

from jsonschema.exceptions import SchemaError
from llguidance import LLMatcher

from .errors import APIError
from .schema_validation import build_validator, json_objects, subschemas

MAX_JSON_NESTING = 256

# The chat template's tool-call framing. The projector that parses output and
# the grammars must agree byte for byte, so every piece is spelled here once.
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "\n<function="
FUNCTION_CLOSE = "</function>\n</tool_call>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "\n</parameter>\n"
# The chat template's think-close token id where no tokenizer contract
# supplies one (tests); the server always passes its validated contract's.
THINK_END_TOKEN_ID = 248069


def function_opening(name):
    """What follows TOOL_CALL_OPEN in a call of tool `name`, up to its
    arguments."""
    return f"{FUNCTION_OPEN}{name}>\n"


@dataclass(frozen=True)
class ToolDialect:
    """One model family's tool-call spelling: the grammar writes it and the
    output projector reads it, so both take it from the same fields.

    ``kind`` "xml" spells calls as markup elements; "python" spells them as
    ``name(argument='value')`` inside a bracketed call list."""

    # A short status/log name.
    name: str
    kind: str = "xml"
    # The text that opens one call; content scanning and the TEXT barrier
    # key on it. XML dialects scan for the element-open; the python dialect
    # scans for the call-list opener.
    call_open: str = TOOL_CALL_OPEN
    # Between the call-open marker and the tool's name, and after it.
    name_prefix: str = FUNCTION_OPEN
    name_close: str = ">\n"
    # Parameter framing: param_open + name + param_name_close + value +
    # param_close.
    param_open: str = PARAMETER_OPEN
    param_name_close: str = ">\n"
    param_close: str = PARAMETER_CLOSE
    # The text that ends one call.
    call_close: str = FUNCTION_CLOSE
    # An optional separator a model may write between calls, consumed before
    # each call's marker ("" where calls run together).
    call_separator: str = ""
    # Whether string values may be wrapped in a CDATA section.
    cdata: bool = False
    # Special-token spellings generation decodes to text: model families
    # mark their call markup special, and the decoder must keep them or the
    # projector never sees them.
    structural: tuple = ()
    # Characters a parameter name may not contain, beyond surrounding space.
    param_name_forbidden: str = "<>\n\r"
    # The lexer pattern an undeclared parameter name matches.
    extra_name_pattern: str = r"[^<>\n\r]+"
    # Text an enum's string value may not spell (the value's closing markup).
    value_barrier: str | None = PARAMETER_CLOSE.rstrip("\n")
    # The regex fragment, an alternation of call markup spellings, that
    # unconstrained TEXT must not contain.
    barrier: str = "<tool_call>"

    def call_opening(self, name):
        """The tokens that begin a call of tool `name`, through the name."""
        if self.kind == "python":
            return f"{self.call_open}{name}("
        return f"{self.call_open}{self.name_prefix}{name}{self.name_close}"

    def argument_grammar(self, schema):
        """The lark grammar of one tool's argument list."""
        if self.kind == "python":
            return _python_argument_grammar(schema)
        return _argument_grammar(schema, self)

    def call_rules(self, names, separator, parallel):
        """(rules, calls, maybe): the lark rules tool_0.. spelling one call
        of each tool each, where the arguments of tool i come from the side
        grammar ``arguments_i``; ``calls`` a run of one or more calls, and
        ``maybe`` that run or nothing."""
        if self.kind == "python":
            rules = [
                f'tool_{index}: {json.dumps(name + "(")} '
                f"@arguments_{index} {json.dumps(')')}"
                for index, name in enumerate(names)
            ]
            choice = "(" + " | ".join(
                f"tool_{index}" for index in range(len(names))
            ) + ")"
            calls = (
                f'{separator} {json.dumps(self.call_open)} {choice}'
                + (f' ({json.dumps(", ")} {choice})*' if parallel else "")
                + f' {json.dumps("]<|tool_call_end|>")}'
            )
            return rules, calls, f"({calls})?"
        optional_separator = (
            f"{json.dumps(self.call_separator)}? " if self.call_separator else ""
        )
        rules = [
            f"tool_{index}: {separator} {optional_separator}"
            f"{json.dumps(self.call_opening(name))} "
            f"@arguments_{index} {json.dumps(self.call_close)}"
            for index, name in enumerate(names)
        ]
        choice = "(" + " | ".join(
            f"tool_{index}" for index in range(len(names))
        ) + ")"
        return (
            rules,
            choice + ("+" if parallel else ""),
            choice + ("*" if parallel else "?"),
        )


def _python_string_literal(value):
    """A python-call string literal as the chat template spells it:
    single-quoted, with backslash, quote, newline and carriage return
    escaped."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f"'{escaped}'"


def _python_scalar(value):
    """The spelling of one scalar argument value in a python-call list."""
    if isinstance(value, str):
        return _python_string_literal(value)
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    return json.dumps(value, allow_nan=False)


# A python-call string: an escape is a backslash followed by the character it
# spells (the template produces \\, \', \n and \r only).
_PYTHON_STRING = r"/'([^'\\\n\r]|\\[\\'nr])*'/"
_PYTHON_NUMBER = r"/-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/"


def _python_value_rules(rule, value_schema):
    """The lark rules naming `rule` the spelling of one argument value."""
    rules = []
    string_schema = raw_string_schema(value_schema)
    schema = _grammar_compatible_schema(value_schema)
    if string_schema is not None:
        if string_schema[0] == "raw":
            body = _PYTHON_STRING
        else:
            choices = [
                json.dumps(_python_string_literal(value)) if value else "''"
                for value in string_schema[1]
            ]
            body = "(" + " | ".join(choices) + ")"
        rules.append(f"{rule}: {body}")
        return rules
    kind = schema.get("type") if isinstance(schema, dict) else None
    kinds = kind if isinstance(kind, list) else [kind]
    union = schema.get("anyOf", schema.get("oneOf")) if isinstance(schema, dict) else None
    if union:
        options = []
        for option_index, option_schema in enumerate(union):
            option_rule = f"{rule}_option_{option_index}"
            rules.extend(_python_value_rules(option_rule, option_schema))
            options.append(option_rule)
        rules.append(f"{rule}: {' | '.join(options)}")
        return rules
    if "enum" in schema or "const" in schema:
        values = schema["enum"] if "enum" in schema else [schema["const"]]
        if all(
            value is None or isinstance(value, (str, bool, int, float))
            for value in values
        ):
            body = " | ".join(json.dumps(_python_scalar(value)) for value in values)
            rules.append(f"{rule}: ({body})")
            return rules
    if kinds == ["boolean"]:
        rules.append(f"{rule}: (True | False)")
        return rules
    if kinds == ["null"]:
        rules.append(f"{rule}: None")
        return rules
    if kinds and all(k in ("integer", "number") for k in kinds):
        rules.append(f"{rule}: {_PYTHON_NUMBER}")
        return rules
    rules.append(
        f"{rule}: %json {json.dumps(schema, separators=(',', ':'))}"
    )
    return rules


def _python_argument_grammar(schema):
    """The argument grammar of a python-call dialect: ``name=value`` pairs
    joined by ", " inside the call's parentheses.

    Parameters generate in their declared order; optional ones may be
    skipped. ``chain_i`` reads parameter i on, and ``seq_i`` the same once a
    value is out, where a comma must precede the next parameter. The chains
    end at ``chain_n``/``seq_n``, which read undeclared ``name=value`` pairs
    when the schema allows additional properties."""
    properties = []
    for name, value_schema in schema["properties"].items():
        if value_schema is False:
            if name in schema["required"]:
                raise APIError(
                    400, f"required tool parameter cannot have a value: {name}"
                )
            continue
        properties.append((name, value_schema))
    required = set(schema["required"])
    rules = []
    for index, (name, value_schema) in enumerate(properties):
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or any(character in name for character in "=(),'\"\\ \t\n\r")
        ):
            raise APIError(400, "invalid tool parameter name")
        rules.extend(_python_value_rules(f"value_{index}", value_schema))
    n = len(properties)
    for index in range(n):
        name, _value_schema = properties[index]
        call = f"{json.dumps(name + '=')} value_{index}"
        if name in required:
            rules.append(f"chain_{index}: {call} seq_{index + 1}")
            rules.append(f'seq_{index}: ", " {call} seq_{index + 1}')
        else:
            rules.append(
                f"chain_{index}: ({call} seq_{index + 1})? chain_{index + 1}"
            )
            rules.append(
                f'seq_{index}: (", " {call} seq_{index + 1})? seq_{index + 1}'
            )
    additional = schema["additionalProperties"]
    if additional is False:
        rules.append(f"chain_{n}:")
        rules.append(f"seq_{n}:")
    else:
        declared = " | ".join(json.dumps(name) for name, _ in properties)
        rules.append(
            "EXTRA_NAME: /[A-Za-z0-9_-]+/" + (f" & ~({declared})" if declared else "")
        )
        rules.extend(_python_value_rules("extra_value", additional))
        rules.append(f"chain_{n}: (EXTRA_NAME \"=\" extra_value tail)?")
        rules.append(f'seq_{n}: (", " EXTRA_NAME \"=\" extra_value tail)?')
        rules.append('tail: (", " EXTRA_NAME "=" extra_value tail)?')
    return "%llguidance {}\nstart: chain_0\n" + "\n".join(rules) + "\n"


# The framing each supported chat template writes its calls in, detected at
# startup from a rendered canary call.
QWEN3_XML = ToolDialect(name="qwen3-xml")

MINICPM5_XML = ToolDialect(
    name="minicpm5-xml",
    call_open="<function",
    name_prefix=' name="',
    name_close='">',
    param_open='<param name="',
    param_name_close='">',
    param_close="</param>",
    call_close="</function>",
    call_separator="<tool_sep>",
    cdata=True,
    structural=("<function", "</function>", "<param", "</param>"),
    param_name_forbidden='"<>\n\r',
    extra_name_pattern=r"[^<>\n\r\"]+",
    value_barrier="</param>",
    barrier="<function|<tool_sep>",
)

LFM25_PYTHON = ToolDialect(
    name="lfm25-python",
    kind="python",
    call_open="<|tool_call_start|>[",
    call_close="]<|tool_call_end|>",
    structural=("<|tool_call_start|>", "<|tool_call_end|>"),
    param_name_forbidden="=(),'\"\\ \t\n\r",
    value_barrier=None,
    barrier=r"<\|tool_call_start\|>",
)

_DIALECT_MARKERS = (
    ("<|tool_call_start|>", LFM25_PYTHON),
    ('<function name="', MINICPM5_XML),
    ("<function=", QWEN3_XML),
    ("<tool_call>", QWEN3_XML),
)


def detect_tool_dialect(rendered):
    """The dialect of a rendered canary tool call, or None where the text
    spells no known framing (requests then keep the default's)."""
    if not isinstance(rendered, str):
        return None
    for marker, dialect in _DIALECT_MARKERS:
        if marker in rendered:
            return dialect
    return None


# Framing projects each tool's fields through schema composition and copies
# the root schema into every field that refers to it. Pathological schemas
# make that quadratic or exponential in their size, so framing all tools of
# a request may produce about this many bytes of schemas.
MAX_FRAMED_SCHEMA_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ToolPolicy:
    validators: dict
    schemas: dict
    required: bool
    parallel: bool
    namespaces: dict = field(default_factory=dict)
    # The served model's tool-call framing; None means the default dialect.
    dialect: ToolDialect | None = None

    @cached_property
    def argument_schemas(self):
        budget = [MAX_FRAMED_SCHEMA_BYTES]
        return {
            name: tool_argument_schema(schema, budget)
            for name, schema in self.schemas.items()
        }


def json_value(value):
    value = value.strip()
    try:
        parsed = json.loads(value, parse_constant=str)
        pending = [(parsed, 0)]
        while pending:
            item, depth = pending.pop()
            if isinstance(item, dict):
                children = item.values()
            elif isinstance(item, list):
                children = item
            else:
                continue
            depth += 1
            if depth > MAX_JSON_NESTING:
                return value
            pending.extend((child, depth) for child in children)
        json.dumps(parsed, allow_nan=False)
        return parsed
    except (ValueError, RecursionError):
        return value


def _remote_ref(schema):
    for node in subschemas(schema):
        if isinstance(node, dict):
            if "$schema" in node and not isinstance(node["$schema"], str):
                raise APIError(400, "$schema must be a string")
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                if key not in node:
                    continue
                # Draft 4 leaves $ref unchecked, and validation fails on
                # anything but a string with an error that is not a
                # validation error.
                ref = node[key]
                if not isinstance(ref, str):
                    raise APIError(400, f"{key} must be a string")
                if not ref.startswith("#"):
                    return ref
    return None


SCHEMA_ANNOTATIONS = {
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
}


# Raw string parameters are framed by the tool-call grammar and validated
# against their complete JSON Schema after parsing. Keeping these assertions
# out of the grammar preserves multiline string payloads.
STRING_SCHEMA_POST_VALIDATION_KEYWORDS = {
    "allOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
}


# The grammar compiler expands these keywords into work proportional to their
# values: a rule per required or optional array item, a state per divisor
# residue. A tiny schema with a huge bound would exhaust memory, so larger
# bounds are left to validation of the complete output.
GRAMMAR_BOUND_KEYWORDS = ("minItems", "maxItems", "multipleOf")
MAX_GRAMMAR_BOUND = 64

# Checking a pattern compiles a grammar, and a client sends the same tools and
# output schema on every turn, so the answers for this many patterns are kept.
PATTERN_CHECK_CACHE_SIZE = 1024

# The whitespace a model chooses between the tokens of constrained output, in
# JSON and around tool calls, is bounded: a model that prefers whitespace to
# every token the grammar allows next would otherwise write it until
# max_tokens, as Qwen models do, most of all under speculative decoding (vLLM
# #38696, #50989). llama.cpp and Outlines bound it more tightly; 64 characters
# still take pretty printing 15 levels deep at four spaces. Whitespace inside
# strings is content and unbounded.
MAX_WHITESPACE = 64
WHITESPACE = rf"[ \t\n\r]{{0,{MAX_WHITESPACE}}}"
WHITESPACE_RULE = f"WS: /{WHITESPACE}/"


@lru_cache(maxsize=PATTERN_CHECK_CACHE_SIZE)
def _grammar_takes_pattern(pattern):
    """Whether the grammar compiler takes a JSON Schema ``pattern``. It keeps
    the pattern's search semantics but rejects look-around, word boundaries
    and backreferences, which are left to validation of the complete output."""
    string = json.dumps({"type": "string", "pattern": pattern})
    return not LLMatcher.validate_grammar(f"%llguidance {{}}\nstart: %json {string}\n")


def _grammar_compatible_schema(schema):
    """Guide generation with supported constraints; validate the original."""
    output = copy.deepcopy(schema)
    for node in subschemas(output):
        if isinstance(node, dict):
            node.pop("propertyNames", None)
            pattern = node.pop("pattern", None)
            if isinstance(pattern, str) and _grammar_takes_pattern(pattern):
                node["pattern"] = pattern
    # A local reference can point anywhere in the document, so any object may
    # be compiled as a schema.
    for node in json_objects(output):
        for key in GRAMMAR_BOUND_KEYWORDS:
            bound = node.get(key)
            if isinstance(bound, (int, float)) and bound > MAX_GRAMMAR_BOUND:
                del node[key]
    if output is True:
        # Any value, with the whitespace between its tokens bounded.
        output = {}
    if isinstance(output, dict):
        output["x-guidance"] = {"lenient": True, "whitespace_pattern": WHITESPACE}
    return output


# Keywords that apply further schemas to the instance a schema describes.
IN_PLACE_APPLICATORS = {
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "dependencies",
    "extends",
    "$dynamicRef",
    "$recursiveRef",
}
SCHEMA_IDENTIFIERS = {
    "$schema",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$defs",
    "definitions",
}
# Keywords by which an object says which properties it takes beyond those it
# declares.
MORE_PROPERTIES = {"additionalProperties", "unevaluatedProperties", "patternProperties"}


def _extended(node):
    """Whether other schemas apply to the instance `node` describes, so that
    `node` may declare only some of its properties."""
    keys = set(node) - SCHEMA_ANNOTATIONS - SCHEMA_IDENTIFIERS
    if keys & IN_PLACE_APPLICATORS or ("$ref" in keys and keys != {"$ref"}):
        return True
    unions = keys & {"anyOf", "oneOf"}
    return bool(unions and keys - unions - {"type"})


def _strict_schema(schema):
    """The schema a strict tool's arguments are generated to, as vLLM and
    SGLang generate them with XGrammar's strict mode: an object that does not
    say which properties it takes beyond those it declares takes none, and
    an array that does not say which items it takes beyond its leading ones
    takes none. Validation keeps the declared schema.

    A schema that other schemas of the same instance extend, as an allOf
    does, may declare only some of its properties, and closing it would
    refuse the others; a schema composed that way is left as declared."""
    if any(_extended(node) for node in subschemas(schema) if isinstance(node, dict)):
        return schema
    output = copy.deepcopy(schema)
    for node in subschemas(output):
        if not isinstance(node, dict):
            continue
        keys = set(node)
        # A union or a reference describes its instance by its alternatives
        # or its target, which are closed in their own places.
        if keys & {"anyOf", "oneOf", "$ref"}:
            continue
        kind = node.get("type")
        kinds = kind if isinstance(kind, list) else [kind]
        if "object" in kinds or (kind is None and "properties" in keys):
            if not keys & MORE_PROPERTIES:
                node["additionalProperties"] = False
        if "array" in kinds or (kind is None and "prefixItems" in keys):
            if isinstance(node.get("items"), list):
                if not keys & {"additionalItems", "unevaluatedItems"}:
                    node["additionalItems"] = False
            elif not keys & {"items", "unevaluatedItems"}:
                node["items"] = False
    return output


def _lookup_tool_reference(ref, root):
    if not isinstance(ref, str) or not ref.startswith("#"):
        raise APIError(400, "unsupported tool parameter reference")
    fragment = unquote(ref[1:])
    if not fragment:
        return root
    if fragment.startswith("/"):
        current = root
        for part in fragment[1:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list) and re.fullmatch(r"0|[1-9][0-9]*", part):
                index = int(part)
                if index < len(current):
                    current = current[index]
                    continue
            elif isinstance(current, dict) and part in current:
                current = current[part]
                continue
            raise APIError(400, f"unresolved tool parameter reference: {ref}")
        return current
    for node in subschemas(root):
        if isinstance(node, dict) and fragment in (
            node.get("$anchor"),
            node.get("$dynamicAnchor"),
        ):
            return node
    raise APIError(400, f"unresolved tool parameter reference: {ref}")


def _resolve_tool_schema(schema, root):
    seen = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        constrained = bool(
            set(schema) - {"$ref", "$defs", "definitions"} - SCHEMA_ANNOTATIONS
        )
        if ref in seen:
            raise APIError(400, "cyclic direct tool parameter reference")
        seen.add(ref)
        current = _lookup_tool_reference(ref, root)
        if constrained:
            # Keep the reference and its intersecting assertions together in
            # the JSON grammar instead of choosing a raw-string encoding.
            return None
        schema = current
    return schema


def _schema_with_root(schema, root):
    def local_refs(value):
        for node in subschemas(value):
            ref = node.get("$ref") if isinstance(node, dict) else None
            if isinstance(ref, str) and ref.startswith("#"):
                yield node

    if next(local_refs(schema), None) is None:
        return schema
    root_defs = root.get("$defs", {}) if isinstance(root, dict) else {}
    schema_defs = schema.get("$defs", {}) if isinstance(schema, dict) else {}
    name = "__splash_root"
    while name in root_defs or name in schema_defs:
        name += "_"
    prefix = f"#/$defs/{name}"

    def rebase(value):
        output = copy.deepcopy(value)
        for node in local_refs(output):
            if node["$ref"] == "#" or node["$ref"].startswith("#/"):
                node["$ref"] = prefix + node["$ref"][1:]
        return output

    output = rebase(schema)
    definitions = dict(output.get("$defs", {}))
    definitions[name] = rebase(root)
    output["$defs"] = definitions
    return output


def raw_string_schema(schema, value_barrier=PARAMETER_CLOSE.rstrip("\n")):
    """How a parameter value is written: ("raw", None) as raw text,
    ("literal", values) as one of `values`, or None as JSON. Framing makes
    each parameter schema self-contained, so it is its own reference root.
    `value_barrier` is the dialect's closing markup an enum value may not
    spell, or None where values are escaped."""
    return _raw_string_schema(schema, schema, frozenset(), {}, value_barrier)


def _raw_string_schema(schema, root, ancestors, results, value_barrier):
    schema = _resolve_tool_schema(schema, root)
    if not isinstance(schema, dict):
        return None
    if id(schema) in ancestors:
        raise APIError(400, "cyclic tool parameter alternatives without a nested value")
    ancestors = ancestors | {id(schema)}
    union = schema.get("anyOf", schema.get("oneOf"))
    if union is not None:
        options = []
        has_other_type = False
        allows_null = False
        for option_schema in union:
            resolved = _resolve_tool_schema(option_schema, root)
            null_only = isinstance(resolved, dict) and (
                resolved.get("type") == "null"
                or resolved.get("const", object()) is None
                or (
                    isinstance(resolved.get("enum"), list)
                    and resolved["enum"]
                    and all(value is None for value in resolved["enum"])
                )
            )
            if null_only:
                allows_null = True
                continue
            # A definition reached through several references is read once.
            if id(resolved) not in results:
                results[id(resolved)] = _raw_string_schema(
                    option_schema, root, ancestors, results, value_barrier
                )
            option = results[id(resolved)]
            if option is None:
                has_other_type = True
            else:
                options.append(option)
        if not options:
            return None
        if has_other_type:
            return None
        if allows_null:
            return None
        if set(schema) - {"anyOf", "oneOf"} - SCHEMA_ANNOTATIONS:
            return None
        if any(option[0] == "raw" for option in options):
            return "raw", None
        values = sum((option[1] for option in options), [])
        return "literal", values

    schema_type = schema.get("type")
    if isinstance(schema_type, list) and "string" in schema_type:
        if set(schema_type) - {"string", "null"}:
            return None
        if "null" in schema_type:
            return None
        schema_type = "string"
    values = schema.get("enum")
    if "const" in schema:
        values = [schema["const"]]
    if schema_type == "null":
        return None
    if schema_type != "string" and not (
        isinstance(values, list) and any(isinstance(value, str) for value in values)
    ):
        return None
    unsupported = (
        set(schema)
        - {
            "type",
            "enum",
            "const",
            "$defs",
            "definitions",
        }
        - SCHEMA_ANNOTATIONS
        - STRING_SCHEMA_POST_VALIDATION_KEYWORDS
    )
    if unsupported:
        return None
    if values is None:
        return "raw", None
    if any(value is not None and not isinstance(value, str) for value in values):
        return None
    if None in values:
        return None
    if value_barrier is not None and any(
        isinstance(value, str) and value_barrier in value for value in values
    ):
        raise APIError(400, "string tool parameter enum contains XML framing")
    return "literal", values


def _schema_combination(keyword, values):
    identity = keyword == "allOf"
    # A definition reached through several references combines once.
    values = list(
        {id(value): value for value in values if value is not identity}.values()
    )
    if not values:
        return identity
    if any(value is not identity and isinstance(value, bool) for value in values):
        return not identity
    if len(values) == 1:
        return values[0]
    combined = {keyword: values}
    # Keep the raw-string transport when every union branch, or at least one
    # intersection, requires a string. Other unions use JSON-encoded values.
    strings = [value.get("type") == "string" for value in values]
    if all(strings) if keyword == "anyOf" else any(strings):
        combined["type"] = "string"
    return combined


def _json_size(value, limit):
    """Approximate serialized size of ``value``, counted no further than ``limit``."""
    size, pending = 0, [value]
    while pending and size <= limit:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        size += len(value) if isinstance(value, str) else 1
    return size


def tool_argument_schema(root, budget):
    """Project object fields for XML framing; validate the untouched schema.

    Cross-field assertions remain on ToolPolicy.validators. This projection
    preserves the set of possible field values rather than choosing a branch
    before the model has supplied the discriminator or dependent properties.
    """

    def charge(size):
        budget[0] -= size
        if budget[0] < 0:
            raise APIError(400, "tool parameter schemas are too complex")

    def combine(shapes, union=False):
        if union:
            shapes = [shape for shape in shapes if shape is not None]
            if not shapes:
                return None
        elif any(shape is None for shape in shapes):
            return None
        if not shapes:
            return {"properties": {}, "required": [], "additionalProperties": True}
        names = dict.fromkeys(name for shape in shapes for name in shape["properties"])
        charge(len(names) * len(shapes))
        required = set(shapes[0]["required"])
        for shape in shapes[1:]:
            if union:
                required.intersection_update(shape["required"])
            else:
                required.update(shape["required"])
        keyword = "anyOf" if union else "allOf"
        return {
            "properties": {
                name: _schema_combination(
                    keyword,
                    [
                        shape["properties"].get(name, shape["additionalProperties"])
                        for shape in shapes
                    ],
                )
                for name in names
            },
            "required": sorted(required),
            "additionalProperties": _schema_combination(
                keyword, [shape["additionalProperties"] for shape in shapes]
            ),
        }

    references = {}

    def project(node, visiting):
        if node is False:
            return None
        if node is True:
            node = {}
        if not isinstance(node, dict):
            raise APIError(400, "tool parameters must allow a JSON object")
        kind = node.get("type", "object")
        if kind != "object" and not (isinstance(kind, list) and "object" in kind):
            return None
        properties = dict(node.get("properties", {}))
        additional = node.get("additionalProperties", True)
        patterns = list(node.get("patternProperties", {}).values())
        if patterns:
            additional = _schema_combination("anyOf", [additional, *patterns])
        required = node.get("required", [])
        for name in required:
            properties.setdefault(name, additional)
        shape = {
            "properties": properties,
            "required": required,
            "additionalProperties": additional,
        }
        shapes = [shape]
        ref = node.get("$ref")
        if ref is not None:
            if ref in visiting:
                raise APIError(400, "cyclic direct tool argument reference")
            if ref not in references:
                resolved = _lookup_tool_reference(ref, root)
                references[ref] = project(resolved, visiting | {ref})
            shapes.append(references[ref])
        for child in node.get("allOf", []):
            shapes.append(project(child, visiting))
        for keyword in ("anyOf", "oneOf"):
            if keyword in node:
                shapes.append(
                    combine([project(child, visiting) for child in node[keyword]], True)
                )
        if "if" in node:
            shapes.append(
                combine(
                    [
                        project(node.get("then", {}), visiting),
                        project(node.get("else", {}), visiting),
                    ],
                    True,
                )
            )
        for child in node.get("dependentSchemas", {}).values():
            shapes.append(
                combine([project(child, visiting), project({}, visiting)], True)
            )
        choices = [node["const"]] if "const" in node else node.get("enum")
        if choices is not None:
            objects = [value for value in choices if isinstance(value, dict)]
            if not objects:
                return None
            shapes.append(
                combine(
                    [
                        {
                            "properties": {
                                name: {"const": value} for name, value in obj.items()
                            },
                            "required": list(obj),
                            "additionalProperties": False,
                        }
                        for obj in objects
                    ],
                    True,
                )
            )
        return combine(shapes)

    def framed(value):
        # Shared definitions repeat in the serialized grammar; count them all.
        charge(_json_size(value, budget[0]))
        framed_value = _schema_with_root(value, root)
        if framed_value is not value:
            charge(_json_size(root, budget[0]))
        return framed_value

    shape = project(root, set())
    if shape is None:
        raise APIError(400, "tool parameters must allow a top-level JSON object")
    shape["type"] = "object"
    shape["properties"] = {
        name: framed(value) for name, value in shape["properties"].items()
    }
    shape["additionalProperties"] = framed(shape["additionalProperties"])
    return shape


def _parameter_rules(rule, prefix, value_schema, dialect):
    rules = []
    closing = json.dumps(dialect.param_close)
    string_schema = raw_string_schema(value_schema, dialect.value_barrier)
    value_schema = _grammar_compatible_schema(value_schema)
    if string_schema is None:
        rules.append(
            f"{rule}: {prefix} "
            f"%json {json.dumps(value_schema, separators=(',', ':'))} "
            f"{closing}"
        )
    elif string_schema[0] == "raw":
        value_rule = f"{rule}_value"
        rules.append(f"{rule}: {prefix} {value_rule}")
        rules.append(f"{value_rule}[suffix={closing}]: /(?s:.*)/")
    else:
        choices = []
        for choice_index, value in enumerate(string_schema[1]):
            if value:
                choices.append(json.dumps(value))
            else:
                empty_rule = f"{rule}_empty_{choice_index}"
                rules.append(f"{empty_rule}:")
                choices.append(empty_rule)
        rules.append(f"{rule}: {prefix} ({' | '.join(choices)}) {closing}")
    return rules


def _argument_grammar(schema, dialect=None):
    if dialect is None:
        dialect = QWEN3_XML
    properties = schema["properties"]
    required = schema["required"]
    rules = []
    sequence = []
    required_sequence = []
    optional_sequence = []
    # A name is a lexeme of its own: one spanning "<parameter=url>" would win
    # over an extra name that starts like it, and lexing cannot back off.
    name_open = json.dumps(dialect.param_open)
    name_close = json.dumps(dialect.param_name_close)
    for index, (name, value_schema) in enumerate(properties.items()):
        if value_schema is False:
            if name in required:
                raise APIError(
                    400, f"required tool parameter cannot have a value: {name}"
                )
            continue
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or any(character in name for character in dialect.param_name_forbidden)
        ):
            raise APIError(400, "invalid tool parameter name")
        rule = f"parameter_{index}"
        item = rule + ("" if name in required else "?")
        sequence.append(item)
        (required_sequence if name in required else optional_sequence).append(item)
        prefix = f"{name_open} {json.dumps(name)} {name_close}"
        rules.extend(_parameter_rules(rule, prefix, value_schema, dialect))
    additional = schema["additionalProperties"]
    if additional is not False:
        # Extra names are the complement of the declared names.
        declared = " | ".join(json.dumps(name) for name in properties)
        rules.append(
            f"EXTRA_NAME: /{dialect.extra_name_pattern}/"
            + (f" & ~({declared})" if declared else "")
        )
        prefix = f"{name_open} EXTRA_NAME {name_close}"
        rules.extend(_parameter_rules("extra", prefix, additional, dialect))
    # A skipped optional field cannot be revisited in an ordered grammar.
    # Also accept required-first order so a model that starts with the required
    # fields can still supply earlier optional fields. Two linear sequences
    # retain required/unique fields without enumerating every permutation.
    required_first = required_sequence + optional_sequence
    start = " ".join(sequence)
    if required_first != sequence:
        start = f"({start}) | ({' '.join(required_first)})"
    if additional is not False:
        start = f"({start}) extra*" if start else "extra*"
    return (
        "%llguidance {}\nstart:"
        + (" " + start if start else "")
        + "\n"
        + "\n".join(rules)
        + "\n"
    )


def json_grammar(schema, thinking, *, think_end_id=None):
    start = "start: " + ("think " if thinking else "") + "WS %json "
    grammar = [
        "%llguidance {}",
        start + json.dumps(_grammar_compatible_schema(schema), separators=(",", ":")),
    ]
    if thinking:
        think_end = (
            THINK_END_TOKEN_ID if think_end_id is None else think_end_id
        )
        grammar.append(f"think: TEXT <[{think_end}]>")
        grammar.append(r"TEXT: /(?s:.*)/ & ~/(?s:.*)<\/think>(?s:.*)/")
    grammar.append(WHITESPACE_RULE)
    return "\n".join(grammar) + "\n"


def normalize_response_format(value):
    if value in (None, {"type": "text"}):
        return None, None
    if not isinstance(value, dict):
        raise APIError(400, "response_format must be an object")
    kind = value.get("type")
    if kind == "json_object":
        schema = {"type": "object"}
    elif kind == "json_schema":
        wrapper = value.get("json_schema")
        schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
        if not isinstance(schema, (dict, bool)):
            raise APIError(400, "response_format.json_schema.schema is required")
    else:
        raise APIError(400, "unsupported response_format")
    if ref := _remote_ref(schema):
        raise APIError(400, f"remote schema reference is not allowed: {ref}")
    try:
        validator = build_validator(schema)
    except SchemaError as error:
        raise APIError(400, f"invalid response schema: {error.message}") from error
    return schema, validator


def normalize_tools(tools, tool_choice, parallel, namespaces=None, dialect=None):
    if parallel is None:
        parallel = True
    if not isinstance(parallel, bool):
        raise APIError(400, "parallel_tool_calls must be a boolean")
    if tools is None:
        if tool_choice not in (None, "none", "auto"):
            raise APIError(400, "tool_choice requires tools")
        return None, None
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array")
    validators = {}
    schemas = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise APIError(400, "only function tools are supported")
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) is None
        ):
            raise APIError(400, "tool name must match [A-Za-z0-9_-]{1,128}")
        if name in validators:
            raise APIError(400, f"duplicate tool name: {name}")
        schema = function.get("parameters")
        if schema is None:
            schema = {}
        if not isinstance(schema, (dict, bool)):
            raise APIError(400, f"invalid tool schema for {name}")
        strict = function.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise APIError(400, f"strict must be a boolean for tool {name}")
        if ref := _remote_ref(schema):
            raise APIError(400, f"remote tool schema reference is not allowed: {ref}")
        try:
            validators[name] = build_validator(schema)
            schemas[name] = _strict_schema(schema) if strict else schema
        except SchemaError as error:
            raise APIError(
                400, f"invalid tool schema for {name}: {error.message}"
            ) from error
    choice = "auto" if tool_choice is None else tool_choice
    if not tools:
        if choice not in ("auto", "none"):
            raise APIError(400, "tool_choice requires at least one tool")
        return None, None
    if choice == "none":
        # The prompt keeps every tool; the grammar and validators allow no call.
        validators, schemas = {}, {}
    elif isinstance(choice, dict):
        function = choice.get("function", {})
        name = function.get("name") if isinstance(function, dict) else None
        if (
            choice.get("type") != "function"
            or not isinstance(name, str)
            or name not in validators
        ):
            raise APIError(400, "invalid named tool_choice")
        # The prompt keeps every tool; the grammar and validators force the
        # call, exactly one, as a forced function is defined.
        validators = {name: validators[name]}
        schemas = {name: schemas[name]}
        parallel = False
    elif choice not in ("auto", "required"):
        raise APIError(400, "invalid tool_choice")
    policy = ToolPolicy(
        validators,
        schemas,
        choice == "required" or isinstance(choice, dict),
        parallel,
        namespaces or {},
        dialect or QWEN3_XML,
    )
    return tools, policy


THINK_END = "</think>"


def tool_grammar(policy, thinking, response_schema=None, *, think_end_id=None):
    dialect = policy.dialect or QWEN3_XML
    try:
        arguments = [
            dialect.argument_grammar(schema)
            for schema in policy.argument_schemas.values()
        ]
    except (AttributeError, TypeError) as error:
        # An older declared dialect leaves newer keywords unchecked, so framing
        # can meet any JSON value where it reads part of a schema.
        raise APIError(400, "unsupported tool parameter schema") from error
    # Text of the model's own may come before a call only where the answer
    # may be text. A required call, like a JSON answer, stands apart from the
    # reasoning and from other calls by whitespace alone.
    separator = "WS" if policy.required or response_schema is not None else "TEXT"
    side_grammars = []
    for index, grammar in enumerate(arguments):
        side_grammars.append(
            {"name": f"arguments_{index}", "lark_grammar": grammar}
        )
    tag_rules, calls, maybe_calls = dialect.call_rules(
        list(policy.argument_schemas), separator, policy.parallel
    )
    calls += " WS"
    thinking_prefix = "think " if thinking else ""
    if not tag_rules:
        # tool_choice "none": neither the text nor a JSON answer starts a call.
        body = "tail" if response_schema is None else "answer"
        start = f"start: {thinking_prefix}{body}"
    elif response_schema is not None:
        body = calls if policy.required else f"({calls} | answer)"
        start = f"start: {thinking_prefix}{body}"
    elif policy.required:
        start = f"start: {thinking_prefix}{calls}"
    else:
        start = f"start: {thinking_prefix}{maybe_calls} tail"
    main = ["%llguidance {}", start]
    if response_schema is not None:
        main.append(
            "answer: WS %json "
            + json.dumps(
                _grammar_compatible_schema(response_schema), separators=(",", ":")
            )
            + " WS"
        )
    # The calls expression ends in WS in every branch, so the rule is bound
    # even where the separator before a call is TEXT.
    main.append(WHITESPACE_RULE)
    if thinking:
        think_end = (
            THINK_END_TOKEN_ID if think_end_id is None else think_end_id
        )
        main.append(f"think: TEXT <[{think_end}]>")
    main.extend(
        [
            "tail: TEXT",
            *tag_rules,
            rf"TEXT: /(?s:.*)/ & ~/(?s:.*)({dialect.barrier}|<\/think>)(?s:.*)/",
        ]
    )
    side_grammars.insert(
        0, {"name": "tool_output", "lark_grammar": "\n".join(main) + "\n"}
    )
    return json.dumps({"grammars": side_grammars}, separators=(",", ":"))

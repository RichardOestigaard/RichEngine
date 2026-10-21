"""The model families RichEngine serves: each family's identity and policy
as data, one JSON file per family in this directory.

A target is identified by its own configuration, never by its repository's
name: an MLX config.json states it, and gguf.model_config derives the same
fields from a GGUF header. Legacy RichEngine packages pack these same
layouts.

Each <family>.json carries:

  name      — the family name model.json and manifests record
  note      — what distinguishes this family, for whoever edits it next
  signature — the text_config fields that identify the architecture, as an
              MLX config states them and gguf.model_config derives them
  forbids   — (field, value) pairs that disqualify the family however the
              config was found
  vision    — false for a family whose tower the runtime does not serve
  defaults  — the packed format install/pack.py writes for it and the
              quantization its target ships
  draft     — {repo, base, signature}: the DFlash2/DFlash/DSpark checkpoint
              trained for the family. base names a shared draft dialect in
              _drafts.json the signature fields extend; a draft spelling
              its own dialect (MiniCPM-V's Draftfit) states every field.
  pack      — the family's packed-package geometry (pack_layouts.Layout and
              DraftLayout fields), its diffusion schedule and its execution
              geometry, for the families a packed format serves
  serving   — the spellings and defaults the server reads at load: the
              think framing, the tool-call dialect's name (the dialect
              bodies live in server/tool_schema.py), the vision tower's
              preprocessing spec
  tuning    — the family's autotune table: knob names from _knobs.json
              plus inline knob objects the family alone sweeps

Shared tables (the draft dialects every family's draft signature extends,
the named knobs every family's sweep references) live in _drafts.json and
_knobs.json; a file named with a leading underscore is data the loader
skips, not a family.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .. import models


@dataclass(frozen=True)
class Draft:
    # The repository of the DFlash2 checkpoint trained for the family, as its
    # release publishes it: config.json and BF16 safetensors. Installations
    # follow its default branch as they follow the target's.
    repo: str
    # The config.json fields, dotted into its objects, that the native draft
    # inspection requires, with the values it requires: a commit stating
    # others is not installed, so it never replaces a draft that loads.
    signature: tuple[tuple[str, object], ...]

    @property
    def layers(self):
        return dict(self.signature)["num_hidden_layers"]


@dataclass(frozen=True)
class FamilyDefaults:
    """The family's install policy the signature cannot state: the packed
    target format install/pack.py writes for it (one of pack.TARGET_FORMATS,
    None for a family served from its MLX or GGUF checkpoint) and the
    quantization its target ships. A family's Draft names the suggested
    draft checkpoint."""

    packed_format: str | None = None
    quant: str = "q4"


@dataclass(frozen=True)
class ModelFamily:
    name: str
    # The text_config fields that identify the architecture, as an MLX config
    # states them and as gguf.model_config derives them from a GGUF header,
    # including every one the native source model inspection requires.
    signature: tuple[tuple[str, object], ...]
    # None for a family with no GPU draft; the n-gram predraft proposes
    # for it.
    draft: Draft | None = None
    # Whether the runtime serves this family's vision tower; a family without
    # one is installed text-only whatever --language-only says.
    vision: bool = True
    # The (field, value) pairs the config must not state: a field naming
    # another architecture's value disqualifies the family however it was
    # found, strict signature or name. A field the config omits contradicts
    # nothing.
    forbids: tuple[tuple[str, object], ...] = ()
    defaults: FamilyDefaults = FamilyDefaults()
    # The family's packed-package geometry, diffusion schedule and execution
    # geometry as pack.py consumes them; None for a family no packed format
    # serves.
    pack: dict | None = None
    # The serving spellings and defaults server/serving.py reads at load:
    # the think framing, the tool-call dialect's name, the vision
    # preprocessing spec.
    serving: dict | None = None
    # The family's autotune table as its file states it: knob names from
    # _knobs.json and inline knob objects, which install/tuning resolves.
    tuning: tuple = ()
    # Upstream repository IDs known to publish the family — error messages
    # and install hints name them.
    repos: tuple[str, ...] = ()


def _freeze(value):
    """JSON's lists and dicts as the tuples and frozen mappings the family
    records carry, so a file's [1, 2] matches a config's (1, 2)."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze(item) for key, item in value.items()}
    return value


def _pairs(mapping):
    return tuple((key, _freeze(value)) for key, value in mapping.items())


def _load():
    """Every family file in this directory, as FAMILIES states them."""
    root = Path(__file__).parent
    bases = {
        name: _freeze(signature)
        for name, signature in json.loads(
            (root / "_drafts.json").read_text()
        ).items()
    }
    # The serving defaults every family shares unless its file overrides
    # them: _serving.json's entries are the base the family's "serving"
    # block merges onto (a family's "sampling" replaces the shared one
    # outright).
    serving_base = json.loads((root / "_serving.json").read_text())
    result = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text())
        draft = None
        if doc.get("draft") is not None:
            spec = doc["draft"]
            signature = dict(bases.get(spec.get("base") or "", {}))
            signature.update(spec.get("signature", {}))
            draft = Draft(spec["repo"], _pairs(signature))
        serving = doc.get("serving")
        if isinstance(serving, dict):
            serving = {**serving_base, **serving}
        result.append(
            ModelFamily(
                doc["name"],
                _pairs(doc["signature"]),
                draft,
                doc.get("vision", True),
                _pairs(doc.get("forbids", {})),
                (
                    FamilyDefaults(
                        doc["defaults"].get("packed_format"),
                        doc["defaults"].get("quant", "q4"),
                    )
                    if doc.get("defaults")
                    else FamilyDefaults()
                ),
                doc.get("pack"),
                serving,
                tuple(doc.get("tuning", ())),
                tuple(doc.get("repos", ())),
            )
        )
    return tuple(result)


FAMILIES = _load()


# Retired family names that installed packages may still carry: each maps
# to the family that serves it (the same architecture). The runtime's
# kModelFamilies keeps the same rows marked Legacy alias; the installer
# never offers them.
LEGACY_NAMES = {
    "Bonsai-2-27B": "Qwen3.8-27B",
}


def named(name):
    """The family called name, or None."""
    return next(
        (family for family in FAMILIES if family.name == LEGACY_NAMES.get(name, name)),
        None,
    )


def _names(family, hint):
    """Whether the hint names the family: its name bounded by characters a
    repository or file name separates tokens with, never inside another
    word (NotBonsai-2-27B names no Bonsai)."""
    pattern = r"(?<![0-9a-z])" + re.escape(family.name.lower()) + r"(?![0-9a-z])"
    return re.search(pattern, hint.lower()) is not None


def family_for(config, name=None):
    """The one family whose architecture the target's config states.

    name is an optional hint such as a repository or GGUF name. It supplies
    the fields a config omits: a family it names wins over a signature that
    matches only because the config omits the deciding field (a GGUF states
    no router_aux_loss_coef), and never over a strict match of its own
    length or longer — a stated discriminator beats the name. A named
    family must still not contradict any field the config states, and a
    field a family forbids disqualifies it however it was found."""
    text = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text, dict):
        # A text-only model (a llama or lfm2 checkpoint) states the same
        # fields flat at its config's root instead of under text_config.
        if isinstance(config, dict) and isinstance(config.get("model_type"), str):
            text = config
        else:
            raise models.ModelError("upstream configuration has no text_config")

    def stated(key, expected, missing_ok):
        value = text.get(key, expected if missing_ok else None)
        if isinstance(value, list):
            value = tuple(value)
        return value == expected

    def forbidden(family):
        # A stated field the family forbids is a contradiction whichever
        # path found it; a field the config omits contradicts nothing.
        # No family declares forbids today; the root-level check stays
        # wired for the next sibling that needs an exclusion. Root-level
        # fields (architectures, model_type on a VL config) state
        # just as much as the text_config's, so they disqualify the same.
        def root_stated(key, value):
            if not isinstance(config, dict) or key not in config:
                return False
            root_value = config[key]
            if isinstance(root_value, list):
                root_value = tuple(root_value)
            return root_value == value

        return any(
            (key in text and stated(key, value, False)) or root_stated(key, value)
            for key, value in family.forbids
        )

    matches = [
        f
        for f in FAMILIES
        if not forbidden(f) and all(stated(k, v, False) for k, v in f.signature)
    ]
    if name:
        named_families = [
            f
            for f in FAMILIES
            if _names(f, name)
            and not forbidden(f)
            and all(stated(k, v, True) for k, v in f.signature)
        ]
        if named_families and max(len(f.signature) for f in named_families) > max(
            (len(f.signature) for f in matches), default=0
        ):
            matches = named_families
    if not matches:
        keys = sorted({key for family in FAMILIES for key, _ in family.signature})
        found = ", ".join(f"{key}={text.get(key)}" for key in keys if key in text)
        raise models.ModelError(
            f"no supported model has this architecture ({found}); "
            f"supported: {', '.join(f.name for f in FAMILIES)}"
        )
    # A family whose signature extends another's (Ornith-1.5-35B-A3B states
    # Qwen3.6-35B-A3B's geometry plus its own router loss) matches beside it;
    # the signature asking the most states the architecture most exactly.
    best = max(matches, key=lambda family: len(family.signature))
    if sum(len(f.signature) == len(best.signature) for f in matches) > 1:
        raise models.ModelError("this architecture matches no single supported model")
    return best

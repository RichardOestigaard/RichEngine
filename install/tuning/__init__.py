"""Per-model autotune: each family's TunableKnobs table is data in its
install/families/<name>.json "tuning" list — knob names from the shared
install/families/_knobs.json library plus inline knob objects the family
alone sweeps — resolved here by family name (the model.json/manifest field
the assembly records). install/autotune.py sweeps the eligible subset; the
result lands in the assembly's tuning.json, which server/model_host.py
applies as engine environment variables on every load."""

import json
from pathlib import Path

from .. import families
from .base import Knob, TunableKnobs, TuneContext, eligible

_KNOBS_PATH = Path(families.__file__).parent / "_knobs.json"

# The shared knob library every family's table references by name.
def _shared_knobs() -> dict[str, Knob]:
    return {
        name: Knob(
            spec["env"],
            tuple(spec["values"]),
            spec["reason"],
            tuple(spec.get("model_types", ())),
            tuple(spec.get("draft_kinds", ())),
            tuple(spec.get("formats", ())),
            spec.get("min_gpu_family", 0),
            spec.get("min_bandwidth_gbps", 0),
            spec.get("diffusion"),
            tuple(spec.get("requires_env", ())),
            spec.get("quality_sensitive", False),
            spec.get("ordered", False),
        )
        for name, spec in json.loads(_KNOBS_PATH.read_text()).items()
    }


def _knob(entry, shared: dict[str, Knob]) -> Knob:
    """One family-table entry: a shared knob's name or an inline spec the
    family alone sweeps."""
    if isinstance(entry, str):
        return shared[entry]
    return Knob(
        entry["env"],
        tuple(entry["values"]),
        entry["reason"],
        tuple(entry.get("model_types", ())),
        tuple(entry.get("draft_kinds", ())),
        tuple(entry.get("formats", ())),
        entry.get("min_gpu_family", 0),
        entry.get("min_bandwidth_gbps", 0),
        entry.get("diffusion"),
        tuple(entry.get("requires_env", ())),
        entry.get("quality_sensitive", False),
        entry.get("ordered", False),
    )


def _tables() -> dict[str, TunableKnobs]:
    shared = _shared_knobs()
    return {
        family.name: TunableKnobs(
            family.name,
            tuple(_knob(entry, shared) for entry in family.tuning),
        )
        for family in families.FAMILIES
        if family.tuning
    }


TUNING_TABLES: dict[str, TunableKnobs] = _tables()


def knobs_for(family: str) -> TunableKnobs | None:
    """The family's sweep table, or None for a family without one."""
    return TUNING_TABLES.get(family)


__all__ = [
    "TUNING_TABLES",
    "Knob",
    "TunableKnobs",
    "TuneContext",
    "eligible",
    "knobs_for",
]

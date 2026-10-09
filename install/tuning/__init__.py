"""Per-model autotune: each family's TunableKnobs table lives in its own
module here, registered by family name (the model.json/manifest field the
assembly records). install/autotune.py sweeps the eligible subset; the
result lands in the assembly's tuning.json, which server/model_host.py
applies as engine environment variables on every load."""

from . import gemma, granite, lfm2, minicpm, ornith, qwen
from .base import Knob, TunableKnobs, TuneContext, eligible

_MODULES = (qwen, ornith, minicpm, lfm2, granite, gemma)

TUNING_TABLES: dict[str, TunableKnobs] = {
    tunable.family: tunable for module in _MODULES for tunable in module.TUNABLE
}


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

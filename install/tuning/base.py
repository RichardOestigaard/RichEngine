"""The autotune knob framework: what a family offers the sweep to try.

A Knob is one RICHENGINE_* environment variable plus the candidate values the
sweep measures; the engine reads these once at process start, so the sweep
spawns one engine per candidate. A knob applies only when every gate on it
matches the TuneContext — model type, draft kind, target format, GPU family
and memory bandwidth — so a family file lists everything it could ever tune
and the sweep keeps the subset this chip and this install can reach. Most
knobs are the engine's disabled-by-default paths (the pool selector, tree
verify for GPU drafts, the chunked GDN scan, packed GGUF, Metal 4): cheaper
to win by trying than to reason about, and the sweep settles each on
measurement.

The knob tables themselves are data: every family's install/families/<name>.json
"tuning" list mixes names of the shared knobs in install/families/_knobs.json
with inline knob objects a family alone sweeps. install/tuning/__init__.py
resolves both into TunableKnobs.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TuneContext:
    """What the sweep knows about this install on this machine: enough to
    drop knobs that cannot run (a GGUF knob on an MLX install, a conv knob
    on a target without one) or that a weaker bandwidth class never wins."""

    # The family name the assembly's record states (FAMILIES' key); "" when
    # the install names a family the table does not know.
    family: str
    chip: str
    # MTLGPUFamilyAppleN; 0 when the chip's family is unknown — family-gated
    # knobs then stay out.
    gpu_family: int
    # The chip's unified-memory bandwidth in GB/s; 0 when unknown.
    bandwidth_gbps: int
    # The family's signature model_type (qwen3_5_text, lfm2, gemma4_text…).
    model_type: str
    # The installed target's format (mlx, gguf, packed-gemma4, …).
    target_format: str
    # The draft kind the descriptor loads: dflash2, dflash, dspark or none.
    draft_kind: str
    # Whether the target is a block-diffusion model (DiffusionGemma).
    diffusion: bool


@dataclass(frozen=True)
class Knob:
    """One sweepable engine switch.

    env — the RICHENGINE_* variable the runtime's Tuning registry reads.
    values — the candidates to measure, each as the variable's text; the
      sweep keeps whichever measured fastest and writes it into the model's
      tuning file.
    reason — one line on what the path changes, for the sweep's log.
    Gates (all must hold): model_types, draft_kinds, formats restrict to
    matching installs; min_gpu_family requires MTLGPUFamilyAppleN — an
    unknown family (0) stays out since the path may not exist; diffusion
    restricts to (True) or from (False) canvas targets;
    min_bandwidth_gbps holds the knob back on chips whose unified-memory
    bandwidth cannot feed the path — an unknown bandwidth (0) is unproven,
    not excluded, and measurement settles it.
    requires_env lists variables that must already be set in the sweep's
    own environment for the knob's path to exist — the ANE wait only moves
    when an ANE predraft artifact is loaded, so the knob stays out unless
    the user armed one.
    quality_sensitive knobs trade output quality for speed (canvas step
    caps, exit thresholds); the sweep leaves them out unless asked.
    ordered knobs list their candidates closest-to-default first; when the
    first loses beyond the sweep's prune depth, the sweep assumes the rest
    deviate more and lose worse — set it only where the knob's effect is
    monotone in its deviation from the default.
    may_diverge knobs may legitimately change the emitted token stream —
    the int8 prefill path's documented near-tie flips, or the packed GGUF
    operands' rare one-ulp divergences — so the sweep's token-parity veto
    does not apply to them. Quality-sensitive knobs imply it.
    agentic knobs only move the engine's constrained-decode (DFA) path:
    inert under the sweep's unconstrained workload, they are measured by
    the agentic pass — a DFA-constrained workload — instead of a family
    table's list, and only when the sweep is asked for it.
    """

    env: str
    values: tuple[str, ...]
    reason: str
    model_types: tuple[str, ...] = ()
    draft_kinds: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    min_gpu_family: int = 0
    min_bandwidth_gbps: int = 0
    diffusion: bool | None = None
    requires_env: tuple[str, ...] = ()
    quality_sensitive: bool = False
    ordered: bool = False
    may_diverge: bool = False
    agentic: bool = False

    def applies(self, context: TuneContext) -> bool:
        return (
            all(name in os.environ for name in self.requires_env)
            and (not self.model_types or context.model_type in self.model_types)
            and (not self.draft_kinds or context.draft_kind in self.draft_kinds)
            and (not self.formats or context.target_format in self.formats)
            and self.min_gpu_family <= context.gpu_family
            and (
                not context.bandwidth_gbps
                or self.min_bandwidth_gbps <= context.bandwidth_gbps
            )
            and (self.diffusion is None or self.diffusion == context.diffusion)
        )


@dataclass(frozen=True)
class TunableKnobs:
    """One model family's sweep table."""

    family: str
    knobs: tuple[Knob, ...]


def eligible(tuning: TunableKnobs, context: TuneContext) -> tuple[Knob, ...]:
    """The family's knobs this install on this chip can actually run."""
    return tuple(knob for knob in tuning.knobs if knob.applies(context))

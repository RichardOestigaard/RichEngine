"""The served model's serving contract: spellings and defaults as data.

Install writes the family's ``serving`` block (install/families/*.json)
into the assembly's model.json, and pack into the package manifest, so a
loaded model carries the spellings its server stack needs: the think
framing, the tool-call dialect's name, the vision tower's pad token and
preprocessing geometry, and the sampling defaults its generation config
states. A record predating the key resolves the same block through its
family name; a record naming no family falls back to probing, as the
server always did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ThinkFraming:
    """How the family's template frames a reasoning block: the close
    spelling, the open spelling, the special token that leads the open
    (decoded for the output parser), and the system-turn flag that marks
    a thinking-enabled prompt where the family has one."""

    end: str = "</think>"
    open: str = "<think>"
    open_token: str | None = "<think>"
    flag: str | None = None


@dataclass(frozen=True)
class VisionSpec:
    """The vision tower's preprocessing and markup contract: the pad token
    the chat template emits per image, and the patch geometry the image
    pipeline resizes into. None for a family serving no tower."""

    pad_token: str = "<|image_pad|>"
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)
    min_pixels: int = 65_536
    # The tower's own resize cap where it states one narrower than the
    # server's default; the frontend prepares under the tighter bound.
    max_pixels: int | None = None


@dataclass(frozen=True)
class VideoSpec:
    """The video pipeline's sampling and markup contract: the pad token the
    chat template emits per clip, the span markers each timestamped frame
    group wraps in, and the frame-sampling and pixel-budget geometry the
    video pipeline decodes with. None for a family serving no video."""

    pad_token: str = "<|video|>"
    vision_start_token: str = "<|vision_start|>"
    vision_end_token: str = "<|vision_end|>"
    # The sampler's target rate and the clip's frame-count bounds.
    fps: float = 2.0
    min_frames: int = 4
    max_frames: int = 384
    # The temporal conv's pair depth: frames are consumed two at a time and
    # the last odd frame pairs with itself.
    temporal_patch_size: int = 2
    # The frame-volume pixel bounds (padded count x h x w) the video
    # processor resizes inside; the budget is shared by every sampled frame.
    min_pixels: int = 4_096
    max_pixels: int = 25_165_824


@dataclass(frozen=True)
class AudioSpec:
    """The audio encoder's front end and markup contract: the pad token the
    chat template emits per clip (one token per 8 mel frames after span
    expansion) and the mel geometry the audio pipeline computes with. None
    for a family serving no audio tower."""

    pad_token: str = "<|audio|>"
    sampling_rate: int = 16_000
    n_mel: int = 128
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    # Mel frames per language token (80 ms at hop 160/16 kHz).
    encoder_stride: int = 8


@dataclass(frozen=True)
class ServingContract:
    """Everything the server stack takes from the served model rather than
    from constants: a model swap changes the contract, never the code."""

    # The name model.json records; informational.
    family: str | None = None
    think: ThinkFraming = ThinkFraming()
    # The tool-call dialect's name (server/tool_schema.py's registry), or
    # None where the family declares none and the template probe decides.
    tool_dialect: str | None = None
    vision: VisionSpec | None = None
    audio: AudioSpec | None = None
    # The video pipeline's contract; None where the family serves no video
    # even though the tower could take frame pairs.
    video: VideoSpec | None = None
    # Sampling defaults the model's generation config states; a request's
    # own value always wins. Keys are protocol.SAMPLING_FIELDS members.
    sampling: dict = field(default_factory=dict)
    # The chat/Responses reasoning effort when a request states none; None
    # keeps the flag's (or no) default.
    reasoning_effort: str | None = None
    # Extra names the model serves under beyond its package id and the
    # command line's --served-model-name list.
    served_names: tuple = ()
    # The engine's default DFA mode when RICHENGINE_DFA_MODE is unset; the
    # family's descriptor declares the same value so both sides of the wire
    # resolve identically. "full" ships constraint DFAs and serves no host
    # mask; None keeps every constraint on the host path.
    dfa_mode: str | None = None
    # A decision model (Liquid d1 family): /v1/systemone and /v1/judgments
    # run the native decision pathway, not the SemIf emulation.
    decision: bool = False
    # An encoder decision model (d1-omni): bidirectional attention, no
    # causal decode.
    encoder: bool = False
    # The model's decision readout temperatures by option group, as the
    # GGUF's lfm2.decision.temperature.* table states them.
    decision_temperatures: dict = field(default_factory=dict)

    @classmethod
    def of(cls, doc, family=None):
        """The contract a ``serving`` block states, validated lazily: a bad
        field fails the load it describes, not a reader of another."""
        if not isinstance(doc, dict):
            doc = {}
        think = doc.get("think")
        vision = doc.get("vision")
        audio = doc.get("audio")
        video = doc.get("video")
        return cls(
            family=family,
            think=(
                ThinkFraming(
                    end=think.get("end", "</think>"),
                    open=think.get("open", "<think>"),
                    open_token=think.get("open_token"),
                    flag=think.get("flag"),
                )
                if isinstance(think, dict)
                else ThinkFraming()
            ),
            tool_dialect=doc.get("tool_dialect"),
            vision=(
                VisionSpec(
                    pad_token=vision.get("pad_token", "<|image_pad|>"),
                    patch_size=int(vision.get("patch_size", 16)),
                    temporal_patch_size=int(vision.get("temporal_patch_size", 2)),
                    merge_size=int(vision.get("merge_size", 2)),
                    image_mean=tuple(vision.get("image_mean", (0.5, 0.5, 0.5))),
                    image_std=tuple(vision.get("image_std", (0.5, 0.5, 0.5))),
                    min_pixels=int(vision.get("min_pixels", 65_536)),
                    max_pixels=(
                        None
                        if vision.get("max_pixels") is None
                        else int(vision["max_pixels"])
                    ),
                )
                if isinstance(vision, dict)
                else None
            ),
            audio=(
                AudioSpec(
                    pad_token=audio.get("pad_token", "<|audio|>"),
                    sampling_rate=int(audio.get("sampling_rate", 16_000)),
                    n_mel=int(audio.get("n_mel", 128)),
                    n_fft=int(audio.get("n_fft", 400)),
                    win_length=int(audio.get("win_length", 400)),
                    hop_length=int(audio.get("hop_length", 160)),
                    encoder_stride=int(audio.get("encoder_stride", 8)),
                )
                if isinstance(audio, dict)
                else None
            ),
            video=(
                VideoSpec(
                    pad_token=video.get("pad_token", '<|video|>'),
                    vision_start_token=video.get(
                        "vision_start_token", '<|vision_start|>'
                    ),
                    vision_end_token=video.get(
                        "vision_end_token", '<|vision_end|>'
                    ),
                    fps=float(video.get("fps", 2.0)),
                    min_frames=int(video.get("min_frames", 4)),
                    max_frames=int(video.get("max_frames", 384)),
                    temporal_patch_size=int(video.get("temporal_patch_size", 2)),
                    min_pixels=int(video.get("min_pixels", 4_096)),
                    max_pixels=int(video.get("max_pixels", 25_165_824)),
                )
                if isinstance(video, dict)
                else None
            ),
            sampling=dict(doc.get("sampling", {})),
            reasoning_effort=doc.get("reasoning_effort"),
            served_names=tuple(doc.get("served_names", ())),
            dfa_mode=doc.get("dfa_mode"),
            decision=bool(doc.get("decision", False)),
            encoder=bool(doc.get("encoder", False)),
            decision_temperatures=dict(doc.get("decision_temperatures") or {}),
        )


def _record(assembly_dir):
    """The assembly's model.json, or a package's manifest.json — whichever
    the installation kind is; {} where neither reads."""
    from install import layout
    from install import models as model_artifacts

    for name in (layout.ASSEMBLY_RECORD, layout.PACKAGE_MANIFEST):
        try:
            doc = model_artifacts.read_json(assembly_dir / name)
        except (model_artifacts.ModelError, OSError):
            continue
        if isinstance(doc, dict):
            return doc
    return {}


def load(assembly_dir):
    """The contract an assembly serves under.

    The record's own ``serving`` block wins; one without it (installed
    before the key existed) resolves its family's file; a record naming no
    family gets the probing defaults the contract fields default to."""
    assembly_dir = Path(assembly_dir)
    record = _record(assembly_dir)
    family_name = record.get("family")
    doc = record.get("serving")
    if family_name:
        from install import families

        family = families.named(family_name)
        if family is not None:
            if doc is None:
                doc = family.serving
            elif isinstance(doc, dict) and family.serving:
                # Keys a record predates — dfa_mode and later additions —
                # come from the family's file; the record's own win.
                doc = {**family.serving, **doc}
    return ServingContract.of(doc, family=family_name)

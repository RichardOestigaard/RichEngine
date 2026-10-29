"""Decode videos to sampled, resized RGB frames and an optional soundtrack.

Mirrors images.py: the server decodes the request's clip with ffmpeg and
reduces it to the fixed representations the engine consumes — one
patch-aligned uint8 RGB frame pair per temporal group (the vision tower's
two temporal conv slices take frame 0 and frame 1 of each pair) and, when
the request asks for it, the clip's soundtrack as float32 [128, T] mel via
audio.py's front end. The engine encodes each pair and each audio piece;
this module only samples, resizes and digs.

Decoding shells out to ffmpeg/ffprobe on PATH; a missing binary fails the
request, not the server. The sampling, timestamps, pairing and soundtrack
interleave reproduce hf/processing_lfm3_omni.py's _video_expansion and the
video processor's sample_frames/smart_resize.
"""

import base64
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from . import audio as audio_input
from .serving import VideoSpec

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE

# Bound the container a request may carry: a base64 data URL's raw bytes.
MAX_VIDEO_BYTES = 128 * 1024 * 1024
# ffmpeg/ffprobe get a hard wall-clock bound per call.
SUBPROCESS_SECONDS = 120
# Decode source frames no larger than this; a 16k frame is rejected early.
MAX_SOURCE_PIXELS = 64 * 1024 * 1024
# One language token covers 8 mel frames (80 ms) of the 16 kHz soundtrack.
SECONDS_PER_AUDIO_TOKEN = 0.08


class VideoError(ValueError):
    pass


class VideoCapacityError(VideoError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedVideo:
    """One clip's sampled frames as temporal pairs: the patch grid every
    frame shares, the per-pair two-frame pixel payloads with their 128-bit
    digests, the per-pair timestamp text the LM prompt spells, and the
    soundtrack mel with the per-pair token bounds when one rides along."""

    grid_height: int  # patches
    grid_width: int
    pair_pixels: tuple  # bytes, two frames each
    pair_digests: tuple  # (lo, hi) per pair
    timestamps: tuple  # float seconds per pair
    # (mel bytes, total mel frames, per-pair audio-token bounds) or None.
    soundtrack: tuple | None = None
    geometry: VideoSpec = field(default=VideoSpec, compare=False)

    @property
    def tokens(self) -> int:
        """Merged language tokens each pair occupies."""
        return (self.grid_height // MERGE) * (self.grid_width // MERGE)

    @property
    def pairs(self) -> int:
        return len(self.pair_pixels)


@dataclass(frozen=True, slots=True)
class _Source:
    fps: float
    total_frames: int
    height: int
    width: int
    has_audio: bool


def decode_data_url(url) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise VideoError("only data: video URLs are supported")
    header, _, encoded = url.partition(",")
    if ";base64" not in header:
        raise VideoError("video data URLs must be base64-encoded")
    if len(encoded) > MAX_VIDEO_BYTES * 4 // 3 + 4:
        raise VideoError("video exceeds the size limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise VideoError("video data URL is not valid base64") from error
    if len(raw) > MAX_VIDEO_BYTES:
        raise VideoError("video exceeds the size limit")
    return raw


def _which(name):
    path = shutil.which(name)
    if path is None:
        raise VideoError(
            f"video input requires {name} on PATH (install ffmpeg)"
        )
    return path


def _run(argv, payload, what):
    try:
        completed = subprocess.run(
            argv,
            input=payload,
            capture_output=True,
            timeout=SUBPROCESS_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise VideoError(f"{what} timed out") from error
    except OSError as error:
        raise VideoError(f"{what} could not run") from error
    if completed.returncode != 0:
        raise VideoError(f"video could not be decoded ({what})")
    return completed.stdout


def _probe(payload: bytes, spec: VideoSpec) -> _Source:
    """The container's fps, frame count, dimensions and audio presence."""
    ffprobe = _which("ffprobe")
    # Most containers probe from a pipe; formats that cannot are retried
    # from a temp file by the caller path that needs them — mp4 seeks fine.
    out = _run(
        [
            ffprobe, "-v", "error", "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames:format=duration",
            "-of", "json", "pipe:0",
        ],
        payload,
        "ffprobe",
    )
    try:
        doc = json.loads(out)
    except ValueError as error:
        raise VideoError("video metadata could not be read") from error
    video_stream = None
    has_audio = False
    for stream in doc.get("streams") or ():
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio":
            has_audio = True
    if video_stream is None:
        raise VideoError("the container carries no video stream")

    def rate(value):
        if not isinstance(value, str) or "/" not in value:
            return None
        num, _, den = value.partition("/")
        try:
            parsed = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
        return parsed if math.isfinite(parsed) and parsed > 0 else None

    fps = rate(video_stream.get("avg_frame_rate")) or rate(
        video_stream.get("r_frame_rate")
    )
    if fps is None:
        raise VideoError("the video stream's frame rate is unknown")
    try:
        total = int(video_stream["nb_frames"])
    except (KeyError, TypeError, ValueError):
        total = 0
    if total <= 0:
        duration = video_stream.get("duration") or (doc.get("format") or {}).get(
            "duration"
        )
        try:
            total = round(float(duration) * fps)
        except (TypeError, ValueError):
            total = 0
    if total <= 0:
        raise VideoError("the video stream's frame count is unknown")
    width = video_stream.get("width") or 0
    height = video_stream.get("height") or 0
    if width <= 0 or height <= 0:
        raise VideoError("the video stream's dimensions are unknown")
    if width * height > MAX_SOURCE_PIXELS:
        raise VideoError("source video exceeds the pixel limit")
    return _Source(round(fps, 6), total, height, width, has_audio)


def sample_indices(total: int, fps_src: float, spec: VideoSpec) -> list:
    """The sampled frame indices: fps-target count clamped to the bounds,
    evenly spaced across the clip, the video processor's own rule."""
    import numpy as np

    count = int(total / fps_src * spec.fps)
    count = min(max(count, spec.min_frames), spec.max_frames, total)
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def smart_resize(num_frames: int, height: int, width: int, spec: VideoSpec):
    """Patch/merge-aligned sides inside the frame-volume budget (padded
    frame count x h x w), the video processor's own smart resize."""
    factor = FACTOR
    temporal = spec.temporal_patch_size
    if num_frames < temporal:
        raise VideoError("video is too short for a temporal pair")
    if height <= 0 or width <= 0:
        raise VideoError("video dimensions must be positive")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise VideoError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal) * temporal
    if t_bar * h_bar * w_bar > spec.max_pixels:
        beta = math.sqrt((num_frames * height * width) / spec.max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < spec.min_pixels:
        beta = math.sqrt(spec.min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _decode_frames(payload: bytes, indices: list, size, spec: VideoSpec):
    """RGB24 rows of the sampled indices resized to `size` (w, h)."""
    ffmpeg = _which("ffmpeg")
    # select keeps only the sampled decoded frames; bicubic matches the
    # reference resample. Indices are non-decreasing and unique.
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    width, height = size
    raw = _run(
        [
            ffmpeg, "-v", "error", "-i", "pipe:0",
            "-vf", f"select='{select}',scale={width}:{height}:flags=bicubic",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        payload,
        "ffmpeg",
    )
    frame_bytes = width * height * 3
    if len(raw) != frame_bytes * len(indices):
        raise VideoError(
            "the decoded frame count does not match the sample "
            f"({len(raw) // frame_bytes} of {len(indices)})"
        )
    return [
        raw[index * frame_bytes : (index + 1) * frame_bytes]
        for index in range(len(indices))
    ]


def _soundtrack(payload: bytes):
    """The clip's soundtrack as (mel bytes, mel frames), or None without
    an audio stream: ffmpeg resamples to 16 kHz mono float32 and
    audio.py's front end computes the log-mel."""
    ffmpeg = _which("ffmpeg")
    try:
        completed = subprocess.run(
            [
                ffmpeg, "-v", "error", "-i", "pipe:0",
                "-vn", "-ac", "1", "-ar", str(audio_input.SAMPLING_RATE),
                "-f", "f32le", "pipe:1",
            ],
            input=payload,
            capture_output=True,
            timeout=SUBPROCESS_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        raise VideoError("the soundtrack could not be decoded") from error
    if completed.returncode != 0 or not completed.stdout:
        return None
    import numpy as np

    samples = np.frombuffer(completed.stdout, dtype="<f4")
    mel = audio_input.log_mel(samples.astype(np.float32))
    return mel.tobytes(), mel.shape[1]


def _audio_bounds(indices, fps: float, tokens: int, spec: VideoSpec):
    """The soundtrack's per-pair token bounds: each group's split lands at
    its first frame's second (processing_lfm3_omni.py's bounds rule)."""
    temporal = spec.temporal_patch_size
    starts = [index / fps for index in indices[::temporal]]
    bounds = (
        [0]
        + [min(tokens, round(start / SECONDS_PER_AUDIO_TOKEN))
           for start in starts[1:]]
        + [tokens]
    )
    return [max(bounds[: i + 1]) for i in range(len(bounds))]


def prepare(payload: bytes, spec: VideoSpec, *, soundtrack: bool = False):
    """Probe, sample, resize and digest one clip's frames; its soundtrack
    rides along when asked and the container carries one."""
    source = _probe(payload, spec)
    indices = sample_indices(source.total_frames, source.fps, spec)
    temporal = spec.temporal_patch_size
    padded = list(indices)
    if pad := -len(padded) % temporal:
        padded.extend([padded[-1]] * pad)
    height, width = smart_resize(
        len(indices), source.height, source.width, spec
    )
    frames = _decode_frames(payload, indices, (width, height), spec)
    frame_bytes = height * width * 3
    pair_pixels, pair_digests = [], []
    for first in range(0, len(frames), temporal):
        second = first + 1 if first + 1 < len(frames) else len(frames) - 1
        pixels = frames[first] + frames[second]
        digest = hashlib.sha256(
            struct.pack("<III", height // PATCH, width // PATCH, temporal)
            + pixels
        )
        pair_pixels.append(pixels)
        pair_digests.append(struct.unpack_from("<QQ", digest.digest()))
    timestamps = tuple(
        (padded[i] + padded[i + temporal - 1]) / (2 * source.fps)
        for i in range(0, len(padded), temporal)
    )
    track = None
    if soundtrack and source.has_audio:
        decoded = _soundtrack(payload)
        if decoded is not None:
            mel, mel_frames = decoded
            tokens = -(-mel_frames // audio_input.ENCODER_STRIDE)
            track = (mel, mel_frames, _audio_bounds(padded, source.fps, tokens, spec))
    elif soundtrack:
        raise VideoError("use_audio_in_video needs a clip with an audio track")
    return PreparedVideo(
        height // PATCH,
        width // PATCH,
        tuple(pair_pixels),
        tuple(pair_digests),
        timestamps,
        track,
        spec,
    )


class PreparedVideos(list):
    """Append-only request batch; its owner travels with the pixel payload."""

    def __init__(self, cache):
        super().__init__()
        self._cache = cache
        self._bytes = 0

    def append(self, video):
        size = sum(len(pixels) for pixels in video.pair_pixels)
        if video.soundtrack:
            size += len(video.soundtrack[0])
        with self._cache._lock:
            if size > self._cache.REQUEST_BUDGET_BYTES - self._cache._request_bytes:
                raise VideoCapacityError(
                    "in-flight video memory budget is full"
                )
            self._cache._request_bytes += size
            self._bytes += size
        super().append(video)

    def __del__(self):
        with self._cache._lock:
            self._cache._request_bytes -= self._bytes


class VideoCache:
    """Byte-budgeted LRU of prepared videos keyed by payload and options."""

    BUDGET_BYTES = 256 * 1024 * 1024
    REQUEST_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: OrderedDict[bytes, PreparedVideo] = OrderedDict()
        self._bytes = 0
        self._request_bytes = 0

    def request_batch(self):
        return PreparedVideos(self)

    def prepare(self, payload: bytes, spec: VideoSpec, *, soundtrack=False):
        key = hashlib.sha256(
            spec.pad_token.encode() + b"\x00" + (b"A" if soundtrack else b"F")
            + payload
        ).digest()
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached
        prepared = prepare(payload, spec, soundtrack=soundtrack)
        size = sum(len(pixels) for pixels in prepared.pair_pixels) + (
            len(prepared.soundtrack[0]) if prepared.soundtrack else 0
        )
        if size > self.BUDGET_BYTES:
            return prepared
        with self._lock:
            if key not in self._entries:
                self._entries[key] = prepared
                self._bytes += size
                while self._bytes > self.BUDGET_BYTES:
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= sum(
                        len(pixels) for pixels in evicted.pair_pixels
                    ) + (
                        len(evicted.soundtrack[0]) if evicted.soundtrack else 0
                    )
        return prepared

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "budget_bytes": self.BUDGET_BYTES,
                "request_bytes": self._request_bytes,
                "request_budget_bytes": self.REQUEST_BUDGET_BYTES,
            }

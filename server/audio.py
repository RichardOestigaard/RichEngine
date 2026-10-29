"""Decode audio and compute the LFM3-Omni log-mel front end in numpy.

Mirrors images.py: the server decodes the request's audio and reduces it to
the fixed front-end representation the engine's audio tower consumes —
float32 [128 mel, T frames] at 10 ms per frame. The engine handles the
conv frontend, encoder and injection; this module only produces the mel.

Decoding supports WAV containers holding PCM (8/16/24/32-bit integer or
32/64-bit float) via the stdlib ``wave`` module, mono or interleaved
multi-channel (mixed down). MP3, OGG and other compressed containers are
rejected; no compressed-audio decoder is vendored. Resampling to the
encoder's 16 kHz uses linear interpolation when the clip's rate differs.

The front end reproduces LFM3-Omni's feature extractor
(dev/omni-reference/hf/feature_extraction_lfm3_omni.py): STFT n_fft=400,
win=400 periodic Hann, hop=160, center=False with the signal right-padded
to whole frames; power spectrum into a 128-bin slaney mel filterbank
(0-8000 Hz, slaney area norm, matching transformers' mel_filter_bank and
llama.cpp's fill_mel_filterbank_matrix); log10 clamped at -12; then
(x+4)/4. One language token covers encoder_stride=8 mel frames (80 ms).
"""

import base64
import hashlib
import io
import math
import struct
import threading
import wave
from collections import OrderedDict
from dataclasses import dataclass

SAMPLING_RATE = 16_000
N_MEL = 128
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
ENCODER_STRIDE = 8
# log10(mel) floored at -12, then (x + 4) / 4.
LOG_FLOOR = -12.0
NORMALIZE_OFFSET = 4.0
NORMALIZE_DIVISOR = 4.0

# Bound the encoded audio a request may carry and one clip's decode result.
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_AUDIO_SECONDS = 600
MAX_AUDIO_CLIPS = 64


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """One clip's mel features: float32 [n_mel, mel_frames] little-endian
    bytes, the placeholder-token count (ceil(frames/8)) and a 128-bit
    content digest for span identity."""

    mel_frames: int
    mel: bytes
    digest_lo: int
    digest_hi: int

    @property
    def tokens(self) -> int:
        return -(-self.mel_frames // ENCODER_STRIDE)


class AudioError(ValueError):
    pass


class AudioCapacityError(AudioError):
    pass


def decode_data_url(data: str) -> bytes:
    """An input_audio part's payload: a bare base64 string or a data: URL."""
    if not isinstance(data, str):
        raise AudioError("input_audio data must be a base64 string")
    encoded = data
    if data.startswith("data:"):
        header, _, encoded = data.partition(",")
        if ";base64" not in header:
            raise AudioError("audio data URLs must be base64-encoded")
    if len(encoded) > MAX_AUDIO_BYTES * 4 // 3 + 4:
        raise AudioError("audio exceeds the size limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise AudioError("audio is not valid base64") from error
    if len(raw) > MAX_AUDIO_BYTES:
        raise AudioError("audio exceeds the size limit")
    return raw


def _pcm_frames(frames, sample_width, channels):
    """Interleaved PCM to mono float32 in [-1, 1]."""
    import numpy as np

    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((raw.shape[0], 4), dtype=np.uint8)
        padded[:, 1:] = raw
        padded[:, 0] = np.where(raw[:, 2] >= 128, 255, 0)
        samples = padded.view("<i4").reshape(-1).astype(np.float32) / 8388608.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioError("unsupported WAV sample width")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def decode_wave(payload: bytes):
    """A WAV container's PCM as mono float32 at its own sample rate."""
    import numpy as np

    try:
        with wave.open(io.BytesIO(payload)) as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            compression = wav.getcomptype()
            if compression != "NONE":
                raise AudioError(
                    "compressed WAV is not supported (PCM only; MP3/OGG unsupported)"
                )
            frames = wav.readframes(wav.getnframes())
    except AudioError:
        raise
    except (wave.Error, EOFError) as error:
        raise AudioError("audio could not be decoded as a WAV file") from error
    if not 1 <= channels <= 8 or rate <= 0:
        raise AudioError("invalid WAV channel or sample-rate fields")
    # IEEE-float WAV keeps compression 'NONE' with 4/8-byte samples; the
    # stdlib reports them through getsampwidth as 4 or 8.
    if width == 8:
        samples = np.frombuffer(frames, dtype="<f8").astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return np.ascontiguousarray(samples), rate
    if width == 4 and not len(frames) % 4:
        # PCM32 and float32 are both four-byte; float WAV files are rare and
        # wave marks them identically, so try PCM first — the common case —
        # then float when the PCM interpretation saturates absurdly.
        pcm = _pcm_frames(frames, 4, channels)
        if np.abs(pcm).max() > 1.5:
            flt = np.frombuffer(frames, dtype="<f4").astype(np.float32)
            if channels > 1:
                flt = flt.reshape(-1, channels).mean(axis=1)
            if np.isfinite(flt).all() and np.abs(flt).max() <= 8.0:
                return np.ascontiguousarray(flt), rate
        return pcm, rate
    return _pcm_frames(frames, width, channels), rate


def resample(samples, rate, target=SAMPLING_RATE):
    """Linear-interpolation resample to the encoder's sample rate."""
    import numpy as np

    if rate == target:
        return samples
    if not isinstance(rate, int) or rate <= 0:
        raise AudioError("invalid source sample rate")
    count = int(round(len(samples) * target / rate))
    if count <= 0:
        raise AudioError("audio is too short to resample")
    positions = np.arange(count, dtype=np.float64) * (rate / target)
    return np.interp(positions, np.arange(len(samples), dtype=np.float64), samples).astype(
        np.float32
    )


def pad_length(num_samples: int) -> int:
    """Right-pad so the last frame is whole (center=False STFT)."""
    if num_samples < N_FFT:
        return N_FFT
    return num_samples + (-(num_samples - N_FFT) % HOP_LENGTH)


def _hann(length: int):
    import numpy as np

    # torch.hann_window defaults to the periodic window.
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def _slaney_mel_filters():
    """The 128-bin slaney filterbank of transformers' mel_filter_bank and
    llama.cpp's fill_mel_filterbank_matrix: slaney hz<->mel, slaney area
    normalization 2/(f_right - f_left)."""
    import numpy as np

    min_log_hz, lin_slope = 1000.0, 3.0 / 200.0
    min_log_mel = min_log_hz * lin_slope
    log_step = math.log(6.4) / 27.0

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        return np.where(
            f < min_log_hz, f * lin_slope, min_log_mel + np.log(f / min_log_hz) / log_step
        )

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m < min_log_mel, m / lin_slope, min_log_hz * np.exp((m - min_log_mel) * log_step))

    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLING_RATE / 2.0), N_MEL + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.arange(N_FFT // 2 + 1, dtype=np.float64) * (SAMPLING_RATE / N_FFT)
    filters = np.zeros((N_MEL, N_FFT // 2 + 1), dtype=np.float64)
    for m in range(N_MEL):
        left, center, right = hz_points[m : m + 3]
        enorm = 2.0 / max(1e-30, right - left)
        up = (bins - left) / max(1e-30, center - left)
        down = (right - bins) / max(1e-30, right - center)
        filters[m] = np.maximum(0.0, np.minimum(up, down)) * enorm
    return filters.astype(np.float32)


_FILTERS = None
_FILTERS_LOCK = threading.Lock()


def log_mel(samples):
    """float32 [n_mel, mel_frames] of a mono 16 kHz clip."""
    import numpy as np

    global _FILTERS
    padded = np.zeros(pad_length(len(samples)), dtype=np.float32)
    padded[: len(samples)] = samples
    count = 1 + (len(padded) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(count, N_FFT),
        strides=(HOP_LENGTH * 4, 4),
        writeable=False,
    )
    windowed = frames * _hann(WIN_LENGTH)[None, :]
    power = np.abs(np.fft.rfft(windowed, n=N_FFT, axis=1)) ** 2
    with _FILTERS_LOCK:
        if _FILTERS is None:
            _FILTERS = _slaney_mel_filters()
        filters = _FILTERS
    mel = filters @ power.T
    log = np.log10(np.maximum(mel, 1e-30))
    np.maximum(log, LOG_FLOOR, out=log)
    return ((log + NORMALIZE_OFFSET) / NORMALIZE_DIVISOR).astype(np.float32)


SPEECH_RATE = 24_000


def pcm_f32_to_s16le(data: bytes) -> bytes:
    """Generated fp32 little-endian PCM to 16-bit little-endian, clamped."""
    import numpy as np

    samples = np.frombuffer(data, dtype="<f4")
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_s16le(pcm: bytes, rate: int = SPEECH_RATE) -> bytes:
    """A mono 16-bit PCM RIFF/WAV wrapping s16le bytes."""
    import io as _io
    import wave as _wave

    out = _io.BytesIO()
    with _wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


def prepare(payload: bytes) -> PreparedAudio:
    """Decode, mix to mono, resample to 16 kHz and compute the mel."""
    samples, rate = decode_wave(payload)
    if not len(samples):
        raise AudioError("audio is empty")
    if len(samples) / rate > MAX_AUDIO_SECONDS:
        raise AudioError("audio exceeds the duration limit")
    mel = log_mel(resample(samples, rate))
    packed = mel.tobytes()
    digest = hashlib.sha256(struct.pack("<II", N_MEL, mel.shape[1]) + packed)
    digest_lo, digest_hi = struct.unpack_from("<QQ", digest.digest())
    return PreparedAudio(mel.shape[1], packed, digest_lo, digest_hi)


class PreparedClips(list):
    """Append-only request batch; its owner travels with the mel payload,
    as PreparedImages does for pixels."""

    def __init__(self, cache):
        super().__init__()
        self._cache = cache
        self._bytes = 0

    def append(self, clip):
        size = len(clip.mel)
        with self._cache._lock:
            if size > self._cache.REQUEST_BUDGET_BYTES - self._cache._request_bytes:
                raise AudioCapacityError("in-flight audio memory budget is full")
            self._cache._request_bytes += size
            self._bytes += size
        super().append(clip)

    def __del__(self):
        with self._cache._lock:
            self._cache._request_bytes -= self._bytes


class AudioCache:
    """Byte-budgeted LRU of prepared mels keyed by the decoded clip bytes."""

    BUDGET_BYTES = 256 * 1024 * 1024
    REQUEST_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: OrderedDict[bytes, PreparedAudio] = OrderedDict()
        self._bytes = 0
        self._request_bytes = 0

    def request_batch(self):
        return PreparedClips(self)

    def prepare(self, payload: bytes) -> PreparedAudio:
        key = hashlib.sha256(payload).digest()
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached
        prepared = prepare(payload)
        if len(prepared.mel) > self.BUDGET_BYTES:
            return prepared
        with self._lock:
            if key not in self._entries:
                self._entries[key] = prepared
                self._bytes += len(prepared.mel)
                while self._bytes > self.BUDGET_BYTES:
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= len(evicted.mel)
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

"""The engine's side of the native wire, for fake engines and golden tests;
production is server/protocol.py."""

import array
import json
import struct
from dataclasses import replace

from server import protocol as wire

_HEADER = struct.Struct("<4sHHHHQI")
_REQUEST = struct.Struct("<QBBQQIIIIffIffffQBIIII")
_IMAGE_SPAN = struct.Struct("<IIIIQQI")
_AUDIO_SPAN = struct.Struct("<IIIQQ")
_MASK_RESPONSE = struct.Struct("<QQI")
_FORCED_TOKENS = struct.Struct("<QI")
_VOCABULARY = struct.Struct("<II")
_CONSTRAINT_DFA = struct.Struct("<QII")
_ID = struct.Struct("<Q")
_LEN = struct.Struct("<I")


def request_frame(**overrides) -> wire.RequestFrame:
    """A valid text request with neutral values, each override replacing one
    field."""
    request = wire.RequestFrame(
        request_id=1,
        priority=wire.RequestPriority.NORMAL,
        absolute_deadline_unix_micros=1,
        remaining_deadline_micros=1,
        logical_max_output_tokens=1,
        prompt_tokens=(1,),
        sampling=wire.SamplingParameters(),
        seed=0,
        constraint=wire.ConstraintMode.NONE,
        image_spans=(),
        image_pixels=b"",
        audio_spans=(),
        audio_mel=b"",
        return_progress=False,
        score_tokens=(),
        generation_prompt_tokens=0,
        flags=wire.RequestFlag(0),
    )
    return replace(request, **overrides)


def status_event(correlation_id: int = 1, **fields) -> wire.StatusJsonEvent:
    """A schema-current status document of a ready, healthy engine; each field
    replaces a top-level key, and None removes it."""
    document = {
        "schema_version": wire.STATUS_SCHEMA_VERSION,
        "ready": True,
        "memory_pressure": "normal",
        "metal": {"healthy": True},
    }
    for key, value in fields.items():
        if value is None:
            document.pop(key, None)
        else:
            document[key] = value
    return wire.StatusJsonEvent(
        correlation_id, json.dumps(document, separators=(",", ":")).encode()
    )


def _event_payload(event: wire.EngineEvent) -> tuple[wire.FrameType, bytes]:
    match event:
        case wire.ReadyEvent():
            return wire.FrameType.READY, struct.pack(
                "<IIBBBx",
                event.max_concurrent_requests,
                event.max_context_tokens,
                event.vision,
                event.audio,
                event.speech,
            )
        case wire.StartEvent():
            return wire.FrameType.START, struct.pack(
                "<QII", event.request_id, event.lane, event.matched_prompt_tokens
            )
        case wire.PromptProgressEvent():
            return wire.FrameType.PROMPT_PROGRESS, struct.pack(
                "<QIQ", event.request_id, event.processed_tokens, event.elapsed_micros
            )
        case wire.TokensEvent():
            return wire.FrameType.TOKENS, struct.pack(
                f"<QII{len(event.tokens)}I",
                event.request_id,
                event.sequence_offset,
                len(event.tokens),
                *event.tokens,
            )
        case wire.MaskRequestEvent():
            return wire.FrameType.MASK_REQUEST, struct.pack(
                f"<QQII{len(event.simulation_tokens)}I",
                event.request_id,
                event.mask_request_id,
                event.words_per_mask,
                len(event.simulation_tokens),
                *event.simulation_tokens,
            )
        case wire.DoneEvent():
            return wire.FrameType.DONE, struct.pack(
                f"<QBIIQQQIII{len(event.option_logits)}f",
                event.request_id,
                event.reason,
                event.prompt_tokens,
                event.completion_tokens,
                event.prefill_micros,
                event.decode_micros,
                event.wall_micros,
                event.drafted_tokens,
                event.accepted_draft_tokens,
                len(event.option_logits),
                *event.option_logits,
            )
        case wire.ErrorEvent():
            return wire.FrameType.ERROR, struct.pack(
                f"<BBQII{len(event.code)}s{len(event.message)}s",
                event.failure_class,
                event.retryable,
                event.request_id,
                len(event.code),
                len(event.message),
                event.code,
                event.message,
            )
        case wire.StatusJsonEvent():
            return wire.FrameType.STATUS_JSON, _ID.pack(
                event.correlation_id
            ) + event.json
    raise TypeError(f"{type(event).__name__} is not an engine event")


def serialize_event(event: wire.EngineEvent) -> bytes:
    """Header and payload exactly as the engine writes them, without
    validation, so fakes can send the server invalid events."""
    frame_type, payload = _event_payload(event)
    header = _HEADER.pack(
        b"RICH",
        wire.PROTOCOL_VERSION,
        wire.FRAME_HEADER_BYTES,
        frame_type,
        0,
        len(payload),
        0,
    )
    return header + payload


def _words(payload: bytes, offset: int, count: int) -> tuple[int, ...]:
    return struct.unpack_from(f"<{count}I", payload, offset)


def _decode_request(payload: bytes) -> wire.RequestFrame:
    (
        request_id,
        priority,
        constraint,
        absolute_deadline,
        remaining_deadline,
        max_output,
        prompt_count,
        span_count,
        audio_count,
        *sampling,
        seed,
        return_progress,
        score_count,
        generation_prompt_tokens,
        flags,
        shared_prefix_tokens,
    ) = _REQUEST.unpack_from(payload)
    offset = _REQUEST.size
    prompt = _words(payload, offset, prompt_count)
    offset += 4 * prompt_count
    spans = tuple(
        wire.ImageSpan(
            *_IMAGE_SPAN.unpack_from(payload, offset + index * _IMAGE_SPAN.size)
        )
        for index in range(span_count)
    )
    offset += _IMAGE_SPAN.size * span_count
    audios = tuple(
        wire.AudioSpan(
            *_AUDIO_SPAN.unpack_from(payload, offset + index * _AUDIO_SPAN.size)
        )
        for index in range(audio_count)
    )
    offset += _AUDIO_SPAN.size * audio_count
    pixel_bytes = sum(span.pixel_bytes for span in spans)
    pixels = bytes(payload[offset : offset + pixel_bytes])
    offset += pixel_bytes
    mel_bytes = sum(span.mel_bytes for span in audios)
    mel = bytes(payload[offset : offset + mel_bytes])
    scores = _words(payload, offset + mel_bytes, score_count)
    return wire.RequestFrame(
        request_id,
        wire.RequestPriority(priority),
        absolute_deadline,
        remaining_deadline,
        max_output,
        prompt,
        wire.SamplingParameters(*sampling),
        seed,
        wire.ConstraintMode(constraint),
        spans,
        pixels,
        audios,
        mel,
        bool(return_progress),
        scores,
        generation_prompt_tokens,
        wire.RequestFlag(flags),
        shared_prefix_tokens,
    )


def decode_client_frame(frame_type: int, payload: bytes) -> wire.ClientMessage:
    """One client frame's message, read as the engine reads it but without
    checking the rules the server's encoder already enforced."""
    match frame_type:
        case wire.FrameType.REQUEST:
            return _decode_request(payload)
        case wire.FrameType.CANCEL:
            return wire.CancelFrame(*_ID.unpack(payload))
        case wire.FrameType.MASK_RESPONSE:
            request_id, mask_request_id, count = _MASK_RESPONSE.unpack_from(payload)
            mask = payload[_MASK_RESPONSE.size : _MASK_RESPONSE.size + 4 * count]
            return wire.MaskResponseFrame(request_id, mask_request_id, bytes(mask))
        case wire.FrameType.FORCED_TOKENS:
            request_id, count = _FORCED_TOKENS.unpack_from(payload)
            tokens = _words(payload, _FORCED_TOKENS.size, count)
            return wire.ForcedTokensFrame(request_id, tokens)
        case wire.FrameType.VOCABULARY:
            vocab, tbytes_len = _VOCABULARY.unpack_from(payload)
            tbytes = payload[
                _VOCABULARY.size : _VOCABULARY.size + tbytes_len
            ]
            offset = _VOCABULARY.size + tbytes_len
            offsets = _words(payload, offset, vocab + 1)
            offset += 4 * (vocab + 1)
            (spec_len,) = _LEN.unpack_from(payload, offset)
            spec_bits = payload[offset + _LEN.size : offset + _LEN.size + spec_len]
            special = tuple(
                index * 8 + bit
                for index, byte in enumerate(spec_bits)
                for bit in range(8)
                if byte >> bit & 1
            )
            tokens = tuple(
                bytes(tbytes[offsets[index] : offsets[index + 1]])
                for index in range(vocab)
            )
            return wire.VocabularyFrame(tokens, special)
        case wire.FrameType.CONSTRAINT_DFA:
            request_id, n_states, _flags = _CONSTRAINT_DFA.unpack_from(payload)
            table_bytes = n_states * 256 * 2
            offset = _CONSTRAINT_DFA.size
            table = payload[offset : offset + table_bytes]
            offset += table_bytes
            accepting_bytes = (n_states + 7) // 8
            accepting = payload[offset : offset + accepting_bytes]
            offset += accepting_bytes
            (eos_count,) = _LEN.unpack_from(payload, offset)
            offset += _LEN.size
            eos_ids = _words(payload, offset, eos_count)
            offset += 4 * eos_count
            (marker_len,) = _LEN.unpack_from(payload, offset)
            offset += _LEN.size
            marker = payload[offset : offset + marker_len]
            offset += marker_len
            # The optional budget section: u32 length plus n_states i16
            # termination bounds.
            costs = ()
            if offset < len(payload):
                (costs_len,) = _LEN.unpack_from(payload, offset)
                offset += _LEN.size
                costs = array.array("h")
                costs.frombytes(payload[offset : offset + costs_len])
                costs = tuple(costs)
            return wire.ConstraintDfaFrame(
                request_id,
                n_states,
                bytes(table),
                bytes(accepting),
                eos_ids,
                bytes(marker),
                costs,
            )
        case wire.FrameType.STATUS_REQUEST:
            return wire.StatusRequestFrame(*_ID.unpack(payload))
        case wire.FrameType.CORPUS_APPEND:
            session, count = struct.unpack_from("<QI", payload)
            return wire.CorpusAppendFrame(
                tuple(_words(payload, 12, count)), session
            )
    raise ValueError(f"frame type {frame_type:#06x} is not a client frame")


class ClientFrameReader:
    """Splits the server's byte stream into client frames."""

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data) -> list[tuple[wire.ClientMessage, bytes]]:
        """Each complete frame in what has arrived so far, as its message and
        its raw bytes."""
        self._buffer.extend(data)
        frames = []
        while len(self._buffer) >= wire.FRAME_HEADER_BYTES:
            magic, version, header_bytes, frame_type, _, payload_bytes, _ = (
                _HEADER.unpack_from(self._buffer)
            )
            if (magic, version, header_bytes) != (
                b"RICH",
                wire.PROTOCOL_VERSION,
                wire.FRAME_HEADER_BYTES,
            ):
                raise ValueError("client frame header is invalid")
            end = wire.FRAME_HEADER_BYTES + payload_bytes
            if len(self._buffer) < end:
                break
            raw = bytes(self._buffer[:end])
            del self._buffer[:end]
            frames.append(
                (decode_client_frame(frame_type, raw[wire.FRAME_HEADER_BYTES :]), raw)
            )
        return frames

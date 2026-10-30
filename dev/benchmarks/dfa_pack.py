"""Emit the dfa-benchmark pack: the packed token vocabulary, a compiled
byte-DFA for a tool-call-style JSON schema, and a tokenized prompt.

The pack mirrors what production ships over the wire — VOCABULARY and
CONSTRAINT_DFA frames — so the native benchmark installs the same payload
with engine.setVocabulary/engine.provideDfa and answers MASK_REQUEST events
with an in-process DFA mirror.

Usage: dfa_pack.py MODEL_ROOT OUT_PATH
Run with the repo venv: .venv/bin/python dev/benchmarks/dfa_pack.py \
    install/models/openbmb/MiniCPM5-2B-MLX build/dfa-bench.pack
"""

import array
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoTokenizer
from llguidance.hf import from_tokenizer as guidance_tokenizer

from server import constraints, dfa, serving

MAGIC = b"DFAPACK\x02"

# A tool-call-shaped payload: literal stretches ({"name":" ... ,"arguments":{"
# produce forced spans), with a free string, an integer, an enum and an array
# of strings for branching variety.
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "arguments": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"enum": ["metric", "imperial"]},
                "alerts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
            },
            "required": ["city", "days", "units"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "arguments"],
    "additionalProperties": False,
}

PROMPT_TEXT = (
    "You are a precise tool-calling assistant. Call the weather tool and "
    "reply only with the JSON object for get_weather(city='Paris', days=3, "
    "units='metric'). Do not explain the call.\n\nTool call: get_weather("
)


def main():
    if len(sys.argv) != 3:
        print("usage: dfa_pack.py MODEL_ROOT OUT_PATH", file=sys.stderr)
        return 2
    model_root = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    hf = AutoTokenizer.from_pretrained(
        model_root / "tokenizer", local_files_only=True,
        fix_mistral_regex=True,
    )
    contract = constraints.validate_tokenizer(
        hf, str(model_root), serving.load(str(model_root))
    )
    guidance = guidance_tokenizer(
        hf,
        n_vocab=contract.vocabulary,
        eos_token=list(contract.eos_tokens),
    )
    vocab_tokens = constraints._vocab_token_bytes(hf, contract.vocabulary)
    special = frozenset(
        token_id
        for token_id in range(contract.vocabulary)
        if guidance.is_special_token(token_id)
    )
    markers = tuple(
        vocab_tokens[token_id] for token_id in sorted(special) if vocab_tokens[token_id]
    )
    spec = dfa.json_spec(SCHEMA, False, think_end=contract.think_end)
    compiled = dfa.compile_dfa(spec, markers)
    if compiled is None:
        raise SystemExit("schema did not compile to a byte DFA")
    eos_states = bytes(
        acc | ok for acc, ok in zip(compiled.accepting, compiled.eos_ok)
    )
    marker = b"".join(m.to_bytes(32, "little") for m in compiled.marker)

    prompt = hf.encode(PROMPT_TEXT, add_special_tokens=False)
    if not prompt or any(not 0 <= t < contract.vocabulary for t in prompt):
        raise SystemExit("prompt tokenization produced out-of-vocabulary ids")

    token_bytes = b"".join(vocab_tokens)
    offsets = array.array("i", [0])
    offset = 0
    for token in vocab_tokens:
        offset += len(token)
        offsets.append(offset)
    spec_bits = bytearray((contract.vocabulary + 7) // 8)
    for token_id in special:
        spec_bits[token_id >> 3] |= 1 << (token_id & 7)
    eos_ids = sorted(contract.eos_tokens)
    if len(eos_ids) > 8:
        raise SystemExit("grammar eos set exceeds the frame limit of 8")

    header = struct.pack(
        "<8s8I",
        MAGIC,
        contract.vocabulary,
        len(token_bytes),
        compiled.n_states,
        len(eos_states),
        len(marker),
        len(eos_ids),
        len(spec_bits),
        len(prompt),
    )
    body = bytearray()
    body += token_bytes
    body += offsets.tobytes()
    body += spec_bits
    body += compiled.table.tobytes()
    body += eos_states
    body += marker
    body += array.array("I", eos_ids).tobytes()
    body += array.array("I", prompt).tobytes()
    # The budget rule's per-state termination bounds, the engine's B_bytes
    # cost: single byte-unmarked walks bottom out at their acceptance.
    costs = compiled.byte_costs()
    body += array.array("h", costs).tobytes()
    bounded = [c for c in costs if c != 32767]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + body)
    print(
        f"dfa-pack: vocab={contract.vocabulary} token_bytes={len(token_bytes)} "
        f"special={len(special)} dfa_states={compiled.n_states} "
        f"eos={eos_ids} prompt_tokens={len(prompt)} "
        f"costs={len(costs)} bounded={len(bounded)} "
        f"max_bound={max(bounded) if bounded else 0} -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())

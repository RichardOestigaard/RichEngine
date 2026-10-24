"""In-process divergence hunt: MiniCPM5 tokenizer, flat 6-field schema.

Walks the DFA along the canonical tokenization of a plausible output and
diffs the DFA's admissible set against LLMatcher's mask at every state.
"""

import array
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from llguidance import LLMatcher, LLExecutor
from llguidance.hf import from_tokenizer as guidance_tokenizer
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask_par

from server import constraints, dfa, serving, tool_schema

TOK_DIR = "install/models/openbmb/MiniCPM5-2B-GGUF:Q4_K_M/tokenizer"
MODEL_DIR = "install/models/openbmb/MiniCPM5-2B-GGUF:Q4_K_M"

SCHEMA = {
    "type": "object",
    "properties": {
        f"field_{i}": {"type": "string"} for i in range(6)
    },
    "required": [f"field_{i}" for i in range(6)],
    "additionalProperties": False,
}


def main():
    hf = AutoTokenizer.from_pretrained(TOK_DIR, local_files_only=True)
    contract = constraints.validate_tokenizer(hf, MODEL_DIR, serving.load(MODEL_DIR))
    print("vocab", contract.vocabulary, "eos", contract.eos_tokens,
          "think_end", contract.think_end_id)
    guidance = guidance_tokenizer(
        hf,
        n_vocab=contract.vocabulary,
        eos_token=list(contract.eos_tokens),
    )
    executor = LLExecutor()
    vocab_tokens = constraints._vocab_token_bytes(hf, contract.vocabulary)
    special = frozenset(
        t for t in range(contract.vocabulary) if guidance.is_special_token(t)
    )
    markers = tuple(
        vocab_tokens[t] for t in sorted(special) if vocab_tokens[t]
    )
    grammar = tool_schema.json_grammar(
        SCHEMA, False, think_end_id=contract.think_end_id,
        think_end_text=contract.think_end,
    )
    spec = dfa.json_spec(SCHEMA, False, think_end=contract.think_end)
    compiled = dfa.compile_dfa(spec, markers)
    print("dfa states:", None if compiled is None else compiled.n_states)
    if compiled is None:
        return
    matcher = LLMatcher(guidance, grammar, log_level=0)
    assert not matcher.is_error(), matcher.get_error()

    def mask_ids(m):
        bitmask = allocate_token_bitmask(1, guidance.vocab_size)
        fill_next_token_bitmask_par(executor, [(m, 0)], bitmask)
        words = array.array("I", bitmask.tobytes())
        return {
            i * 32 + b
            for i, w in enumerate(words)
            for b in range(32)
            if w >> b & 1
        }

    eos = frozenset(contract.eos_tokens)

    # Walk the canonical tokenization of a plausible output; at each step
    # diff the two admissible sets.
    output = '{"field_0":"ok","field_1":"hi","field_2":"yes","field_3":"no","field_4":"done","field_5":"end"}'
    ids = hf.encode(output, add_special_tokens=False)
    print("canonical ids:", ids)
    print("canonical text:", [hf.decode([t]) for t in ids])

    state = 0
    diverged = False
    prefix = []
    for step, tid in enumerate(ids):
        exp = mask_ids(matcher)
        got = set(
            compiled.masks_for_state(state, vocab_tokens, eos, special)
        )
        if exp != got:
            extra = sorted(got - exp)
            missing = sorted(exp - got)
            print(f"step {step} prefix={bytes(b''.join(vocab_tokens[t] for t in prefix))!r}")
            print(f"  DFA-only ({len(extra)}):",
                  [(t, vocab_tokens[t]) for t in extra[:40]])
            print(f"  matcher-only ({len(missing)}):",
                  [(t, vocab_tokens[t]) for t in missing[:40]])
            diverged = True
        if not matcher.consume_token(tid):
            print(f"step {step}: matcher rejects emitted token {tid} {vocab_tokens[tid]!r}")
            diverged = True
            break
        nxt = compiled.advance_token(state, tid, vocab_tokens, eos, special)
        if nxt < 0:
            print(f"step {step}: DFA rejects canonical token {tid} {vocab_tokens[tid]!r}")
            diverged = True
            break
        state = nxt
        prefix.append(tid)
    if not diverged:
        print("no divergence along canonical walk")
    # also compare at start state only exhaustively already done above


if __name__ == "__main__":
    main()

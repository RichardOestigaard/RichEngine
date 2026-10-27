# Integrating with RichEngine

RichEngine serves OpenAI-compatible HTTP on `http://127.0.0.1:8123`
(`RICHENGINE_PORT` to move it). Everything below is optional decoration
on top of plain chat completions.

## Endpoints

### Generation

- `POST /v1/chat/completions` — messages, tools, streaming, `temperature`,
  `max_tokens`, `chat_template_kwargs` (e.g. `enable_thinking`).
- `POST /v1/completions`, `POST /v1/responses`, `POST /v1/messages`
  (Anthropic shape, `POST /v1/messages/count_tokens` for budgeting).
- `POST /v1/judgments`, `POST /v1/systemone` — finite-option scoring
  through the decision pathway.
- `POST /tokenize`, `POST /apply-template` — prompt plumbing probes.

### Corpus (n-gram speculative reuse)

- `POST /v1/corpus` — `{text, emit_variants}` tokenizes and pushes a
  stream into the cross-request n-gram corpus (workspace files, tool
  outputs). `emit_variants: true` also indexes diff-prefixed and
  JSON-escaped copies — use it for any text the model may re-emit in a
  different format.

### Control

- `GET /v1/models`, `GET /status`, `GET /metrics`, `GET /health`,
  `GET /ready`, `GET /v1/disk`
- `POST /v1/models/load|unload|install|tune`, `POST /v1/disk/wipe|dedupe`,
  `POST /v1/tools/fetch`, `POST /v1/tools/search`

## When to call what

| Situation | Do this |
|---|---|
| Agent loop / coding session | Plain chat completions; the corpus self-arms on multi-turn (prompt-prefix containment), tool calls, and session linkage. |
| Your tool reads a file / runs a command | `POST /v1/corpus` with the result — the model can then re-emit it cheaply even if it never appeared in a prompt. |
| Model will emit diffs / JSON-escaped text | Same push, `emit_variants: true`. |
| OpenCode / Claude Code launch | `richengine opencode` / `richengine claude` — corpus defaults on, workspace hook auto-installed. |
| Long open-ended generation | Nothing to do; the EWMA gate keeps corpus out of the way. |
| Per-request isolation | Today: no request-level corpus opt-out; `RICHENGINE_NGRAM_CORPUS=0` at serve disables globally. |

## Knobs (env, serve time)

- `RICHENGINE_NGRAM_CORPUS` — 0 off / 1 all lanes / **2 hinted+self-armed (default)**.
- `RICHENGINE_NGRAM_CORPUS_ALL` — **1 (default)**: prompt+output streams; `0`: output-only.
- `RICHENGINE_NGRAM_CORPUS_MARGIN` — preference margin δ, default 5.
- `RICHENGINE_NGRAM_HINT` — server-side hinting, **default on**, `=0` off.
- `RICHENGINE_TRACE_TOKENS=/path.jsonl` — record every request for offline replay (`dev/benchmarks/ngram_replay.py`).

## Guarantees

Speculative decode is lossless: emitted tokens are byte-identical to
non-speculative. Corpus proposals that fail verify cost rows, never
output correctness.

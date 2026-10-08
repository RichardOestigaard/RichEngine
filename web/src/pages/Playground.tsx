import { createMemo, createSignal, onMount, For, Show } from "solid-js";
import { useSearchParams } from "@solidjs/router";
import { Tab, TabGroup, TabList, TabPanel } from "terracotta";
import { ApiError, api, authorization, errorText, noteAuthStatus } from "../api";
import { useCopied } from "../clipboard";
import { fmtInt } from "../format";
import { pushToast } from "../toast";
import "./playground.css";

/* ---- /tokenize + /apply-template + /v1/completions payload shapes
   (server/frontend.py, server/server.py) ---- */

interface TokenizeResponse {
  tokens?: number[];
  /* Char spans into the submitted content, present with with_pieces. */
  offsets?: [number, number][];
}

interface TemplateResponse {
  prompt?: string;
}

interface CompletionResponse {
  id?: string;
  model?: string;
  choices?: { text?: string; finish_reason?: string }[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

interface MessageRow {
  role: "system" | "user" | "assistant";
  content: string;
}

const ROLES: MessageRow["role"][] = ["system", "user", "assistant"];

const TAB_IDS = new Set(["tokenize", "template", "completions"]);

function isAbort(error: unknown): boolean {
  return (error as { name?: string }).name === "AbortError";
}

/* The ids input accepts "1, 2, 3" or a JSON array "[1,2,3]"; every element
   must be a non-negative integer or the whole parse fails. */
function parseTokenIds(text: string): { ids?: number[]; error?: string } {
  const trimmed = text.trim();
  if (!trimmed) return { error: "token ids must not be empty" };
  let raw: unknown[];
  if (trimmed.startsWith("[")) {
    try {
      raw = JSON.parse(trimmed) as unknown[];
    } catch {
      return { error: "token ids: invalid JSON array" };
    }
    if (!Array.isArray(raw)) return { error: "token ids must be a JSON array" };
  } else {
    raw = trimmed
      .split(/[\s,]+/)
      .filter((part) => part !== "")
      .map(Number);
  }
  if (!raw.length) return { error: "token ids must not be empty" };
  for (const value of raw) {
    if (typeof value !== "number" || !Number.isInteger(value) || value < 0)
      return { error: `${JSON.stringify(value)} is not a non-negative integer` };
  }
  return { ids: raw as number[] };
}

/* All tab state lives at module scope: navigating away and back keeps the
   forms and the last results instead of resetting every tab. An in-flight
   request keeps streaming while the page is away, too. */
const [modelIds, setModelIds] = createSignal<string[]>([]);

/* Tokenize form + result. */
const [content, setContent] = createSignal("");
const [addSpecial, setAddSpecial] = createSignal(false);
const [tokBusy, setTokBusy] = createSignal(false);
const [tokError, setTokError] = createSignal<string | null>(null);
const [tokController, setTokController] = createSignal<AbortController>();
const [tokCancelled, setTokCancelled] = createSignal(false);
const [tokResult, setTokResult] = createSignal<{
  ids: number[];
  chars: number;
  /* The submitted text — the textarea may have moved on since. */
  text: string;
  offsets: [number, number][] | null;
} | null>(null);
/* Chip ↔ source-span hover sync, by token index. */
const [hoverTok, setHoverTok] = createSignal<number | null>(null);

/* Template form + result. */
const [messages, setMessages] = createSignal<MessageRow[]>([
  { role: "user", content: "" },
]);
const [genPrompt, setGenPrompt] = createSignal(true);
const [tplModel, setTplModel] = createSignal("");
const [tplBusy, setTplBusy] = createSignal(false);
const [tplError, setTplError] = createSignal<string | null>(null);
const [tplController, setTplController] = createSignal<AbortController>();
const [tplCancelled, setTplCancelled] = createSignal(false);
const [tplResult, setTplResult] = createSignal<string | null>(null);

/* Completions form + result. */
const [compMode, setCompMode] = createSignal<"text" | "ids">("text");
const [promptText, setPromptText] = createSignal("");
const [idsText, setIdsText] = createSignal("");
const [maxTokens, setMaxTokens] = createSignal("64");
const [temperature, setTemperature] = createSignal("");
const [stopText, setStopText] = createSignal("");
const [compModel, setCompModel] = createSignal("");
const [compStream, setCompStream] = createSignal(true);
const [compBusy, setCompBusy] = createSignal(false);
const [compError, setCompError] = createSignal<string | null>(null);
const [compController, setCompController] = createSignal<AbortController>();
const [compCancelled, setCompCancelled] = createSignal(false);
/* Text streamed so far, while a streaming completion is in flight. */
const [compLive, setCompLive] = createSignal("");
const [compResult, setCompResult] = createSignal<{
  res: CompletionResponse;
  ms: number;
} | null>(null);

export default function Playground() {
  /* Which control last copied. */
  const { copied, copy } = useCopied();

  /* The active tab lives in the hash — #/playground?tab=completions
     survives reloads, and the cross-tool buttons go through setTab. */
  const [params, setParams] = useSearchParams();
  const tab = () => {
    const value = params.tab;
    return typeof value === "string" && TAB_IDS.has(value)
      ? value
      : "tokenize";
  };
  const setTab = (id: string) => setParams({ tab: id }, { replace: true });

  let sourceRef: HTMLDivElement | undefined;

  onMount(() => {
    api<{ data?: { id?: string }[] }>("/v1/models")
      .then((res) => {
        const ids = (res?.data ?? [])
          .map((entry) => entry.id)
          .filter((id): id is string => !!id);
        setModelIds(ids);
        if (ids[0]) {
          if (!tplModel()) setTplModel(ids[0]);
          if (!compModel()) setCompModel(ids[0]);
        }
      })
      .catch(() => {
        /* Model fields stay manual when the list cannot load. */
      });
  });

  async function copyText(which: string, text: string) {
    await copy(which, text);
  }

  /* ---- POST /tokenize ---- */

  function tokenizeRequest(withPieces: boolean, signal: AbortSignal) {
    return api<TokenizeResponse>("/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: content(),
        add_special: addSpecial(),
        with_pieces: withPieces,
      }),
      signal,
    });
  }

  async function submitTokenize(event: SubmitEvent) {
    event.preventDefault();
    if (tokBusy()) return;
    setTokError(null);
    setTokBusy(true);
    setTokCancelled(false);
    const ctrl = new AbortController();
    setTokController(ctrl);
    try {
      /* Ask for per-token source spans; older servers and non-fast
         tokenizers 400 on the option — retry plain. */
      let res: TokenizeResponse;
      try {
        res = await tokenizeRequest(true, ctrl.signal);
      } catch (e) {
        if (e instanceof ApiError && /with_pieces|offset/i.test(e.message))
          res = await tokenizeRequest(false, ctrl.signal);
        else throw e;
      }
      setTokResult({
        ids: res.tokens ?? [],
        chars: content().length,
        text: content(),
        offsets: res.offsets ?? null,
      });
    } catch (e) {
      /* A deliberate cancel is a note, not an error. */
      if (isAbort(e)) setTokCancelled(true);
      else pushToast("error", errorText(e));
    } finally {
      setTokController(undefined);
      setTokBusy(false);
    }
  }

  const charsPerToken = () => {
    const res = tokResult();
    if (!res || !res.ids.length) return "—";
    return `${(res.chars / res.ids.length).toFixed(2)} chars/token`;
  };

  /* Source slices mapped to token indices. Offsets are char spans into the
     submitted text; specials (0,0) and non-monotonic pairs stay unmapped. */
  const tokSpans = createMemo(() => {
    const res = tokResult();
    if (!res || !res.offsets)
      return [] as { i: number; text: string; gap: boolean }[];
    const src = res.text;
    const out: { i: number; text: string; gap: boolean }[] = [];
    let cursor = 0;
    res.offsets.forEach((pair, i) => {
      const start = pair?.[0] ?? 0;
      const end = pair?.[1] ?? 0;
      if (end <= start || start < cursor || end > src.length) return;
      if (start > cursor)
        out.push({ i: -1, text: src.slice(cursor, start), gap: true });
      out.push({ i, text: src.slice(start, end), gap: false });
      cursor = end;
    });
    if (cursor < src.length)
      out.push({ i: -1, text: src.slice(cursor), gap: true });
    return out;
  });

  /* Token index → source piece, for chip tooltips and dimming unmapped ids. */
  const tokPieces = createMemo(() => {
    const map = new Map<number, string>();
    for (const span of tokSpans()) if (!span.gap) map.set(span.i, span.text);
    return map;
  });

  /* Hovering a chip scrolls its source span into view. */
  function hoverChip(i: number) {
    setHoverTok(i);
    sourceRef
      ?.querySelector(`[data-tok="${i}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }

  /* Hand the last tokenization to the completions form as raw ids. */
  function sendToCompletions() {
    setIdsText((tokResult()?.ids ?? []).join(", "));
    setCompMode("ids");
    setTab("completions");
  }

  /* ---- POST /apply-template ---- */

  function setMessage(index: number, patch: Partial<MessageRow>) {
    setMessages((rows) =>
      rows.map((row, i) => (i === index ? { ...row, ...patch } : row))
    );
  }

  function templateBody(): Record<string, unknown> {
    const body: Record<string, unknown> = {
      messages: messages().map((row) => ({ role: row.role, content: row.content })),
      add_generation_prompt: genPrompt(),
    };
    const model = tplModel().trim();
    if (model) body.model = model;
    return body;
  }

  async function submitTemplate(event: SubmitEvent) {
    event.preventDefault();
    if (tplBusy()) return;
    if (!messages().length) {
      setTplError("add at least one message");
      return;
    }
    setTplError(null);
    setTplBusy(true);
    setTplCancelled(false);
    const ctrl = new AbortController();
    setTplController(ctrl);
    try {
      const res = await api<TemplateResponse>("/apply-template", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(templateBody()),
        signal: ctrl.signal,
      });
      setTplResult(res.prompt ?? "");
    } catch (e) {
      if (isAbort(e)) setTplCancelled(true);
      else pushToast("error", errorText(e));
    } finally {
      setTplController(undefined);
      setTplBusy(false);
    }
  }

  /* Hand the rendered prompt back to the tokenizer. */
  function sendToTokenize() {
    setContent(tplResult() ?? "");
    setTab("tokenize");
  }

  /* ---- POST /v1/completions ---- */

  function completionBody(): { body: Record<string, unknown>; error?: string } {
    const body: Record<string, unknown> = {};
    if (compMode() === "text") {
      if (!promptText().trim())
        return { body, error: "prompt must be nonempty" };
      body.prompt = promptText();
    } else {
      const { ids, error } = parseTokenIds(idsText());
      if (error) return { body, error };
      body.prompt = ids;
    }
    const max = maxTokens().trim();
    if (max) {
      const n = Number(max);
      if (!Number.isInteger(n) || n <= 0)
        return { body, error: "max_tokens must be a positive integer" };
      body.max_tokens = n;
    }
    const temp = temperature().trim();
    if (temp) {
      const t = Number(temp);
      if (!Number.isFinite(t))
        return { body, error: "temperature must be a number" };
      body.temperature = t;
    }
    const stops = stopText()
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (stops.length) body.stop = stops;
    const model = compModel().trim();
    if (model) body.model = model;
    return { body };
  }

  /* text_completion chunks: choices[].text deltas; usage arrives on a
     choices-empty chunk when stream_options.include_usage is set. */
  interface CompletionChunk {
    id?: string;
    model?: string;
    choices?: { text?: string; finish_reason?: string }[];
    usage?: CompletionResponse["usage"];
  }

  async function streamCompletion(
    body: Record<string, unknown>,
    signal: AbortSignal
  ): Promise<CompletionResponse> {
    const response = await fetch("/v1/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authorization() },
      body: JSON.stringify({
        ...body,
        stream: true,
        stream_options: { include_usage: true },
      }),
      signal,
    });
    noteAuthStatus(response.status);
    if (!response.ok) {
      const data = (await response.json().catch(() => null)) as {
        error?: { message?: string };
      } | null;
      const message = data?.error?.message;
      throw new ApiError(
        response.status,
        typeof message === "string" && message
          ? message
          : response.statusText
      );
    }
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    const res: CompletionResponse = {};
    let buffer = "";
    let text = "";
    let doneEvent = false;
    const consume = (event: string) => {
      const data = event
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (!data) return;
      if (data === "[DONE]") {
        doneEvent = true;
        return;
      }
      const chunk = JSON.parse(data) as CompletionChunk;
      if (chunk.id) res.id = chunk.id;
      if (chunk.model) res.model = chunk.model;
      if (chunk.usage) res.usage = chunk.usage;
      const choice = chunk.choices?.[0];
      if (choice?.text) {
        text += choice.text;
        setCompLive(text);
      }
      if (choice?.finish_reason)
        res.choices = [{ text, finish_reason: choice.finish_reason }];
    };
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? "";
      for (const event of events) consume(event);
      if (done) break;
    }
    if (buffer) consume(buffer);
    if (!doneEvent) throw new Error("Connection closed early");
    if (!res.choices?.length) res.choices = [{ text }];
    return res;
  }

  async function submitCompletion(event: SubmitEvent) {
    event.preventDefault();
    if (compBusy()) return;
    const { body, error } = completionBody();
    setCompError(error ?? null);
    if (error) return;
    setCompBusy(true);
    setCompCancelled(false);
    setCompLive("");
    const ctrl = new AbortController();
    setCompController(ctrl);
    const started = performance.now();
    try {
      const res = compStream()
        ? await streamCompletion(body, ctrl.signal)
        : await api<CompletionResponse>("/v1/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: ctrl.signal,
          });
      setCompResult({ res, ms: performance.now() - started });
    } catch (e) {
      if (isAbort(e)) {
        setCompCancelled(true);
        /* Keep whatever streamed before the cancel. */
        if (compLive())
          setCompResult({
            res: {
              choices: [{ text: compLive(), finish_reason: "cancelled" }],
            },
            ms: performance.now() - started,
          });
      } else pushToast("error", errorText(e));
    } finally {
      setCompController(undefined);
      setCompBusy(false);
    }
  }

  return (
    <main class="page playground">
      <h1>Playground</h1>
      <p class="lede">Tokenizer and prompt workbench.</p>

      <TabGroup
        class="playground-tabs"
        horizontal
        value={tab()}
        onChange={(value) => setTab(value ?? "tokenize")}
      >
        <TabList class="tab-list">
          <Tab class="tab" value="tokenize" as="button" type="button">
            Tokenize
          </Tab>
          <Tab class="tab" value="template" as="button" type="button">
            Template
          </Tab>
          <Tab class="tab" value="completions" as="button" type="button">
            Completions
          </Tab>
        </TabList>

        {/* ---- POST /tokenize ---- */}
        <TabPanel value="tokenize">
          <form class="panel" onSubmit={submitTokenize}>
            <fieldset disabled={tokBusy()}>
              <label class="field">
                <span class="field-label">Content</span>
                <textarea
                  class="pg-area"
                  rows={6}
                  placeholder="Text to tokenize…"
                  spellcheck={false}
                  value={content()}
                  onInput={(e) => setContent(e.currentTarget.value)}
                />
              </label>
            </fieldset>
            {/* Outside the disabled fieldset so Cancel stays clickable. */}
            <div class="panel-row">
              <label
                class="pg-check grow muted"
                title="Prepend the tokenizer's special tokens (BOS etc.)"
              >
                <input
                  type="checkbox"
                  checked={addSpecial()}
                  onChange={(e) => setAddSpecial(e.currentTarget.checked)}
                />
                add_special — prepend the tokenizer's specials
              </label>
              <Show
                when={tokBusy()}
                fallback={
                  <button
                    type="submit"
                    class="btn small primary"
                    title="Encode the text into token ids"
                  >
                    Tokenize
                  </button>
                }
              >
                <button
                  type="button"
                  class="btn small danger"
                  title="Abort the request"
                  onClick={() => tokController()?.abort()}
                >
                  Cancel
                </button>
              </Show>
            </div>
          </form>

          <Show when={tokError()}>
            <p class="notice error">{tokError()}</p>
          </Show>
          <Show when={tokCancelled()}>
            <p class="notice muted">cancelled</p>
          </Show>

          <Show when={tokResult()}>
            {(res) => (
              <div class="panel pg-result">
                <div class="panel-row">
                  <span class="grow">
                    Ids <span class="badge">{fmtInt(res().ids.length)} tokens</span>
                  </span>
                  <span class="num">{charsPerToken()}</span>
                </div>
                <Show when={res().offsets}>
                  <div class="panel-row chips-row">
                    {/* The submitted text cut into its token spans; hovering
                        a span or an id chip below highlights both ends. */}
                    <div
                      class="tok-source"
                      ref={(el) => (sourceRef = el)}
                      aria-label="Source text marked per token"
                    >
                      <For each={tokSpans()}>
                        {(span) =>
                          span.gap ? (
                            <span class="tok-gap">{span.text}</span>
                          ) : (
                            <span
                              class="tok-span"
                              classList={{
                                alt: span.i % 2 === 1,
                                hot: hoverTok() === span.i,
                              }}
                              data-tok={span.i}
                              title={`token ${res().ids[span.i]}`}
                              onMouseEnter={() => setHoverTok(span.i)}
                              onMouseLeave={() => setHoverTok(null)}
                            >
                              {span.text}
                            </span>
                          )
                        }
                      </For>
                    </div>
                  </div>
                </Show>
                <div class="panel-row chips-row">
                  <div class="token-chips">
                    <For each={res().ids}>
                      {(id, i) => (
                        <span
                          class="chip mono"
                          classList={{
                            hot: hoverTok() === i(),
                            unmapped: !tokPieces().has(i()),
                          }}
                          title={(() => {
                            const piece = tokPieces().get(i());
                            return piece === undefined
                              ? `${id} — no source span (special token)`
                              : `${id} — “${piece}”`;
                          })()}
                          onMouseEnter={() => hoverChip(i())}
                          onMouseLeave={() => setHoverTok(null)}
                        >
                          {id}
                        </span>
                      )}
                    </For>
                  </div>
                </div>
                <div class="panel-row">
                  <button
                    type="button"
                    class="btn small grow-left"
                    title="Copy to clipboard"
                    onClick={() =>
                      void copyText("ids", JSON.stringify(res().ids))
                    }
                  >
                    {copied() === "ids" ? "copied ✓" : "Copy ids"}
                  </button>
                  <button
                    type="button"
                    class="btn small"
                    title="Use these ids as the completions prompt"
                    onClick={sendToCompletions}
                  >
                    Send to completions →
                  </button>
                </div>
              </div>
            )}
          </Show>
        </TabPanel>

        {/* ---- POST /apply-template ---- */}
        <TabPanel value="template">
          <form class="panel" onSubmit={submitTemplate}>
            <fieldset disabled={tplBusy()}>
              <div class="field">
                <span class="field-label">Messages</span>
                <For each={messages()}>
                  {(row, i) => (
                    <div class="msg-row">
                      <select
                        class="pg-input msg-role"
                        aria-label={`Message ${i() + 1} role`}
                        value={row.role}
                        onChange={(e) =>
                          setMessage(i(), {
                            role: e.currentTarget.value as MessageRow["role"],
                          })
                        }
                      >
                        <For each={ROLES}>
                          {(role) => <option value={role}>{role}</option>}
                        </For>
                      </select>
                      <textarea
                        class="pg-area msg-content"
                        rows={2}
                        placeholder="content"
                        aria-label={`Message ${i() + 1} content`}
                        spellcheck={false}
                        value={row.content}
                        onInput={(e) =>
                          setMessage(i(), { content: e.currentTarget.value })
                        }
                      />
                      <button
                        type="button"
                        class="btn small"
                        title={
                          messages().length <= 1
                            ? "At least one message is required"
                            : "Remove message"
                        }
                        aria-label={`Remove message ${i() + 1}`}
                        disabled={messages().length <= 1}
                        onClick={() =>
                          setMessages((rows) =>
                            rows.filter((_, j) => j !== i())
                          )
                        }
                      >
                        ×
                      </button>
                    </div>
                  )}
                </For>
                <button
                  type="button"
                  class="btn small"
                  title="Append another message row"
                  onClick={() =>
                    setMessages((rows) => [
                      ...rows,
                      { role: "user", content: "" },
                    ])
                  }
                >
                  Add message
                </button>
              </div>
              <div class="panel-row">
                <label
                  class="pg-check grow muted"
                  title="Append the assistant turn opener the model expects"
                >
                  <input
                    type="checkbox"
                    checked={genPrompt()}
                    onChange={(e) => setGenPrompt(e.currentTarget.checked)}
                  />
                  add_generation_prompt
                </label>
                <span class="muted">Model</span>
                <input
                  class="pg-input mono w-220"
                  list="playground-models"
                  placeholder="served model id"
                  aria-label="Model"
                  spellcheck={false}
                  value={tplModel()}
                  onInput={(e) => setTplModel(e.currentTarget.value)}
                />
              </div>
            </fieldset>
            {/* Outside the disabled fieldset so Cancel stays clickable. */}
            <div class="panel-row">
              <button
                type="button"
                class="btn small grow-left"
                title="Copy to clipboard"
                onClick={() =>
                  void copyText(
                    "template-body",
                    JSON.stringify(templateBody(), null, 2)
                  )
                }
              >
                {copied() === "template-body" ? "copied ✓" : "Copy request JSON"}
              </button>
              <Show
                when={tplBusy()}
                fallback={
                  <button
                    type="submit"
                    class="btn small primary"
                    title="Apply the chat template to these messages"
                  >
                    Render
                  </button>
                }
              >
                <button
                  type="button"
                  class="btn small danger"
                  title="Abort the request"
                  onClick={() => tplController()?.abort()}
                >
                  Cancel
                </button>
              </Show>
            </div>
          </form>

          <Show when={tplError()}>
            <p class="notice error">{tplError()}</p>
          </Show>
          <Show when={tplCancelled()}>
            <p class="notice muted">cancelled</p>
          </Show>

          <Show when={tplResult()}>
            {(prompt) => (
              <div class="panel pg-result">
                <div class="panel-row">
                  <span class="grow">Rendered prompt</span>
                  <span class="num">
                    {fmtInt(prompt().length)} chars · ~
                    {fmtInt(Math.ceil(prompt().length / 4))} tokens
                  </span>
                </div>
                <pre class="pg-output">{prompt()}</pre>
                <div class="panel-row">
                  <button
                    type="button"
                    class="btn small grow-left"
                    title="Copy to clipboard"
                    onClick={() => void copyText("prompt", prompt())}
                  >
                    {copied() === "prompt" ? "copied ✓" : "Copy prompt"}
                  </button>
                  <button
                    type="button"
                    class="btn small"
                    title="Tokenize the rendered prompt"
                    onClick={sendToTokenize}
                  >
                    Tokenize this →
                  </button>
                </div>
              </div>
            )}
          </Show>
        </TabPanel>

        {/* ---- POST /v1/completions ---- */}
        <TabPanel value="completions">
          <form class="panel" onSubmit={submitCompletion}>
            <fieldset disabled={compBusy()}>
              <div class="panel-row">
                <span class="grow">Prompt</span>
                <div class="seg" role="group" aria-label="Prompt mode">
                  <button
                    type="button"
                    class={`seg-btn${compMode() === "text" ? " on" : ""}`}
                    title="Prompt as plain text"
                    onClick={() => setCompMode("text")}
                  >
                    Text
                  </button>
                  <button
                    type="button"
                    class={`seg-btn${compMode() === "ids" ? " on" : ""}`}
                    title="Prompt as raw token ids"
                    onClick={() => setCompMode("ids")}
                  >
                    Token ids
                  </button>
                </div>
              </div>
              <Show
                when={compMode() === "text"}
                fallback={
                  <label class="field">
                    <span class="field-label">
                      Token ids — "1234, 5678" or a JSON array
                    </span>
                    <input
                      class="pg-input mono"
                      placeholder="128000, 128001"
                      aria-label="Prompt token ids"
                      spellcheck={false}
                      value={idsText()}
                      onInput={(e) => setIdsText(e.currentTarget.value)}
                    />
                  </label>
                }
              >
                <label class="field">
                  <span class="field-label">
                    Text — encoded with the tokenizer's specials
                  </span>
                  <textarea
                    class="pg-area"
                    rows={4}
                    placeholder="Once upon a time"
                    aria-label="Prompt text"
                    spellcheck={false}
                    value={promptText()}
                    onInput={(e) => setPromptText(e.currentTarget.value)}
                  />
                </label>
              </Show>
              <div class="panel-row params">
                <span class="muted">max_tokens</span>
                <input
                  class="pg-input mono w-80"
                  type="number"
                  min={1}
                  step={1}
                  aria-label="Max tokens"
                  value={maxTokens()}
                  onInput={(e) => setMaxTokens(e.currentTarget.value)}
                />
                <span class="muted">temperature</span>
                <input
                  class="pg-input mono w-80"
                  type="number"
                  step={0.1}
                  placeholder="—"
                  aria-label="Temperature"
                  value={temperature()}
                  onInput={(e) => setTemperature(e.currentTarget.value)}
                />
                <span class="muted">stop</span>
                <input
                  class="pg-input mono grow"
                  placeholder="comma-separated"
                  aria-label="Stop sequences"
                  spellcheck={false}
                  value={stopText()}
                  onInput={(e) => setStopText(e.currentTarget.value)}
                />
                <label
                  class="pg-check muted"
                  title="Render the response token by token as it arrives"
                >
                  <input
                    type="checkbox"
                    checked={compStream()}
                    onChange={(e) => setCompStream(e.currentTarget.checked)}
                  />
                  stream
                </label>
              </div>
              <div class="panel-row">
                <span class="grow">
                  Model <span class="muted">— optional</span>
                </span>
                <input
                  class="pg-input mono w-220"
                  list="playground-models"
                  placeholder="served model id"
                  aria-label="Model"
                  spellcheck={false}
                  value={compModel()}
                  onInput={(e) => setCompModel(e.currentTarget.value)}
                />
              </div>
            </fieldset>
            {/* Outside the disabled fieldset so Cancel stays clickable. */}
            <div class="panel-row">
              <button
                type="button"
                class="btn small grow-left"
                title="Copy to clipboard"
                onClick={() =>
                  void copyText(
                    "completion-body",
                    JSON.stringify(completionBody().body, null, 2)
                  )
                }
              >
                {copied() === "completion-body" ? "copied ✓" : "Copy request JSON"}
              </button>
              <Show
                when={compBusy()}
                fallback={
                  <button
                    type="submit"
                    class="btn small primary"
                    title={
                      compStream()
                        ? "Generate a streaming completion"
                        : "Generate a completion (non-streaming)"
                    }
                  >
                    Run
                  </button>
                }
              >
                <button
                  type="button"
                  class="btn small danger"
                  title="Abort the request"
                  onClick={() => compController()?.abort()}
                >
                  Cancel
                </button>
              </Show>
            </div>
          </form>

          <Show when={compError()}>
            <p class="notice error">{compError()}</p>
          </Show>
          <Show when={compCancelled()}>
            <p class="notice muted">cancelled</p>
          </Show>

          <Show when={compBusy() && compStream()}>
            <div class="panel pg-result">
              <div class="panel-row">
                <span class="grow muted">Streaming…</span>
                <span class="num">{fmtInt(compLive().length)} chars</span>
              </div>
              <pre class="pg-output">{compLive()}</pre>
            </div>
          </Show>

          <Show when={compResult()}>
            {(out) => (
              <div class="panel pg-result">
                <div class="panel-row">
                  <span class="grow mono">{out().res.id ?? "completion"}</span>
                  <Show when={out().res.choices?.[0]?.finish_reason}>
                    <span class="badge">
                      {out().res.choices?.[0]?.finish_reason}
                    </span>
                  </Show>
                  <span class="num">{Math.round(out().ms)}ms wall</span>
                </div>
                <pre class="pg-output">{out().res.choices?.[0]?.text ?? ""}</pre>
                <div class="panel-row">
                  <span class="grow">
                    <span class="badge">
                      in {fmtInt(out().res.usage?.prompt_tokens)}
                    </span>{" "}
                    <span class="badge">
                      out {fmtInt(out().res.usage?.completion_tokens)}
                    </span>{" "}
                    <span class="badge">
                      total {fmtInt(out().res.usage?.total_tokens)}
                    </span>
                  </span>
                  <button
                    type="button"
                    class="btn small"
                    title="Copy to clipboard"
                    onClick={() =>
                      void copyText("text", out().res.choices?.[0]?.text ?? "")
                    }
                  >
                    {copied() === "text" ? "copied ✓" : "Copy text"}
                  </button>
                </div>
              </div>
            )}
          </Show>
        </TabPanel>
      </TabGroup>

      <datalist id="playground-models">
        <For each={modelIds()}>{(id) => <option value={id} />}</For>
      </datalist>
    </main>
  );
}

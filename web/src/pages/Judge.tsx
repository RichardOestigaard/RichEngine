import { createSignal, onMount, For, Show } from "solid-js";
import { useSearchParams } from "@solidjs/router";
import { Tab, TabGroup, TabList, TabPanel } from "terracotta";
import { ApiError, api, authorization, errorText, noteAuthStatus } from "../api";
import { Chart } from "../Chart";
import { useCopied } from "../clipboard";
import { fmtInt, fmtMs, pct } from "../format";
import { pushToast } from "../toast";
import "./judge.css";

/* ---- /v1/judgments + /v1/systemone payload shapes (server/judgments.py) ---- */

interface JudgmentResponse {
  id?: string;
  option_ids?: string[];
  probabilities?: number[];
  option_logits?: number[];
  input_tokens?: number;
  answer_token_ids?: number[];
  prompt_sha256?: string;
  prompt_version?: string;
  model?: { id?: string };
  readout?: string;
  probability_status?: string;
  forward_seconds?: number;
  total_seconds?: number;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
}

/* One entry of the answers map: noul carries P(true); choice and score
   carry a probabilities map and a concentration "confidence". */
interface SystemOneAnswer {
  type?: string;
  noul?: number;
  choice?: string;
  score?: number;
  legend?: Record<string, unknown>;
  probabilities?: Record<string, number>;
  confidence?: number;
}

interface SystemOneResponse {
  model?: string;
  answers?: Record<string, SystemOneAnswer>;
  usage?: { input_tokens?: number; output_tokens?: number };
}

/* A systemone 422 lists every bad field, FastAPI-style. */
interface DetailItem {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

class DetailError extends ApiError {
  details: DetailItem[];
  constructor(details: DetailItem[]) {
    super(422, String(details[0]?.msg ?? "unprocessable entity"));
    this.details = details;
  }
}

/* apiPost twin that keeps a 422's detail list for inline rendering;
   every other failure still arrives as a plain ApiError. */
async function postSystemone(
  body: unknown,
  signal?: AbortSignal
): Promise<SystemOneResponse> {
  const response = await fetch("/v1/systemone", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authorization() },
    body: JSON.stringify(body),
    signal,
  });
  const data: unknown = await response.json().catch(() => null);
  noteAuthStatus(response.status);
  if (response.ok) return data as SystemOneResponse;
  const record = (data ?? {}) as Record<string, unknown>;
  if (response.status === 422 && Array.isArray(record.detail)) {
    throw new DetailError(record.detail as DetailItem[]);
  }
  const error = record.error as Record<string, unknown> | undefined;
  const message =
    (typeof error?.message === "string" && error.message) ||
    (Array.isArray(record.detail) ? String(record.detail[0]?.msg) : "") ||
    response.statusText;
  throw new ApiError(response.status, message);
}

function isAbort(error: unknown): boolean {
  return (error as { name?: string }).name === "AbortError";
}

/* state accepts JSON or plain text: a parsed nonempty object, array or
   string wins; anything else posts the raw text. Undefined means empty. */
function parseState(text: string): unknown {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    if (typeof parsed === "string" && parsed) return parsed;
    if (parsed && typeof parsed === "object") {
      const empty = Array.isArray(parsed)
        ? parsed.length === 0
        : Object.keys(parsed).length === 0;
      if (!empty) return parsed;
    }
  } catch {
    /* Not JSON: post the raw text. */
  }
  return trimmed;
}

const fmtSecs = (value: number | null | undefined) =>
  fmtMs(typeof value === "number" ? value * 1000 : undefined);

function headline(answer: SystemOneAnswer): string {
  if (answer.type === "noul") return `P(true) ${pct(answer.noul)}`;
  if (answer.type === "choice") return answer.choice ?? "—";
  if (answer.type === "score")
    return typeof answer.score === "number" ? answer.score.toFixed(2) : "—";
  return "—";
}

/* noul answers carry no probabilities map; synthesize the true/false pair. */
function probabilitiesOf(answer: SystemOneAnswer): [string, number][] {
  if (answer.type === "noul") {
    const p = answer.noul ?? 0;
    return [
      ["true", p],
      ["false", 1 - p],
    ];
  }
  return Object.entries(answer.probabilities ?? {});
}

function describe(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

const QUESTIONS_PLACEHOLDER = `{
  "toxic": { "type": "noul", "instructions": "Is the evidence toxic?" },
  "route": { "type": "choice", "criteria": { "billing": "a billing issue", "support": "a support issue" } },
  "quality": { "type": "score", "criteria": ["poor", "ok", "great"] }
}`;

const SINGLE_STATE_EXAMPLE = `{"claim": "The server rebooted at 03:12", "evidence": "syslog shows a kernel restart at 03:12:04"}`;
const BATCH_STATE_EXAMPLE = `{"ticket": "Invoice #1042 was charged twice and the card on file was declined."}`;

const newJudgmentId = () => `j-${Date.now().toString(36)}`;

/* ---- creator guide: an LLM that edits the forms ---- */

interface ChatReply {
  choices?: { message?: { content?: string } }[];
}

interface GuideMsg {
  role: "user" | "assistant";
  text: string;
}

/* Partial form state the model may return. Unknown keys are ignored. */
interface GuidePatch {
  note?: string;
  single?: {
    id?: string;
    model?: string;
    question?: string;
    state?: unknown;
    options?: { id?: unknown; description?: unknown }[];
  };
  batch?: {
    model?: string;
    state?: unknown;
    questions?: unknown;
  };
}

const GUIDE_SYSTEM = `You configure a "judge" page that scores options against evidence.
It has two forms:

single — POST /v1/judgments: { id: string (auto, only set if asked),
  model: string (optional served model id), question: string (the criterion),
  state: string (the evidence — JSON text or plain text),
  options: [{id, description}] — 2 to 16 entries, unique short ids }

batch — POST /v1/systemone: { model: string (required served model id),
  state: string (shared evidence), questions: object keyed by question id;
  each value is one of:
    {"type":"noul","instructions":"..."}        — scores P(true)
    {"type":"choice","criteria":{id:"desc"...}} — picks one criterion
    {"type":"score","criteria":["poor",...]}    — rates on the listed scale }

The user describes what they want judged; you emit form updates.
Reply with ONLY a JSON object, no prose, no code fences:
{"note":"one short sentence to the user — what you changed, or a clarifying question",
 "single":{...fields to set...}, "batch":{...fields to set...}}
Include a form key only when you want to change that form; include a field only
when you want to change it. For batch.questions emit the full object. Never emit
field values you did not intend to set.`;

/* Pull the first JSON object out of a model reply (fences tolerated). */
function parseGuideReply(raw: string): GuidePatch | null {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const parsed: unknown = JSON.parse(raw.slice(start, end + 1));
    if (parsed && typeof parsed === "object") return parsed as GuidePatch;
  } catch {
    /* Not JSON: show the text as a plain assistant note. */
  }
  return null;
}

const asText = (value: unknown) =>
  typeof value === "string" ? value : JSON.stringify(value, null, 2);

interface OptionRow {
  id: string;
  description: string;
}

export default function Judge() {
  /* Shared: served-model ids for prefills; which form last copied. */
  const [modelIds, setModelIds] = createSignal<string[]>([]);
  const { copied, copy } = useCopied();

  /* The active tab lives in the hash — #/judge?tab=batch survives reloads. */
  const [params, setParams] = useSearchParams();
  const tab = () => (params.tab === "batch" ? "batch" : "single");
  const setTab = (id: string) => setParams({ tab: id }, { replace: true });

  /* Single judgment form — prefilled so the first submit succeeds. */
  const [jId, setJId] = createSignal(newJudgmentId());
  const [jModel, setJModel] = createSignal("");
  const [question, setQuestion] = createSignal(
    "Does the evidence support the claim?"
  );
  const [jState, setJState] = createSignal(SINGLE_STATE_EXAMPLE);
  const [optionRows, setOptionRows] = createSignal<OptionRow[]>([
    { id: "yes", description: "the evidence supports the claim" },
    { id: "no", description: "the evidence does not support the claim" },
  ]);
  const [jBusy, setJBusy] = createSignal(false);
  const [jResult, setJResult] = createSignal<JudgmentResponse | null>(null);
  const [jController, setJController] = createSignal<AbortController>();
  const [jCancelled, setJCancelled] = createSignal(false);

  /* Batch (systemone) form. */
  const [sModel, setSModel] = createSignal("");
  const [sState, setSState] = createSignal(BATCH_STATE_EXAMPLE);
  const [questionsText, setQuestionsText] = createSignal(QUESTIONS_PLACEHOLDER);
  const [sBusy, setSBusy] = createSignal(false);
  const [sDetails, setSDetails] = createSignal<DetailItem[]>([]);
  const [sResult, setSResult] = createSignal<SystemOneResponse | null>(null);
  const [sController, setSController] = createSignal<AbortController>();
  const [sCancelled, setSCancelled] = createSignal(false);

  /* Creator guide: a small LLM chat that patches either form. */
  const [guideOpen, setGuideOpen] = createSignal(true);
  const [guideLog, setGuideLog] = createSignal<GuideMsg[]>([]);
  const [guideInput, setGuideInput] = createSignal("");
  const [guideBusy, setGuideBusy] = createSignal(false);
  const [guideError, setGuideError] = createSignal<string | null>(null);

  onMount(() => {
    api<{ data?: { id?: string }[] }>("/v1/models")
      .then((res) => {
        const ids = (res?.data ?? [])
          .map((entry) => entry.id)
          .filter((id): id is string => !!id);
        setModelIds(ids);
        if (ids[0]) {
          if (!sModel()) setSModel(ids[0]);
          if (!jModel()) setJModel(ids[0]);
        }
      })
      .catch(() => {
        /* Model fields stay manual when the list cannot load. */
      });
  });

  async function copyText(which: string, body: unknown) {
    await copy(which, body);
  }

  /* ---- single judgment ---- */

  function setOption(index: number, key: keyof OptionRow, value: string) {
    setOptionRows((rows) =>
      rows.map((row, i) => (i === index ? { ...row, [key]: value } : row))
    );
  }

  /* errors is keyed by field so each bad input can be marked inline. */
  function judgmentBody(): {
    body: Record<string, unknown>;
    errors: Record<string, string>;
  } {
    const body: Record<string, unknown> = {
      id: jId().trim(),
      question: question().trim(),
      state: parseState(jState()),
      options: optionRows().map((row) => ({
        id: row.id.trim(),
        description: row.description.trim(),
      })),
    };
    const model = jModel().trim();
    if (model) body.model = model;
    const options = body.options as { id: string; description: string }[];
    const errors: Record<string, string> = {};
    if (!body.id) errors.id = "id must be nonempty";
    if (!body.question) errors.question = "question must be nonempty";
    if (body.state === undefined)
      errors.state = "state must be nonempty — JSON value or plain text";
    if (options.length < 2 || options.length > 16)
      errors.options = "options need 2–16 entries";
    else if (options.some((o) => !o.id))
      errors.options = "every option needs an id";
    else if (new Set(options.map((o) => o.id)).size !== options.length)
      errors.options = "option ids must be unique";
    else if (options.some((o) => !o.description))
      errors.options = "every option needs a description";
    return { body, errors };
  }

  const jErrors = () => judgmentBody().errors;
  const jFirstError = () => Object.values(jErrors())[0];
  const jInvalid = (field: string) =>
    jErrors()[field] ? { "aria-invalid": true } : {};

  let jResultEl: HTMLDivElement | undefined;

  async function submitJudgment(event: SubmitEvent) {
    event.preventDefault();
    if (jBusy() || jFirstError()) return;
    const { body } = judgmentBody();
    setJBusy(true);
    setJCancelled(false);
    const ctrl = new AbortController();
    setJController(ctrl);
    try {
      setJResult(
        await api<JudgmentResponse>("/v1/judgments", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: ctrl.signal,
        })
      );
      setJId(newJudgmentId());
      requestAnimationFrame(() =>
        jResultEl?.scrollIntoView({ behavior: "smooth", block: "nearest" })
      );
    } catch (e) {
      /* A deliberate cancel is a note, not an error. */
      if (isAbort(e)) setJCancelled(true);
      else pushToast("error", errorText(e));
    } finally {
      setJController(undefined);
      setJBusy(false);
    }
  }

  const judgmentRows = () => {
    const res = jResult();
    if (!res) return [];
    return (res.option_ids ?? []).map((id, i) => ({
      id,
      probability: res.probabilities?.[i],
      logit: res.option_logits?.[i],
    }));
  };

  /* ---- batch (systemone) ---- */

  /* errors is keyed by field so each bad input can be marked inline. */
  function systemoneBody(): {
    body: Record<string, unknown>;
    errors: Record<string, string>;
  } {
    const model = sModel().trim();
    const body: Record<string, unknown> = {
      model,
      state: parseState(sState()),
    };
    const errors: Record<string, string> = {};
    let questions: unknown;
    try {
      questions = JSON.parse(questionsText().trim() || "null");
    } catch {
      errors.questions = "questions must be valid JSON";
      return { body, errors };
    }
    body.questions = questions;
    if (!model) errors.model = "model is required — the served model id";
    if (body.state === undefined)
      errors.state = "state must be nonempty — JSON value or plain text";
    if (!questions || typeof questions !== "object" || Array.isArray(questions))
      errors.questions = "questions must be a JSON object keyed by question id";
    else if (!Object.keys(questions).length)
      errors.questions = "questions must be nonempty";
    else if (Object.keys(questions).length > 64)
      errors.questions = "at most 64 questions per batch";
    return { body, errors };
  }

  const sErrors = () => systemoneBody().errors;
  const sFirstError = () => Object.values(sErrors())[0];
  const sInvalid = (field: string) =>
    sErrors()[field] ? { "aria-invalid": true } : {};

  let sResultEl: HTMLDivElement | undefined;

  async function submitSystemone(event: SubmitEvent) {
    event.preventDefault();
    if (sBusy() || sFirstError()) return;
    const { body } = systemoneBody();
    setSDetails([]);
    setSBusy(true);
    setSCancelled(false);
    const ctrl = new AbortController();
    setSController(ctrl);
    try {
      setSResult(await postSystemone(body, ctrl.signal));
      requestAnimationFrame(() =>
        sResultEl?.scrollIntoView({ behavior: "smooth", block: "nearest" })
      );
    } catch (e) {
      if (isAbort(e)) setSCancelled(true);
      else if (e instanceof DetailError) setSDetails(e.details);
      else pushToast("error", errorText(e));
    } finally {
      setSController(undefined);
      setSBusy(false);
    }
  }

  const answerEntries = () => Object.entries(sResult()?.answers ?? {});

  /* ---- creator guide ---- */

  /* The guide sees the live form so follow-ups modify rather than rebuild. */
  const guideSnapshot = () => ({
    active_tab: tab(),
    single: {
      id: jId(),
      model: jModel(),
      question: question(),
      state: jState(),
      options: optionRows(),
    },
    batch: { model: sModel(), state: sState(), questions: questionsText() },
  });

  function applyGuide(patch: GuidePatch) {
    const s = patch.single;
    if (s && typeof s === "object") {
      if (typeof s.id === "string" && s.id) setJId(s.id);
      if (typeof s.model === "string") setJModel(s.model);
      if (typeof s.question === "string") setQuestion(s.question);
      if (s.state !== undefined) setJState(asText(s.state));
      if (Array.isArray(s.options)) {
        const rows = s.options
          .filter((o) => o && typeof o.id === "string" && o.id)
          .slice(0, 16)
          .map((o) => ({
            id: String(o.id),
            description: typeof o.description === "string" ? o.description : "",
          }));
        if (rows.length >= 2) setOptionRows(rows);
      }
    }
    const b = patch.batch;
    if (b && typeof b === "object") {
      if (typeof b.model === "string") setSModel(b.model);
      if (b.state !== undefined) setSState(asText(b.state));
      if (b.questions !== undefined) setQuestionsText(asText(b.questions));
    }
  }

  async function sendGuide(event: SubmitEvent) {
    event.preventDefault();
    const text = guideInput().trim();
    if (!text || guideBusy()) return;
    setGuideInput("");
    setGuideError(null);
    const history = guideLog();
    setGuideLog([...history, { role: "user", text }]);
    setGuideBusy(true);
    try {
      const res = await api<ChatReply>("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: sModel() || jModel() || modelIds()[0],
          stream: false,
          messages: [
            { role: "system", content: GUIDE_SYSTEM },
            ...history.map((m) => ({ role: m.role, content: m.text })),
            {
              role: "user",
              content: `Current form state:\n${JSON.stringify(guideSnapshot(), null, 2)}\n\nRequest: ${text}`,
            },
          ],
        }),
      });
      const raw = res?.choices?.[0]?.message?.content ?? "";
      const patch = parseGuideReply(raw);
      let note = raw.trim();
      if (patch) {
        applyGuide(patch);
        note = patch.note ?? "Form updated.";
      }
      setGuideLog((log) => [
        ...log,
        { role: "assistant", text: note || "Done." },
      ]);
    } catch (e) {
      setGuideError(errorText(e));
    } finally {
      setGuideBusy(false);
    }
  }

  return (
    <main class="page judge">
      <h1>Judge</h1>
      <p class="lede">
        Score finite options against evidence — one judgment, or a batch of
        questions over a shared state.
      </p>

      <div class="panel guide">
        <button
          type="button"
          class="guide-head"
          aria-expanded={guideOpen()}
          onClick={() => setGuideOpen(!guideOpen())}
        >
          <span class="grow">
            Creator guide{" "}
            <span class="hint">
              — describe what to judge; it fills the form below. Or edit
              manually.
            </span>
          </span>
          <span class="muted">{guideOpen() ? "▾" : "▸"}</span>
        </button>
        <Show when={guideOpen()}>
          <Show when={guideLog().length > 0 || guideBusy()}>
            <div class="guide-log">
              <For each={guideLog()}>
                {(m) => <div class={`guide-msg ${m.role}`}>{m.text}</div>}
              </For>
              <Show when={guideBusy()}>
                <div class="guide-msg assistant muted">thinking…</div>
              </Show>
            </div>
          </Show>
          <form class="guide-row" onSubmit={sendGuide}>
            <input
              class="judge-input grow"
              placeholder={
                tab() === "batch"
                  ? "e.g. add a question rating tone from rude to warm"
                  : "e.g. judge whether this refund request should be approved"
              }
              aria-label="Describe the judgment to create"
              value={guideInput()}
              onInput={(e) => setGuideInput(e.currentTarget.value)}
            />
            <button
              type="submit"
              class="btn small primary"
              disabled={guideBusy() || !guideInput().trim()}
              title="Send to the served model; the reply edits the form"
            >
              {guideBusy() ? "…" : "Ask"}
            </button>
          </form>
          <Show when={guideError()}>
            <p class="field-error">{guideError()}</p>
          </Show>
        </Show>
      </div>

      <TabGroup
        class="judge-tabs"
        horizontal
        value={tab()}
        onChange={(value) => setTab(value ?? "single")}
      >
        <TabList class="tab-list">
          <Tab class="tab" value="single" as="button" type="button">
            One judgment
          </Tab>
          <Tab class="tab" value="batch" as="button" type="button">
            Batch of questions
          </Tab>
        </TabList>

        {/* ---- POST /v1/judgments ---- */}
        <TabPanel value="single">
          <form class="panel" onSubmit={submitJudgment}>
            <fieldset disabled={jBusy()}>
              <div class="field field-split">
                <label>
                  <span class="field-label">
                    ID <span class="hint">— auto-generated, echoed in the result</span>
                  </span>
                  <input
                    class="judge-input mono"
                    aria-label="Judgment id"
                    spellcheck={false}
                    value={jId()}
                    onInput={(e) => setJId(e.currentTarget.value)}
                    {...jInvalid("id")}
                  />
                  <Show when={jErrors().id}>
                    {(msg) => <p class="field-error">{msg()}</p>}
                  </Show>
                </label>
                <label>
                  <span class="field-label">
                    Model <span class="hint">— optional, defaults to the served model</span>
                  </span>
                  <input
                    class="judge-input mono"
                    list="judge-models"
                    placeholder="served model id"
                    aria-label="Model"
                    spellcheck={false}
                    value={jModel()}
                    onInput={(e) => setJModel(e.currentTarget.value)}
                  />
                </label>
              </div>
              <label class="field">
                <span class="field-label">Question — the criterion to score against</span>
                <input
                  class="judge-input"
                  placeholder="Does the evidence support the claim?"
                  aria-label="Question"
                  spellcheck={false}
                  value={question()}
                  onInput={(e) => setQuestion(e.currentTarget.value)}
                  {...jInvalid("question")}
                />
                <Show when={jErrors().question}>
                  {(msg) => <p class="field-error">{msg()}</p>}
                </Show>
              </label>
              <label class="field">
                <span class="field-label">
                  State — the evidence the model reads (JSON or text)
                </span>
                <textarea
                  class="judge-area"
                  rows={4}
                  placeholder='{"claim": "...", "evidence": "..."} or plain text'
                  aria-label="State"
                  spellcheck={false}
                  value={jState()}
                  onInput={(e) => setJState(e.currentTarget.value)}
                  {...jInvalid("state")}
                />
                <Show when={jErrors().state}>
                  {(msg) => <p class="field-error">{msg()}</p>}
                </Show>
              </label>
              <div class="field">
                <span class="field-label">
                  Options — the answers the model scores (2–16, unique ids)
                </span>
                <Show when={jErrors().options}>
                  {(msg) => <p class="field-error">{msg()}</p>}
                </Show>
                <For each={optionRows()}>
                  {(row, i) => (
                    <div class="option-row">
                      <input
                        class="judge-input mono option-id"
                        placeholder="id"
                        aria-label={`Option ${i() + 1} id`}
                        spellcheck={false}
                        value={row.id}
                        onInput={(e) => setOption(i(), "id", e.currentTarget.value)}
                      />
                      <input
                        class="judge-input grow"
                        placeholder="description"
                        aria-label={`Option ${i() + 1} description`}
                        value={row.description}
                        onInput={(e) =>
                          setOption(i(), "description", e.currentTarget.value)
                        }
                      />
                      <button
                        type="button"
                        class="btn small"
                        title={
                          optionRows().length <= 2
                            ? "At least 2 options are required"
                            : "Remove option"
                        }
                        aria-label={`Remove option ${i() + 1}`}
                        disabled={optionRows().length <= 2}
                        onClick={() =>
                          setOptionRows((rows) => rows.filter((_, j) => j !== i()))
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
                  title={
                    optionRows().length >= 16
                      ? "At most 16 options"
                      : "Append an option (2–16)"
                  }
                  disabled={optionRows().length >= 16}
                  onClick={() =>
                    setOptionRows((rows) => [
                      ...rows,
                      { id: `opt${rows.length + 1}`, description: "" },
                    ])
                  }
                >
                  Add option
                </button>
              </div>
            </fieldset>
            {/* Outside the disabled fieldset so Cancel stays clickable. */}
            <div class="panel-row">
              <button
                type="button"
                class="btn small grow-left"
                title="Copy to clipboard"
                onClick={() => void copyText("single", judgmentBody().body)}
              >
                {copied() === "single" ? "copied ✓" : "Copy request JSON"}
              </button>
              <Show
                when={jBusy()}
                fallback={
                  <button
                    type="submit"
                    class="btn small primary"
                    disabled={!!jFirstError()}
                    title={
                      jFirstError() ??
                      "Submit the judgment — the model scores every option"
                    }
                  >
                    Judge
                  </button>
                }
              >
                <button
                  type="button"
                  class="btn small danger"
                  title="Abort the request"
                  onClick={() => jController()?.abort()}
                >
                  Cancel
                </button>
              </Show>
            </div>
          </form>

          <Show when={jCancelled()}>
            <p class="notice muted">cancelled</p>
          </Show>

          <Show when={jResult()}>
            {(res) => (
              <div class="panel judge-result flash" ref={jResultEl}>
                <div class="panel-row">
                  <span class="grow">
                    Result <span class="mono">{res().id ?? ""}</span>
                  </span>
                  <Show when={res().probability_status}>
                    <span class="badge" title={res().probability_status}>
                      {res().prompt_version ?? "scored"}
                    </span>
                  </Show>
                  <button
                    type="button"
                    class="btn small"
                    title="Copy to clipboard"
                    onClick={() => void copyText("j-result", res())}
                  >
                    {copied() === "j-result" ? "copied ✓" : "Copy JSON"}
                  </button>
                </div>
                <div class="panel-row chart-row">
                  <div>
                    <Chart
                      type="bar"
                      height={170}
                      series={[
                        { name: "p", data: res().probabilities ?? [] },
                      ]}
                      options={{
                        xaxis: { categories: res().option_ids ?? [] },
                        yaxis: { min: 0, max: 1, tickAmount: 5 },
                        tooltip: {
                          y: { formatter: (v: number) => pct(v) },
                        },
                      }}
                    />
                  </div>
                </div>
                <For each={judgmentRows()}>
                  {(row) => (
                    <>
                      <div class="panel-row">
                        <span class="grow mono">{row.id}</span>
                        <span class="num">
                          {typeof row.logit === "number"
                            ? `logit ${row.logit.toFixed(3)}`
                            : ""}
                        </span>
                        <span class="num">{pct(row.probability)}</span>
                      </div>
                      <div class="panel-row meter-row">
                        <div class="meter" title="Probability">
                          <i
                            style={{
                              width: `${Math.max(
                                0,
                                Math.min(1, row.probability ?? 0)
                              ) * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    </>
                  )}
                </For>
                <div class="panel-row">
                  <span class="grow muted meta-line">
                    {res().model?.id ?? "—"} · {fmtInt(res().input_tokens)} tokens
                    in · forward {fmtSecs(res().forward_seconds)} · total{" "}
                    {fmtSecs(res().total_seconds)}
                    <Show when={res().prompt_sha256}>
                      {(sha) => <> · sha <span class="mono">{sha().slice(0, 12)}</span></>}
                    </Show>
                  </span>
                </div>
              </div>
            )}
          </Show>
        </TabPanel>

        {/* ---- POST /v1/systemone ---- */}
        <TabPanel value="batch">
          <form class="panel" onSubmit={submitSystemone}>
            <fieldset disabled={sBusy()}>
              <label class="field">
                <span class="field-label">
                  Model <span class="hint">— the served model id, required</span>
                </span>
                <input
                  class="judge-input mono"
                  list="judge-models"
                  placeholder="served model id"
                  aria-label="Model"
                  spellcheck={false}
                  value={sModel()}
                  onInput={(e) => setSModel(e.currentTarget.value)}
                  {...sInvalid("model")}
                />
                <Show when={sErrors().model}>
                  {(msg) => <p class="field-error">{msg()}</p>}
                </Show>
              </label>
              <label class="field">
                <span class="field-label">
                  State — the evidence shared by every question (JSON or text)
                </span>
                <textarea
                  class="judge-area"
                  rows={4}
                  placeholder='{"ticket": "..."} or plain text'
                  aria-label="State"
                  spellcheck={false}
                  value={sState()}
                  onInput={(e) => setSState(e.currentTarget.value)}
                  {...sInvalid("state")}
                />
                <Show when={sErrors().state}>
                  {(msg) => <p class="field-error">{msg()}</p>}
                </Show>
              </label>
              <div class="field">
                <span class="field-label">
                  Questions — JSON object keyed by question id, ≤64
                </span>
                <p class="hint">
                  types: <code>"noul"</code> scores true/false,{" "}
                  <code>"choice"</code> picks from criteria, <code>"score"</code>{" "}
                  rates on a scale
                </p>
                <textarea
                  class="judge-area tall"
                  rows={9}
                  placeholder={QUESTIONS_PLACEHOLDER}
                  aria-label="Questions"
                  spellcheck={false}
                  value={questionsText()}
                  onInput={(e) => setQuestionsText(e.currentTarget.value)}
                  {...sInvalid("questions")}
                />
                <Show when={sErrors().questions}>
                  {(msg) => <p class="field-error">{msg()}</p>}
                </Show>
                <button
                  type="button"
                  class="btn small"
                  title="Restore the example questions"
                  onClick={() => setQuestionsText(QUESTIONS_PLACEHOLDER)}
                >
                  Insert example
                </button>
              </div>
            </fieldset>
            {/* Outside the disabled fieldset so Cancel stays clickable. */}
            <div class="panel-row">
              <button
                type="button"
                class="btn small grow-left"
                title="Copy to clipboard"
                onClick={() => void copyText("batch", systemoneBody().body)}
              >
                {copied() === "batch" ? "copied ✓" : "Copy request JSON"}
              </button>
              <Show
                when={sBusy()}
                fallback={
                  <button
                    type="submit"
                    class="btn small primary"
                    disabled={!!sFirstError()}
                    title={
                      sFirstError() ??
                      "Submit the batch — the model answers every question"
                    }
                  >
                    Judge all
                  </button>
                }
              >
                <button
                  type="button"
                  class="btn small danger"
                  title="Abort the request"
                  onClick={() => sController()?.abort()}
                >
                  Cancel
                </button>
              </Show>
            </div>
          </form>

          <Show when={sCancelled()}>
            <p class="notice muted">cancelled</p>
          </Show>
          <Show when={sDetails().length > 0}>
            <div class="notice error">
              <For each={sDetails()}>
                {(d) => (
                  <div>
                    <span class="mono">{(d.loc ?? []).join(".") || "body"}</span>
                    : {d.msg ?? "invalid"}
                  </div>
                )}
              </For>
            </div>
          </Show>

          <Show when={sResult()}>
            {(res) => (
              <>
                <div class="panel judge-result flash" ref={sResultEl}>
                  <div class="panel-row">
                    <span class="grow">
                      {fmtInt(answerEntries().length)} answers ·{" "}
                      <span class="mono">{res().model ?? "—"}</span>
                    </span>
                    <span class="num">
                      {fmtInt(res().usage?.input_tokens)} input tokens
                    </span>
                    <button
                      type="button"
                      class="btn small"
                      title="Copy to clipboard"
                      onClick={() => void copyText("s-result", res())}
                    >
                      {copied() === "s-result" ? "copied ✓" : "Copy JSON"}
                    </button>
                  </div>
                </div>
                <For each={answerEntries()}>
                  {([qid, answer]) => (
                    <div class="panel answer-panel">
                      <div class="panel-row">
                        <span class="grow mono">{qid}</span>
                        <span class="badge">{answer.type ?? "?"}</span>
                        <span class="num answer-head">{headline(answer)}</span>
                        <Show when={typeof answer.confidence === "number"}>
                          <span
                            class="badge"
                            title="Normalized-entropy concentration — not calibrated confidence"
                          >
                            conf {pct(answer.confidence)}
                          </span>
                        </Show>
                      </div>
                      <For each={probabilitiesOf(answer)}>
                        {([label, p]) => (
                          <div class="panel-row mini">
                            <span class="mini-label mono">{label}</span>
                            <div class="meter" title="Probability">
                              <i
                                style={{
                                  width: `${Math.max(0, Math.min(1, p)) * 100}%`,
                                }}
                              />
                            </div>
                            <span class="num">{pct(p)}</span>
                          </div>
                        )}
                      </For>
                      <Show when={answer.legend}>
                        <For each={Object.entries(answer.legend ?? {})}>
                          {([idx, desc]) => (
                            <div class="panel-row mini legend">
                              <span class="mini-label mono">{idx}</span>
                              <span class="grow muted">{describe(desc)}</span>
                            </div>
                          )}
                        </For>
                      </Show>
                    </div>
                  )}
                </For>
              </>
            )}
          </Show>
        </TabPanel>
      </TabGroup>

      <datalist id="judge-models">
        <For each={modelIds()}>{(id) => <option value={id} />}</For>
      </datalist>
    </main>
  );
}

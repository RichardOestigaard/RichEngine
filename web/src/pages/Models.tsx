import { createSignal, createEffect, onCleanup, onMount, For, Show } from "solid-js";
import type { JSX } from "solid-js";
import { A } from "@solidjs/router";
import { api, apiPost, authorization, curlExample, errorText, storageGet, storageSet } from "../api";
import { Chart } from "../Chart";
import { fmtBytes, fmtInt, fmtMs, fmtTokens, pct, rate } from "../format";
import { installJob, onInstallDone, pollInstall, setInstallJob } from "../install";
import type { InstallJob } from "../install";
import { onTuneDone, pollTune, setTuneJob, tuneJob } from "../tune";
import type { TuneJob } from "../tune";
import { useCopied } from "../clipboard";
import { useFreshness } from "../fresh";
import { poll } from "../poll";
import { pushToast } from "../toast";
import Confirm from "../Confirm";
import { MODEL_KEY } from "./chat/storage";
import "./models.css";

/* ---- /status payload shapes (runtime/engine/wire/Status.cpp + server/frontend.py) ---- */

interface Batch {
  valid?: boolean;
  width?: number;
  input_tokens?: number;
  output_tokens?: number;
  drafted_tokens?: number;
  accepted_draft_tokens?: number;
  wall_ms?: number;
  tokens_per_second?: number;
}

interface CacheStats {
  entries?: number;
  capacity?: number;
  bytes?: number;
  budget_bytes?: number;
  source_bytes?: number;
  source_budget_bytes?: number;
  request_bytes?: number;
  request_budget_bytes?: number;
  hits?: number;
  misses?: number;
  evictions?: number;
  reused_tokens?: number;
  enabled?: boolean;
}

interface LatencyStage {
  buckets?: Record<string, number>;
  count?: number;
  sum?: number;
}

interface Status {
  schema_version?: number;
  /* The host's lifecycle: loaded | loading | unloaded | failed. */
  model_state?: string;
  /* Load failure detail, present when model_state is "failed". */
  model_error?: string;
  ready?: boolean;
  maximum_context_tokens?: number;
  memory_pressure?: string;
  vision?: boolean;
  input_modalities?: string[];
  metal?: { healthy?: boolean; failure_reason?: string };
  memory_plan?: {
    valid?: boolean;
    maximum_context_tokens?: number;
    budget?: { hard_budget_bytes?: number; kv_capacity_tokens?: number };
  };
  memory_actual?: {
    allocated_bytes?: number;
    current_bytes?: number;
    peak_bytes?: number;
  };
  memory_governor?: {
    limit_bytes?: number;
    charged_bytes?: number;
    headroom_bytes?: number;
    growth_allowed?: boolean;
    denied_reservations?: number;
    system_pressure?: string;
    host_available_bytes?: number;
    host_reserve_bytes?: number;
    host_headroom_bytes?: number;
  };
  kv?: {
    block_tokens?: number;
    pages_allocated?: number;
    pages_active?: number;
    pages_cache?: number;
    pages_free?: number;
    allocated_bytes?: number;
    reclaimable_bytes?: number;
  };
  state?: {
    entries?: number;
    pinned?: number;
    in_use?: number;
    bytes?: number;
    disk_hits?: number;
    disk_promotions?: number;
    offloads?: number;
  };
  disk?: {
    capacity_bytes?: number;
    used_bytes?: number;
    kv_blocks?: number;
    kv_bytes?: number;
    kv_demotions?: number;
    kv_restores?: number;
    kv_pending_pages?: number;
    persistent?: boolean;
    write_behind?: { waiting?: number; durable?: number; refused?: number };
  };
  cache?: {
    hits?: number;
    cold_misses?: number;
    hit_rate?: number;
    kv_hit_tokens?: number;
    kv_disk_hit_tokens?: number;
    reused_tokens?: number;
  };
  scheduler?: {
    queued?: number;
    waiting_resources?: number;
    waiting_prefix?: number;
    prefilling?: number;
    decoding?: number;
    waiting_mask?: number;
    terminal?: number;
    prefill_batches?: number;
    decode_batches?: number;
    decode_batches_by_width?: { b1?: number; b2?: number; b3?: number; b4?: number };
  };
  requests?: { submitted?: number; completed?: number; cancelled?: number; failed?: number };
  metrics?: {
    ttft_ms?: { p50?: number; p95?: number; samples?: number };
    itl_ms?: { p50?: number; p95?: number; samples?: number };
    prefill_tokens_per_second?: number;
    decode_tokens_per_second?: number;
    drafted_tokens?: number;
    accepted_draft_tokens?: number;
    draft_acceptance_rate?: number;
    current_prefill_batch?: Batch;
    current_decode_batch?: Batch;
  };
  warmup?: Record<string, unknown>;
  ane_ffn?: { state?: string; share?: number; reruns?: number; reason?: string };
  model_timing?: {
    prefill?: { last_gpu_ms?: number; last_wall_ms?: number };
    decode?: { last_gpu_ms?: number; last_wall_ms?: number };
  };
  identity?: {
    cache?: { build_id?: string; dtype?: string; block_tokens?: number };
    kv?: { format?: string; quantization?: string; scale_type?: string };
  };
  frontend?: { preparation_capacity?: number; active?: number; waiting?: number };
  grammar_cache?: CacheStats;
  response_store?: CacheStats;
  image_cache?: CacheStats;
  tokenizer_cache?: CacheStats;
  latency?: Record<string, LatencyStage | undefined>;
}

interface ModelEntry {
  id?: string;
  object?: string;
  created?: number;
  owned_by?: string;
  max_model_len?: number;
  context_length?: number;
  vision?: boolean;
  input_modalities?: string[];
  root?: string;
}

interface ModelsResponse {
  object?: string;
  data?: ModelEntry[];
}

/* One /v1/models/available row: an installed selection the load endpoint
   can serve. */
interface AvailableModel {
  model?: string;
  family?: string | null;
  target_format?: string | null;
  vision_format?: string | null;
  bytes?: number | null;
  serving?: boolean;
  /* The link's name under the models root — ".selections/<hash>" for
     installs made with options; POST /v1/models/tune addresses by it. */
  selection?: string;
  /* Auto Tune state on this chip: "untuned" | "partial" | "tuned" |
     "stale", and the knob count the record applies. */
  tuned_knobs?: number | null;
  tuning?: string | null;
  /* The finished sweep's mode ("quick" | "complete") and a count of each
     candidate verdict — kept, rejected, eliminated, pruned, pressured. */
  tune_mode?: string | null;
  tune_verdicts?: Record<string, number> | null;
  /* A tune finished after this model's engine spawned; its winners apply
     only on the next load. */
  requires_reload?: boolean;
  /* False when the family's knob table is unknown — Tune would only fail. */
  tunable?: boolean;
}

/* ---- /v1/models/available + /v1/models/install payload shapes ---- */

interface AvailableResponse {
  object?: string;
  data?: AvailableModel[];
  suggested?: string[];
}

/* ---- small presentational pieces ---- */

function Row(props: { label: string; value?: string; sub?: string }) {
  return (
    <div class="panel-row">
      <span class="grow">{props.label}</span>
      <Show when={props.sub}>
        <span class="num">{props.sub}</span>
      </Show>
      <span class="num">{props.value ?? "—"}</span>
    </div>
  );
}

function MeterRow(props: { label: string; value?: number; max?: number; text: string }) {
  const width = () => {
    const v = props.value ?? 0;
    const m = props.max ?? 0;
    return m > 0 ? Math.min(100, (v / m) * 100) : 0;
  };
  return (
    <>
      <div class="panel-row">
        <span class="grow">{props.label}</span>
        <span class="num">{props.text}</span>
      </div>
      <div class="panel-row meter-row">
        <div
          class="meter"
          role="meter"
          aria-label={props.label}
          aria-valuemin={0}
          aria-valuemax={props.max ?? 0}
          aria-valuenow={props.value ?? 0}
          aria-valuetext={props.text}
        >
          <i style={{ width: `${width()}%` }} />
        </div>
      </div>
    </>
  );
}

function Badge(props: { kind: "ok" | "warn" | "err"; children: JSX.Element }) {
  return <span class={`badge ${props.kind}`}>{props.children}</span>;
}

/* Histogram bucket keys are range labels; order numerically when they
   parse, alphabetically otherwise. */
function bucketOrder(a: string, b: string): number {
  const na = parseFloat(a);
  const nb = parseFloat(b);
  if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
  return a < b ? -1 : a > b ? 1 : 0;
}

/* Collapsible sections remember their open state per visit via
   localStorage; data-name is the key suffix. */
const COLLAPSE_KEY = "richengine:models-collapse:";

function collapseOpen(name: string, fallback = true): boolean {
  const stored = storageGet(COLLAPSE_KEY + name);
  return stored === null ? fallback : stored === "1";
}

function collapseToggle(event: Event) {
  const details = event.currentTarget as HTMLDetailsElement | null;
  const summary = details?.querySelector("summary");
  if (!details || !summary) return;
  summary.title = details.open ? "Collapse section" : "Expand section";
  if (details.dataset.name)
    storageSet(COLLAPSE_KEY + details.dataset.name, details.open ? "1" : "0");
}

/* Telemetry history lives at module scope: navigating away and back keeps
   the collected points instead of restarting every chart empty (same
   pattern as the Metrics page's seriesHistory). */
const [history, setHistory] = createSignal<{
  decode: number[];
  prefill: number[];
  memory: number[];
}>({ decode: [], prefill: [], memory: [] });

/* ---- page ---- */

export default function Models() {
  const [status, setStatus] = createSignal<Status | null>(null);
  const [models, setModels] = createSignal<ModelsResponse | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [modelsError, setModelsError] = createSignal<string | null>(null);
  const [paused, setPaused] = createSignal(false);
  const [metricsText, setMetricsText] = createSignal<string | null>(null);
  const [available, setAvailable] = createSignal<AvailableModel[] | null>(null);
  const [availableError, setAvailableError] = createSignal<string | null>(null);
  const [suggested, setSuggested] = createSignal<string[]>([]);
  const [modelBusy, setModelBusy] = createSignal(false);
  const [confirmUnload, setConfirmUnload] = createSignal<AvailableModel | null>(null);
  const [confirmTune, setConfirmTune] = createSignal<AvailableModel | null>(null);
  const [tuneBusy, setTuneBusy] = createSignal(false);
  /* Heartbeat for the silent gaps between sweep log lines: each candidate
     is an engine launch, so minutes of quiet are normal — the counter
     shows the sweep is alive. */
  const [tick, setTick] = createSignal(0);
  const [lastTailAt, setLastTailAt] = createSignal(Date.now());
  const [installBusy, setInstallBusy] = createSignal(false);
  const [installModel, setInstallModel] = createSignal("");
  const [installRevision, setInstallRevision] = createSignal("");
  const [installDraft, setInstallDraft] = createSignal("");
  const [installLanguageOnly, setInstallLanguageOnly] = createSignal(false);
  const { markOk, updatedAgo, stale } = useFreshness();
  const { copied, copy } = useCopied();
  let logEl: HTMLPreElement | undefined;
  let tuneLogEl: HTMLPreElement | undefined;
  /* Follow the install tail only while the reader stays at the bottom. */
  let logPinned = true;
  let tuneLogPinned = true;

  async function pollStatus() {
    try {
      const next = await api<Status>("/status");
      setStatus(next);
      setError(null);
      markOk();
      setHistory((h) => ({
        decode: [...h.decode, next.metrics?.decode_tokens_per_second ?? 0].slice(-60),
        prefill: [...h.prefill, next.metrics?.prefill_tokens_per_second ?? 0].slice(-60),
        memory: [...h.memory, next.memory_actual?.current_bytes ?? 0].slice(-60),
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function loadMetrics() {
    try {
      const response = await fetch("/metrics", { headers: authorization() });
      setMetricsText(await response.text());
    } catch (e) {
      setMetricsText(e instanceof Error ? e.message : String(e));
    }
  }

  function loadModelsList() {
    api<ModelsResponse>("/v1/models")
      .then(setModels)
      .catch((e) => setModelsError(e instanceof Error ? e.message : String(e)));
  }

  async function copyCurl() {
    await copy("curl", curlExample(model()?.id ?? "MODEL_ID"));
  }

  function loadAvailable() {
    api<AvailableResponse>("/v1/models/available")
      .then((res) => {
        setAvailable(res?.data ?? []);
        setSuggested(res?.suggested ?? []);
        setAvailableError(null);
      })
      .catch((e) => setAvailableError(errorText(e)));
  }

  async function loadModel(model: string) {
    if (modelBusy()) return;
    setModelBusy(true);
    try {
      await apiPost("/v1/models/load", { model });
      pushToast("ok", `Loaded ${model}`);
      // Point the chat composer at the newly served id.
      storageSet(MODEL_KEY, model);
      window.dispatchEvent(
        new CustomEvent("richengine:model-changed", { detail: model })
      );
    } catch (e) {
      pushToast("error", errorText(e));
    }
    setModelBusy(false);
    loadModelsList();
    loadAvailable();
    void pollStatus();
  }

  async function unloadModel() {
    if (modelBusy()) return;
    setConfirmUnload(null);
    setModelBusy(true);
    try {
      await apiPost("/v1/models/unload", {});
      pushToast("ok", "Model unloaded");
    } catch (e) {
      pushToast("error", errorText(e));
    }
    setModelBusy(false);
    loadModelsList();
    loadAvailable();
    void pollStatus();
  }

  /* Install progress polling and its toasts live in ../install — the job
     runs as a server-side subprocess and outlives this page. */
  async function submitInstall(event: SubmitEvent) {
    event.preventDefault();
    const model = installModel().trim();
    if (!model || installBusy() || installJob()?.running) return;
    setInstallBusy(true);
    try {
      const body: Record<string, unknown> = {
        model,
        language_only: installLanguageOnly(),
      };
      const revision = installRevision().trim();
      const draft = installDraft().trim();
      if (revision) body.revision = revision;
      if (draft) body.draft_model = draft;
      logPinned = true;
      setInstallJob(await apiPost<InstallJob>("/v1/models/install", body));
    } catch (e) {
      pushToast("error", errorText(e));
    } finally {
      setInstallBusy(false);
    }
  }

  async function cancelInstall() {
    try {
      await apiPost("/v1/models/install/cancel", {});
      void pollInstall();
    } catch (e) {
      pushToast("error", errorText(e));
    }
  }

  /* Auto Tune: a server-side sweep job, polled in ../tune. The row's Tune
     button opens a dialog that picks the mode — and warns when a model is
     loaded, since its engine shares the GPU and contaminates every
     measurement; the choice the user confirms carries allow_loaded. */
  async function startTune(entry: AvailableModel, mode: "quick" | "complete") {
    const model = entry.model ?? "";
    if (!model || tuneBusy() || tuneJob()?.running) return;
    const state = s().model_state;
    setConfirmTune(null);
    setTuneBusy(true);
    try {
      tuneLogPinned = true;
      setLastTailAt(Date.now());
      const body: Record<string, unknown> = {
        model,
        mode,
        allow_loaded: state === "loaded" || state === "loading",
      };
      /* Optioned installs (.selections/<hash> links) aren't reachable by
         model id — the server resolves the link directly. */
      if (entry.selection) body.selection = entry.selection;
      setTuneJob(await apiPost<TuneJob>("/v1/models/tune", body));
    } catch (e) {
      pushToast("error", errorText(e));
    } finally {
      setTuneBusy(false);
    }
  }

  async function cancelTune() {
    try {
      await apiPost("/v1/models/tune/cancel", {});
      void pollTune();
    } catch (e) {
      pushToast("error", errorText(e));
    }
  }

  const tuneElapsed = () => {
    const started = tuneJob()?.started;
    if (!started) return "";
    const seconds = Math.max(0, Math.round(Date.now() / 1000 - started));
    return seconds < 60
      ? `${seconds}s`
      : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };

  /* Seconds since the sweep last printed — the proof of life between the
     minutes-long engine launches. */
  const tailQuiet = () => {
    tick();
    return Math.max(0, Math.round((Date.now() - lastTailAt()) / 1000));
  };

  /* The sweep's phase as a readable label, with the phase-scoped
     candidate count when the workload is known. */
  const tunePhase = () => {
    const job = tuneJob();
    const total = job?.phase_total ?? 0;
    const done = job?.phase_done ?? 0;
    const counted = total ? ` — ${done}/${total}` : "";
    switch (job?.phase) {
      case "baseline":
        return "measuring the baseline";
      case "sweep":
        return `sweep${counted}`;
      case "recheck":
        return `recheck${counted}`;
      case "prune":
        return `pruning winners${counted}`;
      default:
        return "starting the engine";
    }
  };

  const installElapsed = () => {
    const started = installJob()?.started;
    if (!started) return "";
    const seconds = Math.max(0, Math.round(Date.now() / 1000 - started));
    return seconds < 60
      ? `${seconds}s`
      : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };

  /* Keep the log pinned to its newest line until the reader scrolls up. */
  createEffect(() => {
    installJob()?.tail?.join("\n");
    if (logPinned && logEl) logEl.scrollTop = logEl.scrollHeight;
  });
  createEffect(() => {
    tuneJob()?.tail?.join("\n");
    if (tuneLogPinned && tuneLogEl) tuneLogEl.scrollTop = tuneLogEl.scrollHeight;
  });
  /* The quiet-gap counter: lastTailAt moves whenever the tail grows. */
  createEffect(() => {
    if (tuneJob()?.tail?.length) setLastTailAt(Date.now());
  });

  onMount(() => {
    loadModelsList();
    loadAvailable();
    void pollInstall();
    void pollTune();
    /* Drives the tune panel's quiet-gap counter. */
    const ticker = setInterval(() => setTick((t) => t + 1), 1000);
    onCleanup(() => clearInterval(ticker));
    /* A persisted-open raw section skips the lazy-load toggle. */
    if (collapseOpen("raw-metrics", false)) void loadMetrics();
  });
  /* Refresh the installed list when a module-polled install or tune
     finishes — either changes a row's state. */
  onCleanup(onInstallDone(loadAvailable));
  onCleanup(onTuneDone(loadAvailable));
  onCleanup(poll(() => (paused() ? Promise.resolve() : pollStatus()), 2000));

  const s = () => status() ?? {};
  const model = () => models()?.data?.[0];
  const aliasCount = () => Math.max(0, (models()?.data?.length ?? 0) - 1);
  const aliases = () =>
    (models()?.data ?? [])
      .slice(1)
      .map((m) => m.id)
      .filter(Boolean)
      .join(", ");
  const hardBudget = () =>
    s().memory_plan?.budget?.hard_budget_bytes ?? s().memory_governor?.limit_bytes;
  const pressureKind = (): "ok" | "warn" | "err" => {
    const p = s().memory_pressure;
    return p === "normal" ? "ok" : p === "warning" ? "warn" : "err";
  };

  const warmupSteps = () => {
    const w = s().warmup;
    if (!w) return [];
    const rank = (key: string) => {
      if (key.startsWith("prefill")) return 0;
      const order = ["decode_b1", "decode_b2", "decode_b3", "decode_b4", "composite_state_restore"];
      const i = order.indexOf(key);
      return i === -1 ? order.length + 1 : i + 1;
    };
    return Object.keys(w)
      .filter((k) => k !== "memory_limited_steps" && k !== "detail" && typeof w[k] === "boolean")
      .sort((a, b) => rank(a) - rank(b));
  };
  const warmupLimited = () =>
    (s().warmup?.["memory_limited_steps"] as string[] | undefined) ?? [];

  const latencyChips = () => {
    const l = s().latency;
    if (!l) return [];
    return Object.entries(l)
      .filter((entry): entry is [string, LatencyStage] => !!entry[1] && (entry[1].count ?? 0) > 0)
      .map(([name, v]) => ({
        name,
        count: v.count ?? 0,
        avgMs: (v.count ?? 0) > 0 ? ((v.sum ?? 0) / (v.count ?? 1)) * 1000 : 0,
      }));
  };

  function cacheValue(c: CacheStats | undefined): string {
    if (!c) return "—";
    const parts: string[] = [];
    if (c.entries !== undefined)
      parts.push(`${fmtInt(c.entries)}${c.capacity !== undefined ? `/${fmtInt(c.capacity)}` : ""} entries`);
    const bytes = c.bytes ?? c.source_bytes;
    const budget = c.budget_bytes ?? c.source_budget_bytes;
    if (bytes !== undefined)
      parts.push(`${fmtBytes(bytes)}${budget !== undefined ? `/${fmtBytes(budget)}` : ""}`);
    if (c.hits !== undefined)
      parts.push(`${fmtInt(c.hits)} hits${c.misses !== undefined ? ` · ${fmtInt(c.misses)} misses` : ""}`);
    if (c.evictions !== undefined) parts.push(`${fmtInt(c.evictions)} evictions`);
    if (c.reused_tokens !== undefined) parts.push(`${fmtInt(c.reused_tokens)} reused tokens`);
    return parts.length ? parts.join(" · ") : "—";
  }

  const buildSub = () => {
    const id = s().identity?.cache?.build_id;
    return id ? `build ${id.slice(0, 12)}` : undefined;
  };

  /* Stages that carry a histogram, for the small per-stage bars. */
  const latencyStages = () => {
    const l = s().latency;
    if (!l) return [];
    return Object.entries(l)
      .filter(
        (entry): entry is [string, LatencyStage] =>
          !!entry[1] && Object.keys(entry[1].buckets ?? {}).length > 0
      )
      .map(([name, v]) => {
        const cats = Object.keys(v.buckets ?? {}).sort(bucketOrder);
        return { name, cats, data: cats.map((c) => v.buckets?.[c] ?? 0) };
      });
  };

  /* Load/unload controls freeze while a load is already in flight. */
  const modelWorking = () => modelBusy() || s().model_state === "loading";


  const widthRow = () => {
    const w = s().scheduler?.decode_batches_by_width;
    if (!w) return "—";
    return `b1 ${fmtInt(w.b1)} · b2 ${fmtInt(w.b2)} · b3 ${fmtInt(w.b3)} · b4 ${fmtInt(w.b4)}`;
  };

  return (
    <main class="page">
      <div class="page-head">
        <div>
          <h1>Models</h1>
          <p class="lede">Installed model and engine telemetry.</p>
        </div>
        <div class="head-side">
          <Show when={updatedAgo()}>
            <span
              class={`freshness${stale() ? " stale" : ""}`}
              title="Time since the last successful /status fetch"
            >
              updated {updatedAgo()} ago
            </span>
          </Show>
          <button
            class="btn small"
            title="Copy an OpenAI-compatible curl request for this server"
            onClick={() => void copyCurl()}
          >
            {copied() === "curl" ? "copied ✓" : "Copy curl"}
          </button>
          <button
            class="btn small"
            onClick={() => setPaused(!paused())}
            title={paused() ? "Resume the 2s telemetry poll" : "Pause the 2s telemetry poll"}
          >
            {paused() ? "Resume" : "Pause"}
          </button>
        </div>
      </div>

      <Show when={error()}>
        <p class="notice error">
          Status fetch failed{status() ? " — showing last good data" : ""}: {error()}
        </p>
      </Show>
      <Show when={s().model_state === "failed"}>
        <p class="notice error">
          Model load failed{s().model_error ? `: ${s().model_error}` : ""}
        </p>
      </Show>
      <Show when={modelsError()}>
        <p class="notice error">Model list fetch failed: {modelsError()}</p>
      </Show>

      {/* 1. Overview */}
      <div class="cards">
        <div class="card">
          <div class="card-label">Model</div>
          <div class="card-value model-id">
            <Show
              when={model()?.id}
              fallback={
                <Show when={s().model_state === "unloaded"} fallback="—">
                  <span class="muted">none loaded</span>
                </Show>
              }
            >
              {(id) => (
                <button
                  class="copy-btn"
                  title="Copy"
                  onClick={() => void copy(id(), id())}
                >
                  {copied() === id() ? "copied ✓" : id()}
                </button>
              )}
            </Show>
          </div>
          <Show when={aliasCount() > 0}>
            <div class="card-sub" title={aliases()}>
              +{aliasCount()} names
            </div>
          </Show>
        </div>
        <div class="card">
          <div class="card-label">Context</div>
          <div class="card-value">{fmtTokens(model()?.context_length ?? s().maximum_context_tokens)}</div>
          <div class="card-sub">max context length</div>
        </div>
        <div class="card">
          <div class="card-label">Modalities</div>
          <div class="card-value modalities">
            <Show
              when={(model()?.input_modalities ?? s().input_modalities ?? []).length > 0}
              fallback="—"
            >
              <For each={model()?.input_modalities ?? s().input_modalities ?? []}>
                {(m) => <span class="badge">{m}</span>}
              </For>
            </Show>
          </div>
        </div>
        <div class="card">
          <div class="card-label">Engine</div>
          <div class="card-value">
            <Show when={status()} fallback="—">
              <Show
                when={s().model_state !== "loading" && s().model_state !== "failed"}
                fallback={
                  <Badge kind={s().model_state === "failed" ? "err" : "warn"}>
                    {s().model_state}
                  </Badge>
                }
              >
                <Show when={s().ready} fallback={<Badge kind="warn">not ready</Badge>}>
                  <Badge kind="ok">ready</Badge>
                </Show>
              </Show>
            </Show>
          </div>
          <div class="card-sub">schema v{fmtInt(s().schema_version)}</div>
        </div>
        <div class="card">
          <div class="card-label">Metal</div>
          <div class="card-value">
            <Show when={s().metal} fallback="—">
              <Show when={s().metal?.healthy} fallback={<Badge kind="err">unhealthy</Badge>}>
                <Badge kind="ok">healthy</Badge>
              </Show>
            </Show>
          </div>
          <Show when={s().metal && !s().metal?.healthy && s().metal?.failure_reason}>
            <div class="card-sub">{s().metal?.failure_reason}</div>
          </Show>
        </div>
        <div class="card">
          <div class="card-label">Memory pressure</div>
          <div class="card-value">
            <Show when={s().memory_pressure} fallback="—">
              <Badge kind={pressureKind()}>{s().memory_pressure}</Badge>
            </Show>
          </div>
          <div class="card-sub">
            host available {fmtBytes(s().memory_governor?.host_available_bytes)}
          </div>
        </div>
      </div>

      {/* 1b. Installed models */}
      <section class="section">
        <h2>
          Installed models
          <A href="/disk" class="section-link" title="Where the bytes live on disk">
            Disk →
          </A>
        </h2>
        <Show when={availableError()}>
          <p class="notice error">Installed list fetch failed: {availableError()}</p>
        </Show>
        <div class="panel">
          <Show
            when={available() !== null}
            fallback={
              <div class="panel-row">
                <span class="grow muted">Loading…</span>
              </div>
            }
          >
            <Show
              when={(available()?.length ?? 0) > 0}
              fallback={
                <div class="panel-row">
                  <span class="grow muted">
                    No models installed — install one below.
                  </span>
                </div>
              }
            >
              <For each={available() ?? []}>
                {(entry) => (
                  <div class="panel-row">
                    <span class="grow mono">
                      <button
                        class="copy-btn"
                        title="Copy"
                        onClick={() => void copy(entry.model ?? "", entry.model ?? "")}
                      >
                        {copied() === entry.model ? "copied ✓" : entry.model}
                      </button>
                    </span>
                    <Show when={entry.family}>
                      <span class="badge">{entry.family}</span>
                    </Show>
                    <Show when={entry.target_format}>
                      <span class="badge">{entry.target_format}</span>
                    </Show>
                    <Show when={entry.vision_format}>
                      <span class="badge">{entry.vision_format}</span>
                    </Show>
                    <Show
                      when={
                        (entry.tuned_knobs ?? 0) > 0 && entry.tuning === "tuned"
                      }
                    >
                      <span
                        class="badge ok"
                        title={`${entry.tune_mode ? `${entry.tune_mode} sweep · ` : ""}${entry.tuned_knobs} engine knobs measured on this chip, applied on every load${
                          entry.tune_verdicts
                            ? ` · ${Object.entries(entry.tune_verdicts)
                                .map(([v, n]) => `${n} ${v}`)
                                .join(", ")}`
                            : ""
                        }`}
                      >
                        tuned·{entry.tuned_knobs}
                        {entry.tune_mode === "quick" ? "·quick" : ""}
                      </span>
                    </Show>
                    <Show when={entry.serving && entry.requires_reload}>
                      <span
                        class="badge warn"
                        title="A tune finished while this model was loaded — its new knobs apply on the next load"
                      >
                        reload to apply
                      </span>
                    </Show>
                    <Show when={entry.tuning === "partial"}>
                      <span
                        class="badge warn"
                        title="A sweep kept these knobs but never finished — Tune again to resume"
                      >
                        partial{(entry.tuned_knobs ?? 0) > 0
                          ? `·${entry.tuned_knobs}`
                          : ""}
                      </span>
                    </Show>
                    <Show when={entry.tuning === "stale"}>
                      <span
                        class="badge warn"
                        title="A tune exists but no longer applies — recorded by an older sweep or engine build"
                      >
                        tuning stale
                      </span>
                    </Show>
                    <Show when={entry.tuning === "untuned" && entry.tunable}>
                      <span
                        class="badge"
                        title="Never tuned on this chip — Auto Tune measures and keeps its fastest engine settings"
                      >
                        not tuned
                      </span>
                    </Show>
                    <span class="num">{fmtBytes(entry.bytes)}</span>
                    <button
                      class="btn small"
                      disabled={
                        tuneBusy() ||
                        !!tuneJob()?.running ||
                        entry.tunable === false
                      }
                      title={
                        entry.tunable === false
                          ? "No tunable knobs for this model's family"
                          : tuneJob()?.running
                            ? `A tune of ${tuneJob()?.model} is running`
                            : entry.tuning === "stale"
                              ? "Re-measure this chip's fastest engine knobs — the record went stale"
                              : "Measure this chip's fastest engine knobs for this model"
                      }
                      onClick={() => setConfirmTune(entry)}
                    >
                      {tuneJob()?.running && tuneJob()?.model === entry.model
                        ? "Tuning…"
                        : "Tune"}
                    </button>
                    <Show
                      when={entry.serving}
                      fallback={
                        <button
                          class="btn small"
                          disabled={modelWorking()}
                          title={`Swap the served model to ${entry.model}`}
                          onClick={() => void loadModel(entry.model ?? "")}
                        >
                          {modelWorking() ? "Working…" : "Load"}
                        </button>
                      }
                    >
                      <span class="badge ok">serving</span>
                      <button
                        class="btn small danger"
                        disabled={modelWorking()}
                        title="Free the served model; requests fail until one is loaded again"
                        onClick={() => setConfirmUnload(entry)}
                      >
                        Unload
                      </button>
                    </Show>
                  </div>
                )}
              </For>
            </Show>
          </Show>
        </div>
        <Show when={tuneJob()?.model}>
          <div class="panel install-job">
            <div class="panel-row">
              <span class="grow">
                <Show
                  when={tuneJob()?.done}
                  fallback={
                    <>
                      Tuning <span class="mono">{tuneJob()?.model}</span>…{" "}
                      <span class="muted">{tuneJob()?.mode ?? "quick"}</span>
                    </>
                  }
                >
                  <Show
                    when={tuneJob()?.ok}
                    fallback={
                      <>
                        Tune of <span class="mono">{tuneJob()?.model}</span>{" "}
                        {tuneJob()?.cancelled
                          ? "cancelled"
                          : `failed${tuneJob()?.error ? ` — ${tuneJob()?.error}` : ""}`}
                        <Show when={(tuneJob()?.kept?.length ?? 0) > 0}>
                          {` — kept ${tuneJob()?.kept?.join(", ")} so far; a retry resumes`}
                        </Show>
                      </>
                    }
                  >
                    Tuned <span class="mono">{tuneJob()?.model}</span>
                    <Show when={tuneJob()?.headline}>
                      {` — ${tuneJob()?.headline}`}
                    </Show>
                    <Show
                      when={(tuneJob()?.kept?.length ?? 0) > 0}
                      fallback={" — defaults already won"}
                    >
                      {` · kept ${tuneJob()?.kept?.join(", ")}`}
                    </Show>
                  </Show>
                </Show>
              </span>
              <span class="num">{tuneElapsed()}</span>
              <Show
                when={tuneJob()?.running}
                fallback={
                  <Show when={tuneJob()?.done && tuneJob()?.ok}>
                    <button
                      class="btn small"
                      title="Load this model now"
                      disabled={modelWorking()}
                      onClick={() => void loadModel(tuneJob()?.model ?? "")}
                    >
                      Load it
                    </button>
                  </Show>
                }
              >
                <button
                  class="btn small danger"
                  title="Stop the sweep — its partial record resumes on retry"
                  onClick={() => void cancelTune()}
                >
                  Cancel
                </button>
              </Show>
            </div>
            <Show when={tuneJob()?.running}>
              <div class="panel-row meter-row tune-progress-row">
                <div class="tune-progress">
                  <div
                    class="meter"
                    role="progressbar"
                    aria-label="Sweep progress"
                    aria-valuemin={0}
                    aria-valuemax={tuneJob()?.phase_total || 1}
                    aria-valuenow={tuneJob()?.phase_done ?? 0}
                  >
                    <i
                      classList={{ pending: !(tuneJob()?.phase_total ?? 0) }}
                      style={{
                        width: `${
                          (tuneJob()?.phase_total ?? 0) > 0
                            ? Math.min(
                                100,
                                ((tuneJob()?.phase_done ?? 0) /
                                  (tuneJob()?.phase_total || 1)) *
                                  100
                              )
                            : 100
                        }%`,
                      }}
                    />
                  </div>
                  <div class="tune-sub">
                    <span>{tunePhase()}</span>
                    <Show when={(tuneJob()?.saved ?? 0) > 0}>
                      <span class="chip">{tuneJob()?.saved} skipped by priors</span>
                    </Show>
                    <span class="num">{tailQuiet()}s quiet</span>
                  </div>
                  <Show when={tuneJob()?.current}>
                    {(current) => (
                      <div class="tune-sub">
                        <span class="mono">{current().name}</span>
                        {current().outcome === "kept" ? (
                          <span class="badge ok">kept</span>
                        ) : current().outcome === "failed" ? (
                          <span class="badge err">failed</span>
                        ) : current().outcome === "measuring" ? (
                          <span class="muted">measuring…</span>
                        ) : (
                          <span class="muted">measured</span>
                        )}
                      </div>
                    )}
                  </Show>
                </div>
              </div>
            </Show>
            <Show when={tuneJob()?.done && (tuneJob()?.results?.length ?? 0) > 0}>
              <div class="panel-row">
                <span class="grow muted mono" style="white-space: normal">
                  {tuneJob()?.results?.join("  ·  ")}
                </span>
              </div>
            </Show>
            <Show when={(tuneJob()?.tail?.length ?? 0) > 0}>
              <details
                class="tune-log"
                open={Boolean(tuneJob()?.done && !tuneJob()?.ok)}
              >
                <summary>Sweep log</summary>
                <pre
                  class="install-log"
                  ref={tuneLogEl}
                  onScroll={(event) => {
                    const el = event.currentTarget;
                    tuneLogPinned =
                      el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
                  }}
                >
                  {tuneJob()?.tail?.join("\n")}
                </pre>
              </details>
            </Show>
          </div>
        </Show>
      </section>

      {/* 1c. Install a model */}
      <section class="section">
        <h2>Install a model</h2>
        <form class="panel install-form" onSubmit={submitInstall}>
          <div class="panel-row">
            <span class="grow">Hugging Face repo</span>
            <input
              class="install-input mono"
              list="suggested-models"
              placeholder="owner/repo[:variant]"
              aria-label="Hugging Face repository ID"
              spellcheck={false}
              value={installModel()}
              onInput={(event) => setInstallModel(event.currentTarget.value)}
            />
            <datalist id="suggested-models">
              <For each={suggested()}>{(id) => <option value={id} />}</For>
            </datalist>
          </div>
          <Show when={suggested().length > 0}>
            <div class="panel-row">
              <span class="grow muted">Suggested</span>
              <span class="chips">
                <For each={suggested().slice(0, 6)}>
                  {(id) => (
                    <button
                      type="button"
                      class="chip chip-btn"
                      title={`Fill ${id}`}
                      onClick={() => setInstallModel(id)}
                    >
                      {id}
                    </button>
                  )}
                </For>
              </span>
            </div>
          </Show>
          <div class="panel-row">
            <span class="grow">
              Revision <span class="muted">— optional</span>
            </span>
            <input
              class="install-input mono"
              placeholder="branch, tag or commit"
              aria-label="Revision"
              spellcheck={false}
              value={installRevision()}
              onInput={(event) => setInstallRevision(event.currentTarget.value)}
            />
          </div>
          <div class="panel-row">
            <span class="grow">
              Draft model <span class="muted">— optional</span>
            </span>
            <input
              class="install-input mono"
              placeholder="owner/repo or local directory"
              aria-label="Draft model"
              spellcheck={false}
              value={installDraft()}
              onInput={(event) => setInstallDraft(event.currentTarget.value)}
            />
          </div>
          <div class="panel-row">
            <label class="install-check grow muted">
              <input
                type="checkbox"
                title="Skip vision weights — installs a text-only build"
                checked={installLanguageOnly()}
                onChange={(event) =>
                  setInstallLanguageOnly(event.currentTarget.checked)
                }
              />
              language only — skip vision weights
            </label>
            <button
              class="btn small primary"
              type="submit"
              title="Download and prepare this model from Hugging Face"
              disabled={
                installBusy() || !!installJob()?.running || !installModel().trim()
              }
            >
              {installBusy() ? "Starting…" : "Install"}
            </button>
          </div>
        </form>
        <Show when={installJob()?.model}>
          <div class="panel install-job">
            <div class="panel-row">
              <span class="grow">
                <Show
                  when={installJob()?.done}
                  fallback={
                    <>Installing <span class="mono">{installJob()?.model}</span>…</>
                  }
                >
                  <Show
                    when={installJob()?.ok}
                    fallback={
                      <>
                        Install of <span class="mono">{installJob()?.model}</span>{" "}
                        {installJob()?.cancelled ? "cancelled" : "failed"}
                      </>
                    }
                  >
                    Installed <span class="mono">{installJob()?.model}</span>
                  </Show>
                </Show>
              </span>
              <span class="num">{installElapsed()}</span>
              <Show
                when={installJob()?.running}
                fallback={
                  <Show when={installJob()?.done && installJob()?.ok}>
                    <button
                      class="btn small"
                      title="Load this model now"
                      disabled={modelWorking()}
                      onClick={() => void loadModel(installJob()?.model ?? "")}
                    >
                      Load it
                    </button>
                  </Show>
                }
              >
                <button
                  class="btn small danger"
                  title="Stop the install — partial downloads resume on retry"
                  onClick={() => void cancelInstall()}
                >
                  Cancel
                </button>
              </Show>
            </div>
            <Show when={(installJob()?.tail?.length ?? 0) > 0}>
              <pre
                class="install-log"
                ref={logEl}
                onScroll={(event) => {
                  const el = event.currentTarget;
                  logPinned =
                    el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
                }}
              >
                {installJob()?.tail?.join("\n")}
              </pre>
            </Show>
          </div>
        </Show>
      </section>

      {/* 2. Performance */}
      <Show when={s().metrics}>
        <section class="section">
          <h2>Performance</h2>
          <div class="panel">
            <Row label="Decode throughput" value={`${rate(s().metrics?.decode_tokens_per_second)} tok/s`} />
            <Row label="Prefill throughput" value={`${rate(s().metrics?.prefill_tokens_per_second)} tok/s`} />
            <Show when={history().decode.length > 1}>
              <div class="panel-row chart-row">
                <Chart
                  type="area"
                  height={140}
                  series={[
                    { name: "decode tok/s", data: history().decode },
                    { name: "prefill tok/s", data: history().prefill },
                  ]}
                  options={{
                    legend: { show: true, position: "top", horizontalAlign: "right" },
                    xaxis: { labels: { show: false } },
                    yaxis: {
                      min: 0,
                      labels: { formatter: (v: number) => fmtInt(v) },
                    },
                  }}
                />
              </div>
            </Show>
            <Row
              label="TTFT p50 / p95"
              value={`${fmtMs(s().metrics?.ttft_ms?.p50)} / ${fmtMs(s().metrics?.ttft_ms?.p95)}`}
              sub={`${fmtInt(s().metrics?.ttft_ms?.samples)} samples`}
            />
            <Row
              label="ITL p50 / p95"
              value={`${fmtMs(s().metrics?.itl_ms?.p50)} / ${fmtMs(s().metrics?.itl_ms?.p95)}`}
              sub={`${fmtInt(s().metrics?.itl_ms?.samples)} samples`}
            />
            <Show when={s().metrics?.ttft_ms || s().metrics?.itl_ms}>
              <div class="panel-row chart-row">
                <Chart
                  type="bar"
                  height={110}
                  series={[
                    {
                      name: "p50",
                      data: [
                        s().metrics?.ttft_ms?.p50 ?? 0,
                        s().metrics?.itl_ms?.p50 ?? 0,
                      ],
                    },
                    {
                      name: "p95",
                      data: [
                        s().metrics?.ttft_ms?.p95 ?? 0,
                        s().metrics?.itl_ms?.p95 ?? 0,
                      ],
                    },
                  ]}
                  options={{
                    plotOptions: {
                      bar: { horizontal: true, borderRadius: 3, barHeight: "60%" },
                    },
                    fill: { opacity: 0.9 },
                    legend: { show: true, position: "top", horizontalAlign: "right" },
                    xaxis: {
                      categories: ["TTFT", "ITL"],
                      labels: { formatter: (v: string) => fmtMs(Number(v)) },
                    },
                    tooltip: { y: { formatter: (v: number) => fmtMs(v) } },
                  }}
                />
              </div>
            </Show>
            <Row
              label="Draft acceptance"
              value={pct(s().metrics?.draft_acceptance_rate)}
              sub={`${fmtInt(s().metrics?.accepted_draft_tokens)} / ${fmtInt(s().metrics?.drafted_tokens)} tokens`}
            />
          </div>
        </section>
      </Show>

      {/* 3. Scheduler */}
      <Show when={s().scheduler}>
        <section class="section">
          <h2>Scheduler</h2>
          <div class="panel">
            <Row label="Queued" value={fmtInt(s().scheduler?.queued)} />
            <Row label="Prefilling" value={fmtInt(s().scheduler?.prefilling)} />
            <Row label="Decoding" value={fmtInt(s().scheduler?.decoding)} />
            <Row label="Waiting on resources" value={fmtInt(s().scheduler?.waiting_resources)} />
            <Row label="Waiting on prefix" value={fmtInt(s().scheduler?.waiting_prefix)} />
            <Row label="Waiting on mask" value={fmtInt(s().scheduler?.waiting_mask)} />
            <Row label="Terminal" value={fmtInt(s().scheduler?.terminal)} />
            <Row label="Decode batches by width" value={widthRow()} />
            <Show when={s().scheduler?.decode_batches_by_width}>
              <div class="panel-row chart-row">
                <Chart
                  type="bar"
                  height={90}
                  series={[
                    {
                      name: "batches",
                      data: [
                        s().scheduler?.decode_batches_by_width?.b1 ?? 0,
                        s().scheduler?.decode_batches_by_width?.b2 ?? 0,
                        s().scheduler?.decode_batches_by_width?.b3 ?? 0,
                        s().scheduler?.decode_batches_by_width?.b4 ?? 0,
                      ],
                    },
                  ]}
                  options={{
                    plotOptions: { bar: { borderRadius: 3, columnWidth: "55%" } },
                    fill: { opacity: 0.9 },
                    xaxis: { categories: ["b1", "b2", "b3", "b4"] },
                    yaxis: {
                      min: 0,
                      labels: { formatter: (v: number) => fmtInt(v) },
                    },
                  }}
                />
              </div>
            </Show>
            <Show when={s().metrics?.current_decode_batch?.valid}>
              <Row
                label="Current decode batch"
                value={`width ${fmtInt(s().metrics?.current_decode_batch?.width)} · ${rate(
                  s().metrics?.current_decode_batch?.tokens_per_second
                )} tok/s`}
              />
            </Show>
            <Show when={s().metrics?.current_prefill_batch?.valid}>
              <Row
                label="Current prefill batch"
                value={`${fmtInt(s().metrics?.current_prefill_batch?.input_tokens)} input tokens`}
              />
            </Show>
          </div>
        </section>
      </Show>

      {/* 4. KV cache */}
      <Show when={s().kv}>
        <section class="section">
          <details class="collapse" data-name="kv" open={collapseOpen("kv")} onToggle={collapseToggle}>
            <summary title="Collapse section">KV cache</summary>
            <div class="panel">
            <MeterRow
              label="Pages active"
              value={s().kv?.pages_active}
              max={s().kv?.pages_allocated}
              text={`${fmtInt(s().kv?.pages_active)} / ${fmtInt(s().kv?.pages_allocated)} pages`}
            />
            <Row label="Cache pages" value={fmtInt(s().kv?.pages_cache)} />
            <Row label="Free pages" value={fmtInt(s().kv?.pages_free)} />
            <Row label="Allocated" value={fmtBytes(s().kv?.allocated_bytes)} />
            <Row label="Hit rate" value={pct(s().cache?.hit_rate)} />
            <Row label="KV hit tokens" value={fmtInt(s().cache?.kv_hit_tokens)} />
            <Row label="Reused tokens" value={fmtInt(s().cache?.reused_tokens)} />
            <Row
              label="Hits vs cold misses"
              value={`${fmtInt(s().cache?.hits)} / ${fmtInt(s().cache?.cold_misses)}`}
            />
            </div>
          </details>
        </section>
      </Show>

      {/* 5. Memory */}
      <Show when={s().memory_actual || s().memory_governor}>
        <section class="section">
          <details class="collapse" data-name="memory" open={collapseOpen("memory")} onToggle={collapseToggle}>
            <summary title="Collapse section">Memory</summary>
            <div class="panel">
            <MeterRow
              label="Current usage"
              value={s().memory_actual?.current_bytes}
              max={hardBudget()}
              text={`${fmtBytes(s().memory_actual?.current_bytes)} / ${fmtBytes(hardBudget())}`}
            />
            <Show when={history().memory.length > 1}>
              <div class="panel-row chart-row">
                <Chart
                  type="area"
                  height={120}
                  series={[{ name: "bytes", data: history().memory }]}
                  options={{
                    xaxis: { labels: { show: false } },
                    yaxis: {
                      min: 0,
                      labels: { formatter: (v: number) => fmtBytes(v) },
                    },
                    tooltip: { y: { formatter: (v: number) => fmtBytes(v) } },
                  }}
                />
              </div>
            </Show>
            <Row label="Allocated" value={fmtBytes(s().memory_actual?.allocated_bytes)} />
            <Row label="Current" value={fmtBytes(s().memory_actual?.current_bytes)} />
            <Row label="Peak" value={fmtBytes(s().memory_actual?.peak_bytes)} />
            <Row label="Governor limit" value={fmtBytes(s().memory_governor?.limit_bytes)} />
            <Row label="Governor charged" value={fmtBytes(s().memory_governor?.charged_bytes)} />
            <Row label="Governor headroom" value={fmtBytes(s().memory_governor?.headroom_bytes)} />
            <Row label="Host available" value={fmtBytes(s().memory_governor?.host_available_bytes)} />
            <Row label="Denied reservations" value={fmtInt(s().memory_governor?.denied_reservations)} />
            </div>
          </details>
        </section>
      </Show>

      {/* 6. State & disk */}
      <Show when={s().state || s().disk}>
        <section class="section">
          <details class="collapse" data-name="state-disk" open={collapseOpen("state-disk")} onToggle={collapseToggle}>
            <summary title="Collapse section">State &amp; disk</summary>
            <div class="panel">
            <Row
              label="State entries"
              value={fmtInt(s().state?.entries)}
              sub={`${fmtInt(s().state?.pinned)} pinned · ${fmtInt(s().state?.in_use)} in use`}
            />
            <Row label="Disk hits" value={fmtInt(s().state?.disk_hits)} />
            <Row label="Disk promotions" value={fmtInt(s().state?.disk_promotions)} />
            <Row label="Offloads" value={fmtInt(s().state?.offloads)} />
            <Show when={s().disk}>
              <MeterRow
                label="Disk used"
                value={s().disk?.used_bytes}
                max={s().disk?.capacity_bytes}
                text={`${fmtBytes(s().disk?.used_bytes)} / ${fmtBytes(s().disk?.capacity_bytes)}`}
              />
              <Row
                label="KV on disk"
                value={fmtBytes(s().disk?.kv_bytes)}
                sub={`${fmtInt(s().disk?.kv_blocks)} blocks`}
              />
              <Row
                label="KV demotions / restores"
                value={`${fmtInt(s().disk?.kv_demotions)} / ${fmtInt(s().disk?.kv_restores)}`}
              />
              <Row
                label="Write-behind"
                value={`${fmtInt(s().disk?.write_behind?.waiting)} waiting · ${fmtInt(
                  s().disk?.write_behind?.durable
                )} durable`}
              />
            </Show>
            </div>
          </details>
        </section>
      </Show>

      {/* 7. Warmup */}
      <Show when={s().warmup}>
        <section class="section">
          <details class="collapse" data-name="warmup" open={collapseOpen("warmup")} onToggle={collapseToggle}>
            <summary title="Collapse section">Warmup</summary>
            <div class="panel">
            <For each={warmupSteps()}>
              {(name) => {
                const done = () => s().warmup?.[name] === true;
                const limited = () => warmupLimited().includes(name);
                return (
                  <div class="panel-row check">
                    <span class="grow">{name}</span>
                    <Show when={limited()}>
                      <span class="num">memory limited</span>
                    </Show>
                    <Badge kind={done() ? "ok" : "warn"}>{done() ? "done" : "pending"}</Badge>
                  </div>
                );
              }}
            </For>
            <Show when={warmupLimited().length > 0}>
              <Row label="Memory limited" value={warmupLimited().join(", ")} />
            </Show>
            <Show when={typeof s().warmup?.["detail"] === "string" && s().warmup?.["detail"]}>
              <div class="panel-row">
                <span class="grow muted">{String(s().warmup?.["detail"])}</span>
              </div>
            </Show>
            </div>
          </details>
        </section>
      </Show>

      {/* 8. Frontend */}
      <Show when={s().frontend || s().grammar_cache || s().tokenizer_cache || s().image_cache || s().response_store}>
        <section class="section">
          <details class="collapse" data-name="frontend" open={collapseOpen("frontend")} onToggle={collapseToggle}>
            <summary title="Collapse section">Frontend</summary>
            <div class="panel">
            <Show when={s().frontend}>
              <Row
                label="Preparation"
                value={`${fmtInt(s().frontend?.active)} active / ${fmtInt(
                  s().frontend?.preparation_capacity
                )} capacity`}
                sub={`${fmtInt(s().frontend?.waiting)} waiting`}
              />
            </Show>
            <Row label="Grammar cache" value={cacheValue(s().grammar_cache)} />
            <Row label="Tokenizer cache" value={cacheValue(s().tokenizer_cache)} />
            <Row label="Image cache" value={cacheValue(s().image_cache)} />
            <Row label="Response store" value={cacheValue(s().response_store)} />
            <Show when={latencyChips().length > 0}>
              <div class="panel-row">
                <span class="grow">Latency</span>
                <span class="chips">
                  <For each={latencyChips()}>
                    {(stage) => (
                      <span class="chip">
                        {stage.name} ×{fmtInt(stage.count)} · {fmtMs(stage.avgMs)}
                      </span>
                    )}
                  </For>
                </span>
              </div>
            </Show>
            {/* One compact histogram per stage that reports buckets. */}
            <For each={latencyStages()}>
              {(stage) => (
                <div class="panel-row chart-row">
                  <div>
                    <span class="chart-label">{stage.name}</span>
                    <Chart
                      type="bar"
                      height={80}
                      series={[{ name: stage.name, data: stage.data }]}
                      options={{
                        plotOptions: {
                          bar: { borderRadius: 2, columnWidth: "60%" },
                        },
                        fill: { opacity: 0.9 },
                        xaxis: { categories: stage.cats },
                        yaxis: {
                          min: 0,
                          labels: { formatter: (v: number) => fmtInt(v) },
                        },
                      }}
                    />
                  </div>
                </div>
              )}
            </For>
            </div>
          </details>
        </section>
      </Show>

      {/* 9. Requests */}
      <Show when={s().requests || s().model_timing || s().identity}>
        <section class="section">
          <details class="collapse" data-name="requests" open={collapseOpen("requests")} onToggle={collapseToggle}>
            <summary title="Collapse section">Requests</summary>
            <div class="panel">
            <Show when={s().requests}>
              <Row
                label="Requests"
                value={`${fmtInt(s().requests?.submitted)} submitted · ${fmtInt(
                  s().requests?.completed
                )} completed · ${fmtInt(s().requests?.cancelled)} cancelled · ${fmtInt(
                  s().requests?.failed
                )} failed`}
              />
            </Show>
            <Show when={s().model_timing}>
              <Row
                label="Prefill timing"
                value={`${fmtMs(s().model_timing?.prefill?.last_gpu_ms)} gpu · ${fmtMs(
                  s().model_timing?.prefill?.last_wall_ms
                )} wall`}
              />
              <Row
                label="Decode timing"
                value={`${fmtMs(s().model_timing?.decode?.last_gpu_ms)} gpu · ${fmtMs(
                  s().model_timing?.decode?.last_wall_ms
                )} wall`}
              />
            </Show>
            <Show when={s().ane_ffn?.state && s().ane_ffn?.state !== "off"}>
              <Row
                label="ANE FFN"
                value={`${s().ane_ffn?.state} · ${pct(s().ane_ffn?.share)} share · ${fmtInt(
                  s().ane_ffn?.reruns
                )} reruns`}
                sub={s().ane_ffn?.reason}
              />
            </Show>
            <Show when={s().identity}>
              <Row
                label="KV identity"
                value={`${s().identity?.kv?.format ?? "—"} · ${
                  s().identity?.kv?.quantization ?? "—"
                } · scale ${s().identity?.kv?.scale_type ?? "—"}`}
              />
              <Row
                label="Cache"
                value={`${s().identity?.cache?.dtype ?? "—"} · ${fmtInt(
                  s().identity?.cache?.block_tokens
                )} tok/block`}
                sub={buildSub()}
              />
            </Show>
            </div>
          </details>
        </section>
      </Show>

      <Confirm
        open={confirmUnload() !== null}
        title={`Unload ${confirmUnload()?.model ?? "model"}?`}
        confirmLabel="Unload"
        busy={modelBusy()}
        onConfirm={() => void unloadModel()}
        onClose={() => setConfirmUnload(null)}
      >
        The served model is freed and requests fail until another one is
        loaded. In-flight requests may be interrupted.
      </Confirm>

      <Confirm
        open={confirmTune() !== null}
        title={`Tune ${confirmTune()?.model ?? "model"}?`}
        confirmLabel="Complete"
        confirmClass="btn primary"
        secondaryLabel="Quick"
        secondaryClass="btn"
        busy={tuneBusy()}
        onSecondary={() => {
          const entry = confirmTune();
          if (entry) void startTune(entry, "quick");
        }}
        onConfirm={() => {
          const entry = confirmTune();
          if (entry) void startTune(entry, "complete");
        }}
        onClose={() => setConfirmTune(null)}
      >
        <p>
          <b>Quick</b> sweeps this chip's performance knobs — kernel and
          draft paths. <b>Complete</b> adds quality-trading knobs (faster
          output, lower quality) and a knob-interaction recheck — roughly
          double the time.
        </p>
        <Show
          when={
            s().model_state === "loaded" || s().model_state === "loading"
          }
        >
          <p>
            A model is loaded — its engine shares the GPU and skews every
            measurement. Unload it first for a clean record, or tune anyway.
          </p>
        </Show>
      </Confirm>

      {/* 10. Raw /metrics, fetched on expand */}
      <section class="section">
        <details
          class="metrics-raw"
          data-name="raw-metrics"
          open={collapseOpen("raw-metrics", false)}
          onToggle={(event) => {
            collapseToggle(event);
            if (event.currentTarget.open && metricsText() === null)
              void loadMetrics();
          }}
        >
          <summary title="Expand section">Raw /metrics</summary>
          <Show when={metricsText() !== null} fallback={<p class="muted">Loading…</p>}>
            <pre class="metrics-pre">{metricsText()}</pre>
          </Show>
        </details>
      </section>
    </main>
  );
}

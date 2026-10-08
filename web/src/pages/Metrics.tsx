import { createMemo, createSignal, onCleanup, For, Show } from "solid-js";
import { authorization, errorText, noteAuthStatus } from "../api";
import { Chart } from "../Chart";
import { useCopied } from "../clipboard";
import { useFreshness } from "../fresh";
import { fmtInt, fmtMs } from "../format";
import { poll } from "../poll";
import "./metrics.css";

/* ---- /metrics text exposition (Prometheus wire format) ---- */

interface MetricSample {
  /* Emitted name — `foo_bucket`/`_sum`/`_count` for histogram children. */
  name: string;
  /* Label body without braces, "" when the series carries none. */
  labels: string;
  value: number;
}

interface MetricFamily {
  name: string;
  type: string; /* counter | gauge | histogram | summary | untyped */
  help: string;
  samples: MetricSample[];
}

interface Scrape {
  families: MetricFamily[];
  raw: string;
  sampleCount: number;
  /* Non-comment lines that did not parse as a sample. */
  errors: number;
}

/* Histogram/summary children roll up under the base family when a HELP or
   TYPE declared it (they precede the samples in the exposition). */
const CHILD = /_(bucket|sum|count)$/;

/* parseFloat chokes on the wire's +Inf/NaN spellings; all three are legal. */
function parseValue(raw: string): number | null {
  if (raw === "NaN") return NaN;
  if (raw === "+Inf" || raw === "Inf") return Infinity;
  if (raw === "-Inf") return -Infinity;
  const v = Number.parseFloat(raw);
  return Number.isFinite(v) ? v : null;
}

function parseMetrics(text: string): Omit<Scrape, "raw"> {
  const help = new Map<string, string>();
  const types = new Map<string, string>();
  const families = new Map<string, MetricFamily>();
  let errors = 0;
  let sampleCount = 0;

  const familyOf = (name: string): MetricFamily => {
    let f = families.get(name);
    if (!f) {
      f = {
        name,
        type: types.get(name) ?? "untyped",
        help: help.get(name) ?? "",
        samples: [],
      };
      families.set(name, f);
    }
    return f;
  };

  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    if (s.startsWith("#")) {
      const m = s.match(/^#\s*(HELP|TYPE)\s+(\S+)(?:\s+(.*))?$/);
      if (m) {
        if (m[1] === "HELP") help.set(m[2], (m[3] ?? "").trim());
        else types.set(m[2], (m[3] ?? "untyped").trim());
      }
      /* Other directives (# EOF, # UNIT) carry no signal, not errors. */
      continue;
    }
    /* name{labels} value [timestamp] | name value [timestamp] */
    const m = s.match(/^([^\s{]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$/);
    const value = m ? parseValue(m[3]) : null;
    if (!m || value === null) {
      errors++;
      continue;
    }
    const [, name, labels = ""] = m;
    const base = name.replace(CHILD, "");
    const target =
      base !== name && (types.has(base) || help.has(base)) ? base : name;
    familyOf(target).samples.push({ name, labels, value });
    sampleCount++;
  }

  /* HELP/TYPE may trail a family's samples; backfill before sorting. */
  const out = [...families.values()];
  for (const f of out) {
    f.type = types.get(f.name) ?? f.type;
    f.help = help.get(f.name) ?? f.help;
  }
  out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return { families: out, sampleCount, errors };
}

/* Reflect a collapsible section's open state as a summary tooltip. */
function collapseToggle(event: Event) {
  const details = event.currentTarget as HTMLDetailsElement | null;
  const summary = details?.querySelector("summary");
  if (details && summary)
    summary.title = details.open ? "Collapse section" : "Expand section";
}

/* Series history lives at module scope: navigating away and back keeps the
   collected points instead of restarting every chart empty. */
const seriesHistory = new Map<string, number[]>();

/* Compact series rendering: ints localized, long tails in exponential. */
function fmtValue(v: number): string {
  if (Number.isNaN(v)) return "NaN";
  if (!Number.isFinite(v)) return v > 0 ? "+Inf" : "-Inf";
  const a = Math.abs(v);
  if (a >= 1e15 || (a !== 0 && a < 1e-4)) return v.toExponential(3);
  return Number.isInteger(v) ? fmtInt(v) : String(Number(v.toPrecision(6)));
}

/* ---- page ---- */

export default function Metrics() {
  const [scrape, setScrape] = createSignal<Scrape | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [fetching, setFetching] = createSignal(false);
  const [auto, setAuto] = createSignal(true);
  const [query, setQuery] = createSignal("");
  const [selected, setSelected] = createSignal<string | null>(null);
  const { markOk, updatedAgo, stale } = useFreshness();
  const [durationMs, setDurationMs] = createSignal<number | null>(null);
  /* Bumped so chartSeries re-reads the module-level seriesHistory. */
  const [historyTick, setHistoryTick] = createSignal(0);
  const { copied, copy } = useCopied();

  const seriesKey = (family: string, s: MetricSample) =>
    `${family} ${s.name} ${s.labels}`;

  /* Detail row label: `_bucket{le="0.5"}`, `{route="/v1"}`, or "value". */
  const seriesLabel = (f: MetricFamily, s: MetricSample): string => {
    const suffix = s.name === f.name ? "" : s.name.slice(f.name.length);
    const labels = s.labels ? `{${s.labels}}` : "";
    return `${suffix}${labels}` || "value";
  };

  async function refresh() {
    /* poll() already serializes, but a manual click can overlap it. */
    if (fetching()) return;
    setFetching(true);
    const started = performance.now();
    try {
      const res = await fetch("/metrics", { headers: authorization() });
      noteAuthStatus(res.status);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`.trim());
      const text = await res.text();
      const parsed = parseMetrics(text);
      setDurationMs(performance.now() - started);
      setScrape({ ...parsed, raw: text });
      setError(null);
      markOk();
      for (const f of parsed.families)
        for (const s of f.samples) {
          const key = seriesKey(f.name, s);
          seriesHistory.set(
            key,
            [...(seriesHistory.get(key) ?? []), s.value].slice(-60)
          );
        }
      setHistoryTick((t) => t + 1);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setFetching(false);
    }
  }

  /* 2s scrape poll while auto-refresh is on; freshness clock is shared. */
  onCleanup(poll(() => (auto() ? refresh() : Promise.resolve()), 2000));

  const filtered = () => {
    const q = query().trim().toLowerCase();
    const all = scrape()?.families ?? [];
    return q
      ? all.filter(
          (f) =>
            f.name.toLowerCase().includes(q) ||
            f.help.toLowerCase().includes(q)
        )
      : all;
  };

  const selectedFamily = () =>
    scrape()?.families.find((f) => f.name === selected()) ?? null;

  /* The exposition can run tens of thousands of lines; render a cap and
     report the rest — Copy raw still hands over the full text. */
  const RAW_CAP = 1500;
  const rawView = createMemo(() => {
    const raw = scrape()?.raw ?? "";
    const lines = raw.split("\n");
    return lines.length > RAW_CAP
      ? { text: lines.slice(0, RAW_CAP).join("\n"), more: lines.length - RAW_CAP }
      : { text: raw, more: 0 };
  });

  /* Buckets chart poorly — skip them, but _sum/_count still draw. */
  const chartable = (f: MetricFamily, s: MetricSample) =>
    f.type === "histogram" || f.type === "summary"
      ? !s.name.endsWith("_bucket")
      : true;

  const chartSeries = () => {
    const f = selectedFamily();
    if (!f) return [];
    historyTick();
    return f.samples
      .filter((s) => chartable(f, s))
      .slice(0, 8)
      .map((s) => ({
        name: seriesLabel(f, s),
        data: seriesHistory.get(seriesKey(f.name, s)) ?? [],
      }))
      .filter((s) => s.data.length >= 2);
  };

  async function copyRaw() {
    const raw = scrape()?.raw;
    if (raw === undefined) return;
    await copy("raw", raw);
  }

  return (
    <main class="page">
      <div class="page-head">
        <div>
          <h1>Metrics</h1>
          <p class="lede">Prometheus /metrics explorer.</p>
        </div>
        <div class="head-side">
          <Show when={updatedAgo()}>
            <span
              class={`freshness${stale() ? " stale" : ""}`}
              title="Time since the last successful /metrics fetch"
            >
              fetched {updatedAgo()} ago
            </span>
          </Show>
          <label class="check">
            <input
              type="checkbox"
              title="Scrape /metrics every 2s"
              checked={auto()}
              onChange={(e) => setAuto(e.currentTarget.checked)}
            />
            auto-refresh
          </label>
          <button
            class="btn small"
            title="Fetch /metrics now"
            onClick={() => void refresh()}
            disabled={fetching()}
          >
            {fetching() ? "Fetching…" : "Refresh"}
          </button>
        </div>
      </div>

      <div class="metric-toolbar">
        <input
          class="metric-search"
          type="search"
          placeholder="Filter by metric name or help…"
          aria-label="Filter metrics"
          value={query()}
          onInput={(e) => setQuery(e.currentTarget.value)}
        />
        <Show when={query().trim() && scrape()}>
          <span class="muted match-count">
            {filtered().length} of {scrape()?.families.length ?? 0}
          </span>
        </Show>
        <button
          class="btn small"
          title="Copy the raw Prometheus exposition"
          onClick={() => void copyRaw()}
          disabled={!scrape()}
        >
          {copied() === "raw" ? "copied ✓" : "Copy raw"}
        </button>
      </div>

      <Show when={error()}>
        <p class="notice error">
          Metrics fetch failed
          {scrape() ? " — showing last good data" : ""}: {error()}
        </p>
      </Show>

      <div class="cards">
        <div class="card">
          <div class="card-label">Metric families</div>
          <div class="card-value">{fmtInt(scrape()?.families.length)}</div>
          <div class="card-sub">HELP/TYPE groups</div>
        </div>
        <div class="card">
          <div class="card-label">Samples</div>
          <div class="card-value">{fmtInt(scrape()?.sampleCount)}</div>
          <div class="card-sub">series values</div>
        </div>
        <div class="card">
          <div class="card-label">Scrape</div>
          <div class="card-value">{fmtMs(durationMs())}</div>
          <div class="card-sub">fetch + parse</div>
        </div>
        <Show when={(scrape()?.errors ?? 0) > 0}>
          <div class="card">
            <div class="card-label">Parse errors</div>
            <div class="card-value err">{fmtInt(scrape()?.errors)}</div>
            <div class="card-sub">malformed lines skipped</div>
          </div>
        </Show>
      </div>

      <Show when={scrape()} fallback={<p class="muted">Fetch to load.</p>}>
        {/* Family table: click a row for its series + history. */}
        <section class="section">
          <div class="panel">
            <Show
              when={filtered().length > 0}
              fallback={
                <div class="panel-row metric-empty">
                  No metrics match “{query().trim()}”.
                </div>
              }
            >
              <For each={filtered()}>
                {(f) => (
                  <>
                    <button
                      class={`panel-row metric-row${
                        selected() === f.name ? " sel" : ""
                      }`}
                      title={
                        selected() === f.name
                          ? "Hide detail"
                          : "Show series and history"
                      }
                      onClick={() =>
                        setSelected(selected() === f.name ? null : f.name)
                      }
                    >
                      <span class="grow">{f.name}</span>
                      <span class="badge">{f.type}</span>
                      <span class="num help" title={f.help}>
                        {f.help}
                      </span>
                      <span class="num">{f.samples.length} series</span>
                      <span class="num metric-value">
                        {f.samples.length === 1
                          ? fmtValue(f.samples[0].value)
                          : ""}
                      </span>
                    </button>
                    {/* Detail expands in place so it stays visible wherever
                        the row sits in a long family list. */}
                    <Show when={selected() === f.name}>
                      <div class="metric-detail">
                        <div class="panel-row">
                          <span class="grow mono">{f.name}</span>
                          <span class="badge">{f.type}</span>
                          <span class="num">{f.samples.length} series</span>
                          <button
                            class="btn small"
                            title="Close"
                            aria-label="Close metric detail"
                            onClick={() => setSelected(null)}
                          >
                            close
                          </button>
                        </div>
                        <Show when={f.help}>
                          <div class="panel-row">
                            <span class="grow muted">{f.help}</span>
                          </div>
                        </Show>
                        <For each={f.samples}>
                          {(s) => (
                            <div class="panel-row series-row">
                              <span class="grow mono">{seriesLabel(f, s)}</span>
                              <span class="num mono">{fmtValue(s.value)}</span>
                            </div>
                          )}
                        </For>
                        <Show when={chartSeries().length > 0}>
                          <div class="panel-row chart-row">
                            <div>
                              <span class="chart-label">
                                history · last 60 scrapes
                              </span>
                              <Chart
                                type="area"
                                height={140}
                                series={chartSeries()}
                                options={{
                                  yaxis: {
                                    labels: {
                                      formatter: (v: number) => fmtValue(v),
                                    },
                                  },
                                }}
                              />
                            </div>
                          </div>
                        </Show>
                      </div>
                    </Show>
                  </>
                )}
              </For>
            </Show>
          </div>
        </section>

        {/* Raw exposition, for copy/paste debugging. */}
        <section class="section">
          <details class="metrics-raw" onToggle={collapseToggle}>
            <summary title="Expand section">Raw /metrics</summary>
            <pre class="metrics-pre">{rawView().text}</pre>
            <Show when={rawView().more > 0}>
              <p class="muted">
                … {fmtInt(rawView().more)} more lines — Copy raw has the full
                exposition.
              </p>
            </Show>
          </details>
        </section>
      </Show>
    </main>
  );
}

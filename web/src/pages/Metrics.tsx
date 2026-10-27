import { createEffect, createMemo, createSignal, For, onCleanup, Index, Show } from "solid-js";
import { authorization, errorText, noteAuthStatus } from "../api";
import { Chart, fmtTooltipRows } from "../Chart";
import type { ChartAppend } from "../Chart";
import { useCopied } from "../clipboard";
import { useFreshness } from "../fresh";
import { fmtBytes, fmtInt, fmtMs, pct } from "../format";
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
   collected points instead of restarting every chart empty. Each point
   carries its own scrape time, so a series that appears mid-session never
   borrows a shared clock. richengine_info's instance label keys the map —
   a restarted engine clears it. */
const seriesHistory = new Map<string, { t: number; v: number }[]>();
let instanceId: string | undefined;

type Unit = "ms" | "s" | "bytes" | "ratio" | "count" | "none";

/* Unit conventions from the metric name: *_seconds and *_milliseconds are
   times, *_bytes sizes, *_ratio/_rate fractions. */
function baseUnit(name: string): Unit {
  if (name.endsWith("_milliseconds")) return "ms";
  if (name.endsWith("_seconds")) return "s";
  if (name.endsWith("_bytes")) return "bytes";
  if (name.endsWith("_ratio") || name.endsWith("_rate")) return "ratio";
  return "none";
}

/* A sample's unit: histogram/summary _sum keeps the family unit while
   _count and _bucket are plain counts. */
function unitOf(f: MetricFamily, s?: MetricSample): Unit {
  if (s && s.name !== f.name)
    return s.name.endsWith("_sum") ? baseUnit(f.name) : "count";
  return baseUnit(f.name);
}

function fmtUnit(unit: Unit | undefined, v: number): string {
  switch (unit) {
    case "ms":
      return fmtMs(v);
    case "s":
      return fmtMs(v * 1000);
    case "bytes":
      return fmtBytes(v);
    case "ratio":
      return pct(v);
    default:
      return fmtValue(v);
  }
}

/* Families render under section heads; *_seconds histograms always land
   in Latency, the rest group by the name's second token. */
const SECTION_ORDER = [
  "Throughput",
  "Requests & scheduler",
  "Latency",
  "KV & cache",
  "Memory & hardware",
  "Masks & constraints",
  "Images",
  "Other",
] as const;

function sectionOf(name: string): string {
  if (name.endsWith("_seconds")) return "Latency";
  switch (name.replace(/^richengine_/, "").split("_")[0]) {
    case "ttft":
    case "itl":
    case "prefill":
    case "decode":
    case "drafted":
    case "accepted":
    case "draft":
    case "capacity":
      return "Throughput";
    case "requests":
    case "frontend":
    case "transport":
    case "scheduler":
    case "admission":
      return "Requests & scheduler";
    case "kv":
    case "state":
    case "cache":
    case "response":
      return "KV & cache";
    case "memory":
    case "metal":
    case "thermal":
      return "Memory & hardware";
    case "mask":
    case "masks":
    case "constraint":
      return "Masks & constraints";
    case "image":
    case "images":
      return "Images";
    default:
      return "Other";
  }
}

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
  /* Bumped so chartData re-reads the module-level seriesHistory. */
  const [historyTick, setHistoryTick] = createSignal(0);
  /* Rate view for cumulative families: a counter's ever-rising total says
     little — the per-scrape Δv/Δt is the interesting shape. */
  const [rateMode, setRateMode] = createSignal(false);
  const [chartAppend, setChartAppend] = createSignal<{
    signature: string;
    append: ChartAppend;
  } | null>(null);
  /* Hero cards append once per scrape through this shared x. */
  const [heroAppend, setHeroAppend] = createSignal<{
    x: number;
    y: Map<string, number>;
  } | null>(null);
  const [advOpen, setAdvOpen] = createSignal(false);
  const { copied, copy } = useCopied();

  const seriesKey = (family: string, s: MetricSample) =>
    `${family} ${s.name} ${s.labels}`;

  /* Detail row label: `_bucket{le="0.5"}`, `{route="/v1"}`, or "value". */
  const seriesLabel = (f: MetricFamily, s: MetricSample): string => {
    const suffix = s.name === f.name ? "" : s.name.slice(f.name.length);
    const labels = s.labels ? `{${s.labels}}` : "";
    return `${suffix}${labels}` || "value";
  };

  /* Buckets chart poorly — skip them, but _sum/_count still draw. */
  const chartable = (f: MetricFamily, s: MetricSample) =>
    f.type === "histogram" || f.type === "summary"
      ? !s.name.endsWith("_bucket")
      : true;

  /* Cumulative families: declared counters/histograms/summaries, plus the
     _total convention for metrics the exposition leaves untyped. */
  const rateable = (f: MetricFamily) =>
    f.type === "counter" ||
    f.type === "histogram" ||
    f.type === "summary" ||
    (f.samples.length > 0 &&
      f.samples.every(
        (s) => s.name === f.name && f.name.endsWith("_total")
      ));

  const chartSeries = (f: MetricFamily) =>
    f.samples
      .filter((s) => chartable(f, s))
      .map((s) => {
        const key = seriesKey(f.name, s);
        return {
          key,
          label: seriesLabel(f, s),
          unit: unitOf(f, s),
          pts: seriesHistory.get(key) ?? [],
        };
      })
      .filter((s) => s.pts.length >= 2)
      .slice(0, 8);

  const chartableCount = (f: MetricFamily) =>
    f.samples.filter((s) => chartable(f, s)).length;

  /* The next chart point for a series: its value, or its per-scrape rate
     in rate mode. */
  const appendY = (pts: { t: number; v: number }[], rate: boolean) => {
    const n = pts.length;
    if (!n) return NaN;
    if (!rate) return pts[n - 1].v;
    if (n < 2) return NaN;
    const dt = (pts[n - 1].t - pts[n - 2].t) / 1000;
    const dv = pts[n - 1].v - pts[n - 2].v;
    return Number.isFinite(dv) && dv >= 0 && dt > 0 ? dv / dt : NaN;
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
      const now = Date.now();
      /* A new engine instance invalidates every collected series. */
      const inst = parsed.families
        .find((f) => f.name === "richengine_info")
        ?.samples[0]?.labels.match(/instance="([^"]*)"/)?.[1];
      if (inst !== instanceId) {
        instanceId = inst;
        seriesHistory.clear();
      }
      for (const f of parsed.families)
        for (const s of f.samples) {
          const key = seriesKey(f.name, s);
          seriesHistory.set(
            key,
            [
              ...(seriesHistory.get(key) ?? []),
              { t: now, v: s.value },
            ].slice(-60)
          );
        }
      const selectedFamily = parsed.families.find(
        (f) => f.name === selected()
      );
      const series = selectedFamily ? chartSeries(selectedFamily) : [];
      const rate = rateMode() && !!selectedFamily && rateable(selectedFamily);
      setHistoryTick((t) => t + 1);
      setHeroAppend({
        x: now,
        y: new Map(
          HEROES.map((hero) => [hero.key, heroLatest(hero.key)] as const)
        ),
      });
      setChartAppend(
        selectedFamily && series.length
          ? {
              signature: series.map((s) => s.key).join("\n"),
              append: {
                x: now,
                y: series.map((s) => appendY(s.pts, rate)),
              },
            }
          : null
      );
    } catch (e) {
      setError(errorText(e));
      /* A failed scrape still advances the timeline: NaN samples draw as
         gaps instead of a line connecting across the outage. */
      const now = Date.now();
      for (const [key, ys] of seriesHistory)
        seriesHistory.set(key, [...ys, { t: now, v: NaN }].slice(-60));
      const f = selectedFamily();
      const series = f ? chartSeries(f) : [];
      setHistoryTick((t) => t + 1);
      setHeroAppend({
        x: now,
        y: new Map(HEROES.map((hero) => [hero.key, NaN] as const)),
      });
      setChartAppend(
        f && series.length
          ? {
              signature: series.map((s) => s.key).join("\n"),
              append: { x: now, y: series.map(() => NaN) },
            }
          : null
      );
    } finally {
      setFetching(false);
    }
  }

  /* 2s scrape poll while auto-refresh is on; freshness clock is shared. */
  onCleanup(poll(() => (auto() ? refresh() : Promise.resolve()), 2000));

  /* Typing a query opens Advanced once; closing it manually sticks. */
  let prevEmpty = true;
  createEffect(() => {
    const empty = !query().trim();
    if (!empty && prevEmpty) setAdvOpen(true);
    prevEmpty = empty;
  });

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

  const grouped = () => {
    const groups = new Map<string, MetricFamily[]>();
    for (const f of filtered()) {
      const section = sectionOf(f.name);
      const list = groups.get(section) ?? [];
      list.push(f);
      groups.set(section, list);
    }
    return SECTION_ORDER.filter((title) => groups.has(title)).map(
      (title) => ({ title, families: groups.get(title)! })
    );
  };

  const selectedFamily = () =>
    scrape()?.families.find((f) => f.name === selected()) ?? null;

  const selectFamily = (name: string) => {
    setSelected(selected() === name ? null : name);
    setRateMode(false);
  };

  /* Key indicators, read straight out of the latest scrape. */
  const metricValue = (name: string) =>
    scrape()?.families.find((f) => f.name === name)?.samples[0]?.value;

  /* Hero cards: the handful of metrics worth a chart by default.
     Everything else waits behind the Advanced fold. */
  const TTFT_KEY = "richengine_ttft_seconds";
  interface HeroDef {
    key: string;
    label: string;
    sub: string;
    unit: Unit;
  }
  const HEROES: HeroDef[] = [
    { key: "richengine_decode_tokens_per_second", label: "Decode", sub: "output throughput", unit: "none" },
    { key: "richengine_prefill_tokens_per_second", label: "Prefill", sub: "prompt throughput", unit: "none" },
    { key: "richengine_memory_current_bytes", label: "Memory", sub: "resident in use / limit", unit: "bytes" },
    { key: "richengine_scheduler_queued", label: "Queued", sub: "requests waiting", unit: "count" },
    { key: TTFT_KEY, label: "TTFT", sub: "avg time to first token", unit: "s" },
  ];

  const heroExists = (name: string) =>
    Boolean(scrape()?.families.some((f) => f.name === name));

  /* History for a hero's own-named series (buckets never reach it). */
  const heroPts = (name: string) => {
    const f = scrape()?.families.find((x) => x.name === name);
    const s = f?.samples.find((s) => chartable(f, s)) ?? f?.samples[0];
    if (!f || !s) return [];
    return seriesHistory.get(seriesKey(f.name, s)) ?? [];
  };

  /* Avg TTFT per interval: Δsum/Δcount between consecutive scrapes —
     the histogram's two monotone children cancel into a latency. */
  const ttftPts = () => {
    const f = scrape()?.families.find((x) => x.name === TTFT_KEY);
    const sumS = f?.samples.find((s) => s.name.endsWith("_sum"));
    const cntS = f?.samples.find((s) => s.name.endsWith("_count"));
    if (!f || !sumS || !cntS) return [];
    const sp = seriesHistory.get(seriesKey(f.name, sumS)) ?? [];
    const cp = seriesHistory.get(seriesKey(f.name, cntS)) ?? [];
    const pts: { t: number; v: number }[] = [];
    for (let i = 1, n = Math.min(sp.length, cp.length); i < n; i++) {
      const ds = sp[i].v - sp[i - 1].v;
      const dc = cp[i].v - cp[i - 1].v;
      const dt = sp[i].t - sp[i - 1].t;
      pts.push({
        t: sp[i].t,
        v:
          Number.isFinite(ds) && Number.isFinite(dc) && dc > 0 && dt > 0
            ? ds / dc
            : NaN,
      });
    }
    return pts;
  };

  const heroSeries = (name: string) =>
    name === TTFT_KEY ? ttftPts() : heroPts(name);

  const heroReady = (name: string) => {
    historyTick();
    return heroSeries(name).length >= 2;
  };

  const heroLatest = (name: string): number => {
    const pts = heroSeries(name);
    return pts.length ? pts[pts.length - 1].v : NaN;
  };

  const heroData = (name: string) => {
    const pts = heroSeries(name);
    return {
      x: pts.map((p) => p.t),
      series: [{ label: name, y: pts.map((p) => p.v) }],
    };
  };

  const heroAppendFor = (key: string): ChartAppend | null => {
    const h = heroAppend();
    const y = h?.y.get(key);
    return h && y !== undefined ? { x: h.x, y } : null;
  };

  const heroValue = (key: string): string => {
    if (key === "richengine_decode_tokens_per_second") return liveRate(key);
    if (key === "richengine_prefill_tokens_per_second") return liveRate(key);
    if (key === "richengine_memory_current_bytes") return liveMemory();
    if (key === "richengine_scheduler_queued")
      return fmtInt(metricValue(key));
    const v = heroLatest(TTFT_KEY);
    return Number.isFinite(v) ? fmtMs(v * 1000) : "—";
  };

  const liveRate = (name: string) => {
    const v = metricValue(name);
    return v === undefined ? "—" : `${fmtInt(v)} tok/s`;
  };

  const liveMemory = () => {
    const current = metricValue("richengine_memory_current_bytes");
    const limit = metricValue("richengine_memory_limit_bytes");
    if (current === undefined) return "—";
    return limit !== undefined
      ? `${fmtBytes(current)} / ${fmtBytes(limit)}`
      : fmtBytes(current);
  };

  /* Y-axis/tooltip formatter for the open family's chart: units resolve
     per call so selection and rate toggles stay right without a rebuild. */
  const fmtChartValue = (v: number, label?: string) => {
    const d = chartData();
    const unit =
      d?.units.get(label ?? d?.series[0]?.label ?? "") ?? "none";
    return `${fmtUnit(unit, v)}${rateMode() ? "/s" : ""}`;
  };

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

  /* The chart needs a reactive yes/no once any series reaches 2 points —
     tracked via historyTick. chartData itself stays untracked by the
     scrape clock: the data prop holds a mount/rebuild snapshot while the
     stream flows through append(). */
  const chartReady = () => {
    const f = selectedFamily();
    if (!f) return false;
    historyTick();
    return chartSeries(f).length > 0;
  };

  const chartData = () => {
    const f = selectedFamily();
    if (!f) return null;
    const series = chartSeries(f);
    if (!series.length) return null;
    /* Shared x axis = the union of every charted series' scrape times;
       points a series missed pad with NaN gaps instead of shifting. */
    const times = [
      ...new Set(series.flatMap((s) => s.pts.map((p) => p.t))),
    ].sort((a, b) => a - b);
    const rate = rateMode() && rateable(f);
    return {
      signature: series.map((s) => s.key).join("\n"),
      units: new Map(series.map((s) => [s.label, s.unit])),
      x: times,
      series: series.map(({ label, pts }) => {
        if (!rate) {
          const at = new Map(pts.map((p) => [p.t, p.v]));
          return { label, y: times.map((t) => at.get(t) ?? NaN) };
        }
        /* Δv/Δt between this series' own scrapes; counter resets and
           failed scrapes draw as gaps. */
        const y = new Map<number, number>();
        for (let i = 1; i < pts.length; i++) {
          const dt = (pts[i].t - pts[i - 1].t) / 1000;
          const dv = pts[i].v - pts[i - 1].v;
          if (Number.isFinite(dv) && dv >= 0 && dt > 0)
            y.set(pts[i].t, dv / dt);
        }
        return { label, y: times.map((t) => y.get(t) ?? NaN) };
      }),
    };
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

      <Show when={error()}>
        <p class="notice error">
          Metrics fetch failed
          {scrape() ? " — showing last good data" : ""}: {error()}
        </p>
      </Show>

      {/* Hero dashboard: the metrics worth a chart, no interaction needed.
          Each card's live value reads from the latest scrape; its chart
          mounts once two points exist and grows via heroAppend. */}
      <div class="hero-grid">
        <For each={HEROES}>
          {(hero) => (
            <Show when={heroExists(hero.key)}>
              <div class="card hero-card">
                <div class="card-label">{hero.label}</div>
                <div class="card-value">{heroValue(hero.key)}</div>
                <div class="card-sub">{hero.sub}</div>
                <Show when={heroReady(hero.key)}>
                  <div class="hero-chart">
                    <Chart
                      type="area"
                      height={88}
                      data={heroData(hero.key)}
                      append={heroAppendFor(hero.key)}
                      options={{
                        labels: {
                          y: (v: number) => fmtUnit(hero.unit, v),
                        },
                        window: 60,
                        xScale: "time",
                        zoom: { wheel: false },
                        tooltip: {
                          format: fmtTooltipRows((v) =>
                            fmtUnit(hero.unit, v)
                          ),
                        },
                        yMin: 0,
                      }}
                    />
                  </div>
                </Show>
              </div>
            </Show>
          )}
        </For>
      </div>

      <Show when={(scrape()?.errors ?? 0) > 0}>
        <p class="notice error">
          {fmtInt(scrape()?.errors)} malformed metric line
          {scrape()?.errors === 1 ? "" : "s"} skipped — see Advanced.
        </p>
      </Show>

      <Show when={scrape()} fallback={<p class="muted">Loading…</p>}>
        {/* Everything below the fold: meta counts, the filterable family
            explorer, and the raw exposition. */}
        <details
          class="advanced"
          open={advOpen()}
          onToggle={(e) => setAdvOpen(e.currentTarget.open)}
        >
          <summary title="Expand section">
            Advanced
            <span class="muted">
              {" "}
              — all {fmtInt(scrape()?.families.length)} families, filter,
              raw exposition
            </span>
          </summary>
          <div class="advanced-body">
            <div class="cards meta-cards">
              <div class="card">
                <div class="card-label">Metric families</div>
                <div class="card-value">
                  {fmtInt(scrape()?.families.length)}
                </div>
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

        {/* Family table grouped by subsystem: click a row for its series +
            history. */}
        <Show
          when={filtered().length > 0}
          fallback={
            <div class="panel">
              <div class="panel-row metric-empty">
                No metrics match “{query().trim()}”.
              </div>
            </div>
          }
        >
          {/* Index, not For: each scrape parses fresh family objects, so
              reference-keyed rows would remount — and the open detail's
              Chart with them — every 2s. Index keeps DOM by position and
              re-reads items through accessors. */}
          <Index each={grouped()}>
            {(group) => (
              <section class="section">
                <h2>{group().title}</h2>
                <div class="panel">
                  <Index each={group().families}>
                    {(f) => (
                      <>
                        <button
                          class={`panel-row metric-row${
                            selected() === f().name ? " sel" : ""
                          }`}
                          title={
                            selected() === f().name
                              ? "Hide detail"
                              : "Show series and history"
                          }
                          onClick={() => selectFamily(f().name)}
                        >
                          <span class="grow">{f().name}</span>
                          <span class="badge">{f().type}</span>
                          <span class="num help" title={f().help}>
                            {f().help}
                          </span>
                          <span class="num">{f().samples.length} series</span>
                          <span class="num metric-value">
                            {f().samples.length === 1
                              ? fmtUnit(unitOf(f()), f().samples[0].value)
                              : ""}
                          </span>
                        </button>
                        {/* Detail expands in place so it stays visible
                            wherever the row sits in a long family list. */}
                        <Show when={selected() === f().name}>
                          <div class="metric-detail">
                            <div class="panel-row">
                              <span class="grow mono">{f().name}</span>
                              <span class="badge">{f().type}</span>
                              <span class="num">
                                {f().samples.length} series
                              </span>
                              <Show when={rateable(f())}>
                                <div
                                  class="segmented"
                                  role="radiogroup"
                                  aria-label="Chart mode"
                                >
                                  <button
                                    type="button"
                                    role="radio"
                                    aria-checked={!rateMode()}
                                    classList={{ active: !rateMode() }}
                                    title="Cumulative totals"
                                    onClick={() => setRateMode(false)}
                                  >
                                    total
                                  </button>
                                  <button
                                    type="button"
                                    role="radio"
                                    aria-checked={rateMode()}
                                    classList={{ active: rateMode() }}
                                    title="Per-second rate between scrapes"
                                    onClick={() => setRateMode(true)}
                                  >
                                    rate
                                  </button>
                                </div>
                              </Show>
                              <button
                                class="btn small"
                                title="Close"
                                aria-label="Close metric detail"
                                onClick={() => setSelected(null)}
                              >
                                close
                              </button>
                            </div>
                            <Show when={f().help}>
                              <div class="panel-row">
                                <span class="grow muted">{f().help}</span>
                              </div>
                            </Show>
                            <Index each={f().samples}>
                              {(s) => (
                                <div class="panel-row series-row">
                                  <span class="grow mono">
                                    {seriesLabel(f(), s())}
                                  </span>
                                  <span class="num mono">
                                    {fmtUnit(unitOf(f(), s()), s().value)}
                                  </span>
                                </div>
                              )}
                            </Index>
                            <Show
                              when={chartReady()}
                              fallback={
                                <Show
                                  when={f().samples.some((s) =>
                                    chartable(f(), s)
                                  )}
                                >
                                  <div class="panel-row">
                                    <span class="grow muted">
                                      Collecting history — the chart appears
                                      after 2 scrapes.
                                    </span>
                                  </div>
                                </Show>
                              }
                            >
                              <Show when={chartData()}>
                                {(data) => (
                                  <div class="panel-row chart-row">
                                    <div>
                                      <span class="chart-label">
                                        {rateMode()
                                          ? "rate per second"
                                          : "history"}{" "}
                                        · last 60 scrapes
                                        {chartableCount(f()) > 8 &&
                                          ` · first 8 of ${chartableCount(
                                            f()
                                          )} series`}
                                      </span>
                                      <Chart
                                        type="area"
                                        height={140}
                                        data={{
                                          x: data().x,
                                          series: data().series,
                                        }}
                                        append={
                                          chartAppend()?.signature ===
                                          data().signature
                                            ? chartAppend()?.append
                                            : null
                                        }
                                        options={{
                                          legend: { position: "top" },
                                          labels: {
                                            y: (v) => fmtChartValue(v),
                                          },
                                          window: 60,
                                          xScale: "time",
                                          zoom: { wheel: false },
                                          tooltip: {
                                            format:
                                              fmtTooltipRows(fmtChartValue),
                                          },
                                          yMin: 0,
                                        }}
                                      />
                                    </div>
                                  </div>
                                )}
                              </Show>
                            </Show>
                          </div>
                        </Show>
                      </>
                    )}
                  </Index>
                </div>
              </section>
            )}
          </Index>
        </Show>

            {/* Raw exposition, for copy/paste debugging. */}
            <section class="section">
              <details class="metrics-raw" onToggle={collapseToggle}>
                <summary title="Expand section">Raw /metrics</summary>
                <pre class="metrics-pre">{rawView().text}</pre>
                <Show when={rawView().more > 0}>
                  <p class="muted">
                    … {fmtInt(rawView().more)} more lines — Copy raw has the
                    full exposition.
                  </p>
                </Show>
              </details>
            </section>
          </div>
        </details>
      </Show>
    </main>
  );
}

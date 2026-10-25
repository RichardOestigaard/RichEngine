import {
  createEffect,
  createMemo,
  createSignal,
  Show,
  untrack,
} from "solid-js";
import { Chart as RcChart } from "resolvcharts-solid";
import type {
  ChartInstance,
  ChartOptions,
  ChartType,
  HoverInfo,
} from "resolvcharts-solid";
import type { TooltipFormatter } from "resolvcharts";
import { theme } from "./theme";
import "./chart.css";

type ChartData = ChartOptions["data"];
type ExtraOptions = Omit<
  ChartOptions,
  "type" | "data" | "height" | "theme" | "animate"
>;
type ResolvedOptions = Omit<ChartOptions, "type" | "data">;

/* Shared ResolvCharts wrapper. CSS vars become literal theme tokens; option
   changes recreate the chart because only data can be patched in place. */
const osDarkQuery = matchMedia("(prefers-color-scheme: dark)");
const [osDark, setOsDark] = createSignal(osDarkQuery.matches);
osDarkQuery.addEventListener("change", (e) => setOsDark(e.matches));

const motionQuery = matchMedia("(prefers-reduced-motion: reduce)");
const [noMotion, setNoMotion] = createSignal(motionQuery.matches);
motionQuery.addEventListener("change", (e) => setNoMotion(e.matches));

/* Bumped a microtask after the theme signal fires — dataset.theme lands
   after the signal, so the vars are only fresh by then. */
const [paletteTick, setPaletteTick] = createSignal(0);

function palette() {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) =>
    css.getPropertyValue(name).trim() || fallback;
  return {
    accent: v("--accent", "#4f46e5"),
    muted: v("--muted", "#83837d"),
    border: v("--border", "#e7e7e3"),
    text: v("--text", "#191917"),
    composer: v("--composer", "#ffffff"),
    shadow: v("--shadow", "0 4px 14px rgba(0, 0, 0, 0.08)"),
    warn: v("--warn", "#b45309"),
    danger: v("--danger", "#dc2626"),
    ok: v("--ok", "#16a34a"),
  };
}

const seriesColors = (p: ReturnType<typeof palette>) => [
  p.accent,
  p.warn,
  p.danger,
  p.muted,
];

const appendTypes: ReadonlySet<ChartType> = new Set([
  "line",
  "area",
  "scatter",
  "bubble",
  "rangearea",
  "sparkline",
  "baseline",
]);

export interface ChartAppend {
  x: number;
  y: number | number[];
}

/* The fmt callback may take the series label for per-row units. */
export const fmtTooltipRows =
  (fmt: (v: number, label?: string) => string): TooltipFormatter =>
  (info: HoverInfo) => {
    if (info.kind === "point")
      return {
        rows: info.values.map((v) => ({
          color: v.color,
          label: v.label,
          value: fmt(v.y, v.label),
        })),
      };
    if (info.kind === "bar")
      return {
        rows: [
          {
            color: info.color,
            label: `${info.category} ${info.label}`.trim(),
            value: fmt(info.value, info.label),
          },
        ],
      };
    return null;
  };

export interface ChartProps {
  type: ChartType;
  data: ChartData;
  append?: ChartAppend | null;
  options?: ExtraOptions;
  height?: number;
}

export function Chart(props: ChartProps) {
  let chart: ChartInstance | undefined;
  let lastAppend = props.append;

  createEffect(() => {
    theme();
    osDark();
    queueMicrotask(() => setPaletteTick((t) => t + 1));
  });

  const options = createMemo<ResolvedOptions>(() => {
    paletteTick();
    const p = palette();
    return {
      height: props.height,
      accent: p.accent,
      curve: "smooth",
      animate: noMotion() ? false : 200,
      theme: {
        bg: p.composer,
        text: p.text,
        textMuted: p.muted,
        grid: p.border,
        axisBaseline: p.border,
        crosshair: p.muted,
        series: seriesColors(p),
        contextGray: p.muted,
        up: p.ok,
        down: p.danger,
        tooltipBg: p.composer,
        tooltipText: p.text,
        tooltipMuted: p.muted,
        tooltipBorder: p.border,
        tooltipShadow: p.shadow,
        fontFamily: getComputedStyle(document.body).fontFamily,
      },
      ...props.options,
    };
  });

  createEffect(() => {
    const pending = props.append;
    if (pending && appendTypes.has(props.type)) {
      if (pending !== lastAppend) chart?.append(pending.x, pending.y);
      lastAppend = pending;
      return;
    }
    lastAppend = pending ?? null;
  });

  return (
    <div class="chart-frame">
      <Show when={options()} keyed>
        {(opts) => {
          lastAppend = untrack(() => props.append);
          return (
            <RcChart
              {...opts}
              legend={opts.legend ?? false}
              ref={(c) => (chart = c ?? undefined)}
              type={props.type}
              data={props.data}
            />
          );
        }}
      </Show>
    </div>
  );
}

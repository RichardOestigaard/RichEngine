import { createEffect, createMemo, createSignal, onCleanup, onMount } from "solid-js";
import type ApexCharts from "apexcharts";
import type { ApexOptions } from "apexcharts";
import { theme } from "./theme";

/* Shared ApexCharts wrapper. Apex wants literal colors, so the page
   palette is read out of the CSS vars on <html>. Options rebuild when
   the theme (or the OS scheme while theme is "auto") changes, which
   triggers updateOptions — no remount needed.

   ApexCharts is imported dynamically so its ~600KB bundle stays a lazy
   chunk; and the instance is driven directly rather than through
   solid-apexcharts, whose deferred series/options effects can fire
   before the chart exists. */

const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const [osDark, setOsDark] = createSignal(darkQuery.matches);
darkQuery.addEventListener("change", (e) => setOsDark(e.matches));

const motionQuery = matchMedia("(prefers-reduced-motion: reduce)");
const [noMotion, setNoMotion] = createSignal(motionQuery.matches);
motionQuery.addEventListener("change", (e) => setNoMotion(e.matches));

/* Bumped a microtask after the theme signal fires — dataset.theme lands
   after the signal, so the vars are only fresh by then. */
const [paletteTick, setPaletteTick] = createSignal(0);

function dark(): boolean {
  const t = theme();
  return t === "dark" || t === "black" || (t === "auto" && osDark());
}

function palette() {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) =>
    css.getPropertyValue(name).trim() || fallback;
  return {
    accent: v("--accent", "#4f46e5"),
    muted: v("--muted", "#83837d"),
    border: v("--border", "#e7e7e3"),
    bg: v("--bg", "#fdfdfc"),
    text: v("--text", "#191917"),
    bubble: v("--bubble", "#efefec"),
    warn: v("--warn", "#b45309"),
    danger: v("--danger", "#dc2626"),
  };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/* Caller options win; nested keys (chart, xaxis, …) merge instead of
   clobbering the defaults. */
function mergeOptions(base: ApexOptions, extra?: ApexOptions): ApexOptions {
  if (!extra) return base;
  const out: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(extra)) {
    const existing = out[key];
    out[key] =
      isObject(existing) && isObject(value)
        ? mergeOptions(existing as ApexOptions, value as ApexOptions)
        : value;
  }
  return out as ApexOptions;
}

export interface ChartProps {
  type: NonNullable<NonNullable<ApexOptions["chart"]>["type"]>;
  series: NonNullable<ApexOptions["series"]>;
  options?: ApexOptions;
  height?: number | string;
}

export function Chart(props: ChartProps) {
  let el!: HTMLDivElement;
  let chart: ApexCharts | undefined;

  createEffect(() => {
    theme();
    osDark();
    queueMicrotask(() => setPaletteTick((t) => t + 1));
  });

  const options = createMemo<ApexOptions>(() => {
    paletteTick();
    const mode: "dark" | "light" = dark() ? "dark" : "light";
    const p = palette();
    const defaults: ApexOptions = {
      chart: {
        background: "transparent",
        foreColor: p.muted,
        fontFamily: "inherit",
        toolbar: { show: false },
        zoom: { enabled: false },
        animations: {
          enabled: !noMotion(),
          speed: 200,
          dynamicAnimation: { enabled: !noMotion(), speed: 200 },
        },
      },
      theme: { mode },
      colors: [p.accent, p.warn, p.danger, p.muted],
      grid: {
        borderColor: p.border,
        strokeDashArray: 3,
        padding: { top: 0, right: 6, bottom: 0, left: 6 },
      },
      dataLabels: { enabled: false },
      legend: { show: false },
      stroke: { curve: "smooth", width: 1.5 },
      fill: { opacity: 0.15 },
      markers: { size: 0 },
      tooltip: {
        theme: mode,
        style: { fontFamily: "inherit" },
        x: { show: false },
      },
      xaxis: {
        axisBorder: { show: false },
        axisTicks: { show: false },
        labels: { style: { colors: p.muted, fontFamily: "inherit" } },
        tooltip: { enabled: false },
      },
      yaxis: {
        labels: { style: { colors: p.muted, fontFamily: "inherit" } },
      },
    };
    return mergeOptions(defaults, props.options);
  });

  function apply() {
    // Read the reactive inputs before the early return — bailing first
    // would leave the effect with no tracked dependencies.
    const opts = options();
    const series = props.series;
    const motion = noMotion();
    const instance = chart;
    if (!instance) return;
    // updateOptions accepts a series key, so options and data flow
    // through the one call.
    void instance.updateOptions({ ...opts, series }, false, !motion);
  }

  onMount(async () => {
    const Apex = (await import("apexcharts")).default;
    // The page may have navigated away while the chunk loaded.
    if (!el.isConnected) return;
    const instance = new Apex(el, {
      ...options(),
      chart: {
        ...options().chart,
        type: props.type,
        height: props.height ?? 140,
        width: "100%",
      },
      series: props.series,
    });
    await instance.render();
    chart = instance;
    // Props that changed during the async init need one replay.
    apply();
  });

  /* Skipped until the instance exists; updates during init are caught
     by the apply() after render. */
  createEffect(apply);

  onCleanup(() => {
    void chart?.destroy();
    chart = undefined;
  });

  return <div ref={el} />;
}

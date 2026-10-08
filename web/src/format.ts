/* Number formatting shared by the telemetry pages: one convention
   everywhere — 1024-based IEC units — so the same byte count reads
   identically on the Models and Disk pages. */

export function fmtInt(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.round(value).toLocaleString("en-US")
    : "—";
}

export function fmtBytes(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = Math.max(0, value);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${unit === 0 || size >= 100 ? Math.round(size) : size.toFixed(1)} ${units[unit]}`;
}

export function fmtMs(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(1)} ms`;
}

export function pct(value: number | null | undefined): string {
  return typeof value !== "number" || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(1)}%`;
}

export function rate(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(1)
    : "—";
}

export function fmtTokens(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (value >= 1000) return `${Math.round(value / 1000)}k tokens`;
  return `${Math.round(value)} tokens`;
}

/* "2:41 PM" for a timestamp, today or any day. */
export function fmtClock(ts: number | undefined): string {
  if (!ts) return "";
  return new Date(ts).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

/* "just now", "3m ago", "2h ago", "Mar 4". */
export function fmtAgo(ts: number | undefined): string {
  if (!ts) return "";
  const seconds = (Date.now() - ts) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(ts).toLocaleDateString([], { month: "short", day: "numeric" });
}

/* Shared poll loop for the telemetry pages and the readiness pill: one
   timer, rescheduled only after the request settles so slow polls never
   pile up; a hidden tab keeps the cadence but skips the request, so a
   foregrounded page refreshes within one interval. Returns the stop
   function for onCleanup. */

export function poll(fn: () => Promise<void>, intervalMs: number): () => void {
  let timer: number | undefined;
  let stopped = false;
  const tick = async () => {
    try {
      if (!document.hidden) await fn();
    } finally {
      if (!stopped) timer = window.setTimeout(tick, intervalMs);
    }
  };
  void tick();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
  };
}

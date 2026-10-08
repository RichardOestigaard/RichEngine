import { createSignal } from "solid-js";
import { poll } from "./poll";

/* One shared 1s clock for "updated Ns ago" labels — poll() skips hidden
   tabs and the tick never hits the network. */
const [nowTick, setNowTick] = createSignal(Date.now());
poll(async () => {
  setNowTick(Date.now());
}, 1000);

/* Freshness for a polled endpoint: markOk() on each success, updatedAgo()
   renders "Ns ago"/"Nm ago", stale() once the last success is staleMs old. */
export function useFreshness(staleMs = 10_000) {
  const [lastOk, setLastOk] = createSignal<number | null>(null);
  const updatedAgo = () => {
    const t = lastOk();
    if (t === null) return null;
    const secs = Math.max(0, Math.round((nowTick() - t) / 1000));
    return secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m`;
  };
  const stale = () => {
    const t = lastOk();
    return t !== null && nowTick() - t > staleMs;
  };
  return { markOk: () => setLastOk(Date.now()), updatedAgo, stale };
}

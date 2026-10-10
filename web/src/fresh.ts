import { createSignal, getOwner, onCleanup } from "solid-js";
import { poll } from "./poll";

/* One shared 1s clock for "updated Ns ago" labels — poll() skips hidden
   tabs and the tick never hits the network. The clock starts on the first
   useFreshness consumer and stops when the last one unmounts, so pages
   that never show freshness don't wake the tab every second. */
const [nowTick, setNowTick] = createSignal(Date.now());
let consumers = 0;
let stopClock: (() => void) | undefined;

function acquireClock(): void {
  consumers += 1;
  if (!stopClock) {
    stopClock = poll(async () => {
      setNowTick(Date.now());
    }, 1000);
  }
  if (getOwner()) {
    onCleanup(() => {
      consumers -= 1;
      if (consumers === 0) {
        stopClock?.();
        stopClock = undefined;
      }
    });
  }
}

/* Freshness for a polled endpoint: markOk() on each success, updatedAgo()
   renders "Ns ago"/"Nm ago", stale() once the last success is staleMs old. */
export function useFreshness(staleMs = 10_000) {
  acquireClock();
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

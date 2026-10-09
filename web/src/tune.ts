import { createSignal } from "solid-js";
import { api } from "./api";
import { poll } from "./poll";
import { pushToast } from "./toast";

/* Shared tune-job state. The sweep runs as a server-side subprocess and
   outlives the Models page, so the signal and its completion toasts live at
   module scope, same as install.ts. */

export interface TuneJob {
  running?: boolean;
  model?: string;
  mode?: string;
  started?: number;
  done?: boolean;
  ok?: boolean;
  cancelled?: boolean;
  returncode?: number | null;
  kept?: string[] | null;
  /* "KNOB=value +4.1%" lines — every measured candidate's swing vs the
     baseline, biggest first. */
  results?: string[] | null;
  /* The payoff number — "decode +9.8% @b4" — when the record shows one. */
  headline?: string | null;
  /* The sweep's own error line on failure. */
  error?: string | null;
  candidates_done?: number;
  candidates_total?: number;
  tail?: string[];
}

const [tuneJob, setTuneJob] = createSignal<TuneJob | null>(null);
export { tuneJob, setTuneJob };

/* Subscribers notified when a tune finishes successfully — the Models page
   uses it to refresh /v1/models/available. Returns an unsubscribe. */
type DoneCallback = (job: TuneJob) => void;
const doneCallbacks = new Set<DoneCallback>();

export function onTuneDone(cb: DoneCallback): () => void {
  doneCallbacks.add(cb);
  return () => doneCallbacks.delete(cb);
}

export async function pollTune() {
  try {
    const next = await api<TuneJob>("/v1/models/tune");
    const prev = tuneJob();
    setTuneJob(next);
    if (prev?.running && next.done) {
      if (next.ok) {
        pushToast(
          "ok",
          next.kept?.length
            ? `Tuned ${next.model}: ${next.kept.join(", ")}.`
            : `Tuned ${next.model}: the defaults already won.`
        );
      } else if (next.cancelled) {
        pushToast("error", `Tune of ${next.model} cancelled; it resumes where it stopped.`);
      } else {
        pushToast(
          "error",
          `Tune of ${next.model} failed${
            next.returncode != null ? ` (exit ${next.returncode})` : ""
          }.`
        );
      }
      /* Any done state can have written winners — cancelled sweeps keep a
         partial record — so listeners refresh on every completion. */
      for (const cb of doneCallbacks) cb(next);
    }
  } catch {
    /* Best-effort like the rest of telemetry: keep the last reading. */
  }
}

/* Always-on 2s poll started at import; poll() already skips hidden tabs. */
poll(pollTune, 2000);

import { createSignal } from "solid-js";
import { api } from "./api";
import { poll } from "./poll";
import { pushToast } from "./toast";

/* Shared install-job state. The install runs as a server-side subprocess
   and outlives the Models page, so the signal and its completion toasts
   live at module scope: navigating away no longer loses the feedback. */

export interface InstallJob {
  running?: boolean;
  model?: string;
  started?: number;
  done?: boolean;
  ok?: boolean;
  cancelled?: boolean;
  returncode?: number | null;
  tail?: string[];
}

const [installJob, setInstallJob] = createSignal<InstallJob | null>(null);
export { installJob, setInstallJob };

/* Subscribers notified when an install finishes successfully — the Models
   page uses it to refresh /v1/models/available. Returns an unsubscribe. */
type DoneCallback = (job: InstallJob) => void;
const doneCallbacks = new Set<DoneCallback>();

export function onInstallDone(cb: DoneCallback): () => void {
  doneCallbacks.add(cb);
  return () => doneCallbacks.delete(cb);
}

export async function pollInstall() {
  try {
    const next = await api<InstallJob>("/v1/models/install");
    const prev = installJob();
    setInstallJob(next);
    if (prev?.running && next.done) {
      if (next.ok) {
        pushToast("ok", `Installed ${next.model}.`);
        for (const cb of doneCallbacks) cb(next);
      } else if (next.cancelled) {
        pushToast("error", `Install of ${next.model} cancelled.`);
      } else {
        pushToast(
          "error",
          `Install of ${next.model} failed${
            next.returncode != null ? ` (exit ${next.returncode})` : ""
          }.`
        );
      }
    }
  } catch {
    /* Best-effort like the rest of telemetry: keep the last reading. */
  }
}

/* Always-on 2s poll started at import; poll() already skips hidden tabs. */
poll(pollInstall, 2000);

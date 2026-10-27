import { createSignal } from "solid-js";
import { api, apiPost, storageSet } from "./api";
import { MODEL_KEY } from "./pages/chat/storage";
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
        /* Install → Load → Chat as one pipeline: a successful install
           loads its model and toasts a link to start chatting. */
        void autoLoad(next.model ?? "");
        for (const cb of doneCallbacks) cb(next);
      } else if (next.cancelled) {
        pushToast("error", `Install of ${next.model} cancelled.`, undefined, {
          label: "Retry",
          run: () => void retryInstall(next.model),
        });
      } else {
        pushToast(
          "error",
          `Install of ${next.model} failed${
            next.returncode != null ? ` (exit ${next.returncode})` : ""
          }.`,
          undefined,
          { label: "Retry", run: () => void retryInstall(next.model) }
        );
      }
    }
  } catch {
    /* Best-effort like the rest of telemetry: keep the last reading. */
  }
}

/* Retry with just the model id — the job record doesn't keep revision,
   draft or language_only. */
async function retryInstall(model?: string) {
  if (!model) return;
  try {
    setInstallJob(await apiPost<InstallJob>("/v1/models/install", { model }));
  } catch {
    /* The poll picks the state back up; a failed retry just toasts. */
  }
}

/* A finished install loads its resident and points the composer at it.
   Failure degrades to "go load it yourself" rather than silence. */
async function autoLoad(model: string) {
  if (!model) return;
  try {
    await apiPost("/v1/models/load", { model });
    storageSet(MODEL_KEY, model);
    window.dispatchEvent(
      new CustomEvent("richengine:model-changed", { detail: model })
    );
    pushToast("ok", `${model} ready — start chatting`, 10000, {
      label: "Open chat",
      run: () =>
        window.dispatchEvent(
          new CustomEvent("richengine:navigate", { detail: "/" })
        ),
    });
  } catch {
    pushToast("ok", `Installed ${model} — load it from Models`, 10000, {
      label: "Open Models",
      run: () =>
        window.dispatchEvent(
          new CustomEvent("richengine:navigate", { detail: "/models" })
        ),
    });
  }
}

/* Latest download percentage parsed from the tail — HF progress lines
   carry "NN%"; undefined when nothing has reported yet. */
export function installPct(): number | undefined {
  const tail = installJob()?.tail;
  for (let i = (tail?.length ?? 0) - 1; i >= 0 && i > (tail?.length ?? 0) - 40; i--) {
    const m = tail![i].match(/(\d{1,3})%/);
    if (m) return Math.min(100, Number(m[1]));
  }
  return undefined;
}

/* Download rate off the same progress line ("1.4 GB/s"). */
export function installRate(): string | undefined {
  const tail = installJob()?.tail;
  for (let i = (tail?.length ?? 0) - 1; i >= 0 && i > (tail?.length ?? 0) - 40; i--) {
    const m = tail![i].match(/([\d.]+\s*[GMK]B\/s)/);
    if (m) return m[1];
  }
  return undefined;
}

/* Always-on 2s poll started at import; poll() already skips hidden tabs. */
poll(pollInstall, 2000);

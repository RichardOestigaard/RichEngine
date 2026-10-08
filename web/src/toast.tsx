import { For } from "solid-js";
import type { JSX } from "solid-js";
import { Toast, Toaster, ToasterStore, useToaster } from "terracotta";
import "./toast.css";

interface ToastItem {
  kind: "ok" | "error";
  text: string;
}

/* Shared queue — pushToast works from anywhere, no context needed. */
const store = new ToasterStore<ToastItem>();

export function pushToast(kind: "ok" | "error", text: string, timeoutMs?: number): void {
  const id = store.create({ kind, text });
  const ms = timeoutMs ?? (kind === "error" ? 6000 : 4000);
  if (ms > 0) window.setTimeout(() => store.remove(id), ms);
}

export default function Toasts(): JSX.Element {
  const queue = useToaster(store);
  return (
    <Toaster class="toasts">
      <For each={queue()}>
        {(toast) => (
          <Toast class={`toast ${toast.data.kind}`}>
            <span class="toast-text">{toast.data.text}</span>
            <button
              class="toast-close"
              type="button"
              aria-label="Dismiss"
              onClick={() => store.remove(toast.id)}
            >
              ×
            </button>
          </Toast>
        )}
      </For>
    </Toaster>
  );
}

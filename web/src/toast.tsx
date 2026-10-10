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
const timers = new Map<string, number>();

function dismiss(id: string): void {
  const timer = timers.get(id);
  if (timer !== undefined) {
    window.clearTimeout(timer);
    timers.delete(id);
  }
  store.remove(id);
}

export function pushToast(kind: "ok" | "error", text: string, timeoutMs?: number): void {
  const id = store.create({ kind, text });
  const ms = timeoutMs ?? (kind === "error" ? 6000 : 4000);
  if (ms > 0) timers.set(id, window.setTimeout(() => dismiss(id), ms));
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
              onClick={() => dismiss(toast.id)}
            >
              ×
            </button>
          </Toast>
        )}
      </For>
    </Toaster>
  );
}

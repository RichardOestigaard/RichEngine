import { For, Show } from "solid-js";
import type { JSX } from "solid-js";
import { Toast, Toaster, ToasterStore, useToaster } from "terracotta";
import "./toast.css";

interface ToastItem {
  kind: "ok" | "error";
  text: string;
  /* Optional action button (Undo and friends); runs once, then dismisses. */
  action?: { label: string; run(): void };
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

export function pushToast(kind: "ok" | "error", text: string, timeoutMs?: number, action?: ToastItem["action"]): void {
  const id = store.create({ kind, text, ...(action ? { action } : {}) });
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
            <Show when={toast.data.action}>
              {(action) => (
                <button
                  class="toast-action"
                  type="button"
                  onClick={() => {
                    action().run();
                    dismiss(toast.id);
                  }}
                >
                  {action().label}
                </button>
              )}
            </Show>
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

import { createSignal, onCleanup } from "solid-js";

/* Click-to-copy with a brief "copied ✓" state per key, so rows sharing a
   page light up only where clicked. Non-strings are copied as JSON. */
export function useCopied(timeoutMs = 1400) {
  const [copied, setCopied] = createSignal<string | null>(null);
  let timer: number | undefined;
  onCleanup(() => window.clearTimeout(timer));

  async function copy(key: string, value: unknown) {
    const text =
      typeof value === "string" ? value : JSON.stringify(value, null, 2);
    try {
      await navigator.clipboard?.writeText(text);
    } catch {
      /* Clipboard unavailable: the brief state still confirms intent. */
    }
    setCopied(key);
    window.clearTimeout(timer);
    timer = window.setTimeout(() => setCopied(null), timeoutMs);
  }

  return { copied, copy };
}

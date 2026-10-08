import { createSignal } from "solid-js";

const KEY = "richengine-api-key";

function readKey(): string {
  try {
    return localStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
}

const [key, setKey] = createSignal(readKey());

export function apiKey(): string {
  return key();
}

export function setApiKey(value: string) {
  setKey(value);
  try {
    localStorage.setItem(KEY, value);
  } catch {
    /* Storage full or disabled: the key still works for this tab. */
  }
}

export function authorization(): Record<string, string> {
  const value = apiKey();
  return value ? { Authorization: `Bearer ${value}` } : {};
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function errorMessage(data: unknown, fallback: string): string {
  if (data && typeof data === "object") {
    const error = (data as Record<string, unknown>).error;
    if (error && typeof error === "object") {
      const message = (error as Record<string, unknown>).message;
      if (typeof message === "string" && message) return message;
    }
    const detail = (data as Record<string, unknown>).detail;
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
  }
  return fallback;
}

export function errorText(error: unknown): string {
  return error instanceof Error && error.message ? error.message : String(error);
}

/* True after the last API answer was a 401; cleared by the next successful
   request. TopBar reflects it in the status pill. */
const [unauthorized, setUnauthorized] = createSignal(false);
export { unauthorized };

/* For raw fetch callers that bypass api() (TopBar's /ready probe). */
export function noteAuthStatus(status: number) {
  setUnauthorized(status === 401);
}

export async function api<T = unknown>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 15_000
): Promise<T> {
  const method = (init.method ?? "GET").toUpperCase();
  const response = await fetch(path, {
    ...init,
    headers: { ...(init.headers ?? {}), ...authorization() },
    // A stalled GET must not stall a polling loop forever; POSTs (wipe,
    // dedupe) legitimately run long and keep their own lifetime.
    signal:
      init.signal ??
      (method === "GET" || method === "HEAD"
        ? AbortSignal.timeout(timeoutMs)
        : undefined),
  });
  const data = await response.json().catch(() => null);
  noteAuthStatus(response.status);
  if (!response.ok) throw new ApiError(response.status, errorMessage(data, response.statusText));
  return data as T;
}

export function apiPost<T = unknown>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/* An OpenAI-compatible curl for the chat endpoint, bearer header included
   only when a key is stored. */
export function curlExample(model: string): string {
  const body = JSON.stringify({
    model,
    messages: [{ role: "user", content: "Hello!" }],
  });
  return [
    `curl -sS ${location.origin}/v1/chat/completions \\`,
    `  -H 'Content-Type: application/json' \\`,
    ...(apiKey() ? [`  -H 'Authorization: Bearer ${apiKey()}' \\`] : []),
    `  -d '${body}'`,
  ].join("\n");
}

export function storageGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function storageSet(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

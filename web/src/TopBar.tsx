import { A, useLocation } from "@solidjs/router";
import { createEffect, createSignal, For, onCleanup, Show } from "solid-js";
import { api, apiKey, setApiKey, authorization, noteAuthStatus, unauthorized } from "./api";
import { decisionModel, setDecisionModel } from "./decision";
import { fmtBytes } from "./format";
import { installJob } from "./install";
import { poll } from "./poll";
import { togglePalette } from "./shortcuts";
import { cycleTheme, theme, THEME_LABELS } from "./theme";
import type { Theme } from "./theme";

/* /ready answers one of these; "down" is the frontend's own verdict for an
   unreachable server or an unrecognized state — "unloaded" is a reachable
   server with no model, not an outage. */
type ReadyState = "ready" | "loading" | "unloaded" | "down" | null;

interface MemoryUse {
  current: number;
  peak?: number;
  limit?: number;
  pressure?: string;
}

const ICON_PROPS = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  "stroke-width": 1.8,
  "stroke-linecap": "round",
} as const;

function ThemeIcon(props: { name: Theme }) {
  return (
    <Show when={props.name === "auto"} fallback={
      <Show when={props.name === "light"} fallback={
        <Show when={props.name === "dark"} fallback={
          /* black: a filled disc with a fine ring, read as "solid dark" */
          <svg {...ICON_PROPS}>
            <circle cx="12" cy="12" r="8" fill="currentColor" stroke="none" />
            <circle cx="12" cy="12" r="9.2" opacity=".45" />
          </svg>
        }>
          {/* dark: crescent moon */}
          <svg {...ICON_PROPS}>
            <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
          </svg>
        </Show>
      }>
        {/* light: sun */}
        <svg {...ICON_PROPS}>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      </Show>
    }>
      {/* auto: half-filled disc */}
      <svg {...ICON_PROPS}>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" stroke="none" />
      </svg>
    </Show>
  );
}

export default function TopBar() {
  const location = useLocation();
  const [ready, setReady] = createSignal<ReadyState>(null);
  const [servedModels, setServedModels] = createSignal<
    { id: string; decision?: boolean }[]
  >([]);
  const [memory, setMemory] = createSignal<MemoryUse | null>(null);

  async function checkReady() {
    try {
      const response = await fetch("/ready", {
        headers: authorization(),
        signal: AbortSignal.timeout(8000),
      });
      noteAuthStatus(response.status);
      const body = (await response.json().catch(() => null)) as {
        status?: string;
      } | null;
      const status = body?.status;
      setReady(
        response.ok
          ? "ready"
          : status === "loading" || status === "unloaded"
            ? status
            : "down"
      );
      if (response.ok) void loadServedModel();
      void loadStatus();
    } catch {
      setReady("down");
      setMemory(null);
    }
  }

  /* Model memory rides the ready poll: present only while an app is
     loaded, so the chip hides itself when nothing is serving. */
  async function loadStatus() {
    try {
      const res = await api<{
        memory_actual?: { current_bytes?: number; peak_bytes?: number };
        memory_governor?: { limit_bytes?: number };
        memory_pressure?: string;
      }>("/status");
      const current = res.memory_actual?.current_bytes;
      setMemory(
        typeof current === "number"
          ? {
              current,
              peak: res.memory_actual?.peak_bytes,
              limit: res.memory_governor?.limit_bytes,
              pressure: res.memory_pressure,
            }
          : null
      );
    } catch {
      setMemory(null);
    }
  }

  async function loadServedModel() {
    try {
      const res = await api<{ data?: { id?: string; decision?: boolean }[] }>(
        "/v1/models"
      );
      const list = (res.data ?? [])
        .filter((entry): entry is { id: string; decision?: boolean } =>
          typeof entry.id === "string"
        )
        .map((entry) => ({ id: entry.id, decision: entry.decision }));
      setServedModels(list);
      // Chat and Playground stay while any resident can generate text;
      // an all-decision pool hides them.
      setDecisionModel(list.length > 0 && list.every((m) => m.decision));
    } catch {
      setServedModels([]);
      setDecisionModel(false);
    }
  }

  /* Runs once on mount, then retries whenever the API key changes. */
  createEffect(() => {
    apiKey();
    void loadServedModel();
  });

  window.addEventListener("richengine:model-changed", loadServedModel);
  onCleanup(() =>
    window.removeEventListener("richengine:model-changed", loadServedModel)
  );

  onCleanup(poll(checkReady, 5000));

  const active = (path: string) =>
    path === "/" ? location.pathname === "/" : location.pathname.startsWith(path);

  return (
    <header class="topbar">
      <A href="/" class="brand" title="RichEngine — Chat">
        <img src="/favicon.ico" alt="" width="18" height="18" />
        RichEngine
      </A>
      <nav class="topnav" aria-label="Pages">
        <Show when={!decisionModel()}>
          <A href="/" class="navlink" classList={{ active: active("/") }} title="Chat (⌘1)">Chat</A>
        </Show>
        <A href="/models" class="navlink" classList={{ active: active("/models") }} title="Models (⌘2)">Models</A>
        <A href="/disk" class="navlink" classList={{ active: active("/disk") }} title="Disk (⌘3)">Disk</A>
        <Show when={!decisionModel()}>
          <A href="/playground" class="navlink" classList={{ active: active("/playground") }} title="Playground (⌘4)">Playground</A>
        </Show>
        <A href="/judge" class="navlink" classList={{ active: active("/judge") }} title="Judge (⌘5)">Judge</A>
        <A href="/metrics" class="navlink" classList={{ active: active("/metrics") }} title="Metrics (⌘6)">Metrics</A>
        <A href="/settings" class="navlink" classList={{ active: active("/settings") }} title="Settings (⌘7)">Settings</A>
      </nav>
      <button
        type="button"
        class="palette-trigger"
        title="Command palette (⌘K)"
        aria-label="Open command palette"
        onClick={() => togglePalette()}
      >
        <kbd class="kbd">⌘K</kbd>
      </button>
      <button
        type="button"
        class="theme-btn"
        title={`Theme: ${THEME_LABELS[theme()]} — click to change`}
        aria-label={`Theme: ${THEME_LABELS[theme()]}. Click to change theme.`}
        onClick={() => cycleTheme()}
      >
        <ThemeIcon name={theme()} />
      </button>
      <Show when={servedModels().length}>
        <A
          href="/models"
          class="top-model"
          title={`${servedModels()
            .map((m) => m.id)
            .join(", ")} — served models; open Models`}
        >
          <For each={servedModels()}>
            {(m, i) => (
              <>
                {i() > 0 && " · "}
                {m.id}
                <Show when={m.decision}>
                  {" "}
                  <span
                    class="badge"
                    title="Decision model — answers /v1/systemone and /v1/judgments in one forward pass; no text generation"
                  >
                    decision
                  </span>
                </Show>
              </>
            )}
          </For>
        </A>
      </Show>
      <Show when={installJob()?.running}>
        <A
          href="/models"
          class="top-install"
          title={`Installing ${installJob()?.model ?? "model"} — view progress`}
        >
          <span class="dot" aria-hidden="true" />
          installing…
        </A>
      </Show>
      <div class="topright">
        <Show when={memory()}>
          {(mem) => (
            <A
              href="/metrics"
              class="top-mem"
              classList={{
                warn:
                  mem().pressure === "warning" ||
                  mem().pressure === "critical",
              }}
              title={`Model memory in use — peak ${
                mem().peak !== undefined ? fmtBytes(mem().peak) : "—"
              }${
                mem().limit !== undefined
                  ? ` · limit ${fmtBytes(mem().limit)}`
                  : ""
              }${
                mem().pressure ? ` · pressure ${mem().pressure}` : ""
              } — open Metrics`}
            >
              {fmtBytes(mem().current)}
              {mem().limit !== undefined ? ` / ${fmtBytes(mem().limit)}` : ""}
            </A>
          )}
        </Show>
        <span
          class="status-pill"
          classList={{
            ok: ready() === "ready" && !unauthorized(),
            down: ready() === "down",
            warn: unauthorized() || ready() === "loading",
          }}
          title={
            unauthorized()
              ? "API key required or rejected — enter it at right"
              : ready() === null
                ? "Checking server"
                : ready() === "ready"
                  ? "Engine ready"
                  : ready() === "loading"
                    ? "Model is loading"
                    : ready() === "unloaded"
                      ? "Server up — no model loaded; open Models to load one"
                      : "Engine unavailable"
          }
        >
          {unauthorized()
            ? "auth"
            : ready() === null
              ? "…"
              : ready() === "unloaded"
                ? "no model"
                : ready()}
        </span>
        <input
          class="api-key"
          classList={{ invalid: unauthorized() }}
          type="password"
          autocomplete="off"
          spellcheck={false}
          placeholder="API key (if required)"
          aria-label="API key"
          aria-invalid={unauthorized()}
          title="Sent as a Bearer token with every API request. Stored in localStorage."
          value={apiKey()}
          onInput={(event) => setApiKey(event.currentTarget.value)}
        />
      </div>
    </header>
  );
}

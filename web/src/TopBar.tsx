import { A, useLocation } from "@solidjs/router";
import { createEffect, createSignal, onCleanup, Show } from "solid-js";
import { api, apiKey, setApiKey, authorization, noteAuthStatus, unauthorized } from "./api";
import { installJob } from "./install";
import { poll } from "./poll";
import { togglePalette } from "./shortcuts";
import { cycleTheme, theme, THEME_LABELS } from "./theme";
import type { Theme } from "./theme";

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
  const [ready, setReady] = createSignal<boolean | null>(null);
  const [servedModel, setServedModel] = createSignal<string | null>(null);

  async function checkReady() {
    try {
      const response = await fetch("/ready", {
        headers: authorization(),
        signal: AbortSignal.timeout(8000),
      });
      noteAuthStatus(response.status);
      setReady(response.ok);
      if (response.ok) void loadServedModel();
    } catch {
      setReady(false);
    }
  }

  async function loadServedModel() {
    try {
      const res = await api<{ data?: { id?: string }[] }>("/v1/models");
      setServedModel(res.data?.[0]?.id ?? null);
    } catch {
      setServedModel(null);
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
        <A href="/" class="navlink" classList={{ active: active("/") }} title="Chat (⌘1)">Chat</A>
        <A href="/models" class="navlink" classList={{ active: active("/models") }} title="Models (⌘2)">Models</A>
        <A href="/disk" class="navlink" classList={{ active: active("/disk") }} title="Disk (⌘3)">Disk</A>
        <A href="/playground" class="navlink" classList={{ active: active("/playground") }} title="Playground (⌘4)">Playground</A>
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
      <Show when={servedModel()}>
        <A href="/models" class="top-model" title={`${servedModel()} — served model; open Models`}>{servedModel()}</A>
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
        <span
          class="status-pill"
          classList={{
            ok: ready() === true && !unauthorized(),
            down: ready() === false,
            warn: unauthorized(),
          }}
          title={
            unauthorized()
              ? "API key required or rejected — enter it at right"
              : ready() === null
                ? "Checking server"
                : ready()
                  ? "Engine ready"
                  : "Engine unavailable"
          }
        >
          {unauthorized() ? "auth" : ready() === null ? "…" : ready() ? "ready" : "down"}
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

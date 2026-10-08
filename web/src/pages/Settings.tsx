import { createSignal, For, onMount, Show } from "solid-js";
import {
  api,
  apiKey,
  setApiKey,
  authorization,
  errorText,
  noteAuthStatus,
  storageGet,
  storageSet,
} from "../api";
import Confirm from "../Confirm";
import { fmtBytes } from "../format";
import { setTheme, theme, THEME_LABELS } from "../theme";
import type { Theme } from "../theme";
import { pushToast } from "../toast";
import {
  conversations,
  deleteConversation,
  EFFORT_KEY,
  MODEL_KEY,
  SIDEBAR_KEY,
  STORE_KEY,
  SYSTEM_KEY,
} from "./chat/storage";
import {
  SEARCH_PROVIDER_LABELS,
  SEARCH_PROVIDER_NEEDS_KEY,
  SEARCH_PROVIDERS,
  searchApiKey,
  searchInstance,
  searchProvider,
  setSearchApiKey,
  setSearchInstance,
  setSearchProvider,
} from "./chat/tools";
import type { SearchProvider } from "./chat/tools";
import "./settings.css";

/* Settings: browser-local preferences (theme, API key, chat defaults) plus
   the localStorage/sessionStorage the chat page leaves behind. */

const THEMES: Theme[] = ["auto", "light", "dark", "black"];

const THEME_TITLES: Record<Theme, string> = {
  auto: "Follow the OS theme",
  light: "Light theme",
  dark: "Dark theme",
  black: "True-black theme",
};

const EFFORTS: { value: string; label: string }[] = [
  { value: "", label: "Default" },
  { value: "xhigh", label: "XHigh" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
  { value: "none", label: "Off" },
];

const DRAFT_PREFIX = "richengine:draft:";

/* theme.ts and api.ts keep their storage keys private; both re-read them
   on load, so removing by name here is safe. */
const RESET_KEYS = [
  "richengine-theme",
  "richengine-api-key",
  "richengine:chat-tools",
  "richengine:search-provider",
  "richengine:search-api-key",
  "richengine:search-instance",
  MODEL_KEY,
  EFFORT_KEY,
  SYSTEM_KEY,
  SIDEBAR_KEY,
];

interface ModelsResponse {
  data?: { id?: string }[];
}

interface InstanceInfo {
  id?: string;
  pid?: number;
  model?: string | null;
  host?: string;
  port?: number;
  /* Epoch seconds. */
  started_at?: number;
}

/* The destructive action waiting on AlertDialog confirmation. */
type ConfirmTarget = "chats" | "drafts" | "reset";

function chatCount(): number {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(STORE_KEY) ?? "null");
    return Array.isArray(stored) ? stored.length : 0;
  } catch {
    return 0;
  }
}

function draftCount(): number {
  try {
    let count = 0;
    for (let i = 0; i < sessionStorage.length; i++)
      if (sessionStorage.key(i)?.startsWith(DRAFT_PREFIX)) count++;
    return count;
  } catch {
    return 0;
  }
}

export default function Settings() {
  const savedSidebar = storageGet(SIDEBAR_KEY);
  const [showKey, setShowKey] = createSignal(false);
  const [keyTest, setKeyTest] = createSignal<"idle" | "busy" | "ok" | "err">("idle");
  const [keyTestText, setKeyTestText] = createSignal("");
  const [modelIds, setModelIds] = createSignal<string[] | null>(null);
  const [modelError, setModelError] = createSignal("");
  const [modelDefault, setModelDefault] = createSignal(storageGet(MODEL_KEY) ?? "");
  const savedEffort = storageGet(EFFORT_KEY) ?? "";
  const [effort, setEffort] = createSignal(
    EFFORTS.some((entry) => entry.value === savedEffort) ? savedEffort : ""
  );
  const [system, setSystem] = createSignal(storageGet(SYSTEM_KEY) ?? "");
  /* Same default as Chat: open unless the narrow-screen drawer applies. */
  const [sidebarOpen, setSidebarOpen] = createSignal(
    savedSidebar !== null
      ? savedSidebar === "1"
      : !window.matchMedia("(max-width: 720px)").matches
  );
  const [chats, setChats] = createSignal(chatCount());
  const [drafts, setDrafts] = createSignal(draftCount());
  const [estimate, setEstimate] = createSignal<StorageEstimate | null>(null);
  const [estimateFailed, setEstimateFailed] = createSignal(false);
  const [instance, setInstance] = createSignal<InstanceInfo | null>(null);
  const [confirm, setConfirm] = createSignal<ConfirmTarget | null>(null);

  onMount(() => {
    api<ModelsResponse>("/v1/models")
      .then((res) =>
        setModelIds((res.data ?? []).flatMap((m) => (m.id ? [m.id] : [])))
      )
      .catch((e) => setModelError(errorText(e)));
    /* estimate() is missing in some browsers — degrade to a dash. */
    if (navigator.storage?.estimate)
      navigator.storage.estimate().then(setEstimate, () => setEstimateFailed(true));
    else setEstimateFailed(true);
    api<{ instance?: InstanceInfo }>("/status")
      .then((res) => setInstance(res.instance ?? null))
      .catch(() => {
        /* Best-effort: the About rows fall back to unavailable. */
      });
  });

  /* A stored id the server no longer lists stays selectable so the real
     preference is visible instead of silently flipping to default. */
  const modelOptions = () => {
    const ids = modelIds() ?? [];
    const current = modelDefault();
    return current && !ids.includes(current) ? [current, ...ids] : ids;
  };

  const storageText = () => {
    const est = estimate();
    if (!est) return estimateFailed() ? "unavailable" : "…";
    const usage = fmtBytes(est.usage);
    return typeof est.quota === "number" && est.quota > 0
      ? `${usage} of ${fmtBytes(est.quota)}`
      : usage;
  };

  const uptime = () => {
    const started = instance()?.started_at;
    if (!started) return "";
    const secs = Math.max(0, Math.round(Date.now() / 1000 - started));
    const days = Math.floor(secs / 86400);
    const hours = Math.floor((secs % 86400) / 3600);
    const mins = Math.floor((secs % 3600) / 60);
    if (days) return `${days}d ${hours}h`;
    if (hours) return `${hours}h ${mins}m`;
    if (mins) return `${mins}m ${secs % 60}s`;
    return `${secs}s`;
  };

  /* Probe /v1/models, not /ready — the server exempts /ready from auth, so
     it can never reject a bad key. noteAuthStatus keeps the status pill
     honest on a 401. */
  async function testConnection() {
    if (keyTest() === "busy") return;
    setKeyTest("busy");
    try {
      const response = await fetch("/v1/models", {
        headers: authorization(),
        signal: AbortSignal.timeout(8000),
      });
      noteAuthStatus(response.status);
      if (response.ok) {
        setKeyTest("ok");
        setKeyTestText(apiKey() ? "key accepted" : "reachable — no key required");
      } else if (response.status === 401) {
        setKeyTest("err");
        setKeyTestText("key rejected");
        pushToast("error", "Connection test failed: API key rejected (HTTP 401)");
      } else {
        setKeyTest("err");
        setKeyTestText(`HTTP ${response.status}`);
        pushToast("error", `Connection test failed: HTTP ${response.status} ${response.statusText}`.trim());
      }
    } catch (error) {
      setKeyTest("err");
      setKeyTestText(errorText(error));
      pushToast("error", `Connection test failed: ${errorText(error)}`);
    }
  }

  function onModelChange(value: string) {
    setModelDefault(value);
    storageSet(MODEL_KEY, value);
    /* Open pages follow the stored pick. */
    window.dispatchEvent(new CustomEvent("richengine:model-changed", { detail: value }));
  }

  function clearChats() {
    /* Keep the live signal in step so an already-open Chat page doesn't
       repersist the deleted list on its next save. */
    for (const item of conversations()) deleteConversation(item.id);
    try {
      localStorage.removeItem(STORE_KEY);
    } catch {
      /* Storage disabled. */
    }
    window.dispatchEvent(new CustomEvent("richengine:chats-cleared"));
    setChats(0);
    pushToast("ok", "Cleared all saved chats.");
  }

  function clearDrafts() {
    try {
      /* Collect first: removing keys shifts the index space mid-loop. */
      const keys: string[] = [];
      for (let i = 0; i < sessionStorage.length; i++) {
        const key = sessionStorage.key(i);
        if (key?.startsWith(DRAFT_PREFIX)) keys.push(key);
      }
      for (const key of keys) sessionStorage.removeItem(key);
    } catch {
      /* Storage disabled. */
    }
    setDrafts(0);
    pushToast("ok", "Cleared unsent drafts.");
  }

  function resetSettings() {
    try {
      for (const key of RESET_KEYS) localStorage.removeItem(key);
    } catch {
      /* Storage disabled: the reload still drops in-memory state. */
    }
    location.reload();
  }

  /* Runs whichever action the confirm dialog is currently describing. */
  function runConfirmed() {
    const target = confirm();
    setConfirm(null);
    if (target === "chats") clearChats();
    else if (target === "drafts") clearDrafts();
    else if (target === "reset") resetSettings();
  }

  const confirmTitle = () =>
    confirm() === "chats"
      ? "Clear all chats?"
      : confirm() === "drafts"
        ? "Clear drafts?"
        : "Reset all settings?";

  const confirmLabel = () => (confirm() === "reset" ? "Reset and reload" : "Clear");

  const confirmDesc = () => {
    const target = confirm();
    if (target === "chats")
      return `This permanently deletes ${chats() === 1 ? "the saved conversation" : `all ${chats()} saved conversations`} stored in this browser. This cannot be undone.`;
    if (target === "drafts")
      return `This deletes ${drafts()} unsent composer draft${drafts() === 1 ? "" : "s"} held in this tab session.`;
    return "This resets the theme, API key, model, thinking effort, system prompt and sidebar preference, then reloads the page. Saved chats and drafts stay.";
  };

  return (
    <main class="page settings-page">
      <h1>Settings</h1>
      <p class="lede">Preferences and browser data for this UI.</p>

      {/* 1. Appearance */}
      <section class="section">
        <h2>Appearance</h2>
        <div class="panel">
          <div class="panel-row">
            <span class="grow">Theme</span>
            <div class="segmented" role="radiogroup" aria-label="Theme">
              <For each={THEMES}>
                {(value) => (
                  <button
                    type="button"
                    role="radio"
                    aria-checked={theme() === value}
                    title={THEME_TITLES[value]}
                    classList={{ active: theme() === value }}
                    onClick={() => setTheme(value)}
                  >
                    {THEME_LABELS[value]}
                  </button>
                )}
              </For>
            </div>
          </div>
        </div>
      </section>

      {/* 2. Connection */}
      <section class="section">
        <h2>Connection</h2>
        <div class="panel">
          <div class="panel-row">
            <span class="grow">API key</span>
            <input
              class="settings-input"
              type={showKey() ? "text" : "password"}
              autocomplete="off"
              spellcheck={false}
              placeholder="API key (if required)"
              aria-label="API key"
              value={apiKey()}
              onInput={(event) => setApiKey(event.currentTarget.value)}
            />
            <button
              type="button"
              class="btn small"
              title={showKey() ? "Hide the key" : "Reveal the stored key"}
              aria-label={showKey() ? "Hide the API key" : "Reveal the API key"}
              onClick={() => setShowKey(!showKey())}
            >
              {showKey() ? "Hide" : "Show"}
            </button>
          </div>
          <div class="panel-row wrap">
            <span class="grow field-note">
              Sent as Bearer token with every request · stored in localStorage
            </span>
            <Show when={keyTest() === "ok"}>
              <span class="badge ok">{keyTestText()}</span>
            </Show>
            <Show when={keyTest() === "err"}>
              <span class="badge err">{keyTestText()}</span>
            </Show>
            <button
              type="button"
              class="btn small"
              title="Verify the key against /v1/models"
              disabled={keyTest() === "busy"}
              onClick={() => void testConnection()}
            >
              {keyTest() === "busy" ? "Testing…" : "Test"}
            </button>
            <button
              type="button"
              class="btn small"
              title="Remove the stored key"
              disabled={!apiKey()}
              onClick={() => setApiKey("")}
            >
              Clear
            </button>
          </div>
        </div>
      </section>

      {/* 3. Chat defaults */}
      <section class="section">
        <h2>Chat defaults</h2>
        <Show when={modelError()} keyed>
          {(text: string) => <p class="notice error">Model list fetch failed: {text}</p>}
        </Show>
        <div class="panel">
          <div class="panel-row">
            <span class="grow">Model</span>
            <select
              class="settings-select"
              aria-label="Default model"
              value={modelDefault()}
              onChange={(event) => onModelChange(event.currentTarget.value)}
            >
              <option value="">Server default</option>
              <For each={modelOptions()}>
                {(id) => <option value={id}>{id}</option>}
              </For>
            </select>
          </div>
          <div class="panel-row">
            <span class="grow">Thinking effort</span>
            <select
              class="settings-select"
              aria-label="Thinking effort"
              value={effort()}
              onChange={(event) => {
                setEffort(event.currentTarget.value);
                storageSet(EFFORT_KEY, event.currentTarget.value);
              }}
            >
              <For each={EFFORTS}>
                {(entry) => <option value={entry.value}>{entry.label}</option>}
              </For>
            </select>
          </div>
          <div class="panel-row col">
            <span>System prompt</span>
            <textarea
              class="settings-area"
              rows={3}
              placeholder="Optional instructions for every message…"
              aria-label="System prompt"
              value={system()}
              onInput={(event) => {
                setSystem(event.currentTarget.value);
                storageSet(SYSTEM_KEY, event.currentTarget.value);
              }}
            />
            <span class="field-note">Also editable in the chat sidebar.</span>
          </div>
          <div class="panel-row">
            <label
              class="settings-check grow"
              title="Open the conversation sidebar on wide screens"
            >
              <input
                type="checkbox"
                checked={sidebarOpen()}
                onChange={(event) => {
                  setSidebarOpen(event.currentTarget.checked);
                  storageSet(SIDEBAR_KEY, event.currentTarget.checked ? "1" : "0");
                }}
              />
              Sidebar open by default
            </label>
          </div>
        </div>
      </section>

      {/* 4. Web search — the chat page's web_search tool provider */}
      <section class="section">
        <h2>Web search</h2>
        <div class="panel">
          <div class="panel-row">
            <span class="grow">Search provider</span>
            <select
              class="settings-select"
              aria-label="Search provider"
              value={searchProvider()}
              onChange={(event) =>
                setSearchProvider(event.currentTarget.value as SearchProvider)
              }
            >
              <For each={SEARCH_PROVIDERS}>
                {(id) => (
                  <option value={id}>{SEARCH_PROVIDER_LABELS[id]}</option>
                )}
              </For>
            </select>
          </div>
          <Show when={SEARCH_PROVIDER_NEEDS_KEY[searchProvider()]}>
            <div class="panel-row">
              <span class="grow">Provider API key</span>
              <input
                class="settings-input"
                type="password"
                autocomplete="off"
                spellcheck={false}
                placeholder="API key"
                aria-label="Search provider API key"
                value={searchApiKey()}
                onInput={(event) => setSearchApiKey(event.currentTarget.value)}
              />
            </div>
            <div class="panel-row wrap">
              <span class="grow field-note">
                Sent only to this local server and the search provider · stored in localStorage
              </span>
            </div>
          </Show>
          <Show when={searchProvider() === "searxng"}>
            <div class="panel-row">
              <span class="grow">Instance URL</span>
              <input
                class="settings-input"
                type="text"
                autocomplete="off"
                spellcheck={false}
                placeholder="https://searx.example.com"
                aria-label="SearXNG instance URL"
                value={searchInstance()}
                onInput={(event) =>
                  setSearchInstance(event.currentTarget.value)
                }
              />
            </div>
          </Show>
          <div class="panel-row wrap">
            <span class="grow field-note">
              The chat page's web_search tool queries this provider.
            </span>
          </div>
        </div>
      </section>

      {/* 5. Local data */}
      <section class="section">
        <h2>Local data</h2>
        <div class="panel">
          <div class="panel-row">
            <span class="grow">Browser storage</span>
            <span class="num">{storageText()}</span>
          </div>
          <div class="panel-row">
            <span class="grow">Saved chats</span>
            <span class="num">{chats()}</span>
          </div>
          <div class="panel-row">
            <span class="grow">Unsent drafts</span>
            <span class="num">{drafts()}</span>
          </div>
        </div>
        <div class="panel danger-zone">
          <div class="panel-row wrap">
            <span class="grow muted">
              Delete every saved conversation stored in this browser.
            </span>
            <button
              type="button"
              class="btn small danger"
              title="Delete all stored conversations from this browser"
              disabled={!chats()}
              onClick={() => setConfirm("chats")}
            >
              Clear all chats
            </button>
          </div>
          <div class="panel-row wrap">
            <span class="grow muted">
              Delete unsubmitted composer drafts held for this tab session.
            </span>
            <button
              type="button"
              class="btn small danger"
              title="Delete unsent composer drafts"
              disabled={!drafts()}
              onClick={() => setConfirm("drafts")}
            >
              Clear drafts
            </button>
          </div>
          <div class="panel-row wrap">
            <span class="grow muted">
              Reset theme, API key and chat defaults, then reload.
            </span>
            <button
              type="button"
              class="btn small danger"
              title="Clear every saved preference and reload"
              onClick={() => setConfirm("reset")}
            >
              Reset all settings
            </button>
          </div>
        </div>
      </section>

      {/* 5. About */}
      <section class="section">
        <h2>About</h2>
        <div class="panel">
          <div class="panel-row">
            <span class="grow">Server</span>
            <span class="num mono">{location.origin}</span>
          </div>
          <Show
            when={instance()}
            fallback={
              <div class="panel-row">
                <span class="grow muted">Instance details unavailable</span>
              </div>
            }
          >
            {(info) => (
              <>
                <div class="panel-row">
                  <span class="grow">Instance</span>
                  <span class="num mono">
                    {info().id}
                    <Show when={info().pid !== undefined}> · pid {info().pid}</Show>
                  </span>
                </div>
                <div class="panel-row">
                  <span class="grow">Listening</span>
                  <span class="num mono">
                    {info().host}:{info().port}
                  </span>
                </div>
                <Show when={info().model}>
                  {(model) => (
                    <div class="panel-row">
                      <span class="grow">Model</span>
                      <span class="num mono">{model()}</span>
                    </div>
                  )}
                </Show>
                <Show when={uptime()}>
                  <div class="panel-row">
                    <span class="grow">Uptime</span>
                    <span class="num">{uptime()}</span>
                  </div>
                </Show>
              </>
            )}
          </Show>
        </div>
      </section>

      <Confirm
        open={confirm() !== null}
        title={confirmTitle()}
        confirmLabel={confirmLabel()}
        onConfirm={runConfirmed}
        onClose={() => setConfirm(null)}
      >
        {confirmDesc()}
      </Confirm>
    </main>
  );
}

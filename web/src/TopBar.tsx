import { A, useLocation, useNavigate } from "@solidjs/router";
import { createEffect, createSignal, For, onCleanup, Show } from "solid-js";
import { api, apiKey, authorization, noteAuthStatus, storageSet, unauthorized } from "./api";
import { decisionModel, setDecisionModel } from "./decision";
import { fmtBytes, fmtTokens } from "./format";
import { installJob, installPct } from "./install";
import { MODEL_KEY } from "./pages/chat/storage";
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

/* One /v1/models entry: a served name. Entries with `root` are aliases of
   the resident whose first name is root; entries without it are that
   resident's primary name. */
interface ServedEntry {
  id: string;
  root?: string;
  decision?: boolean;
  vision?: boolean;
  context_length?: number;
}

/* One row of /status.residents, keyed by the loaded model id — always one
   of the resident group's served names. */
interface ResidentRow {
  state?: string;
  error?: string;
  context_length?: number;
  vision?: boolean;
  decision?: boolean;
  memory_bytes?: number;
  memory_peak_bytes?: number;
}

/* /status.pool: resident capacity, present only on pooled servers. */
interface PoolInfo {
  max_residents?: number;
  slots?: number;
  free?: number;
}

/* A resident group: one engine process plus every name it answers to. */
interface ResidentGroup {
  id: string;
  names: string[];
  decision?: boolean;
  vision?: boolean;
  context?: number;
  state?: string;
  error?: string;
  memoryBytes?: number;
  memoryPeak?: number;
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
  const navigate = useNavigate();
  const [ready, setReady] = createSignal<ReadyState>(null);
  const [servedModels, setServedModels] = createSignal<ServedEntry[]>([]);
  const [residentsMap, setResidentsMap] = createSignal<
    Record<string, ResidentRow>
  >({});
  const [poolInfo, setPoolInfo] = createSignal<PoolInfo | null>(null);
  const [memory, setMemory] = createSignal<MemoryUse | null>(null);
  /* /ready's residents tally counts hosts by lifecycle — including a
     loading resident that has no app and no /v1/models entry yet. */
  const [readyCounts, setReadyCounts] = createSignal<{
    total?: number;
    loading?: number;
    failed?: number;
  } | null>(null);
  const [menuOpen, setMenuOpen] = createSignal(false);
  let menuEl: HTMLDivElement | undefined;
  let navEl: HTMLElement | undefined;

  /* The sliding pill under the nav tracks the active link's box; CSS vars
     carry the position so the transition lives in styles.css. */
  function placeInd() {
    if (!navEl) return;
    const link = navEl.querySelector<HTMLElement>(".navlink.active");
    if (!link) {
      navEl.style.setProperty("--ind-o", "0");
      return;
    }
    navEl.style.setProperty("--ind-x", `${link.offsetLeft}px`);
    navEl.style.setProperty("--ind-w", `${link.offsetWidth}px`);
    navEl.style.setProperty("--ind-o", "1");
  }

  createEffect(() => {
    // Reposition after route swaps and after the Chat link mounts/hides.
    void location.pathname;
    void decisionModel();
    queueMicrotask(placeInd);
  });
  window.addEventListener("resize", placeInd);
  onCleanup(() => window.removeEventListener("resize", placeInd));

  async function checkReady() {
    try {
      const response = await fetch("/ready", {
        headers: authorization(),
        signal: AbortSignal.timeout(8000),
      });
      noteAuthStatus(response.status);
      const body = (await response.json().catch(() => null)) as {
        status?: string;
        residents?: { total?: number; loading?: number; failed?: number };
      } | null;
      const status = body?.status;
      setReadyCounts(body?.residents ?? null);
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
      setResidentsMap({});
      setPoolInfo(null);
    }
  }

  /* Resident state and pool capacity ride the ready poll: /status reports
     the default resident's memory plus a residents map and a pool summary
     on pooled servers. */
  async function loadStatus() {
    try {
      const res = await api<{
        memory_actual?: { current_bytes?: number; peak_bytes?: number };
        memory_governor?: { limit_bytes?: number };
        memory_pressure?: string;
        residents?: Record<string, ResidentRow>;
        pool?: PoolInfo;
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
      setResidentsMap(res.residents ?? {});
      setPoolInfo(res.pool ?? null);
    } catch {
      setMemory(null);
      setResidentsMap({});
      setPoolInfo(null);
    }
  }

  async function loadServedModel() {
    try {
      const res = await api<{ data?: ServedEntry[] }>("/v1/models");
      const list = (res.data ?? []).filter(
        (entry): entry is ServedEntry => typeof entry.id === "string"
      );
      setServedModels(list);
      /* Chat and Playground stay while any resident can generate text; a
         pool where every resident is a decision model hides them. Aliases
         (entries with root) repeat their resident's flag — count only
         primaries. */
      const primaries = list.filter((m) => !m.root);
      setDecisionModel(
        primaries.length > 0 && primaries.every((m) => m.decision)
      );
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

  /* A pointerdown outside the open menu closes it; clicks inside do not.
     Escape closes it too, in the capture phase. */
  createEffect(() => {
    if (!menuOpen()) return;
    const onDown = (event: PointerEvent) => {
      if (!menuEl?.contains(event.target as Node)) setMenuOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        setMenuOpen(false);
      }
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey, true);
    onCleanup(() => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey, true);
    });
  });

  /* /v1/models order is default-resident-first; the residents map joins by
     the loaded model id, which is always one of the group's names. */
  const residents = (): ResidentGroup[] => {
    const map = residentsMap();
    const groups: ResidentGroup[] = [];
    for (const entry of servedModels()) {
      const key = entry.root ?? entry.id;
      let group = groups.find((g) => g.id === key);
      if (!group) {
        group = {
          id: key,
          names: [],
          decision: entry.decision,
          vision: entry.vision,
          context: entry.context_length,
        };
        groups.push(group);
      }
      group.names.push(entry.id);
      const row = map[entry.id];
      if (row) {
        group.state = row.state;
        group.error = row.error;
        group.context = row.context_length ?? group.context;
        group.vision = row.vision ?? group.vision;
        group.decision = row.decision ?? group.decision;
        group.memoryBytes = row.memory_bytes;
        group.memoryPeak = row.memory_peak_bytes;
      }
    }
    return groups;
  };

  /* Worst-case resident state drives the dot: failed, then loading, then
     ready. A served name with no residents row counts as loaded; the
     /ready tally adds app-less loading and failed hosts the groups can't
     name. */
  const aggregate = () => {
    const groups = residents();
    const counts = readyCounts();
    if (!groups.length && !counts?.total) return "unloaded" as const;
    if (groups.some((g) => g.state === "failed") || (counts?.failed ?? 0) > 0)
      return "failed" as const;
    if (groups.some((g) => g.state === "loading") || (counts?.loading ?? 0) > 0)
      return "loading" as const;
    if (!groups.length) return "unloaded" as const;
    return "ready" as const;
  };

  const dotKind = () => {
    if (unauthorized()) return "warn";
    if (ready() === "down") return "err";
    if (ready() === null) return "";
    const agg = aggregate();
    if (agg === "failed") return "err";
    if (agg === "ready") return "ok";
    return "warn";
  };

  const menuLabel = () => {
    if (unauthorized()) return "auth";
    const r = ready();
    if (r === null) return "…";
    if (r === "down") return "down";
    const groups = residents();
    if (!groups.length)
      return r === "loading" || (readyCounts()?.loading ?? 0) > 0
        ? "loading"
        : "no model";
    return groups[0].id;
  };

  const menuTitle = () => {
    if (unauthorized()) return "API key required or rejected";
    const r = ready();
    if (r === null) return "Checking server";
    if (r === "down") return "Engine unavailable";
    const groups = residents();
    if (!groups.length)
      return "No model loaded — open Models to load one";
    return groups
      .map((g) => `${g.id}: ${g.state ?? "loaded"}`)
      .join(" · ");
  };

  const residentSub = (g: ResidentGroup) => {
    const parts: string[] = [];
    if (g.state && g.state !== "loaded") parts.push(g.state);
    if (g.context) parts.push(`${fmtTokens(g.context)} ctx`);
    if (typeof g.memoryBytes === "number")
      parts.push(fmtBytes(g.memoryBytes));
    if (g.names.length > 1)
      parts.push(`+${g.names.length - 1} name${g.names.length > 2 ? "s" : ""}`);
    return parts.join(" · ");
  };

  const rowDot = (g: ResidentGroup) =>
    g.state === "failed" ? "err" : g.state === "loading" ? "warn" : "ok";

  /* Picking a resident points the chat composer at it. A decision resident
     can't generate — its surface is Judge, so it navigates there instead. */
  function pickResident(g: ResidentGroup) {
    setMenuOpen(false);
    if (g.decision) {
      navigate("/judge");
      return;
    }
    storageSet(MODEL_KEY, g.id);
    window.dispatchEvent(
      new CustomEvent("richengine:model-changed", { detail: g.id })
    );
    navigate("/");
  }

  /* Capacity reads only matter past the single-model default: a lone
     resident at max_residents=1 would otherwise show "1/1 · 0 free" noise. */
  const capacityText = () => {
    const pool = poolInfo();
    if (!pool?.max_residents || pool.max_residents <= 1) return "";
    const used = pool.max_residents - (pool.free ?? 0);
    return `${used}/${pool.max_residents} residents`;
  };

  const freeSlots = () => {
    const pool = poolInfo();
    return pool?.max_residents && pool.max_residents > 1 ? pool.free : undefined;
  };

  /* Per-resident memory when the server reports it, else the default
     resident's memory_actual. */
  const displayMemory = (): MemoryUse | null => {
    const withMemory = residents().filter(
      (g) => typeof g.memoryBytes === "number"
    );
    const base = memory();
    if (!withMemory.length) return base;
    return {
      current: withMemory.reduce((t, g) => t + (g.memoryBytes ?? 0), 0),
      peak:
        withMemory.length === 1 ? withMemory[0].memoryPeak : undefined,
      limit: base?.limit,
      pressure: base?.pressure,
    };
  };

  const active = (path: string) =>
    path === "/" ? location.pathname === "/" : location.pathname.startsWith(path);

  return (
    <>
    <header class="topbar">
      <A href="/" class="brand" title="RichEngine — Chat">
        <img src="/favicon.ico" alt="" width="18" height="18" />
        RichEngine
      </A>
      <nav class="topnav" aria-label="Pages" ref={(el) => (navEl = el)}>
        <span class="nav-ind" aria-hidden="true" />
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
        {/* Keyed remount plays the icon's entrance on every theme change. */}
        <Show when={theme()} keyed>
          {(name) => <ThemeIcon name={name} />}
        </Show>
      </button>
      <div class="model-menu" ref={(el) => (menuEl = el)}>
        <button
          type="button"
          class="top-model"
          aria-haspopup="menu"
          aria-expanded={menuOpen()}
          title={menuTitle()}
          onClick={() => setMenuOpen(!menuOpen())}
        >
          <span class={`dot ${dotKind()}`} aria-hidden="true" />
          <span class="name">{menuLabel()}</span>
          <Show when={residents().length > 1}>
            <span class="count">+{residents().length - 1}</span>
          </Show>
          <svg
            class="chev"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
        <Show when={menuOpen()}>
          <div class="model-pop" role="menu" aria-label="Resident models">
            <div class="model-pop-title">
              <span>Resident models</span>
              <Show when={capacityText()}>
                <span>{capacityText()}</span>
              </Show>
            </div>
            <Show
              when={residents().length}
              fallback={
                <div class="model-row-empty">
                  No model loaded — open Models to load one.
                </div>
              }
            >
              <For each={residents()}>
                {(g, i) => (
                  <button
                    type="button"
                    class="model-row"
                    role="menuitem"
                    title={
                      g.decision
                        ? `${g.id} — open Judge`
                        : `Use ${g.id} in chat`
                    }
                    onClick={() => pickResident(g)}
                  >
                    <span class={`dot ${rowDot(g)}`} aria-hidden="true" />
                    <span class="model-row-text">
                      <span class="model-row-name">
                        {g.id}
                        <Show when={i() === 0}>
                          {" "}
                          <span class="badge">default</span>
                        </Show>
                        <Show when={g.decision}>
                          {" "}
                          <span
                            class="badge"
                            title="Decision model — answers /v1/systemone and /v1/judgments in one forward pass; no text generation"
                          >
                            decision
                          </span>
                        </Show>
                      </span>
                      <Show when={residentSub(g)}>
                        <span
                          class="model-row-sub"
                          title={g.state === "failed" ? g.error : undefined}
                        >
                          {residentSub(g)}
                        </span>
                      </Show>
                    </span>
                  </button>
                )}
              </For>
            </Show>
            <div class="model-pop-foot">
              <span class="muted">
                <Show when={freeSlots() !== undefined}>
                  {`${freeSlots()} slot${freeSlots() === 1 ? "" : "s"} free`}
                </Show>
              </span>
              <A href="/models" onClick={() => setMenuOpen(false)}>
                Manage models
              </A>
            </div>
          </div>
        </Show>
      </div>
      <Show when={installJob()?.running}>
        <A
          href="/models"
          class="top-install"
          title={`Installing ${installJob()?.model ?? "model"} — view progress`}
        >
          <span class="dot" aria-hidden="true" />
          {installPct() !== undefined
            ? `installing ${installPct()}%`
            : "installing…"}
        </A>
      </Show>
      <div class="topright">
        <Show when={displayMemory()}>
          {(mem) => (
            <A
              href="/metrics"
              class="top-mem"
              classList={{
                warn:
                  mem().pressure === "warning" ||
                  mem().pressure === "critical",
              }}
              title={`Resident memory in use — peak ${
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
        <Show when={unauthorized() || apiKey()}>
          <A
            href="/settings"
            class="top-lock"
            classList={{ warn: unauthorized() }}
            title={
              unauthorized()
                ? "API key required or rejected — open Settings"
                : "API key set — manage in Settings"
            }
            aria-label={
              unauthorized()
                ? "API key required or rejected — open Settings"
                : "API key set — manage in Settings"
            }
          >
            <svg {...ICON_PROPS}>
              <rect x="5" y="11" width="14" height="9" rx="2" />
              <path d="M8 11V7a4 4 0 0 1 8 0v4" />
            </svg>
          </A>
        </Show>
      </div>
    </header>
    {/* Global outage banner — per-page notices describe last-good data;
        this one says the engine itself is gone. */}
    <Show when={ready() === "down"}>
      <div class="topdown" role="alert">
        Engine unreachable — retrying every 5s
      </div>
    </Show>
    </>
  );
}

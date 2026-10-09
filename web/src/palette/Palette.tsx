import { useNavigate } from "@solidjs/router";
import {
  createEffect,
  createMemo,
  createSignal,
  For,
  onCleanup,
  onMount,
  Show,
} from "solid-js";
import {
  CommandBar,
  CommandBarDescription,
  CommandBarOverlay,
  CommandBarPanel,
  CommandBarTitle,
  Dialog,
  DialogOverlay,
  DialogPanel,
  DialogTitle,
} from "terracotta";
import { api, curlExample, storageGet, storageSet } from "../api";
import { useCopied } from "../clipboard";
import { MODEL_KEY } from "../pages/chat/storage";
import { cycleTheme, theme, THEME_LABELS } from "../theme";
import {
  closePalette,
  closeShortcuts,
  isEditableTarget,
  openPalette,
  openShortcuts,
  paletteOpen,
  shortcutsOpen,
  togglePalette,
  toggleShortcuts,
} from "../shortcuts";
import "./palette.css";

/* Session cache for /v1/models — survives palette open/close, refetched only
   after a load failure (cache stays null). */
let modelCache: string[] | null = null;

interface Row {
  id: string;
  title: string;
  hint?: string;
  kbd?: string;
  run: () => void;
}

const SHORTCUTS: [string, string][] = [
  ["⌘K", "Command palette"],
  ["⌘1", "Go to Chat"],
  ["⌘2", "Go to Models"],
  ["⌘3", "Go to Disk"],
  ["⌘4", "Go to Playground"],
  ["⌘5", "Go to Judge"],
  ["⌘6", "Go to Metrics"],
  ["⌘7", "Go to Settings"],
  ["Esc", "Stop generation / close"],
  ["⏎", "Send message (chat)"],
  ["⇧⏎", "Newline (chat)"],
  ["/", "Focus message input (chat)"],
  ["⌘B", "Toggle sidebar (chat)"],
  ["?", "This list"],
];

export default function Palette() {
  const navigate = useNavigate();
  const [query, setQuery] = createSignal("");
  const [mode, setMode] = createSignal<"root" | "models">("root");
  const [active, setActive] = createSignal(0);
  const [models, setModels] = createSignal<string[] | null>(modelCache);
  const [modelsError, setModelsError] = createSignal(false);
  const { copied, copy } = useCopied();
  let inputRef: HTMLInputElement | undefined;
  let listRef: HTMLDivElement | undefined;

  function currentModel(): string {
    return storageGet(MODEL_KEY) ?? "";
  }

  function currentModelLabel(): string {
    return currentModel() || "Default";
  }

  async function copyBaseUrl() {
    await copy("base", `${location.origin}/v1`);
  }

  async function loadModels() {
    if (modelCache) {
      setModels(modelCache);
      return;
    }
    setModelsError(false);
    try {
      const res = await api<{ data?: { id?: string }[] }>("/v1/models");
      const ids = (res.data ?? [])
        .map((m) => m.id)
        .filter((id): id is string => typeof id === "string" && id.length > 0);
      modelCache = ids;
      setModels(ids);
    } catch {
      setModelsError(true);
    }
  }

  function enterModels() {
    setMode("models");
    setQuery("");
    setActive(0);
    if (models() === null && !modelsError()) void loadModels();
  }

  function chooseModel(id: string) {
    storageSet(MODEL_KEY, id);
    window.dispatchEvent(new CustomEvent("richengine:model-changed", { detail: id }));
    closePalette();
  }

  const rootActions = (): Row[] => [
    {
      id: "new-chat",
      title: "New chat",
      run: () => {
        // Chat mounts its listener on navigation; dispatch once it attaches.
        navigate("/");
        setTimeout(
          () => window.dispatchEvent(new CustomEvent("richengine:new-chat")),
          0
        );
        closePalette();
      },
    },
    {
      id: "focus-input",
      title: "Focus message input",
      kbd: "/",
      run: () => {
        navigate("/");
        setTimeout(
          () => window.dispatchEvent(new CustomEvent("richengine:focus-input")),
          0
        );
        closePalette();
      },
    },
    {
      id: "toggle-sidebar",
      title: "Toggle sidebar",
      kbd: "⌘B",
      run: () => {
        window.dispatchEvent(new CustomEvent("richengine:toggle-sidebar"));
        closePalette();
      },
    },
    { id: "go-chat", title: "Go to Chat", kbd: "⌘1", run: () => { navigate("/"); closePalette(); } },
    { id: "go-models", title: "Go to Models", kbd: "⌘2", run: () => { navigate("/models"); closePalette(); } },
    { id: "go-disk", title: "Go to Disk", kbd: "⌘3", run: () => { navigate("/disk"); closePalette(); } },
    { id: "go-playground", title: "Go to Playground", kbd: "⌘4", run: () => { navigate("/playground"); closePalette(); } },
    { id: "go-judge", title: "Go to Judge", kbd: "⌘5", run: () => { navigate("/judge"); closePalette(); } },
    { id: "go-metrics", title: "Go to Metrics", kbd: "⌘6", run: () => { navigate("/metrics"); closePalette(); } },
    { id: "go-settings", title: "Go to Settings", kbd: "⌘7", run: () => { navigate("/settings"); closePalette(); } },
    { id: "switch-model", title: "Switch model…", hint: currentModelLabel(), run: enterModels },
    {
      id: "tune-model",
      title: "Tune a model…",
      hint: "Models",
      // The sweep's per-model buttons live on the Models page.
      run: () => { navigate("/models"); closePalette(); },
    },
    {
      id: "shortcuts",
      title: "Keyboard shortcuts",
      kbd: "?",
      run: () => {
        closePalette();
        openShortcuts();
      },
    },
    {
      id: "theme",
      title: "Cycle theme",
      hint: THEME_LABELS[theme()],
      // Stays open so repeated runs cycle through every theme.
      run: () => cycleTheme(),
    },
    {
      id: "export-chat",
      title: "Export chat as Markdown",
      run: () => {
        navigate("/");
        setTimeout(
          () =>
            window.dispatchEvent(new CustomEvent("richengine:export-chat")),
          0
        );
        closePalette();
      },
    },
    {
      id: "copy-base",
      title: "Copy API base URL",
      hint: copied() === "base" ? "Copied ✓" : `${location.origin}/v1`,
      run: copyBaseUrl,
    },
    {
      id: "copy-curl",
      title: "Copy curl example",
      hint: copied() === "curl" ? "Copied ✓" : "POST /v1/chat/completions",
      run: async () => {
        await copy(
          "curl",
          curlExample(currentModel() || modelCache?.[0] || "MODEL_ID")
        );
      },
    },
  ];

  const rows = createMemo<Row[]>(() => {
    const q = query().trim().toLowerCase();
    if (mode() === "models") {
      const cur = currentModel();
      const list = [{ id: "", title: "Default" }, ...(models() ?? []).map((id) => ({ id, title: id }))];
      return list
        .filter((m) => !q || m.title.toLowerCase().includes(q))
        .map((m) => ({
          id: m.id,
          title: m.title,
          hint: cur === m.id ? "✓ current" : "",
          run: () => chooseModel(m.id),
        }));
    }
    return rootActions().filter((a) => !q || a.title.toLowerCase().includes(q));
  });

  /* Keep active index in range when the row set changes. */
  createEffect(() => {
    const n = rows().length;
    if (active() >= n) setActive(Math.max(0, n - 1));
  });

  /* Keep the active row visible. */
  createEffect(() => {
    active();
    listRef
      ?.querySelector(".palette-row.active")
      ?.scrollIntoView({ block: "nearest" });
  });

  /* Reset state whenever the palette opens; the panel focuses the input. */
  createEffect(() => {
    if (paletteOpen()) {
      setMode("root");
      setQuery("");
      setActive(0);
      queueMicrotask(() => inputRef?.focus());
    }
  });

  /* Focus trapping and Escape are handled by the terracotta panels; this
     handles the option-list navigation keys inside the query input. */
  function onInputKey(e: KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, rows().length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      rows()[active()]?.run();
    } else if (e.key === "Backspace" && mode() === "models" && query() === "") {
      setMode("root");
      setActive(0);
    }
    /* Escape handled globally on capture. */
  }

  /* All global key handling lives here, on capture, so palette Esc wins over
     page-level handlers (e.g. chat's abort-stream Esc). Capturing also beats
     CommandBar's own ⌘K listener, which defers to preventDefault. */
  function onGlobalKey(e: KeyboardEvent) {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      closeShortcuts();
      togglePalette();
      return;
    }
    const pages: Record<string, string> = {
      "1": "/",
      "2": "/models",
      "3": "/disk",
      "4": "/playground",
      "5": "/judge",
      "6": "/metrics",
      "7": "/settings",
    };
    if (mod && pages[e.key]) {
      e.preventDefault();
      closePalette();
      closeShortcuts();
      navigate(pages[e.key]);
      return;
    }
    if (mod && (e.key === "b" || e.key === "B")) {
      e.preventDefault();
      window.dispatchEvent(new CustomEvent("richengine:toggle-sidebar"));
      return;
    }
    if (e.key === "Escape") {
      if (paletteOpen()) {
        e.preventDefault();
        e.stopPropagation();
        closePalette();
      } else if (shortcutsOpen()) {
        e.preventDefault();
        e.stopPropagation();
        closeShortcuts();
      }
      return;
    }
    if (
      e.key === "/" &&
      !paletteOpen() &&
      !shortcutsOpen() &&
      !isEditableTarget(e.target)
    ) {
      e.preventDefault();
      window.dispatchEvent(new CustomEvent("richengine:focus-input"));
    }
    if (e.key === "?" && !paletteOpen() && !isEditableTarget(e.target)) {
      e.preventDefault();
      toggleShortcuts();
    }
  }

  onMount(() => {
    window.addEventListener("keydown", onGlobalKey, { capture: true });
  });
  onCleanup(() => {
    window.removeEventListener("keydown", onGlobalKey, { capture: true });
  });

  const modelRowsBlocked = () =>
    mode() === "models" && (modelsError() || models() === null);

  return (
    <>
      <CommandBar
        isOpen={paletteOpen()}
        onChange={(open) => (open ? openPalette() : closePalette())}
      >
        <CommandBarOverlay class="palette-overlay" />
        <CommandBarPanel class="palette-panel">
          <CommandBarTitle class="sr-only">Command palette</CommandBarTitle>
          <CommandBarDescription class="sr-only">
            Run a command or pick a model
          </CommandBarDescription>
          <div class="palette-input-row">
            <input
              ref={inputRef}
              class="palette-input"
              type="text"
              autofocus
              spellcheck={false}
              role="combobox"
              aria-expanded="true"
              aria-controls="palette-list"
              aria-activedescendant={
                rows().length ? `palette-row-${active()}` : undefined
              }
              placeholder={mode() === "models" ? "Select a model…" : "Type a command…"}
              aria-label={mode() === "models" ? "Select a model" : "Command palette"}
              value={query()}
              onInput={(e) => {
                setQuery(e.currentTarget.value);
                setActive(0);
              }}
              onKeyDown={onInputKey}
            />
            <Show when={mode() === "models"}>
              <span class="palette-back">⌫ back</span>
            </Show>
          </div>
          <div
            class="palette-list"
            id="palette-list"
            role="listbox"
            ref={listRef}
          >
            <Show when={mode() === "models" && modelsError()}>
              <div class="palette-empty">Couldn't load models</div>
            </Show>
            <Show when={mode() === "models" && !modelsError() && models() === null}>
              <div class="palette-empty">Loading models…</div>
            </Show>
            <For each={rows()}>
              {(row, i) => (
                <div
                  class="palette-row"
                  id={`palette-row-${i()}`}
                  classList={{ active: i() === active() }}
                  role="option"
                  aria-selected={i() === active()}
                  onMouseEnter={() => setActive(i())}
                  onClick={() => row.run()}
                >
                  <span class="palette-row-title">{row.title}</span>
                  <Show when={row.kbd}>
                    <kbd class="kbd">{row.kbd}</kbd>
                  </Show>
                  <Show when={row.hint}>
                    <span class="palette-row-hint">{row.hint}</span>
                  </Show>
                </div>
              )}
            </For>
            <Show when={rows().length === 0 && !modelRowsBlocked()}>
              <div class="palette-empty">No matches</div>
            </Show>
          </div>
          <div class="palette-footer">↑↓ navigate · ↵ run · esc close</div>
        </CommandBarPanel>
      </CommandBar>

      <Dialog
        isOpen={shortcutsOpen()}
        onChange={(open) => (open ? openShortcuts() : closeShortcuts())}
      >
        <DialogOverlay class="palette-overlay" />
        <DialogPanel class="palette-panel palette-panel-small">
          <DialogTitle as="div" class="palette-title">
            Keyboard shortcuts
          </DialogTitle>
          <div class="palette-list">
            <For each={SHORTCUTS}>
              {([keys, desc]) => (
                <div class="palette-row palette-row-static">
                  <span class="palette-row-title">{desc}</span>
                  <span class="palette-row-hint">
                    <kbd class="kbd">{keys}</kbd>
                  </span>
                </div>
              )}
            </For>
          </div>
          <div class="palette-footer">esc or ? close</div>
        </DialogPanel>
      </Dialog>
    </>
  );
}

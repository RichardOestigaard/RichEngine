import { createSignal, For, Show } from "solid-js";
import { storageGet, storageSet } from "../../api";
import { fmtAgo } from "../../format";
import { conversations, SYSTEM_KEY } from "./storage";
import type { Conversation } from "./types";

export default function Sidebar(props: {
  open: boolean;
  activeId: string | null;
  disabled: boolean;
  onNew(): void;
  onOpen(id: string): void;
  onDelete(id: string): void;
  onRename(id: string, title: string): void;
}) {
  const [filter, setFilter] = createSignal("");
  const [system, setSystem] = createSignal(storageGet(SYSTEM_KEY) ?? "");

  const filtered = () => {
    const query = filter().trim().toLowerCase();
    const list = conversations();
    return query
      ? list.filter((conv) => conv.title.toLowerCase().includes(query))
      : list;
  };

  const filtering = () => Boolean(filter().trim());

  // Recency buckets; conversations arrive sorted by updated, so same-label
  // runs merge into one group as the list is walked.
  const groups = () => {
    const day = 86400000;
    const now = new Date();
    const today = new Date(
      now.getFullYear(), now.getMonth(), now.getDate()
    ).getTime();
    const out: { label: string; items: Conversation[] }[] = [];
    for (const conv of filtered()) {
      const label =
        conv.updated >= today
          ? "Today"
          : conv.updated >= today - day
            ? "Yesterday"
            : conv.updated >= today - 6 * day
              ? "This week"
              : "Older";
      const last = out[out.length - 1];
      if (last?.label === label) last.items.push(conv);
      else out.push({ label, items: [conv] });
    }
    return out;
  };

  const renderRecent = (conv: Conversation) => (
    <Recent
      conv={conv}
      active={conv.id === props.activeId}
      disabled={props.disabled}
      onOpen={() => props.onOpen(conv.id)}
      onDelete={() => props.onDelete(conv.id)}
      onRename={(title) => props.onRename(conv.id, title)}
    />
  );

  return (
    <aside id="sidebar">
      <button
        id="new-chat"
        type="button"
        title="Start a new chat"
        disabled={props.disabled}
        onClick={() => props.onNew()}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.8"
          aria-hidden="true"
        >
          <path d="M12 5H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-6" />
          <path d="m15 4 5 5M14 10l6-6" />
        </svg>
        New chat
      </button>
      <button
        id="export-chat"
        type="button"
        disabled={!props.activeId}
        title="Download the current chat as Markdown"
        onClick={() =>
          window.dispatchEvent(new CustomEvent("richengine:export-chat"))
        }
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.8"
          aria-hidden="true"
        >
          <path d="M12 3v12m0 0l-4-4m4 4l4-4" />
          <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
        </svg>
        Export
      </button>
      <Show when={conversations().length > 1}>
        <input
          class="recent-search"
          type="search"
          placeholder="Search chats"
          aria-label="Search chats"
          value={filter()}
          onInput={(event) => setFilter(event.currentTarget.value)}
        />
      </Show>
      <Show when={filtering()}>
        <div class="recent-label">Recents</div>
      </Show>
      <nav id="recents" aria-label="Recent chats">
        <Show
          when={filtering()}
          fallback={
            <For each={groups()}>
              {(group) => (
                <>
                  <div class="recent-label group">{group.label}</div>
                  <For each={group.items}>{renderRecent}</For>
                </>
              )}
            </For>
          }
        >
          <For each={filtered()}>{renderRecent}</For>
        </Show>
        <Show when={filter() && !filtered().length}>
          <div class="recent-empty muted">No matching chats</div>
        </Show>
      </nav>
      <details class="sys-prompt">
        <summary>System prompt</summary>
        <textarea
          rows={3}
          placeholder="Optional instructions for every message…"
          aria-label="System prompt"
          value={system()}
          onInput={(event) => {
            setSystem(event.currentTarget.value);
            storageSet(SYSTEM_KEY, event.currentTarget.value);
          }}
        />
        <div class="sys-hint muted">
          Sent as the first message of every request. Clearing disables it.
        </div>
      </details>
    </aside>
  );
}

function Recent(props: {
  conv: Conversation;
  active: boolean;
  disabled: boolean;
  onOpen(): void;
  onDelete(): void;
  onRename(title: string): void;
}) {
  const [editing, setEditing] = createSignal(false);
  let editEl: HTMLInputElement | undefined;
  let cancelled = false;

  function startEdit() {
    cancelled = false;
    setEditing(true);
  }

  function commit() {
    if (!editing() || cancelled || !editEl) return;
    const title = editEl.value.replace(/\s+/g, " ").trim();
    if (title && title !== props.conv.title) props.onRename(title);
    setEditing(false);
  }

  return (
    <div class="recent" classList={{ active: props.active }}>
      <Show
        when={editing()}
        fallback={
          <>
            <button
              type="button"
              class="recent-title"
              title={props.conv.title}
              disabled={props.disabled}
              onClick={() => props.onOpen()}
            >
              {props.conv.title}
            </button>
            <span
              class="recent-ago muted"
              title={new Date(props.conv.updated).toLocaleString()}
            >
              {fmtAgo(props.conv.updated)}
            </span>
            <button
              type="button"
              class="recent-act"
              aria-label={`Rename ${props.conv.title}`}
              title="Rename"
              disabled={props.disabled}
              onClick={startEdit}
            >
              ✎
            </button>
            <button
              type="button"
              class="recent-act"
              aria-label={`Delete ${props.conv.title}`}
              title="Delete"
              disabled={props.disabled}
              onClick={() => props.onDelete()}
            >
              ×
            </button>
          </>
        }
      >
        <input
          class="recent-edit"
          aria-label="Chat title"
          value={props.conv.title}
          ref={(el) => {
            editEl = el;
            queueMicrotask(() => {
              el.focus();
              el.select();
            });
          }}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") editEl?.blur();
            else if (e.key === "Escape") {
              cancelled = true;
              setEditing(false);
            }
          }}
        />
      </Show>
    </div>
  );
}

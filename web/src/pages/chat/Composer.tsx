import { createEffect, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { storageGet } from "../../api";
import { pushToast } from "../../toast";
import { MODEL_KEY } from "./storage";
import { jsonOutput, setJsonOutput, setToolEnabled, TOOL_HINTS, TOOL_IDS, TOOL_LABELS, toolsEnabled } from "./tools";
import { devMode } from "../../devmode";
import { hasSamplingFor, SAMPLING_FIELDS, samplingFor } from "../../sampling";
import type { Attachment, ModelInfo, QueuedTurn } from "./types";

// field-sizing grows the textarea in CSS; the manual measure is the fallback.
const canFieldSize =
  typeof CSS !== "undefined" && CSS.supports("field-sizing", "content");

/* A clipboard dump past either threshold attaches as pasted.txt instead of
   flooding the box: the composer stays readable and the size shows up in the
   token meter rather than as a wall of text. */
const PASTE_FILE_CHARS = 2000;
const PASTE_FILE_LINES = 20;

const EFFORT_LABELS: Record<string, string> = {
  xhigh: "XHigh",
  medium: "Medium",
  low: "Low",
  none: "Off",
};

export default function Composer(props: {
  running: boolean;
  canSend: boolean;
  queued: QueuedTurn[];
  onRemoveQueued(turn: QueuedTurn): void;
  onEditQueued(turn: QueuedTurn): void;
  attachments: Attachment[];
  attachmentsVersion: number;
  imagesAccepted: boolean;
  pdfsAccepted: boolean;
  tokenText: string;
  tokenPct: number;
  models: ModelInfo[];
  model: string;
  effort: string;
  input: string;
  onSend(): void;
  onToggle(): void;
  onInput(value: string): void;
  onModelChange(value: string): void;
  onEffortChange(value: string): void;
  onAddFiles(files: File[]): void;
  onRemoveAttachment(attachment: Attachment): void;
  /* Shell-style history: the handler returns true when it moved the draft,
     false when the caret should move normally. */
  onRecallPrev?(): boolean;
  onRecallNext?(): boolean;
  inputRef?(el: HTMLTextAreaElement): void;
}) {
  let inputEl: HTMLTextAreaElement | undefined;
  let fileInput: HTMLInputElement | undefined;
  let toolsEl: HTMLDivElement | undefined;
  const [toolsOpen, setToolsOpen] = createSignal(false);
  const anyToolOn = () =>
    TOOL_IDS.some((id) => toolsEnabled()[id]) || jsonOutput();

  // The sampling badge names the model's overridden fields (Models → ⚙);
  // developer mode shows it, nothing else cares.
  const samplingNote = () => {
    if (!devMode() || !hasSamplingFor(props.model)) return "";
    const settings = samplingFor(props.model);
    const names = SAMPLING_FIELDS.filter(
      (field) => settings[field.key] !== undefined
    )
      .map((field) => `${field.key}=${settings[field.key]}`)
      .join(", ");
    return names ? `Sampling overrides for ${props.model}: ${names}` : "";
  };

  // A pointerdown outside the open menu closes it; clicks inside do not.
  // Escape closes it too, in the capture phase: the chat's own Escape
  // handler aborts a generation, which a menu dismissal must not reach.
  createEffect(() => {
    if (!toolsOpen()) return;
    const onDown = (event: PointerEvent) => {
      if (!toolsEl?.contains(event.target as Node)) setToolsOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        setToolsOpen(false);
      }
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey, true);
    onCleanup(() => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey, true);
    });
  });

  function resize() {
    if (canFieldSize || !inputEl) return;
    inputEl.style.height = "auto";
    inputEl.style.height = `${inputEl.scrollHeight}px`;
  }

  createEffect(() => {
    void props.input;
    queueMicrotask(resize);
  });

  // Clearing attachments also resets the file input, like clearAttachments did.
  createEffect(() => {
    if (!props.attachments.length && fileInput) fileInput.value = "";
  });

  onMount(() => {
    const onModelChanged = () =>
      props.onModelChange(storageGet(MODEL_KEY) ?? "");
    window.addEventListener("richengine:model-changed", onModelChanged);
    onCleanup(() =>
      window.removeEventListener("richengine:model-changed", onModelChanged)
    );
    queueMicrotask(() => inputEl?.focus());
  });

  const acceptedFiles = () =>
    [
      "text/plain",
      props.imagesAccepted ? "image/*" : "",
      props.pdfsAccepted ? "application/pdf" : "",
    ]
      .filter(Boolean)
      .join(",");

  function onPaste(event: ClipboardEvent) {
    if (props.imagesAccepted) {
      const images = Array.from(event.clipboardData?.items ?? [])
        .filter((item) => item.type.startsWith("image/"))
        .map((item) => item.getAsFile())
        .filter((file): file is File => Boolean(file));
      if (images.length) {
        if (!event.clipboardData?.getData("text/plain")) event.preventDefault();
        props.onAddFiles(images);
        return;
      }
    }
    const text = event.clipboardData?.getData("text/plain") ?? "";
    if (
      text.length >= PASTE_FILE_CHARS ||
      text.split("\n").length > PASTE_FILE_LINES
    ) {
      event.preventDefault();
      props.onAddFiles([new File([text], "pasted.txt", { type: "text/plain" })]);
      pushToast("ok", "Large paste attached as pasted.txt");
    }
  }

  function onKeyDown(event: KeyboardEvent) {
    // Safari can end composition before the confirming Enter keydown.
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      props.onSend();
    } else if (event.key === "ArrowUp" && props.onRecallPrev?.()) {
      event.preventDefault();
    } else if (event.key === "ArrowDown" && props.onRecallNext?.()) {
      event.preventDefault();
    }
  }

  return (
    <form
      id="form"
      onSubmit={(event) => {
        event.preventDefault();
        props.onToggle();
      }}
    >
      <input
        id="file-input"
        type="file"
        accept={acceptedFiles()}
        multiple
        hidden
        aria-hidden="true"
        ref={(el) => (fileInput = el)}
        onChange={(event) => {
          props.onAddFiles(Array.from(event.currentTarget.files ?? []));
          event.currentTarget.value = "";
        }}
      />
      {/* Queued turns wait here, anchored to the box's top edge — out of
          the message log — and send in order once the reply ends. */}
      <Show when={props.queued.length}>
        <div class="queue-tray" role="status" aria-label="Queued messages">
          <For each={props.queued}>
            {(turn, i) => (
              <div
                class="queue-row"
                title="Queued — sends when the current reply finishes"
              >
                <button
                  type="button"
                  class="queue-edit"
                  title="Edit — pull back into the composer"
                  onClick={() => props.onEditQueued(turn)}
                >
                  <span class="queue-pos">
                    {i() === 0 ? "Next" : `#${i() + 1}`}
                  </span>
                  <span class="queue-text">
                    {turn.text ||
                      turn.attachments.map((a) => a.name).join(", ") ||
                      "Attachment"}
                  </span>
                  <Show when={turn.attachments.length}>
                    <span class="queue-atts">
                      <For each={turn.attachments}>
                        {(attachment) =>
                          attachment.kind === "image" && attachment.url ? (
                            <img
                              src={attachment.url}
                              alt={attachment.name || "Attached image"}
                            />
                          ) : (
                            <span class="queue-file" title={attachment.name}>
                              📄 {attachment.name}
                            </span>
                          )
                        }
                      </For>
                    </span>
                  </Show>
                </button>
                <button
                  type="button"
                  class="queue-remove"
                  aria-label="Remove queued message"
                  title="Remove queued message"
                  onClick={() => props.onRemoveQueued(turn)}
                >
                  ×
                </button>
              </div>
            )}
          </For>
        </div>
      </Show>
      <div class="composer">
        <div id="attachments" aria-label="Attachments">
          <For each={props.attachments}>
            {(attachment) => {
              // Attachment fields are mutated by FileReader callbacks;
              // attachmentsVersion re-reads them.
              const url = () => {
                void props.attachmentsVersion;
                return attachment.url;
              };
              const errored = () => {
                void props.attachmentsVersion;
                return attachment.error;
              };
              const name = () => attachment.name || "Attached file";
              return (
                <div
                  class="attachment"
                  classList={{
                    pdf: attachment.kind === "pdf",
                    txt: attachment.kind === "text",
                  }}
                  title={name()}
                >
                  <Show
                    when={url()}
                    fallback={
                      <span
                        class="attachment-status"
                        classList={{ error: errored() }}
                        role="status"
                      >
                        {errored() ? "Read failed" : "Loading…"}
                      </span>
                    }
                  >
                    <Show
                      when={attachment.kind === "image"}
                      fallback={
                        <span class="attachment-pdf">
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                            <path d="M6 3h8l5 5v13H6z" />
                            <path d="M14 3v5h5" />
                          </svg>
                          <span class="attachment-pdf-name">{name()}</span>
                        </span>
                      }
                    >
                      <img src={url()!} alt={name()} />
                    </Show>
                  </Show>
                  <button
                    type="button"
                    class="remove-image"
                    aria-label={`Remove ${name()}`}
                    onClick={() => props.onRemoveAttachment(attachment)}
                  >
                    ×
                  </button>
                </div>
              );
            }}
          </For>
        </div>
        <textarea
          id="input"
          rows="1"
          placeholder="Message RichEngine"
          aria-label="Message RichEngine"
          title={
            props.running
              ? "Enter queues the message · Shift+Enter for a new line"
              : "Enter to send · Shift+Enter for a new line"
          }
          autofocus
          enterkeyhint="send"
          ref={(el) => {
            inputEl = el;
            props.inputRef?.(el);
          }}
          value={props.input}
          onInput={(event) => props.onInput(event.currentTarget.value)}
          onPaste={onPaste}
          onKeyDown={onKeyDown}
        />
      </div>
      <div class="toolbar">
        <button
          id="attach"
          type="button"
          aria-label="Attach images, PDFs, or text files"
          title="Attach images, PDFs, or text files"
          onClick={() => fileInput?.click()}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.8"
            stroke-linecap="round"
          >
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>
        <div class="tools-menu" ref={(el) => (toolsEl = el)}>
          <button
            id="tools"
            type="button"
            aria-label="Built-in tools"
            aria-expanded={toolsOpen()}
            title={
              props.running
                ? "Tools are fixed until this reply finishes"
                : "Built-in tools"
            }
            disabled={props.running}
            classList={{ on: anyToolOn() }}
            onClick={() => setToolsOpen(!toolsOpen())}
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.8"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M14.7 6.3a4.4 4.4 0 0 0-6 5.6L3 17.6V21h3.4l5.7-5.7a4.4 4.4 0 0 0 5.6-6l-3 3-2.1-2.1 3-3z" />
            </svg>
          </button>
          <Show when={toolsOpen()}>
            <div class="tools-pop" role="menu" aria-label="Built-in tools">
              <div class="tools-pop-title">Built-in tools</div>
              <For each={TOOL_IDS}>
                {(id) => (
                  <label class="tools-row">
                    <input
                      type="checkbox"
                      checked={toolsEnabled()[id]}
                      onChange={(event) =>
                        setToolEnabled(id, event.currentTarget.checked)
                      }
                    />
                    <span class="tools-row-text">
                      <span class="tools-row-label">{TOOL_LABELS[id]}</span>
                      <span class="tools-row-hint">{TOOL_HINTS[id]}</span>
                    </span>
                  </label>
                )}
              </For>
              <div class="tools-pop-title">Output</div>
              <label class="tools-row">
                <input
                  type="checkbox"
                  checked={jsonOutput()}
                  onChange={(event) =>
                    setJsonOutput(event.currentTarget.checked)
                  }
                />
                <span class="tools-row-text">
                  <span class="tools-row-label">JSON output</span>
                  <span class="tools-row-hint">
                    Guaranteed-valid JSON response
                  </span>
                </span>
              </label>
            </div>
          </Show>
        </div>
        <Show when={samplingNote()}>
          <span class="sampling-badge" title={samplingNote()}>
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.8"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.09a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.09a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </span>
        </Show>
        <Show when={jsonOutput()}>
          <button
            type="button"
            class="json-pill"
            title="Replies are constrained to JSON — click to turn off"
            aria-label="JSON output on, click to turn off"
            onClick={() => setJsonOutput(false)}
          >
            JSON ×
          </button>
        </Show>
        {/* While a reply streams the pickers lock into a static badge that
            states what's in flight, instead of two greyed-out selects. */}
        <Show
          when={!props.running}
          fallback={
            <span
              class="effort static"
              title="Model and thinking effort are fixed until this reply finishes"
            >
              {props.model || "Default"}
              {props.effort
                ? ` · ${EFFORT_LABELS[props.effort] ?? props.effort}`
                : ""}
            </span>
          }
        >
          <label class="effort picker" hidden={!props.models.length}>
            <span>Model</span>
            <select
              id="model"
              aria-label="Model"
              value={props.model}
              onChange={(event) => props.onModelChange(event.currentTarget.value)}
            >
              <option value="">Default</option>
              <For each={props.models}>
                {(item) =>
                  item?.id ? <option value={item.id}>{item.id}</option> : null
                }
              </For>
            </select>
          </label>
          <label class="effort">
            <span>Thinking</span>
            <select
              id="effort"
              aria-label="Thinking effort"
              value={props.effort}
              onChange={(event) => props.onEffortChange(event.currentTarget.value)}
            >
              <option value="">Default</option>
              <option value="xhigh">XHigh</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="none">Off</option>
            </select>
          </label>
        </Show>
        <div class="spacer" />
        <Show when={props.running}>
          <span class="stop-hint muted">Esc to stop · Enter to queue</span>
        </Show>
        <Show when={props.tokenText}>
          <span
            class="token-meter"
            classList={{
              warn: props.tokenPct >= 80 && props.tokenPct < 95,
              danger: props.tokenPct >= 95,
            }}
            title="Estimated prompt size — chat history plus this draft"
          >
            {props.tokenText}
          </span>
        </Show>
        {/* aria-disabled keeps the button focusable and clickable — a click
            with nothing to send just focuses the textarea. */}
        <button
          id="send"
          aria-disabled={!props.running && !props.canSend}
          aria-label={props.running ? "Stop" : "Send"}
          title={
            props.running
              ? "Stop — Enter queues a message"
              : props.canSend
                ? "Send"
                : "Type a message to send"
          }
          onClick={() => {
            if (!props.running && !props.canSend) inputEl?.focus();
          }}
        >
          {props.running ? "■" : "↑"}
        </button>
      </div>
    </form>
  );
}

import { createEffect, createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { storageGet } from "../../api";
import { MODEL_KEY } from "./storage";
import { setToolEnabled, TOOL_HINTS, TOOL_IDS, TOOL_LABELS, toolsEnabled } from "./tools";
import type { Attachment, ModelInfo } from "./types";

// field-sizing grows the textarea in CSS; the manual measure is the fallback.
const canFieldSize =
  typeof CSS !== "undefined" && CSS.supports("field-sizing", "content");

export default function Composer(props: {
  running: boolean;
  canSend: boolean;
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
  onRecall?(): void;
  inputRef?(el: HTMLTextAreaElement): void;
}) {
  let inputEl: HTMLTextAreaElement | undefined;
  let fileInput: HTMLInputElement | undefined;
  let toolsEl: HTMLDivElement | undefined;
  const [toolsOpen, setToolsOpen] = createSignal(false);
  const anyToolOn = () => TOOL_IDS.some((id) => toolsEnabled()[id]);

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
      props.imagesAccepted ? "image/*" : "",
      props.pdfsAccepted ? "application/pdf" : "",
    ]
      .filter(Boolean)
      .join(",");

  function onPaste(event: ClipboardEvent) {
    // Without image input the browser pastes as usual.
    if (!props.imagesAccepted) return;
    const images = Array.from(event.clipboardData?.items ?? [])
      .filter((item) => item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter((file): file is File => Boolean(file));
    if (!images.length) return;
    if (!event.clipboardData?.getData("text/plain")) event.preventDefault();
    props.onAddFiles(images);
  }

  function onKeyDown(event: KeyboardEvent) {
    // Safari can end composition before the confirming Enter keydown.
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      props.onSend();
    } else if (event.key === "ArrowUp" && !props.input && props.onRecall) {
      event.preventDefault();
      props.onRecall();
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
                  classList={{ pdf: attachment.kind === "pdf" }}
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
          title="Enter to send · Shift+Enter for a new line"
          autofocus
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
          aria-label="Add images or PDFs"
          title="Attach images or PDFs"
          hidden={!props.imagesAccepted && !props.pdfsAccepted}
          disabled={props.running}
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
            title="Built-in tools"
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
            </div>
          </Show>
        </div>
        <label class="effort picker" hidden={!props.models.length}>
          <span>Model</span>
          <select
            id="model"
            aria-label="Model"
            disabled={props.running}
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
            disabled={props.running}
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
        <div class="spacer" />
        <Show when={props.running}>
          <span class="stop-hint muted">Esc to stop</span>
        </Show>
        <Show when={props.tokenText}>
          <span
            class="token-meter"
            classList={{
              warn: props.tokenPct >= 80 && props.tokenPct < 95,
              danger: props.tokenPct >= 95,
            }}
            title="Approximate prompt tokens in this draft"
          >
            {props.tokenText}
          </span>
        </Show>
        <button
          id="send"
          disabled={!props.running && !props.canSend}
          aria-label={props.running ? "Stop" : "Send"}
          title={props.running ? "Esc to stop" : "Send"}
        >
          {props.running ? "■" : "↑"}
        </button>
      </div>
    </form>
  );
}

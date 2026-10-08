import { createSignal, For, Show } from "solid-js";
import { fmtClock } from "../../format";
import { createMarkdownPainter, displayContent } from "./markdown";
import type { MarkdownPainter } from "./markdown";
import { userParts } from "./storage";
import type { Message } from "./types";

// The box carries its message entry, matching the vanilla version's
// box._message: the delegated regenerate handler reads it back.
export type MessageElement = HTMLElement & { _message?: Message };

export interface BoxHandle {
  el: MessageElement;
  painter: MarkdownPainter;
}

export default function MessageView(props: {
  msg: Message;
  live?: boolean;
  error?: boolean;
  register?: (handle: BoxHandle) => void;
}) {
  const msg = props.msg;

  function mount(el: HTMLElement) {
    const box = el as MessageElement;
    box._message = msg;
    if (msg.role === "assistant") {
      const painter = createMarkdownPainter(box);
      painter.update(typeof msg.content === "string" ? msg.content : "");
      props.register?.({ el: box, painter });
    }
  }

  if (msg.role === "user") {
    const parts = userParts(msg.content);
    return (
      <div
        class="message user"
        ref={mount}
        title={
          msg.created ? new Date(msg.created).toLocaleString() : undefined
        }
      >
        {parts.text ? <div class="user-text">{parts.text}</div> : null}
        {parts.images.length ? (
          <div class="user-images">
            {parts.images.map((url) => (
              <img src={url} alt="Attached image" />
            ))}
          </div>
        ) : null}
        {parts.files.length ? (
          <div class="user-files">
            {parts.files.map((file) => (
              <span class="file-chip" title={file.name}>
                📄 {file.name}
              </span>
            ))}
          </div>
        ) : null}
        <div class="actions" hidden={!!props.live}>
          <button
            class="act copy-msg"
            type="button"
            title="Copy message"
            aria-label="Copy message"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
              <rect x="9" y="9" width="11" height="11" rx="2" />
              <path d="M5 15V5a1 1 0 0 1 1-1h9" />
            </svg>
          </button>
          <button
            class="act edit-msg"
            type="button"
            title="Edit and resend"
            aria-label="Edit and resend"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
              <path d="M17 3l4 4L8 20l-5 1 1-5z" />
            </svg>
          </button>
        </div>
      </div>
    );
  }

  const reasoning = msg.reasoning_content ?? "";
  const content = typeof msg.content === "string" ? msg.content : "";
  const clock = fmtClock(msg.created);
  const baseStats =
    msg.stats ||
    (msg.tps
      ? `${Number(msg.tps).toLocaleString(undefined, { maximumFractionDigits: 1 })} tok/s`
      : "");
  const savedStats = [baseStats, clock].filter(Boolean).join(" · ");

  return (
    <div
      class="message assistant"
      classList={{ live: !!props.live, error: !!props.error }}
      ref={mount}
    >
      <details hidden={!reasoning} open={!!props.live}>
        <summary>
          <span class="think-label">{props.live ? "Thinking…" : "Thought"}</span>
        </summary>
        <div class="reasoning">{displayContent(reasoning)}</div>
      </details>
      {/* Live turns fill this row themselves (paintToolRuns); a stored
          message renders the runs it finished with. */}
      <div class="tool-runs" hidden={!msg.tool_runs?.length}>
        <For each={msg.tool_runs ?? []}>
          {(run) => {
            const [open, setOpen] = createSignal(false);
            const title = () =>
              [
                run.detail,
                run.elapsed !== undefined ? `${run.elapsed.toFixed(1)}s` : "",
                run.result ? "click for the result" : "",
              ]
                .filter(Boolean)
                .join(" · ");
            return (
              <>
                <button
                  type="button"
                  class={`tool-chip ${run.status}`}
                  classList={{ open: open() }}
                  title={title() || undefined}
                  onClick={() => run.result && setOpen(!open())}
                >
                  {run.detail ? `${run.name} · ${run.detail}` : run.name}
                </button>
                <pre class="tool-out" hidden={!open()}>
                  {run.result ?? ""}
                </pre>
              </>
            );
          }}
        </For>
      </div>
      <div class="content" />
      <div
        class="waiting"
        hidden={!props.live || Boolean(content || reasoning)}
        role="status"
        aria-label="Generating response"
      />
      <div class="stats" hidden={!savedStats}>
        {savedStats}
      </div>
      <div class="actions" hidden={!!props.live}>
        <Show
          when={!props.error}
          fallback={
            <button
              class="act retry"
              type="button"
              title="Retry the request"
              aria-label="Retry the request"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
                <path d="M21 12a9 9 0 1 1-2.6-6.4" />
                <path d="M21 4v5h-5" />
              </svg>
            </button>
          }
        >
          <button
            class="act copy-msg"
            type="button"
            title="Copy response"
            aria-label="Copy response"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
              <rect x="9" y="9" width="11" height="11" rx="2" />
              <path d="M5 15V5a1 1 0 0 1 1-1h9" />
            </svg>
          </button>
          <button
            class="act regenerate"
            type="button"
            title="Regenerate"
            aria-label="Regenerate response"
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.8"
              stroke-linecap="round"
            >
              <path d="M21 12a9 9 0 1 1-2.6-6.4" />
              <path d="M21 4v5h-5" />
            </svg>
          </button>
        </Show>
      </div>
    </div>
  );
}

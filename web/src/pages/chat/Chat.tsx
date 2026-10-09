import {
  batch,
  createEffect,
  createMemo,
  createSignal,
  For,
  onCleanup,
  onMount,
  Show,
} from "solid-js";
import { A } from "@solidjs/router";
import { api, apiKey, ApiError, authorization, errorText, storageGet, storageSet } from "../../api";
import { pushToast } from "../../toast";
import { isEditableTarget } from "../../shortcuts";
import {
  chatMarkdown,
  conversations,
  deleteConversation,
  persistChat,
  renameConversation,
  requestMessages,
  statsText,
  titleFor,
  userParts,
  EFFORT_KEY,
  MODEL_KEY,
  SIDEBAR_KEY,
} from "./storage";
import { displayContent } from "./markdown";
import { activeTools, runTool, toolDetail } from "./tools";
import type { Attachment, Message, MessageContent, ModelInfo, ToolRun, Usage } from "./types";
import Sidebar from "./Sidebar";
import Composer from "./Composer";
import MessageView from "./Message";
import type { BoxHandle, MessageElement } from "./Message";
import "./chat.css";

interface StreamChunk {
  error?: { message?: string };
  prompt_progress?: { processed?: number; total?: number; cache?: number };
  metrics?: { request_latency?: { stream_tokens_per_second?: number } };
  usage?: Usage;
  choices?: {
    finish_reason?: string | null;
    delta?: {
      reasoning_content?: string;
      content?: string;
      tool_calls?: {
        index?: number;
        id?: string;
        function?: { name?: string; arguments?: string };
      }[];
    };
  }[];
}

/* One generation round can answer or end in calls; each call's id, name and
   argument text arrives spread across chunks, joined here by its index. */
interface ToolCallDraft {
  id: string;
  name: string;
  arguments: string;
}

interface StreamState {
  firstDeltaAt: number;
  finish: string | null;
  calls: (ToolCallDraft | undefined)[];
}

const ALLOWED_EFFORTS = new Set(["", "xhigh", "medium", "low", "none"]);
/* A model still calling after this many rounds is stuck; the last streamed
   answer ends the turn without honoring further calls. */
const MAX_TOOL_ROUNDS = 8;
// ~33% base64 inflation keeps 100 MB of file under the 128M request limit.
const MAX_FILE_BYTES = 100 * 1024 * 1024;

function draftKey(id: string | null): string {
  return `richengine:draft:${id ?? "<new>"}`;
}

function isAbort(error: unknown): boolean {
  return (error as { name?: string }).name === "AbortError";
}

export default function Chat() {
  const savedEffort = storageGet(EFFORT_KEY);
  const savedSidebar = storageGet(SIDEBAR_KEY);
  const [messages, setMessages] = createSignal<Message[]>([]);
  const [activeId, setActiveId] = createSignal<string | null>(null);
  const [attachments, setAttachments] = createSignal<Attachment[]>([]);
  const [attachmentsVersion, setAttachmentsVersion] = createSignal(0);
  const [input, setInput] = createSignal("");
  const [controller, setController] = createSignal<AbortController>();
  const [live, setLive] = createSignal<Message>();
  const [errorNote, setErrorNote] = createSignal<Message>();
  // The drawer exists only at the chat.css 720px breakpoint; on wide screens
  // the sidebar is always shown and the flag is just a remembered preference.
  const [menuOpen, setMenuOpen] = createSignal(
    savedSidebar !== null
      ? savedSidebar === "1"
      : !window.matchMedia("(max-width: 720px)").matches
  );
  const [modelData, setModelData] = createSignal<ModelInfo[]>([]);
  const [model, setModel] = createSignal(storageGet(MODEL_KEY) ?? "");
  const [effort, setEffort] = createSignal(
    ALLOWED_EFFORTS.has(savedEffort ?? "") ? savedEffort ?? "" : ""
  );
  // Until /v1/models says otherwise; a text-only model takes no attachments.
  const [imagesAccepted, setImagesAccepted] = createSignal(true);
  const [pdfsAccepted, setPdfsAccepted] = createSignal(true);
  const [serverDown, setServerDown] = createSignal(false);
  // A reachable server with no model loaded is its own state.
  const [noModel, setNoModel] = createSignal(false);
  const [modelLoading, setModelLoading] = createSignal(false);
  const [modelError, setModelError] = createSignal("");
  const [lightbox, setLightbox] = createSignal<string | null>(null);
  const [dragging, setDragging] = createSignal(0);
  const [tokenCount, setTokenCount] = createSignal<number | null>(null);
  const running = () => controller() !== undefined;

  let inputEl: HTMLTextAreaElement | undefined;
  let srEl!: HTMLDivElement;
  let toBottomEl!: HTMLButtonElement;
  let liveBox: BoxHandle | undefined;
  // Auto-scroll only follows when the reader is already at the bottom.
  let pinned = true;
  let paintQueued = false;

  // Capabilities and limits follow the selected model, falling back to the
  // server default (the first entry) when the picker is on "Default".
  const currentModel = () =>
    modelData().find((item) => item?.id === model()) ?? modelData()[0];
  const contextLen = () => Number(currentModel()?.context_length) || 0;

  const canSend = createMemo(() => {
    attachmentsVersion();
    return (
      Boolean(input().trim() || attachments().length) &&
      attachments().every((attachment) => attachment.url)
    );
  });

  function nearBottom() {
    return scrollY + innerHeight >= document.documentElement.scrollHeight - 80;
  }

  function maybeScroll() {
    if (pinned) scrollTo(0, document.documentElement.scrollHeight);
  }

  function pinAndScroll() {
    pinned = true;
    if (toBottomEl) toBottomEl.hidden = true;
    scrollTo(0, document.documentElement.scrollHeight);
  }

  function announce(text: string) {
    if (srEl) srEl.textContent = text;
  }

  // The draft lives in sessionStorage under the conversation it belongs to,
  // so a reload or a chat switch never loses an unsubmitted message.
  function saveDraft() {
    try {
      const text = input();
      if (text) sessionStorage.setItem(draftKey(activeId()), text);
      else sessionStorage.removeItem(draftKey(activeId()));
    } catch {
      /* Storage disabled or full: the draft stays in memory. */
    }
  }

  function restoreDraft() {
    let text = "";
    try {
      text = sessionStorage.getItem(draftKey(activeId())) ?? "";
    } catch {
      /* Same as above. */
    }
    setInput(text);
  }

  createEffect(saveDraft);
  createEffect(() => storageSet(SIDEBAR_KEY, menuOpen() ? "1" : "0"));

  async function loadModels() {
    try {
      const data = await api<{ data?: ModelInfo[] }>("/v1/models");
      const models = data?.data;
      if (!Array.isArray(models)) return;
      setModelData(models);
      setServerDown(false);
      setNoModel(models.length === 0);
      if (models.length === 0) void checkModelState();
      else {
        window.clearTimeout(modelTimer);
        setModelLoading(false);
        setModelError("");
      }
      const saved = storageGet(MODEL_KEY);
      setModel(saved && models.some((item) => item?.id === saved) ? saved : "");
      applyModel();
    } catch (error) {
      // An answered error is a reachable server; a network failure means the
      // server is gone — banner instead of just a toast.
      if (!(error instanceof ApiError)) setServerDown(true);
      pushToast("error", errorText(error));
    }
  }

  // An empty /v1/models can mean the host is still loading a model; /status
  // reports the lifecycle, so poll it until the list is worth re-fetching.
  let modelTimer: number | undefined;
  async function checkModelState() {
    window.clearTimeout(modelTimer);
    try {
      const status = await api<{ model_state?: string; model_error?: string }>(
        "/status"
      );
      const state = status?.model_state;
      if (state === "loading") {
        setModelLoading(true);
        setModelError("");
        modelTimer = window.setTimeout(() => void checkModelState(), 3000);
      } else if (state === "loaded") {
        setModelLoading(false);
        void loadModels();
      } else {
        setModelLoading(false);
        setModelError(status?.model_error ?? "");
      }
    } catch {
      /* A failed probe just keeps the no-model banner. */
    }
  }

  createEffect(() => {
    apiKey();
    void loadModels();
  });

  function applyModel() {
    const modalities = currentModel()?.input_modalities;
    if (!Array.isArray(modalities)) return;
    setImagesAccepted(modalities.includes("image"));
    setPdfsAccepted(modalities.includes("pdf"));
    if (!modalities.includes("image") && !modalities.includes("pdf"))
      setAttachments([]);
    else setAttachments((list) =>
      list.filter(
        (a) =>
          (a.kind === "image" && modalities.includes("image")) ||
          (a.kind === "pdf" && modalities.includes("pdf"))
      )
    );
  }

  function onModelChange(value: string) {
    setModel(value);
    storageSet(MODEL_KEY, value);
    applyModel();
  }

  function onEffortChange(value: string) {
    setEffort(value);
    storageSet(EFFORT_KEY, value);
  }

  function readFile(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  async function addFiles(files: File[]) {
    const accepted = files
      .filter(
        (file) =>
          (file.type.startsWith("image/") && imagesAccepted()) ||
          (file.type === "application/pdf" && pdfsAccepted())
      )
      .filter((file) => {
        if (file.size <= MAX_FILE_BYTES) return true;
        pushToast(
          "error",
          `${file.name || "File"} exceeds the 128 MB request limit`
        );
        return false;
      });
    if (!accepted.length) {
      if (files.length && !files.some((file) => file.size > MAX_FILE_BYTES))
        pushToast("error", "This model accepts no such attachments");
      return;
    }
    // Readers update stable entries, even if rollback moves them to a new draft.
    const selected: Attachment[] = accepted.map((file) => ({
      name: file.name,
      url: null,
      error: false,
      kind: file.type === "application/pdf" ? "pdf" : "image",
    }));
    setAttachments((list) => [...list, ...selected]);
    await Promise.all(
      accepted.map(async (file, index) => {
        const attachment = selected[index];
        try {
          attachment.url = await readFile(file);
        } catch {
          attachment.error = true;
          pushToast("error", `Could not read ${file.name || "file"}`);
        }
        if (attachments().includes(attachment))
          setAttachmentsVersion((version) => version + 1);
      })
    );
  }

  function onRemoveAttachment(attachment: Attachment) {
    setAttachments((list) => list.filter((value) => value !== attachment));
  }

  async function complete() {
    if (controller() || !canSend()) return;
    const text = input().trim();
    const pending = attachments();
    const content: MessageContent = pending.length
      ? [
          ...(text ? [{ type: "text" as const, text }] : []),
          ...pending.map((attachment) =>
            attachment.kind === "pdf"
              ? {
                  type: "file" as const,
                  file: { file_data: attachment.url!, filename: attachment.name },
                }
              : {
                  type: "image_url" as const,
                  image_url: { url: attachment.url! },
                }
          ),
        ]
      : text;
    setInput("");
    setAttachments([]);
    pinned = true;
    if (toBottomEl) toBottomEl.hidden = true;
    setMessages((list) => [
      ...list,
      { role: "user", content, created: Date.now() },
    ]);
    scrollTo(0, document.documentElement.scrollHeight);
    try {
      await streamReply(text || pending[0]?.name || "Image");
    } catch (error) {
      // A stopped stream keeps its partial output; other errors roll back.
      if (isAbort(error)) return;
      setMessages((list) => list.slice(0, -1));
      if (input() === "") setInput(text);
      // Restore the submitted entries alongside the next draft's pending reads.
      setAttachments((list) => [...pending, ...list]);
      setErrorNote({ role: "assistant", content: errorText(error) });
    }
  }

  function paintStats(box: HTMLElement, assistant: Message) {
    const stats = box.querySelector<HTMLElement>(".stats");
    if (!stats) return;
    const text = statsText(assistant, contextLen());
    stats.textContent = text;
    stats.hidden = !text;
  }

  // Re-rendering the whole reply every token is wasteful; coalesce paint and
  // the follow-scroll to one layout per frame.
  function schedulePaint(getText: () => string) {
    if (paintQueued) return;
    paintQueued = true;
    requestAnimationFrame(() => {
      paintQueued = false;
      const handle = liveBox;
      if (!handle?.el.isConnected) return;
      handle.painter.update(getText());
      maybeScroll();
    });
  }

  function consume(
    event: string,
    assistant: Message,
    state: StreamState
  ): boolean {
    const data = event
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data || data === "[DONE]") return data === "[DONE]";
    const chunk = JSON.parse(data) as StreamChunk;
    if (chunk.error) throw new Error(chunk.error.message);
    const box = liveBox?.el;
    const stats = box?.querySelector<HTMLElement>(".stats");
    const progress = chunk.prompt_progress;
    if (
      stats &&
      progress &&
      Number.isFinite(progress.processed) &&
      Number.isFinite(progress.total)
    ) {
      const cached = progress.cache
        ? ` (${progress.cache.toLocaleString()} cached)`
        : "";
      stats.textContent = `Prefill ${progress.processed!.toLocaleString()}/${progress.total!.toLocaleString()} tok${cached}`;
      stats.hidden = false;
    }
    const speed = Number(chunk.metrics?.request_latency?.stream_tokens_per_second);
    if (Number.isFinite(speed) && speed > 0) assistant.tps = speed;
    if (chunk.usage) assistant.usage = chunk.usage;
    if ((assistant.tps || assistant.usage) && box) paintStats(box, assistant);
    const choice = chunk.choices?.[0];
    if (choice?.finish_reason) state.finish = choice.finish_reason;
    const delta = choice?.delta;
    for (const call of delta?.tool_calls ?? []) {
      const index = call.index ?? 0;
      const entry =
        state.calls[index] ??
        (state.calls[index] = { id: "", name: "", arguments: "" });
      if (call.id) entry.id = call.id;
      if (call.function?.name) entry.name += call.function.name;
      if (call.function?.arguments) entry.arguments += call.function.arguments;
    }
    // A chunk without text has nothing to show or scroll to.
    if (!delta?.reasoning_content && !delta?.content) return false;
    const waiting = box?.querySelector<HTMLElement>(".waiting");
    if (waiting) waiting.hidden = true;
    if (delta.reasoning_content) {
      assistant.reasoning_content =
        (assistant.reasoning_content ?? "") + delta.reasoning_content;
      const details = box?.querySelector<HTMLDetailsElement>("details");
      if (details) {
        details.hidden = false;
        const reasoning = details.querySelector<HTMLElement>(".reasoning");
        if (reasoning)
          reasoning.textContent = displayContent(assistant.reasoning_content);
      }
    }
    if (delta.content) {
      if (!state.firstDeltaAt) state.firstDeltaAt = performance.now();
      assistant.content = (assistant.content as string) + delta.content;
      schedulePaint(() => assistant.content as string);
      // Before the server reports real throughput, estimate from streamed
      // text (~4 chars per token) so the reader sees progress right away.
      if (!assistant.tps && stats) {
        const elapsed = (performance.now() - state.firstDeltaAt) / 1000;
        if (elapsed > 0.5) {
          stats.textContent = `~${Math.round(
            (assistant.content as string).length / 4 / elapsed
          ).toLocaleString()} tok/s`;
          stats.hidden = false;
        }
      }
    }
    maybeScroll();
    return false;
  }

  function finishLive(assistant: Message) {
    const handle = liveBox;
    if (!handle?.el.isConnected) return;
    const waiting = handle.el.querySelector<HTMLElement>(".waiting");
    if (waiting) waiting.hidden = true;
    handle.painter.update(
      typeof assistant.content === "string" ? assistant.content : ""
    );
    paintStats(handle.el, assistant);
  }

  function commitAssistant(assistant: Message, firstPrompt: string) {
    batch(() => {
      setMessages((list) => [...list, assistant]);
      setLive(undefined);
    });
    const id = persistChat(firstPrompt, activeId(), messages());
    setActiveId(id);
    // A conversation created this turn earns its title from the reply;
    // the prompt-derived one stays when the user renamed meanwhile.
    if (
      conversations().find((item) => item.id === id)?.title ===
      titleFor(firstPrompt)
    ) {
      void autoTitle(id, firstPrompt);
    }
  }

  /* Ask the served model for a short title, once per new chat, in the
     background; a failure or a same-time rename simply keeps the prompt
     title. The title shares the one generation slot, so it waits for a
     quiet moment rather than queue a fast follow-up turn behind a
     24-token rename. */
  async function autoTitle(id: string, firstPrompt: string) {
    try {
      for (let attempt = 0; ; attempt++) {
        const status = await api<{ transport?: { pending?: number } }>(
          "/status"
        ).catch(() => null);
        const pending = status?.transport?.pending;
        // An unreadable status means try now anyway; a busy engine waits.
        if (pending === undefined || pending === 0) break;
        if (attempt >= 12) return;
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authorization() },
        body: JSON.stringify({
          model: model() || modelData()[0]?.id,
          stream: false,
          max_tokens: 24,
          messages: [
            {
              role: "user",
              content:
                "Write a title for a chat that begins with this message. " +
                "At most 6 words, plain text, no quotes or punctuation at the ends.\n\n" +
                firstPrompt.slice(0, 500),
            },
          ],
        }),
      });
      if (!response.ok) return;
      const data = await response.json();
      const raw: unknown = data?.choices?.[0]?.message?.content;
      if (typeof raw !== "string") return;
      const title = raw
        .replace(/^["'“”‘’`\s]+|["'“”‘’`\s.!?]+$/g, "")
        .replace(/\s+/g, " ")
        .slice(0, 60);
      const conversation = conversations().find((item) => item.id === id);
      if (
        !title ||
        !conversation ||
        conversation.title !== titleFor(firstPrompt)
      )
        return;
      renameConversation(id, title);
    } catch {
      // A title is nice to have; the prompt-derived one stays.
    }
  }

  /* Live boxes render the chips themselves: the runs list is written into
     the .tool-runs row MessageView always carries, replacing its children. */
  function toolChipTitle(run: ToolRun): string {
    return [
      run.detail,
      run.elapsed !== undefined ? `${run.elapsed.toFixed(1)}s` : "",
      run.result ? "click for the result" : "",
    ]
      .filter(Boolean)
      .join(" · ");
  }

  function paintToolRuns(assistant: Message) {
    const box = liveBox?.el;
    if (!box) return;
    const runs = assistant.tool_runs ?? [];
    if (runs.length) {
      const waiting = box.querySelector<HTMLElement>(".waiting");
      if (waiting) waiting.hidden = true;
    }
    const row = box.querySelector<HTMLElement>(".tool-runs");
    if (!row) return;
    row.hidden = !runs.length;
    row.replaceChildren(
      ...runs.flatMap((run) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = `tool-chip ${run.status}`;
        chip.textContent = run.detail ? `${run.name} · ${run.detail}` : run.name;
        chip.title = toolChipTitle(run);
        const out = document.createElement("pre");
        out.className = "tool-out";
        out.hidden = true;
        out.textContent = run.result ?? "";
        chip.addEventListener("click", () => {
          if (!run.result) return;
          out.hidden = !out.hidden;
          chip.classList.toggle("open", !out.hidden);
        });
        return [chip, out];
      })
    );
  }

  /* Run the calls a tool_calls finish produced and append the transcript —
     the round's own text, then each call and its result — to the wire
     history the next round resends. Calls execute in parallel (a search
     and a fetch should not wait on each other); results append to wire in
     call order once every call lands. */
  async function runToolCalls(
    wire: Record<string, unknown>[],
    assistant: Message,
    roundText: { content: string; reasoning: string },
    calls: ToolCallDraft[],
    round: number,
    signal: AbortSignal
  ) {
    wire.push({
      role: "assistant",
      content: roundText.content,
      ...(roundText.reasoning
        ? { reasoning_content: roundText.reasoning }
        : {}),
      tool_calls: calls.map((call, index) => ({
        id: call.id || `call_${round}_${index}`,
        type: "function",
        function: { name: call.name, arguments: call.arguments },
      })),
    });
    const jobs = calls.map((call) => {
      const run: ToolRun = { name: call.name || "tool", detail: "", status: "running" };
      let args: Record<string, unknown> | null = null;
      try {
        const parsed: unknown = call.arguments.trim()
          ? JSON.parse(call.arguments)
          : {};
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
          args = parsed as Record<string, unknown>;
      } catch {
        /* args stays null; the result below reports the bad call. */
      }
      if (args) run.detail = toolDetail(call.name, args);
      (assistant.tool_runs ??= []).push(run);
      return { call, run, args };
    });
    paintToolRuns(assistant);
    const results = await Promise.all(
      jobs.map(async ({ call, run, args }) => {
        const started = performance.now();
        let result: string;
        if (args === null) {
          run.status = "error";
          result = "Error: tool call arguments were not valid JSON";
        } else {
          try {
            result = await runTool(call.name, args, signal);
            run.status = "done";
          } catch (error) {
            run.status = "error";
            result = `Error: ${errorText(error)}`;
          }
        }
        run.elapsed = Math.round((performance.now() - started) / 100) / 10;
        run.result = result.length > 8000 ? `${result.slice(0, 8000)}…` : result;
        paintToolRuns(assistant);
        return result;
      })
    );
    for (const [index, { call }] of jobs.entries())
      wire.push({
        role: "tool",
        tool_call_id: call.id || `call_${round}_${index}`,
        content: results[index],
      });
  }

  // Streams the assistant reply for the user message already at the tail.
  // Regenerate truncates back to a user turn and calls this directly.
  async function streamReply(firstPrompt: string) {
    const assistant: Message = {
      role: "assistant",
      content: "",
      reasoning_content: "",
      created: Date.now(),
    };
    const state: StreamState = { firstDeltaAt: 0, finish: null, calls: [] };
    liveBox = undefined;
    setLive(assistant);
    maybeScroll();
    const ctrl = new AbortController();
    setController(ctrl);
    // A busy server queues the request; say so if nothing has streamed yet.
    const queueTimer = window.setTimeout(async () => {
      if (assistant.content || assistant.reasoning_content) return;
      const detail = await api<{ transport?: { pending?: number } }>(
        "/status"
      ).catch(() => null);
      const pending = detail?.transport?.pending;
      if (typeof pending === "number" && pending > 1 && liveBox) {
        const stats = liveBox.el.querySelector<HTMLElement>(".stats");
        if (stats && stats.hidden) {
          stats.textContent = `Queued · ${pending - 1} ahead`;
          stats.hidden = false;
        }
      }
    }, 600);
    try {
      // The wire history differs from the display one mid-turn: it also
      // carries the assistant's calls and their results until the model
      // answers. Neither side of that transcript is stored or shown.
      const wire = requestMessages(messages());
      const tools = activeTools();
      if (tools.some((tool) => tool.function.name === "memory_save")) {
        const note =
          "This chat has built-in memory tools: memory_save stores a fact " +
          "under a key, memory_list recalls every stored fact and " +
          "memory_delete forgets one. Memory persists across chats in this " +
          "browser.";
        const system = wire.find((message) => message.role === "system");
        if (system && typeof system.content === "string")
          system.content = `${system.content}\n\n${note}`;
        else wire.unshift({ role: "system", content: note });
      }
      let round = 0;
      while (true) {
        state.calls = [];
        state.finish = null;
        const contentStart = (assistant.content as string).length;
        const reasoningStart = (assistant.reasoning_content ?? "").length;
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authorization() },
          body: JSON.stringify({
            messages: wire,
            stream: true,
            stream_options: { include_usage: true },
            ...(model() ? { model: model() } : {}),
            ...(effort() ? { reasoning_effort: effort() } : {}),
            ...(tools.length ? { tools } : {}),
          }),
          signal: ctrl.signal,
        });
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as {
            error?: { message?: string };
          } | null;
          throw new Error(body?.error?.message || response.statusText);
        }
        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let doneEvent = false;
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop() ?? "";
          for (const event of events)
            doneEvent = consume(event, assistant, state) || doneEvent;
          if (done) break;
        }
        if (buffer) doneEvent = consume(buffer, assistant, state) || doneEvent;
        if (!doneEvent) throw new Error("Connection closed early");
        const calls = state.calls.filter(
          (call): call is ToolCallDraft => Boolean(call?.name)
        );
        if (
          state.finish !== "tool_calls" ||
          !calls.length ||
          round >= MAX_TOOL_ROUNDS
        ) {
          // The round cap leaves unanswered calls; say so instead of
          // dropping them silently.
          if (calls.length && round >= MAX_TOOL_ROUNDS) {
            (assistant.tool_runs ??= []).push({
              name: `stopped after ${MAX_TOOL_ROUNDS} tool rounds`,
              detail: "",
              status: "error",
              result:
                `The model kept calling tools: ${calls
                  .map((call) => call.name)
                  .join(", ")} ` +
                "went unanswered so the turn could end.",
            });
            paintToolRuns(assistant);
          }
          break;
        }
        round += 1;
        await runToolCalls(
          wire,
          assistant,
          {
            content: (assistant.content as string).slice(contentStart),
            reasoning: (assistant.reasoning_content ?? "").slice(reasoningStart),
          },
          calls,
          round,
          ctrl.signal
        );
        if (ctrl.signal.aborted)
          throw new DOMException("The operation was aborted.", "AbortError");
      }
      finishLive(assistant);
      assistant.stats = statsText(assistant, contextLen());
      announce(
        `Response complete${assistant.stats ? ` · ${assistant.stats}` : ""}`
      );
      commitAssistant(assistant, firstPrompt);
    } catch (error) {
      // Interrupted streams keep whatever arrived: a deliberate stop is marked
      // "stopped", a dropped connection "interrupted". Nothing arrived yet is
      // still an error the caller reports.
      const partial = Boolean(
        assistant.content || assistant.reasoning_content
      );
      if (!isAbort(error) && !partial) {
        setLive(undefined);
        throw error;
      }
      if (isAbort(error)) assistant.stopped = true;
      else assistant.interrupted = true;
      const label = assistant.stopped ? "stopped" : "interrupted";
      assistant.stats = statsText(assistant, contextLen()) || label;
      finishLive(assistant);
      announce(`Response ${label}`);
      commitAssistant(assistant, firstPrompt);
    } finally {
      window.clearTimeout(queueTimer);
      setController(undefined);
      liveBox = undefined;
      inputEl?.focus();
    }
  }

  // Edit a past user turn: pull its text and attachments back into the
  // composer and drop it and everything after.
  function editMessage(box: MessageElement) {
    if (controller()) return;
    const index = messages().indexOf(box._message as Message);
    if (index < 0 || messages()[index]?.role !== "user") return;
    const parts = userParts(messages()[index].content);
    setInput(parts.text);
    setAttachments([
      ...parts.images.map((url) => ({
        name: "image",
        url,
        error: false,
        kind: "image" as const,
      })),
      ...parts.files.map((file) => ({
        name: file.name,
        url: file.url,
        error: false,
        kind: "pdf" as const,
      })),
    ]);
    setMessages((list) => list.slice(0, index));
    setErrorNote(undefined);
    inputEl?.focus();
  }

  function copyMessage(box: MessageElement, button: HTMLElement) {
    const msg = box._message;
    if (!msg) return;
    const text =
      msg.role === "user"
        ? userParts(msg.content).text
        : typeof msg.content === "string"
          ? msg.content
          : "";
    void navigator.clipboard?.writeText(text).then(() => {
      button.classList.add("copied");
      setTimeout(() => button.classList.remove("copied"), 1200);
    });
  }

  function exportChat() {
    const list = messages();
    if (!list.length) {
      pushToast("error", "Nothing to export");
      return;
    }
    const conversation = conversations().find((item) => item.id === activeId());
    const title = conversation?.title || titleFor(userParts(list[0].content).text || "chat") || "chat";
    const blob = new Blob([chatMarkdown(title, list)], {
      type: "text/markdown",
    });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `${title.replace(/[^\w-]+/g, "-").slice(0, 50)}.md`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    pushToast("ok", "Chat exported as Markdown");
  }

  // Regenerate: drop the turn from this assistant message on and resend.
  function regenerate(box: MessageElement) {
    if (controller()) return;
    // The box carries its message entry; error notes hold none and miss here.
    const index = messages().indexOf(box._message as Message);
    if (index < 0 || messages()[index]?.role !== "assistant") return;
    const prompt = messages()
      .slice(0, index)
      .reverse()
      .find((message) => message.role === "user");
    const { text } = userParts(prompt?.content ?? "");
    setMessages((list) => list.slice(0, index));
    setErrorNote(undefined);
    pinned = true;
    streamReply(text || "Image").catch((error) => {
      if (isAbort(error)) return;
      setErrorNote({ role: "assistant", content: errorText(error) });
    });
  }

  function reset() {
    setAttachments([]);
    setErrorNote(undefined);
    setMenuOpen(false);
    pinAndScroll();
    inputEl?.focus();
  }

  function newChat() {
    if (controller()) {
      pushToast("error", "Stop the response before starting a new chat");
      return;
    }
    saveDraft();
    setActiveId(null);
    setMessages([]);
    restoreDraft();
    reset();
  }

  function openChat(id: string) {
    if (controller()) {
      pushToast("error", "Stop the response before switching chats");
      return;
    }
    const conversation = conversations().find((item) => item.id === id);
    if (!conversation) return;
    saveDraft();
    setActiveId(id);
    setMessages(conversation.messages);
    restoreDraft();
    reset();
  }

  function onDeleteConversation(id: string) {
    deleteConversation(id);
    if (activeId() === id) newChat();
    // After newChat's save, so the outgoing draft cannot resurrect under the
    // deleted id.
    try {
      sessionStorage.removeItem(draftKey(id));
    } catch {
      /* Storage disabled: the stale draft is harmless. */
    }
  }

  // ArrowUp in an empty composer pulls the last user turn back for resending.
  function recallLast() {
    const last = messages()
      .slice()
      .reverse()
      .find((message) => message.role === "user");
    if (!last) return;
    setInput(userParts(last.content).text);
  }

  function onChatClick(event: MouseEvent) {
    const target = event.target as HTMLElement;
    const copy = target.closest<HTMLElement>(".copy");
    if (copy) {
      const text =
        copy.closest(".codeblock")?.querySelector("code")?.textContent ?? "";
      navigator.clipboard?.writeText(text).then(() => {
        copy.textContent = "Copied";
        setTimeout(() => {
          copy.textContent = "Copy";
        }, 1200);
      });
      return;
    }
    const box = target.closest(".message") as MessageElement | null;
    const copyMsg = target.closest<HTMLElement>(".copy-msg");
    if (copyMsg && box) {
      copyMessage(box, copyMsg);
      return;
    }
    if (target.closest(".edit-msg")) {
      if (box) editMessage(box);
      return;
    }
    if (target.closest(".retry")) {
      void complete();
      return;
    }
    const retry = target.closest(".regenerate");
    if (retry) {
      if (box) regenerate(box);
      return;
    }
    const zoom = target.closest<HTMLImageElement>(".user-images img");
    if (zoom?.src) setLightbox(zoom.src);
  }

  // A hidden tab keeps generating; the title says so, then reports done.
  const baseTitle = document.title;
  let generated = false;
  createEffect(() => {
    if (running()) {
      generated = true;
      document.title = "● Generating — RichEngine";
    } else if (generated) {
      document.title = document.hidden ? "✓ Done — RichEngine" : baseTitle;
      if (!document.hidden) generated = false;
    } else {
      document.title = baseTitle;
    }
  });

  // Debounced token estimate for the draft, via the server's own counter.
  let tokenTimer: number | undefined;
  createEffect(() => {
    const draft = input();
    const history = messages();
    const picked = model() || modelData()[0]?.id;
    window.clearTimeout(tokenTimer);
    // Nothing to count: an empty history and draft 400s on the server.
    if (running() || (!draft.trim() && !history.length)) {
      setTokenCount(null);
      return;
    }
    tokenTimer = window.setTimeout(async () => {
      try {
        const res = await api<{ input_tokens?: number }>(
          "/v1/messages/count_tokens",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...(picked ? { model: picked } : {}),
              messages: [
                ...requestMessages(history),
                ...(draft.trim()
                  ? [{ role: "user", content: draft }]
                  : []),
              ],
            }),
          }
        );
        setTokenCount(
          typeof res?.input_tokens === "number" ? res.input_tokens : null
        );
      } catch {
        setTokenCount(null);
      }
    }, 400);
  });
  onCleanup(() => window.clearTimeout(tokenTimer));

  const tokenText = () => {
    const n = tokenCount();
    if (n === null) return "";
    const ctx = contextLen();
    return `≈${n.toLocaleString()} tok` +
      (ctx ? ` · ${Math.min(100, Math.round((n / ctx) * 100))}% ctx` : "");
  };

  const tokenPct = () => {
    const n = tokenCount();
    const ctx = contextLen();
    return n !== null && ctx ? Math.min(100, Math.round((n / ctx) * 100)) : 0;
  };

  onMount(() => {
    restoreDraft();
    const onScroll = () => {
      pinned = nearBottom();
      if (toBottomEl) toBottomEl.hidden = pinned;
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.key === "/" &&
        !event.metaKey && !event.ctrlKey && !event.altKey &&
        !isEditableTarget(event.target) &&
        !document.querySelector(".palette-overlay")
      ) {
        event.preventDefault();
        inputEl?.focus();
        return;
      }
      if (event.key !== "Escape") return;
      if (lightbox()) {
        setLightbox(null);
        return;
      }
      // The command palette owns Escape while it is open.
      if (controller() && !document.querySelector(".palette-overlay"))
        controller()!.abort();
    };
    const onNewChat = () => newChat();
    const onExport = () => exportChat();
    const onToggleSidebar = () => setMenuOpen((open) => !open);
    const onFocusInput = () => inputEl?.focus();
    const onVisible = () => {
      if (!document.hidden) document.title = baseTitle;
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("richengine:new-chat", onNewChat);
    window.addEventListener("richengine:export-chat", onExport);
    window.addEventListener("richengine:toggle-sidebar", onToggleSidebar);
    window.addEventListener("richengine:focus-input", onFocusInput);
    onCleanup(() => {
      window.removeEventListener("scroll", onScroll);
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("richengine:new-chat", onNewChat);
      window.removeEventListener("richengine:export-chat", onExport);
      window.removeEventListener("richengine:toggle-sidebar", onToggleSidebar);
      window.removeEventListener("richengine:focus-input", onFocusInput);
      window.clearTimeout(modelTimer);
      document.title = baseTitle;
    });
  });

  const acceptsFiles = () => imagesAccepted() || pdfsAccepted();

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(0);
    const files = Array.from(event.dataTransfer?.files ?? []);
    if (files.length) void addFiles(files);
  }

  return (
    <div
      class="chat-shell"
      classList={{
        empty: !messages().length,
        "menu-open": menuOpen(),
        dragging: dragging() > 0,
      }}
      onDragEnter={(event) => {
        if (acceptsFiles()) {
          event.preventDefault();
          setDragging((n) => n + 1);
        }
      }}
      onDragOver={(event) => {
        if (acceptsFiles()) event.preventDefault();
      }}
      onDragLeave={(event) => {
        if (!event.relatedTarget) setDragging(0);
        else setDragging((n) => Math.max(0, n - 1));
      }}
      onDrop={onDrop}
    >
      <Sidebar
        open={menuOpen()}
        activeId={activeId()}
        disabled={running()}
        onNew={newChat}
        onOpen={openChat}
        onDelete={onDeleteConversation}
        onRename={(id, title) => renameConversation(id, title)}
      />
      <div class="scrim" onClick={() => setMenuOpen(false)} />
      <button
        class="menu-btn"
        type="button"
        aria-label={menuOpen() ? "Close sidebar" : "Open sidebar"}
        aria-expanded={menuOpen()}
        onClick={() => setMenuOpen(!menuOpen())}
      >
        <svg
          width="19"
          height="19"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.8"
        >
          <path d="M4 7h16M4 12h16M4 17h16" />
        </svg>
      </button>
      <main>
        <div
          id="chat"
          role="log"
          aria-live="off"
          aria-busy={running()}
          aria-relevant="additions"
          aria-label="Conversation"
          onClick={onChatClick}
        >
          <Show when={serverDown()}>
            <div class="down-banner" role="alert">
              Server unreachable — start one with
              {" "}<code>richengine serve --model &lt;MODEL&gt;</code>, then{" "}
              <button
                type="button"
                class="down-retry"
                onClick={() => void loadModels()}
              >
                retry
              </button>
              .
            </div>
          </Show>
          <Show when={modelLoading() && !serverDown()}>
            <div class="down-banner" role="status">Model is loading…</div>
          </Show>
          <Show when={noModel() && !serverDown() && !modelLoading()}>
            <div class="down-banner" role="status">
              {modelError() || (
                <>
                  No model loaded — open{" "}
                  <A href="/models" class="down-retry">Models</A> to load one.
                </>
              )}
            </div>
          </Show>
          <Show when={!messages().length}>
            <div id="empty">
              How can I help?
              <div class="empty-hints">
                <span>⌘K commands</span>
                <span>Drop images or PDFs to attach</span>
                <span>/ focuses the composer</span>
              </div>
            </div>
          </Show>
          <For each={messages()}>{(message) => <MessageView msg={message} />}</For>
          <Show when={live()} keyed>
            {(assistant) => (
              <MessageView
                msg={assistant}
                live
                register={(handle) => (liveBox = handle)}
              />
            )}
          </Show>
          <Show when={errorNote()} keyed>
            {(note) => <MessageView msg={note} error />}
          </Show>
        </div>
      </main>
      <Show when={dragging() > 0}>
        <div class="drop-overlay">Drop to attach</div>
      </Show>
      <Show when={lightbox()}>
        {(src) => (
          <div
            class="lightbox"
            role="dialog"
            aria-label="Image preview"
            onClick={() => setLightbox(null)}
          >
            <img src={src()} alt="Attached image, enlarged" />
          </div>
        )}
      </Show>
      <div id="sr-status" class="visually-hidden" role="status" ref={srEl} />
      <button
        id="to-bottom"
        type="button"
        hidden
        aria-label="Scroll to latest message"
        title="Jump to latest (auto-scroll resumes)"
        onClick={pinAndScroll}
      >
        ↓ Latest
      </button>
      <Composer
        running={running()}
        canSend={canSend()}
        attachments={attachments()}
        attachmentsVersion={attachmentsVersion()}
        imagesAccepted={imagesAccepted()}
        pdfsAccepted={pdfsAccepted()}
        tokenText={tokenText()}
        tokenPct={tokenPct()}
        models={modelData()}
        model={model()}
        effort={effort()}
        input={input()}
        onSend={() => void complete()}
        onToggle={() => {
          const ctrl = controller();
          if (ctrl) ctrl.abort();
          else void complete();
        }}
        onInput={setInput}
        onModelChange={onModelChange}
        onEffortChange={onEffortChange}
        onAddFiles={(files) => void addFiles(files)}
        onRemoveAttachment={onRemoveAttachment}
        onRecall={recallLast}
        inputRef={(el) => (inputEl = el)}
      />
    </div>
  );
}

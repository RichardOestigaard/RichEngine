import { createSignal } from "solid-js";
import { storageGet, storageSet } from "../../api";
import { fmtClock } from "../../format";
import type { Conversation, Message, MessageContent, Part, ToolRun } from "./types";

export const STORE_KEY = "richengine-chats";
export const MODEL_KEY = "richengine-chat-model";
export const EFFORT_KEY = "richengine-thinking-effort";
export const SYSTEM_KEY = "richengine-system-prompt";
export const SIDEBAR_KEY = "richengine:sidebar";

function cleanUserContent(content: unknown): MessageContent | null {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;
  const clean = content.flatMap((part): Part[] => {
    if (!part || typeof part !== "object") return [];
    const entry = part as Record<string, unknown>;
    if (entry.type === "text" && typeof entry.text === "string")
      return [{ type: "text", text: entry.text }];
    if (entry.type === "image_url") {
      const url = (entry.image_url as Record<string, unknown> | undefined)?.url;
      if (typeof url === "string" && url.startsWith("data:image/"))
        return [{ type: "image_url", image_url: { url } }];
    }
    if (entry.type === "file") {
      const file = entry.file as Record<string, unknown> | undefined;
      const data = file?.file_data;
      if (typeof data === "string" && data.startsWith("data:application/pdf")) {
        const filename = file?.filename;
        return [
          {
            type: "file",
            file: {
              file_data: data,
              ...(typeof filename === "string" ? { filename } : {}),
            },
          },
        ];
      }
    }
    return [];
  });
  return clean.length ? clean : null;
}

function cleanMessage(message: unknown): Message | null {
  if (!message || typeof message !== "object") return null;
  const entry = message as Record<string, unknown>;
  if (entry.role !== "user" && entry.role !== "assistant") return null;
  const content = entry.role === "user" ? cleanUserContent(entry.content) : entry.content;
  if (content === null || (entry.role === "assistant" && typeof content !== "string")) return null;
  const clean: Message = { role: entry.role, content: content as MessageContent };
  if (entry.role === "assistant") {
    if (typeof entry.reasoning_content === "string") clean.reasoning_content = entry.reasoning_content;
    if (typeof entry.tps === "number" && Number.isFinite(entry.tps) && entry.tps > 0) clean.tps = entry.tps;
    if (typeof entry.stats === "string") clean.stats = entry.stats;
    if (Array.isArray(entry.tool_runs)) {
      const runs = entry.tool_runs.flatMap((run): ToolRun[] => {
        if (!run || typeof run !== "object") return [];
        const item = run as Record<string, unknown>;
        if (typeof item.name !== "string" || !item.name) return [];
        return [{
          name: item.name,
          detail: typeof item.detail === "string" ? item.detail.slice(0, 200) : "",
          status: item.status === "running" || item.status === "error" ? item.status : "done",
          ...(typeof item.result === "string" && item.result
            ? { result: item.result.slice(0, 4000) }
            : {}),
          ...(typeof item.elapsed === "number" && Number.isFinite(item.elapsed)
            ? { elapsed: item.elapsed }
            : {}),
        }];
      });
      // A stored chat reopens finished: a run still marked running shows done.
      if (runs.length) clean.tool_runs = runs.map(run => run.status === "running" ? { ...run, status: "done" } : run);
    }
  }
  return clean;
}

function load(): Conversation[] {
  try {
    const stored: unknown = JSON.parse(storageGet(STORE_KEY) ?? "null");
    if (!Array.isArray(stored)) return [];
    const seen = new Set<string>();
    return stored.flatMap((conversation): Conversation[] => {
      if (!conversation || typeof conversation !== "object") return [];
      const entry = conversation as Record<string, unknown>;
      if (typeof entry.id !== "string" || seen.has(entry.id) ||
          typeof entry.title !== "string" ||
          typeof entry.updated !== "number" || !Number.isFinite(entry.updated) ||
          !Array.isArray(entry.messages)) return [];
      const messages = entry.messages.map(cleanMessage).filter((m): m is Message => m !== null);
      if (!messages.length) return [];
      seen.add(entry.id);
      return [{ id: entry.id, title: entry.title, updated: entry.updated, messages }];
    });
  } catch { return []; }
}

const [conversations, setConversations] = createSignal<Conversation[]>(load());
export { conversations };

function hasMedia(conversation: Conversation): boolean {
  return conversation.messages.some(message => Array.isArray(message.content) &&
    (message.content as Part[]).some(part => part.type !== "text"));
}

function withoutMedia(conversation: Conversation): Conversation {
  const messages = conversation.messages.map(message => Array.isArray(message.content) ? {
    ...message,
    content: (message.content as Part[]).map((part): Part =>
      part.type === "text" ? part
        : { type: "text", text: part.type === "image_url" ? "[Image not saved]" : "[Document not saved]" })
  } : message);
  return { ...conversation, messages };
}

// The storage write serializes every chat, images included; it runs on an
// idle slice so a completed turn never blocks the next keystroke, and is
// flushed synchronously when the page goes away.
let saveScheduled = false;
const idleRun: (fn: () => void, options?: { timeout: number }) => void =
  typeof window.requestIdleCallback === "function"
    ? (fn, options) => { window.requestIdleCallback(fn, options); }
    : (fn) => { setTimeout(fn, 300); };

function persistChats(): void {
  // Browser storage holds a few MB per site, which one photo can fill. Rather
  // than fail every later save, store the newest chats that fit, each with its
  // images if they still fit and without them otherwise, so a photo too large
  // to store does not cost older chats theirs. The open page keeps its images.
  const all = conversations();
  if (!storageSet(STORE_KEY, JSON.stringify(all))) {
    const stored: Conversation[] = [];
    for (const conversation of all) {
      const candidates = hasMedia(conversation) ? [conversation, withoutMedia(conversation)] : [conversation];
      const fitting = candidates.find(candidate => storageSet(STORE_KEY, JSON.stringify([...stored, candidate])));
      if (!fitting) break;
      stored.push(fitting);
    }
  }
}

export function scheduleSave(): void {
  setConversations([...conversations()].sort((a, b) => b.updated - a.updated).slice(0, 20));
  if (saveScheduled) return;
  saveScheduled = true;
  idleRun(() => { saveScheduled = false; persistChats(); }, { timeout: 2000 });
}

addEventListener("pagehide", () => {
  if (saveScheduled) { saveScheduled = false; persistChats(); }
});

// Find-or-create the conversation for activeId, point it at the live message
// list, and return its id so the caller can track it as active.
export function persistChat(firstPrompt: string, activeId: string | null, messages: Message[]): string {
  let conversation = conversations().find(item => item.id === activeId);
  if (!conversation) {
    conversation = { id: chatId(), title: titleFor(firstPrompt), messages, updated: Date.now() };
    setConversations([...conversations(), conversation]);
  }
  if (!conversation.title) conversation.title = titleFor(firstPrompt);
  conversation.messages = messages;
  conversation.updated = Date.now();
  scheduleSave();
  return conversation.id;
}

export function deleteConversation(id: string): void {
  setConversations(conversations().filter(conversation => conversation.id !== id));
  scheduleSave();
}

export function renameConversation(id: string, title: string): void {
  const conversation = conversations().find(item => item.id === id);
  const clean = title.replace(/\s+/g, " ").trim();
  if (conversation && clean && clean !== conversation.title) {
    conversation.title = clean.length > 60 ? `${clean.slice(0, 60)}…` : clean;
    scheduleSave();
  } else {
    // Signal equivalent of the original renderRecents(): reset the row.
    setConversations([...conversations()]);
  }
}

export function titleFor(text: string): string {
  const title = text.replace(/\s+/g, " ").trim();
  return title.length > 34 ? `${title.slice(0, 34)}…` : title;
}

// crypto.randomUUID exists only in secure contexts, which excludes a LAN
// address over plain HTTP; getRandomValues works everywhere.
export function chatId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)),
    byte => byte.toString(16).padStart(2, "0")).join("");
}

export function userParts(content: MessageContent): { text: string; images: string[]; files: { name: string; url: string }[] } {
  if (typeof content === "string") return { text: content, images: [], files: [] };
  return {
    text: content.flatMap(part => part.type === "text" ? [part.text] : []).join("\n"),
    images: content.flatMap(part => part.type === "image_url" ? [part.image_url.url] : []),
    files: content.flatMap(part =>
      part.type === "file"
        ? [{ name: part.file.filename || "document.pdf", url: part.file.file_data }]
        : [])
  };
}

// Client-side annotations never go back to the server: speed, usage and the
// stopped/interrupted flags are display state, not request fields. A system
// prompt set in the sidebar travels as the first message.
export function requestMessages(messages: Message[]): Record<string, unknown>[] {
  const cleaned = messages.map(
    ({ tps, usage, stats, tool_runs, stopped, interrupted, created, ...message }) => message
  );
  const system = storageGet(SYSTEM_KEY)?.trim();
  return system
    ? [{ role: "system", content: system }, ...cleaned]
    : cleaned;
}

/** The conversation as a Markdown document for download/copy. */
export function chatMarkdown(title: string, messages: Message[]): string {
  const out = [`# ${title}`, ""];
  for (const message of messages) {
    const clock = fmtClock(message.created);
    const when = clock ? ` · ${clock}` : "";
    if (message.role === "user") {
      const parts = userParts(message.content);
      out.push(`## You${when}`, "");
      if (parts.text) out.push(parts.text, "");
      for (const file of parts.files) out.push(`📎 ${file.name}`, "");
      if (parts.images.length) out.push(`[${parts.images.length} image(s) attached]`, "");
    } else {
      out.push(`## RichEngine${when}`, "");
      const reasoning = message.reasoning_content?.trim();
      if (reasoning)
        out.push("> **Thinking**", ...reasoning.split("\n").map(line => `> ${line}`), "");
      const content = typeof message.content === "string" ? message.content.trim() : "";
      if (content) out.push(content, "");
      if (message.stats) out.push(`_${message.stats}_`, "");
    }
  }
  return out.join("\n");
}

// Live stats during the stream, real usage and context share once the
// final usage chunk arrives. The last rendered line is kept on the message
// so a reloaded chat shows the same numbers.
export function statsText(assistant: Message, contextLen: number): string {
  const parts: string[] = [];
  if (assistant.tps) parts.push(`${Number(assistant.tps).toLocaleString(undefined, { maximumFractionDigits: 1 })} tok/s`);
  const usage = assistant.usage;
  if (usage) {
    const total = usage.prompt_tokens + usage.completion_tokens;
    parts.push(`${total.toLocaleString()} tok` + (contextLen ? ` · ${Math.min(100, Math.round(total / contextLen * 100))}% ctx` : ""));
    const cached = usage.prompt_tokens_details?.cached_tokens;
    if (cached) parts.push(`${cached.toLocaleString()} cached`);
  }
  if (assistant.stopped) parts.push("stopped");
  if (assistant.interrupted) parts.push("interrupted");
  return parts.join(" · ");
}

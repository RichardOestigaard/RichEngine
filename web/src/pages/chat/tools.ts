import { createSignal } from "solid-js";
import { apiPost } from "../../api";

/* The chat page's built-in tools. They ride the OpenAI tools field the
   engine already supports: the page lists them per request, runs the calls
   the model emits, and feeds the results back for the next round. */

export type ToolId = "time" | "fetch" | "search" | "memory";
export const TOOL_IDS: ToolId[] = ["time", "fetch", "search", "memory"];
export const TOOL_LABELS: Record<ToolId, string> = {
  time: "Time & date",
  fetch: "Web fetch",
  search: "Web search",
  memory: "Memory",
};
export const TOOL_HINTS: Record<ToolId, string> = {
  time: "Current local time and date",
  fetch: "Read a public web page",
  search: "Search the configured engine",
  memory: "Remember facts across chats",
};

const TOOLS_KEY = "richengine:chat-tools";
const MEMORY_KEY = "richengine:memory";
const SEARCH_PROVIDER_KEY = "richengine:search-provider";
const SEARCH_API_KEY_KEY = "richengine:search-api-key";
const SEARCH_INSTANCE_KEY = "richengine:search-instance";
const MAX_MEMORY_ENTRIES = 200;
const MAX_MEMORY_KEY_CHARS = 100;
const MAX_MEMORY_VALUE_CHARS = 4000;

function loadEnabled(): Record<ToolId, boolean> {
  const enabled = { time: true, fetch: true, search: true, memory: true } as Record<
    ToolId,
    boolean
  >;
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(TOOLS_KEY) ?? "{}");
    if (stored && typeof stored === "object")
      for (const id of TOOL_IDS) {
        const value = (stored as Record<string, unknown>)[id];
        if (typeof value === "boolean") enabled[id] = value;
      }
  } catch {
    /* Storage disabled or corrupt: the defaults stand. */
  }
  return enabled;
}

const [toolsEnabled, setToolsEnabled] =
  createSignal<Record<ToolId, boolean>>(loadEnabled());
export { toolsEnabled };

/* The web_search provider picks in Settings: bing needs nothing, brave and
   tavily take the stored API key, searxng the stored instance URL. */
export const SEARCH_PROVIDERS = ["bing", "brave", "tavily", "searxng"] as const;
export type SearchProvider = (typeof SEARCH_PROVIDERS)[number];
export const SEARCH_PROVIDER_LABELS: Record<SearchProvider, string> = {
  bing: "Bing (no key)",
  brave: "Brave Search API",
  tavily: "Tavily",
  searxng: "SearXNG instance",
};
export const SEARCH_PROVIDER_NEEDS_KEY: Record<SearchProvider, boolean> = {
  bing: false,
  brave: true,
  tavily: true,
  searxng: false,
};

function storedString(key: string, fallback = ""): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function storedProvider(): SearchProvider {
  const value = storedString(SEARCH_PROVIDER_KEY, "bing");
  return SEARCH_PROVIDERS.includes(value as SearchProvider)
    ? (value as SearchProvider)
    : "bing";
}

const [searchProvider, setSearchProviderSignal] =
  createSignal<SearchProvider>(storedProvider());
const [searchApiKey, setSearchApiKeySignal] = createSignal(
  storedString(SEARCH_API_KEY_KEY)
);
const [searchInstance, setSearchInstanceSignal] = createSignal(
  storedString(SEARCH_INSTANCE_KEY)
);
export { searchProvider, searchApiKey, searchInstance };

function persist(key: string, value: string): void {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* The setting still applies to this tab. */
  }
}

export function setSearchProvider(value: SearchProvider): void {
  setSearchProviderSignal(value);
  persist(SEARCH_PROVIDER_KEY, value);
}

export function setSearchApiKey(value: string): void {
  setSearchApiKeySignal(value);
  persist(SEARCH_API_KEY_KEY, value);
}

export function setSearchInstance(value: string): void {
  setSearchInstanceSignal(value);
  persist(SEARCH_INSTANCE_KEY, value);
}

export function setToolEnabled(id: ToolId, on: boolean): void {
  const next = { ...toolsEnabled(), [id]: on };
  setToolsEnabled(next);
  try {
    localStorage.setItem(TOOLS_KEY, JSON.stringify(next));
  } catch {
    /* The toggle still applies to this tab. */
  }
}

interface ToolDef {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    // Not strict: a strict tool's arguments are grammar-generated, which
    // fails on dialects whose call markup is a special token (minicpm5).
    // Free-form calls still read back type-converted by the schema.
  };
}

const DEFS: { id: ToolId; def: ToolDef }[] = [
  {
    id: "time",
    def: {
      type: "function",
      function: {
        name: "get_current_time",
        description:
          "Get the current local date, time, timezone and weekday. Call this before answering anything time-sensitive.",
        parameters: { type: "object", properties: {}, additionalProperties: false },
      },
    },
  },
  {
    id: "fetch",
    def: {
      type: "function",
      function: {
        name: "web_fetch",
        description:
          "Fetch a public web page and return its text content. Use it for a URL the user mentions or for current information. Private and local addresses are refused.",
        parameters: {
          type: "object",
          properties: {
            url: { type: "string", description: "The http(s) URL to fetch" },
          },
          required: ["url"],
          additionalProperties: false,
        },
      },
    },
  },
  {
    id: "search",
    def: {
      type: "function",
      function: {
        name: "web_search",
        description:
          "Search the web and return matching pages as title, URL and snippet. Use it for current information or to find a page before reading it with web_fetch.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string", description: "The search query" },
            count: {
              type: "integer",
              description: "How many results to return, at most 10",
            },
          },
          required: ["query"],
          additionalProperties: false,
        },
      },
    },
  },
  {
    id: "memory",
    def: {
      type: "function",
      function: {
        name: "memory_save",
        description:
          "Remember a fact across chats in this browser. Store it under a short lowercase key, overwriting any fact already under it.",
        parameters: {
          type: "object",
          properties: {
            key: { type: "string", description: "Short key, e.g. user_name" },
            value: { type: "string", description: "The fact to remember" },
          },
          required: ["key", "value"],
          additionalProperties: false,
        },
      },
    },
  },
  {
    id: "memory",
    def: {
      type: "function",
      function: {
        name: "memory_list",
        description: "List every fact remembered across chats: key, value and when it was saved.",
        parameters: { type: "object", properties: {}, additionalProperties: false },
      },
    },
  },
  {
    id: "memory",
    def: {
      type: "function",
      function: {
        name: "memory_delete",
        description: "Forget a remembered fact by its key.",
        parameters: {
          type: "object",
          properties: {
            key: { type: "string", description: "The key to forget" },
          },
          required: ["key"],
          additionalProperties: false,
        },
      },
    },
  },
];

export function activeTools(): ToolDef[] {
  const on = toolsEnabled();
  return DEFS.filter((entry) => on[entry.id]).map((entry) => entry.def);
}

/* What the tool chip shows next to the tool name while it runs. */
export function toolDetail(name: string, args: Record<string, unknown>): string {
  if (name === "web_search" && typeof args.query === "string")
    return args.query.slice(0, 60);
  if (name === "web_fetch" && typeof args.url === "string") {
    try {
      return new URL(args.url).hostname;
    } catch {
      return args.url.slice(0, 60);
    }
  }
  if (
    (name === "memory_save" || name === "memory_delete") &&
    typeof args.key === "string"
  )
    return args.key;
  return "";
}

interface MemoryEntry {
  value: string;
  updated: number;
}

function readMemory(): Record<string, MemoryEntry> {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(MEMORY_KEY) ?? "{}");
    if (!stored || typeof stored !== "object" || Array.isArray(stored))
      return {};
    const entries: Record<string, MemoryEntry> = {};
    for (const [key, entry] of Object.entries(stored)) {
      const item = entry as Record<string, unknown>;
      if (typeof item?.value === "string")
        entries[key] = { value: item.value, updated: Number(item.updated) || 0 };
    }
    return entries;
  } catch {
    return {};
  }
}

function writeMemory(entries: Record<string, MemoryEntry>): string | null {
  try {
    localStorage.setItem(MEMORY_KEY, JSON.stringify(entries));
    return null;
  } catch {
    return "storage is unavailable or full; the fact was not saved";
  }
}

function memorySave(args: Record<string, unknown>): string {
  const key = String(args.key ?? "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .slice(0, MAX_MEMORY_KEY_CHARS);
  const value = String(args.value ?? "").slice(0, MAX_MEMORY_VALUE_CHARS);
  if (!key || !value) return "Error: key and value are required";
  const entries = readMemory();
  if (!(key in entries) && Object.keys(entries).length >= MAX_MEMORY_ENTRIES) {
    // Full: drop the oldest entry rather than refuse the save.
    const oldest = Object.entries(entries).sort(
      (a, b) => a[1].updated - b[1].updated
    )[0];
    if (oldest) delete entries[oldest[0]];
  }
  entries[key] = { value, updated: Date.now() };
  const problem = writeMemory(entries);
  return problem ? `Error: ${problem}` : `Saved memory "${key}".`;
}

function memoryList(): string {
  const entries = readMemory();
  const sorted = Object.entries(entries).sort(
    (a, b) => b[1].updated - a[1].updated
  );
  if (!sorted.length) return "No memories are stored.";
  return JSON.stringify(
    sorted.map(([key, entry]) => ({
      key,
      value: entry.value,
      saved: new Date(entry.updated).toISOString(),
    }))
  );
}

function memoryDelete(args: Record<string, unknown>): string {
  const key = String(args.key ?? "").trim().toLowerCase();
  const entries = readMemory();
  if (!(key in entries)) return `No memory is stored under "${key}".`;
  delete entries[key];
  const problem = writeMemory(entries);
  return problem ? `Error: ${problem}` : `Deleted memory "${key}".`;
}

/* Run one call and answer the text the model sees as its result. Tool
   failures return text too: a call that errored keeps the turn alive. */
export async function runTool(
  name: string,
  args: Record<string, unknown>,
  signal?: AbortSignal
): Promise<string> {
  switch (name) {
    case "get_current_time": {
      const now = new Date();
      return JSON.stringify({
        iso: now.toISOString(),
        local: now.toLocaleString(undefined, {
          dateStyle: "full",
          timeStyle: "long",
        }),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        unix_seconds: Math.floor(now.getTime() / 1000),
      });
    }
    case "web_fetch": {
      if (typeof args.url !== "string" || !args.url.trim())
        return "Error: url is required";
      try {
        return JSON.stringify(
          await apiPost("/v1/tools/fetch", { url: args.url.trim() }, signal)
        );
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    }
    case "web_search": {
      if (typeof args.query !== "string" || !args.query.trim())
        return "Error: query is required";
      const provider = searchProvider();
      if (SEARCH_PROVIDER_NEEDS_KEY[provider] && !searchApiKey().trim())
        return `Error: ${provider} search needs an API key — set it in Settings → Web search`;
      if (provider === "searxng" && !searchInstance().trim())
        return "Error: searxng needs an instance URL — set it in Settings → Web search";
      try {
        return JSON.stringify(
          await apiPost(
            "/v1/tools/search",
            {
              query: args.query.trim(),
              provider,
              api_key: searchApiKey().trim(),
              instance: searchInstance().trim(),
              ...(typeof args.count === "number" && Number.isFinite(args.count)
                ? { count: Math.round(args.count) }
                : {}),
            },
            signal
          )
        );
      } catch (error) {
        return `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    }
    case "memory_save":
      return memorySave(args);
    case "memory_list":
      return memoryList();
    case "memory_delete":
      return memoryDelete(args);
    default:
      return `Error: unknown tool "${name}"`;
  }
}

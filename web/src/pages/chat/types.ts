export type Part =
  | { type: "text"; text: string; name?: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "file"; file: { file_data: string; filename?: string } };
export type MessageContent = string | Part[];
export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  prompt_tokens_details?: { cached_tokens?: number };
}
export interface ToolRun {
  name: string;
  detail: string;
  status: "running" | "done" | "error";
  /* The tool's reply, shown when the chip expands; seconds the call took. */
  result?: string;
  elapsed?: number;
}
export interface Message {
  role: "user" | "assistant";
  content: MessageContent;
  created?: number;
  reasoning_content?: string;
  tps?: number;
  usage?: Usage;
  stats?: string;
  tool_runs?: ToolRun[];
  /* The reply's declared shape; "json" renders the raw content inside a
     json code block. The wire keeps the model's own text either way. */
  format?: "json";
  /* Speculative counts the done event reported (metrics.decode): draft
     proposals verified, and how many of them the output accepted. */
  drafted?: number;
  accepted_draft?: number;
  stopped?: boolean;
  interrupted?: boolean;
  /* Seconds the reasoning block streamed for; shown on the fold once the
     answer starts (or the turn ends with thinking unfinished). */
  think_secs?: number;
}
export interface Conversation {
  id: string;
  title: string;
  updated: number;
  messages: Message[];
}
export interface Attachment {
  name: string;
  /* For text attachments this is the raw pasted/file text, not a data URL. */
  url: string | null;
  error: boolean;
  kind: "image" | "pdf" | "text";
}
/* Enter during a run stashes the turn here instead of dropping it; queued
   turns stack in a tray above the composer and drain in order once the
   reply ends — becoming real messages only when their send starts. */
export interface QueuedTurn {
  content: MessageContent;
  text: string;
  attachments: Attachment[];
}
export interface ModelInfo {
  id: string;
  context_length?: number;
  input_modalities?: string[];
}

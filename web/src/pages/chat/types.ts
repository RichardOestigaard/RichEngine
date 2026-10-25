export type Part =
  | { type: "text"; text: string }
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
  stopped?: boolean;
  interrupted?: boolean;
}
export interface Conversation {
  id: string;
  title: string;
  updated: number;
  messages: Message[];
}
export interface Attachment {
  name: string;
  url: string | null;
  error: boolean;
  kind: "image" | "pdf";
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

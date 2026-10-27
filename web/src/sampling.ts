import { createSignal } from "solid-js";

/* Per-model sampling overrides sent on chat requests. A field left unset
   keeps the served model's own default (its generation config, else the
   chat endpoint's — server/frontend.py SAMPLING_NUMBERS + TOP_K_DEFAULT).
   Stored in one localStorage map keyed by the install selection name, the
   same string requests put in `model`. */

export interface SamplingSettings {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  repetition_penalty?: number;
}

export interface SamplingField {
  key: keyof SamplingSettings;
  label: string;
  hint: string;
  step: number;
  integer?: boolean;
  /* The range the server validates; an out-of-range value 400s the
     request, so the form refuses it at save time. */
  valid: (value: number) => boolean;
  range: string;
}

export const SAMPLING_FIELDS: SamplingField[] = [
  {
    key: "temperature",
    label: "Temperature",
    hint: "0 is greedy",
    step: 0.05,
    valid: (v) => v >= 0 && v <= 2,
    range: "0 – 2",
  },
  {
    key: "top_p",
    label: "Top P",
    hint: "nucleus cutoff",
    step: 0.01,
    valid: (v) => v > 0 && v <= 1,
    range: "(0, 1]",
  },
  {
    key: "top_k",
    label: "Top K",
    hint: "0 keeps every token",
    step: 1,
    integer: true,
    valid: (v) => Number.isInteger(v) && v >= 0,
    range: "integer ≥ 0",
  },
  {
    key: "min_p",
    label: "Min P",
    hint: "floor vs the top token's mass",
    step: 0.01,
    valid: (v) => v >= 0 && v <= 1,
    range: "0 – 1",
  },
  {
    key: "presence_penalty",
    label: "Presence penalty",
    hint: "token seen before",
    step: 0.05,
    valid: (v) => v >= -2 && v <= 2,
    range: "-2 – 2",
  },
  {
    key: "frequency_penalty",
    label: "Frequency penalty",
    hint: "scaled by occurrences",
    step: 0.05,
    valid: (v) => v >= -2 && v <= 2,
    range: "-2 – 2",
  },
  {
    key: "repetition_penalty",
    label: "Repetition penalty",
    hint: "1 leaves logits unchanged",
    step: 0.01,
    valid: (v) => v > 0,
    range: "> 0",
  },
];

const SAMPLING_KEY = "richengine:model-sampling";

function loadMap(): Record<string, SamplingSettings> {
  try {
    const stored: unknown = JSON.parse(
      localStorage.getItem(SAMPLING_KEY) ?? "{}"
    );
    if (stored && typeof stored === "object")
      return stored as Record<string, SamplingSettings>;
  } catch {
    /* Storage disabled or corrupt: defaults stand. */
  }
  return {};
}

const [samplingMap, setSamplingMap] =
  createSignal<Record<string, SamplingSettings>>(loadMap());

export function samplingFor(model: string | undefined): SamplingSettings {
  return (model && samplingMap()[model]) || {};
}

export function hasSamplingFor(model: string | undefined): boolean {
  return Object.keys(samplingFor(model)).length > 0;
}

export function saveSamplingFor(
  model: string,
  next: SamplingSettings
): void {
  const map = { ...samplingMap() };
  if (Object.keys(next).length) map[model] = next;
  else delete map[model];
  setSamplingMap(map);
  try {
    localStorage.setItem(SAMPLING_KEY, JSON.stringify(map));
  } catch {
    /* The setting still applies to this tab. */
  }
}

/* The request-body fields of a model's overrides: only set, finite values,
   ready to spread. */
export function requestSampling(
  model: string | undefined
): Partial<SamplingSettings> {
  const out: Partial<SamplingSettings> = {};
  const settings = samplingFor(model);
  for (const field of SAMPLING_FIELDS) {
    const value = settings[field.key];
    if (typeof value === "number" && Number.isFinite(value))
      out[field.key] = value;
  }
  return out;
}

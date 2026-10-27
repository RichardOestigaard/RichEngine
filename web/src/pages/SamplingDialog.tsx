import { createEffect, createSignal, For, Show } from "solid-js";
import {
  Dialog,
  DialogDescription,
  DialogOverlay,
  DialogPanel,
  DialogTitle,
} from "terracotta";
import { pushToast } from "../toast";
import {
  SAMPLING_FIELDS,
  samplingFor,
  saveSamplingFor,
  type SamplingSettings,
} from "../sampling";

/* The per-model sampling editor the Models page's cog opens: number fields
   the request sends verbatim, empty meaning the model's own default. Saved
   under the install selection name chat requests use. Styles reuse the
   .confirm-* modal chrome plus .sampling-* in models.css. */
export default function SamplingDialog(props: {
  model: string | null;
  onClose: () => void;
}) {
  const [values, setValues] = createSignal<Record<string, string>>({});
  const [invalid, setInvalid] = createSignal<Set<string>>(new Set());

  // Re-open (or switch models) reloads the stored overrides.
  createEffect(() => {
    const model = props.model;
    if (!model) return;
    const stored = samplingFor(model);
    const next: Record<string, string> = {};
    for (const field of SAMPLING_FIELDS) {
      const value = stored[field.key];
      next[field.key] = value === undefined ? "" : String(value);
    }
    setValues(next);
    setInvalid(new Set<string>());
  });

  function save() {
    const model = props.model;
    if (!model) return;
    const next: SamplingSettings = {};
    const bad = new Set<string>();
    for (const field of SAMPLING_FIELDS) {
      const raw = (values()[field.key] ?? "").trim();
      if (!raw) continue;
      const value = Number(raw);
      if (!Number.isFinite(value) || !field.valid(value)) {
        bad.add(field.key);
        continue;
      }
      next[field.key] = value;
    }
    setInvalid(bad);
    if (bad.size) {
      const names = SAMPLING_FIELDS.filter((field) => bad.has(field.key))
        .map((field) => field.label)
        .join(", ");
      pushToast("error", `${names}: out of range`);
      return;
    }
    saveSamplingFor(model, next);
    pushToast(
      "ok",
      Object.keys(next).length
        ? `Saved sampling for ${model}`
        : `Sampling for ${model} back to defaults`
    );
    props.onClose();
  }

  return (
    <Dialog
      isOpen={props.model !== null}
      onChange={(open) => {
        if (!open) props.onClose();
      }}
    >
      <DialogOverlay class="confirm-overlay" />
      <DialogPanel class="confirm-panel sampling-panel">
        <DialogTitle as="div" class="confirm-title">
          Sampling · {props.model}
        </DialogTitle>
        <DialogDescription class="confirm-desc">
          Overrides sent on every chat request. Empty keeps the model's own
          defaults.
        </DialogDescription>
        <div class="sampling-grid">
          <For each={SAMPLING_FIELDS}>
            {(field) => (
              <label
                class="sampling-field"
                classList={{ invalid: invalid().has(field.key) }}
              >
                <span class="sampling-label">{field.label}</span>
                <input
                  type="number"
                  inputmode="decimal"
                  step={field.step}
                  placeholder={field.range}
                  value={values()[field.key] ?? ""}
                  onInput={(event) => {
                    setValues({
                      ...values(),
                      [field.key]: event.currentTarget.value,
                    });
                    if (invalid().has(field.key)) {
                      const next = new Set(invalid());
                      next.delete(field.key);
                      setInvalid(next);
                    }
                  }}
                />
                <span class="sampling-hint">{field.hint}</span>
              </label>
            )}
          </For>
        </div>
        <div class="confirm-actions">
          <button
            type="button"
            class="btn"
            onClick={() => {
              const cleared: Record<string, string> = {};
              for (const field of SAMPLING_FIELDS) cleared[field.key] = "";
              setValues(cleared);
              setInvalid(new Set<string>());
            }}
          >
            Reset
          </button>
          <span style="flex: 1" />
          <button type="button" class="btn" onClick={props.onClose}>
            Cancel
          </button>
          <button type="button" class="btn primary" onClick={save}>
            Save
          </button>
        </div>
        <Show when={props.model}>
          <p class="sampling-note muted">
            Saved for <span class="mono">{props.model}</span> — applies to
            requests naming this model.
          </p>
        </Show>
      </DialogPanel>
    </Dialog>
  );
}

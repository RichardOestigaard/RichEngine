import { createSignal } from "solid-js";
import { storageGet, storageSet } from "./api";

/* Developer mode (Settings → Developer): surfaces engine internals — draft
   acceptance per reply and the model's sampling overrides on the composer.
   Persisted like the theme. */
const DEV_MODE_KEY = "richengine:dev-mode";

const [devMode, setDevModeSignal] = createSignal(
  storageGet(DEV_MODE_KEY) === "1"
);

export function setDevMode(on: boolean): void {
  setDevModeSignal(on);
  storageSet(DEV_MODE_KEY, on ? "1" : "0");
}

export { devMode };

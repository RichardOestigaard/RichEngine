import { createSignal } from "solid-js";
import { storageGet, storageSet } from "./api";

/* Theme preference: "auto" follows the OS (unset attribute falls back to the
   prefers-color-scheme rules), the rest force their palette. Applied to
   documentElement.dataset.theme at import time so first paint matches. */

export type Theme = "auto" | "light" | "dark" | "black";

const KEY = "richengine-theme";
const THEMES: Theme[] = ["auto", "light", "dark", "black"];

export const THEME_LABELS: Record<Theme, string> = {
  auto: "Auto",
  light: "Light",
  dark: "Dark",
  black: "Black",
};

function read(): Theme {
  const value = storageGet(KEY) as Theme | null;
  return value && THEMES.includes(value) ? value : "auto";
}

function apply(value: Theme) {
  document.documentElement.dataset.theme = value;
}

const [theme, setThemeSignal] = createSignal<Theme>(read());
export { theme };

export function setTheme(value: Theme) {
  setThemeSignal(value);
  storageSet(KEY, value);
  apply(value);
}

export function cycleTheme() {
  setTheme(THEMES[(THEMES.indexOf(theme()) + 1) % THEMES.length]);
}

apply(read());

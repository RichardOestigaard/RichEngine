import { createSignal } from "solid-js";

/* Leaf module shared by TopBar (calls togglePalette) and palette/Palette.tsx
   (renders the overlays). No imports from either, so no import cycle. */

const [paletteOpen, setPaletteOpen] = createSignal(false);
const [shortcutsOpen, setShortcutsOpen] = createSignal(false);

export { paletteOpen, shortcutsOpen };

export function openPalette() {
  setPaletteOpen(true);
}

export function closePalette() {
  setPaletteOpen(false);
}

export function togglePalette() {
  setPaletteOpen((open) => !open);
}

export function openShortcuts() {
  setShortcutsOpen(true);
}

export function closeShortcuts() {
  setShortcutsOpen(false);
}

export function toggleShortcuts() {
  setShortcutsOpen((open) => !open);
}

/** True while focus is somewhere `?`/letter keys would type text. */
export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

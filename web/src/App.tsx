import type { ParentProps } from "solid-js";
import { createEffect, onCleanup, Show } from "solid-js";
import { useLocation, useNavigate } from "@solidjs/router";
import TopBar from "./TopBar";
import Toasts from "./toast";
import Palette from "./palette/Palette";

const TITLES: [RegExp, string][] = [
  [/^\/models/, "Models"],
  [/^\/disk/, "Disk"],
  [/^\/playground/, "Playground"],
  [/^\/judge/, "Judge"],
  [/^\/metrics/, "Metrics"],
  [/^\/settings/, "Settings"],
];

export default function App(props: ParentProps) {
  const location = useLocation();
  const navigate = useNavigate();
  /* Per-route document titles: browser tabs and screen readers announce
     the page instead of a static name. */
  createEffect(() => {
    const found = TITLES.find(([pattern]) => pattern.test(location.pathname));
    document.title = found ? `${found[1]} — RichEngine` : "RichEngine";
  });
  /* Module-scope code (install.ts) can't useNavigate; it asks and App
     answers. */
  const onNavigate = (event: Event) =>
    navigate((event as CustomEvent<string>).detail);
  window.addEventListener("richengine:navigate", onNavigate);
  onCleanup(() =>
    window.removeEventListener("richengine:navigate", onNavigate)
  );
  return (
    <>
      <a href="#content" class="skip-link">Skip to content</a>
      <TopBar />
      {/* A div, not main: Chat renders its own <main> landmark. Keying on
          the path remounts each page into the page-in animation. */}
      <div id="content" tabindex="-1" class="page-host">
        <Show when={location.pathname} keyed>
          {props.children}
        </Show>
      </div>
      <Toasts />
      <Palette />
    </>
  );
}

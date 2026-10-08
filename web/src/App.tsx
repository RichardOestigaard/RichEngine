import type { ParentProps } from "solid-js";
import { createEffect } from "solid-js";
import { useLocation } from "@solidjs/router";
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
  /* Per-route document titles: browser tabs and screen readers announce
     the page instead of a static name. */
  createEffect(() => {
    const found = TITLES.find(([pattern]) => pattern.test(location.pathname));
    document.title = found ? `${found[1]} — RichEngine` : "RichEngine";
  });
  return (
    <>
      <a href="#content" class="skip-link">Skip to content</a>
      <TopBar />
      {/* A div, not main: Chat renders its own <main> landmark. */}
      <div id="content" tabindex="-1" class="page-host">
        {props.children}
      </div>
      <Toasts />
      <Palette />
    </>
  );
}

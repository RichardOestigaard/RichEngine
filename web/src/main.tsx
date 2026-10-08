import { render } from "solid-js/web";
import { HashRouter, Route } from "@solidjs/router";
import App from "./App";
import Chat from "./pages/chat/Chat";
import Models from "./pages/Models";
import Disk from "./pages/Disk";
import Judge from "./pages/Judge";
import Metrics from "./pages/Metrics";
import Playground from "./pages/Playground";
import Settings from "./pages/Settings";
import "./styles.css";

render(
  () => (
    <HashRouter root={App}>
      <Route path="/" component={Chat} />
      <Route path="/models" component={Models} />
      <Route path="/disk" component={Disk} />
      <Route path="/judge" component={Judge} />
      <Route path="/metrics" component={Metrics} />
      <Route path="/playground" component={Playground} />
      <Route path="/settings" component={Settings} />
    </HashRouter>
  ),
  document.getElementById("root")!
);

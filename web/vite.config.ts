import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

export default defineConfig({
  plugins: [solid()],
  // `vite dev` proxies the API to a `richengine serve` instance; override
  // the target with RICHENGINE_SERVER or RICHENGINE_PORT.
  server: {
    proxy: Object.fromEntries(
      ["/v1", "/status", "/ready", "/health", "/metrics", "/tokenize", "/apply-template"].map(
        (path) => [
          path,
          process.env.RICHENGINE_SERVER ??
            `http://127.0.0.1:${process.env.RICHENGINE_PORT ?? "8000"}`,
        ]
      )
    ),
  },
  build: {
    // Served by server.py at / with assets under /assets/.
    outDir: "../server/webui",
    emptyOutDir: true,
  },
});

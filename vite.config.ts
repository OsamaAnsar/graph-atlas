import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
  base: "./",
  resolve: {
    alias: {
      "node:async_hooks": fileURLToPath(new URL("./src/shims/async-hooks.ts", import.meta.url)),
      async_hooks: fileURLToPath(new URL("./src/shims/async-hooks.ts", import.meta.url)),
    },
  },
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
});

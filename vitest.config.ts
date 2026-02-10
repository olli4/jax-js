import path from "node:path";

import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      // During testing, resolve @jax-js/jax to the TypeScript source so that
      // stack traces (e.g. from checkLeaks) show original file:line:col
      // instead of dist/index.js positions.
      "@jax-js/jax": path.resolve(__dirname, "src/index.ts"),
    },
  },
  esbuild: {
    supported: {
      using: false, // Needed to lower 'using' statements in tests.
    },
  },
  test: {
    watch: false, // Run once and exit, don't wait for 'q'
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright(),
      // https://vitest.dev/config/browser/playwright.html
      instances: [{ browser: "chromium" }],
    },
    coverage: {
      // coverage is disabled by default, run with `pnpm test:coverage`.
      enabled: false,
      provider: "v8",
    },
    passWithNoTests: true,
    exclude: ["**/node_modules/**", "**/dist/**", "test/deno/**", "tmp/**"],
    setupFiles: ["test/setup.ts"],
  },
});

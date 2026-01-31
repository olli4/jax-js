import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  esbuild: {
    supported: {
      using: false, // Needed to lower 'using' statements in tests.
    },
  },
  test: {
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright(),
      // https://vitest.dev/config/browser/playwright.html
      instances: [
        {
          browser: "chromium",
          launch: {
            args: [
              // WebGPU on headless server (requires NVIDIA Vulkan ICD)
              "--headless=new",
              "--use-angle=vulkan",
              "--enable-features=Vulkan",
              "--disable-vulkan-surface",
              "--enable-unsafe-webgpu",
              "--disable-software-rasterizer",
              "--no-sandbox",
            ],
          },
        },
      ],
    },
    coverage: {
      // coverage is disabled by default, run with `pnpm test:coverage`.
      enabled: false,
      provider: "v8",
    },
    exclude: ["**/node_modules/**", "**/dist/**", "test/deno/**"],
    setupFiles: ["test/setup.ts"],
  },
});

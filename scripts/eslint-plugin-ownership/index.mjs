/**
 * Local ESLint plugin for jax-js ownership principles.
 *
 * Rules here intentionally target the repository's explicit ownership model:
 * - explicit retention boundaries via `.ref`
 * - symmetric release via `.dispose()` / transfer / return
 */

import requireRetainedRelease from "./rules/require-retained-release.mjs";
import requireTryFinallySymmetry from "./rules/require-try-finally-symmetry.mjs";
import requireWrapperDisposeSymmetry from "./rules/require-wrapper-dispose-symmetry.mjs";

const plugin = {
  meta: {
    name: "@local/jax-ownership",
    version: "0.0.1",
  },
  rules: {
    "require-retained-release": requireRetainedRelease,
    "require-try-finally-symmetry": requireTryFinallySymmetry,
    "require-wrapper-dispose-symmetry": requireWrapperDisposeSymmetry,
  },
  configs: {
    recommended: {
      plugins: {
        "@ownership": null,
      },
      rules: {
        "@ownership/require-retained-release": "warn",
        "@ownership/require-try-finally-symmetry": "warn",
        "@ownership/require-wrapper-dispose-symmetry": "warn",
      },
    },
  },
};

plugin.configs.recommended.plugins["@ownership"] = plugin;

export default plugin;

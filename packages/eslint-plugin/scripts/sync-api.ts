#!/usr/bin/env tsx
/**
 * sync-api.ts — Auto-generate the API surface lists from jax-js source code.
 *
 * Reads `src/frontend/core.ts` (Tracer class) and `src/frontend/array.ts`
 * (Array class + factory functions), extracts public getters, methods,
 * and exported factory functions, and writes the result to
 * `src/api-surface.generated.ts`.
 *
 * Usage:
 *   pnpm --filter @jax-js/eslint-plugin sync
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const ROOT = resolve(__dirname, "../../..");
const OUT = resolve(__dirname, "../src/api-surface.generated.ts");

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

function readSource(relPath: string): string {
  return readFileSync(resolve(ROOT, relPath), "utf-8");
}

interface ClassMembers {
  getters: string[];
  methods: string[];
}

/**
 * Extract public getters and methods from a class body.
 *
 * Matches:
 *   `  get name()` → getter
 *   `  name(`      → method
 *   `  async name(`→ method
 *
 * Skips constructors, private/internal names (`_`/`#`), and well-known
 * internal helpers.
 */
function extractClassMembers(
  source: string,
  classPattern: RegExp,
): ClassMembers {
  const lines = source.split("\n");
  const startIdx = lines.findIndex((l) => classPattern.test(l));
  if (startIdx === -1) throw new Error(`Class not found: ${classPattern}`);

  // Find the closing brace at depth 0.
  let depth = 0;
  let endIdx = startIdx;
  for (let i = startIdx; i < lines.length; i++) {
    for (const ch of lines[i]) {
      if (ch === "{") depth++;
      if (ch === "}") depth--;
    }
    if (depth === 0) {
      endIdx = i;
      break;
    }
  }

  const body = lines.slice(startIdx, endIdx + 1).join("\n");

  const getters: string[] = [];
  for (const m of body.matchAll(/^\s{2}get (\w+)\s*\(/gm)) {
    getters.push(m[1]);
  }

  const SKIP = new Set([
    "constructor",
    "get",
    "set",
    "async",
    "static",
    "fullLower", // internal tracing
    "updateRc", // internal rc management
    "prepare", // internal pending execution
    "prepareSync", // internal pending execution
    "submit", // internal pending execution
    "processPrimitive", // internal tracing
  ]);

  const methods: string[] = [];
  for (const m of body.matchAll(/^\s{2}(?:async )?([a-zA-Z]\w*)\s*[(<]/gm)) {
    const name = m[1];
    if (SKIP.has(name) || name.startsWith("_") || getters.includes(name)) {
      continue;
    }
    methods.push(name);
  }

  return {
    getters: [...new Set(getters)],
    methods: [...new Set(methods)],
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const coreSource = readSource("src/frontend/core.ts");
const arraySource = readSource("src/frontend/array.ts");

const tracer = extractClassMembers(
  coreSource,
  /^export abstract class Tracer\b/,
);
const array = extractClassMembers(
  arraySource,
  /^export class Array extends Tracer\b/,
);

// Merge getters and methods from both classes (Array extends Tracer).
const allGetters = [...new Set([...tracer.getters, ...array.getters])].sort();
const allMethods = [...new Set([...tracer.methods, ...array.methods])].sort();

const output = `\
// AUTO-GENERATED — do not edit manually.
// Run \`pnpm --filter @jax-js/eslint-plugin sync\` to regenerate.
//
// Source: scripts/sync-api.ts reads src/frontend/core.ts and
// src/frontend/array.ts, extracting public getters and methods.
//
// The derived constant sets (NON_CONSUMING_PROPS, ARRAY_RETURNING_METHODS,
// CONSUMING_TERMINAL_METHODS) are computed in shared.ts from these lists.

/** Public getters on Tracer and Array classes (non-consuming accesses). */
export const EXTRACTED_GETTERS = [
${allGetters.map((n) => `  "${n}",`).join("\n")}
] as const;

/** Public methods on Tracer and Array classes (consuming operations). */
export const EXTRACTED_METHODS = [
${allMethods.map((n) => `  "${n}",`).join("\n")}
] as const;
`;

writeFileSync(OUT, output, "utf-8");
console.log(`Wrote ${OUT}`);
console.log(`  ${allGetters.length} getters, ${allMethods.length} methods`);

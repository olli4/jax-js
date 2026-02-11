/**
 * sync-lists.test.ts
 *
 * Verifies that the auto-generated API surface file is up to date
 * with the actual jax-js source code, and that the manual sets
 * maintain their invariants.
 *
 * If a developer adds a new method or getter to the Tracer or Array
 * class, this test will fail with:
 *
 *   "api-surface.generated.ts is out of date.
 *    Run: pnpm --filter @jax-js/eslint-plugin sync"
 *
 * After re-running `sync`, the new name is automatically picked up
 * by NON_CONSUMING_PROPS or ARRAY_RETURNING_METHODS (derived sets).
 *
 * The only cases requiring manual edits after `sync`:
 *   - New terminal method (returns non-Array) → add to CONSUMING_TERMINAL_METHODS
 *   - New method unique to jax-js → add to UNAMBIGUOUS_ARRAY_METHODS
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  EXTRACTED_GETTERS,
  EXTRACTED_METHODS,
} from "../src/api-surface.generated";
import {
  ARRAY_RETURNING_METHODS,
  UNAMBIGUOUS_ARRAY_METHODS,
} from "../src/shared";

const __dir = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dir, "../../..");

// ---------------------------------------------------------------------------
// Extraction logic (mirrors scripts/sync-api.ts)
// ---------------------------------------------------------------------------

function readSource(relPath: string): string {
  return readFileSync(resolve(ROOT, relPath), "utf-8");
}

function extractClassMembers(
  source: string,
  classPattern: RegExp,
): { getters: string[]; methods: string[] } {
  const lines = source.split("\n");
  const startIdx = lines.findIndex((l) => classPattern.test(l));
  if (startIdx === -1) throw new Error(`Class not found: ${classPattern}`);

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
  const SKIP = new Set([
    "constructor",
    "get",
    "set",
    "async",
    "static",
    "fullLower",
    "updateRc",
    "prepare",
    "prepareSync",
    "submit",
    "processPrimitive",
  ]);

  const getters: string[] = [];
  for (const m of body.matchAll(/^\s{2}get (\w+)\s*\(/gm)) getters.push(m[1]);

  const methods: string[] = [];
  for (const m of body.matchAll(/^\s{2}(?:async )?([a-zA-Z]\w*)\s*[(<]/gm)) {
    const n = m[1];
    if (!SKIP.has(n) && !n.startsWith("_") && !getters.includes(n))
      methods.push(n);
  }

  return { getters: [...new Set(getters)], methods: [...new Set(methods)] };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("sync-lists", () => {
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

  const freshGetters = [
    ...new Set([...tracer.getters, ...array.getters]),
  ].sort();
  const freshMethods = [
    ...new Set([...tracer.methods, ...array.methods]),
  ].sort();

  it("EXTRACTED_GETTERS matches current source", () => {
    expect(
      [...EXTRACTED_GETTERS],
      "api-surface.generated.ts is out of date.\n" +
        "Run: pnpm --filter @jax-js/eslint-plugin sync",
    ).toEqual(freshGetters);
  });

  it("EXTRACTED_METHODS matches current source", () => {
    expect(
      [...EXTRACTED_METHODS],
      "api-surface.generated.ts is out of date.\n" +
        "Run: pnpm --filter @jax-js/eslint-plugin sync",
    ).toEqual(freshMethods);
  });

  it("UNAMBIGUOUS_ARRAY_METHODS is a subset of ARRAY_RETURNING_METHODS", () => {
    for (const name of UNAMBIGUOUS_ARRAY_METHODS) {
      expect(
        ARRAY_RETURNING_METHODS.has(name),
        `UNAMBIGUOUS_ARRAY_METHODS contains "${name}" which is not in ARRAY_RETURNING_METHODS`,
      ).toBe(true);
    }
  });
});

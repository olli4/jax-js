// AUTO-GENERATED — do not edit manually.
// Run `pnpm --filter @jax-js/eslint-plugin sync` to regenerate.
//
// Source: scripts/sync-api.ts reads src/frontend/core.ts and
// src/frontend/array.ts, extracting public getters and methods.
//
// The derived constant sets (NON_CONSUMING_PROPS, ARRAY_RETURNING_METHODS,
// CONSUMING_TERMINAL_METHODS) are computed in shared.ts from these lists.

/** Public getters on Tracer and Array classes (non-consuming accesses). */
export const EXTRACTED_GETTERS = [
  "aval",
  "device",
  "dtype",
  "ndim",
  "ref",
  "refCount",
  "shape",
  "size",
  "weakType",
] as const;

/** Public methods on Tracer and Array classes (consuming operations). */
export const EXTRACTED_METHODS = [
  "add",
  "all",
  "any",
  "argsort",
  "astype",
  "blockUntilReady",
  "data",
  "dataSync",
  "diagonal",
  "dispose",
  "div",
  "equal",
  "flatten",
  "greater",
  "greaterEqual",
  "item",
  "js",
  "jsAsync",
  "less",
  "lessEqual",
  "max",
  "mean",
  "min",
  "mod",
  "mul",
  "neg",
  "notEqual",
  "prod",
  "ravel",
  "reshape",
  "slice",
  "sort",
  "sub",
  "sum",
  "toString",
  "transpose",
] as const;

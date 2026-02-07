/**
 * @file Scan plan construction — determines the execution strategy for a scan.
 *
 * P0: fallback-only. P2–P4 will add compiled-loop and preencoded-routine.
 */

import type { Backend, Executable } from "../backend";
import { byteWidth } from "../alu";
import type { ScanPath } from "../utils";
import type { Jaxpr } from "./jaxpr";
import type { JitProgram } from "./jit";

// ---------------------------------------------------------------------------
// ScanPlan: a discriminated union of execution strategies
// ---------------------------------------------------------------------------

export type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | { path: "compiled-loop"; executable: Executable; params?: any }
  | { path: "preencoded-routine"; preencodedParams: any };

// ---------------------------------------------------------------------------
// Path acceptance checking (for testing / debugging)
// ---------------------------------------------------------------------------

/**
 * Check if a chosen scan path satisfies the acceptPath constraint.
 * Returns an error message if the path is not allowed, or null if OK.
 *
 * Special case: an empty array `[]` always rejects, showing the chosen path.
 */
export function checkAcceptedPath(
  chosenPath: ScanPath,
  acceptPath: string | string[] | undefined,
  extraInfo?: string,
): string | null {
  if (!acceptPath) return null;

  const allowedPaths = Array.isArray(acceptPath) ? acceptPath : [acceptPath];
  const suffix = extraInfo ? ` (${extraInfo})` : "";

  if (allowedPaths.length === 0) {
    return `Scan path debug: chose "${chosenPath}"${suffix}`;
  }

  if (!allowedPaths.includes(chosenPath)) {
    return (
      `Scan acceptPath constraint not satisfied: ` +
      `got "${chosenPath}" but accepted paths are [${allowedPaths.map((p) => `"${p}"`).join(", ")}]${suffix}`
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// Buffer size helpers (shared by backends in later phases)
// ---------------------------------------------------------------------------

/**
 * Extract buffer sizes and strides from body jaxpr for native scan codegen.
 * Shared by WebGPU and WASM native scan implementations.
 */
export function getScanBufferSizes(
  bodyJaxpr: Jaxpr,
  numConsts: number,
  numCarry: number,
  numX: number,
) {
  const constAvals = bodyJaxpr.inBinders
    .slice(0, numConsts)
    .map((v) => v.aval);
  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry, numConsts + numCarry + numX)
    .map((v) => v.aval);
  const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);

  return {
    constSizes: constAvals.map((a) => a.size * byteWidth(a.dtype)),
    carrySizes: carryAvals.map((a) => a.size * byteWidth(a.dtype)),
    xsStrides: xAvals.map((a) => a.size * byteWidth(a.dtype)),
    ysStrides: yAvals.map((a) => a.size * byteWidth(a.dtype)),
  };
}

// ---------------------------------------------------------------------------
// planScan: decide which execution strategy to use
// ---------------------------------------------------------------------------

/**
 * Choose a scan execution strategy.
 *
 * Priority: compiled-loop > preencoded-routine > fallback.
 * P0 always returns fallback. P2–P4 will add compiled-loop and
 * preencoded-routine paths.
 */
export function planScan(
  _backend: Backend,
  bodyProgram: JitProgram,
  _bodyJaxpr: Jaxpr,
  _length: number,
  _numCarry: number,
  _numConsts: number,
  _numX: number,
  _numY: number,
  _reverse: boolean,
  acceptPath?: ScanPath | ScanPath[],
): ScanPlan {
  // P0: fallback only. P2–P4 will try compiled-loop and preencoded-routine
  // before falling through here.

  const dispatchCount = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ).length;
  const extraInfo = `${dispatchCount} dispatch${dispatchCount !== 1 ? "es" : ""} per iteration`;

  const pathError = checkAcceptedPath("fallback", acceptPath, extraInfo);
  if (pathError) throw new Error(pathError);

  return { path: "fallback", extraInfo };
}

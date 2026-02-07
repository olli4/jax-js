/**
 * @file Scan plan construction — determines the execution strategy for a scan.
 *
 * P0: fallback-only. P2–P4 will add compiled-loop and preencoded-routine.
 */

import { byteWidth, Kernel, Reduction } from "../alu";
import type { Backend, Executable } from "../backend";
import type { NativeScanGeneralParams, WasmBackend } from "../backend/wasm";
import { Routine } from "../routine";
import { DEBUG } from "../utils";
import type { ScanPath } from "../utils";
import type { Jaxpr } from "./jaxpr";
import type { JitId, JitProgram, JitStep } from "./jit";

// ---------------------------------------------------------------------------
// ScanPlan: a discriminated union of execution strategies
// ---------------------------------------------------------------------------

export type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | {
      path: "compiled-loop";
      executable: Executable;
      params?: NativeScanGeneralParams;
      internalSizes?: number[];
    }
  | { path: "preencoded-routine"; preencodedParams: any };

type ExecuteStep = Extract<JitStep, { type: "execute" }>;

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
  const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
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
 * Try to prepare a WASM native scan executable.
 *
 * Builds GeneralScanStep[] from the body program's execute steps, maps
 * slot IDs to internal buffer indices, determines CarryOutputSource and
 * YOutputSource for each output, and calls backend.prepareNativeScanGeneral().
 *
 * Returns null if the body can't be compiled (e.g. unsupported step types,
 * unmapped slots).
 */
function tryPrepareWasmNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  executeSteps: ExecuteStep[],
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): {
  executable: Executable;
  internalSizes: number[];
  params?: NativeScanGeneralParams;
} | null {
  if (DEBUG >= 2) {
    console.log(
      `[wasm-scan] trying with numCarry=${numCarry}, numY=${numY}, steps=${executeSteps.length}`,
    );
  }

  // Check for unsupported routines
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      if (DEBUG >= 1)
        console.log(
          `[wasm-scan] skipped, routine in scan body not yet supported in P2`,
        );
      return null;
    }
  }

  const numInputs = numConsts + numCarry + numX;

  const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(
    bodyJaxpr,
    numConsts,
    numCarry,
    numX,
  );

  // Build mapping from JitId (output slot) to internal buffer index
  const slotToInternal = new Map<JitId, number>();
  const internalSizes: number[] = [];

  for (const step of executeSteps) {
    const source = step.source;
    if (source instanceof Kernel) {
      const internalIdx = internalSizes.length;
      slotToInternal.set(step.outputs[0], internalIdx);
      internalSizes.push(source.size * byteWidth(source.dtype));
    }
  }

  // Build steps with reindexed inputs
  type LocalStep = import("../backend/wasm").GeneralScanStep;
  const steps: LocalStep[] = [];

  for (const step of executeSteps) {
    const source = step.source;
    if (!(source instanceof Kernel)) continue;

    // Map each input JitId to either a jaxpr input index or an internal buffer
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        inputSlots.push(inputId);
      } else {
        const internalIdx = slotToInternal.get(inputId);
        if (internalIdx === undefined) {
          if (DEBUG >= 1)
            console.log(
              `[wasm-scan] skipped, input ${inputId} not found in slot mapping`,
            );
          return null;
        }
        inputSlots.push(numInputs + internalIdx);
      }
    }

    // Reindex kernel expressions to use our inputSlots mapping
    const reindexMap = inputSlots;
    const reindexedExp = source.exp.reindexGids(reindexMap);
    const reindexedReduction = source.reduction
      ? new Reduction(
          source.reduction.dtype,
          source.reduction.op,
          source.reduction.size,
          source.reduction.epilogue.reindexGids(reindexMap),
        )
      : undefined;
    const reindexedKernel = new Kernel(
      numInputs + internalSizes.length,
      source.size,
      reindexedExp,
      reindexedReduction,
    );

    const internalIdx = slotToInternal.get(step.outputs[0])!;
    steps.push({
      source: reindexedKernel,
      inputSlots,
      outputInternalIdx: internalIdx,
    });
  }

  // Determine carry output sources
  const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
  const carryInputSlots = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  type LocalCarrySource = import("../backend/wasm").CarryOutputSource;
  const carryOutSources: LocalCarrySource[] = [];
  for (const slot of carryOutSlots) {
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      carryOutSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(
          `[wasm-scan] skipped, carry output slot ${slot} not produced by any execute step`,
        );
      return null;
    }
    carryOutSources.push({ type: "internal", internalIdx });
  }

  // Determine Y output sources
  const xsInputSlots = bodyProgram.inputs.slice(
    numConsts + numCarry,
    numConsts + numCarry + numX,
  );
  const yOutputSlots = bodyProgram.outputs.slice(numCarry);

  type LocalYSource = import("../backend/wasm").YOutputSource;
  const yOutputSources: LocalYSource[] = [];
  for (const slot of yOutputSlots) {
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      yOutputSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    const xsIdx = xsInputSlots.indexOf(slot);
    if (xsIdx !== -1) {
      yOutputSources.push({ type: "xs-passthrough", xsIdx });
      continue;
    }
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(`[wasm-scan] skipped, Y output slot ${slot} not found`);
      return null;
    }
    yOutputSources.push({ type: "internal", internalIdx });
  }

  // Build params and prepare the native scan
  const wasmBackend = backend as WasmBackend;
  if (!wasmBackend.prepareNativeScanGeneral) {
    if (DEBUG >= 2)
      console.log("[wasm-scan] backend has no prepareNativeScanGeneral");
    return null;
  }

  const params: NativeScanGeneralParams = {
    length,
    numConsts,
    constSizes,
    numCarry,
    carrySizes,
    numX,
    xsStrides,
    numY,
    ysStrides,
    internalSizes,
    steps,
    carryOutSources,
    yOutputSources,
    reverse,
  };

  try {
    const exe = wasmBackend.prepareNativeScanGeneral(params);
    if (exe) {
      if (DEBUG >= 1) {
        console.log(
          `[wasm-scan] SUCCESS! Using WASM native scan with ${steps.length} steps`,
        );
      }
      return { executable: exe, internalSizes, params };
    }
    return null;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("[wasm-scan] preparation failed:", e);
    }
    return null;
  }
}

/**
 * Try to prepare a native scan executable (any backend).
 */
function tryPrepareNativeScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): {
  executable: Executable;
  internalSizes: number[];
  params?: NativeScanGeneralParams;
} | null {
  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as ExecuteStep[];
  if (executeSteps.length === 0) {
    if (DEBUG >= 1) console.log("[compiled-loop] skipped, no execute steps");
    return null;
  }

  if (backend.type === "wasm") {
    return tryPrepareWasmNativeScan(
      backend,
      bodyProgram,
      bodyJaxpr,
      executeSteps,
      length,
      numCarry,
      numConsts,
      numX,
      numY,
      reverse,
    );
  }

  // P3: WebGPU multi-kernel scan will be handled here
  if (DEBUG >= 1)
    console.log(
      `[compiled-loop] skipped, backend=${backend.type} not supported yet`,
    );
  return null;
}

/**
 * Choose a scan execution strategy.
 *
 * Priority: compiled-loop > preencoded-routine > fallback.
 */
export function planScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
  acceptPath?: ScanPath | ScanPath[],
): ScanPlan {
  // Try compiled-loop (WASM native scan)
  const nativeScanResult = tryPrepareNativeScan(
    backend,
    bodyProgram,
    bodyJaxpr,
    length,
    numCarry,
    numConsts,
    numX,
    numY,
    reverse,
  );

  if (nativeScanResult) {
    const pathError = checkAcceptedPath("compiled-loop", acceptPath);
    if (pathError) throw new Error(pathError);
    return {
      path: "compiled-loop",
      executable: nativeScanResult.executable,
      internalSizes: nativeScanResult.internalSizes,
      params: nativeScanResult.params,
    };
  }

  // P4: preencoded-routine will be tried here

  // Fallback: JS loop
  const dispatchCount = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ).length;
  const extraInfo = `${dispatchCount} dispatch${dispatchCount !== 1 ? "es" : ""} per iteration`;

  const pathError = checkAcceptedPath("fallback", acceptPath, extraInfo);
  if (pathError) throw new Error(pathError);

  return { path: "fallback", extraInfo };
}

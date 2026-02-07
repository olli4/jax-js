import { byteWidth, DType, Kernel } from "../alu";
import type { Backend, Executable } from "../backend";
import type { NativeScanGeneralParams } from "../backend/wasm";
import type {
  NativeScanMultiParams,
  NativeScanMultiStep,
  PreparedPreencodedScan,
} from "../backend/webgpu";
import { Routine, Routines } from "../routine";
import { DEBUG, prod, type ScanPath } from "../utils";
import type { Jaxpr } from "./jaxpr";
import type { JitId, JitProgram, JitStep } from "./jit";

export type ScanPlanResult =
  | {
      path: "compiled-loop";
      executable: Executable;
      params?: NativeScanGeneralParams;
    }
  | {
      path: "preencoded-routine";
      preencodedParams: PreparedPreencodedScan;
    }
  | {
      path: "fallback";
      extraInfo?: string;
    };

type ExecuteStep = Extract<JitStep, { type: "execute" }>;

/**
 * Check if a chosen scan path satisfies the acceptPath constraint.
 * Returns an error message if the path is not allowed, or null if OK.
 *
 * Special case: an empty array `[]` always rejects, showing the chosen path.
 */
function checkAcceptedPath(
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

/**
 * Extract buffer sizes and strides from body jaxpr for native scan codegen.
 * Shared by WebGPU and WASM native scan implementations.
 */
function getScanBufferSizes(
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

/**
 * Try to prepare a preencoded scan for routine bodies (matmul, conv, etc.).
 */
function tryPreparePreencodedScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): PreparedPreencodedScan | null {
  if (backend.type !== "webgpu") {
    if (DEBUG >= 2)
      console.log("Preencoded scan: skipped, unsupported backend");
    return null;
  }

  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as ExecuteStep[];
  if (executeSteps.length !== 1) {
    if (DEBUG >= 2)
      console.log(
        `Preencoded scan: skipped, ${executeSteps.length} execute steps (need exactly 1)`,
      );
    return null;
  }

  const execStep = executeSteps[0];
  if (!(execStep.source instanceof Routine)) {
    if (DEBUG >= 2) console.log("Preencoded scan: skipped, not a Routine");
    return null;
  }

  if (numCarry !== numY) {
    if (DEBUG >= 2)
      console.log(
        `Preencoded scan: skipped, numCarry=${numCarry} !== numY=${numY}`,
      );
    return null;
  }

  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry)
    .map((v) => v.aval);

  const carrySizes = carryAvals.map((a) => a.size * byteWidth(a.dtype));
  const xsElemStrides = xAvals.map((a) => a.size);
  const ysElemStrides = carryAvals.map((a) => a.size);

  if (!backend.prepareRoutineSync) {
    if (DEBUG >= 2)
      console.log(
        "Preencoded scan: skipped, backend has no prepareRoutineSync",
      );
    return null;
  }

  let bodyRoutineExe;
  try {
    bodyRoutineExe = backend.prepareRoutineSync(execStep.source);
  } catch (e) {
    if (DEBUG >= 2)
      console.warn("Preencoded scan: prepareRoutineSync failed:", e);
    return null;
  }

  if (!backend.preparePreencodedScan) {
    if (DEBUG >= 2)
      console.log(
        "Preencoded scan: skipped, backend has no preparePreencodedScan",
      );
    return null;
  }

  const preencodedScanParams = {
    length,
    carrySizes,
    xsElemStrides,
    ysElemStrides,
    bodyRoutine: bodyRoutineExe,
    numCarry,
    numX,
    numY,
    numConsts,
    reverse,
    routineInputJitIds: execStep.inputs,
    routineOutputJitIds: execStep.outputs,
  };

  try {
    const prepared = backend.preparePreencodedScan(preencodedScanParams);
    if (prepared && DEBUG >= 1) {
      console.log(
        `Preencoded scan: SUCCESS! Using WebGPU preencoded scan for ${execStep.source.name}`,
      );
    }
    return prepared;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("Preencoded scan preparation failed:", e);
    }
    return null;
  }
}

/**
 * Try to prepare a native scan for WebGPU with kernel-only body.
 */
function tryPrepareWebGPUNativeScan(
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
): { executable: Executable } | null {
  if (DEBUG >= 2)
    console.log(
      `[webgpu-scan] trying with numCarry=${numCarry}, numY=${numY}, steps=${executeSteps.length}`,
    );

  const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(
    bodyJaxpr,
    numConsts,
    numCarry,
    numX,
  );

  const numInputs = numConsts + numCarry + numX;

  if (executeSteps.length === 1 && numCarry === 1 && numY === 1) {
    const step = executeSteps[0];
    const kernel = step.source as Kernel;

    const reindexMap: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        reindexMap.push(inputId);
      } else {
        if (DEBUG >= 2)
          console.log("[webgpu-scan] single kernel has internal buffer ref");
        return null;
      }
    }

    const reindexedExp = kernel.exp.reindexGids(reindexMap);
    const reindexedReduction = kernel.reduction?.reindexGids(reindexMap);
    const reindexedKernel = Kernel.single(
      numInputs,
      kernel.size,
      reindexedExp,
      reindexedReduction,
    );

    // Convert single kernel to multi-step format
    const multiStep: NativeScanMultiStep = {
      kernel: reindexedKernel,
      inputs: reindexMap,
      outputCarryIdx: 0,
      outputSize: reindexedKernel.size,
    };

    const params: NativeScanMultiParams = {
      length,
      numConsts,
      constSizes,
      numCarry,
      carrySizes,
      numX: xsStrides.length,
      xsStrides,
      numY: ysStrides.length,
      ysStrides,
      steps: [multiStep],
      reverse,
    };

    const webgpuBackend = backend as any;
    if (!webgpuBackend.prepareNativeScanMulti) {
      if (DEBUG >= 2)
        console.log("[webgpu-scan] backend has no prepareNativeScanMulti");
      return null;
    }

    try {
      const exe = webgpuBackend.prepareNativeScanMulti(params);
      if (exe && DEBUG >= 1) {
        console.log(
          "[webgpu-scan] SUCCESS! Using WebGPU native scan (single kernel)",
        );
      }
      return exe ? { executable: exe } : null;
    } catch (e) {
      if (DEBUG >= 2)
        console.warn("[webgpu-scan] prepareNativeScanMulti failed:", e);
    }
    return null;
  }

  if (numCarry !== numY && numY !== 0) {
    if (DEBUG >= 2)
      console.log(
        `[webgpu-scan] multi-kernel requires numCarry === numY or numY === 0, got ${numCarry} !== ${numY}`,
      );
    return null;
  }

  const hasInternalDeps = executeSteps.some((step) =>
    step.inputs.some((inputId) => inputId >= numInputs),
  );
  if (hasInternalDeps) {
    if (DEBUG >= 2)
      console.log(
        "[webgpu-scan] multi-kernel: internal buffer dependencies not supported, falling back",
      );
    return null;
  }

  interface SlotSource {
    stepIdx: number;
    outputIdxInStep: number;
    step: ExecuteStep;
  }
  const slotToSource = new Map<number, SlotSource>();
  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    for (let outIdx = 0; outIdx < step.outputs.length; outIdx++) {
      slotToSource.set(step.outputs[outIdx], {
        stepIdx: i,
        outputIdxInStep: outIdx,
        step,
      });
    }
  }

  const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
  const carryInputSlots = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  interface CarrySourceInfo {
    source: SlotSource;
    carryIdx: number;
  }
  const carrySourceInfos: CarrySourceInfo[] = [];
  for (let carryIdx = 0; carryIdx < numCarry; carryIdx++) {
    const slot = carryOutSlots[carryIdx];
    const passthroughIdx = carryInputSlots.indexOf(slot);
    if (passthroughIdx !== -1) {
      if (DEBUG >= 2)
        console.log(
          `[webgpu-scan] multi-kernel: carry ${carryIdx} is passthrough, not supported`,
        );
      return null;
    }

    const source = slotToSource.get(slot);
    if (!source) {
      if (DEBUG >= 2)
        console.log(
          `[webgpu-scan] multi-kernel: carry output ${carryIdx} (slot ${slot}) not produced by any step`,
        );
      return null;
    }

    carrySourceInfos.push({ source, carryIdx });
  }

  const multiSteps: NativeScanMultiParams["steps"] = [];

  let totalOutputs = 0;
  for (const step of executeSteps) totalOutputs += step.outputs.length;

  const slotToOutputIdx = new Map<number, number>();
  let outputIdx = 0;
  for (const step of executeSteps) {
    for (const outId of step.outputs) {
      slotToOutputIdx.set(outId, outputIdx++);
    }
  }

  for (const { source, carryIdx } of carrySourceInfos) {
    const { step, outputIdxInStep } = source;
    const stepSource = step.source;

    const inputs: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        inputs.push(inputId);
      } else {
        const outIdx = slotToOutputIdx.get(inputId);
        if (outIdx !== undefined) {
          inputs.push(numInputs + outIdx);
        } else {
          if (DEBUG >= 2)
            console.log(
              `[webgpu-scan] multi-kernel: input ${inputId} not mapped`,
            );
          return null;
        }
      }
    }

    let reindexedKernel: Kernel;
    if (stepSource instanceof Kernel) {
      const output = stepSource.outputs[outputIdxInStep];
      const reindexedExp = output.exp.reindexGids(inputs);
      const reindexedReduction = output.reduction?.reindexGids(inputs);
      reindexedKernel = Kernel.single(
        numInputs + totalOutputs,
        output.size,
        reindexedExp,
        reindexedReduction,
      );
    } else {
      if (DEBUG >= 2)
        console.log(
          "[webgpu-scan] multi-kernel: unexpected source type at step",
        );
      return null;
    }

    multiSteps.push({
      kernel: reindexedKernel,
      inputs,
      outputCarryIdx: carryIdx,
      outputSize: reindexedKernel.size,
    });
  }

  const params: NativeScanMultiParams = {
    length,
    numConsts,
    constSizes,
    numCarry,
    carrySizes,
    numX,
    xsStrides,
    numY,
    ysStrides,
    steps: multiSteps,
    reverse,
  };

  const webgpuBackend = backend as any;
  if (!webgpuBackend.prepareNativeScanMulti) {
    if (DEBUG >= 2)
      console.log("[webgpu-scan] backend has no prepareNativeScanMulti");
    return null;
  }

  try {
    const exe = webgpuBackend.prepareNativeScanMulti(params);
    if (exe && DEBUG >= 1) {
      console.log(
        `[webgpu-scan] SUCCESS! Using WebGPU native scan (${multiSteps.length} kernels)`,
      );
    }
    return exe ? { executable: exe } : null;
  } catch (e) {
    if (DEBUG >= 2)
      console.warn("[webgpu-scan] prepareNativeScanMulti failed:", e);
  }
  return null;
}

/**
 * Try to prepare a native scan for WASM backend.
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
  if (DEBUG >= 2)
    console.log(
      `[wasm-scan] trying with numCarry=${numCarry}, numY=${numY}, steps=${executeSteps.length}`,
    );

  // Check which routines are used and build routine info for WASM imports
  const usedRoutines = new Set<Routines>();
  const supportedRoutines = new Set([
    Routines.Cholesky,
    Routines.Sort,
    Routines.TriangularSolve,
    Routines.LU,
    Routines.Argsort,
  ]);

  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      const routineName = step.source.name as Routines;
      if (!supportedRoutines.has(routineName)) {
        // Unsupported routine - fall back to JS loop
        if (DEBUG >= 1)
          console.log(
            `[wasm-scan] skipped, unsupported routine in scan body: ${Routines[routineName]}`,
          );
        return null;
      }
      usedRoutines.add(routineName);
    }
  }

  if (DEBUG >= 1) {
    const routineNames = [...usedRoutines].map((r) => Routines[r]);
    console.log(
      `[wasm-scan] Analyzing body: ${executeSteps.length} execute steps, numCarry=${numCarry}, numY=${numY}` +
        (routineNames.length > 0
          ? `, routines: ${routineNames.join(", ")}`
          : ""),
    );
  }

  // Number of jaxpr inputs
  const numInputs = numConsts + numCarry + numX;

  // Get buffer sizes using shared helper
  const { constSizes, carrySizes, xsStrides, ysStrides } = getScanBufferSizes(
    bodyJaxpr,
    numConsts,
    numCarry,
    numX,
  );

  // Build a mapping from JitId (output slot) to internal buffer index
  // Multi-output routines need multiple internal buffers
  const slotToInternal = new Map<JitId, number>();
  const stepToInternalBase = new Map<number, number>(); // step index -> first internal buffer index
  const internalSizes: number[] = [];

  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;
    stepToInternalBase.set(i, internalSizes.length);

    if (source instanceof Kernel) {
      if (source.isMultiOutput) {
        // Multi-output Kernel: multiple outputs
        for (let outIdx = 0; outIdx < source.outputs.length; outIdx++) {
          const internalIdx = internalSizes.length;
          slotToInternal.set(step.outputs[outIdx], internalIdx);
          internalSizes.push(
            source.outputs[outIdx].size * byteWidth(source.dtypeAt(outIdx)),
          );
        }
      } else {
        // Single-output Kernel
        const internalIdx = internalSizes.length;
        slotToInternal.set(step.outputs[0], internalIdx);
        internalSizes.push(source.size * byteWidth(source.dtype));
      }
    } else {
      // Routine: may have multiple outputs
      const routine = source as Routine;
      for (let outIdx = 0; outIdx < step.outputs.length; outIdx++) {
        const internalIdx = internalSizes.length;
        slotToInternal.set(step.outputs[outIdx], internalIdx);
        const outShape = routine.type.outputShapes[outIdx];
        const outDtype = routine.type.outputDtypes[outIdx];
        internalSizes.push(prod(outShape) * byteWidth(outDtype));
      }
    }
  }

  // Calculate aux buffer size for routines that need it
  // Sort needs aux buffer of sortDim * elementSize
  // Argsort needs aux buffer of sortDim * 4 (for i32 indices)
  let auxBufferSize = 0;
  let elementSize: 4 | 8 = 4;
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      const routine = step.source;
      const dtype = routine.type.inputDtypes[0];
      elementSize = byteWidth(dtype) as 4 | 8;
      if (routine.name === Routines.Sort) {
        // Sort needs aux buffer of size sortDim * elementSize
        const inputShape = routine.type.inputShapes[0];
        const sortDim = inputShape[inputShape.length - 1];
        auxBufferSize = Math.max(auxBufferSize, sortDim * elementSize);
      } else if (routine.name === Routines.Argsort) {
        // Argsort needs aux buffer of size sortDim * 4 (i32 indices)
        const inputShape = routine.type.inputShapes[0];
        const sortDim = inputShape[inputShape.length - 1];
        auxBufferSize = Math.max(auxBufferSize, sortDim * 4);
      }
    }
  }

  // Build input slot mapping for each step
  // Each step's source has inputs that reference jaxpr inputs or internal buffers
  // - [0, numConsts): constant
  // - [numConsts, numConsts+numCarry): carry
  // - [numConsts+numCarry, numInputs): xs
  // - [numInputs, ...): internal buffer from previous step
  type LocalGeneralScanStep = {
    source: Kernel | Routine;
    inputSlots: number[];
    outputInternalIdx: number;
    outputInternalIndices?: number[];
    routineCallInfo?: {
      routineInfoIdx: number;
      staticParams: number[];
    };
  };

  // Build routineInfos array for WASM imports (size-specialized)
  // Note: ScanRoutineInfo is defined in wasm.ts with dtype, sizeParams, unitDiagonal, lower
  type ScanRoutineInfo = {
    routine: Routines;
    exportName: string;
    numParams: number;
    dtype: "f32" | "f64";
    sizeParams: number[];
    unitDiagonal?: boolean;
    lower?: boolean;
  };
  const routineInfos: ScanRoutineInfo[] = [];
  // Map from step index to routine info index (for steps that are routines)
  const stepToRoutineInfoIdx = new Map<number, number>();

  // First pass: collect routine infos from all routine steps
  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    if (step.source instanceof Routine) {
      const routine = step.source;
      const routineName = routine.name as Routines;
      const isF64 = routine.type.inputDtypes[0] === DType.Float64;
      const dtype: "f32" | "f64" = isF64 ? "f64" : "f32";

      const routineInfoIdx = routineInfos.length;
      stepToRoutineInfoIdx.set(i, routineInfoIdx);

      // Build routine info with size params for size-specialized modules
      if (routineName === Routines.Cholesky) {
        const inputShape = routine.type.inputShapes[0];
        const n = inputShape[inputShape.length - 1];
        routineInfos.push({
          routine: routineName,
          exportName: "cholesky", // size-specialized module exports simple name
          numParams: 2, // (inPtr, outPtr) - no n param for size-specialized
          dtype,
          sizeParams: [n],
        });
      } else if (routineName === Routines.Sort) {
        const inputShape = routine.type.inputShapes[0];
        const n = inputShape[inputShape.length - 1];
        routineInfos.push({
          routine: routineName,
          exportName: "sort", // (dataPtr, auxPtr)
          numParams: 2,
          dtype,
          sizeParams: [n],
        });
      } else if (routineName === Routines.TriangularSolve) {
        const aShape = routine.type.inputShapes[0];
        const bShape = routine.type.inputShapes[1];
        const n = aShape[aShape.length - 1];
        // batchRows is the number of columns in B (number of RHS vectors)
        const batchRows = bShape[bShape.length - 1];
        const unitDiagonal = routine.params?.unitDiagonal ?? false;
        // Primitive.TriangularSolve is always upper triangular
        // (lower=true case is handled by flipping matrices in core.ts)
        const lower = false;
        routineInfos.push({
          routine: routineName,
          exportName: "triangular_solve", // (aPtr, bPtr, xPtr)
          numParams: 3,
          dtype,
          sizeParams: [n, batchRows],
          unitDiagonal,
          lower,
        });
      } else if (routineName === Routines.LU) {
        const inputShape = routine.type.inputShapes[0];
        const m = inputShape[inputShape.length - 2];
        const n = inputShape[inputShape.length - 1];
        routineInfos.push({
          routine: routineName,
          exportName: "lu", // (aPtr, luPtr, pivPtr, permPtr)
          numParams: 4,
          dtype,
          sizeParams: [m, n],
        });
      } else if (routineName === Routines.Argsort) {
        const inputShape = routine.type.inputShapes[0];
        const n = inputShape[inputShape.length - 1];
        routineInfos.push({
          routine: routineName,
          exportName: "argsort", // (dataPtr, outPtr, idxPtr, auxPtr)
          numParams: 4,
          dtype,
          sizeParams: [n],
        });
      }
    }
  }

  const steps: LocalGeneralScanStep[] = [];
  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;

    // step.inputs are JitIds that the source reads from
    // We need to classify each: is it a jaxpr input or an internal buffer?
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        // It's a jaxpr input (const, carry, or xs)
        inputSlots.push(inputId);
      } else {
        // It's an internal buffer - find which step produced it
        const internalIdx = slotToInternal.get(inputId);
        if (internalIdx === undefined) {
          if (DEBUG >= 1)
            console.log(
              `[wasm-scan] skipped, input ${inputId} not found in slot mapping`,
            );
          return null;
        }
        // Internal buffers are indexed after jaxpr inputs
        inputSlots.push(numInputs + internalIdx);
      }
    }

    if (source instanceof Kernel) {
      // Reindex kernel gids to use our inputSlots mapping
      const reindexMap = inputSlots;
      // Handle both single and multi-output kernels uniformly
      const reindexedOutputs = source.outputs.map((out) => ({
        size: out.size,
        exp: out.exp.reindexGids(reindexMap),
        reduction: out.reduction?.reindexGids(reindexMap),
      }));
      const reindexedKernel = Kernel.multi(
        numInputs + internalSizes.length, // nargs: can read from jaxpr inputs + all internals
        reindexedOutputs,
      );

      const internalBase = stepToInternalBase.get(i)!;
      if (source.isMultiOutput) {
        const outputInternalIndices: number[] = [];
        for (let outIdx = 0; outIdx < source.outputs.length; outIdx++) {
          outputInternalIndices.push(internalBase + outIdx);
        }
        steps.push({
          source: reindexedKernel,
          inputSlots,
          outputInternalIdx: internalBase,
          outputInternalIndices,
        });
      } else {
        steps.push({
          source: reindexedKernel,
          inputSlots,
          outputInternalIdx: internalBase,
        });
      }
    } else {
      // Routine: build routineCallInfo with static params
      const routine = source as Routine;
      const routineName = routine.name as Routines;
      const routineInfoIdx = stepToRoutineInfoIdx.get(i)!;

      // Get internal buffer indices for all outputs
      const internalBase = stepToInternalBase.get(i)!;
      const numOutputs = routine.type.outputShapes.length;
      const outputInternalIndices: number[] = [];
      for (let outIdx = 0; outIdx < numOutputs; outIdx++) {
        outputInternalIndices.push(internalBase + outIdx);
      }

      // Build static params based on routine type
      // Size params are extracted here for codegenNativeScanGeneral to use when
      // calling size-specialized routine modules
      let staticParams: number[] = [];
      if (routineName === Routines.Cholesky) {
        const inputShape = routine.type.inputShapes[0];
        const n = inputShape[inputShape.length - 1];
        staticParams = [n];
      } else if (routineName === Routines.Sort) {
        const inputShape = routine.type.inputShapes[0];
        const n = inputShape[inputShape.length - 1];
        staticParams = [n];
      } else if (routineName === Routines.TriangularSolve) {
        const aShape = routine.type.inputShapes[0];
        const bShape = routine.type.inputShapes[1];
        const n = aShape[aShape.length - 1];
        // batchRows is the number of columns in B (number of RHS vectors)
        const batchRows = bShape[bShape.length - 1];
        const numBatches = 1;
        const unitDiagonal = routine.params?.unitDiagonal ? 1 : 0;
        // Primitive.TriangularSolve is always upper triangular (lower=0)
        const lower = 0;
        staticParams = [n, batchRows, numBatches, unitDiagonal, lower];
      } else if (routineName === Routines.LU) {
        const inputShape = routine.type.inputShapes[0];
        const m = inputShape[inputShape.length - 2];
        const n = inputShape[inputShape.length - 1];
        staticParams = [m, n];
      } else if (routineName === Routines.Argsort) {
        const inputShape = routine.type.inputShapes[0];
        const n = inputShape[inputShape.length - 1];
        staticParams = [n];
      }

      steps.push({
        source,
        inputSlots,
        outputInternalIdx: internalBase,
        outputInternalIndices,
        routineCallInfo: {
          routineInfoIdx,
          staticParams,
        },
      });
    }
  }

  // Find carry output sources: which internal buffer provides each carry output
  // Also handle passthrough (carry input returned as carry output unchanged)
  const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
  const carryInputSlots = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  type CarryOutputSource = {
    type: "passthrough" | "internal";
    carryIdx?: number;
    internalIdx?: number;
  };
  const carryOutSources: CarryOutputSource[] = [];
  for (const slot of carryOutSlots) {
    // Check if it's a passthrough from carry input
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      carryOutSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    // Otherwise it should be from an internal buffer
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

  // Find Y output sources: passthrough from carry, passthrough from xs, or internal buffer
  type YOutputSource = {
    type: "passthrough" | "xs-passthrough" | "internal";
    carryIdx?: number;
    xsIdx?: number;
    internalIdx?: number;
  };

  // Get xs input slots for xs passthrough detection
  const xsInputSlots = bodyProgram.inputs.slice(
    numConsts + numCarry,
    numConsts + numCarry + numX,
  );

  const yOutputSlots = bodyProgram.outputs.slice(numCarry);
  const yOutputSources: YOutputSource[] = [];

  for (const slot of yOutputSlots) {
    // Check if it's a passthrough from carry input
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      yOutputSources.push({ type: "passthrough", carryIdx });
      continue;
    }

    // Check if it's a passthrough from xs input
    const xsIdx = xsInputSlots.indexOf(slot);
    if (xsIdx !== -1) {
      yOutputSources.push({ type: "xs-passthrough", xsIdx });
      continue;
    }

    // Otherwise it should be from an internal buffer
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(`[wasm-scan] skipped, Y output slot ${slot} not found`);
      return null;
    }
    yOutputSources.push({ type: "internal", internalIdx });
  }

  // Try to prepare general native scan
  if (!backend.prepareNativeScanGeneral) {
    if (DEBUG >= 2)
      console.log("[wasm-scan] backend has no prepareNativeScanGeneral");
    return null;
  }

  const params = {
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
    auxBufferSize,
    elementSize,
    routineInfos: routineInfos.length > 0 ? routineInfos : undefined,
  };

  try {
    const exe = backend.prepareNativeScanGeneral(params);
    if (exe) {
      if (DEBUG >= 1) {
        const hasRoutines = steps.some((s) => s.source instanceof Routine);
        console.log(
          `[wasm-scan] SUCCESS! Using WASM native scan with ${steps.length} steps` +
            (hasRoutines ? " (includes routines)" : ""),
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
 * Try to prepare a native scan executable.
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
): { executable: Executable; params?: NativeScanGeneralParams } | null {
  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as ExecuteStep[];
  if (executeSteps.length === 0) {
    if (DEBUG >= 1) console.log("[compiled-loop] skipped, no execute steps");
    return null;
  }

  const allKernels = executeSteps.every((s) => s.source instanceof Kernel);

  if (backend.type === "webgpu" && allKernels) {
    return tryPrepareWebGPUNativeScan(
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

  if (DEBUG >= 1)
    console.log(
      `[compiled-loop] skipped, backend=${backend.type} not supported`,
    );
  return null;
}

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
): ScanPlanResult {
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
      params: nativeScanResult.params,
    };
  }

  const preencodedParams = tryPreparePreencodedScan(
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

  if (preencodedParams) {
    const pathError = checkAcceptedPath("preencoded-routine", acceptPath);
    if (pathError) throw new Error(pathError);
    return { path: "preencoded-routine", preencodedParams };
  }

  const dispatchCount = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ).length;
  const extraInfo =
    backend.type === "webgpu"
      ? `${dispatchCount} GPU dispatch${dispatchCount !== 1 ? "es" : ""} per iteration`
      : undefined;
  const pathError = checkAcceptedPath("fallback", acceptPath, extraInfo);
  if (pathError) throw new Error(pathError);

  return { path: "fallback", extraInfo };
}

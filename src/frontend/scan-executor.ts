/**
 * @file Unified scan executor — single execution path for all scan strategies.
 *
 * This replaces v1's dual loop (eager Primitive.Scan + JIT scanRunner) with one
 * function that handles ownership, flush, and dispatch for all backends and all
 * plan paths (fallback, compiled-loop, preencoded-routine).
 *
 * P0: only the fallback path is implemented.
 */

import { byteWidth } from "../alu";
import type { Backend, Slot } from "../backend";
import type { PendingExecute } from "./array";
import { ShapedArray } from "./core";
import type { Jaxpr } from "./jaxpr";
import type { JitProgram } from "./jit";
import type { ScanPlan } from "./scan-plan";
import type { WasmBackend } from "../backend/wasm";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface ExecuteScanParams {
  backend: Backend;
  plan: ScanPlan;
  bodyProgram: JitProgram;
  bodyJaxpr: Jaxpr;
  length: number;
  numCarry: number;
  numConsts: number;
  numX: number;
  numY: number;
  reverse: boolean;
  constSlots: Slot[];
  initCarrySlots: Slot[];
  xsSlots: Slot[];
  xsAvals: ShapedArray[];
  /** Preallocated output slots: [carry_out..., stacked_ys...] */
  outputSlots: Slot[];
}

export interface ExecuteScanResult {
  outputs: Slot[];
  pending: PendingExecute[];
}

/**
 * Execute a scan loop. Dispatches to the appropriate strategy based on the plan.
 *
 * Ownership contract:
 * - constSlots: borrowed (incRef'd before each body call, not consumed)
 * - initCarrySlots: consumed (absorbed into first iteration)
 * - xsSlots: borrowed (sliced per iteration via createView or readSync)
 * - outputSlots: filled in-place (carry outputs + stacked Y outputs)
 */
export function executeScan(params: ExecuteScanParams): ExecuteScanResult {
  switch (params.plan.path) {
    case "fallback":
      return executeScanFallback(params);
    case "compiled-loop":
      return executeScanCompiledLoop(params);
    case "preencoded-routine":
      // P4: backend.dispatchPreencodedScan()
      throw new Error("preencoded-routine scan not yet implemented (P4)");
  }
}

// ---------------------------------------------------------------------------
// Fallback: JS loop calling bodyProgram.execute() per iteration
// ---------------------------------------------------------------------------

function executeScanFallback(params: ExecuteScanParams): ExecuteScanResult {
  const {
    backend,
    bodyProgram,
    bodyJaxpr,
    length,
    numCarry,
    numConsts: _numConsts,
    numX: _numX,
    numY,
    reverse,
    constSlots,
    initCarrySlots,
    xsSlots,
    xsAvals,
    outputSlots,
  } = params;

  // Compute per-xs byte strides (size of one iteration's slice)
  const xsStrides = xsAvals.map((aval) => aval.size * byteWidth(aval.dtype));

  // Compute per-y byte strides from body jaxpr outputs
  const yOutAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);
  const ysStrides = yOutAvals.map((aval) => aval.size * byteWidth(aval.dtype));

  // Detect which body outputs share the same Jaxpr variable between
  // carry_out and y_out. This is the "shared-slot protection" case.
  const bodyCarryOutVars = bodyJaxpr.outs.slice(0, numCarry);
  const bodyYOutVars = bodyJaxpr.outs.slice(numCarry);
  const sharedCarryYIndices = new Set<number>();
  for (let ci = 0; ci < numCarry; ci++) {
    for (let yi = 0; yi < numY; yi++) {
      if (bodyCarryOutVars[ci] === bodyYOutVars[yi]) {
        sharedCarryYIndices.add(ci);
      }
    }
  }

  // Current carry slots — start with initCarry (we own these)
  let carry = initCarrySlots.slice();

  // Y output slots from the preallocated outputs
  const ysOutputSlots = outputSlots.slice(numCarry);

  // Track pending operations
  const pending: PendingExecute[] = [];

  for (let step = 0; step < length; step++) {
    const i = reverse ? length - 1 - step : step;

    // Invariant 1: Flush pending ops before each body invocation
    flushPending(pending);

    // Slice xs for this iteration
    const xSlices = sliceXsAtIteration(backend, xsSlots, xsStrides, xsAvals, i);

    // IncRef consts so body can consume them
    for (const slot of constSlots) backend.incRef(slot);

    // Build body inputs: [consts, carry, xSlices]
    // carry is consumed (body takes ownership)
    const bodyInputs = [...constSlots, ...carry, ...xSlices];

    // Execute body
    const bodyResult = bodyProgram.execute(bodyInputs);
    pending.push(...bodyResult.pending);

    // Flush pending ops from body execution before reading output slots
    // (the body's kernels must be dispatched before we can copy from them)
    flushPending(pending);

    const newCarry = bodyResult.outputs.slice(0, numCarry);
    const ySlices = bodyResult.outputs.slice(numCarry);

    // Invariant 4: Shared-slot protection
    // If a carry_out and y_out share the same slot, incRef before stacking
    // (because stacking will copy from it, and the next iteration will
    // overwrite the carry slot)
    for (const ci of sharedCarryYIndices) {
      backend.incRef(newCarry[ci]);
    }

    // Invariant 3: Y stacking — copy y slices into preallocated output buffers
    for (let yi = 0; yi < numY; yi++) {
      if (ysStrides[yi] > 0) {
        copySliceToBuffer(
          backend,
          ysOutputSlots[yi],
          ySlices[yi],
          i,
          ysStrides[yi],
          ysStrides[yi],
        );
      }
      // Free the y slice (it's been copied into the output buffer)
      backend.decRef(ySlices[yi]);
    }

    // Invariant 2: Carry lifecycle — old carry was consumed by body.execute()
    // (body takes ownership of inputs). The new carry is from body outputs.
    carry = newCarry;
  }

  // Flush any remaining pending ops before writing final carry
  flushPending(pending);

  // Write final carry to output slots
  const carryOutputSlots = outputSlots.slice(0, numCarry);
  for (let ci = 0; ci < numCarry; ci++) {
    if (carry[ci] === carryOutputSlots[ci]) {
      // Slot is already the output (shouldn't normally happen in fallback)
      continue;
    }
    // Copy carry data into the preallocated output slot
    const carrySize =
      bodyJaxpr.outs[ci].aval.size * byteWidth(bodyJaxpr.outs[ci].aval.dtype);
    copySliceToBuffer(
      backend,
      carryOutputSlots[ci],
      carry[ci],
      0,
      0,
      carrySize,
    );
    backend.decRef(carry[ci]);
  }

  return { outputs: outputSlots, pending };
}

// ---------------------------------------------------------------------------
// Compiled-loop: WASM or WebGPU native scan
// ---------------------------------------------------------------------------

function executeScanCompiledLoop(params: ExecuteScanParams): ExecuteScanResult {
  const {
    backend,
    plan,
    numCarry,
    numY,
    constSlots,
    initCarrySlots,
    xsSlots,
    outputSlots,
  } = params;

  if (plan.path !== "compiled-loop") throw new Error("unreachable");
  if (!plan.params) throw new Error("compiled-loop plan missing params");

  const carryOutSlots = outputSlots.slice(0, numCarry);
  const ysStackedSlots = outputSlots.slice(numCarry, numCarry + numY);

  // Dispatch to the backend's native scan implementation
  const wasmBackend = backend as WasmBackend;
  wasmBackend.dispatchNativeScanGeneral(
    plan.executable,
    plan.params,
    constSlots,
    initCarrySlots,
    xsSlots,
    carryOutSlots,
    ysStackedSlots,
  );

  return { outputs: outputSlots, pending: [] };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Flush all pending GPU/WASM operations. */
function flushPending(pending: PendingExecute[]): void {
  for (const p of pending) {
    p.prepareSync();
    p.submit();
  }
  pending.length = 0;
}

/**
 * Slice xs buffers at a given iteration index.
 * Returns new slots (views or copies) for each xs input.
 */
function sliceXsAtIteration(
  backend: Backend,
  xsSlots: Slot[],
  xsStrides: number[],
  xsAvals: ShapedArray[],
  iterIdx: number,
): Slot[] {
  const slices: Slot[] = [];
  for (let j = 0; j < xsSlots.length; j++) {
    const srcOffset = iterIdx * xsStrides[j];
    const sliceSize = xsAvals[j].size * byteWidth(xsAvals[j].dtype);

    // Read the slice and create a new slot with a copy of the data.
    // This is the simple approach for the fallback path. Native paths
    // (P2-P4) use buffer offsets or views instead.
    const data = backend.readSync(xsSlots[j], srcOffset, sliceSize);
    const slot = backend.malloc(sliceSize, data);
    slices.push(slot);
  }
  return slices;
}

/**
 * Copy a slice from src slot into dst slot at a given iteration offset.
 */
function copySliceToBuffer(
  backend: Backend,
  dst: Slot,
  src: Slot,
  iterIdx: number,
  strideBytes: number,
  sliceBytes: number,
): void {
  const dstOffset = iterIdx * strideBytes;
  if (backend.copyBufferToBuffer) {
    backend.copyBufferToBuffer(src, 0, dst, dstOffset, sliceBytes);
  } else {
    // Fallback: read + write via malloc. This is slow but correct.
    const data = backend.readSync(src, 0, sliceBytes);
    // Write into dst at the correct offset.
    // We need a backend method for this. For now, use readSync of entire dst,
    // patch, and write back. This is very inefficient but correct for P0.
    const dstData = backend.readSync(dst);
    const dstArray = new Uint8Array(dstData.buffer.slice(0));
    dstArray.set(data, dstOffset);

    // We need to write this back. The only way without copyBufferToBuffer
    // is to create a new slot and swap. This is a P0 hack — all real
    // backends (WASM, WebGPU) implement copyBufferToBuffer.
    //
    // For WASM, we can write directly into the memory buffer.
    // For CPU, same. So copyBufferToBuffer should always exist.
    // Throw if it doesn't — this forces us to add copyBufferToBuffer.
    throw new Error(
      "internal: copySliceToBuffer requires backend.copyBufferToBuffer",
    );
  }
}

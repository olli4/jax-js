// lax.scan primitive implementation for jax-js
//
// Scans a function over leading axis of input arrays while carrying state.
// This enables expressing recurrences (like Kalman filters) in a functional style.
//
// This version uses the Scan primitive for efficient execution.

import type { Array } from "../frontend/array";
import { zeros } from "../frontend/array";
import { bind, getAval, Primitive, ShapedArray } from "../frontend/core";
import { makeJaxpr } from "../frontend/jaxpr";
import * as tree from "../tree";
import type { JsTree } from "../tree";
import type { ScanPath } from "../utils";

/**
 * Options for {@link scan}.
 */
export interface ScanOptions {
  /**
   * Number of iterations. If not provided, inferred from the leading axis of `xs`.
   * Required when `xs` is empty or null.
   */
  length?: number;

  /**
   * If `true`, process `xs` in reverse order (from last to first element).
   * The output `ys` will also be in reverse order.
   * @default false
   */
  reverse?: boolean;

  /**
   * Accept only specific scan implementation paths. Throws an error if the
   * actual path chosen is not in the list.
   *
   * Valid paths:
   * - `"compiled-loop"` — entire scan loop compiled to native code (WASM/WebGPU)
   * - `"preencoded-routine"` — pre-encoded GPU dispatches for routine bodies (WebGPU only)
   * - `"fallback"` — JS loop calling body program per iteration
   */
  acceptPath?: ScanPath | ScanPath[];

  /**
   * Control gradient checkpointing during reverse-mode autodiff.
   *
   * - `undefined` or `true` (default): use √N checkpointing with segment size ceil(√N)
   * - A positive integer: use that as the segment size
   * - `false`: store all intermediate carries (O(N) memory, no recomputation)
   */
  checkpoint?: boolean | number;
}

/**
 * Scan a function over leading array axes while carrying along state.
 *
 * Think of `scan` as a functional `reduce` that also returns all intermediate
 * results. It iterates over the leading axis of `xs`, threading a "carry" value
 * through each step and collecting outputs.
 *
 * @param f - Step function `(carry, x) => [newCarry, y]`
 * @param init - Initial carry value. Can be a single array or a pytree of arrays.
 * @param xs - Input sequence to scan over, or `null` for carry-only scans.
 * @param options - Scan options (length, reverse, acceptPath, checkpoint)
 * @returns `[finalCarry, ys]`
 *
 * @see {@link https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html | JAX lax.scan}
 */
export function scan<
  Carry extends JsTree<Array>,
  X extends JsTree<Array> | null,
  Y extends JsTree<Array> | null,
>(
  f: (carry: Carry, x: X) => [Carry, Y],
  init: Carry,
  xs: X,
  options?: ScanOptions,
): [Carry, Y] {
  const opts: ScanOptions = options ?? {};
  const { length: lengthOpt, reverse = false, acceptPath, checkpoint } = opts;

  // Handle xs=null case (carry-only scan with no input arrays)
  const xsIsNull = xs === null;

  // Flatten inputs for primitive call
  const [initFlat, initTreedef] = tree.flatten(init);
  const [xsFlat, xsTreedef] = xsIsNull ? [[], null] : tree.flatten(xs);

  // Determine scan length from input
  const n = lengthOpt ?? (xsFlat.length > 0 ? xsFlat[0].shape[0] : 0);

  // Length-0 is handled after tracing (we need Y treedef from the trace).

  // Get abstract values for carry and x_slice (xs with leading dim removed)
  const carryAvals = initFlat.map((arr) => ShapedArray.fromAval(getAval(arr)));
  const xSliceAvals = xsFlat.map((arr) => {
    const aval = getAval(arr);
    // Remove leading dimension for x_slice
    return new ShapedArray(aval.shape.slice(1), aval.dtype, aval.weakType);
  });

  // Create a wrapper function that takes flat arrays and returns flat arrays.
  // yTreedef_ is captured by the closure and set during tracing.
  let yTreedef_: tree.JsTreeDef | undefined;
  const flatF = (
    carryFlat: Array[],
    xSliceFlat: Array[],
  ): [Array[], Array[]] => {
    const carry = tree.unflatten(initTreedef, carryFlat) as Carry;
    // For xs=null, pass null/undefined to body function (matches JAX behavior)
    const xSlice = xsIsNull
      ? xs
      : (tree.unflatten(xsTreedef!, xSliceFlat) as X);
    const [newCarry, y] = f(carry, xSlice as X);
    const [newCarryFlat] = tree.flatten(newCarry);
    // tree.flatten handles null as empty node with no leaves (NodeType.None)
    const [yFlat, yTreedef] = tree.flatten(y as JsTree<Array>);
    yTreedef_ = yTreedef;
    return [newCarryFlat, yFlat];
  };

  // Trace the function to get jaxpr
  const traceFn = (...args: Array[]): Array[] => {
    const numCarry = carryAvals.length;
    const carryFlat = args.slice(0, numCarry);
    const xSliceFlat = args.slice(numCarry);
    const [newCarryFlat, yFlat] = flatF(carryFlat, xSliceFlat);
    return [...newCarryFlat, ...yFlat];
  };

  // Build abstract values for tracing
  const traceAvals = [...carryAvals, ...xSliceAvals];

  // Trace to get jaxpr
  const { jaxpr: closedJaxpr, treedef: _outTreedef } = makeJaxpr(traceFn)(
    ...traceAvals,
  );
  const jaxpr = closedJaxpr.jaxpr;
  const consts = closedJaxpr.consts;

  // Handle length-0 scans: return (init, empty_ys) like JAX.
  if (n === 0) {
    if (xsIsNull && lengthOpt === undefined) {
      throw new Error("scan: length option is required when xs is null");
    }

    // Final carry is the initial carry (transfer ownership).
    const finalCarryFlat = initFlat.map((arr) => arr.ref);

    // Build empty Y leaves based on jaxpr outputs.
    const numCarry = initFlat.length;
    const yOutAtoms = jaxpr.outs.slice(numCarry);

    const yFlatEmpty = yOutAtoms.map((atom) => {
      const aval = atom.aval;
      return zeros([0, ...aval.shape], { dtype: aval.dtype });
    });

    // Reconstruct pytrees
    const finalCarry = tree.unflatten(initTreedef, finalCarryFlat) as Carry;
    const ys =
      yTreedef_ === tree.JsTreeDef.none
        ? (null as Y)
        : (tree.unflatten(yTreedef_!, yFlatEmpty) as Y);

    // Dispose inputs and tracing artifacts
    initFlat.forEach((arr) => arr.dispose());
    xsFlat.forEach((arr) => arr.dispose());
    closedJaxpr.dispose();

    return [finalCarry, ys];
  }

  // Call the Scan primitive
  // Args: [...consts, ...initCarry, ...xs]
  const scanArgs = [
    ...consts.map((c) => c.ref),
    ...initFlat.map((arr) => arr.ref),
    ...xsFlat.map((arr) => arr.ref),
  ];

  const numCarry = initFlat.length;
  const numConsts = consts.length;

  const results = bind(Primitive.Scan, scanArgs, {
    jaxpr,
    numCarry,
    numConsts,
    length: n,
    reverse,
    acceptPath,
    checkpoint,
  });

  // Dispose original inputs
  initFlat.forEach((arr) => arr.dispose());
  xsFlat.forEach((arr) => arr.dispose());
  closedJaxpr.dispose();

  // Split results into carry and ys
  const carryOut = results.slice(0, numCarry) as Array[];
  const ysFlat = results.slice(numCarry) as Array[];

  // Reconstruct pytrees
  const finalCarry = tree.unflatten(initTreedef, carryOut) as Carry;
  const ys = tree.unflatten(yTreedef_!, ysFlat) as Y;

  return [finalCarry, ys];
}

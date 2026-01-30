// lax.scan primitive implementation for jax-js
//
// Scans a function over leading axis of input arrays while carrying state.
// This enables expressing recurrences (like Kalman filters) in a functional style.
//
// This version uses the Scan primitive for efficient execution.

import type { Array, ArrayLike } from "../frontend/array";
import { bind, Primitive, ShapedArray, getAval } from "../frontend/core";
import { makeJaxpr } from "../frontend/jaxpr";
import * as numpy from "./numpy";
import * as tree from "../tree";
import type { JsTree, MapJsTree } from "../tree";

/**
 * Scan a function over the leading axis of input arrays.
 *
 * This is the JAX-style functional loop that threads a carry through successive
 * applications of `f`, accumulating outputs along the way.
 *
 * @param f - Step function: (carry, x) => [newCarry, output]
 * @param init - Initial carry value (pytree of arrays)
 * @param xs - Input sequence (pytree, each leaf is an array to scan over axis 0)
 * @param length - Optional length override (for when xs is null/empty)
 * @returns [finalCarry, stackedOutputs] - Final carry and outputs stacked along axis 0
 *
 * @example
 * ```ts
 * // Cumulative sum
 * const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
 *   const sum = np.add(carry, x);
 *   return [sum, sum.ref];
 * };
 * const [final, sums] = await scan(step, np.array([0.0]), np.array([[1], [2], [3], [4], [5]]));
 * // final = [15.0], sums = [[1], [3], [6], [10], [15]]
 * ```
 */
export function scan<
  Carry extends JsTree<Array>,
  X extends JsTree<Array>,
  Y extends JsTree<Array>,
>(
  f: (carry: Carry, x: X) => [Carry, Y],
  init: Carry,
  xs: X,
  length?: number,
): [Carry, Y] {
  // Flatten inputs for primitive call
  const [initFlat, initTreedef] = tree.flatten(init);
  const [xsFlat, xsTreedef] = tree.flatten(xs);
  
  // Determine scan length from input
  const n = length ?? (xsFlat.length > 0 ? xsFlat[0].shape[0] : 0);
  
  if (n === 0) {
    throw new Error("scan: cannot determine length from empty inputs");
  }
  
  // Get abstract values for carry and x_slice (xs with leading dim removed)
  const carryAvals = initFlat.map((arr) => ShapedArray.fromAval(getAval(arr)));
  const xSliceAvals = xsFlat.map((arr) => {
    const aval = getAval(arr);
    // Remove leading dimension for x_slice
    return new ShapedArray(aval.shape.slice(1), aval.dtype, aval.weakType);
  });
  
  // Create a wrapper function that takes flat arrays and returns flat arrays
  const flatF = (carryFlat: Array[], xSliceFlat: Array[]): [Array[], Array[]] => {
    const carry = tree.unflatten(initTreedef, carryFlat) as Carry;
    const xSlice = tree.unflatten(xsTreedef, xSliceFlat) as X;
    const [newCarry, y] = f(carry, xSlice);
    const [newCarryFlat] = tree.flatten(newCarry);
    const [yFlat, yTreedef] = tree.flatten(y);
    // Store yTreedef for later reconstruction
    (flatF as any)._yTreedef = yTreedef;
    return [newCarryFlat, yFlat];
  };
  
  // Trace the function to get jaxpr
  // makeJaxpr expects a function that takes individual arrays as arguments
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
  const { jaxpr: closedJaxpr, treedef: outTreedef } = makeJaxpr(traceFn)(...traceAvals);
  const jaxpr = closedJaxpr.jaxpr;
  const consts = closedJaxpr.consts;
  
  // Call the Scan primitive
  // Args: [...consts, ...initCarry, ...xs]
  const scanArgs = [...consts.map(c => c.ref), ...initFlat.map(arr => arr.ref), ...xsFlat.map(arr => arr.ref)];
  
  const numCarry = initFlat.length;
  const numConsts = consts.length;
  
  const results = bind(
    Primitive.Scan,
    scanArgs,
    { jaxpr, numCarry, numConsts, length: n },
  );
  
  // Dispose original inputs
  initFlat.forEach(arr => arr.dispose());
  xsFlat.forEach(arr => arr.dispose());
  closedJaxpr.dispose();
  
  // Split results into carry and ys
  const carryOut = results.slice(0, numCarry) as Array[];
  const ysFlat = results.slice(numCarry) as Array[];
  
  // Reconstruct pytrees
  const finalCarry = tree.unflatten(initTreedef, carryOut) as Carry;
  const yTreedef = (flatF as any)._yTreedef;
  const ys = tree.unflatten(yTreedef, ysFlat) as Y;
  
  return [finalCarry, ys];
}

/**
 * Stack a list of pytrees along a new leading axis.
 * Each pytree in the list must have the same structure.
 */
export function stackPyTree<T extends JsTree<Array>>(trees: T[]): T {
  if (trees.length === 0) {
    throw new Error("stackPyTree: empty list");
  }

  const [firstLeaves, treedef] = tree.flatten(trees[0]);
  const allLeaves = trees.map((t) => tree.leaves(t));

  // Number of leaves per tree
  const numLeaves = firstLeaves.length;

  // Stack each leaf position across all trees
  const stackedLeaves: Array[] = [];
  for (let leafIdx = 0; leafIdx < numLeaves; leafIdx++) {
    const toStack = allLeaves.map((leaves) => leaves[leafIdx].ref);
    // Use np.stack to combine along new axis 0
    const stacked = numpy.stack(toStack, 0);
    stackedLeaves.push(stacked);
  }

  // Reconstruct pytree with stacked leaves
  return tree.unflatten(treedef, stackedLeaves) as T;
}

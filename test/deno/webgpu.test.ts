/**
 * Deno-based WebGPU tests for jax-js
 *
 * Run with: deno test --unstable-webgpu --allow-read --allow-env test/deno/
 *
 * This uses Deno's native wgpu-rs WebGPU implementation which works
 * headless without X11, enabling hardware GPU testing on servers.
 *
 * jax-js includes a workaround for Deno's `createComputePipelineAsync` bug
 * by using the synchronous `createComputePipeline` when running in Deno.
 *
 * jax-js Reference Counting:
 * - Arrays passed to functions are CONSUMED (refcount -1, internal dispose)
 * - .data() CONSUMES the array (reads data then disposes)
 * - Use .ref to KEEP an array alive for reuse
 * - Don't manually .dispose() arrays that are consumed
 */

import {
  assertEquals,
  assertAlmostEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";

// Import jax-js from local build
import {
  init,
  defaultDevice,
  numpy as np,
  lax,
  jit,
  grad,
} from "../../dist/index.js";

// Check if WebGPU is available
const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

Deno.test({
  name: "WebGPU adapter available (hardware GPU)",
  ignore: !hasWebGPU,
  fn: async () => {
    const adapter = await navigator.gpu.requestAdapter();
    console.log("GPU Adapter:", adapter?.info);
    assertEquals(adapter !== null, true, "Should have a GPU adapter");
    assertEquals(
      adapter?.info?.isFallbackAdapter,
      false,
      "Should be hardware GPU, not fallback"
    );
  },
});

Deno.test({
  name: "jax-js init detects webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    console.log("Available devices:", devices);
    assertEquals(devices.includes("webgpu"), true, "WebGPU should be available");
  },
});

// Test with WebGPU backend (uses sync createComputePipeline workaround)
Deno.test({
  name: "basic array operations on webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // a and b are CONSUMED by np.add
    // c is CONSUMED by .data()
    // No manual dispose needed
    const a = np.array([1, 2, 3, 4]);
    const b = np.array([5, 6, 7, 8]);
    const c = np.add(a, b);

    const result = await c.data();
    assertEquals(Array.from(result), [6, 8, 10, 12]);
  },
});

Deno.test({
  name: "matmul on webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    const c = np.matmul(a, b);

    const result = await c.data();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assertEquals(Array.from(result), [19, 22, 43, 50]);
  },
});

Deno.test({
  name: "JIT compilation on webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // x.ref keeps x alive for second use, x.ref keeps for third use
    const f = (x: np.Array) => {
      const two = np.array([2]);
      const x2 = np.multiply(x.ref, x.ref);  // need .ref on both since we use x again
      return np.add(x2, np.multiply(x, two));  // x consumed here
    };
    const jitF = jit(f);

    const x = np.array([1, 2, 3, 4]);
    const result = jitF(x);  // x is consumed by jitF

    const data = await result.data();  // result is consumed by .data()
    // x^2 + 2x for x = [1,2,3,4] => [3, 8, 15, 24]
    assertEquals(Array.from(data), [3, 8, 15, 24]);
  },
});

Deno.test({
  name: "autodiff (grad) on webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // f(x) = sum(x^2), so grad(f)(x) = 2x
    const f = (x: np.Array) => np.sum(np.multiply(x.ref, x));
    const gradF = grad(f);

    const x = np.array([1, 2, 3]);
    const dx = gradF(x);  // x is consumed

    const data = await dx.data();  // dx is consumed by .data()
    // d/dx(sum(x^2)) = 2x => [2, 4, 6]
    assertEquals(Array.from(data), [2, 4, 6]);
  },
});

Deno.test({
  name: "lax.scan on webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // Cumulative sum using scan
    // carry and x are consumed by np.add
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry.ref, newCarry];  // .ref keeps it alive for both slots
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    // Use .ref since we're calling .data() twice
    const finalData = await finalCarry.data();
    assertAlmostEquals(finalData[0], 15.0, 1e-5);

    const outputData = await outputs.data();
    assertEquals(Array.from(outputData), [1, 3, 6, 10, 15]);
  },
});

Deno.test({
  name: "reduction operations on webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    const x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Use .ref to keep x alive for multiple operations
    const sumAll = np.sum(x.ref);
    const sumAxis0 = np.sum(x.ref, 0);
    const sumAxis1 = np.sum(x, 1);  // last use, no .ref needed

    // Each .data() consumes its array
    assertEquals(Array.from(await sumAll.data()), [21]);
    assertEquals(Array.from(await sumAxis0.data()), [5, 7, 9]);
    assertEquals(Array.from(await sumAxis1.data()), [6, 15]);
  },
});

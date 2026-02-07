/**
 * Deno WebGPU tests for pool peak memory guarantee.
 *
 * Verifies that the WebGPU buffer pool + JIT recycling never causes
 * physical GPU memory (live + pooled) to exceed the program's peak live bytes.
 *
 * Key mechanism: `configurePool()` is called before each JIT execution with
 * compile-time `peakBytes` and `mallocSizes`. The pool evicts stale entries
 * and caps retained bytes so pool + live ≤ peakBytes.
 *
 * Run with:
 *   pnpm build && deno test --no-check --unstable-webgpu --allow-read --allow-env test/deno/pool-memory.test.ts
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/mod.ts";

import {
  init,
  defaultDevice,
  numpy as np,
  lax,
  jit,
  getBackend,
  blockUntilReady,
} from "../../dist/index.js";

import { withLeakCheck } from "./harness.ts";

// Check if WebGPU is available
const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

// Helper to get the WebGPU backend's gpuAllocatedBytes method.
// Uses (backend as any) since gpuAllocatedBytes is WebGPU-specific.
function getGpuBytes(): number {
  const backend = getBackend() as any;
  if (typeof backend.gpuAllocatedBytes !== "function") {
    throw new Error("gpuAllocatedBytes not available on backend");
  }
  return backend.gpuAllocatedBytes();
}

function getSlotCount(): number {
  const backend = getBackend() as any;
  return backend.slotCount();
}

// Initialize WebGPU once for all tests
if (hasWebGPU) {
  const devices = await init();
  if (devices.includes("webgpu")) {
    defaultDevice("webgpu");
  }
}

// ---------------------------------------------------------------------------
// Test: Repeated JIT calls don't grow GPU memory beyond peak
// ---------------------------------------------------------------------------
Deno.test({
  name: "pool: repeated JIT calls stay within peak memory",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    // A JIT function with intermediates: x.add(1).mul(2).sub(3)
    // This creates/frees intermediate buffers that the pool may retain.
    const f = jit((x: any) => x.add(1).mul(2).sub(3));

    const x = np.ones([1024]); // 4096 bytes

    // Warm up: first call compiles + runs
    const warmup = f(x.ref);
    await warmup.data();

    // After warmup, record baseline GPU bytes.
    // This includes the pool potentially holding buffers from warmup.
    const baselineBytes = getGpuBytes();

    // Run many iterations. If pool grows unbounded, bytes will increase.
    const N = 20;
    for (let i = 0; i < N; i++) {
      const result = f(x.ref);
      await result.data(); // consumes result
    }

    const afterBytes = getGpuBytes();

    // GPU bytes should NOT grow after the first call.
    // The pool should be stable: configurePool sets budget = peakBytes,
    // so pool + live stays flat across calls.
    assert(
      afterBytes <= baselineBytes,
      `GPU bytes grew: ${baselineBytes} → ${afterBytes} (delta: ${afterBytes - baselineBytes})`,
    );

    x.dispose();
    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// Test: Multi-output JIT (recycling active) stays within peak
// ---------------------------------------------------------------------------
Deno.test({
  name: "pool: multi-output JIT stays within peak memory",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    // Two outputs of the same size → recycling kicks in
    const f = jit((x: any) => [x.ref.add(1), x.mul(2)]);

    const x = np.ones([2048]); // 8192 bytes

    // Warmup
    const [a, b] = f(x.ref) as any[];
    await a.data();
    await b.data();

    const baselineBytes = getGpuBytes();

    for (let i = 0; i < 15; i++) {
      const [r1, r2] = f(x.ref) as any[];
      await r1.data();
      await r2.data();
    }

    const afterBytes = getGpuBytes();
    assert(
      afterBytes <= baselineBytes,
      `GPU bytes grew: ${baselineBytes} → ${afterBytes}`,
    );

    x.dispose();
    f.dispose();
  }),
});

// ---------------------------------------------------------------------------
// Test: Different-shaped JIT calls evict stale pool entries
// ---------------------------------------------------------------------------
Deno.test({
  name: "pool: shape-varying JIT calls don't accumulate stale buffers",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    // Two different JIT functions with different buffer sizes.
    // When we switch between them, configurePool should evict stale entries.
    const fSmall = jit((x: any) => x.add(1).mul(2));
    const fLarge = jit((x: any) => x.add(1).mul(2));

    const xSmall = np.ones([256]); // 1024 bytes
    const xLarge = np.ones([4096]); // 16384 bytes

    // Warmup both
    (await fSmall(xSmall.ref)).dispose();
    (await fLarge(xLarge.ref)).dispose();

    // Record baseline after both have run
    const baselineBytes = getGpuBytes();

    // Interleave calls: each configurePool evicts the other's stale sizes
    for (let i = 0; i < 10; i++) {
      (await fSmall(xSmall.ref)).dispose();
      (await fLarge(xLarge.ref)).dispose();
    }

    const afterBytes = getGpuBytes();
    assert(
      afterBytes <= baselineBytes,
      `GPU bytes grew after interleaved calls: ${baselineBytes} → ${afterBytes}`,
    );

    xSmall.dispose();
    xLarge.dispose();
    fSmall.dispose();
    fLarge.dispose();
  }),
});

// ---------------------------------------------------------------------------
// Test: Scan cumsum stays within peak memory
// ---------------------------------------------------------------------------
Deno.test({
  name: "pool: scan cumsum stays within peak memory",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const scanF = jit((xs: any) => {
      const init = np.zeros([64]);
      return lax.scan(
        (carry: any, x: any) => {
          const s = carry.add(x);
          return [s.ref, s];
        },
        init,
        xs.reshape([-1, 64]),
      );
    });

    const xs = np.ones([6400]); // 100 rows of 64

    // Warmup
    const [c, ys] = scanF(xs.ref) as any[];
    await c.data();
    await ys.data();

    const baselineBytes = getGpuBytes();

    // Run again, pool should be stable
    for (let i = 0; i < 5; i++) {
      const [c2, ys2] = scanF(xs.ref) as any[];
      await c2.data();
      await ys2.data();
    }

    const afterBytes = getGpuBytes();
    assert(
      afterBytes <= baselineBytes,
      `GPU bytes grew after scan: ${baselineBytes} → ${afterBytes}`,
    );

    xs.dispose();
    scanF.dispose();
  }),
});

// ---------------------------------------------------------------------------
// Test: gpuAllocatedBytes tracks correctly for a unique buffer size
// ---------------------------------------------------------------------------
Deno.test({
  name: "pool: gpuAllocatedBytes tracks creates and pool returns",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    // Use a unique size unlikely to be in the pool from previous tests
    const uniqueSize = 7919; // prime number of f32 elements → 31676 bytes
    const arr = np.ones([uniqueSize]);
    await blockUntilReady(arr);

    const withArrayBytes = getGpuBytes();

    // Consume the array — buffer returns to pool (or is destroyed)
    await arr.data();

    const afterBytes = getGpuBytes();
    // After dispose: if pool accepted it, bytes stay same (pool holds it).
    // If pool rejected it, bytes decrease. Either way, should not increase.
    assert(
      afterBytes <= withArrayBytes,
      `GPU bytes increased after dispose: ${withArrayBytes} → ${afterBytes}`,
    );
    // And should not go negative relative to zero
    assert(afterBytes >= 0, `GPU bytes negative: ${afterBytes}`);
  }),
});

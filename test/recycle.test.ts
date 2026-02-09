/**
 * Tests for JIT-level buffer recycling.
 *
 * Buffer recycling replaces free→malloc pairs of the same byte size with a
 * "recycle" step that reuses the backend Slot, avoiding alloc/free overhead.
 */
import {
  checkLeaks,
  defaultDevice,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "@jax-js/jax";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

describe("buffer recycling", () => {
  it("chain of same-size operations is correct", () => {
    // a → b → c → d, all same shape — intermediates b, c are recyclable.
    const f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
    const x = np.array([1, 2, 3, 4]);
    // (([1,2,3,4] + 1) * 2) - 3 = ([2,3,4,5]*2)-3 = [4,6,8,10]-3 = [1,3,5,7]
    expect(f(x).js()).toEqual([1, 3, 5, 7]);
    f.dispose();
  });

  it("recycling preserves correctness with different sizes", () => {
    // Mix of shapes: reduction produces a different size, no recycling there.
    const f = jit((x: np.Array) => x.mul(2).sum().add(1));
    const x = np.array([1, 2, 3]);
    // (1*2 + 2*2 + 3*2) + 1 = 12 + 1 = 13
    expect(f(x).js()).toEqual(13);
    f.dispose();
  });

  it("multi-step chain does not leak slots", async () => {
    checkLeaks.start();
    const f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4));
    const x = np.array([10, 20, 30, 40]);
    const result = f(x);
    await result.data();
    result.dispose();
    x.dispose();
    f.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);
  });

  it("works with grad through chained ops", () => {
    // f(x) = x^2, f'(x) = 2x
    const f = (x: np.Array) => np.multiply(x, x).sum();
    const df = grad(f);
    const x = np.array([1, 2, 3]);
    expect(df(x).js()).toEqual([2, 4, 6]);
  });

  it("works correctly with scan", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const init = np.array([0, 0]);
    const run = jit((init: np.Array, xs: np.Array) => lax.scan(step, init, xs));
    const [carry, ys] = run(init, xs) as [np.Array, np.Array];
    expect(carry.js()).toEqual([9, 12]);
    expect(ys.js()).toEqual([
      [1, 2],
      [4, 6],
      [9, 12],
    ]);
    run.dispose();
  });
});

describe("buffer recycling (WASM)", () => {
  let previousDevice: string | undefined;

  beforeAll(async () => {
    const devices = await init();
    previousDevice = devices.includes("webgpu")
      ? "webgpu"
      : devices.includes("wasm")
        ? "wasm"
        : undefined;
    if (devices.includes("wasm")) {
      defaultDevice("wasm");
    }
  });

  afterAll(() => {
    if (previousDevice) defaultDevice(previousDevice as any);
  });

  it("chained ops produce correct results on WASM", () => {
    const f = jit((x: np.Array) => x.add(1).mul(2).sub(3));
    const x = np.array([1, 2, 3, 4]);
    expect(f(x).js()).toEqual([1, 3, 5, 7]);
    f.dispose();
  });

  it("does not leak slots on WASM", async () => {
    checkLeaks.start();
    const f = jit((x: np.Array) => x.add(1).mul(2).sub(3).add(4));
    const x = np.array([10, 20, 30, 40]);
    const result = f(x);
    await result.data();
    result.dispose();
    x.dispose();
    f.dispose();
    expect(checkLeaks.stop().leaked).toBe(0);
  });
});

import { defaultDevice, devices, init, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

// Tests run on all available backends: CPU, WASM, and WebGPU
suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("argmax correctness", () => {
    test("argmax returns first index on ties", () => {
      const x = np.array([3, 5, 5, 2, 5, 1]);
      const result = np.argmax(x);
      expect(result.js()).toEqual(1); // First occurrence of maximum (5)
    });

    test("argmax works with negative values", () => {
      const x = np.array([-10, -5, -20, -5, -15]);
      const result = np.argmax(x);
      expect(result.js()).toEqual(1); // First occurrence of maximum (-5)
    });

    test("argmax handles all equal values", () => {
      const x = np.ones([5]);
      const result = np.argmax(x);
      expect(result.js()).toEqual(0); // Returns first index
    });

    test("argmax 2D various shapes", () => {
      const x = np.array([
        [1, 2, 3],
        [6, 5, 4],
        [7, 8, 9],
      ]);

      expect(np.argmax(x).js()).toEqual(8); // Global argmax
      expect(np.argmax(x, 0).js()).toEqual([2, 2, 2]); // Column-wise
      expect(np.argmax(x, 1).js()).toEqual([2, 0, 2]); // Row-wise
    });
  });

  suite("argmin correctness", () => {
    test("argmin returns first index on ties", () => {
      const x = np.array([3, 1, 5, 1, 1, 2]);
      const result = np.argmin(x);
      expect(result.js()).toEqual(1); // First occurrence of minimum (1)
    });

    test("argmin handles all equal values", () => {
      const x = np.ones([5]).mul(42);
      const result = np.argmin(x);
      expect(result.js()).toEqual(0); // Returns first index
    });

    test("argmin 2D various shapes", () => {
      const x = np.array([
        [9, 8, 7],
        [4, 5, 6],
        [3, 2, 1],
      ]);

      expect(np.argmin(x).js()).toEqual(8); // Global argmin
      expect(np.argmin(x, 0).js()).toEqual([2, 2, 2]); // Column-wise
      expect(np.argmin(x, 1).js()).toEqual([2, 0, 2]); // Row-wise
    });
  });
});

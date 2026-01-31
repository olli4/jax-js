/**
 * Tests for lax.scan implementation
 */

import { describe, it, expect, beforeAll, beforeEach, suite, test } from "vitest";
import { init, defaultDevice, numpy as np, lax, tree, devices, jit, Device } from "../src";

const devicesAvailable = await init();

describe("lax.scan", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("stackPyTree", () => {
    it("stacks list of pytrees", async () => {
      const tree1 = { a: np.array([1.0]), b: np.array([2.0]) };
      const tree2 = { a: np.array([3.0]), b: np.array([4.0]) };
      const tree3 = { a: np.array([5.0]), b: np.array([6.0]) };

      const stacked = lax.stackPyTree([tree1, tree2, tree3]);

      const aData = await (stacked as { a: np.Array; b: np.Array }).a.data();
      const bData = await (stacked as { a: np.Array; b: np.Array }).b.data();

      expect(Array.from(aData)).toEqual([1, 3, 5]);
      expect(Array.from(bData)).toEqual([2, 4, 6]);
    });
  });

  describe("scan basic", () => {
    it("computes cumulative sum", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [finalCarry, outputs] = await lax.scan(step, init, xs);

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(15.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([1, 3, 6, 10, 15]);
    });

    it("computes factorial-like recurrence", async () => {
      // x(t) = x(t-1) * t
      const step = (carry: np.Array, t: np.Array): [np.Array, np.Array] => {
        const newCarry = np.multiply(carry, t);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([1.0]);
      const ts = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [final, outputs] = await lax.scan(step, init, ts);

      const finalData = await final.data();
      expect(finalData[0]).toBeCloseTo(120.0); // 5!

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([1, 2, 6, 24, 120]);
    });
  });

  describe("scan with pytree carry", () => {
    it("tracks two values simultaneously", async () => {
      type Carry = { sum: np.Array; count: np.Array };

      const step = (
        carry: Carry,
        x: np.Array,
      ): [Carry, np.Array] => {
        const newSum = np.add(carry.sum, x);
        const newCount = np.add(carry.count, np.array([1.0]));
        return [
          { sum: newSum, count: newCount },
          np.divide(newSum.ref, newCount.ref),
        ];
      };

      const init = { sum: np.array([0.0]), count: np.array([0.0]) };
      const xs = np.array([[2.0], [4.0], [6.0], [8.0]]);

      const [final, runningMeans] = await lax.scan(step, init, xs);

      const sumData = await final.sum.data();
      const countData = await final.count.data();

      expect(sumData[0]).toBeCloseTo(20.0);
      expect(countData[0]).toBeCloseTo(4.0);

      // Running means: 2/1, 6/2, 12/3, 20/4 = 2, 3, 4, 5
      const meanData = await runningMeans.data();
      expect(Array.from(meanData)).toEqual([2, 3, 4, 5]);
    });
  });

  describe("scan with pytree inputs", () => {
    it("handles pytree xs", async () => {
      type X = { a: np.Array; b: np.Array };
      type Carry = np.Array;
      type Y = np.Array;

      const step = (carry: Carry, x: X): [Carry, Y] => {
        const sum = np.add(x.a, x.b);
        const newCarry = np.add(carry, sum);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = {
        a: np.array([[1.0], [2.0], [3.0]]),
        b: np.array([[10.0], [20.0], [30.0]]),
      };

      const [final, outputs] = await lax.scan(step, init, xs);

      // (1+10) + (2+20) + (3+30) = 11 + 22 + 33 = 66
      const finalData = await final.data();
      expect(finalData[0]).toBeCloseTo(66.0);
    });
  });
});

/**
 * Multi-backend scan tests - runs on all available devices including WebGPU
 */
suite.each(devices)("lax.scan device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("cumulative sum", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeCloseTo(15.0);

    const outputData = await outputs.data();
    expect(Array.from(outputData)).toEqual([1, 3, 6, 10, 15]);
  });

  test("jit + scan", async () => {
    // Note: lax.scan is async, so we test jit on the step function itself
    const step = jit((carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    });

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeCloseTo(6.0);

    const outputData = await outputs.data();
    expect(Array.from(outputData)).toEqual([1, 3, 6]);
  });

  test("pytree carry with multiple arrays", async () => {
    type Carry = { sum: np.Array; product: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const newSum = np.add(carry.sum.ref, x.ref);
      const newProduct = np.multiply(carry.product, x);
      return [
        { sum: newSum, product: newProduct },
        newSum.ref,
      ];
    };

    const initCarry = { sum: np.array([0.0]), product: np.array([1.0]) };
    const xs = np.array([[2.0], [3.0], [4.0]]);

    const [final, outputs] = await lax.scan(step, initCarry, xs);

    const sumData = await final.sum.data();
    const productData = await final.product.data();

    expect(sumData[0]).toBeCloseTo(9.0);      // 2+3+4
    expect(productData[0]).toBeCloseTo(24.0); // 2*3*4

    const outputData = await outputs.data();
    expect(Array.from(outputData)).toEqual([2, 5, 9]);
  });

  test("larger iteration count", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    };

    const n = 100;
    const initCarry = np.array([0.0]);
    const xs = np.ones([n, 1]);

    const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeCloseTo(n);
  });

  test("elementwise ops in scan body", async () => {
    // More complex body: carry = tanh(carry + x * 0.1)
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const scaled = np.multiply(x, np.array([0.1]));
      const added = np.add(carry, scaled);
      const newCarry = np.tanh(added);
      return [newCarry, newCarry.ref];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    // Should converge towards tanh saturation
    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeGreaterThan(0);
    expect(finalData[0]).toBeLessThan(1);
  });
});
/**
 * Tests for lax.scan implementation
 */

import { describe, it, expect, beforeAll } from "vitest";
import { init, defaultDevice, numpy as np, lax, tree } from "../src";

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

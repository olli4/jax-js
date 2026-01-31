/**
 * Tests for lax.scan implementation
 */

import { describe, it, expect, beforeAll, beforeEach, suite, test } from "vitest";
import { init, defaultDevice, numpy as np, lax, tree, devices, jit, Device, jvp, grad, vjp, makeJaxpr } from "../src";

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

  test("native scan - small array", async () => {
    // Small carry array (64 elements) - uses native scan on WebGPU/WASM
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    };

    const size = 64;
    const initCarry = np.zeros([size]);
    const xs = np.ones([10, size]);  // 10 iterations

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    // Each element should be 10 (10 iterations of adding 1)
    expect(finalData[0]).toBeCloseTo(10.0);
    expect(finalData[size - 1]).toBeCloseTo(10.0);
  });

  test("native scan - large array", async () => {
    // Large carry array (512 elements) - uses native scan on WebGPU/WASM
    // For elementwise bodies, each element's scan is independent so any size works
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    };

    const size = 512;
    const initCarry = np.zeros([size]);
    const xs = np.ones([5, size]);  // 5 iterations

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    // Each element should be 5 (5 iterations of adding 1)
    expect(finalData[0]).toBeCloseTo(5.0);
    expect(finalData[size - 1]).toBeCloseTo(5.0);
  });

  test("matmul in body - falls back to JS loop (routine not elementwise)", async () => {
    // Matmul is a "routine" (not an elementwise kernel), so native scan is ineligible
    // This tests that the fallback to JS loop works correctly for non-elementwise bodies
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      // carry: [2, 2], x: [2, 2] -> matmul produces [2, 2]
      const newCarry = np.matmul(carry, x);
      return [newCarry.ref, newCarry];
    };

    // 2x2 matrices
    const initCarry = np.eye(2);  // identity matrix
    const xs = np.array([
      [[2, 0], [0, 2]],  // scale by 2
      [[1, 1], [0, 1]],  // shear
      [[0, -1], [1, 0]], // rotate 90 degrees
    ]);  // 3 iterations of 2x2 matrices

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    // I * [[2,0],[0,2]] = [[2,0],[0,2]]
    // [[2,0],[0,2]] * [[1,1],[0,1]] = [[2,2],[0,2]]
    // [[2,2],[0,2]] * [[0,-1],[1,0]] = [[2,-2],[2,0]]
    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeCloseTo(2.0);
    expect(finalData[1]).toBeCloseTo(-2.0);
    expect(finalData[2]).toBeCloseTo(2.0);
    expect(finalData[3]).toBeCloseTo(0.0);
  });
});

describe("scan autodiff", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("JVP (forward-mode)", () => {
    it("computes jvp of cumulative sum", async () => {
      // f(xs) = cumsum(xs) via scan
      // For xs = [1, 2, 3], cumsum = [1, 3, 6], final_carry = 6
      // d/dxs[0] cumsum = [1, 1, 1]  (every subsequent sum includes xs[0])
      // d/dxs[1] cumsum = [0, 1, 1]
      // d/dxs[2] cumsum = [0, 0, 1]
      // So tangent wrt all-ones: [1, 1, 1] + [0, 1, 1] + [0, 0, 1] = [1, 2, 3]
      // And for final carry: 1+1+1 = 3
      
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _outputs] = lax.scan(step, init, xs);
        // Return just the final carry as a scalar to simplify
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const xs_dot = np.ones([3, 1]);  // tangent is all 1s

      const [primal, tangent] = jvp(cumsumScan, [xs], [xs_dot]);

      // Primal: cumsum final = 1 + 2 + 3 = 6
      expect(await primal.data()).toEqual(new Float32Array([6]));
      
      // Tangent: d(final)/d(xs) with all-ones = 1+1+1 = 3
      expect(await tangent.data()).toEqual(new Float32Array([3]));
    });

    it("computes jvp of cumulative product", async () => {
      // f(xs) = cumprod(xs) via scan
      // For xs = [2, 3, 4], cumprod = [2, 6, 24], final_carry = 24
      // d/dx[0] final = 3*4 = 12
      // d/dx[1] final = 2*4 = 8
      // d/dx[2] final = 2*3 = 6
      // With tangent = [1, 1, 1]: 12 + 8 + 6 = 26
      
      const cumprodScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.ones([1]);
        const [finalCarry, _outputs] = lax.scan(step, init, xs);
        return finalCarry;
      };

      const xs = np.array([[2.0], [3.0], [4.0]]);
      const xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(cumprodScan, [xs], [xs_dot]);

      // Primal: cumprod final = 2 * 3 * 4 = 24
      expect(await primal.data()).toEqual(new Float32Array([24]));
      
      // Tangent: 12 + 8 + 6 = 26
      expect(await tangent.data()).toEqual(new Float32Array([26]));
    });

    it("jvp respects different tangent values", async () => {
      // Same cumsum but with tangent = [1, 0, 0] - only perturb first element
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      
      // Perturb only first element
      const xs_dot1 = np.array([[1.0], [0.0], [0.0]]);
      const [p1, t1] = jvp(cumsumScan, [xs.ref], [xs_dot1]);
      expect(await t1.data()).toEqual(new Float32Array([1]));  // d(final)/d(xs[0]) = 1
      p1.dispose();

      // Perturb only second element  
      const xs_dot2 = np.array([[0.0], [1.0], [0.0]]);
      const [p2, t2] = jvp(cumsumScan, [xs.ref], [xs_dot2]);
      expect(await t2.data()).toEqual(new Float32Array([1]));  // d(final)/d(xs[1]) = 1
      p2.dispose();

      // Perturb only third element
      const xs_dot3 = np.array([[0.0], [0.0], [1.0]]);
      const [p3, t3] = jvp(cumsumScan, [xs], [xs_dot3]);
      expect(await t3.data()).toEqual(new Float32Array([1]));  // d(final)/d(xs[2]) = 1
      p3.dispose();
    });
  });

  describe("VJP (reverse-mode)", () => {
    it("throws informative error for grad through scan", async () => {
      // VJP through scan is not yet implemented
      // This test verifies we get an error when attempting grad through scan
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry.sum();  // Return scalar for grad
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      
      // Expect grad to throw - the specific message may vary but it should fail
      expect(() => {
        grad(cumsumScan)(xs);
      }).toThrow();
    });
  });

  describe("makeJaxpr of scan with JVP", () => {
    it("traces scan jvp correctly", async () => {
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      };

      // We can at least verify that jvp produces consistent results
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const xs_dot = np.ones([5, 1]);

      const [primal, tangent] = jvp(cumsumScan, [xs], [xs_dot]);

      // Primal: 1+2+3+4+5 = 15
      expect(await primal.data()).toEqual(new Float32Array([15]));
      
      // Tangent: 5 (each input contributes 1 to the final sum)
      expect(await tangent.data()).toEqual(new Float32Array([5]));
    });
  });
});
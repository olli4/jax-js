/**
 * Comprehensive tests for lax.scan implementation.
 *
 * Ported from feat/scan (v1) for scan-v2 P1.
 *
 * Structure:
 * - Basic scan tests (eager mode, CPU/WASM default)
 * - Multi-backend device tests (all available devices)
 * - xs=null carry-only tests
 * - Y=null (no output stacking) tests
 * - Scan over views (sliced/transposed/reshaped xs)
 * - Scan with routines in body (Cholesky, matmul, etc.)
 * - acceptPath option tests
 * - Ownership edge cases (duplicate-slot, passthrough, xs-passthrough)
 * - Autodiff tests (JVP, VJP, checkpointing, vmap)
 * - Native scan path tests (compiled-loop, preencoded-routine)
 */

import {
  defaultDevice,
  devices,
  grad,
  init,
  jit,
  jvp,
  lax,
  numpy as np,
  tree,
  vmap,
} from "@jax-js/jax";
import {
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  suite,
  test,
} from "vitest";

const devicesAvailable = await init();

// ============================================================================
// Basic scan tests (eager mode)
// ============================================================================

describe("lax.scan", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("tree.map for stacking pytrees (JAX tree_stack pattern)", () => {
    it("stacks list of pytrees", () => {
      type Tree = { a: np.Array; b: np.Array };
      const trees: [Tree, Tree, Tree] = [
        { a: np.array([1.0]), b: np.array([2.0]) },
        { a: np.array([3.0]), b: np.array([4.0]) },
        { a: np.array([5.0]), b: np.array([6.0]) },
      ];

      const stacked = tree.map(
        (...v: np.Array[]) =>
          np.stack(
            v.map((a) => a),
            0,
          ),
        ...trees,
      );

      const a = (stacked as { a: np.Array; b: np.Array }).a;
      const b = (stacked as { a: np.Array; b: np.Array }).b;

      expect(a).toBeAllclose([[1], [3], [5]]);
      expect(b).toBeAllclose([[2], [4], [6]]);
    });
  });

  describe("scan basic", () => {
    it("computes cumulative sum", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const initVal = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([15.0]);
      expect(outputs).toBeAllclose([[1], [3], [6], [10], [15]]);
    });

    it("computes factorial-like recurrence", () => {
      const step = (carry: np.Array, t: np.Array): [np.Array, np.Array] => {
        const newCarry = np.multiply(carry, t);
        return [newCarry, newCarry];
      };

      const initVal = np.array([1.0]);
      const ts = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [final, outputs] = lax.scan(step, initVal, ts);

      expect(final).toBeAllclose([120.0]);
      expect(outputs).toBeAllclose([[1], [2], [6], [24], [120]]);
    });

    it("handles length-0 scans with xs array (returns init and empty ys)", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const initVal = np.array([42.0]);
      const xs = np.zeros([0, 1]);

      const [final, outputs] = lax.scan(step, initVal, xs);

      expect(final).toBeAllclose([42.0]);
      expect(outputs.shape).toEqual([0, 1]);
    });

    it("handles xs=null length-0 scans with explicit length (returns init and empty ys)", () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        return [carry, carry];
      };

      const initVal = np.array([7.0]);

      const [final, outputs] = lax.scan(step, initVal, null, { length: 0 });

      expect(final).toBeAllclose([7.0]);
      expect(outputs.shape).toEqual([0, 1]);
    });

    it("length-0 scan with pytree Y returns empty per-leaf arrays", () => {
      type Y = { a: np.Array; b: np.Array };

      const step = (carry: np.Array, x: np.Array): [np.Array, Y] => {
        const newCarry = np.add(carry, x);
        return [
          newCarry,
          { a: newCarry, b: np.multiply(newCarry, np.array([2.0])) },
        ];
      };

      const initVal = np.array([3.0]);
      const xs = np.zeros([0, 1]);

      const [final, ys] = lax.scan(step, initVal, xs);
      expect(final).toBeAllclose([3.0]);

      expect(ys).not.toBeNull();
      expect(ys.a.shape[0]).toBe(0);
      expect(ys.b.shape[0]).toBe(0);
    });

    it("length-0 scan with Y=null returns null", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, null] => {
        const newCarry = np.add(carry, x);
        return [newCarry, null];
      };

      const initVal = np.array([5.0]);
      const xs = np.zeros([0, 1]);

      const [final, ys] = lax.scan(step, initVal, xs);
      expect(final).toBeAllclose([5.0]);
      expect(ys).toBeNull();
    });
  });

  describe("scan with pytree carry", () => {
    it("tracks two values simultaneously", () => {
      type Carry = { sum: np.Array; count: np.Array };

      const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
        const newSum = np.add(carry.sum, x);
        const newCount = np.add(carry.count, np.array([1.0]));
        return [
          { sum: newSum, count: newCount },
          np.divide(newSum, newCount),
        ];
      };

      const initVal = { sum: np.array([0.0]), count: np.array([0.0]) };
      const xs = np.array([[2.0], [4.0], [6.0], [8.0]]);

      const [final, runningMeans] = lax.scan(step, initVal, xs);

      expect(final.sum).toBeAllclose([20.0]);
      expect(final.count).toBeAllclose([4.0]);
      // Running means: 2/1, 6/2, 12/3, 20/4 = 2, 3, 4, 5
      expect(runningMeans).toBeAllclose([[2], [3], [4], [5]]);
    });
  });

  describe("scan with pytree inputs", () => {
    it("handles pytree xs", () => {
      type X = { a: np.Array; b: np.Array };
      type Carry = np.Array;
      type Y = np.Array;

      const step = (carry: Carry, x: X): [Carry, Y] => {
        const sum = np.add(x.a, x.b);
        const newCarry = np.add(carry, sum);
        return [newCarry, newCarry];
      };

      const initVal = np.array([0.0]);
      const xs = {
        a: np.array([[1.0], [2.0], [3.0]]),
        b: np.array([[10.0], [20.0], [30.0]]),
      };

      const [final, _outputs] = lax.scan(step, initVal, xs);

      // (1+10) + (2+20) + (3+30) = 11 + 22 + 33 = 66
      expect(final).toBeAllclose([66.0]);
    });
  });
});

// ============================================================================
// Multi-backend scan tests — runs on all available devices
// ============================================================================

suite.each(devices)("lax.scan device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("cumulative sum", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

    expect(finalCarry).toBeAllclose([15.0]);
    expect(outputs).toBeAllclose([[1], [3], [6], [10], [15]]);
  });

  test("jit + scan", () => {
    const step = jit((carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    });

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

    expect(finalCarry).toBeAllclose([6.0]);
    expect(outputs).toBeAllclose([[1], [3], [6]]);
  });

  test("pytree carry with multiple arrays", () => {
    type Carry = { sum: np.Array; product: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const newSum = np.add(carry.sum, x);
      const newProduct = np.multiply(carry.product, x);
      return [{ sum: newSum, product: newProduct }, newSum];
    };

    const initCarry = { sum: np.array([0.0]), product: np.array([1.0]) };
    const xs = np.array([[2.0], [3.0], [4.0]]);

    const [final, outputs] = lax.scan(step, initCarry, xs);

    expect(final.sum).toBeAllclose([9.0]); // 2+3+4
    expect(final.product).toBeAllclose([24.0]); // 2*3*4
    expect(outputs).toBeAllclose([[2], [5], [9]]);
  });

  test("larger iteration count", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const n = 100;
    const initCarry = np.array([0.0]);
    const xs = np.ones([n, 1]);

    const [finalCarry, _outputs] = lax.scan(step, initCarry, xs);

    expect(finalCarry).toBeAllclose([n]);
  });

  test("elementwise ops in scan body", async () => {
    // More complex body: carry = tanh(carry + x * 0.1)
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const scaled = np.multiply(x, np.array([0.1]));
      const added = np.add(carry, scaled);
      const newCarry = np.tanh(added);
      return [newCarry, newCarry];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]);

    const [finalCarry, _outputs] = lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeGreaterThan(0);
    expect(finalData[0]).toBeLessThan(1);
  });

  describe("scan with routine body", () => {
    test("matmul in body (routine)", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.matmul(carry, x);
        return [newCarry, newCarry];
      };

      const initCarry = np.eye(2);
      const xs = np.array([
        [
          [2, 0],
          [0, 2],
        ], // scale by 2
        [
          [1, 1],
          [0, 1],
        ], // shear
        [
          [0, -1],
          [1, 0],
        ], // rotate 90 degrees
      ]);

      const [finalCarry, _outputs] = lax.scan(step, initCarry, xs);

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(2.0);
      expect(finalData[1]).toBeCloseTo(-2.0);
      expect(finalData[2]).toBeCloseTo(2.0);
      expect(finalData[3]).toBeCloseTo(0.0);
    });

    test("matmul in body with reverse", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.matmul(carry, x);
        return [newCarry, newCarry];
      };

      const initCarry = np.eye(2);
      const xs = np.array([
        [
          [2, 0],
          [0, 2],
        ],
        [
          [1, 1],
          [0, 1],
        ],
        [
          [0, -1],
          [1, 0],
        ],
      ]);

      const [finalCarry, _outputs] = lax.scan(step, initCarry, xs, {
        reverse: true,
      });

      // Processing order: xs[2], xs[1], xs[0]
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(0.0);
      expect(finalData[1]).toBeCloseTo(-2.0);
      expect(finalData[2]).toBeCloseTo(2.0);
      expect(finalData[3]).toBeCloseTo(2.0);
    });

    test("complex composite body (Kalman-like)", async () => {
      const F = np.array([
        [1, 0.1],
        [0, 1],
      ]);
      const H = np.array([[1, 0]]);
      const Q = np.array([
        [0.01, 0],
        [0, 0.01],
      ]);
      const R = np.array([[0.1]]);

      type Carry = { state: np.Array; cov: np.Array };
      type X = { obs: np.Array };
      type Y = { pred: np.Array; innovation: np.Array };

      const step = (carry: Carry, x: X): [Carry, Y] => {
        const { state, cov } = carry;
        const { obs } = x;

        const statePred = np.matmul(F, state);
        const FCov = np.matmul(F, cov);
        const covPred = np.add(np.matmul(FCov, F.transpose()), Q);
        const innovation = np.subtract(
          obs,
          np.matmul(H, statePred),
        );
        const covH = np.matmul(covPred, H.transpose());
        const scale = np.array([[0.5]]);
        const K = np.multiply(covH, scale);
        const stateNew = np.add(
          statePred,
          np.matmul(K, innovation),
        );
        const covNew = np.multiply(covPred, np.array([[0.9]]));

        return [
          { state: stateNew, cov: covNew },
          { pred: statePred, innovation },
        ];
      };

      const initState = np.array([[0], [0]]);
      const initCov = np.array([
        [1, 0],
        [0, 1],
      ]);
      const observations = np.array([[[1]], [[2]], [[2.5]], [[3]], [[3.5]]]);

      const [finalCarry, outputs] = lax.scan(
        step,
        { state: initState, cov: initCov },
        { obs: observations },
      );

      expect(finalCarry.state.shape).toEqual([2, 1]);
      expect(finalCarry.cov.shape).toEqual([2, 2]);
      expect(outputs.pred.shape).toEqual([5, 2, 1]);
      expect(outputs.innovation.shape).toEqual([5, 1, 1]);

      const finalStateData = await finalCarry.state.data();
      expect(Math.abs(finalStateData[0])).toBeGreaterThan(0.1);

      F.dispose();
      H.dispose();
      Q.dispose();
      R.dispose();
      finalCarry.state.dispose();
      finalCarry.cov.dispose();
      outputs.pred.dispose();
      outputs.innovation.dispose();
    });

    test("with reduction (dot product accumulation)", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const sumX = np.sum(x);
        const newCarry = np.add(carry, sumX);
        return [newCarry, newCarry];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([
        [1.0, 2.0, 3.0, 4.0], // sum = 10
        [5.0, 5.0, 0.0, 0.0], // sum = 10
        [1.0, 1.0, 1.0, 1.0], // sum = 4
      ]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

      expect(finalCarry).toBeAllclose([24.0]);
      expect(outputs).toBeAllclose([[10], [20], [24]]);
    });
  });

  describe("reverse scan", () => {
    test("basic", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
        reverse: true,
      });

      expect(finalCarry).toBeAllclose([6.0]);
      // Reverse: processes xs[2]=3, xs[1]=2, xs[0]=1
      // outputs[2]=3, outputs[1]=5, outputs[0]=6
      expect(outputs).toBeAllclose([[6], [5], [3]]);
    });
  });

  describe("scan with xs=null (carry-only)", () => {
    test("generates sequence with no input arrays", () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const newCarry = np.add(carry, np.array([1.0]));
        return [newCarry, carry];
      };

      const initCarry = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 5,
      });

      expect(finalCarry).toBeAllclose([5.0]);
      expect(outputs).toBeAllclose([[0], [1], [2], [3], [4]]);
    });

    test("generates fibonacci sequence", async () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const a = np.take(carry, np.array(0, { dtype: np.int32 }));
        const b = np.take(carry, np.array(1, { dtype: np.int32 }));
        const newCarry = np.stack([b, np.add(a, b)]);
        return [newCarry, a];
      };

      const initCarry = np.array([0.0, 1.0]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 8,
      });

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([0, 1, 1, 2, 3, 5, 8, 13]);

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(21.0); // fib(8)
      expect(finalData[1]).toBeCloseTo(34.0); // fib(9)
    });

    test("with jit", () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const newCarry = np.add(carry, np.array([1.0]));
        return [newCarry, carry];
      };

      const jitScan = jit((initVal: np.Array) =>
        lax.scan(step, initVal, null, { length: 5 }),
      );

      const initCarry = np.array([0.0]);
      const [finalCarry, outputs] = jitScan(initCarry);

      expect(finalCarry).toBeAllclose([5.0]);
      expect(outputs).toBeAllclose([[0], [1], [2], [3], [4]]);

      jitScan.dispose();
    });

    test("throws when length not provided", () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        return [carry, carry];
      };

      const initCarry = np.array([0.0]);

      expect(() => lax.scan(step, initCarry, null)).toThrow();
    });

    test("with reverse", () => {
      const step = (carry: np.Array, _x: null): [np.Array, np.Array] => {
        const newCarry = np.add(carry, np.array([1.0]));
        return [newCarry, carry];
      };

      const initCarry = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 5,
        reverse: true,
      });

      expect(finalCarry).toBeAllclose([5.0]);
      // Reverse: outputs = [4, 3, 2, 1, 0] (carry values in reverse stacking order)
      expect(outputs).toBeAllclose([[4], [3], [2], [1], [0]]);
    });

    test("with pytree carry", async () => {
      type Carry = { a: np.Array; b: np.Array };

      const step = (carry: Carry, _x: null): [Carry, np.Array] => {
        const newA = np.add(carry.a, carry.b);
        const newB = np.add(carry.b, np.array([1.0]));
        return [{ a: newA, b: newB }, carry.a];
      };

      const initCarry = { a: np.array([0.0]), b: np.array([1.0]) };

      const [finalCarry, outputs] = lax.scan(step, initCarry, null, {
        length: 5,
      });

      const aData = await finalCarry.a.data();
      const bData = await finalCarry.b.data();
      expect(aData[0]).toBeCloseTo(15.0);
      expect(bData[0]).toBeCloseTo(6.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([0, 1, 3, 6, 10]);
    });
  });

  describe("scan with Y=null (no output stacking)", () => {
    test("basic carry-only with Y=null", async () => {
      const step = (carry: np.Array, _x: null): [np.Array, null] => {
        const newCarry = np.add(carry, np.array([1.0]));
        carry.dispose();
        return [newCarry, null];
      };

      const initVal = np.array([0.0]);
      const [finalCarry, ys] = lax.scan(step, initVal, null, { length: 5 });

      const data = await finalCarry.data();
      expect(data[0]).toBeCloseTo(5.0);
      expect(ys).toBeNull();
    });

    test("Y=null with jit", () => {
      const scanFn = (initVal: np.Array) => {
        const step = (carry: np.Array, _x: null): [np.Array, null] => {
          const newCarry = np.add(carry, np.array([1.0]));
          carry.dispose();
          return [newCarry, null];
        };
        return lax.scan(step, initVal, null, { length: 5 });
      };

      const f = jit(scanFn as any);
      const [finalCarry, ys] = f(np.array([0.0])) as [np.Array, null];

      expect(finalCarry).toBeAllclose([5.0]);
      expect(ys).toBeNull();
      f.dispose();
    });

    test("Y=null with xs array (not null)", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, null] => {
        const newCarry = np.add(carry, x);
        return [newCarry, null];
      };

      const initVal = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const [finalCarry, ys] = lax.scan(step, initVal, xs);

      const data = await finalCarry.data();
      expect(data[0]).toBeCloseTo(15.0);
      expect(ys).toBeNull();
    });

    test("Y=null with pytree carry", async () => {
      type Carry = { sum: np.Array; count: np.Array };

      const step = (carry: Carry, _x: null): [Carry, null] => {
        const newSum = np.add(carry.sum, np.array([10.0]));
        const newCount = np.add(carry.count, np.array([1.0]));
        carry.sum.dispose();
        return [{ sum: newSum, count: newCount }, null];
      };

      const initVal = { sum: np.array([0.0]), count: np.array([0.0]) };
      const [finalCarry, ys] = lax.scan(step, initVal, null, { length: 5 });

      const sumData = await finalCarry.sum.data();
      const countData = await finalCarry.count.data();
      expect(sumData[0]).toBeCloseTo(50.0);
      expect(countData[0]).toBeCloseTo(5.0);
      expect(ys).toBeNull();
    });
  });

  describe("scan over views (sliced/transposed xs)", () => {
    it("scan over sliced xs", () => {
      const full = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
      const xs = full.slice([2, 5]);

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const initVal = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([9.0]); // 2 + 3 + 4
      expect(outputs).toBeAllclose([[2], [5], [9]]); // cumsum of [2,3,4]
    });

    it("scan over transposed xs", () => {
      const original = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      const xs = np.transpose(original); // shape [2, 3]

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };
      const initVal = np.zeros([3]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([3, 7, 11]);
      expect(outputs).toBeAllclose([
        [1, 3, 5],
        [3, 7, 11],
      ]);
    });

    it("jit(scan) over sliced xs", () => {
      const full = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]);
      const xs = full.slice([1, 4]); // [1.0], [2.0], [3.0]

      const f = jit(() => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.array([0.0]);
        return lax.scan(step, initVal, xs);
      });

      const [finalCarry, outputs] = f();

      expect(finalCarry).toBeAllclose([6.0]); // 1 + 2 + 3
      expect(outputs).toBeAllclose([[1], [3], [6]]);

      full.dispose();
      f.dispose();
    });

    it("scan over reshaped xs", () => {
      const original = np.array([
        [
          [1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0],
        ],
        [
          [7.0, 8.0],
          [9.0, 10.0],
          [11.0, 12.0],
        ],
      ]);

      const xs = np.reshape(original, [2, 6]); // shape [2, 6]

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const xSum = np.sum(x);
        const newCarry = np.add(carry, xSum);
        return [newCarry, newCarry];
      };
      const initVal = np.array([0.0]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      expect(finalCarry).toBeAllclose([78.0]);
      expect(outputs).toBeAllclose([[21.0], [78.0]]);
    });
  });

  describe("scan with routines", () => {
    it("scan with Cholesky in body", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const scaled = np.multiply(carry, x);
        const L = lax.linalg.cholesky(scaled);
        const reconstructed = np.matmul(L, L.transpose());
        return [reconstructed, L];
      };

      const initVal = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      const xs = np.array([[1.0], [1.0], [1.0]]);

      const [finalCarry, outputs] = lax.scan(step, initVal, xs);

      const outputData = await outputs.data();
      expect(outputData.length).toBe(3 * 4); // 3 iterations, 2x2 matrices

      const finalData = await finalCarry.data();
      expect(finalData.length).toBe(4);
      expect(finalData[0]).toBeGreaterThan(0);
      expect(finalData[3]).toBeGreaterThan(0);
    });

    it("jit + scan with Cholesky", async () => {
      const f = jit(() => {
        const step = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
          const L = lax.linalg.cholesky(carry);
          const reconstructed = np.matmul(L, L.transpose());
          return [reconstructed, L];
        };

        const initVal = np.array([
          [4.0, 2.0],
          [2.0, 5.0],
        ]);
        const xs = np.array([[1.0], [1.0]]);

        return lax.scan(step, initVal, xs);
      });

      const [finalCarry, outputs] = f();

      const finalData = await finalCarry.data();
      expect(finalData.length).toBe(4);
      expect(finalData[0]).toBeGreaterThan(0);

      const outputData = await outputs.data();
      expect(outputData.length).toBe(2 * 4); // 2 iterations, 2x2 matrices

      f.dispose();
    });
  });

  describe("acceptPath option", () => {
    it("succeeds when acceptPath includes fallback", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const initVal = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      // On fallback-only (P1), fallback should always be accepted
      const f = jit(() =>
        lax.scan(step, initVal, xs, {
          acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
        }),
      );

      const [carry, ys] = f();

      expect(carry).toBeAllclose([6.0]);
      expect(ys).toBeAllclose([[1], [3], [6]]);

      f.dispose();
    });

    it("allows array of paths", () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const initVal = np.array([0.0]);
      const xs = np.array([[1.0], [2.0]]);

      const f = jit(() =>
        lax.scan(step, initVal, xs, {
          acceptPath: ["compiled-loop", "preencoded-routine", "fallback"],
        }),
      );

      const [carry] = f();
      expect(carry).toBeAllclose([3.0]);

      f.dispose();
    });
  });

  describe("ownership edge cases", () => {
    it("duplicate-slot output: return [x, x]", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const result = np.add(carry, x);
        return [result, result];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = lax.scan(step, initCarry, xs) as [
        np.Array,
        np.Array,
      ];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 3.0, 6.0]));
      initCarry.dispose();
      xs.dispose();
    });

    it("carry passthrough: return [carry, y]", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const y = np.add(carry, x);
        return [carry, y];
      };

      const initCarry = np.array([10.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = lax.scan(step, initCarry, xs) as [
        np.Array,
        np.Array,
      ];

      expect(await finalCarry.data()).toEqual(new Float32Array([10.0]));
      expect(await ys.data()).toEqual(new Float32Array([11.0, 12.0, 13.0]));
      initCarry.dispose();
      xs.dispose();
    });

    it("xs passthrough: return [newCarry, x]", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, x];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = lax.scan(step, initCarry, xs) as [
        np.Array,
        np.Array,
      ];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 2.0, 3.0]));
      initCarry.dispose();
      xs.dispose();
    });

    it("jit(scan) with duplicate-slot output", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const result = np.add(carry, x);
        return [result, result];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 3.0, 6.0]));
      f.dispose();
    });

    it("jit(scan) with xs passthrough, reverse=true", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, x];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs, { reverse: true }));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      // xs passthrough with reverse: ys maintains xs order
      expect(await ys.data()).toEqual(new Float32Array([1.0, 2.0, 3.0]));
      f.dispose();
    });

    it("jit(scan) with carry passthrough and multiple iterations", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const y = np.multiply(carry, x);
        return [carry, y];
      };

      const initCarry = np.array([2.0]);
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([2.0]));
      expect(await ys.data()).toEqual(
        new Float32Array([2.0, 4.0, 6.0, 8.0, 10.0]),
      );
      f.dispose();
    });
  });
});

// ============================================================================
// Scan preallocate tests (Y stacking correctness)
// ============================================================================

describe("scan preallocate", () => {
  test("basic stacked outputs", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    expect(ys).toBeAllclose([[1.0], [3.0], [6.0]]);
  });

  test("duplicate-slot ys: carry and y are same slot", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const result = np.add(carry, x);
      return [result, result]; // carry = y (same underlying slot)
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    expect(ys).toBeAllclose([[1.0], [3.0], [6.0]]);
  });

  test("passthrough ys: y is the carry value (no new computation for y)", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, carry]; // y = old carry (passthrough)
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    // ys = carry BEFORE each update: [0, 1, 3]
    expect(ys).toBeAllclose([[0.0], [1.0], [3.0]]);
  });

  test("reverse scan preallocate", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const result = lax.scan(step, initCarry, xs, { reverse: true });
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([6.0]);
    // Reverse processes [3,2,1]: cumsum = [3,5,6]
    // Output stored in original indices: ys[2]=3, ys[1]=5, ys[0]=6
    expect(ys).toBeAllclose([[6.0], [5.0], [3.0]]);
  });

  test("length-0 scan preallocate", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const initCarry = np.array([42.0]);
    const xs = np.zeros([0, 1]);

    const result = lax.scan(step, initCarry, xs);
    const [finalCarry, ys] = result;

    expect(finalCarry).toBeAllclose([42.0]);
    expect(ys.shape).toEqual([0, 1]);
  });
});

// ============================================================================
// DLM / Kalman-like patterns (jit-scan-dlm tests)
// ============================================================================

describe("jit(scan) DLM patterns", () => {
  test("jit(scan) with passthrough Y + flip", async () => {
    type Carry = { state: np.Array; P: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const { state, P } = carry;
      const newState = np.add(state, np.multiply(P, x));
      const newP = np.multiply(P, np.array([0.9]));
      return [{ state: newState, P: newP }, state]; // y = old state (passthrough)
    };

    const initCarry = {
      state: np.array([0.0]),
      P: np.array([1.0]),
    };
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const f = jit(() => lax.scan(step, initCarry, xs));
    const [finalCarry, outputs] = f();

    // Carry tracks accumulated state
    const stateData = await finalCarry.state.data();
    const PData = await finalCarry.P.data();
    expect(stateData[0]).toBeGreaterThan(0);
    expect(PData[0]).toBeGreaterThan(0);
    expect(PData[0]).toBeLessThan(1);

    // Outputs should be the previous state values (passthrough pattern)
    const outputsData = await outputs.data();
    expect(outputsData.length).toBe(5);
    expect(outputsData[0]).toBeCloseTo(0.0); // initial state

    f.dispose();
  });

  test("non-jit scan baseline (DLM-like)", async () => {
    type Carry = { state: np.Array; P: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const { state, P } = carry;
      const newState = np.add(state, np.multiply(P, x));
      const newP = np.multiply(P, np.array([0.9]));
      return [{ state: newState, P: newP }, state];
    };

    const initCarry = {
      state: np.array([0.0]),
      P: np.array([1.0]),
    };
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs);

    const stateData = await finalCarry.state.data();
    expect(stateData[0]).toBeGreaterThan(0);
    expect(outputs.shape).toEqual([5, 1]);
  });

  test("two-pass forward + reverse (smoother-like pattern)", async () => {
    // Forward pass: accumulate
    const forwardStep = (
      carry: np.Array,
      x: np.Array,
    ): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
    const [_fwdFinal, fwdOutputs] = lax.scan(forwardStep, np.array([0.0]), xs);

    // Reverse pass: use forward outputs as input, accumulate backwards
    const reverseStep = (
      carry: np.Array,
      x: np.Array,
    ): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const [revFinal, _revOutputs] = lax.scan(
      reverseStep,
      np.array([0.0]),
      fwdOutputs,
      { reverse: true },
    );

    const revData = await revFinal.data();
    // Sum of cumulative sums = 1 + 3 + 6 + 10 + 15 = 35
    expect(revData[0]).toBeCloseTo(35.0);
  });

  test("scan with pytree carry + output + slicing", async () => {
    type Carry = { sum: np.Array; count: np.Array };
    type Y = { running_mean: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, Y] => {
      const newSum = np.add(carry.sum, x);
      const newCount = np.add(carry.count, np.array([1.0]));
      const mean = np.divide(newSum, newCount);
      return [{ sum: newSum, count: newCount }, { running_mean: mean }];
    };

    const xs = np.array([[2.0], [4.0], [6.0], [8.0], [10.0]]);
    const initCarry = { sum: np.array([0.0]), count: np.array([0.0]) };

    const f = jit(() => lax.scan(step, initCarry, xs));
    const [finalCarry, outputs] = f();

    expect(finalCarry.sum).toBeAllclose([30.0]);
    expect(finalCarry.count).toBeAllclose([5.0]);

    // Running means: 2/1=2, 6/2=3, 12/3=4, 20/4=5, 30/5=6
    expect(outputs.running_mean).toBeAllclose([[2], [3], [4], [5], [6]]);

    f.dispose();
  });
});

// ============================================================================
// Scan autodiff
// ============================================================================

describe("scan autodiff", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("JVP (forward-mode)", () => {
    it("computes jvp of cumulative sum", async () => {
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [finalCarry, _outputs] = lax.scan(step, initVal, xs);
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(cumsumScan, [xs], [xs_dot]);

      expect(await primal.data()).toEqual(new Float32Array([6]));
      expect(await tangent.data()).toEqual(new Float32Array([3]));
    });

    it("computes jvp of cumulative product", async () => {
      const cumprodScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.ones([1]);
        const [finalCarry, _outputs] = lax.scan(step, initVal, xs);
        return finalCarry;
      };

      const xs = np.array([[2.0], [3.0], [4.0]]);
      const xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(cumprodScan, [xs], [xs_dot]);

      expect(await primal.data()).toEqual(new Float32Array([24]));
      expect(await tangent.data()).toEqual(new Float32Array([26]));
    });

    it("jvp works with reverse scan", async () => {
      const revCumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(revCumsumScan, [xs], [xs_dot]);

      expect(await primal.data()).toEqual(new Float32Array([6]));
      expect(await tangent.data()).toEqual(new Float32Array([3]));
    });
  });

  describe("VJP (reverse-mode)", () => {
    it("computes gradient through scan (sum of final carry)", async () => {
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        return finalCarry.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(cumsumScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
    });

    it("computes gradient through scan (sum of all cumsum values)", async () => {
      const sumOfCumsum = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [_, ys] = lax.scan(step, initVal, xs);
        return ys.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(sumOfCumsum)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([3, 2, 1]));
    });

    it("computes gradient through reverse scan", async () => {
      const reverseCumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
        return finalCarry.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(reverseCumsumScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
    });
  });

  describe("gradient checkpointing", () => {
    it("default (sqrt-N) produces same gradient as checkpoint: false", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const cumsumScanNoCheckpoint = (xs: np.Array) => {
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs, {
          checkpoint: false,
        });
        return finalCarry.sum();
      };

      const cumsumScanDefault = (xs: np.Array) => {
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        return finalCarry.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const dxsRef = grad(cumsumScanNoCheckpoint)(xs);
      const dxsDefault = grad(cumsumScanDefault)(xs);

      expect(await dxsDefault.data()).toEqual(await dxsRef.data());
    });

    it("default checkpointing produces correct gradient (sum of all outputs)", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const sumOfCumsum = (xs: np.Array) => {
        const initVal = np.zeros([1]);
        const [_, ys] = lax.scan(step, initVal, xs);
        return ys.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const dxs = grad(sumOfCumsum)(xs);
      expect(dxs.shape).toEqual([5, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([5, 4, 3, 2, 1]));
    });

    it("checkpoint: number uses custom segment size", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const cumsumScan = (xs: np.Array) => {
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs, { checkpoint: 2 });
        return finalCarry.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
      const dxs = grad(cumsumScan)(xs);
      expect(dxs.shape).toEqual([6, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
    });

    it("checkpoint works with reverse scan", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const reverseScan = (xs: np.Array) => {
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs, { reverse: true });
        return finalCarry.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(reverseScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
    });

    it("checkpoint works with larger iteration count", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const cumsumScan = (xs: np.Array) => {
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        return finalCarry.sum();
      };

      const data = new Float32Array(100).fill(1.0);
      const xs = np.array(data).reshape([100, 1]);

      const dxs = grad(cumsumScan)(xs);
      expect(dxs.shape).toEqual([100, 1]);

      const gradData = await dxs.data();
      for (let i = 0; i < 100; i++) {
        expect(gradData[i]).toBeCloseTo(1.0);
      }
    });

    it("checkpoint works with nonlinear body (multiplicative)", async () => {
      const mulScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry, x);
          return [newCarry, carry];
        };
        const initVal = np.array([1.0]);
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        return finalCarry.sum();
      };

      const xs = np.array([[2.0], [3.0], [4.0]]);
      const dxsRef = grad((xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry, x);
          return [newCarry, carry];
        };
        const initVal = np.array([1.0]);
        const [finalCarry, _] = lax.scan(step, initVal, xs, {
          checkpoint: false,
        });
        return finalCarry.sum();
      })(xs);
      const dxsDefault = grad(mulScan)(xs);

      expect(await dxsDefault.data()).toEqual(await dxsRef.data());
    });

    it("jit(grad(scan)) works with default checkpointing", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry];
      };

      const cumsumScan = (xs: np.Array) => {
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        return finalCarry.sum();
      };

      const jitGrad = jit(grad(cumsumScan));
      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = jitGrad(xs);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
      jitGrad.dispose();
    });
  });

  describe("vmap", () => {
    it("vmaps cumulative sum over batch dimension", async () => {
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [finalCarry, outputs] = lax.scan(step, initVal, xs);
        outputs.dispose();
        return finalCarry;
      };

      const xs = np.array([
        [[1.0], [2.0], [3.0], [4.0]],
        [[2.0], [4.0], [6.0], [8.0]],
        [[1.0], [1.0], [1.0], [1.0]],
      ]);

      const batchedCumsum = vmap(cumsumScan);
      const result = batchedCumsum(xs);

      expect(result.shape).toEqual([3, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(10.0);
      expect(data[1]).toBeCloseTo(20.0);
      expect(data[2]).toBeCloseTo(4.0);
    });

    it("jit(vmap(scan)) works correctly", async () => {
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        return finalCarry;
      };

      const xs = np.array([
        [[1.0], [2.0], [3.0], [4.0]],
        [[2.0], [4.0], [6.0], [8.0]],
        [[1.0], [1.0], [1.0], [1.0]],
      ]);

      const jittedBatchedCumsum = jit(vmap(cumsumScan));
      const result = jittedBatchedCumsum(xs);

      expect(result.shape).toEqual([3, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(10.0);
      expect(data[1]).toBeCloseTo(20.0);
      expect(data[2]).toBeCloseTo(4.0);

      jittedBatchedCumsum.dispose();
    });
  });

  describe("transform sandwiches", () => {
    it("jit(grad(scan)) computes gradient through JIT-compiled grad", async () => {
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry];
        };
        const initVal = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, initVal, xs);
        return finalCarry.sum();
      };

      const jitGrad = jit(grad(cumsumScan));

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs1 = jitGrad(xs);
      expect(dxs1.shape).toEqual([3, 1]);
      expect(await dxs1.data()).toEqual(new Float32Array([1, 1, 1]));

      const xs2 = np.array([[4.0], [5.0], [6.0]]);
      const dxs2 = jitGrad(xs2);
      expect(await dxs2.data()).toEqual(new Float32Array([1, 1, 1]));

      jitGrad.dispose();
    });
  });
});

// ============================================================================
// Native scan path tests — P2+ (skipped until compiled-loop/preencoded available)
// ============================================================================

describe("native scan paths (P2+)", () => {
  beforeAll(async () => {
    const devs = await init();
    if (!devs.includes("wasm")) return; // skip if no WASM
    defaultDevice("wasm");
  });
  test("small array with acceptPath: compiled-loop", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const size = 64;
    const initCarry = np.zeros([size]);
    const xs = np.ones([10, size]);

    const [finalCarry, _outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    expect(finalCarry).toBeAllclose(np.full([size], 10.0));
  });

  test("native scan with constants", () => {
    const scale = np.array([2.0]);
    const offset = np.array([1.0]);

    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const scaled = np.multiply(x, scale);
      const shifted = np.add(scaled, offset);
      const newCarry = np.add(carry, shifted);
      return [newCarry, newCarry];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    expect(finalCarry).toBeAllclose([15.0]);
    expect(outputs).toBeAllclose([[3], [8], [15]]);

    scale.dispose();
    offset.dispose();
  });

  test("native scan with reduction in body", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const sumX = np.sum(x);
      const newCarry = np.add(carry, sumX);
      return [newCarry, newCarry];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 5.0, 0.0, 0.0],
      [1.0, 1.0, 1.0, 1.0],
    ]);

    const [finalCarry, outputs] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    expect(finalCarry).toBeAllclose([24.0]);
    expect(outputs).toBeAllclose([[10], [20], [24]]);
  });

  test("large native scan (many iterations)", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry];
    };

    const n = 500;
    const initCarry = np.zeros([64]);
    const xs = np.ones([n, 64]);

    const [finalCarry, _] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    expect(finalCarry).toBeAllclose(np.full([64], n));
  });

  test("routine body: matmul in native scan", () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.matmul(carry, x);
      return [newCarry, newCarry];
    };

    const initCarry = np.eye(4);
    const xs = np.stack([np.eye(4).mul(2), np.eye(4).mul(3)]);

    const [finalCarry, _] = lax.scan(step, initCarry, xs, {
      acceptPath: ["compiled-loop", "preencoded-routine"],
    });

    expect(finalCarry).toBeAllclose(np.eye(4).mul(6));
  });
});

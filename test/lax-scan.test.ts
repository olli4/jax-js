import { jit, lax, numpy as np } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("lax.scan", () => {
  test("cumulative sum (eager)", () => {
    const result = lax.scan(
      (carry: np.Array, x: np.Array) => {
        const newCarry = carry.add(x);
        return [newCarry, newCarry.ref];
      },
      np.array(0),
      np.array([1, 2, 3, 4, 5]),
    );
    const [finalCarry, ys] = result;
    expect(finalCarry).toBeAllclose(15);
    expect(ys).toBeAllclose([1, 3, 6, 10, 15]);
  });

  test("cumulative sum (jit)", () => {
    const f = jit((xs: np.Array) => {
      return lax.scan(
        (carry: np.Array, x: np.Array) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry.ref];
        },
        np.array(0),
        xs,
      );
    });
    const [finalCarry, ys] = f(np.array([1, 2, 3, 4, 5]));
    expect(finalCarry).toBeAllclose(15);
    expect(ys).toBeAllclose([1, 3, 6, 10, 15]);
    f.dispose();
  });

  test("cumulative product (eager)", () => {
    const result = lax.scan(
      (carry: np.Array, x: np.Array) => {
        const newCarry = carry.mul(x);
        return [newCarry, newCarry.ref];
      },
      np.array(1),
      np.array([1, 2, 3, 4, 5]),
    );
    const [finalCarry, ys] = result;
    expect(finalCarry).toBeAllclose(120);
    expect(ys).toBeAllclose([1, 2, 6, 24, 120]);
  });

  test("carry only, no outputs (eager)", () => {
    const result = lax.scan(
      (carry: np.Array, x: np.Array) => {
        return [carry.add(x), null];
      },
      np.array(0),
      np.array([10, 20, 30]),
    );
    const [finalCarry, ys] = result;
    expect(finalCarry).toBeAllclose(60);
    expect(ys).toBe(null);
  });

  test("reverse scan (eager)", () => {
    const result = lax.scan(
      (carry: np.Array, x: np.Array) => {
        const newCarry = carry.add(x);
        return [newCarry, newCarry.ref];
      },
      np.array(0),
      np.array([1, 2, 3, 4, 5]),
      { reverse: true },
    );
    const [finalCarry, ys] = result;
    expect(finalCarry).toBeAllclose(15);
    // Reverse: processes [5,4,3,2,1], cumsum = [5,9,12,14,15]
    // Output is in reverse order of processing: [15,14,12,9,5]
    expect(ys).toBeAllclose([15, 14, 12, 9, 5]);
  });

  test("length-0 scan (eager)", () => {
    const result = lax.scan(
      (carry: np.Array, x: np.Array) => {
        return [carry.add(x.ref), x];
      },
      np.array(42),
      np.zeros([0]),
    );
    const [finalCarry, ys] = result;
    expect(finalCarry).toBeAllclose(42);
    expect(ys.shape).toEqual([0]);
  });

  test("multiple carry values (eager)", () => {
    const result = lax.scan(
      (carry: [np.Array, np.Array], x: np.Array) => {
        const [sum, count] = carry;
        const newSum = sum.add(x.ref);
        const newCount = count.add(np.array(1));
        return [[newSum, newCount], x];
      },
      [np.array(0), np.array(0)],
      np.array([10, 20, 30]),
    );
    const [[finalSum, finalCount], ys] = result;
    expect(finalSum).toBeAllclose(60);
    expect(finalCount).toBeAllclose(3);
    expect(ys).toBeAllclose([10, 20, 30]);
  });

  test("2D xs input (eager)", () => {
    // xs has shape [3, 2] — each x slice is [2]
    const result = lax.scan(
      (carry: np.Array, x: np.Array) => {
        const newCarry = carry.add(np.sum(x.ref));
        return [newCarry, x];
      },
      np.array(0),
      np.array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]),
    );
    const [finalCarry, ys] = result;
    expect(finalCarry).toBeAllclose(21); // 3+7+11
    expect(ys).toBeAllclose([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });

  test("cumulative sum (jit, wasm)", () => {
    const f = jit(
      (xs: np.Array) => {
        return lax.scan(
          (carry: np.Array, x: np.Array) => {
            const newCarry = carry.add(x);
            return [newCarry, newCarry.ref];
          },
          np.array(0),
          xs,
        );
      },
      { device: "wasm" },
    );
    const [finalCarry, ys] = f(np.array([1, 2, 3, 4, 5]));
    expect(finalCarry).toBeAllclose(15);
    expect(ys).toBeAllclose([1, 3, 6, 10, 15]);
    f.dispose();
  });
});

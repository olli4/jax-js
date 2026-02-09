/**
 * Tests for the non-consuming ownership model.
 *
 * Operations do NOT consume their inputs — no .ref needed for multi-use.
 * Memory is managed via explicit .dispose() or the `using` keyword.
 */
import {
  getBackend,
  grad,
  jit,
  lax,
  numpy as np,
  valueAndGrad,
} from "@jax-js/jax";
import { describe, expect, it } from "vitest";

function slotCount(): number {
  return (getBackend() as any).slotCount();
}

describe("non-consuming ownership model", () => {
  describe("operations do not consume inputs", () => {
    it("input used twice without .ref", () => {
      const f = jit((x: np.Array) => x.mul(x));
      expect(f(np.array([2, 3])).js()).toEqual([4, 9]);
      f.dispose();
    });

    it("input used three times without .ref", () => {
      const f = jit((x: np.Array) => x.mul(x).add(x));
      expect(f(np.array([1, 2, 3])).js()).toEqual([2, 6, 12]);
      f.dispose();
    });

    it("intermediate used twice without .ref", () => {
      const f = jit((x: np.Array) => {
        const a = x.mul(2);
        return a.add(a);
      });
      expect(f(np.array([5, 10])).js()).toEqual([20, 40]);
      f.dispose();
    });

    it("multiple inputs used multiple times", () => {
      const f = jit((x: np.Array, y: np.Array) => {
        return x.mul(y).add(x.add(y));
      });
      expect(f(np.array([1, 2]), np.array([3, 4])).js()).toEqual([7, 14]);
      f.dispose();
    });

    it("closures captured by jit", () => {
      const scale = np.array([10, 10, 10]);
      const f = jit((x: np.Array) => x.mul(scale));
      expect(f(np.array([1, 2, 3])).js()).toEqual([10, 20, 30]);
      expect(f(np.array([4, 5, 6])).js()).toEqual([40, 50, 60]);
      f.dispose();
      scale.dispose();
    });

    it("works with reduction", () => {
      const f = jit((x: np.Array) => x.mul(x).sum());
      expect(f(np.array([1, 2, 3])).js()).toEqual(14);
      f.dispose();
    });

    it("works with 2D arrays", () => {
      const f = jit((x: np.Array) => {
        const t = x.add(x);
        return t.sum(-1);
      });
      expect(
        f(
          np.array([
            [1, 2],
            [3, 4],
          ]),
        ).js(),
      ).toEqual([6, 14]);
      f.dispose();
    });

    it("mandelbrot pattern without .ref", () => {
      const f = jit(
        (A: np.Array, B: np.Array, V: np.Array, X: np.Array, Y: np.Array) => {
          const Asq = A.mul(A);
          const Bsq = B.mul(B);
          V = V.add(Asq.add(Bsq).less(100).astype(np.float32));
          const A2 = np.clip(Asq.sub(Bsq).add(X), -50, 50);
          const B2 = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
          return [A2, B2, V];
        },
      );

      const size = 4;
      const [A2, B2, V2] = f(
        np.zeros([size]),
        np.zeros([size]),
        np.zeros([size]),
        np.linspace(-2, 1, size),
        np.linspace(-1, 1, size),
      ) as np.Array[];
      expect(V2.js()).toEqual([1, 1, 1, 1]);
      A2.dispose();
      B2.dispose();
      V2.dispose();
      f.dispose();
    });

    it("cached across calls with same shape", () => {
      const f = jit((x: np.Array) => x.mul(x));
      expect(f(np.array([1, 2])).js()).toEqual([1, 4]);
      expect(f(np.array([3, 4])).js()).toEqual([9, 16]);
      f.dispose();
    });
  });

  describe("does not leak memory", () => {
    it("multi-use input does not leak", () => {
      const before = slotCount();
      const f = jit((x: np.Array) => x.mul(x).add(1));
      const input = np.array([2, 3, 4]);
      const result = f(input);
      expect(result.js()).toEqual([5, 10, 17]);
      result.dispose();
      input.dispose();
      f.dispose();
      expect(slotCount()).toBe(before);
    });

    it("unused input does not leak", () => {
      const before = slotCount();
      const f = jit((_x: np.Array, y: np.Array) => y.mul(2));
      const x = np.array([1]);
      const y = np.array([5]);
      const result = f(x, y);
      expect(result.js()).toEqual([10]);
      result.dispose();
      x.dispose();
      y.dispose();
      f.dispose();
      expect(slotCount()).toBe(before);
    });
  });

  describe("with grad", () => {
    it("grad through jit without .ref", () => {
      const f = jit((x: np.Array) => x.mul(x).sum());
      const df = grad(f);
      const x = np.array([1, 2, 3]);
      expect(df(x).js()).toEqual([2, 4, 6]);
      f.dispose();
    });

    it("grad with multi-use input", () => {
      // f(x) = (x*x + x).sum() => f'(x) = 2x + 1
      const f = jit((x: np.Array) => x.mul(x).add(x).sum());
      const df = grad(f);
      expect(df(np.array([1, 2, 3])).js()).toEqual([3, 5, 7]);
      f.dispose();
    });

    it("valueAndGrad without .ref", () => {
      const f = jit((x: np.Array) => x.mul(x).sum());
      const vg = valueAndGrad(f);
      const [val, dx] = vg(np.array([1, 2, 3]));
      expect(val.js()).toEqual(14);
      expect(dx.js()).toEqual([2, 4, 6]);
      f.dispose();
    });
  });

  describe("with scan", () => {
    it("scan inside jit without .ref", () => {
      const f = jit((xs: np.Array) => {
        const init = np.array([0]);
        return lax.scan(
          (carry: np.Array, x: np.Array) => {
            const s = carry.add(x);
            return [s, s];
          },
          init,
          xs.reshape([-1, 1]),
        );
      });
      const [carry, ys] = f(np.array([1, 2, 3, 4, 5])) as [np.Array, np.Array];
      expect(carry.js()).toEqual([15]);
      expect(ys.js()).toEqual([[1], [3], [6], [10], [15]]);
      f.dispose();
    });

    it("scan body with multi-use carry", () => {
      const f = jit((xs: np.Array) => {
        const init = np.array([1]);
        return lax.scan(
          (carry: np.Array, x: np.Array) => {
            const s = carry.mul(carry).add(x);
            return [s, s];
          },
          init,
          xs.reshape([-1, 1]),
        );
      });
      const [carry, ys] = f(np.array([1, 2, 3])) as [np.Array, np.Array];
      expect(carry.js()).toEqual([39]);
      expect(ys.js()).toEqual([[2], [6], [39]]);
      f.dispose();
    });
  });

  describe("edge cases", () => {
    it("scalar inputs", () => {
      const f = jit((x: np.Array) => x.mul(x).add(x));
      expect(f(np.array(3)).js()).toEqual(12);
      f.dispose();
    });

    it(".ref still works for backward compat", () => {
      const f = jit((x: np.Array) => x.ref.mul(x));
      expect(f(np.array([2, 3])).js()).toEqual([4, 9]);
      f.dispose();
    });
  });
});

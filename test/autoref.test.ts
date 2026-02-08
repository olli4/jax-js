/**
 * Tests for jit ref validation.
 *
 * jit() always traces without enforcing .ref/.dispose() (internal autoRef),
 * then validates that the user's .ref usage is correct for eager-mode compat.
 * Missing .ref → actionable error. Extra .ref → actionable error.
 * Correct .ref → identical compiled code, works eagerly too.
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

describe("jit ref validation", () => {
  describe("correct .ref usage passes validation", () => {
    it("single-use inputs need no .ref", () => {
      const f = jit((x: np.Array) => x.mul(2));
      expect(f(np.array([1, 2, 3])).js()).toEqual([2, 4, 6]);
      f.dispose();
    });

    it("double-use input with one .ref", () => {
      const f = jit((x: np.Array) => x.ref.mul(x));
      expect(f(np.array([2, 3])).js()).toEqual([4, 9]);
      f.dispose();
    });

    it("triple-use input with two .ref", () => {
      const f = jit((x: np.Array) => x.ref.mul(x.ref).add(x));
      expect(f(np.array([1, 2, 3])).js()).toEqual([2, 6, 12]);
      f.dispose();
    });

    it("intermediate used twice with one .ref", () => {
      const f = jit((x: np.Array) => {
        const a = x.mul(2);
        return a.ref.add(a);
      });
      expect(f(np.array([5, 10])).js()).toEqual([20, 40]);
      f.dispose();
    });

    it("multiple inputs with correct refs", () => {
      const f = jit((x: np.Array, y: np.Array) => {
        // x used twice, y used twice
        return x.ref.mul(y.ref).add(x.add(y));
      });
      expect(f(np.array([1, 2]), np.array([3, 4])).js()).toEqual([7, 14]);
      f.dispose();
    });

    it("closures are consumed (use .ref to keep)", () => {
      const scale = np.array([10, 10, 10]);
      // .ref before jit so the caller keeps their reference
      const f = jit((x: np.Array) => x.mul(scale.ref));
      expect(f(np.array([1, 2, 3])).js()).toEqual([10, 20, 30]);
      // scale is still alive — .ref preserved the caller's reference
      expect(f(np.array([4, 5, 6])).js()).toEqual([40, 50, 60]);
      f.dispose();
      scale.dispose();
    });

    it("does not leak slots", () => {
      const before = slotCount();
      const f = jit((x: np.Array) => x.ref.mul(x).add(1));
      const result = f(np.array([2, 3, 4]));
      expect(result.js()).toEqual([5, 10, 17]);
      f.dispose();
      expect(slotCount()).toBe(before);
    });

    it("works with reduction", () => {
      const f = jit((x: np.Array) => x.ref.mul(x).sum());
      expect(f(np.array([1, 2, 3])).js()).toEqual(14);
      f.dispose();
    });

    it("works with 2D arrays", () => {
      const f = jit((x: np.Array) => {
        const t = x.ref.add(x);
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

    it("mandelbrot pattern with correct refs", () => {
      const f = jit(
        (A: np.Array, B: np.Array, V: np.Array, X: np.Array, Y: np.Array) => {
          // A used 3x: mul(A,A), mul(A,B), and consumed by mul
          // B used 2x: mul(B,B), mul(A,B)
          // Asq used 2x: add(Asq,Bsq) and sub(Asq,Bsq)
          // Bsq used 2x: add(Asq,Bsq) and sub(Asq,Bsq)
          const Asq = A.ref.mul(A.ref);
          const Bsq = B.ref.ref.mul(B);
          V = V.add(Asq.ref.add(Bsq.ref).less(100).astype(np.float32));
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
      expect(V2.ref.js()).toEqual([1, 1, 1, 1]);
      A2.dispose();
      B2.dispose();
      V2.dispose();
      f.dispose();
    });

    it("cached across calls with same shape", () => {
      const f = jit((x: np.Array) => x.ref.mul(x));
      expect(f(np.array([1, 2])).js()).toEqual([1, 4]);
      expect(f(np.array([3, 4])).js()).toEqual([9, 16]);
      f.dispose();
    });
  });

  describe("missing .ref detected", () => {
    it("input used twice without .ref", () => {
      expect(() => jit((x: np.Array) => x.mul(x))(np.array([1]))).toThrow(
        /ref validation failed/,
      );
    });

    it("error mentions the argument index", () => {
      expect(() => jit((x: np.Array) => x.mul(x))(np.array([1]))).toThrow(
        /argument 0/,
      );
    });

    it("error says how many .ref are needed", () => {
      expect(() =>
        jit((x: np.Array) => x.mul(x).add(x))(np.array([1])),
      ).toThrow(/need 2/);
    });

    it("intermediate used twice without .ref", () => {
      expect(() =>
        jit((x: np.Array) => {
          const a = x.mul(2);
          return a.add(a);
        })(np.array([1])),
      ).toThrow(/ref validation failed/);
    });

    it("multiple missing refs reported together", () => {
      try {
        jit((x: np.Array, y: np.Array) => {
          // Both x and y used twice without .ref
          return x.mul(y).add(x.add(y));
        })(np.array([1]), np.array([2]));
        expect.unreachable("should have thrown");
      } catch (e: any) {
        expect(e.message).toMatch(/argument 0/);
        expect(e.message).toMatch(/argument 1/);
      }
    });
  });

  describe("extra .ref detected", () => {
    it("unused extra .ref on input", () => {
      // x used once but has 1 .ref (rc=2, used=1)
      expect(() =>
        jit((x: np.Array) => {
          x.ref; // extra ref, never consumed
          return x.mul(2);
        })(np.array([1])),
      ).toThrow(/ref validation failed/);
    });

    it("error says to remove extra .ref", () => {
      expect(() =>
        jit((x: np.Array) => {
          x.ref;
          return x.mul(2);
        })(np.array([1])),
      ).toThrow(/Remove 1/);
    });
  });

  describe("validateRefs: false skips validation", () => {
    it("missing .ref does not throw", () => {
      const f = jit((x: np.Array) => x.mul(x), { validateRefs: false });
      // Would normally fail validation, but validation is skipped
      expect(f(np.array([2, 3])).js()).toEqual([4, 9]);
      f.dispose();
    });
  });

  describe("with grad", () => {
    it("grad through jit with correct refs", () => {
      const f = jit((x: np.Array) => x.ref.mul(x).sum());
      const df = grad(f);
      const x = np.array([1, 2, 3]);
      // d/dx (x^2).sum() = 2*x
      expect(df(x).js()).toEqual([2, 4, 6]);
      f.dispose();
    });

    it("grad with multi-use and correct refs", () => {
      // f(x) = (x*x + x).sum() => f'(x) = 2x + 1
      const f = jit((x: np.Array) => x.ref.mul(x.ref).add(x).sum());
      const df = grad(f);
      expect(df(np.array([1, 2, 3])).js()).toEqual([3, 5, 7]);
      f.dispose();
    });

    it("valueAndGrad with correct refs", () => {
      const f = jit((x: np.Array) => x.ref.mul(x).sum());
      const vg = valueAndGrad(f);
      const [val, dx] = vg(np.array([1, 2, 3]));
      expect(val.js()).toEqual(14);
      expect(dx.js()).toEqual([2, 4, 6]);
      f.dispose();
    });
  });

  describe("with scan", () => {
    it("scan inside jit with correct refs", () => {
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

    it("scan body with multi-use carry needs .ref", () => {
      // Scan traces its body independently via makeJaxpr with
      // { validateRefs: false }, so .ref is still needed in scan bodies.
      const f = jit((xs: np.Array) => {
        const init = np.array([1]);
        return lax.scan(
          (carry: np.Array, x: np.Array) => {
            // carry used twice → needs .ref inside scan body
            const s = carry.ref.mul(carry).add(x);
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
    it("unused input does not leak", () => {
      const before = slotCount();
      // _x unused → would cause "extra .ref" validation error
      // Use validateRefs: false since unused args trigger validation
      const f = jit((_x: np.Array, y: np.Array) => y.mul(2), {
        validateRefs: false,
      });
      const result = f(np.array([1]), np.array([5]));
      expect(result.js()).toEqual([10]);
      f.dispose();
      expect(slotCount()).toBe(before);
    });

    it("scalar inputs", () => {
      const f = jit((x: np.Array) => x.ref.mul(x.ref).add(x));
      expect(f(np.array(3)).js()).toEqual(12);
      f.dispose();
    });
  });
});

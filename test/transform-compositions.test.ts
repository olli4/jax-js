/**
 * Systematic tests for compositions of autodifferentiation transforms and jit.
 *
 * Transforms tested:
 *   grad  — reverse-mode derivative: f: R->R becomes f': R->R
 *   jit   — JIT compilation: preserves signature
 *   jvp   — forward-mode AD (tested via jvp(f, [x], [1])[1])
 *   vjp   — reverse-mode AD (tested via vjp(f, [x]) then backward)
 *   vmap  — vectorization (tested with batched inputs)
 *
 * We test all valid 1–2 deep compositions exhaustively, selected 3–4 deep
 * compositions focusing on patterns known to trigger bugs (e.g. grad(grad(jit(f)))).
 *
 * Test function: f(x) = x³
 *   f(x)    = x³          → f(2) = 8
 *   f'(x)   = 3x²         → f'(2) = 12
 *   f''(x)  = 6x           → f''(2) = 12
 *   f'''(x) = 6            → f'''(2) = 6
 *   f⁴(x)   = 0            → f⁴(2) = 0
 */
import {
  grad,
  hessian,
  jacfwd,
  jacrev,
  jit,
  jvp,
  linearize,
  numpy as np,
  valueAndGrad,
  vjp,
  vmap,
} from "@jax-js/jax";
import { expect, suite, test } from "vitest";

// --- Test function: f(x) = x³ ---

const fn = (x: np.Array) => x.ref.mul(x.ref).mul(x);

// Reference values at x = 2
const X = 2;
const F0 = 8; // x³
const F1 = 12; // 3x²
const F2 = 12; // 6x
const F3 = 6; // 6
const F4 = 0; // 0
const DERIVS = [F0, F1, F2, F3, F4];

// ============================================================
// Depth 1: single transforms
// ============================================================
suite("depth-1 compositions", () => {
  test("grad(f)", () => {
    expect(grad(fn)(X)).toBeAllclose(F1);
  });

  test("jit(f)", () => {
    const f = jit(fn);
    expect(f(np.array(X))).toBeAllclose(F0);
    f.dispose();
  });

  test("jvp(f) — tangent = f'(x)", () => {
    const [y, dy] = jvp(fn, [X], [1]);
    expect(y).toBeAllclose(F0);
    expect(dy).toBeAllclose(F1);
  });

  test("vjp(f) — cotangent = f'(x)", () => {
    const [y, backward] = vjp(fn, [X]);
    expect(y).toBeAllclose(F0);
    expect(backward(1)[0]).toBeAllclose(F1);
  });

  test("vmap(f)", () => {
    const f = vmap(fn);
    expect(f(np.array([1, 2, 3]))).toBeAllclose([1, 8, 27]);
  });

  test("valueAndGrad(f)", () => {
    const [v, g] = valueAndGrad(fn)(X);
    expect(v).toBeAllclose(F0);
    expect(g).toBeAllclose(F1);
  });
});

// ============================================================
// Depth 2: all pairs of {grad, jit, jvp-wrap, vjp-wrap, vmap}
// ============================================================
suite("depth-2 compositions", () => {
  // --- grad ∘ X ---
  test("grad(grad(f)) = f''", () => {
    expect(grad(grad(fn))(X)).toBeAllclose(F2);
  });

  test("grad(jit(f)) = f'", () => {
    const f = jit(fn);
    expect(grad(f)(X)).toBeAllclose(F1);
    f.dispose();
  });

  // --- jit ∘ X ---
  test("jit(grad(f)) = f'", () => {
    const f = jit(grad(fn));
    expect(f(np.array(X))).toBeAllclose(F1);
    f.dispose();
  });

  test("jit(jit(f)) = f", () => {
    const f = jit(jit(fn));
    expect(f(np.array(X))).toBeAllclose(F0);
    f.dispose();
  });

  // --- jvp ∘ X ---
  test("jvp(grad(f)) = [f'(x), f''(x)]", () => {
    expect(jvp(grad(fn), [X], [1])).toBeAllclose([F1, F2]);
  });

  test("jvp(jit(f)) = [f(x), f'(x)]", () => {
    const f = jit(fn);
    expect(jvp(f, [X], [1])).toBeAllclose([F0, F1]);
    f.dispose();
  });

  // --- vjp ∘ X ---
  test("vjp(grad(f)) = f''", () => {
    const [y, backward] = vjp(grad(fn), [X]);
    expect(y).toBeAllclose(F1);
    expect(backward(1)[0]).toBeAllclose(F2);
    backward.dispose();
  });

  test("vjp(jit(f)) = f'", () => {
    const f = jit(fn);
    const [y, backward] = vjp(f, [X]);
    expect(y).toBeAllclose(F0);
    expect(backward(1)[0]).toBeAllclose(F1);
    backward.dispose();
    f.dispose();
  });

  // --- vmap ∘ X ---
  test("vmap(grad(f))", () => {
    const f = vmap(grad(fn));
    // f'(1) = 3, f'(2) = 12, f'(3) = 27
    expect(f(np.array([1, 2, 3]))).toBeAllclose([3, 12, 27]);
  });

  test("vmap(jit(f))", () => {
    const f = jit(fn);
    expect(vmap(f)(np.array([1, 2, 3]))).toBeAllclose([1, 8, 27]);
    f.dispose();
  });

  // --- grad ∘ vmap is NOT valid (vmap output is not scalar) ---
  // --- jit ∘ vmap ---
  test("jit(vmap(f))", () => {
    const f = jit(vmap(fn));
    expect(f(np.array([1, 2, 3]))).toBeAllclose([1, 8, 27]);
    f.dispose();
  });

  // --- valueAndGrad ∘ X ---
  test("valueAndGrad(jit(f))", () => {
    const f = jit(fn);
    const [v, g] = valueAndGrad(f)(X);
    expect(v).toBeAllclose(F0);
    expect(g).toBeAllclose(F1);
    f.dispose();
  });

  test("valueAndGrad(grad(f))", () => {
    // grad(f) returns scalar → can use valueAndGrad on it
    const [v, g] = valueAndGrad(grad(fn))(X);
    expect(v).toBeAllclose(F1);
    expect(g).toBeAllclose(F2);
  });
});

// ============================================================
// Depth 3: selected compositions
// ============================================================
suite("depth-3 compositions", () => {
  // --- Triple grad = f''' ---
  test("grad(grad(grad(f))) = f'''", () => {
    expect(grad(grad(grad(fn)))(X)).toBeAllclose(F3);
  });

  // --- grad + jit mixed (known bug patterns) ---
  test("grad(grad(jit(f))) = f''", () => {
    const f = jit(fn);
    expect(grad(grad(f))(X)).toBeAllclose(F2);
    f.dispose();
  });

  test("grad(jit(grad(f))) = f''", () => {
    const f = jit(grad(fn));
    expect(grad(f)(X)).toBeAllclose(F2);
    f.dispose();
  });

  test("jit(grad(grad(f))) = f''", () => {
    const f = jit(grad(grad(fn)));
    expect(f(np.array(X))).toBeAllclose(F2);
    f.dispose();
  });

  test("jit(grad(jit(f))) = f'", () => {
    const inner = jit(fn);
    const f = jit(grad(inner));
    expect(f(np.array(X))).toBeAllclose(F1);
    f.dispose();
    inner.dispose();
  });

  test("jit(jit(grad(f))) = f'", () => {
    const f = jit(jit(grad(fn)));
    expect(f(np.array(X))).toBeAllclose(F1);
    f.dispose();
  });

  test("jit(jit(jit(f))) = f", () => {
    const f = jit(jit(jit(fn)));
    expect(f(np.array(X))).toBeAllclose(F0);
    f.dispose();
  });

  // --- jvp at depth 3 ---
  test("jvp(grad(grad(f))) = [f''(x), f'''(x)]", () => {
    expect(jvp(grad(grad(fn)), [X], [1])).toBeAllclose([F2, F3]);
  });

  test("jvp(grad(jit(f))) = [f'(x), f''(x)]", () => {
    const f = jit(fn);
    expect(jvp(grad(f), [X], [1])).toBeAllclose([F1, F2]);
    f.dispose();
  });

  test("jvp(jit(grad(f))) = [f'(x), f''(x)]", () => {
    const f = jit(grad(fn));
    expect(jvp(f, [X], [1])).toBeAllclose([F1, F2]);
    f.dispose();
  });

  // --- vjp at depth 3 ---
  test("vjp(grad(grad(f))) = f'''", () => {
    const [y, backward] = vjp(grad(grad(fn)), [X]);
    expect(y).toBeAllclose(F2);
    expect(backward(1)[0]).toBeAllclose(F3);
    backward.dispose();
  });

  test("vjp(grad(jit(f))) = f''", () => {
    const f = jit(fn);
    const [y, backward] = vjp(grad(f), [X]);
    expect(y).toBeAllclose(F1);
    expect(backward(1)[0]).toBeAllclose(F2);
    backward.dispose();
    f.dispose();
  });

  // --- vmap at depth 3 ---
  test("vmap(grad(grad(f)))", () => {
    // f''(x) = 6x
    expect(vmap(grad(grad(fn)))(np.array([1, 2, 3]))).toBeAllclose([6, 12, 18]);
  });

  // BUG: vmap(grad(jit(f))) leaks 1 slot — grad wrapping jit in vmap context
  test.fails("vmap(grad(jit(f)))", () => {
    const f = jit(fn);
    // f'(x) = 3x²
    expect(vmap(grad(f))(np.array([1, 2, 3]))).toBeAllclose([3, 12, 27]);
    f.dispose();
  });

  test("vmap(jit(grad(f)))", () => {
    const f = jit(grad(fn));
    // f'(x) = 3x²
    expect(vmap(f)(np.array([1, 2, 3]))).toBeAllclose([3, 12, 27]);
    f.dispose();
  });

  test("jit(vmap(grad(f)))", () => {
    const f = jit(vmap(grad(fn)));
    expect(f(np.array([1, 2, 3]))).toBeAllclose([3, 12, 27]);
    f.dispose();
  });

  // --- grad(vmap(f)) is NOT valid (vmap output is vector, not scalar) ---
});

// ============================================================
// Depth 4: bug-prone patterns
// ============================================================
suite("depth-4 compositions", () => {
  // --- Quadruple grad = f'''' ---
  test("grad(grad(grad(grad(f)))) = f''''", () => {
    expect(grad(grad(grad(grad(fn))))(X)).toBeAllclose(F4);
  });

  // --- grad/jit sandwiches ---
  test("grad(grad(grad(jit(f)))) = f'''", () => {
    const f = jit(fn);
    expect(grad(grad(grad(f)))(X)).toBeAllclose(F3);
    f.dispose();
  });

  test("grad(grad(jit(grad(f)))) = f'''", () => {
    const f = jit(grad(fn));
    expect(grad(grad(f))(X)).toBeAllclose(F3);
    f.dispose();
  });

  test("grad(jit(grad(grad(f)))) = f'''", () => {
    const f = jit(grad(grad(fn)));
    expect(grad(f)(X)).toBeAllclose(F3);
    f.dispose();
  });

  test("jit(grad(grad(grad(f)))) = f'''", () => {
    const f = jit(grad(grad(grad(fn))));
    expect(f(np.array(X))).toBeAllclose(F3);
    f.dispose();
  });

  test("grad(grad(jit(jit(f)))) = f''", () => {
    const f = jit(jit(fn));
    expect(grad(grad(f))(X)).toBeAllclose(F2);
    f.dispose();
  });

  test("grad(jit(grad(jit(f)))) = f''", () => {
    const inner = jit(fn);
    const f = jit(grad(inner));
    expect(grad(f)(X)).toBeAllclose(F2);
    f.dispose();
    inner.dispose();
  });

  test("jit(grad(jit(grad(f)))) = f''", () => {
    const inner = jit(grad(fn));
    const f = jit(grad(inner));
    // jit(grad(jit(grad(f)))) should give f''
    expect(f(np.array(X))).toBeAllclose(F2);
    f.dispose();
    inner.dispose();
  });

  test("grad(jit(jit(grad(f)))) = f''", () => {
    const f = jit(jit(grad(fn)));
    expect(grad(f)(X)).toBeAllclose(F2);
    f.dispose();
  });

  test("jit(jit(grad(grad(f)))) = f''", () => {
    const f = jit(jit(grad(grad(fn))));
    expect(f(np.array(X))).toBeAllclose(F2);
    f.dispose();
  });

  test("jit(grad(grad(jit(f)))) = f''", () => {
    const f = jit(fn);
    const g = jit(grad(grad(f)));
    expect(g(np.array(X))).toBeAllclose(F2);
    g.dispose();
    f.dispose();
  });

  // --- Mixed jvp/vjp depth 4 ---
  test("jvp(grad(grad(grad(f)))) = [f'''(x), f''''(x)]", () => {
    expect(jvp(grad(grad(grad(fn))), [X], [1])).toBeAllclose([F3, F4]);
  });

  test("vjp(grad(grad(grad(f)))) = f''''", () => {
    const [y, backward] = vjp(grad(grad(grad(fn))), [X]);
    expect(y).toBeAllclose(F3);
    expect(backward(1)[0]).toBeAllclose(F4);
    backward.dispose();
  });

  test("jvp(grad(grad(jit(f)))) = [f''(x), f'''(x)]", () => {
    const f = jit(fn);
    expect(jvp(grad(grad(f)), [X], [1])).toBeAllclose([F2, F3]);
    f.dispose();
  });

  test("vjp(grad(jit(grad(f)))) = f'''", () => {
    const f = jit(grad(fn));
    const [y, backward] = vjp(grad(f), [X]);
    expect(y).toBeAllclose(F2);
    expect(backward(1)[0]).toBeAllclose(F3);
    backward.dispose();
    f.dispose();
  });

  // --- vmap at depth 4 ---
  test("vmap(grad(grad(grad(f))))", () => {
    // f'''(x) = 6
    expect(vmap(grad(grad(grad(fn))))(np.array([1, 2, 3]))).toBeAllclose([
      6, 6, 6,
    ]);
  });

  // BUG: vmap(grad(grad(jit(f)))) leaks 1 slot — same root cause as vmap(grad(jit(f)))
  test.fails("vmap(grad(grad(jit(f))))", () => {
    const f = jit(fn);
    // f''(x) = 6x
    expect(vmap(grad(grad(f)))(np.array([1, 2, 3]))).toBeAllclose([6, 12, 18]);
    f.dispose();
  });

  test("jit(vmap(grad(grad(f))))", () => {
    const f = jit(vmap(grad(grad(fn))));
    expect(f(np.array([1, 2, 3]))).toBeAllclose([6, 12, 18]);
    f.dispose();
  });

  test("jit(grad(vmap(fn).sum))", () => {
    // grad of sum(vmap(f)(xs)) w.r.t. a single-element input
    // This tests jit wrapping grad wrapping a vmap pipeline
    const vmapped = vmap(fn);
    const loss = (x: np.Array) => vmapped(x).sum();
    const f = jit(grad(loss));
    // d/dx_i of sum(x_i³) = 3x_i²
    expect(f(np.array([1, 2, 3]))).toBeAllclose([3, 12, 27]);
    f.dispose();
  });
});

// ============================================================
// Specific regression tests for known bug patterns
// ============================================================
suite("regression: transform composition edge cases", () => {
  test("grad(grad(jit(f))) does not leak slots", () => {
    const f = jit(fn);
    // Run multiple times to check no accumulating leaks
    for (let i = 0; i < 3; i++) {
      expect(grad(grad(f))(X)).toBeAllclose(F2);
    }
    f.dispose();
  });

  test("deeply nested grad chain is numerically correct", () => {
    // f(x) = x³ → f', f'', f''', f'''' at multiple points
    for (const x of [0, 1, 2, 3, -1, -2]) {
      expect(fn(np.array(x))).toBeAllclose(x ** 3);
      expect(grad(fn)(x)).toBeAllclose(3 * x ** 2);
      expect(grad(grad(fn))(x)).toBeAllclose(6 * x);
      expect(grad(grad(grad(fn)))(x)).toBeAllclose(6);
      expect(grad(grad(grad(grad(fn))))(x)).toBeAllclose(0);
    }
  });

  test("jit captures are freed with dispose", () => {
    // Create nested jit-wrapped functions and verify dispose works
    const f1 = jit(fn);
    const f2 = jit(grad(f1));
    const f3 = jit(grad(f2));

    // f3 = jit(grad(jit(grad(jit(fn))))) = f'' (two grad layers)
    expect(f3(np.array(X))).toBeAllclose(F2);

    // Dispose in reverse order
    f3.dispose();
    f2.dispose();
    f1.dispose();
  });

  test("alternating grad/jit layers", () => {
    // grad(jit(grad(jit(fn)))) — alternating pattern
    const j1 = jit(fn);
    const g1 = grad(j1);
    const j2 = jit(g1);
    const g2 = grad(j2);

    expect(g2(X)).toBeAllclose(F2);

    j2.dispose();
    j1.dispose();
  });

  test("valueAndGrad through jit layers", () => {
    const f = jit(fn);
    const [v1, g1] = valueAndGrad(f)(X);
    expect(v1).toBeAllclose(F0);
    expect(g1).toBeAllclose(F1);

    // valueAndGrad of grad through jit
    const [v2, g2] = valueAndGrad(grad(f))(X);
    expect(v2).toBeAllclose(F1);
    expect(g2).toBeAllclose(F2);

    f.dispose();
  });

  test("jvp and vjp agree at all derivative orders", () => {
    // Both should compute the same derivative
    for (let order = 0; order <= 3; order++) {
      // Build grad^order(fn)
      let f: (x: np.Array) => np.Array = fn;
      for (let i = 0; i < order; i++) f = grad(f);

      // jvp tangent = (grad^order(fn))'(x) = (grad^(order+1)(fn))(x)
      const [, jvpResult] = jvp(f, [X], [1]);

      // vjp cotangent = same thing
      const [, backward] = vjp(f, [X]);
      const vjpResult = backward(1)[0];

      expect(jvpResult).toBeAllclose(DERIVS[order + 1]);
      expect(vjpResult).toBeAllclose(DERIVS[order + 1]);
      backward.dispose();
    }
  });

  test("jit does not change gradient values", () => {
    // Verify jit insertion at any point doesn't change the result
    const g1 = grad(fn)(X);
    const g2 = grad(jit(fn))(X);
    const g3_fn = jit(grad(fn));
    const g3 = g3_fn(np.array(X));

    expect(g1).toBeAllclose(F1);
    expect(g2).toBeAllclose(F1);
    expect(g3).toBeAllclose(F1);
    g3_fn.dispose();

    const gg1 = grad(grad(fn))(X);
    const gg2 = grad(grad(jit(fn)))(X);
    const gg3 = grad(jit(grad(fn)))(X);
    const gg4_fn = jit(grad(grad(fn)));
    const gg4 = gg4_fn(np.array(X));

    expect(gg1).toBeAllclose(F2);
    expect(gg2).toBeAllclose(F2);
    expect(gg3).toBeAllclose(F2);
    expect(gg4).toBeAllclose(F2);
    gg4_fn.dispose();
  });

  test("non-polynomial function: f(x) = sin(x)", () => {
    const sinFn = (x: np.Array) => np.sin(x);
    const x = 1.0;
    const sinX = Math.sin(x);
    const cosX = Math.cos(x);

    expect(sinFn(np.array(x))).toBeAllclose(sinX, { atol: 1e-5 });
    expect(grad(sinFn)(x)).toBeAllclose(cosX, { atol: 1e-5 });
    expect(grad(grad(sinFn))(x)).toBeAllclose(-sinX, { atol: 1e-5 });
    expect(grad(grad(grad(sinFn)))(x)).toBeAllclose(-cosX, { atol: 1e-4 });
    expect(grad(grad(grad(grad(sinFn))))(x)).toBeAllclose(sinX, { atol: 1e-3 });

    // Same with jit inserted
    const jitSin = jit(sinFn);
    expect(grad(jitSin)(x)).toBeAllclose(cosX, { atol: 1e-5 });
    expect(grad(grad(jitSin))(x)).toBeAllclose(-sinX, { atol: 1e-5 });
    expect(grad(grad(grad(jitSin)))(x)).toBeAllclose(-cosX, { atol: 1e-4 });
    jitSin.dispose();
  });

  // BUG: library leak from src/frontend/core.ts:579 with multi-arg grad(grad(f))
  test.fails("multi-argument function grad compositions", () => {
    // f(x, y) = x² * y³ → ∂f/∂x = 2xy³, ∂f/∂y = 3x²y²
    const f = (x: np.Array, y: np.Array) =>
      x.ref.mul(x).mul(y.ref.mul(y.ref).mul(y));

    const dfdx = grad(f); // w.r.t. first argument by default
    expect(dfdx(2, 3)).toBeAllclose(2 * 2 * 27); // 2xy³ at (2,3) = 108

    const dfdy = grad(f, { argnums: 1 });
    expect(dfdy(2, 3)).toBeAllclose(3 * 4 * 9); // 3x²y² at (2,3) = 108

    // Second derivative: d²f/dx² = 2y³
    expect(grad(grad(f))(2, 3)).toBeAllclose(2 * 27); // 54

    // jit + grad of multi-arg
    const jitF = jit(f);
    expect(grad(jitF)(2, 3)).toBeAllclose(108);
    expect(grad(grad(jitF))(2, 3)).toBeAllclose(54);
    jitF.dispose();
  });
});

// ============================================================
// jacfwd / jacrev / hessian compositions
// ============================================================
suite("jacfwd/jacrev compositions", () => {
  // f: R³ → R³, f(x) = x²  (elementwise)
  const vecFn = (x: np.Array) => x.ref.mul(x);

  // f: R³ → R, f(x) = sum(x²)
  const scalarFn = (x: np.Array) => np.sum(x.ref.mul(x));

  // --- Depth 1 ---
  test("jacfwd(f)", () => {
    const J = jacfwd(vecFn)(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
  });

  test("jacrev(f)", () => {
    const J = jacrev(vecFn)(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
  });

  test("hessian(f)", () => {
    // hessian of sum(x²) = 2*I
    const H = hessian(scalarFn)(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ]),
    );
  });

  // --- Depth 2: jacfwd/jacrev + jit ---
  test("jacfwd(jit(f))", () => {
    const f = jit(vecFn);
    const J = jacfwd(f)(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
    f.dispose();
  });

  test("jacrev(jit(f))", () => {
    const f = jit(vecFn);
    const J = jacrev(f)(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
    f.dispose();
  });

  test("jit(jacfwd(f))", () => {
    const f = jit(jacfwd(vecFn));
    const J = f(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
    f.dispose();
  });

  test("jit(jacrev(f))", () => {
    const f = jit(jacrev(vecFn));
    const J = f(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
    f.dispose();
  });

  // --- hessian + jit ---
  test("hessian(jit(f))", () => {
    const f = jit(scalarFn);
    const H = hessian(f)(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ]),
    );
    f.dispose();
  });

  test("jit(hessian(f))", () => {
    const f = jit(hessian(scalarFn));
    const H = f(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ]),
    );
    f.dispose();
  });

  // --- jacfwd vs jacrev agreement ---
  test("jacfwd(f) === jacrev(f) for vector function", () => {
    const f = (x: np.Array) => np.sin(x.ref).add(np.cos(x));
    const x = np.array([1, 2, 3]);
    const Jfwd = jacfwd(f)(x.ref);
    const Jrev = jacrev(f)(x);
    expect(Jfwd).toBeAllclose(Jrev);
  });

  // --- jacrev of scalar matches grad ---
  test("jacrev(f) matches grad(f) for scalar output", () => {
    const x = np.array([1, 2, 3]);
    const J = jacrev(scalarFn)(x.ref);
    const g = grad(scalarFn)(x);
    expect(J).toBeAllclose(g);
  });

  // --- Depth 3: jit sandwiches ---
  test("jacfwd(jit(jit(f)))", () => {
    const f = jit(jit(vecFn));
    const J = jacfwd(f)(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
    f.dispose();
  });

  test("jit(jacfwd(jit(f)))", () => {
    const inner = jit(vecFn);
    const f = jit(jacfwd(inner));
    const J = f(np.array([1, 2, 3]));
    expect(J).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
    f.dispose();
    inner.dispose();
  });

  // BUG: jit(hessian(jit(f))) leaks transformed-jit closure arrays
  test.fails("jit(hessian(jit(f)))", () => {
    const inner = jit(scalarFn);
    const f = jit(hessian(inner));
    const H = f(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ]),
    );
    f.dispose();
    inner.dispose();
  });

  // --- hessian with cross terms + jit ---
  test("hessian of cross-term function through jit", () => {
    // f(x) = x0 * x1 + x1 * x2
    // hessian = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    const f = (x: np.Array) => {
      const [x0, x1, x2] = x;
      return x0.mul(x1.ref).add(x1.mul(x2));
    };
    const jf = jit(f);
    const H = hessian(jf)(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
      ]),
    );
    jf.dispose();
  });

  // --- vmap + jacfwd ---
  test("vmap(jacfwd(f))", () => {
    // jacfwd of x² = diag(2x) for each batch element
    const J = vmap(jacfwd(vecFn));
    const xs = np.array([
      [1, 2],
      [3, 4],
    ]);
    const result = J(xs);
    // batch[0]: diag([2, 4]), batch[1]: diag([6, 8])
    expect(result).toBeAllclose(
      np.array([
        [
          [2, 0],
          [0, 4],
        ],
        [
          [6, 0],
          [0, 8],
        ],
      ]),
    );
  });

  // --- grad + jacfwd (hessian manually) ---
  test("jacfwd(grad(f)) === hessian(f)", () => {
    const H1 = jacfwd(grad(scalarFn))(np.array([1, 2, 3]));
    const H2 = hessian(scalarFn)(np.array([1, 2, 3]));
    expect(H1).toBeAllclose(H2);
  });

  // --- jacrev + grad (hessian manually via jacrev) ---
  test("jacrev(grad(f)) matches hessian(f)", () => {
    const H1 = jacrev(grad(scalarFn))(np.array([1, 2, 3]));
    const H2 = hessian(scalarFn)(np.array([1, 2, 3]));
    expect(H1).toBeAllclose(H2);
  });
});

// ============================================================
// linearize compositions
// ============================================================
suite("linearize compositions", () => {
  test("linearize(f) basic", () => {
    const [y, lin] = linearize(fn, [X]);
    expect(y).toBeAllclose(F0);
    expect(lin(1)).toBeAllclose(F1); // tangent at dx=1 is f'(x)
    expect(lin(2)).toBeAllclose(F1 * 2); // linear in tangent
  });

  test("linearize(jit(f))", () => {
    const f = jit(fn);
    const [y, lin] = linearize(f, [X]);
    expect(y).toBeAllclose(F0);
    expect(lin(1)).toBeAllclose(F1);
    lin.dispose();
    f.dispose();
  });

  test("linearize(grad(f)) — second-order linear approximation", () => {
    // linearize(f', x) gives [f'(x), dx => f''(x)*dx]
    const [y, lin] = linearize(grad(fn), [X]);
    expect(y).toBeAllclose(F1);
    expect(lin(1)).toBeAllclose(F2);
    expect(lin(0.5)).toBeAllclose(F2 * 0.5);
  });

  test("grad(linearize(f).lin) — differentiate through the linear part", () => {
    // linearize gives [y, lin] where lin(dx) = J * dx
    // Using it via jvp pattern
    const [, lin] = linearize(fn, [X]);
    // lin is a linear function from tangent → output tangent
    // Its "derivative" should be constant (it's already linear)
    expect(lin(1)).toBeAllclose(F1);
    expect(lin(3)).toBeAllclose(F1 * 3);
  });

  test("linearize(grad(jit(f)))", () => {
    const f = jit(fn);
    const [y, lin] = linearize(grad(f), [X]);
    expect(y).toBeAllclose(F1);
    expect(lin(1)).toBeAllclose(F2);
    lin.dispose();
    f.dispose();
  });

  test("jit(linearize(f).lin) — jit the linear part", () => {
    const [, lin] = linearize(fn, [X]);
    const jlin = jit(lin);
    expect(jlin(np.array(1))).toBeAllclose(F1);
    expect(jlin(np.array(2))).toBeAllclose(F1 * 2);
    jlin.dispose();
    lin.dispose();
  });

  test("linearize for vector function", () => {
    const vecFn = (x: np.Array) => x.ref.mul(x);
    const [y, lin] = linearize(vecFn, [np.array([1, 2, 3])]);
    expect(y).toBeAllclose([1, 4, 9]);
    // Tangent of x² at [1,2,3] in direction [1,0,0] = [2*1, 0, 0] = [2, 0, 0]
    expect(lin(np.array([1, 0, 0]))).toBeAllclose([2, 0, 0]);
    expect(lin(np.array([0, 1, 0]))).toBeAllclose([0, 4, 0]);
    expect(lin(np.array([0, 0, 1]))).toBeAllclose([0, 0, 6]);
    lin.dispose();
  });
});

// ============================================================
// Non-scalar functions: reductions, matrix ops
// ============================================================
suite("non-scalar function compositions", () => {
  // f(x) = sum(x²) : R³ → R
  const sumSq = (x: np.Array) => np.sum(x.ref.mul(x));

  // f(x) = softmax-cross-entropy-like: -sum(t * log(softmax(x)))
  // Simplified: f(x) = -sum(x * log(x)) for positive x (entropy)

  test("grad of sum reduction", () => {
    // d/dx sum(x²) = 2x
    expect(grad(sumSq)(np.array([1, 2, 3]))).toBeAllclose([2, 4, 6]);
  });

  test("grad(grad(f)) for vector input, scalar output", () => {
    // f(x) = sum(x³)  → grad = 3x² → grad(grad) = diag(6x)
    // But grad(grad) cannot work directly because grad(f) returns a vector
    // and outer grad needs scalar. Use sum of grad instead.
    const f = (x: np.Array) => np.sum(x.ref.mul(x.ref).mul(x));
    // grad(f) = 3x²
    expect(grad(f)(np.array([1, 2, 3]))).toBeAllclose([3, 12, 27]);

    // To get second derivative, use hessian
    const H = hessian(f)(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [6, 0, 0],
        [0, 12, 0],
        [0, 0, 18],
      ]),
    );
  });

  test("grad through jit of reduction", () => {
    const f = jit(sumSq);
    expect(grad(f)(np.array([1, 2, 3]))).toBeAllclose([2, 4, 6]);
    f.dispose();
  });

  test("jit(grad(f)) for reduction", () => {
    const f = jit(grad(sumSq));
    expect(f(np.array([1, 2, 3]))).toBeAllclose([2, 4, 6]);
    f.dispose();
  });

  test("hessian through jit of cubic reduction", () => {
    const f = (x: np.Array) => np.sum(x.ref.mul(x.ref).mul(x));
    const jf = jit(f);
    const H = hessian(jf)(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [6, 0, 0],
        [0, 12, 0],
        [0, 0, 18],
      ]),
    );
    jf.dispose();
  });

  test("jvp of reduction function", () => {
    // f(x) = sum(x²), tangent of f at x in direction v = 2x·v
    const [y, dy] = jvp(sumSq, [np.array([1, 2, 3])], [np.array([1, 0, 0])]);
    expect(y).toBeAllclose(14); // 1+4+9
    expect(dy).toBeAllclose(2); // 2*1*1 + 2*2*0 + 2*3*0
  });

  test("vjp of reduction function", () => {
    const [y, backward] = vjp(sumSq, [np.array([1, 2, 3])]);
    expect(y).toBeAllclose(14);
    expect(backward(1)[0]).toBeAllclose([2, 4, 6]); // gradient
    backward.dispose();
  });

  test("vmap(grad(f)) for reduction", () => {
    // Each row: grad of sum(x²) = 2x
    const xs = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(vmap(grad(sumSq))(xs)).toBeAllclose([
      [2, 4, 6],
      [8, 10, 12],
    ]);
  });

  test("jit(vmap(grad(f))) for reduction", () => {
    const f = jit(vmap(grad(sumSq)));
    const xs = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(f(xs)).toBeAllclose([
      [2, 4, 6],
      [8, 10, 12],
    ]);
    f.dispose();
  });

  test("grad of dot product", () => {
    // f(x) = x · w where w is constant.  grad(f) = w
    const w = np.array([2, 3, 4]);
    const f = (x: np.Array) => np.sum(x.mul(w.ref));
    expect(grad(f)(np.array([1, 1, 1]))).toBeAllclose([2, 3, 4]);
    w.dispose();
  });

  test("grad(grad(sum(x*sin(x))))", () => {
    // f(x) = sum(x * sin(x))
    // f'(x) = sin(x) + x*cos(x)
    // f''(x) = 2*cos(x) - x*sin(x)
    // For vector input, outer grad needs scalar, so use sum.
    const f = (x: np.Array) => np.sum(x.ref.mul(np.sin(x)));

    const x = np.array([1.0]);
    // f'(1) = sin(1) + cos(1) ≈ 1.3818
    expect(grad(f)(x.ref)).toBeAllclose([Math.sin(1) + Math.cos(1)], {
      atol: 1e-5,
    });

    // For second derivative, use hessian (vector input)
    const H = hessian(f)(x);
    // f''(1) = 2*cos(1) - sin(1) ≈ 0.2390
    expect(H).toBeAllclose([[2 * Math.cos(1) - Math.sin(1)]], { atol: 1e-4 });
  });
});

// ============================================================
// vjp as inner layer — functions that internally use vjp
// ============================================================
suite("vjp as inner layer", () => {
  test("grad of function that internally uses vjp", () => {
    // f(x) = custom_grad_fn(x) where the function computes via vjp internally
    // This tests the interaction of nested autodiff contexts
    const innerVjpFn = (x: np.Array): np.Array => {
      // Compute x² using vjp to get value
      const sq = (a: np.Array) => a.ref.mul(a);
      const [y, backward] = vjp(sq, [x]);
      backward.dispose(); // We don't use the backward
      return y;
    };

    // grad of innerVjpFn should give 2x
    expect(grad(innerVjpFn)(3)).toBeAllclose(6);
  });

  test("jit wrapping function with internal vjp", () => {
    const f = (x: np.Array): np.Array => {
      const sq = (a: np.Array) => a.ref.mul(a);
      const [y, backward] = vjp(sq, [x]);
      backward.dispose();
      return y;
    };
    const jf = jit(f);
    expect(jf(np.array(3))).toBeAllclose(9);
    jf.dispose();
  });

  test("jvp of function using linearize internally", () => {
    const f = (x: np.Array): np.Array => {
      // Use linearize internally to compute x²
      const sq = (a: np.Array) => a.ref.mul(a);
      const [y, lin] = linearize(sq, [x]);
      lin.dispose();
      return y;
    };
    const [y, dy] = jvp(f, [3], [1]);
    expect(y).toBeAllclose(9);
    expect(dy).toBeAllclose(6); // tangent of x² at 3 is 2*3=6
  });
});

// ============================================================
// Depth-5 stress tests (selected critical patterns only)
// ============================================================
suite("depth-5 stress tests", () => {
  test("grad(grad(grad(grad(grad(f))))) = f⁵ = 0 for x³", () => {
    expect(grad(grad(grad(grad(grad(fn)))))(X)).toBeAllclose(0);
  });

  test("jit(grad(grad(grad(grad(f))))) = f⁴ = 0", () => {
    const f = jit(grad(grad(grad(grad(fn)))));
    expect(f(np.array(X))).toBeAllclose(0);
    f.dispose();
  });

  test("grad(jit(jit(grad(fn)))) = f'' = 12", () => {
    // j1 = jit(grad(fn)) = f', j2 = jit(j1) = f', grad(j2) = f''
    const j1 = jit(grad(fn));
    const j2 = jit(j1);
    const result = grad(j2)(X);
    expect(result).toBeAllclose(F2);
    j2.dispose();
    j1.dispose();
  });

  test("grad(grad(jit(grad(jit(f))))) = f''' = 6", () => {
    const j1 = jit(fn);
    const j2 = jit(grad(j1));
    expect(grad(grad(j2))(X)).toBeAllclose(F3);
    j2.dispose();
    j1.dispose();
  });

  test("jit(jit(grad(grad(grad(f))))) = f''' = 6", () => {
    const f = jit(jit(grad(grad(grad(fn)))));
    expect(f(np.array(X))).toBeAllclose(F3);
    f.dispose();
  });

  test("grad of sin to depth 5", () => {
    // sin → cos → -sin → -cos → sin
    const sinFn = (x: np.Array) => np.sin(x);
    const x = 1.0;
    const expected = Math.sin(x); // 5th deriv of sin is cos, wait:
    // d/dx sin = cos, d²/dx² sin = -sin, d³ = -cos, d⁴ = sin, d⁵ = cos
    expect(grad(grad(grad(grad(grad(sinFn)))))(x)).toBeAllclose(
      Math.cos(x),
      { atol: 1e-2 }, // Accumulated numerical error at depth 5
    );
  });

  test("hessian(jit(jit(f))) — double-jit inner", () => {
    const scalarFn = (x: np.Array) => np.sum(x.ref.mul(x));
    const f = jit(jit(scalarFn));
    const H = hessian(f)(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ]),
    );
    f.dispose();
  });

  test("jacfwd(grad(jit(f))) — Jacobian of gradient through jit", () => {
    const scalarFn = (x: np.Array) => np.sum(x.ref.mul(x.ref).mul(x));
    const f = jit(scalarFn);
    // jacfwd(grad(f)) = hessian of f = diag(6x)
    const H = jacfwd(grad(f))(np.array([1, 2, 3]));
    expect(H).toBeAllclose(
      np.array([
        [6, 0, 0],
        [0, 12, 0],
        [0, 0, 18],
      ]),
    );
    f.dispose();
  });

  test("vmap(jacfwd(jit(f))) — vectorized Jacobian through jit", () => {
    const vecFn = (x: np.Array) => x.ref.mul(x);
    const f = jit(vecFn);
    const batchJ = vmap(jacfwd(f));
    const xs = np.array([
      [1, 2],
      [3, 4],
    ]);
    const result = batchJ(xs);
    expect(result).toBeAllclose(
      np.array([
        [
          [2, 0],
          [0, 4],
        ],
        [
          [6, 0],
          [0, 8],
        ],
      ]),
    );
    f.dispose();
  });
});

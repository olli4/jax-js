import { jit, lax, numpy as np } from "@jax-js/jax";

// This example draws a Mandelbrot fractal using lax.scan for the iteration loop.
// The scan compiles to a fused GPU loop — all 100 iterations run in a single
// WebGPU/WASM dispatch, with no JS overhead per step.
const width = 750;
const height = 600;
const iters = 100;

type Carry = { A: np.Array; B: np.Array; V: np.Array };

const calculateMandelbrot = jit((X: np.Array, Y: np.Array) => {
  const step = (carry: Carry, _x: null): [Carry, null] => {
    const { A, B, V } = carry;
    const Asq = A.ref.mul(A.ref);
    const Bsq = B.ref.mul(B.ref);
    const newV = V.add(Asq.ref.add(Bsq.ref).less(100).astype(np.float32));
    const newA = np.clip(Asq.sub(Bsq).add(X.ref), -50, 50);
    const newB = np.clip(A.mul(B).mul(2).add(Y.ref), -50, 50);
    return [{ A: newA, B: newB, V: newV }, null];
  };

  const init: Carry = {
    A: np.zeros(X.shape),
    B: np.zeros(Y.shape),
    V: np.zeros(X.shape),
  };
  const [final, _ys] = lax.scan(step, init, null, { length: iters });
  return final.V;
});

const x = np.linspace(-2, 0.5, width);
const y = np.linspace(-1, 1, height);
const [X, Y] = np.meshgrid([x, y]);

const ar = calculateMandelbrot(X, Y);
calculateMandelbrot.dispose();
const image = np.subtract(1, ar.div(iters));

// The REPL has a displayImage() builtin for drawing image pixels.
await displayImage(image);

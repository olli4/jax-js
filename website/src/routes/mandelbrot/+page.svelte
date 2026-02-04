<script lang="ts">
  import { defaultDevice, init, jit, lax, numpy as np } from "@jax-js/jax";
  import { onMount } from "svelte";

  const width = 1000;
  const height = 800;

  let milliseconds = $state(0);

  onMount(async () => {
    await init("webgpu");
    defaultDevice("webgpu");
  });

  function mandelbrotIteration(
    A: np.Array,
    B: np.Array,
    V: np.Array,
    X: np.Array,
    Y: np.Array,
  ) {
    const Asq = A.ref.mul(A.ref);
    const Bsq = B.ref.mul(B.ref);
    V = V.add(Asq.ref.add(Bsq.ref).less(100).astype(np.float32));
    const A2 = np.clip(Asq.sub(Bsq).add(X), -50, 50);
    const B2 = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
    return [A2, B2, V];
  }

  const mandelbrotMultiple = (iters: number) =>
    jit((A: np.Array, B: np.Array, V: np.Array, X: np.Array, Y: np.Array) => {
      for (let i = 0; i < iters; i++) {
        [A, B, V] = mandelbrotIteration(A, B, V, X.ref, Y.ref);
      }
      X.dispose();
      Y.dispose();
      return [A, B, V];
    });

  function calculateMandelbrot(iters: number) {
    const x = np.linspace(-2, 0.5, width);
    const y = np.linspace(-1, 1, height);

    const [X, Y] = np.meshgrid([x, y]);

    const f = jit(mandelbrotIteration);

    let A = np.zeros(X.shape);
    let B = np.zeros(Y.shape);
    let V = np.zeros(X.shape);
    for (let i = 0; i < iters; i++) {
      console.log(`Iteration ${i + 1}/${iters}`);
      [A, B, V] = f(A, B, V, X.ref, Y.ref);
    }
    X.dispose();
    Y.dispose();
    A.dispose();
    B.dispose();
    f.dispose();

    return V;
  }

  function calculateMandelbrotJit10(iters: number) {
    const x = np.linspace(-2, 0.5, width);
    const y = np.linspace(-1, 1, height);

    const [X, Y] = np.meshgrid([x, y]);

    const f = mandelbrotMultiple(10);

    let A = np.zeros(X.shape);
    let B = np.zeros(Y.shape);
    let V = np.zeros(X.shape);
    for (let i = 0; i < iters / 10; i++) {
      console.log(`Iteration ${i + 1}/${iters / 10}`);
      [A, B, V] = f(A, B, V, X.ref, Y.ref);
    }
    X.dispose();
    Y.dispose();
    A.dispose();
    B.dispose();
    f.dispose();

    return V;
  }

  function calculateMandelbrotScan(iters: number) {
    const x = np.linspace(-2, 0.5, width);
    const y = np.linspace(-1, 1, height);

    const [X, Y] = np.meshgrid([x, y]);

    // Use lax.scan with Y=null (no output stacking needed)
    const f = jit(
      (
        A: np.Array,
        B: np.Array,
        V: np.Array,
        X: np.Array,
        Y: np.Array,
      ): [np.Array, np.Array, np.Array] => {
        type Carry = { A: np.Array; B: np.Array; V: np.Array };

        const step = (
          carry: Carry,
          _x: null,
        ): [Carry, null] => {
          const { A, B, V } = carry;
          const Asq = A.ref.mul(A.ref);
          const Bsq = B.ref.mul(B.ref);
          const newV = V.add(Asq.ref.add(Bsq.ref).less(100).astype(np.float32));
          const newA = np.clip(Asq.sub(Bsq).add(X.ref), -50, 50);
          const newB = np.clip(A.mul(B).mul(2).add(Y.ref), -50, 50);
          return [{ A: newA, B: newB, V: newV }, null];
        };

        const init: Carry = { A, B, V };
        const [final, _ys] = lax.scan(step, init, null, { length: iters });
        X.dispose();
        Y.dispose();
        return [final.A, final.B, final.V];
      },
    );

    const A = np.zeros(X.shape);
    const B = np.zeros(Y.shape);
    const V = np.zeros(X.shape);
    const [_A2, _B2, V2] = f(A, B, V, X, Y);
    f.dispose();
    _A2.dispose();
    _B2.dispose();

    return V2;
  }

  let canvas: HTMLCanvasElement;

  function renderMandelbrot(result: Int32Array) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    for (let i = 0; i < result.length; i++) {
      const value = 255 * (1 - result[i] / 100);
      data[i * 4] = value; // Red
      data[i * 4 + 1] = value; // Green
      data[i * 4 + 2] = value; // Blue
      data[i * 4 + 3] = 255; // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
  }
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">mandelbrot in jax-js</h1>

  <p class="mb-4">
    NumPy + GPU + JIT, in JavaScript! Open the browser console to see more.
  </p>

  <div class="flex flex-wrap gap-2 mb-4">
    <button
      onmousedown={async () => {
        const start = performance.now();
        const result = (await calculateMandelbrot(100).data()) as Int32Array;
        milliseconds = performance.now() - start;
        console.log(`Mandelbrot calculated in ${milliseconds} ms`);
        renderMandelbrot(result);
      }}
    >
      JS Loop (100 iters)
    </button>

    <button
      onmousedown={async () => {
        const start = performance.now();
        const result = (await calculateMandelbrotJit10(100).data()) as Int32Array;
        milliseconds = performance.now() - start;
        console.log(`Mandelbrot (jit10) calculated in ${milliseconds} ms`);
        renderMandelbrot(result);
      }}
    >
      JS Loop (10×10 iters)
    </button>

    <button
      onmousedown={async () => {
        const start = performance.now();
        const result = (await calculateMandelbrotScan(100).data()) as Int32Array;
        milliseconds = performance.now() - start;
        console.log(`Mandelbrot (scan) calculated in ${milliseconds} ms`);
        renderMandelbrot(result);
      }}
    >
      lax.scan (100 iters)
    </button>
  </div>

  {#if milliseconds}
    <span class="text-sm">Computed in {milliseconds.toFixed(1)} ms</span>
  {/if}

  <canvas bind:this={canvas} {width} {height} class="my-8"></canvas>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border px-2 hover:bg-gray-100 active:scale-95;
  }
</style>

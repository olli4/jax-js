<script lang="ts">
  import { init, numpy as np, setBackend } from "@jax-js/jax";
  import { onMount } from "svelte";

  onMount(async () => {
    await init("webgpu");
    setBackend("webgpu");
  });

  function mandelbrotIteration(
    A: np.Array,
    B: np.Array,
    X: np.Array,
    Y: np.Array,
  ) {
    const A2 = np.clip(A.ref.mul(A.ref).sub(B.ref.mul(B.ref)).add(X), -50, 50);
    const B2 = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
    return [A2, B2];
  }

  function calculateMandelbrot(iters: number) {
    const x = np.linspace(-2, 0.5, 500);
    const y = np.linspace(-1, 1, 400);

    const [X, Y] = np.meshgrid([x, y]);

    let A = np.zeros(X.shape);
    let B = np.zeros(Y.shape);
    for (let i = 0; i < iters; i++) {
      [A, B] = mandelbrotIteration(A, B, X.ref, Y.ref);
    }
    X.dispose();
    Y.dispose();

    return A.ref.mul(A).add(B.ref.mul(B)).less(100);
  }

  let canvas: HTMLCanvasElement;

  function renderMandelbrot(result: Int32Array) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    for (let i = 0; i < result.length; i++) {
      const value = result[i] ? 255 : 0;
      data[i * 4] = value; // Red
      data[i * 4 + 1] = value; // Green
      data[i * 4 + 2] = value; // Blue
      data[i * 4 + 3] = 255; // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
  }
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">mandelbrot</h1>

  <button
    onclick={async () => {
      // TODO: async does not work yet?
      const result = calculateMandelbrot(50).dataSync() as Int32Array;
      console.log(result);
      renderMandelbrot(result);
    }}
  >
    Calculate Mandelbrot
  </button>

  <canvas bind:this={canvas} width="500" height="400" class="my-8"></canvas>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border px-2 hover:bg-gray-100 active:scale-95;
  }
</style>

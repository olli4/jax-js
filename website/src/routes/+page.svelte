<script lang="ts">
  import { base } from "$app/paths";
</script>

<main class="mx-auto my-12 px-4 sm:px-6 max-w-screen-md">
  <div class="mb-10">
    <h1 class="text-6xl text-center font-bold mb-4">jax-js</h1>
    <p class="italic text-center">Numerical and GPU computing on the web</p>
  </div>

  <p class="mb-5">
    In 2025, most software is written in JavaScript and runs in browsers. But ML
    libraries don't.
  </p>

  <p class="mb-5">What if they did?</p>

  <pre class="mb-5 text-center"><code>npm install @jax-js/core</code></pre>

  <p class="mb-5">
    And you had access to familiar array APIs, autograd, and linear algebra.
    This is <em>all you need</em> for neural networks (AI), physics simulations,
    statistics, image processing, and more. No setup—just start programming.
  </p>

  <div class="mb-5 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4">
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">julia set</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">mnist</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">llama chatbot</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">bayesian regression</span>
    </div>
  </div>

  <p class="mb-5">
    <code>jax-js</code> is a numerical computing library for web developers.
    Like Python libraries (NumPy, PyTorch, JAX), it's carefully hand-optimized,
    including a JIT compiler and kernel scheduler for running on your GPU. But
    it's also
    <strong>extremely simple and portable across the web</strong>. It runs on
    Chrome, Firefox, Safari, iOS, Arc, and other browsers.
  </p>

  <div class="mb-5 border h-48 flex items-center justify-center square-grid">
    <p class="text-sm italic text-center">
      Performance graph of flops: cpu, wasm, webgl, webgpu
    </p>
  </div>

  <p class="mb-5">
    How is this possible? In short: because of modern technologies like
    WebAssembly, WebGL, and WebGPU that give us fast, low-level access to
    hardware. <span class="italic"
      >(Performance benchmarks coming soon—expect <code>jax-js</code> to outperform
      TensorFlow.js and others.)</span
    >
  </p>

  <p class="mb-5">
    Want to distribute software? Make a website. And if you want to run neural
    networks or numerical code on that website, use <code>jax-js</code>.
  </p>

  <p>
    Try it in the
    <a href="{base}/repl" class="link">jax-js REPL</a>.
  </p>
</main>

<style lang="postcss">
  @reference "$app.css";

  .square-grid {
    background-size: 40px 40px;
    background-image:
      linear-gradient(to right, var(--color-gray-200) 1px, transparent 1px),
      linear-gradient(to bottom, var(--color-gray-200) 1px, transparent 1px);
    background-position: center center;
  }

  a.link {
    @apply underline;
  }
</style>

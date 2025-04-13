<script lang="ts">
  import { base } from "$app/paths";
</script>

<svelte:head>
  <title>jax-js</title>
</svelte:head>

<main class="mx-auto my-12 px-4 sm:px-6 max-w-screen-md">
  <div class="mb-10">
    <h1 class="text-6xl text-center font-bold mb-4">jax-js</h1>
    <p class="italic text-center">
      Numerical computing foundations for the web
    </p>
  </div>

  <p class="mb-6">
    Have you ever wanted to use NumPy or PyTorch from the browser?
  </p>

  <p class="mb-6">
    Really, machine learning belongs in the browser—instantly accessible and
    running locally. But the best libraries are in Python; they weren't
    <em>built for</em> the web. We're trying to change that.
  </p>

  <pre class="mb-6 text-center"><code>npm install @jax-js/core</code></pre>

  <p class="mb-6">
    <code>jax-js</code> brings high-performance WebAssembly and GPU kernels to JavaScript,
    with a simple API. So you can do neural networks, statistics, image processing,
    and all sorts of numerical computing.
  </p>

  <div class="mb-6 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4">
    <!-- TODO: Images -->
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

  <p class="mb-6">
    Like the <a href="https://jax.dev" target="_blank" class="link"
      >JAX Python library</a
    >
    it's based on, <code>jax-js</code> is carefully optimized, including a JIT
    compiler and GPU kernel scheduler. It's also
    <strong>extremely fast and portable</strong>, running on Chrome, Firefox,
    Safari, iOS, and Android, while having best-in-class performance.
  </p>

  <div class="mb-6 border h-48 flex items-center justify-center square-grid">
    <p class="text-sm italic text-center">
      <!-- TODO: Insert graph -->
      Performance graph of flops: cpu, wasm, webgl, webgpu
    </p>
  </div>

  <p class="mb-6">
    How is this possible? <code>jax-js</code> compiles math into optimized
    WebAssembly, WebGL, and WebGPU code on modern browsers. Its generated code
    is even faster than specialized neural network libraries like
    <a href="https://www.tensorflow.org/js/" target="_blank" class="link"
      >Google's TensorFlow</a
    >
    at running their own models.
  </p>

  <p class="mb-6">
    <span class="italic"
      >XXX Performance benchmarks coming soon—though see
      <a href="{base}/matmul" class="link">here</a> for some initial results.</span
    >
  </p>

  <p class="mb-6">
    NumPy (2005), PyTorch (2016), and JAX (2018) revolutionized numerical
    computing and sparked a flourishing ecosystem in Python. <code>jax-js</code>
    aspires to bring that same foundation to JavaScript.
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

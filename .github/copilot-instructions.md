# GitHub Copilot instructions for jax-js

These notes help AI coding agents be immediately productive. The document has two parts:

1. **Repository Overview** — General jax-js knowledge for any development work
2. **Scan Feature Reference** — `lax.scan` implementation details and backend-specific behavior

---

# Part 1: Repository Overview

## What is jax-js?

jax-js is a JavaScript/TypeScript port of [JAX](https://github.com/google/jax), Google's library for
high-performance numerical computing. It brings **numerical computing and machine learning** to the
web as a first-class capability — not just model inference, but the full stack: array operations,
automatic differentiation, JIT compilation, and composable transformations.

- **NumPy-like array operations** in the browser or Node.js
- **Automatic differentiation** — compute gradients of arbitrary functions for ML training
- **JIT compilation** — trace functions once, compile to optimized GPU/WASM code
- **Transformations** — `grad`, `vmap`, `jit` compose to build complex ML pipelines

The target audience is broad: scientists, artists, ML practitioners, data scientists — anyone who
needs numerical computing but shouldn't have to write GPU shaders or manage a Python environment.
jax-js makes the JAX/NumPy API available anywhere a browser (or Node.js/Deno) runs.

### Key concepts

**Tracing:** When you call a jit-wrapped function, jax-js executes it with "tracer" objects instead
of real arrays. This records what operations happen, producing an intermediate representation
(Jaxpr) that can be compiled to efficient native code. The function runs once for tracing, then the
compiled version runs for all subsequent calls.

**Kernels:** A "kernel" is a compiled computation unit. On GPU, this is a WGSL shader program; on
WASM, it's a WebAssembly module. jax-js _fuses_ multiple elementwise operations into single kernels
to minimize dispatch overhead — instead of launching one GPU dispatch per `add`, `mul`, `exp`, it
combines them into one.

**Autodiff intuition:** `grad(f)` returns a new function that computes `f`'s gradient. Internally,
jax-js traces `f` to build a computation graph, then applies the chain rule automatically. The `jvp`
(forward-mode) and `vjp` (reverse-mode) primitives enable efficient gradient computation for
different use cases.

## Project vision & design philosophy

jax-js solves two problems simultaneously: **numerical computing in the browser** (statistics,
signal processing, simulations, classical ML) and **GPU compute in the browser** (WebGPU shaders,
parallel number crunching). JAX's tracing-based design is the ideal bridge — it lets users write
high-level array code that compiles down to fast native kernels automatically.

### Generative compiler over static kernel libraries

The central architectural bet is that **generating kernels from an IR at runtime beats shipping a
library of pre-compiled kernels** (which is what TensorFlow.js and ONNX Runtime Web do). This means:

- New operations don't require new hand-written kernels — `matmul` is just reshape + multiply +
  reduce, compiled via `jit`.
- Kernel fusion happens automatically across arbitrary operation chains.
- Performance scales with compiler improvements, not manual kernel engineering.
- The library stays lightweight — XLA is 200+ KLoC, far too large for a browser bundle.

This approach draws from tinygrad's insight that a minimal set of operations + a view-tracking
system (ShapeTracker) + simple fusion heuristics can get you surprisingly far. Combined with JAX's
tracing model, it yields a composable system where `grad`, `jit`, and `vmap` all interoperate
naturally.

### Design tradeoffs to keep in mind

- **"80% of XLA" philosophy** — jax-js aims for the sweet spot of correctness and performance
  without squeezing out every last drop. We don't know what hardware we're running on (it's a
  browser library), so we target **3–5× of optimal** rather than peak.
- **Lightweight over exhaustive** — prefer a small, composable set of primitives over a large
  surface area of specialized ops. If something can be expressed via existing primitives and `jit`,
  that's preferred over adding a new backend kernel.
- **Move semantics over GC** — JavaScript has no reliable destructor, so jax-js uses explicit
  ownership (`.ref` / `.dispose()`). This is a deliberate choice, not a workaround. It enables
  predictable memory management on GPU, which is essential for real applications.
- **Compounding returns** — every improvement to the compiler makes _all_ operations faster, every
  new primitive gets autodiff for free, every `jit`-wrapped function gets kernel fusion
  automatically. Prioritize work that compounds.

### Development priorities

When deciding what to work on, prefer work in this order:

1. **Correctness first** — tests, reference-counting discipline, cross-backend consistency.
2. **API breadth** — approximate NumPy/JAX API compatibility (see `FEATURES.md` for the status
   table). The goal is that common JAX/NumPy patterns can be ported with minimal changes.
3. **Performance** — there is significant headroom, especially in:
   - WASM backend (SIMD, loop unrolling, multi-threading via SharedArrayBuffer).
   - Transformer inference (currently ~1/3 of raw matmul GFLOP/s).
   - Conv2d and other operations that haven't been tuned yet.
4. **Demos and applications** — fluid simulations, neural networks, audio processing, embedding
   search, fractals. These serve as integration tests and showcase what's possible.

### What jax-js is NOT

- Not a model-serving runtime like ONNX Runtime — it's a **framework** for writing and training
  numerical programs, not just running pre-packaged inference.
- Not trying to match XLA's peak CUDA performance — the target is the browser and web runtimes.
- Not a Python replacement — it's for when you want numerical computing **where JavaScript runs**.

## Architecture overview

- **Core library** (`@jax-js/jax`, root `src/`): array API, autodiff (`grad`, `jvp`, `vjp`), JIT
  compilation, device placement.
  - Frontend modules in `src/frontend/`: `array.ts` (Array class), `jit.ts` (kernel fusion),
    `jvp.ts`/`linearize.ts` (forward/reverse AD), `vmap.ts` (vectorization), `convolution.ts`.
  - Library namespaces in `src/library/`: `numpy.ts`, `lax.ts`, `nn.ts`, `random.ts`,
    `scipy-special.ts`, `numpy-linalg.ts`, `numpy-fft.ts`.
- **Backends** (`src/backend/`): `cpu.ts` (debug only), `wasm.ts` + `wasm/`, `webgl.ts` + `webgl/`,
  `webgpu.ts` + `webgpu/` (ML compiler & shader codegen).
- **Aux packages**: `packages/loaders` (safetensors, OPFS cache, BPE tokenizers), `packages/onnx`
  (ONNX model → native ops), `packages/optax` (optimizers).
- **Website & demos**: `website/` — live examples that double as integration tests.

## Developer workflows

```bash
pnpm install                       # requires pnpm ≥ 10
pnpm run build                     # tsdown → dist/*.js, dist/*.d.ts
pnpm run build:watch               # watch mode
pnpm exec playwright install       # one-time: install Chromium for WebGPU tests
pnpm test                          # Vitest + Playwright (browser + node)
pnpm run test:all                  # Vitest + Deno WebGPU (auto-fallback)
pnpm run test:deno                 # Deno WebGPU tests only (headless GPU)
pnpm test test/conv.test.ts        # single file
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm -C website dev                # local dev server for demos
```

### Pre-commit CI checks

Husky runs `lint-staged` on commit, which auto-fixes ESLint and Prettier issues on staged files. The
pre-commit hook also runs the full Vitest suite and Deno WebGPU tests (`pnpm vitest run` +
`pnpm run build` + `pnpm run test:deno`).

**Before any commit**, also run these checks manually to catch issues early:

```bash
pnpm build                         # Build the project
pnpm check                         # TypeScript type checking
pnpm exec playwright install chromium-headless-shell  # (if not already installed)
pnpm test                          # Run all Vitest tests
pnpm run test:deno                 # Run Deno WebGPU tests
```

These match the checks in `.github/workflows/ci.yaml`.

### Rebase to main

When the user asks to "rebase to main", perform these steps:

1. `git fetch origin` — update remote refs
2. `git rebase origin/main` — rebase current branch onto latest main
3. If conflicts occur, resolve them (prefer keeping both sides' intent), then
   `git add <resolved files>` and `GIT_EDITOR=true git rebase --continue` (use `GIT_EDITOR=true` to
   avoid opening an interactive editor)
4. `pnpm vitest run` — verify all tests pass after rebase
5. `git push --force-with-lease` — update remote branch (safe force-push)

**Important:** Always use `GIT_EDITOR=true` when running `git rebase --continue` to prevent the
terminal from getting stuck in vim. Always use `--force-with-lease` (not `--force`) for the push.

### Temporary files

Use `tmp/` in the project root for temporary/scratch files. This directory is gitignored and allows
file operations without manual approval in VS Code.

### Debug logging

**IMPORTANT:** Do NOT use environment variables like `DEBUG=1`. Use the runtime function:

```typescript
import { setDebug } from "@jax-js/jax";
setDebug(1); // Enable debug logging BEFORE any jit compilation
```

| Level | Output                                           |
| ----- | ------------------------------------------------ |
| 0     | No debug output (default)                        |
| 1     | JIT compile logs, scan path selection            |
| 2     | Shader code (WGSL/WebAssembly), detailed tracing |
| 3     | Expressions and metadata                         |
| 4     | JIT programs, tuning details                     |
| 5     | Most verbose operation traces                    |

### WebGPU testing on headless servers

For GPU tests on a headless server, Chrome requires specific flags:

```bash
google-chrome --headless=new --use-angle=vulkan --enable-features=Vulkan \
  --disable-vulkan-surface --enable-unsafe-webgpu --disable-software-rasterizer
```

**Alternative: Deno for headless hardware WebGPU** — Deno's WebGPU (based on wgpu-rs) works headless
without X11:

```bash
curl -fsSL https://deno.land/install.sh | sh
deno eval --unstable-webgpu 'const a = await navigator.gpu.requestAdapter(); console.log(a?.info)'
pnpm run test:deno
```

### GPU device permissions (Linux)

```bash
sudo usermod -a -G render,video $USER  # then log out/in
```

## Reference-counting & ownership (critical)

Function arguments are **consumed by default** (refcount −1). Reuse an array by accessing `.ref`
(refcount +1). The `.data()` method also **consumes the array** after reading.

```ts
// BAD: consumes x twice
function foo(x) {
  return x.add(x.mul(y));
}

// GOOD: .ref keeps x alive for the second use
function foo(x) {
  return x.ref.add(x.mul(y));
}

// .data() consumes - don't dispose after!
const result = await arr.data(); // arr is now consumed
```

Canonical examples: `test/refcount.test.ts`, `test/conv.test.ts`, `test/deno/webgpu.test.ts`.

### Memory lifecycle

A **Slot** is jax-js's internal handle to a backend memory allocation (WASM pointer or GPU buffer).

1. **Array creation** — `np.array(...)` allocates a backend `Slot` with refcount = 1.
2. **`.ref` accessor** — increments the Array object's `#rc`; same underlying Slot.
3. **Function call** — passing an Array decrements `#rc` by 1 (ownership transfer).
4. **`.data()` / `.dataSync()`** — reads buffer, then calls `dispose()` internally.
5. **`.dispose()`** — decrements `#rc`; when 0, calls `backend.decRef(slot)`.
6. **Pending ops** — `PendingExecute` (batched GPU commands awaiting submission) holds refs on Slots
   until `submit()`.

### Backend memory comparison

| Aspect        | Wasm (`src/backend/wasm.ts`)              | WebGPU (`src/backend/webgpu.ts`)                      |
| ------------- | ----------------------------------------- | ----------------------------------------------------- |
| Allocation    | `WasmAllocator` over `WebAssembly.Memory` | `device.createBuffer()` with `GPUBufferUsage.STORAGE` |
| Slot tracking | `Map<Slot, {ptr, size, ref}>`             | `Map<Slot, {buffer, size, ref}>`                      |
| Sync read     | Direct memory view                        | `SyncReader` with staging buffer + `mapAsync`         |
| Dispatch      | Instantiate Wasm module, call exported fn | `commandEncoder.dispatchWorkgroups()`, queue submit   |

## WebGPU Backend Architecture

This section explains WebGPU constraints relevant to jax-js development. Assumes familiarity with
GPU concepts (buffers, shaders, workgroups) but not WebGPU-specific details.

### WebGPU compute model primer

WebGPU exposes GPU compute via **compute shaders** written in WGSL (WebGPU Shading Language). Key
concepts:

- **Workgroup**: A group of threads that execute together and can share memory. Threads within a
  workgroup can synchronize via `workgroupBarrier()`.
- **Dispatch**: Launches a grid of workgroups. Each thread gets a unique `global_invocation_id`.
- **Storage buffers**: GPU memory readable/writable by shaders. Used for inputs and outputs.
- **Uniform buffers**: Small, read-only memory for constants. Faster than storage buffers.

**Critical limitation:** There is **no global barrier** in WebGPU. Threads in different workgroups
cannot synchronize within a single dispatch. This fundamentally shapes how jax-js implements
operations.

### Hard limits and how jax-js handles them

| Limit                              | Typical Value | Impact on jax-js                                   |
| ---------------------------------- | ------------- | -------------------------------------------------- |
| `maxStorageBuffersPerShaderStage`  | 8-10          | Limits kernel inputs; excess args trigger fallback |
| `maxComputeWorkgroupsPerDimension` | 65535         | Large arrays need 2D grid splitting                |
| `maxComputeWorkgroupSizeX`         | 256           | Limits threads per workgroup (Sort workgroup size) |
| `minUniformBufferOffsetAlignment`  | 256 bytes     | Dynamic uniform offsets must be 256-byte aligned   |
| `minStorageBufferOffsetAlignment`  | 256 bytes     | Can't use buffer offsets for arbitrary strides     |

**Storage buffer limit handling:**

```typescript
// src/backend/webgpu.ts
const maxArgs = limits.maxStorageBuffersPerShaderStage - 1; // Reserve 1 for output
if (kernel.nargs > maxArgs) {
  throw new Error(`Kernel has ${kernel.nargs} args, max is ${maxArgs}`);
}
```

When a fused kernel would exceed the buffer limit, compilation fails. The frontend must split into
smaller kernels (handled by `splitGraphDataflow()` in JIT).

**Grid size handling:**

When array size exceeds 65535 workgroups, `calculateGrid()` in `codegen.ts` splits into a 2D grid:

```typescript
// If size > 65535, split into 2D grid: (65535, ceil(size/65535))
const gridX = Math.min(size, 65535);
const gridY = Math.ceil(size / 65535);
```

Shader code uses `global_invocation_id.x + global_invocation_id.y * 65535u` to reconstruct the
linear index.

**Storage buffer offset alignment:**

The 256-byte `minStorageBufferOffsetAlignment` means you can't bind a buffer at arbitrary offsets.
For scan with strides like 48 bytes (a 4×3 f32 matrix), buffer offsets fail. jax-js solves this with
**uniform-based offsets**: bind the entire buffer, pass the element offset as a uniform variable.
The scan-v2 preencoded-routine path will use this technique (see Part 2).

### Features exploited

| Feature                     | How jax-js uses it                                               | Location                           |
| --------------------------- | ---------------------------------------------------------------- | ---------------------------------- |
| **shader-f16**              | Float16 dtype support; requested at device creation              | `src/backend.ts` feature requests  |
| **Workgroup shared memory** | Sort uses `var<workgroup>` for bitonic sort local exchanges      | `src/backend/webgpu/routines.ts`   |
| **workgroupBarrier()**      | Synchronizes threads within Sort workgroups                      | `bitonicSortShader` in routines.ts |
| **storageBarrier()**        | Memory fence for shared variable consistency                     | `bitonicSortShader` in routines.ts |
| **Pipeline caching**        | Compiled pipelines stored by shader hash                         | `pipelineCache` in webgpu.ts       |
| **Command batching**        | Multiple dispatches encoded before single queue.submit()         | `PendingExecute` in webgpu.ts      |
| **WGSL copy shader**        | Byte-level buffer copy when `copyBufferToBuffer` alignment fails | `COPY_SHADER_CODE` in webgpu.ts    |

**Scan will additionally use** (implemented in v2):

| Feature               | How scan uses it                                        | v1 reference location                   |
| --------------------- | ------------------------------------------------------- | --------------------------------------- |
| **Ping-pong buffers** | Carry state alternates between two buffers across iters | `dispatchPreencodedScan()` in webgpu.ts |
| **Uniform buffers**   | Per-iteration offsets for preencoded routine scan       | `scan-wrapper.ts`                       |

**Pipeline caching detail:**

```typescript
// Pipelines cached by shader source hash
const cacheKey = hashShaderSource(shaderCode);
if (pipelineCache.has(cacheKey)) {
  return pipelineCache.get(cacheKey);
}
const pipeline = device.createComputePipeline({ ... });
pipelineCache.set(cacheKey, pipeline);
```

This avoids recompiling identical shaders (common with JIT-generated kernels).

**Synchronous readback trick:**

WebGPU normally requires async `buffer.mapAsync()` for reading GPU data. jax-js implements
`SyncReader` (`src/backend/webgpu/reader.ts`) using an offscreen canvas with webgpu context —
borrowed from TensorFlow.js. This enables `.dataSync()` for debugging, though `.data()` (async) is
preferred for performance.

### Features NOT exploited (opportunities)

| Feature                   | What it enables                                     | Why not used yet                         |
| ------------------------- | --------------------------------------------------- | ---------------------------------------- |
| **Subgroups**             | SIMD-width operations (shuffle, reduce within wave) | Requires `subgroups` feature; not stable |
| **Indirect dispatch**     | GPU-driven workgroup counts                         | No dynamic control flow needs it yet     |
| **Texture sampling**      | Hardware-accelerated interpolation                  | All ops use storage buffers currently    |
| **Tiled matrix multiply** | Shared memory blocking for large matmuls            | Matmul uses simple row×col accumulation  |
| **Atomic operations**     | Lock-free reductions, histograms                    | Reductions done via shader accumulation  |
| **timestamp-query**       | GPU-side profiling                                  | Requested but not wired up for profiling |
| **Render pipelines**      | Visualization without readback                      | Would need separate rendering path       |

**Subgroups opportunity:**

Subgroups enable operations like `subgroupAdd()` that sum across a SIMD lane (typically 32-64
threads) without explicit barriers. This would accelerate reductions significantly:

```wgsl
// Current (sequential accumulation):
var acc = 0.0;
for (var i = 0u; i < size; i++) { acc += data[i]; }

// With subgroups (parallel within wave):
let partial = subgroupAdd(data[local_id]);
if (subgroup_invocation_id == 0) { atomicAdd(&result, partial); }
```

**Tiled matmul opportunity:**

Current matmul computes `C[i,j] = sum(A[i,:] * B[:,j])` with each thread doing a full dot product.
Tiled matmul loads tiles of A and B into shared memory, enabling data reuse:

```wgsl
// Tiled approach (not implemented):
var<workgroup> tileA: array<f32, 16*16>;
var<workgroup> tileB: array<f32, 16*16>;
// Load tiles collaboratively, compute partial products, accumulate
```

This is a standard GPU optimization that could provide 5-10× speedup for large matrices.

### WebGPU-specific scan constraints

The "no global barrier" limitation creates scan-specific constraints documented in Part 2:

| Constraint                        | Why it exists                                                | Consequence                   |
| --------------------------------- | ------------------------------------------------------------ | ----------------------------- |
| Per-element independence required | No cross-workgroup sync between iterations                   | Complex bodies → JS fallback  |
| numCarry ≠ numY unsupported       | compiled-loop shader assumes 1:1 carry↔output mapping       | Falls back to JS loop         |
| Internal buffer deps unsupported  | Shader can't allocate scratch temporaries between statements | Mandelbrot pattern → fallback |
| Sort in scan uses fallback        | Sort shader already uses uniforms (offset conflict)          | Uniform buffer contention     |

WASM backend handles all these cases because it can allocate temporaries and has true sequential
control flow. WebGPU is more restricted but faster when patterns fit.

### Key WebGPU files

| File                                 | Purpose                                            |
| ------------------------------------ | -------------------------------------------------- |
| `src/backend.ts`                     | WebGPU init, adapter/device creation, feature reqs |
| `src/backend/webgpu.ts`              | Main backend: kernels, scan, command encoding      |
| `src/backend/webgpu/codegen.ts`      | `calculateGrid()`, WGSL helpers, `ShaderInfo`      |
| `src/backend/webgpu/routines.ts`     | Bitonic sort, Cholesky, LU, TriangularSolve WGSL   |
| `src/backend/webgpu/scan-wrapper.ts` | Transforms routine shaders for scan with offsets   |
| `src/backend/webgpu/reader.ts`       | `SyncReader` for synchronous buffer readback       |
| `src/backend/webgpu/builtins.ts`     | Shader snippets for special functions (erf, etc.)  |

### Autodiff and ownership

> ⚠️ **Critical difference from Python JAX:** Letting `vjpFn` go out of scope will **NOT** free GPU
> memory. You **MUST** call `.dispose()` explicitly.

```ts
const [y, vjpFn] = vjp(f, [x]);
const dx = vjpFn(dy);
vjpFn.dispose(); // free captured forward-pass intermediates

const jitF = jit((x) => np.multiply(x, np.array([2])));
const result = jitF(x);
jitF.dispose(); // free captured constants
```

## Codegen architecture

Expression translation and shader generation share common code between regular kernels and scan.
Understanding this structure is essential for adding scan codegen paths.

**WASM Backend:**

| Function                        | Role                                                |
| ------------------------------- | --------------------------------------------------- |
| `translateExpCore()`            | Shared core handling all `AluOp` cases              |
| `TranslateExpContext` interface | Callbacks for `getVariable` and `handleGlobalIndex` |
| `translateExp()`                | Wrapper with bounds-check GlobalIndex               |
| `codegenWasmKernel()`           | Entry point, dispatches based on `isMultiOutput`    |
| `codegenWasmSinglePath()`       | Single-output kernel (supports reduction)           |
| `codegenWasmMultiPath()`        | Multi-output kernel (no reduction)                  |

Scan v2 will add `translateExpWithGeneralScanContext()` (const/carry/xs/internal classification) and
`codegenNativeScanGeneral()` (full scan loop codegen) — these are ported from feat/scan.

**WebGPU Backend:**

| Function                    | Role                                                    |
| --------------------------- | ------------------------------------------------------- |
| `translateAluOpToWgsl()`    | Binary/unary ops, comparisons, casts, ternary           |
| `translateErfToWgsl()`      | Erf/Erfc with f32 precision wrapper                     |
| `gen()` in `pipelineSource` | CSE (common subexpression elimination) + special cases  |
| `createShaderEmitter()`     | Returns `{emit, pushIndent, popIndent, getCode}` helper |

Scan v2 will add `genScanExpressionWithRidx` (scan-specific GlobalIndex + inline generation) and
`nativeScanMultiShaderSource()` (full scan shader) — ported from feat/scan.

**Backend Interface:**

The `Backend` interface uses unified methods for single and multi-output kernels:

- `prepareKernel()` / `prepareKernelSync()` — each backend checks `kernel.isMultiOutput` internally
- WebGPU expands multi-output kernels into separate shader dispatches
- WebGL throws on multi-output (not supported)

## Routine system

Routines are backend-specific operations (sort, cholesky, etc.) that can't be expressed as fused
elementwise kernels. They exist in three implementations:

| Backend    | Implementation          | Location                         | Algorithm Style            |
| ---------- | ----------------------- | -------------------------------- | -------------------------- |
| **CPU**    | JavaScript (TypedArray) | `src/routine.ts`                 | Sequential (for debugging) |
| **WASM**   | wasmblr (runtime gen)   | `src/backend/wasm/routines/*.ts` | Sequential (optimized)     |
| **WebGPU** | Hand-written WGSL       | `src/backend/webgpu/routines.ts` | Parallel (GPU-optimized)   |

1. **CPU backend assumes WASM unavailable** — exists for environments without WebAssembly
2. **WebGPU uses different algorithms** — GPU parallelism requires fundamentally different
   approaches:
   - Sort: Bitonic sort (parallel) vs merge sort (sequential)
   - Cholesky: Column-parallel Cholesky-Crout vs row-by-row Cholesky-Banachiewicz

### wasmblr

**Problem:** Hand-writing WASM bytecode is error-prone and unmaintainable.

**Solution:** wasmblr — a custom WASM bytecode assembler with a high-level helper layer (WasmHl).

**Benefits:**

- Runtime JIT compilation (no separate build step, no pre-compiled binaries)
- Single TypeScript syntax throughout the codebase
- Ergonomic helpers for control flow (`forLoop`, `whileLoop`, `ifElse`) and memory access
- SIMD-ready (v128, i32x4, f32x4 types available)
- Small output (~1KB per routine)
- **Size specialization**: Matrix dimensions baked at compile time enable loop unrolling and
  constant propagation
- **LRU caching**: 64-entry cache amortizes compilation cost across calls

**Key WasmHl helpers:**

- `forLoop(i, start, end, body)` — for loop with expression start/end
- `forLoopDown(i, start, end, body)` — downward for loop
- `forLoopUnrolled(n, body, threshold?)` — fully unrolls small fixed-size loops (default ≤8)
- `whileLoop(cond, body)` — while loop with condition callback
- `ifElse(resultType, then, else?)` — conditional with optional else
- `load(dtype, base, indexExpr)` — load from base + index × elemSize
- `store(dtype, base, indexExpr, valueExpr)` — store to memory
- `index2D(row, cols, col)` — compute row × cols + col
- `binOp(dtype, op)` — binary operation (add, sub, mul, div)
- `const(dtype, value)` — push constant onto stack

**SIMD helpers (f32x4/f64x2):**

- `loadF32x4(base, indexExpr)` — load 4 floats as v128
- `storeF32x4(base, indexExpr, valueExpr)` — store v128 as 4 floats
- `f32x4Hsum()` — horizontal sum v128 → f32
- `f64x2Hsum()` — horizontal sum v128 → f64
- `simdReductionF32(acc, k, end, rowABase, rowBBase, op)` — SIMD dot product with automatic tail
  handling
- `simdReductionF64(acc, k, end, rowABase, rowBBase, op)` — same for f64x2

**SIMD speedup by matrix size:**

| Matrix Size | f32x4 Speedup | f64x2 Speedup |
| ----------- | ------------- | ------------- |
| n < 32      | ~0.8x (skip)  | ~0.9x (skip)  |
| n = 32      | ~1.1x         | ~1.0x         |
| n = 64      | ~1.7x         | ~1.3x         |
| n = 128     | ~3.0x         | ~1.8x         |
| n = 256     | ~3.8x         | ~1.9x         |

SIMD is automatically selected for Cholesky when `dtype === "f32" && n >= 32`.

**Calling routines from scan loops:** Scan modules use WASM imports to call routines from separate
wasmblr modules. This avoids code duplication (each routine is 1-3KB) while keeping the entire loop
in native code. See `codegenNativeScanGeneral()` in `src/backend/wasm.ts` on feat/scan.

### Autodiff of routines

Routines remain **opaque primitives** — the Jaxpr just contains `cholesky a`. The internal algorithm
is NOT traced.

The JVP rule defines the derivative **in terms of other primitives**:

```typescript
[Primitive.Cholesky]([a], [da]) {
  const L = cholesky(a.ref);
  da = da.ref.add(mT(da)).mul(0.5);  // Symmetrize
  const W = triangularSolve(L.ref, da, { lower: true });
  const ST = triangularSolve(L.ref, mT(W), { lower: true });
  const dL = batchMatmulT(L.ref, triu(ST, 1).add(triu(ST)).mul(0.5));
  return [[L], [dL]];
}
```

The gradient is computed by:

1. **JVP tracing** → produces a Jaxpr containing `cholesky`, `triangular_solve`, matmul, etc.
2. **Transpose** → walks the JVP Jaxpr and transposes each primitive

The result (`grad(sum(cholesky))`) produces a **fully expanded Jaxpr** with ~30 operations. The
derivative of `cholesky` requires `triangular_solve` — both are Routines that dispatch to native
WASM.

### Adding a new routine (checklist)

| Step | File                                   | What to add                                                          |
| ---- | -------------------------------------- | -------------------------------------------------------------------- |
| 1    | `src/backend/wasm/routines/<name>.ts`  | Size-specialized wasmblr implementation (sizes as compile-time args) |
| 2    | `src/backend/wasm/routines/index.ts`   | Export the build function                                            |
| 3    | `src/backend/wasm/routine-provider.ts` | Add builder to `routineBuilders` map with size key generation        |
| 4    | `src/routine.ts`                       | Add to `Routines` enum                                               |
| 5    | `src/frontend/core.ts`                 | Add to `routinePrimitives` map                                       |
| 6    | `src/backend/wasm.ts`                  | Add dispatch case with size params                                   |
| opt  | `src/routine.ts`                       | Add CPU fallback in `runCpuRoutine()`                                |
| opt  | `src/frontend/jvp.ts`                  | Add JVP rule if autodiff needed                                      |
| opt  | `src/frontend/linearize.ts`            | Add transpose rule if grad needed                                    |

**Size key convention:** Cache keys include dtype and all size dimensions, e.g., `cholesky_f32_4` or
`triangular_solve_f64_8_16_lower_unit`.

### WASM feature opportunities (assessed Feb 2026)

| Priority | Feature            | Browser risk       | Impact      | Notes                                                                                |
| -------- | ------------------ | ------------------ | ----------- | ------------------------------------------------------------------------------------ |
| Medium   | i64 in wasmblr     | None (MVP)         | Medium-High | Unlocks proper f64 builtins (exp/log/sin/erf) and simplifies Threefry PRNG           |
| Medium   | Relaxed SIMD (FMA) | Safari unsupported | High        | `f32x4.relaxed_madd` for 2× dot-product throughput; needs runtime feature detection  |
| Low      | Threads / atomics  | Needs COOP/COEP    | Very High   | SharedArrayBuffer + Workers for parallel matmul/routines; major architectural change |
| Low      | Sign extension ops | None               | Low         | `i32.extend8_s` etc.; marginal for float-focused workloads                           |

## Deno WebGPU test guidelines

**Critical: Avoid creating multiple `GPUDevice` instances**

- **Always reuse jax-js's WebGPU device** instead of calling `navigator.gpu.requestAdapter()` +
  `adapter.requestDevice()`.
- Creating a second `GPUDevice` can destabilize Deno's WebGPU runtime and cause flakiness, memory
  leaks, or segfaults across test files.
- Use the `getJaxJsWebGPUDevice()` helper pattern to access the backend's device.
- **Never call `device.destroy()`** on the shared backend device — let the backend manage its
  lifecycle.

**Import from `dist/` not `src/`**

- Deno tests MUST import from `../../dist/index.js` to share backend instances across test modules.
- Mixed `src/` vs `dist/` imports create separate module graphs with separate backend instances,
  causing leak detection failures.

**Buffer cleanup**

- Track all created `GPUBuffer`s in an array: `const createdBuffers: GPUBuffer[] = []`.
- Destroy them in `finally` blocks: `for (const b of createdBuffers) b.destroy()`.
- Call `await device.queue.onSubmittedWorkDone()` before destroying buffers.

**Memory leak detection:**

```ts
import { withLeakCheck, getSlotCount, assertNoLeaks } from "./harness.ts";

Deno.test({
  name: "my test",
  fn: withLeakCheck(async () => {
    const result = await someComputation();
    await result.data();
    jitF.dispose();
  }),
});
```

## Exports & public API

All public symbols must be exported from `src/index.ts`. Key exports:

- Transformations: `jit`, `grad`, `valueAndGrad`, `jvp`, `vjp`, `vmap`, `jacfwd`, `jacrev`,
  `hessian`, `linearize`, `makeJaxpr`
- Device control: `init`, `defaultDevice`, `devicePut`, `blockUntilReady`, `devices`, `getBackend`
- Namespaces: `numpy`, `lax`, `nn`, `random`, `scipySpecial`, `tree`
- Testing utilities: `setScanBodyStepsCallback`, `ScanPath` (type)

## Extending the codebase

| Area             | Key files                                          | Notes                                                          |
| ---------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| New numpy/lax op | `src/library/{numpy,lax}.ts`                       | Follow existing function signatures; add to exports if public. |
| Backend kernel   | `src/backend/webgpu/builtins.ts`, shader templates | Mirror existing patterns; test on Chromium via Playwright.     |
| Loader/tokenizer | `packages/loaders/src/`                            | See `safetensors.ts`, `tokenizers.ts`.                         |
| ONNX op          | `packages/onnx/src/ops.ts`                         | Implement lowering; wire in `index.ts`.                        |

## JIT compiler internals

The JIT system lives in `src/frontend/jit.ts` and `src/frontend/jaxpr.ts`.

**Pipeline:**

1. **Tracing** – `makeJaxpr(f)` traces a function to produce a `Jaxpr` (intermediate representation
   in A-Normal Form, where every subexpression is named)
2. **Simplification** – `jaxpr.flatten().simplify()` canonicalizes the graph
3. **Graph splitting** – `splitGraphDataflow()` identifies "black nodes" (operations that can't be
   fused, like reductions, routines, or `DynamicUpdateSlice`) vs fusable elementwise ops
4. **Kernel fusion** – Consecutive elementwise ops merge into a single `Kernel` (multi-output if
   needed)
5. **Compilation** – `jitCompile(backend, jaxpr)` emits a `JitProgram` (list of `JitStep`s)
6. **Execution** – `JitProgram.execute(slots)` runs steps, managing memory lifetime

**Multi-output kernel fusion:**

When multiple non-fusable outputs have the same size and no reductions, they are batched into a
multi-output `Kernel`. This reduces kernel dispatch overhead for functions with multiple outputs.

- Inputs are unioned across all outputs, and expressions are reindexed accordingly
- Example: `(a + b, a - b, a * b)` becomes one Kernel with 3 outputs and 2 inputs
- Batching triggers at flush points: reductions, routines, size changes, or end of compilation

**Key types:**

| Type                              | File         | Role                                               |
| --------------------------------- | ------------ | -------------------------------------------------- |
| `Jaxpr`, `JaxprEqn`, `Var`, `Lit` | `jaxpr.ts`   | IR nodes and bindings                              |
| `JitProgram`, `JitStep`           | `jit.ts`     | Compiled program + step types                      |
| `Kernel`                          | `alu.ts`     | Fused kernel (1..N outputs), `KernelOutput[]`      |
| `KernelOutput`                    | `alu.ts`     | `{ size, exp, reduction? }` for each kernel output |
| `Routine`                         | `routine.ts` | Backend-specific op (sort, cholesky, etc.)         |

**JitStep types:**

| Type                 | Purpose                                                             |
| -------------------- | ------------------------------------------------------------------- |
| `execute`            | Dispatch a `Kernel` or `Routine` with inputs→outputs                |
| `copy`               | `DynamicUpdateSlice`: clone dst buffer, patch src region(s) into it |
| `malloc`             | Allocate a buffer                                                   |
| `incref`             | Increment refcount on a slot                                        |
| `free`               | Decrement refcount on a slot                                        |
| `scan`               | JS fallback scan loop                                               |
| `compiled-loop`      | Native WASM/WebGPU scan                                             |
| `preencoded-routine` | Pre-encoded WebGPU routine scan                                     |

**Kernel class (unified single/multi-output):**

The `Kernel` class uses an `outputs: KernelOutput[]` array to support 1..N outputs:

- `Kernel.single(nargs, size, exp, reduction?)` — single-output kernel
- `Kernel.multi(nargs, outputs[])` — multi-output kernel
- `kernel.isMultiOutput` — true if `outputs.length > 1`
- `kernel.dtypeAt(i)` — dtype of output `i`

**Codegen simplifications:**

- Simplify index math when emitting WGSL/WASM (e.g., elide stride=1 multiplies).
- Drop additions by 0 for offsets and avoid redundant casts in view maps.
- Apply the same simplifications in loop index maps and scan offset logic.

**Adding a new primitive:**

1. Declare in `Primitive` enum (`src/frontend/core.ts`)
2. Add tracing rule in `implRules` / `jvpRules` / `transposeRules`
3. If fusable elementwise, add ALU lowering in `jit.ts`
4. If needs dedicated kernel, register in `routinePrimitives` and implement in `src/backend/*`
5. If copy-like (e.g., `DynamicUpdateSlice`), handle as a special case in `jitCompile()` and add to
   `splitGraphDataflow()` black node classification

## Common pitfalls

- Forgetting `.ref` → double-consume → `ReferenceError` in tests
- Exporting a symbol from library but not `src/index.ts` → missing from published types
- Changing WebGPU shaders without browser tests → silent breakage
- **CPU backend GlobalView detection**: Collect both `AluOp.GlobalIndex` AND `AluOp.GlobalView`
  (internal ALU expression types) when finding used input buffers
- **JIT pending ops before scan**: Flush pending ops before scan step execution

## Known flaky tests

- **LU JVP finite-differences** (`test/lax-linalg.test.ts`): Occasionally fails with precision
  errors at the edge of f32 machine epsilon. Not a bug — inherent to finite-difference verification.
- **Deno WebGPU tests** (`test/deno/`): When running all Deno test files together in a single
  `deno test` invocation, GPU state pollution between files causes memory leak detection failures.
  The `test:deno` script runs each file as a separate `deno test` command (chained with `&&`).

> ⚠️ **IMPORTANT: Deno WebGPU test isolation** - Due to Deno's module caching and GPU state
> persistence between test files, running all Deno tests together in a single process causes
> spurious memory leak failures. The `test:deno` script chains separate `deno test` commands for
> each file to ensure proper isolation:
>
> ```bash
> pnpm run test:deno  # Runs each file separately (RECOMMENDED)
> ```
>
> Do NOT run `deno test test/deno/` directly - use the script instead. All test files use
> `withLeakCheck` from harness.ts for memory leak detection.

## Commit checklist

**Before every commit**, AI agents MUST:

1. Run pre-commit CI checks (see above)
2. Ensure the **pre-commit hook** is installed (run `pnpm prepare` if needed). The repository will
   run linting and the _full test suite_ automatically when you commit.
3. Run the _full test suite_ locally (`pnpm vitest run`) after finishing code changes to verify
   there are no regressions.
4. Update documentation when adding new features or APIs
5. Add/adjust tests exercising `.ref` and `.dispose()` for new behavior — add focused unit tests for
   any bugfixes or edge cases
6. Export new public symbols from `src/index.ts`
7. Update `FEATURES.md` for user-visible changes

## Documentation files

| File                              | Purpose                                    | When to update                 |
| --------------------------------- | ------------------------------------------ | ------------------------------ |
| `README.md`                       | Main project intro, tutorial               | Major features, API changes    |
| `FEATURES.md`                     | JAX/NumPy API compatibility table          | New supported functions        |
| `.github/copilot-instructions.md` | AI agent onboarding, scan feature tracking | New patterns, scan development |
| `packages/*/README.md`            | Package-specific docs                      | Package feature changes        |

## Where to start reading

- Entry & exports: `src/index.ts`
- Memory model: `test/refcount.test.ts`
- Backends: `src/backend/webgpu/`, `src/backend/wasm/`
- Demos: `website/src/routes/repl/`, `website/src/routes/mobileclip/`
- Deno WebGPU tests: `test/deno/webgpu.test.ts` — headless hardware GPU testing
- Scan tests: `test/lax-scan.test.ts` — comprehensive scan suite (~1700 lines)

---

# Part 2: Scan Feature Reference

`lax.scan` implements sequential loops with carry state — the JAX equivalent of a for-loop that
threads state through each iteration. It supports JIT compilation, automatic differentiation
(`grad`, `jvp`, `vjp`), `vmap`, and native compilation on WASM and WebGPU backends.

## Architecture overview

```
lax.scan(f, init, xs)
  │
  ▼
Primitive.Scan → makeJaxpr(body) → bodyJaxpr
  │
  ▼ (inside jitCompile)
planScan(backend, bodyProgram, bodyJaxpr, ...) → ScanPlan
  │
  ├─ { path: "compiled-loop" }       ← WASM compiled module or WebGPU multi-kernel shader
  ├─ { path: "preencoded-routine" }  ← WebGPU uniform-offset routine scan
  └─ { path: "fallback" }            ← JS loop calling bodyProgram.execute() per iteration
  │
  ▼
JitStep { type: "scan", plan: ScanPlan, ... }
  │
  ▼ (inside JitProgram.execute)
executeScan(backend, step)
  ├─ flush pending ops on all inputs (ONE policy)
  ├─ preallocate Y stacked buffers if direct-write eligible
  ├─ dispatch based on plan.path
  ├─ manage carry lifecycle, shared-slot guards, duplicate-slot incRef
  └─ return carry + stacked ys
```

The planner produces a `ScanPlan` data structure; the executor interprets it. This gives one
execution path for all backends, one ownership policy, and one flush discipline.

## Key files

| File                                 | Purpose                                                   |
| ------------------------------------ | --------------------------------------------------------- |
| `src/library/lax-scan.ts`            | Public `lax.scan()` API, tree handling, tracing           |
| `src/frontend/scan-plan.ts`          | `ScanPlan` type, `planScan()`, path selection heuristics  |
| `src/frontend/scan-executor.ts`      | `executeScan()` — unified scan loop for all backends      |
| `src/frontend/core.ts`               | `Primitive.Scan`, `DynamicUpdateSlice`                    |
| `src/frontend/jaxpr.ts`              | Abstract eval rules for Scan and DUS                      |
| `src/frontend/jit.ts`                | Scan JitStep, `Primitive.Scan` case in `jitCompile()`     |
| `src/frontend/array.ts`              | Eager `Primitive.Scan` impl, `copySliceToBuffer`          |
| `src/frontend/jvp.ts`                | Scan JVP rule (wrapper jaxpr for input/output reordering) |
| `src/frontend/linearize.ts`          | Scan transpose rule (√N checkpointing), partial eval      |
| `src/frontend/vmap.ts`               | Scan vmap rule (batch axis management)                    |
| `src/backend/wasm.ts`                | WASM compiled-loop scan codegen                           |
| `src/backend/webgpu.ts`              | WebGPU multi-kernel + preencoded-routine scan             |
| `src/backend/webgpu/scan-wrapper.ts` | WGSL shader transformer for preencoded scan               |
| `test/lax-scan.test.ts`              | Comprehensive scan test suite (~1700 lines)               |
| `test/deno/webgpu.test.ts`           | Deno WebGPU scan tests (cumsum, matmul)                   |

## Execution paths

| Path                   | Backend | When chosen                                             | How it works                                               |
| ---------------------- | ------- | ------------------------------------------------------- | ---------------------------------------------------------- |
| **compiled-loop**      | WASM    | Elementwise body, no routines, numCarry == numY         | Generates a single WebAssembly module for the full loop    |
| **compiled-loop**      | WebGPU  | Elementwise body, no routines, numCarry == numY         | Generates WGSL shader dispatched once per iteration        |
| **preencoded-routine** | WebGPU  | Body contains routines (sort, cholesky, matmul)         | Pre-encodes routine commands, replays with uniform offsets |
| **fallback**           | All     | Complex bodies, internal buffer deps, unsupported cases | JS loop calling `bodyProgram.execute()` per iteration      |

## Autodiff

- **JVP**: Creates a wrapper jaxpr that reorders inputs from scan layout (interleaved
  primal/tangent) to JVP layout (all primals then all tangents). Runs a doubled scan with
  `numCarry*2`.
- **Transpose (VJP/grad)**: √N checkpointing by default (Griewank–Walther). Forward pass stores
  `ceil(√N)` checkpoints, backward pass recomputes intermediate carries per segment. Supports
  `checkpoint: false` (O(N) memory) and `checkpoint: number` (custom segment size).
- **Vmap**: Moves batch to axis 0 for carry/consts, axis 1 for xs (after scan length axis), vmaps
  the body, then moves ys batch back to axis 0.

## Ownership rules

1. **Flush before body**: Before each iteration, pending GPU commands are submitted.
2. **Carry lifecycle**: Carry slots are owned by the loop. Non-aliased old carry slots are freed
   after each iteration.
3. **Y stacking**: Direct-write (preferred) preallocates full Y output buffers, copies each
   iteration's y into the correct slice. Collect-and-stack as fallback.
4. **Shared-slot protection**: When carry_out[i] aliases y_out[j], incRef before stacking.
5. **xs slicing**: Iteration `i` reads at offset `i * stride` (or `(length-1-i) * stride` for
   reverse).

## WebGPU scan constraints

The "no global barrier" limitation creates scan-specific constraints:

| Constraint                        | Why it exists                                                | Consequence                   |
| --------------------------------- | ------------------------------------------------------------ | ----------------------------- |
| Per-element independence required | No cross-workgroup sync between iterations                   | Complex bodies → JS fallback  |
| numCarry ≠ numY unsupported       | compiled-loop shader assumes 1:1 carry↔output mapping       | Falls back to JS loop         |
| Internal buffer deps unsupported  | Shader can't allocate scratch temporaries between statements | Mandelbrot pattern → fallback |
| Sort in scan uses preencoded      | Sort shader already uses workgroup-level sync                | Requires preencoded path      |

WASM backend handles all these cases because it can allocate temporaries and has true sequential
control flow. WebGPU is more restricted but faster when patterns fit.

### Kalman / tree-carry path analysis (v1 = v2, no regression)

Kalman-like patterns have **tree-structured carry** (e.g., `{state, covDiag}`) which flattens to
`numCarry=2, numY=2`. The body uses `np.matmul` which lowers to `Kernel` (Mul→Reduce), **not**
`Routine`, creating deep **internal buffer dependencies** (each matmul output feeds the next).

| Backend | Path          | Why                                                                        |
| ------- | ------------- | -------------------------------------------------------------------------- |
| WASM    | compiled-loop | Internal deps + multi-carry fully supported via `codegenNativeScanGeneral` |
| WebGPU  | fallback      | Internal deps rejected (no global barrier between dispatches)              |

This is **identical between v1 and v2** — v1 also rejected Kalman on WebGPU at the same
`hasInternalDeps` check. The WASM compiled-loop handles it in both versions via
`translateExpWithGeneralScanContext` which resolves internal buffer references at codegen time, with
separate carry buffer pointers per tree leaf (`carrySizes[]`, `carryOutSources[]`).

v2's WebGPU path adds an explicit Y≠carry slot match check (`yOutIds[i] !== carryOutIds[i]`) that v1
enforced only implicitly in the shader codegen, but internal deps alone already reject Kalman before
that check is reached.

**Deno limitation:** WebGPU fallback uses `readSync` → `SyncReader` → `OffscreenCanvas`, which is
unavailable in Deno. Kalman WebGPU benchmarks can only run via Vitest (headful Chromium).

## How to access v1 (feat/scan) source

The feat/scan branch contains the v1 implementation. **Never switch to it** — use `git show`:

```bash
git show feat/scan:src/library/lax-scan.ts
git show feat/scan:src/frontend/scan-plan.ts
git show feat/scan:src/backend/wasm.ts
git show feat/scan:src/backend/webgpu.ts
git diff main..feat/scan -- src/library/lax-scan.ts
```

## Scan v2 review: gaps vs feat/scan (v1)

v2 is a clean architectural rewrite of v1 with a unified `ScanPlan` + `executeScan()` replacing v1's
three separate JitStep types (`"scan"`, `"compiled-loop"`, `"preencoded-routine"`) and `ScanRunner`
callback pattern. The autodiff stack (JVP, √N checkpointing, vmap) is **identical** between v1 and
v2. All three execution paths (compiled-loop, preencoded-routine, fallback) are implemented and
working for elementwise-only scan bodies.

### Feature gaps to port from v1

| Gap                           | Severity | What's missing                                                                                                                                                                                                                                                                                                                                         | v1 reference                                                                                                |
| ----------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| WASM routine imports in scan  | Critical | `routine-provider.ts` (261 lines), `routines/` dir (7 files: cholesky, cholesky-simd, triangular-solve, lu, sort, argsort, index), `#getRoutineModuleForScan()`, `#getRoutineInstanceForModule()`, `ScanRoutineInfo` type, `auxBufferSize` in scan-plan. Without this, scan bodies with routines on WASM fall to JS fallback instead of compiled WASM. | `feat/scan:src/backend/wasm/routine-provider.ts`, `feat/scan:src/backend/wasm/routines/`                    |
| Multi-output kernel           | Major    | `KernelOutput` interface, `Kernel.multi()`, `isMultiOutput`, `dtypeAt(i)`, `codegenWasmMultiPath`. Bodies producing multiple outputs can't fuse into one kernel dispatch.                                                                                                                                                                              | `feat/scan:src/alu.ts` (KernelOutput, Kernel class), `feat/scan:src/backend/wasm.ts` (codegenWasmMultiPath) |
| WASM instance caching         | Moderate | `#instanceCache: WeakMap<WebAssembly.Module, WebAssembly.Instance>` avoids re-instantiating WASM modules on repeated dispatch. v2 creates fresh instances every time.                                                                                                                                                                                  | `feat/scan:src/backend/wasm.ts` L171, L314-337, L606-628                                                    |
| WASM routine dispatch         | Moderate | Dedicated `#dispatchCholesky/Sort/LU/TriangularSolve/Argsort` methods using size-specialized wasmblr modules. v2 falls back to `runCpuRoutine` for all routines.                                                                                                                                                                                       | `feat/scan:src/backend/wasm.ts` L387-545                                                                    |
| Error handling in plan        | Minor    | `prepareNativeScanGeneral` and `prepareNativeScanMulti` should wrap in try/catch and return null (graceful fallback) instead of throwing.                                                                                                                                                                                                              | `feat/scan:src/frontend/scan-plan.ts`                                                                       |
| `slotCount()` on WASM backend | Minor    | Memory leak detection helper.                                                                                                                                                                                                                                                                                                                          | `feat/scan:src/backend/wasm.ts` L195                                                                        |

### Missing tests (v1 had ~30 more)

| Category                        | Count | Description                                               |
| ------------------------------- | ----- | --------------------------------------------------------- |
| WASM routine scan               | 7     | Cholesky/sort/LU/triangularSolve/argsort in compiled-loop |
| Known-limitations path tests    | 7     | Documents which path each pattern takes per backend       |
| Advanced vmap/grad compositions | 6     | `grad(vmap)`, `vmap(grad)`, `vmap(jit)`, equivalence      |
| Extra JVP/checkpoint            | 4     | Different tangent values, `checkpoint: 1`                 |
| WebGL backend                   | 2     | WebGL fallback path                                       |
| Routine grad flow               | 1     | `grad` through mixed kernel+routine body                  |
| makeJaxpr tracing               | 1     | Scan JVP trace                                            |

### v2 improvements over v1

- Unified `ScanPlan` discriminated union + `executeScan()` in dedicated `scan-executor.ts`
- Single `type: "scan"` JitStep instead of three step types
- Better typing (`WebGPUBackend` / `WasmBackend` casts instead of `as any`)
- Explicit Y-carry slot match validation in WebGPU path
- Bool dtype `(access != 0)` handling in WebGPU scan expressions
- Buffer min-size guard `Math.max(size, 4)` for preencoded ping-pong buffers
- New DLM pattern tests (4) and scan preallocate tests (5)

### Porting priority

1. **WASM routine infrastructure** — port `routine-provider.ts`, `routines/` dir, routine import
   linking in `dispatchNativeScanGeneral`, enable routine steps in `tryPrepareWasmNativeScan`
2. **WASM instance caching** — add `#instanceCache` WeakMap for dispatch and scan dispatch
3. **Multi-output kernel** — port `KernelOutput`, `Kernel.multi()`, `codegenWasmMultiPath`
4. **Error handling** — wrap plan preparation in try/catch for graceful fallback
5. **Missing tests** — port path-documentation tests, advanced transform compositions

### Benchmark results (WASM, Chromium headless, Feb 2026)

All benchmarks use the `compiled-loop` path (planner auto-selects). Run with
`pnpm vitest bench bench/scan.bench.ts`.

| Pattern      | N   | Size | ops/sec |
| ------------ | --- | ---- | ------- |
| cumsum       | 100 | 64   | ~84K    |
| cumsum-large | 500 | 256  | ~12K    |
| carry-only   | 200 | 32   | ~123K   |
| reduction    | 100 | 64   | ~74K    |
| reverse      | 200 | 64   | ~70K    |
| kalman       | 200 | 4    | ~70K    |

### Benchmark results (WebGPU, Deno wgpu-rs, Feb 2026)

Hardware GPU benchmarks via Deno's native WebGPU. Uses `compiled-loop` path. Run with
`pnpm run bench:deno` (requires `pnpm build` first).

| Pattern      | N   | Size | iter/s |
| ------------ | --- | ---- | ------ |
| cumsum       | 100 | 64   | ~10K   |
| cumsum-large | 500 | 256  | ~7.6K  |
| reverse      | 200 | 64   | ~7.7K  |

Note: Kalman (tree-structured carry) requires fallback path which uses `readSync`/`OffscreenCanvas`,
unavailable in Deno. Run Kalman WebGPU benchmarks via Vitest when headful Chromium is available.

### Benchmark files

| File                      | Runner | Backend     | Patterns                                          |
| ------------------------- | ------ | ----------- | ------------------------------------------------- |
| `bench/scan.bench.ts`     | Vitest | WASM+WebGPU | cumsum, carry-only, reduction, reverse, kalman    |
| `test/deno/scan.bench.ts` | Deno   | WebGPU      | cumsum, reverse (compiled-loop only, no fallback) |

Commands:

```bash
pnpm vitest bench bench/scan.bench.ts   # Vitest (Chromium headless, WASM + WebGPU)
pnpm run bench:deno                     # Deno WebGPU (headless hardware GPU)
```

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
- Scan tests: `test/lax-scan.test.ts` — comprehensive scan suite (~2000 lines)

---

# Part 2: Scan v2 Planning

This section tracks the scan-v2 rewrite plan. It replaces the scan-v1 (feat/scan) approach of three
separate JIT step types with a single loop construct and unified executor.

## feat/scan complexity audit

feat/scan added ~8,800 lines of real source across 87 commits. This is the breakdown of where that
complexity lives, what is essential vs crud, and what v2 should carry forward vs eliminate.

### Line budget (feat/scan vs main)

| Layer              | Files                                             | Lines added | Essential?                                                                                                           |
| ------------------ | ------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| Frontend: planner  | `scan-plan.ts` (new)                              | 1,075       | ~400 reusable (buffer sizing, path heuristics); ~675 is backend-specific param construction that belongs in backends |
| Frontend: JIT      | `jit.ts`                                          | +700        | ~200 essential (scan compilation, body tracing); ~500 is three step types with repeated fields/pprint/execute        |
| Frontend: array.ts | `array.ts`                                        | +550        | ~200 essential (DUS, buffer copy); ~350 is duplicate JIT-vs-eager scan loops                                         |
| Frontend: autodiff | `linearize.ts`, `jvp.ts`, `vmap.ts`               | +1,136      | ~1,000 essential (inherent math); ~136 structural overhead                                                           |
| Frontend: API      | `lax-scan.ts` (new)                               | 484         | ~484 all essential — cleanest layer                                                                                  |
| Backend: WebGPU    | `webgpu.ts`, `scan-wrapper.ts`                    | +1,830      | ~1,500 essential (irreducibly backend-specific); ~330 param-passing glue                                             |
| Backend: WASM      | `wasm.ts`, `routine-provider.ts`, `wasmblr-hl.ts` | +2,950      | ~2,400 essential (codegen, routines, wasmblr-hl); ~550 param-passing glue                                            |
| Backend: interface | `backend.ts`                                      | +80         | ~40 essential; ~40 can collapse if executor owns dispatch                                                            |
| Tests              | 5 files                                           | 4,463       | ~4,463 all essential — port directly                                                                                 |
| **Total source**   |                                                   | **~8,800**  | **~6,500 essential + ~2,300 eliminable**                                                                             |

### Top-7 pain points (what v2 must fix)

**1. Three JIT step types with identical shapes (~500 lines of duplication)**

`scan`, `compiled-loop`, and `preencoded-routine` each repeat ~12 fields (`length`, `numCarry`,
`numConsts`, `reverse`, `consts`, `initCarry`, `xs`, `outputs`), near-identical `pprint` cases, and
the same `execute` pattern (flushPending → resolveScanSlots → dispatch). v2 replaces all three with
one `scan` step that holds a `ScanPlan` discriminated union, eliminating the repetition.

**2. Duplicate fallback loops — JIT scanRunner vs eager Primitive.Scan (~350 lines)**

`array.ts` implements the JS scan body loop twice: once as a `ScanRunner` callback for JIT-compiled
scans, once in the `Primitive.Scan` handler for eager scans. They share slot-wrapping,
body-execution, duplicate-slot detection, and Y-stacking logic but differ subtly in ownership (JIT
skips dispose to preserve pending ops; eager disposes). v2 uses a single `executeScan()` function
called from both paths.

**3. Scattered pending-ops flush discipline (~7 manual flush sites)**

Pending GPU commands are flushed at: JIT execute, scanRunner, fallback loop, DUS handler, and WebGPU
dispatch functions. Missing any site causes silent data corruption. v2 flushes at exactly two
points: before each body invocation and before reading final outputs. One policy, one place.

**4. Ownership rules differ between paths (~400 lines of ad-hoc guards)**

Shared-slot protection (carry ≡ Y slot), passthrough detection, duplicate-slot `incRef`, and
preallocated-slot cleanup each live in separate code paths with subtly different semantics. v2
centralizes all ownership invariants in the single executor, implemented once.

**5. scan-plan.ts mixes analysis with backend param construction (1,075 lines)**

`scan-plan.ts` classifies buffers, computes sizes, builds backend-specific params
(`NativeScanMultiParams`, `NativeScanGeneralParams`, `PreparedPreencodedScan`), reindexes kernels,
and dispatches to backends — all in one file. v2 splits this: buffer/stride analysis → `ScanPlan`
construction (backend-agnostic, ~400 lines); param building → backend executor method
(backend-specific, moves into backends).

**6. Backend codegen shares no code (WASM vs WebGPU: ~4,700 lines total)**

WASM generates a monolithic WebAssembly module (`codegenNativeScanGeneral`: 590 lines). WebGPU has
three sub-paths (multi-kernel shader, pre-encoded routine, JS fallback). They share only
`getScanBufferSizes` (12 lines). This divergence is mostly **inherent** — the backends compile to
fundamentally different targets. v2 accepts this: the shared abstraction is the `ScanPlan`, not the
codegen.

**7. Autodiff: 580-line transpose rule with triple jaxpr construction**

The transpose rule builds three jaxprs (primal-only, tangent-only, transposed- tangent), manages √N
checkpointing, and has its own carry ownership system. This is **inherent complexity** from the math
(JAX-Python has the same structure). v2 keeps this largely as-is; the main simplification is that
scan is `Primitive.Scan` everywhere, no detection heuristics change.

### What is and isn't eliminable

| Category                                                                      | v1 lines               | v2 estimate                | Reduction         |
| ----------------------------------------------------------------------------- | ---------------------- | -------------------------- | ----------------- |
| Eliminable crud (duplicate step types, dual loops, scattered flush/ownership) | ~2,300                 | 0                          | **-2,300**        |
| Backend-agnostic planner (buffer roles, sizes, strides, path selection)       | ~675 (in scan-plan.ts) | ~400 (ScanPlan)            | **-275**          |
| Backend-specific codegen (WASM module gen, WGSL shader gen, scan-wrapper)     | ~3,900                 | ~3,500 (port with cleanup) | **-400**          |
| Autodiff rules (JVP, transpose, vmap)                                         | ~1,136                 | ~1,100 (minor rename)      | **-36**           |
| API + core primitives (`lax-scan.ts`, `core.ts`, `jaxpr.ts`)                  | ~600                   | ~600 (port directly)       | **0**             |
| Tests                                                                         | 4,463                  | 4,463 (port directly)      | **0**             |
| New unified executor                                                          | 0                      | ~300                       | **+300**          |
| **Total**                                                                     | **~13,074**            | **~10,363**                | **-2,711 (~21%)** |

The net reduction is modest (~21%) because most of the v1 complexity is **essential**: backends are
irreducibly different, autodiff math doesn't change, and tests must be preserved. The real win is
**structural** — one execution path instead of six, one ownership policy instead of three.

## Scan v2 design (scan-v2 branch)

### Goals

1. One JIT step type (`scan`) with a `ScanPlan` that captures the execution strategy.
2. One `executeScan()` function that handles ownership, flush, and dispatch for all backends.
3. Backend-specific codegen stays in backends, called via the plan.
4. Port v1 tests as the primary correctness gate.

### Architecture: ScanPlan replaces three step types

```
lax.scan(f, init, xs)
  │
  ▼
Primitive.Scan → makeJaxpr(body) → bodyJaxpr
  │
  ▼ (inside jitCompile)
planScan(backend, bodyProgram, bodyJaxpr, ...) → ScanPlan
  │
  ├─ { path: "compiled-loop", executable }      ← WASM compiled module or WebGPU multi-kernel shader
  ├─ { path: "preencoded-routine", prepared }    ← WebGPU uniform-offset routine scan
  └─ { path: "fallback" }                        ← JS loop calling bodyProgram.execute() per iteration
  │
  ▼
JitStep { type: "scan", plan: ScanPlan, consts, initCarry, xs, outputs, ... }
  │
  ▼ (inside JitProgram.execute)
executeScan(backend, step)
  ├─ flush pending ops on all inputs (ONE policy)
  ├─ preallocate Y stacked buffers if direct-write eligible
  ├─ dispatch based on plan.path (compiled-loop | preencoded-routine | fallback)
  ├─ manage carry lifecycle, shared-slot guards, duplicate-slot incRef (ONE place)
  └─ return carry + stacked ys
```

Key difference from v1: the planner produces a data structure (`ScanPlan`), not a JIT step type. The
executor interprets it. This eliminates the three parallel step type definitions, three pprint
cases, and three execute branches.

### What does NOT change from v1

These pieces port with minimal modification:

- **`lax-scan.ts`** — public API, tree handling, tracing. Port directly.
- **`core.ts` / `jaxpr.ts`** — `Primitive.Scan`, `DynamicUpdateSlice`, abstract eval. Port directly.
- **Autodiff** — JVP rule (`jvp.ts`), transpose rule with √N checkpointing (`linearize.ts`), vmap
  rule (`vmap.ts`). Port directly, rename as needed.
- **Backend codegen** — WASM `codegenNativeScanGeneral` + routine-provider + wasmblr-hl, WebGPU
  `nativeScanMultiShaderSource` + scan-wrapper + preencoded dispatch. Port from feat/scan with
  cleanup.
- **Tests** — all 4,463 lines. Port directly — they are the main asset.

### What changes from v1

| v1 pattern                                                                    | v2 replacement                                                                       | Impact                                                                    |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| Three `JitStep` union members (`scan`, `compiled-loop`, `preencoded-routine`) | One `JitStep` member (`scan`) with `plan: ScanPlan`                                  | -200 lines JIT type defs, -150 lines pprint, -150 lines execute branching |
| Dual scan loop in `array.ts` (`ScanRunner` + `Primitive.Scan`)                | Single `executeScan()` extracted to `scan-executor.ts`                               | -350 lines, one ownership policy                                          |
| 7 manual flush sites                                                          | 2 flush calls in `executeScan()` (before body, before output read)                   | eliminates a class of silent bugs                                         |
| `scan-plan.ts` builds backend params                                          | `planScan()` returns `ScanPlan` (data only); backends call their own prep            | -275 lines, cleaner separation                                            |
| `Backend` interface gains 6 optional scan methods                             | 2-3 methods: `prepareScan?(plan)`, `dispatchScan?(prepared, slots)`, `copyBuffer?()` | simpler interface                                                         |

### Dropped: memref / index_map IR layer

The original v2 plan proposed `src/ir/memref.ts` and `src/ir/index_map.ts` for explicit view/stride
semantics. After auditing v1, this is **over-engineering for the current need**: scan bodies access
data at `base + i * stride`, which is a `{buffer, byteOffset, byteStride}` tuple — not a
multi-dimensional view.

The `ScanPlan` stores buffer roles (const/carry/xs/ys/internal), byte sizes, and per-iteration
strides directly. If view-aware codegen or kernel packing becomes necessary later, memref/index_map
can be introduced then as a refinement, not a prerequisite.

### How to access v1 (feat/scan) source

The feat/scan branch contains the v1 implementation. **Never switch to it** — use `git show` to read
individual files without leaving the scan-v2 branch:

```bash
# Read a file at its feat/scan version:
git show feat/scan:src/library/lax-scan.ts
git show feat/scan:src/frontend/core.ts
git show feat/scan:src/frontend/array.ts
git show feat/scan:src/frontend/jit.ts
git show feat/scan:src/frontend/scan-plan.ts
git show feat/scan:src/backend/wasm.ts
git show feat/scan:src/backend/webgpu.ts
git show feat/scan:src/backend/webgpu/scan-wrapper.ts
git show feat/scan:test/lax-scan.test.ts

# Search for a pattern in v1:
git show feat/scan:src/frontend/jit.ts | grep -n "Primitive.Scan"

# Diff v1 against main to see only scan additions:
git diff main..feat/scan -- src/library/lax-scan.ts
```

### P0 detailed specification

P0 delivers a working `lax.scan()` that executes via the JS fallback loop path only. No WASM
codegen, no WebGPU shaders — just tracing, JIT compilation, and a JS loop calling
`bodyProgram.execute()` per iteration. This establishes the correct API surface and ownership model
that P1–P6 build on.

#### P0 file manifest

| File                            | Action       | Source                                                    | Description                                                              |
| ------------------------------- | ------------ | --------------------------------------------------------- | ------------------------------------------------------------------------ |
| `src/library/lax-scan.ts`       | Create (new) | Port from `feat/scan:src/library/lax-scan.ts` (484 lines) | Public `lax.scan()` API                                                  |
| `src/frontend/core.ts`          | Edit         | Port additions from `feat/scan:src/frontend/core.ts`      | Add `Primitive.Scan`, `DynamicUpdateSlice`, params                       |
| `src/frontend/jaxpr.ts`         | Edit         | Port additions from `feat/scan:src/frontend/jaxpr.ts`     | Add abstract eval rules for Scan and DUS                                 |
| `src/frontend/jit.ts`           | Edit         | New v2 design                                             | Add `scan` JitStep (single type), Primitive.Scan case in `jitCompile()`  |
| `src/frontend/scan-executor.ts` | Create (new) | New v2 design                                             | `executeScan()` — unified scan loop                                      |
| `src/frontend/scan-plan.ts`     | Create (new) | Partially port from feat/scan                             | `ScanPlan` type, `planScan()` (fallback-only in P0)                      |
| `src/frontend/array.ts`         | Edit         | Port additions from feat/scan                             | `Primitive.Scan` eager impl + `DynamicUpdateSlice`, `#copySliceToBuffer` |
| `src/library/lax.ts`            | Edit         | —                                                         | Wire `scan` export                                                       |
| `src/index.ts`                  | Edit         | —                                                         | Export `ScanPath`, `setScanBodyStepsCallback`                            |

#### P0 data flow (end-to-end)

```
User calls lax.scan(f, init, xs, { length?, reverse? })
  │
  ▼ lax-scan.ts: scan()
1. tree.flatten(init) → [initFlat, carryTreeDef]
2. tree.flatten(xs)   → [xsFlat, xsTreeDef]
3. Determine scan length from xs shapes (or explicit length param)
4. Build flatF: (carry, xSlice) → { carry, y }
   - Unflattens carry via carryTreeDef, xSlice via xsTreeDef
   - Calls user's f(carry, xSlice)
   - Flattens results back
5. traceFn = (carryAvals, xAvals) → flatF(carryAvals, xAvals)
6. bodyJaxpr = makeJaxpr(traceFn)(...traceAvals)
7. Handle length === 0 early return: carry = init, ys = empty
8. bind(Primitive.Scan, [...consts, ...initFlat, ...xsFlat], {
     jaxpr: bodyJaxpr, numCarry, numConsts, length, reverse,
     acceptPath, checkpoint
   })
9. Split results → carry (first numCarry) + ys (rest)
10. Unflatten carry via carryTreeDef, each y via yTreeDef
  │
  ▼ Two execution paths from bind():
  ├─ EAGER (tracing off): Primitive.Scan impl rule in array.ts
  │    → calls executeScan() with bodyProgram from jitCompile(bodyJaxpr)
  │
  └─ JIT (inside jit()): Primitive.Scan recorded in Jaxpr
       → jitCompile() encounters Primitive.Scan
       → calls planScan() → ScanPlan { path: "fallback" }  (P0 only)
       → emits JitStep { type: "scan", plan, ... }
       → JitProgram.execute() calls executeScan()
```

#### P0 types to add

**In `core.ts`:**

```typescript
// Add to Primitive enum:
Scan = "scan",

// Add to PrimitiveParamsImpl:
[Primitive.Scan]: {
  jaxpr: Jaxpr;
  numCarry: number;
  numConsts: number;
  length: number;
  reverse: boolean;
  acceptPath?: ScanPath | ScanPath[];
  checkpoint?: boolean | number;
};
```

**`DynamicUpdateSlice` function** — needed by direct-write Y stacking. Port from feat/scan
`core.ts`. It takes `(operand, update, startIndices)` and produces an array equal to `operand`
except for the slice starting at `startIndices` which is replaced by `update`. In jit, it lowers to
a buffer copy step.

**In `scan-plan.ts`:**

```typescript
export type ScanPath = "compiled-loop" | "preencoded-routine" | "fallback";

export type ScanPlan =
  | { path: "fallback"; extraInfo?: string }
  | { path: "compiled-loop"; executable: Executable; params?: any }
  | { path: "preencoded-routine"; preencodedParams: any };

// P0 implementation: always returns fallback
export function planScan(...): ScanPlan {
  return { path: "fallback" };
}
```

P2–P4 will populate the compiled-loop and preencoded-routine variants. The `any` types will be
replaced with `NativeScanGeneralParams` and `PreparedPreencodedScan` when those are ported.

**In `jit.ts`:**

```typescript
// ONE scan step type (replaces v1's three):
| {
    type: "scan";
    plan: ScanPlan;
    bodyProgram: JitProgram;
    bodyJaxpr: Jaxpr;
    length: number;
    numCarry: number;
    numConsts: number;
    numX: number;
    numY: number;
    reverse: boolean;
    consts: JitId[];
    initCarry: JitId[];
    xs: JitId[];
    xsAvals: ShapedArray[];
    outputs: JitId[];  // [carry_out..., stacked_ys...]
  }
```

The `plan` field is checked at execute time. P0's plan is always `{ path: "fallback" }`, so the JIT
execute path calls `executeScan()` with the fallback loop. P2–P4 add compiled-loop and
preencoded-routine dispatch.

#### Ownership rules (critical — v2 unification)

v1's #1 bug source was divergent ownership between the eager and JIT paths. v2 enforces ONE policy
in `executeScan()`:

**Invariant 1: Flush before body.** Before each loop iteration calls `bodyProgram.execute()`, all
pending GPU commands are submitted. This ensures carry slots and xs slices contain up-to-date data.

**Invariant 2: Carry lifecycle.** Carry slots are owned by the loop. On each iteration:

1. Pass carry slots to body as inputs (consume semantics — body takes ownership)
2. Body returns new carry slots as outputs
3. Old carry slots may alias new ones (passthrough) — detect via slot identity
4. Non-aliased old carry slots are freed after the iteration

**Invariant 3: Y stacking.** Two strategies:

- **Direct-write** (preferred): Preallocate full Y output buffers, copy each iteration's y into the
  correct slice via `copySliceToBuffer`. No concatenation.
- **Collect-and-stack** (fallback): Collect y slices in an array, concatenate at end. Only used when
  direct-write fails (e.g., zero-size ys).

**Invariant 4: Shared-slot protection.** When carry_out[i] aliases y_out[j] (same Jaxpr variable),
the same slot would be both reused as carry input and stacked as Y output. Protect by `incRef` on
the shared slot before stacking.

**Invariant 5: xs slicing.** For each xs input, iteration `i` reads an `xsAvals[j].size`-element
slice at offset `i * stride` (or `(length-1-i) * stride` for reverse). The executor computes the
byte offset and uses `backend.createView()` or similar.

**Invariant 6: No dispose in JIT path.** When `executeScan()` is called from `JitProgram.execute()`,
carry/y slots are tracked by the JIT scope — the executor must NOT dispose them. When called from
the eager `Primitive.Scan` impl, the executor owns the slots and disposes them normally. Distinguish
via a parameter (e.g., `disposeInputs: boolean`).

#### executeScan() function signature (scan-executor.ts)

```typescript
export interface ExecuteScanParams {
  backend: Backend;
  plan: ScanPlan;
  bodyProgram: JitProgram;
  bodyJaxpr: Jaxpr;
  length: number;
  numCarry: number;
  numConsts: number;
  numX: number;
  numY: number;
  reverse: boolean;
  constSlots: Slot[];
  initCarrySlots: Slot[];
  xsSlots: Slot[];
  xsAvals: ShapedArray[];
  outputSlots: Slot[]; // preallocated by JIT or eager path
}

export function executeScan(params: ExecuteScanParams): {
  outputs: Slot[]; // [carry_out..., stacked_ys...]
  pending: PendingExecute[];
};
```

The function dispatches on `params.plan.path`:

- `"fallback"` → JS loop (P0)
- `"compiled-loop"` → `backend.dispatchNativeScanGeneral()` (P2/P3)
- `"preencoded-routine"` → `backend.dispatchPreencodedScan()` (P4)

#### Fallback loop pseudocode (P0 scope)

```
function executeScanFallback(params):
  carry = params.initCarrySlots  (ref'd, not consumed)
  ys = [][]  // numY arrays of length `length` slices
  pending = []

  for i in 0..length (reversed if params.reverse):
    // Invariant 1: flush
    FlushPending(pending)

    // Slice xs for this iteration
    xSlices = sliceXsAtIteration(params.xsSlots, params.xsAvals, i)

    // Call body: [consts, carry, xSlices] → [newCarry, ySlices]
    bodyInputSlots = [...params.constSlots.map(incRef), ...carry, ...xSlices]
    { outputs, pending: newPending } = params.bodyProgram.execute(bodyInputSlots)
    pending.push(...newPending)

    newCarry = outputs.slice(0, params.numCarry)
    ySlices = outputs.slice(params.numCarry)

    // Invariant 4: shared-slot protection
    for each shared carry/y slot: incRef

    // Invariant 3: Y stacking (direct-write)
    for j in 0..numY:
      copySliceToBuffer(outputSlots[numCarry + j], ySlices[j], i, ysStride[j])

    // Invariant 2: free old carry (skip if passthrough)
    carry = newCarry

  // Write final carry to output slots
  for j in 0..numCarry:
    copyToSlot(outputSlots[j], carry[j])

  return { outputs: outputSlots, pending }
```

#### DynamicUpdateSlice (DUS) handling

`DynamicUpdateSlice` is a primitive used by direct-write Y stacking. It copies a source slice into a
destination buffer at a specified offset. In JIT, it compiles to a `copy` step (buffer clone +
patch).

**What to port from feat/scan:**

- `DynamicUpdateSlice()` function in `core.ts`
- Abstract eval rule in `jaxpr.ts` (output shape = operand shape)
- JIT lowering in `jit.ts` — detect DUS in `splitGraphDataflow()` as a "black node" and emit a
  `JitStep { type: "copy" }` step
- Backend `copyBuffer(src, dst, srcOffset, dstOffset, size)` method

**For P0**, DUS can be implemented directly using `backend.copyBuffer()` in the `executeScan()`
fallback loop without going through the JIT copy step. The JIT path for DUS is only needed when scan
bodies themselves contain DUS (rare).

#### Commit granularity for P0

P0 should be built in these sub-commits (each passing tests):

1. **core + jaxpr primitives**: Add `Primitive.Scan` to enum, `PrimitiveParams`, abstract eval rule.
   Add `DynamicUpdateSlice` primitive + eval rule. Existing tests still pass (no behavior change).

2. **lax-scan.ts + lax.ts wiring**: Create `src/library/lax-scan.ts` ported from feat/scan. Add
   `scan` to `lax.ts` exports. Add `ScanPath` export to `index.ts`. At this point `lax.scan()`
   traces and calls `bind(Primitive.Scan, ...)`, but execution will error because no impl rule
   exists yet.

3. **scan-plan.ts + scan-executor.ts**: Create `ScanPlan` type and fallback-only `planScan()`.
   Create `executeScan()` with the fallback loop. These are new files that nothing calls yet.

4. **array.ts eager impl**: Add `Primitive.Scan` to the impl rules in `array.ts`. This calls
   `jitCompile(bodyJaxpr)` + `executeScan()`. Now `lax.scan()` works in eager mode.

5. **jit.ts compilation**: Add scan JitStep type, handle `Primitive.Scan` in `jitCompile()`, handle
   `"scan"` step in `JitProgram.execute()`. Now `jit(f)` where `f` uses `lax.scan()` works.

6. **Exports + smoke test**: Export public symbols, add a basic scan test (cumsum). Verify build +
   check + vitest + deno all pass.

### Implementation phases

| Phase                              | Description                                                                                                                  | New/adapted lines | Port from v1                                                                                         | Risk                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------- |
| **P0: API + primitive + fallback** | `lax-scan.ts`, `Primitive.Scan` in core/jaxpr, `planScan()` returning fallback-only `ScanPlan`, `executeScan()` with JS loop | ~800 new          | `lax-scan.ts` (484), `core.ts` (+50), `jaxpr.ts` (+60)                                               | Low                           |
| **P1: Tests on fallback**          | Port all scan tests, run on JS fallback path only                                                                            | ~0 new            | 4,463 lines of tests                                                                                 | Low — tests catch regressions |
| **P2: WASM compiled-loop**         | Port `codegenNativeScanGeneral`, `routine-provider`, wire into `planScan`                                                    | ~100 new glue     | `wasm.ts` scan codegen (~600), `routine-provider.ts` (261), `wasmblr-hl.ts` (if not already on main) | Medium — mostly mechanical    |
| **P3: WebGPU multi-kernel**        | Port `nativeScanMultiShaderSource`, `genScanExpressionWithRidx`, wire into `planScan`                                        | ~100 new glue     | `webgpu.ts` scan functions (~600)                                                                    | Medium — WebGPU-specific      |
| **P4: WebGPU preencoded-routine**  | Port `scan-wrapper.ts`, pre-encoded dispatch, ping-pong carry                                                                | ~50 new glue      | `scan-wrapper.ts` (386), dispatch logic (~200)                                                       | Medium — uniform offset logic |
| **P5: Autodiff**                   | Port JVP, transpose (with √N checkpointing), vmap rules                                                                      | ~50 adaptation    | `jvp.ts` (+200), `linearize.ts` (+825), `vmap.ts` (+111)                                             | Low — mostly rename           |
| **P6: Cleanup**                    | Delete v1 scan code from feat/scan, collapse scan-plan into planner, update docs                                             | net -1,500        | —                                                                                                    | Low — test suite validates    |

**Estimated total v2 source (excluding tests): ~6,500 lines** vs v1's ~8,800. The reduction is ~26%,
but the structural improvement (one execution path, one ownership policy) is the primary value — not
the line count.

### Progress tracker

Update this table as phases are completed. Each phase should be marked with its current status and
any notes about deviations from the plan.

| Phase                              | Status      | Notes                                                                                                                                         |
| ---------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0: API + primitive + fallback** | Done        | 6 sub-commits: core primitives, lax-scan.ts, scan-plan/executor, array.ts eager, jit.ts, smoke tests. 9 scan tests pass (eager + JIT + wasm). |
| **P1: Tests on fallback**          | Not started |                                                                                                                                               |
| **P2: WASM compiled-loop**         | Not started |                                                                                                                                               |
| **P3: WebGPU multi-kernel**        | Not started |                                                                                                                                               |
| **P4: WebGPU preencoded-routine**  | Not started |                                                                                                                                               |
| **P5: Autodiff**                   | Not started |                                                                                                                                               |
| **P6: Cleanup**                    | Not started |                                                                                                                                               |

Status values: `Not started` · `In progress` · `Done` · `Blocked`

### Risks and guardrails

- **Performance regression from extra indirection.** The `ScanPlan` → executor dispatch adds one
  level vs v1's direct step execution. Profile after P2/P3 to verify no measurable overhead.
- **WebGPU still needs three sub-paths.** The "no global barrier" hardware constraint means compiled
  shaders, pre-encoded routines, and JS fallback are irreducibly different strategies. The v2 win is
  that they share one executor for ownership/flush, not that they merge into one codegen path.
- **Autodiff rules are orthogonal.** JVP/transpose/vmap operate on the Jaxpr, not the execution
  layer. Switching from v1's step types to `ScanPlan` does not simplify the math. Budget time to
  port but don't expect simplification.
- **Edge cases live in the tests.** feat/scan accumulated 87 commits of fixes (length-0, reverse,
  passthrough, shared-slot). Port tests before code to avoid re-discovering these.

## Lessons learned from scan-v1 (feat/scan)

- Split execution paths created duplicated ownership rules; v2 must have one executor.
- Pending ops must be flushed before loop bodies read slots; make this a single policy.
- WebGPU alignment limits require uniform-offset addressing for strided access.
- Direct-write Y stacking is essential for long scans and must be part of the core plan.
- Fallback paths must share body execution semantics with compiled paths to avoid drift.
- `scan-plan.ts` grew to 1,075 lines because it mixed analysis with backend param construction — v2
  should split planner (data) from backend prep (codegen).
- The JIT-vs-eager dual loop was the #1 source of bugs: shared-slot guards, pending-op flush, and
  dispose semantics diverged incrementally across commits.

## Reusable pieces from scan-v1 (with modifications)

When implementing scan-v2, check the reusable list and function-level inventory below (feat/scan)
before writing new code.

- Backend/WebGPU: WGSL copy shader for unaligned buffer copies and ping-pong carry dispatch logic in
  [src/backend/webgpu.ts](src/backend/webgpu.ts).
- Backend/WebGPU: Uniform-offset routine wrapper and offset buffer builder in
  [src/backend/webgpu/scan-wrapper.ts](src/backend/webgpu/scan-wrapper.ts).
- Backend/WebGPU: Scan-wrapper tests that validate shader rewriting in
  [src/backend/webgpu/scan-wrapper.test.ts](src/backend/webgpu/scan-wrapper.test.ts).
- Backend/WASM: General scan param model and routine import wiring in
  [src/backend/wasm.ts](src/backend/wasm.ts).
- Backend/WASM: Loop and memory helpers in
  [src/backend/wasm/wasmblr-hl.ts](src/backend/wasm/wasmblr-hl.ts).
- Backend/WASM: Routine modules and providers for scan bodies that call routines in
  [src/backend/wasm/routine-provider.ts](src/backend/wasm/routine-provider.ts) and
  [src/backend/wasm/routines](src/backend/wasm/routines).
- Frontend: Path selection and buffer sizing helpers in
  [src/frontend/scan-plan.ts](src/frontend/scan-plan.ts).
- Frontend: Fallback loop, DUS-based Y stacking, and shared-slot protection in
  [src/frontend/array.ts](src/frontend/array.ts).
- Tests: Scan preallocate coverage in
  [test/scan-preallocate.test.ts](test/scan-preallocate.test.ts).
- Tests: Scan correctness and regression suite in [test/lax-scan.test.ts](test/lax-scan.test.ts) and
  [test/jit-scan-dlm.test.ts](test/jit-scan-dlm.test.ts).
- Tests/Deno: Leak harness and WebGPU coverage in [test/deno/harness.ts](test/deno/harness.ts) and
  [test/deno/webgpu.test.ts](test/deno/webgpu.test.ts).
- Tests/Deno: Preencoded routine coverage in
  [test/deno/preencoded-scan.test.ts](test/deno/preencoded-scan.test.ts) and
  [test/deno/preencoded-scan-integration.test.ts](test/deno/preencoded-scan-integration.test.ts).
- Benchmarks/Deno: Scan overhead and Kalman trace benchmarks in
  [test/deno/scan-overhead.bench.ts](test/deno/scan-overhead.bench.ts) and
  [test/deno/dlm-kalman-trace.bench.ts](test/deno/dlm-kalman-trace.bench.ts).
- Scripts: Unified test runner for Deno fallback in [scripts/test-all.sh](scripts/test-all.sh).

### Function-level inventory (feat/scan)

- src/backend/webgpu/scan-wrapper.ts: `parseBufferBindings`, `transformArrayAccesses`,
  `wrapRoutineForScan`, `createScanOffsetsUniform`, `createAllIterationsOffsetsBuffer`,
  `getOffsetBindings`, `generateOffsetsStruct`, `findMainBodyStart`, `generateOffsetDeclarations`.
- src/backend/webgpu.ts (scan-specific): `dispatchNativeScanGeneral`, `prepareNativeScanMulti`,
  `getPreencodedScanAlignment`, `preparePreencodedScan`, `dispatchPreencodedScan`,
  `genScanExpressionWithRidx`, `nativeScanMultiShaderSource`.
- src/backend/wasm.ts (scan-specific): `prepareNativeScanGeneral`, `dispatchNativeScanGeneral`,
  `codegenNativeScanGeneral`, `translateExpWithGeneralScanContext`.
- src/backend/wasm/wasmblr-hl.ts: `WasmHl` methods `forLoop`, `forLoopDown`, `whileLoop`, `ifElse`,
  `addr`, `load`, `loadDirect`, `storeAddr`, `storeDirect`, `store`, `memcpy`, `memcpyDynamic`,
  `index2D`, `get`, `getExpr`, `loadF32x4`, `storeAddrF32x4`, `storeDirectF32x4`, `storeF32x4`,
  `f32x4Hsum`, `f32x4Splat`, `f64x2Hsum`, `f64x2Splat`, `const`, `sqrt`, `binOp`, `eq`, `ltS`,
  `leS`, `forLoopUnrolled`, `simdReductionF32`, `simdReductionF64`.
- src/backend/wasm/routine-provider.ts: `getCholeskyModule`, `getTriangularSolveModule`,
  `getLUModule`, `getSortModule`, `getArgsortModule`, `clearRoutineCache`, `getRoutineCacheSize`,
  plus key helpers `choleskyKey`, `triangularSolveKey`, `luKey`, `sortKey`, `argsortKey`.
- src/frontend/scan-plan.ts: `checkAcceptedPath`, `getScanBufferSizes`, `tryPreparePreencodedScan`,
  `tryPrepareWebGPUNativeScan`, `tryPrepareWasmNativeScan`, `tryPrepareNativeScan`, `planScan`.
- src/frontend/array.ts (scan fallback): `Array.#stackScanYs`, `Array.#runScanFallbackLoop` and the
  `ScanRunner`-driven loop in `Primitive.Jit` and `Primitive.Scan`.
- test/deno/harness.ts: `getSlotCount`, `assertNoLeaks`, `withLeakCheck`, `leakCheckTest`.
- test/scan-preallocate.test.ts: inline scan step lambdas covering preallocated Y stacking,
  duplicate-slot Y, passthrough Y, reverse scan, length-0.
- test/lax-scan.test.ts: helper lambdas like `cumsumScan`, `cumprodScan`, `reverseCumsumScan`,
  `sumOfCumsum`, `cumsumWithOutputs`, `cumsumWithJitBody`, `cumsumWithSum`, plus `loss` helpers in
  grad/vmap sections.
- test/jit-scan-dlm.test.ts: `forwardStep` (Kalman-like), `runSmoother` (two-pass pattern) in
  regression coverage.
- test/deno/preencoded-scan.test.ts: `getJaxJsWebGPUDevice` (backend bootstrap), plus matmul scan
  step lambdas.
- test/deno/scan-overhead.bench.ts: `complexStep` benchmark body.
- test/deno/dlm-kalman-trace.bench.ts: `forwardStep` Kalman benchmark body.

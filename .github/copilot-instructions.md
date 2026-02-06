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

Husky runs `lint-staged` on commit, which auto-fixes ESLint and Prettier issues on staged files.

**Before any commit**, also run these checks (not enforced by husky):

```bash
pnpm build                         # Build the project
pnpm check                         # TypeScript type checking
pnpm exec playwright install chromium-headless-shell  # (if not already installed)
pnpm test                          # Run all tests
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
See [WebGPU preencoded-routine details](#webgpu-preencoded-routine-details-routine-body) in Part 2.

### Features exploited

| Feature                     | How jax-js uses it                                               | Location                                |
| --------------------------- | ---------------------------------------------------------------- | --------------------------------------- |
| **shader-f16**              | Float16 dtype support; requested at device creation              | `src/backend.ts` feature requests       |
| **Workgroup shared memory** | Sort uses `var<workgroup>` for bitonic sort local exchanges      | `src/backend/webgpu/routines.ts`        |
| **workgroupBarrier()**      | Synchronizes threads within Sort workgroups                      | `bitonicSortShader` in routines.ts      |
| **storageBarrier()**        | Memory fence for shared variable consistency                     | `bitonicSortShader` in routines.ts      |
| **Pipeline caching**        | Compiled pipelines stored by shader hash                         | `pipelineCache` in webgpu.ts            |
| **Command batching**        | Multiple dispatches encoded before single queue.submit()         | `PendingExecute` in webgpu.ts           |
| **Ping-pong buffers**       | Scan carry state alternates between two buffers                  | `dispatchPreencodedScan()` in webgpu.ts |
| **Uniform buffers**         | Per-iteration offsets for preencoded scan                        | `scan-wrapper.ts`                       |
| **WGSL copy shader**        | Byte-level buffer copy when `copyBufferToBuffer` alignment fails | `COPY_SHADER_CODE` in webgpu.ts         |

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
   fused, like reductions or routines) vs fusable elementwise ops
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

**Kernel class (unified single/multi-output):**

The `Kernel` class uses an `outputs: KernelOutput[]` array to support 1..N outputs:

- `Kernel.single(nargs, size, exp, reduction?)` — single-output kernel
- `Kernel.multi(nargs, outputs[])` — multi-output kernel
- `kernel.isMultiOutput` — true if `outputs.length > 1`
- `kernel.dtypeAt(i)` — dtype of output `i`

**Adding a new primitive:**

1. Declare in `Primitive` enum (`src/frontend/core.ts`)
2. Add tracing rule in `implRules` / `jvpRules` / `transposeRules`
3. If fusable elementwise, add ALU lowering in `jit.ts`
4. If needs dedicated kernel, register in `routinePrimitives` and implement in `src/backend/*`

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

# Part 2: Scan Feature Reference

This section documents the `lax.scan` implementation architecture, design choices, and
backend-specific behavior.

> **API Stability:** The scan feature is under active development. Breaking API changes may occur
> without deprecation warnings. No external users depend on this API yet.

## Overview & Motivation

`lax.scan` applies a function over the leading axis of arrays, threading carry state — essential for
RNNs, Kalman filters, cumulative operations, and other sequential computations.

**Signature:**

```ts
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs, options);
// f: (carry, x) => [newCarry, y]
```

**Options:**

- `length?: number` — Number of iterations (inferred from xs if not provided)
- `reverse?: boolean` — Process xs in reverse order (default: false)
- `acceptPath?: ScanPath | ScanPath[]` — Accept only these paths; throws if actual path not in list
- `checkpoint?: boolean | number` — Control gradient checkpointing for `grad(scan)`. Default
  (undefined/true) uses √N checkpointing. A number specifies the segment size. `false` stores all
  carries (O(N) memory).
- `preallocateY?: boolean` — When true (default) and using the JS fallback scan path, preallocates
  the stacked Y output buffer and writes each iteration's Y directly via `copyBufferToBuffer`
  (4-byte aligned) or the WGSL copy shader (unaligned). Avoids stack overflow on long scans and
  reduces O(length) intermediate arrays from `coreConcatenate`. No effect on compiled-loop or
  preencoded-routine paths (they already write directly).

**Scan paths (`ScanPath` type):**

- `"compiled-loop"` — Entire scan loop compiled to native code (WASM module or WebGPU shader)
- `"preencoded-routine"` — Pre-encoded GPU command dispatches with uniform offsets per iteration
  (WebGPU only)
- `"fallback"` — JS loop calling body program per iteration (one or more JS↔backend boundary
  crossings)

Use `acceptPath: ["compiled-loop", "preencoded-routine"]` in tests to ensure native compilation
doesn't regress.

**xs=null and Y=null (jax-js extensions):**

Unlike Python JAX, jax-js supports null inputs and outputs for efficiency:

- **xs=null:** When xs is null, you must provide `length` option. Body receives null as x.
- **Y=null:** Body can return `[newCarry, null]` to skip output stacking entirely.

See [API Contract](#scan-reference-contract) for code examples and ownership details.

**Use cases:**

- Cumulative sum/product
- RNN/LSTM forward pass
- Kalman filter (forward and backward passes)
- Any sequential state machine

**Key files:**

| File                                 | Role                                                          |
| ------------------------------------ | ------------------------------------------------------------- |
| `src/library/lax-scan.ts`            | Public API                                                    |
| `src/frontend/core.ts`               | `Primitive.Scan` enum + params type                           |
| `src/frontend/jaxpr.ts`              | Abstract eval rule                                            |
| `src/frontend/array.ts`              | Scan impl rule with `scanRunner` callback                     |
| `src/frontend/jit.ts`                | JIT step types: `scan`, `compiled-loop`, `preencoded-routine` |
| `src/frontend/linearize.ts`          | JVP + transpose rules for autodiff                            |
| `src/frontend/vmap.ts`               | Scan vmap rule (batches independent scans)                    |
| `src/backend/wasm.ts`                | Compiled-loop codegen                                         |
| `src/backend/webgpu.ts`              | Compiled-loop + preencoded-routine for routines               |
| `src/backend/webgpu/scan-wrapper.ts` | WGSL shader transformer for uniform offsets                   |

---

## Feature Status by Backend

### CPU Backend

The CPU backend uses JavaScript-interpreted evaluation. It serves as the reference implementation
for correctness testing. All scan tests are in [test/lax-scan.test.ts](test/lax-scan.test.ts).

| Feature / Test             | Status  | Notes                                |
| -------------------------- | ------- | ------------------------------------ |
| `scan basic`               | ✅ Pass |                                      |
| `scan with pytree carry`   | ✅ Pass | pytree = nested dict/array structure |
| `reverse scan`             | ✅ Pass |                                      |
| `jit + scan`               | ✅ Pass |                                      |
| `JVP (forward-mode)`       | ✅ Pass |                                      |
| `VJP (reverse-mode)`       | ✅ Pass |                                      |
| `vmap`                     | ✅ Pass |                                      |
| `vmap` > `jit(vmap(scan))` | ✅ Pass |                                      |
| `vmap` > `vmap(jit(scan))` | ✅ Pass |                                      |
| `scan over views`          | ✅ Pass | sliced/transposed xs                 |

### WASM Backend

The WASM backend supports **compiled-loop**: the entire scan loop is compiled into a WebAssembly
module, eliminating JS/WASM boundary overhead per iteration.

| Feature / Test                      | Status  | Notes                               |
| ----------------------------------- | ------- | ----------------------------------- |
| `scan basic`                        | ✅ Pass |                                     |
| `scan with pytree carry`            | ✅ Pass |                                     |
| `reverse scan`                      | ✅ Pass |                                     |
| `jit + scan`                        | ✅ Pass |                                     |
| `JVP (forward-mode)`                | ✅ Pass |                                     |
| `VJP (reverse-mode)`                | ✅ Pass |                                     |
| `vmap`                              | ✅ Pass |                                     |
| `vmap` > `jit(vmap(scan))`          | ✅ Pass |                                     |
| `vmap` > `vmap(jit(scan))`          | ✅ Pass |                                     |
| `scan over views`                   | ✅ Pass | sliced/transposed xs                |
| `compiled-loop`                     | ✅ Pass |                                     |
| `compiled-loop` > `with constants`  | ✅ Pass |                                     |
| `compiled-loop` > `reverse=true`    | ✅ Pass | all variants support reverse        |
| `scan with routine body`            | ✅ Pass |                                     |
| `routine in scan body`              | ✅ Pass | uses compiled-loop via WASM imports |
| `grad` through `scan` with routines | ✅ Pass | works via compiled-loop             |

**Performance benchmarks:**

- Fallback (JS loop): ~500 iter/sec
- Compiled-loop: ~50-80M iter/sec (small elementwise bodies, L=1000)
- Compiled-loop with direct-write: **40-65% faster** than without for small bodies

**Small scan throughput (L=1000 iterations, WASM compiled-loop):**

| Body Pattern               | Throughput    | Notes                                    |
| -------------------------- | ------------- | ---------------------------------------- |
| Cumsum (scalar)            | ~62M iter/sec | 1 carry, 1 Y, direct-write active        |
| Carry-only (4×4, Y=null)   | ~50M iter/sec | 1 carry, no Y output                     |
| Elementwise (n=4, Y=carry) | ~78M iter/sec | 1 carry, 1 Y, direct-write active        |
| Passthrough Y (4×4)        | ~35M iter/sec | Y = old carry, direct-write not eligible |

**Scan vs jit(loop) overhead:**

Compiled-loop is consistently faster than manual `jit(loop)` at all tested sizes due to eliminating
JS↔WASM boundary crossings per iteration:

| Matrix Size | Overhead | Notes                                |
| ----------- | -------- | ------------------------------------ |
| 16×16       | **-98%** | Scan FASTER (single WASM invocation) |
| 32×32       | **-98%** | Scan FASTER                          |
| 64×64       | **-96%** | Scan FASTER                          |
| 128×128     | **-84%** | Scan FASTER                          |

Compiled-loop compiles the entire loop into one WASM module, avoiding per-iteration overhead. Memory
copies within the loop use `memory.copy` (bulk memory) for efficient carry/output transfers.

### WebGPU Backend

The WebGPU backend keeps data on GPU between iterations. Supports **compiled-loop** for elementwise
kernels, **multi-kernel scan** for bodies with multiple independent kernels, and
**preencoded-routine** for single-routine bodies meeting specific requirements (currently Cholesky).

| Feature / Test                     | Status      | Notes                                        |
| ---------------------------------- | ----------- | -------------------------------------------- |
| `scan basic`                       | ✅ Pass     | uses compiled-loop on WebGPU                 |
| `scan with pytree carry`           | ✅ Pass     |                                              |
| `reverse scan`                     | ✅ Pass     | uses compiled-loop with dataIdx              |
| `jit + scan`                       | ✅ Pass     |                                              |
| `JVP (forward-mode)`               | ✅ Pass     |                                              |
| `VJP (reverse-mode)`               | ✅ Pass     |                                              |
| `vmap`                             | ✅ Pass     |                                              |
| `vmap` > `jit(vmap(scan))`         | ✅ Pass     |                                              |
| `vmap` > `vmap(jit(scan))`         | ✅ Pass     |                                              |
| `scan over views`                  | ✅ Pass     | sliced/transposed xs                         |
| `compiled-loop`                    | ✅ Pass     | kernel gids reindexed to scan layout         |
| `compiled-loop` > `with reduction` | ✅ Pass     | e.g., `carry += sum(x)` or matmul            |
| `compiled-loop` > `with reverse`   | ✅ Pass     | uses dataIdx like WASM                       |
| `compiled-loop` > `with constants` | ✅ Pass     | captured constants bound as storage          |
| `multi-kernel scan`                | ✅ Pass     | derives output mapping from body outputs     |
| `preencoded-routine` (Cholesky)    | ✅ Pass     | requires passthrough pattern (numCarry=numY) |
| Mixed kernel+routine bodies        | ⚠️ Fallback | e.g., Kalman filter, lstsq                   |
| Multi-routine bodies               | ⚠️ Fallback | e.g., Cholesky→TriSolve→TriSolve             |
| Sort in scan body                  | ⚠️ Fallback | Sort already uses uniforms (conflict)        |

**Note on numCarry ≠ numY:** WebGPU compiled-loop requires `numCarry === numY`. When they differ,
WebGPU falls back to JS loop. WASM compiled-loop handles this case.

**Tested on:** NVIDIA RTX 4070 Ti SUPER via Deno WebGPU (headless, no X11)

### WebGL Backend

The WebGL backend has **no compiled-loop support**. All scans use the JS fallback path, which
executes the body program per iteration. This works correctly but lacks optimization.

| Feature / Test | Status      | Notes                                         |
| -------------- | ----------- | --------------------------------------------- |
| `scan basic`   | ⚠️ Untested | Uses fallback path; requires browser with GPU |
| `jit + scan`   | ⚠️ Untested | Uses fallback path                            |

**Note:** WebGL tests exist in `test/lax-scan.test.ts` but are **untested in CI** because:

- Deno doesn't provide WebGL (only WebGPU)
- Playwright's headless Chromium doesn't expose WebGL in the test environment
- The dev system lacks a display for headed browser testing

The fallback `scanRunner` is backend-agnostic and tested with CPU/WASM/WebGPU, so WebGL should work
identically. To verify manually, run website demos in a WebGL-capable browser.

---

## Design Choices & Rationales

### Why compiled-loop vs preencoded-routine vs fallback?

| Approach               | How it works                                  | When used                                        |
| ---------------------- | --------------------------------------------- | ------------------------------------------------ |
| **compiled-loop**      | Entire scan loop in native code (WASM/shader) | Elementwise kernels (WASM+WebGPU), WASM routines |
| **preencoded-routine** | Pre-encode dispatches with uniform offsets    | WebGPU single-routine bodies (e.g., Cholesky)    |
| **fallback**           | JS loop calling body program per iteration    | Unsupported patterns, mixed bodies, Sort         |

**Rationale:** compiled-loop is preferred because:

1. Eliminates JS↔native boundary per iteration (~5000× speedup for WASM)
2. Enables compiler optimizations across iterations
3. Single WASM module instantiation vs N calls

preencoded-routine is used for WebGPU routines that can't be inlined into a shader. It transforms
routine shaders to accept per-iteration offsets via uniforms, enabling fused dispatch.

### Why wasmblr for WASM routines?

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

See the [Routine System](#routine-system) section for implementation details and wasmblr patterns.

### Why 3 routine implementations (CPU/WASM/WebGPU)?

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

**Calling routines from scan loops:** Scan modules use WASM imports to call routines from separate
wasmblr modules. This avoids code duplication (each routine is 1-3KB) while keeping the entire loop
in native code (~1.5M iter/sec). See `codegenNativeScanGeneral()` in `src/backend/wasm.ts`.

---

## API Contract

### Scan reference contract

This contract applies to both `lax.scan()` and `jit(() => lax.scan(...))()`:

**Inputs — consumed:**

```ts
const [carry, ys] = lax.scan(f, init, xs);
// init and xs are consumed (refcount -1)
// Use .ref if you need them after: lax.scan(f, init.ref, xs.ref)
```

**xs=null for carry-only scans:**

```ts
// When xs is null, you must provide length option
const [carry, ys] = lax.scan(f, init, null, { length: 100 });
// No memory allocated for xs - useful for generators, RNG sequences, etc.
// Body receives null as second argument: f(carry, null) => [newCarry, y]
```

**Y=null to skip output stacking:**

```ts
// Return null as Y to avoid allocating stacked outputs
const [carry, nullYs] = lax.scan(f, init, xs);
// f: (carry, x) => [newCarry, null]
// nullYs is null, not an empty array - no memory allocated for outputs
// Useful when you only need the final carry (e.g., Mandelbrot iteration count)
```

**Body function — managed references:**

Scan _manages_ the lifecycle of `carry` and `x` — don't dispose them manually. However, standard
consumption rules apply inside the body (same as regular functions):

- **Single use:** `np.add(carry, x)` — no `.ref` needed, the operation consumes them.
- **Multiple uses:** Use `.ref` to keep alive for additional uses.

```ts
// ✓ Works: .ref keeps carry alive, then bare carry consumed in return
const step = (carry, x) => {
  const newCarry = np.add(carry.ref, x); // .ref: we'll use carry again
  return [newCarry, carry]; // carry consumed here
};

// ✗ Fails: can't use carry in TWO separate operations after .ref
const step = (carry, x) => {
  const a = np.add(carry.ref, x); // first operation
  const b = np.add(a, carry); // ERROR: second operation on carry
  return [b, a.ref];
};
```

**Workaround for complex bodies:** Use pytree carries so each field can be `.ref`'d independently:

```ts
const f = (carry: { a: np.Array; b: np.Array }, x) => {
  // carry.a and carry.b are separate arrays with independent refcounts
  const newA = np.add(carry.a.ref, carry.b.ref);
  const newB = np.add(carry.b, np.array([1.0])); // carry.b consumed (last use)
  return [{ a: newA, b: newB }, carry.a]; // carry.a consumed (last use)
};
```

**Outputs — caller owns:**

```ts
const [finalCarry, stackedYs] = lax.scan(f, init, xs);
// Caller owns these — dispose when done:
finalCarry.dispose();
stackedYs.dispose(); // or skip if Y=null
```

**Common patterns:**

| Pattern      | Code                                   | Notes                           |
| ------------ | -------------------------------------- | ------------------------------- |
| Simple body  | `return [newCarry, y]`                 | Two distinct arrays             |
| Passthrough  | `return [newCarry.ref, newCarry]`      | Same array in both              |
| Pytree carry | `return [{ a: a.ref, b }, { out: a }]` | Mix of refs                     |
| Keep inputs  | `scan(f, init.ref, xs.ref)`            | Don't consume inputs            |
| Carry-only   | `scan(f, init, null, { length: N })`   | No xs allocation (saves memory) |
| No Y output  | `return [newCarry, null]`              | No ys allocation (saves memory) |

---

## Implementation Architecture

### Execution flow

```
lax.scan(f, init, xs, { reverse })
  → Trace f → bodyJaxpr (once)
  → Primitive.Scan(jaxpr, numCarry, numConsts, length, reverse)
  → scanRunner (JS loop or native):
      for i in 0..length:
        idx = reverse ? (length-1-i) : i
        xSlice = xs[idx] via ShapeTracker (zero-copy view)
        [carry, y] = bodyProgram.execute(carry, xSlice)
      return [carry, stack(ys)]
```

The `scanRunner` is a callback passed from `array.ts` to `JitProgram.execute()`. It implements the
fallback path — a JS loop that calls `bodyProgram.execute()` per iteration. The compiled-loop and
preencoded-routine paths bypass `scanRunner` entirely by running the loop in WASM/GPU code.

**Argument layout:**

```
Primitive args:   [...consts, ...initCarry, ...xs]
Body jaxpr input: [...consts, ...carry, ...x_slice]
```

### JIT and scan interaction

Understanding how JIT interacts with scan is crucial for performance:

**Without JIT wrapper:** `lax.scan(f, init, xs)`

```
1. Trace body function f → bodyJaxpr
2. JIT-compile bodyJaxpr → bodyProgram (this ALWAYS happens)
3. Execute via scanRunner callback:
   - compiled-loop/preencoded-routine: Single step runs entire loop
   - fallback: JS loop calling bodyProgram.execute() per iteration
```

**With JIT wrapper:** `jit(() => lax.scan(f, init, xs))()`

```
1. Trace outer function → outerJaxpr containing Primitive.Scan
2. JIT-compile outerJaxpr → outerProgram containing scan step
3. Trace body function f → bodyJaxpr (nested inside scan step compilation)
4. JIT-compile bodyJaxpr → bodyProgram
5. Execute outerProgram:
   - compiled-loop/preencoded-routine: outerProgram runs entire loop in WASM/GPU code
   - fallback: outerProgram has scan step, calls scanRunner with bodyProgram
```

**Key insight:** The body function is **always** JIT-compiled into a `bodyProgram`. The difference
is whether the **outer loop** runs natively (compiled-loop/preencoded-routine) or via JS (fallback).

**Why use `jit()` wrapper:**

- **Enables compiled-loop/preencoded-routine** — Without jit, scan uses `scanRunner` (JS loop). With
  jit, the compiler can emit steps that run entirely in WASM/GPU.
- **Caches compilation** — `jit((xs) => scan(...))` compiles once, runs many times.
- **Captures constants** — Closed-over arrays become constants in the compiled program.

**When to use `jit()` wrapper:**

| Pattern                              | Use case                | Notes                                  |
| ------------------------------------ | ----------------------- | -------------------------------------- |
| `lax.scan(f, init, xs)`              | Simple scans, debugging | Body still JIT-compiled                |
| `jit(() => lax.scan(...))()`         | Performance-critical    | Enables native loop execution          |
| `jit((xs) => lax.scan(f, init, xs))` | Reusable function       | Cached compilation, constants captured |

**Transform compositions:**

| Composition       | Works? | Notes                                       |
| ----------------- | ------ | ------------------------------------------- |
| `jit(scan(...))`  | ✅     | JIT wraps scan, body is JIT-compiled        |
| `scan(jit(...))`  | ⚠️     | JIT inside body adds overhead per iteration |
| `grad(jit(scan))` | ❌     | Not supported — jit captures forward pass   |
| `jit(grad(scan))` | ✅     | Correct pattern for gradients               |
| `vmap(jit(scan))` | ✅     | Each batch element runs JIT-compiled scan   |
| `jit(vmap(scan))` | ✅     | Outer JIT, inner vmap, body compiled once   |

**Transform sandwiches:** Compositions like `jit(grad(scan))` where transforms wrap each other. The
test suite verifies these work correctly by comparing against reference implementations.

### How consumption rules are enforced

The scan body's consumption rules work at both trace time and execution time via different
mechanisms:

**Trace time (JaxprTracer):** When `makeJaxpr` traces the body function, `carry` and `x` are
`JaxprTracer` objects with explicit reference counting (`src/frontend/jaxpr.ts:567-590`):

```typescript
class JaxprTracer extends Tracer {
  #rc: number = 1; // Reference count enforced even for tracers

  get ref() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
    this.#rc++;
    return this;
  }
  dispose() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
    this.#rc--;
  }
}
```

And `processPrimitive` consumes all input tracers:

```typescript
const avalsIn = tracers.map((t) => {
  t.dispose(); // Standard consumption!
  return t.aval;
});
```

**Execution time (array.ts):** The scan impl rule creates `.ref` copies internally before calling
`evalJaxpr` (`src/frontend/array.ts:1414-1420`):

```typescript
const jaxprInputs = [
  ...constViews.map((c) => c.ref),
  ...carry.map((c) => c.ref), // Scan creates .ref
  ...xSlice,
];
// evalJaxpr consumes jaxprInputs
```

**Why this matters:** Code that works during tracing (with `jit()`) will also work at execution time
(without `jit()`), and vice versa. A body function that double-consumes a variable will fail during
tracing, catching the bug early rather than silently producing incorrect gradients.

### Debugging scan paths

**Verify expected path with acceptPath:**

```ts
// Throws if actual path is not in accepted list
const [carry, ys] = await lax.scan(f, init, xs, {
  acceptPath: ["compiled-loop", "preencoded-routine"],
});

// Accept only a specific path
await lax.scan(f, init, xs, { acceptPath: "compiled-loop" });

// Discover which path was chosen (always throws, shows path)
await lax.scan(f, init, xs, { acceptPath: [] });
// Error: Scan path debug: chose "compiled-loop"
// For WebGPU fallback, also shows dispatch count:
// Error: Scan path debug: chose "fallback" (2 GPU dispatches per iteration)
```

**Enable debug logging:**

```ts
import { setDebug } from "@jax-js/jax";
setDebug(1); // Shows scan path selection reason
setDebug(2); // Shows shader/WASM code
```

**Common fallback reasons:**

| Reason               | Debug message                                | Fix                           |
| -------------------- | -------------------------------------------- | ----------------------------- |
| Internal buffer deps | "internal buffer dependencies not supported" | Simplify body or use WASM     |
| Carry passthrough    | "carry is passthrough, not supported"        | Ensure kernel produces carry  |
| numCarry ≠ numY      | "numCarry !== numY"                          | Match carry/output counts     |
| Unsupported routine  | "unsupported routine in scan body"           | Use supported routine or WASM |

### JIT step types

| ScanPath | Meaning | Internal Step Types |
| -------- | ------- | ------------------- |

| `fallback` | JS loop with per-iteration dispatch | `scan` |

### Body composition types

Scan bodies are classified by what operations they contain:

| Body Type                | Description                           | Example                          |
| ------------------------ | ------------------------------------- | -------------------------------- |
| **kernel-only**          | Only elementwise/reduction kernels    | `carry + x`, `sum(x)`            |
| **routine body**         | Single routine operation              | `cholesky(x)`, `sort(x)`         |
| **mixed kernel+routine** | Both kernels and routines in one body | `scale * x` then `cholesky(...)` |

**Execution path by body type and backend:**

| Body Type                | WASM          | WebGPU                            |
| ------------------------ | ------------- | --------------------------------- |
| kernel-only (simple)     | compiled-loop | compiled-loop                     |
| kernel-only (with deps¹) | compiled-loop | **fallback**                      |
| routine body (single)    | compiled-loop | preencoded-routine (or fallback²) |
| mixed kernel+routine     | compiled-loop | **fallback** (common in practice) |
| multiple routines        | compiled-loop | **fallback** (e.g., lstsq)        |

¹ "With deps" = internal buffer dependencies between steps, or carry passthrough pattern. ² Sort
uses fallback due to uniform buffer conflict; LU uses fallback due to multi-output (numCarry ≠
numY).

**Why `lax.linalg.triangularSolve` creates a mixed body:**

The high-level `triangularSolve` API handles `leftSide` and `lower` parameters by adding
transpose/flip operations around the primitive routine:

```ts
// What lax.linalg.triangularSolve(L, b, { leftSide: true, lower: true }) compiles to:
b_transposed = moveaxis(b, -2, -1); // Kernel step 1
L_flipped = flip(L, [-2, -1]); // Kernel step 2 (for lower=true)
x = Primitive.TriangularSolve(L_flipped, b_transposed); // Routine step
result = flip(moveaxis(x, -2, -1), [-1]); // Kernel step 3
```

This creates a mixed kernel+routine body with 3+ steps, so WebGPU falls back. WASM compiled-loop
handles it. If performance is critical, consider using WASM backend for linalg-heavy scan bodies.

**Definition: Internal buffer dependencies**

When one kernel step reads from another step's output within the same body:

```ts
// Body with internal deps (WebGPU falls back):
const body = (carry, x) => {
  const Asq = carry.A.mul(carry.A); // Step 1: produces Asq
  const newA = Asq.sub(carry.B); // Step 2: reads Asq (internal dep!)
  return [{ A: newA, B: carry.B }, newA];
};
```

WASM handles this by allocating temporary buffers. WebGPU's shader codegen doesn't support it yet.

**Definition: Carry passthrough**

When an output carry slot directly references the input carry without a kernel producing it:

```ts
// Carry passthrough (WebGPU multi-kernel falls back):
const body = (carry, x) => {
  const newA = carry.A.add(x);
  return [{ A: newA, B: carry.B.ref }, newA]; // B is passthrough!
};
```

WebGPU multi-kernel scan requires every carry output to be produced by a kernel step.

WASM's unified `codegenNativeScanGeneral` handles all body types via compiled-loop. WebGPU has more
constraints and falls back to JS loop for complex patterns.

### Terminology glossary

The documentation uses descriptive terms that map to code constructs:

| Doc Term               | Code Step Type       | Backend      | Description                                |
| ---------------------- | -------------------- | ------------ | ------------------------------------------ |
| **compiled-loop**      | `compiled-loop`      | WASM, WebGPU | Entire scan loop compiled to native code   |
| **preencoded-routine** | `preencoded-routine` | WebGPU       | Routine body with uniform offsets per iter |
| **fallback**           | `scan`               | All          | JS loop calling body program per iteration |

Note: `preencoded-routine` transforms routine shaders to use uniform-based offsets for xs buffers,
then dispatches all iterations with pre-encoded commands. Both `compiled-loop` and
`preencoded-routine` implement the "fast" scan path.

### Compiled-loop routing

The `tryPrepareNativeScan()` dispatcher routes to backend-specific implementations:

- **WebGPU kernel-only** → `tryPrepareWebGPUNativeScan()` → uses `prepareNativeScan()` or
  `prepareNativeScanMulti()`
- **WebGPU routine body** → `tryPreparePreencodedScan()` → uses `preparePreencodedScan()`
- **WASM (kernels + routines)** → `tryPrepareWasmNativeScan()` → uses `prepareNativeScanGeneral()`

### Compiled-loop eligibility

**WASM compiled-loop** (via `tryPrepareWasmNativeScan`):

- All body steps are Kernels or supported Routines
- Constants allowed, reductions allowed
- Any `numCarry`/`numY` combination
- Y outputs can be: carry passthrough, xs passthrough, or internal buffer
- Internal buffer dependencies between steps: **supported**
- Supported routines: Cholesky, Sort, TriangularSolve, LU, Argsort

**WebGPU compiled-loop (single kernel)** (via `prepareNativeScan`):

- Exactly 1 Kernel step (single-output)
- `numCarry === 1` and `numY === 1`
- Constants supported, reverse supported

**WebGPU compiled-loop (multi-kernel)** (via `prepareNativeScanMulti`):

- Multiple Kernel steps (kernel-only body)
- `numCarry === numY` (or `numY === 0`)
- **No internal buffer dependencies** between steps (falls back otherwise)
- **No carry passthrough** (every carry must be produced by a kernel)

**WebGPU preencoded-routine** (via `tryPreparePreencodedScan`):

- Exactly 1 Routine step (single routine body)
- `numCarry === numY` (passthrough pattern)
- Routine must not already use uniforms (excludes Sort)

### WASM compiled-loop details

All WASM scan variants use `codegenNativeScanGeneral` in `src/backend/wasm.ts`:

1. **Pre-analysis** — Build `directWriteMap` deciding which internal buffers can be redirected
2. Import routine functions and helper math functions
3. Allocate WASM locals: iteration counter, data index, element indices
4. **Step 1**: Copy initCarry to carryOut (working buffer) via `memory.copy`
5. **Step 2**: Main scan loop (iter = 0..length):
   - Compute dataIdx (reverse-aware: `length - 1 - iter` or `iter`)
   - **Step 2a**: For each step, execute kernel or call imported routine
   - **Step 2b**: Copy Y outputs to `ysStacked` at iteration offset
   - **Step 2c**: Copy carry outputs to `carryOut` for next iteration
6. Return compiled `WebAssembly.Module`

**Direct-write optimization (pre-analysis phase):**

Before generating any WASM code, `codegenNativeScanGeneral` analyzes the scan body to build a
`directWriteMap: Map<internalIdx, { carryIdx, yIdx? }>`. This maps internal buffer indices to their
redirect targets. The analysis walks expression trees via `AluExp.fold()` to collect carry read
patterns per step.

When a kernel step is eligible for direct-write:

- **Step 2a**: The kernel's store instruction targets `carryOut[carryIdx]` instead of
  `internals[internalIdx]`. If `yIdx` is also set, uses `local.tee` to store the computed value to
  both `carryOut` and `ysStacked[yIdx]` in a single expression evaluation.
- **Step 2b**: The `memory.copy` for this Y output is skipped (already written inline).
- **Step 2c**: The `memory.copy` for this carry output is skipped (already written inline).

For multi-output kernels, each output is analyzed independently — some may use direct-write while
others fall back to internal buffers.

Eligibility conditions (all must be met):

1. Output produced by a Kernel step (not a Routine)
2. Kernel has no reduction (prevents self-overwrite during inner loop)
3. Internal buffer not read by any other step (no data dependencies)
4. Maps to exactly one carry output
5. No Y output is a passthrough from the target carry (passthrough reads OLD carry, but direct-write
   overwrites carry during the kernel loop)
6. No later step reads the target carry as input (later steps should see the carry from the START of
   the iteration, not the partially-overwritten value)

**Why condition 5 matters:**

```ts
// This body has Y = old carry (passthrough):
const step = (carry, x) => {
  const newC = carry.add(x);
  return [newC, carry]; // Y reads OLD carry value
};
```

If we direct-wrote `newC` to `carryOut` during the kernel loop (element by element), then the
passthrough copy `Y = carry` would read a mix of old and new carry values. The passthrough copy in
Step 2b reads from `carryOut`, and at element `i`, elements `0..i-1` would already be overwritten.
This is why direct-write is disabled when any Y output is a passthrough from the target carry.

**Why condition 6 matters:**

```ts
// Multi-step body where step 2 reads the carry that step 1 writes to:
const step = (carry, x) => {
  const a = carry.A.add(x); // Step 1: writes to carry.A
  const b = carry.A.ref.mul(x); // Step 2: reads carry.A (needs OLD value!)
  return [{ A: a, B: b }, null];
};
```

If step 1 direct-wrote to `carryOut.A`, step 2 would read partially-overwritten values instead of
the carry entering the iteration. Direct-write is disabled for `carry.A` in this case.

**Performance impact:**

For small scan bodies (L=1000), eliminating `memory.copy` provides **40-65% speedup**:

| Pattern             | Without direct-write | With direct-write | Speedup |
| ------------------- | -------------------- | ----------------- | ------- |
| Cumsum (scalar)     | ~44M iter/sec        | ~62M iter/sec     | +41%    |
| Elementwise (n=4)   | ~48M iter/sec        | ~78M iter/sec     | +63%    |
| Carry-only (4×4)    | ~40M iter/sec        | ~50M iter/sec     | +25%    |
| Passthrough Y (4×4) | ~35M iter/sec        | ~35M iter/sec     | N/A     |

The elementwise case benefits most because it eliminates both internal→carry AND internal→Y copies.
Carry-only (Y=null) only eliminates the carry copy. Passthrough Y is ineligible (condition 5).

**Y output sources (`YOutputSource` type):**

| Type             | Source                              | Use case                         |
| ---------------- | ----------------------------------- | -------------------------------- |
| `passthrough`    | Copy from carry input               | `return [newC, oldC.ref]`        |
| `xs-passthrough` | Copy from xs slice at current iter  | `return [newC, x.ref]`           |
| `internal`       | Copy from internal buffer (compute) | `return [newC, someComputation]` |

**Carry output sources (`CarryOutputSource` type):**

| Type          | Source                    | Use case                                 |
| ------------- | ------------------------- | ---------------------------------------- |
| `passthrough` | Copy from carry input     | `return [oldC.ref, y]` (carry unchanged) |
| `internal`    | Copy from internal buffer | `return [computation, y]`                |

### WebGPU compiled-loop details

Shader codegen in `nativeScanShaderSource()`:

```wgsl
for (var iter: u32 = 0; iter < length; iter++) {
  var acc: f32 = 0.0;  // reduction identity
  for (var ridx: u32 = 0; ridx < reductionSize; ridx++) {
    acc = acc + /* expression using ridx */;
  }
  carry[gidx] = /* epilogue using acc */;
  ys[iter * carrySize + gidx] = carry[gidx];
}
```

Key insight: Thread `i` only reads/writes `carry[i]` and `xs[:,i]`. No `workgroupBarrier()` needed.

### WebGPU preencoded-routine details (routine body)

For routine bodies, the approach uses pre-encoded dispatches with uniform-based offsets.

**Why uniform-based (not buffer offsets):**

- `minStorageBufferOffsetAlignment` is 256 bytes on most GPUs
- Typical strides (e.g., 120 bytes) fail alignment requirements
- Solution: Bind entire buffers, pass offset as uniform variable

**Implementation:**

The `wrapRoutineForScan` function transforms routine shaders to add offset uniforms:

1. Parse buffer bindings from WGSL source
2. Identify which bindings need offsets using `ScanBindingInfo` mapping:
   - `routineInputJitIds` maps routine input bindings → body jaxpr JitIds
   - Inputs with JitId ≥ numConsts+numCarry are xs (need offsets)
   - Outputs are always carry (ys are filled via copy-after-iteration)
3. Generate `ScanOffsets` struct with offset fields for xs bindings only
4. Transform array accesses to add offset (e.g., `x[idx]` → `x[x_offset + idx]`)

**Dispatch architecture:**

- Ping-pong buffers for carry (iteration n reads from one, writes to other)
- Stacked ys buffers are filled by `copyBufferToBuffer` after each iteration
- Separate uniform bind groups per iteration (dynamic offsets not supported with auto layout)

### Critical implementation patterns

**Pending ops flush:** Scan execution requires flushing pending ops before scan step:

```ts
case "scan": {
  for (const p of pending) { p.prepareSync(); p.submit(); }
  pending.length = 0;
}
```

**IncRef for duplicate slots:** When body outputs contain duplicate slots (passthrough):

```ts
const seenSlots = new Set<Slot>();
const outArrays = bodyOuts.map((slot) => {
  if (seenSlots.has(slot)) backend.incRef(slot);
  else seenSlots.add(slot);
  return new Array({ source: slot, ... });
});
```

---

## Autodiff Support

### JVP (Forward-mode AD)

JVP tracing produces a doubled scan: primals + tangents flow together:

- Body becomes `(carryP, carryT, xP, xT) → (newCarryP, newCarryT, yP, yT)`
- Single scan executes both primal and tangent computation

### VJP/Grad (Reverse-mode AD)

Uses the JVP-transpose pattern for control flow:

```
grad(f)(xs)
  → vjp(f, [xs])
  → linearizeFlatUtil(f, primals)
  → partialEvalFlat (JVP'd scan with doubled args)
  → transpose(jvpResult, cotangents)
  → Scan transpose rule: iterate backward, transpose each step
```

**Key insights:**

1. Forward pass stores √N checkpoint carries by default (or all N if `checkpoint: false`)
2. Backward pass iterates from `length-1` to `0`, recomputing from checkpoints as needed
3. `evalJaxprTransposed` propagates "known" status for residuals

**Gradient checkpointing (`checkpoint` option):**

By default, the backward pass uses √N checkpointing: only O(√N) intermediate carries are stored, and
the rest are recomputed from the nearest checkpoint during the backward pass. This trades ~2×
compute for dramatically reduced memory. Set `checkpoint: false` to store all N carries (O(N)
memory, no recomputation).

```ts
// Default: √N checkpointing is automatic
const loss = (xs) => {
  const [carry, _] = lax.scan(step, init, xs);
  return carry.sum();
};
const dxs = grad(loss)(xs); // O(√N) memory

// Opt out: store all carries
const dxs2 = grad((xs) => {
  const [carry, _] = lax.scan(step, init, xs, { checkpoint: false });
  return carry.sum();
})(xs); // O(N) memory, no recomputation
```

Implementation (in `linearize.ts` transpose rule):

1. **Checkpoint forward pass**: Run forward, save carries every `segmentSize` iterations
2. **Segment-based backward**: For each segment (reverse order):
   - Recompute forward from the segment's checkpoint to recover all segment carries
   - Run transposed body backward through the segment
3. Helper functions `runOneForwardStep` and `runOneBackwardStep` eliminate code duplication

### Vmap (Vectorized Scan)

Each batch element runs an independent scan:

1. Move batch dims: consts/carry → axis 0, xs → axis 1
2. Create vmapped body jaxpr with batch at axis 0
3. Run single scan over batched arrays
4. Move ys batch from axis 1 to axis 0

**Compositions work:** `jit(vmap(scan))` and `vmap(jit(scan))`

**Transform sandwiches tested:** The `test/lax-scan.test.ts` "transform sandwiches" suite verifies
additional compositions: `jit(grad(scan))`, `grad(vmap(scan))`, `vmap(grad(scan))`, and `vmap(scan)`
vs `scan(vmap(body))` equivalence. Note: `grad(jit(scan))` is not supported — use `jit` inside the
grad-wrapped function instead.

---

## Routine System

### Implementation status

| Routine             | Status         | Source                                          | Notes                                          |
| ------------------- | -------------- | ----------------------------------------------- | ---------------------------------------------- |
| **Cholesky**        | ✅ Implemented | `src/backend/wasm/routines/cholesky.ts`         | f32/f64, single/batched                        |
| **TriangularSolve** | ✅ Implemented | `src/backend/wasm/routines/triangular-solve.ts` | Upper/lower triangular, unit/non-unit diagonal |
| **LU**              | ✅ Implemented | `src/backend/wasm/routines/lu.ts`               | Partial pivoting                               |
| **Sort**            | ✅ Implemented | `src/backend/wasm/routines/sort.ts`             | Bottom-up merge sort, NaN-aware                |
| **Argsort**         | ✅ Implemented | `src/backend/wasm/routines/argsort.ts`          | Stable merge sort on indices                   |

Routines are compiled at runtime using wasmblr — no separate build step required.

**Key files:**

| File                                   | Purpose                                |
| -------------------------------------- | -------------------------------------- |
| `src/backend/wasm/wasmblr.ts`          | Low-level WASM bytecode assembler      |
| `src/backend/wasm/wasmblr-hl.ts`       | High-level helper layer (WasmHl class) |
| `src/backend/wasm/routines/*.ts`       | Size-specialized routine codegen       |
| `src/backend/wasm/routine-provider.ts` | 64-entry LRU module cache (by size)    |

### Adding a new routine (checklist)

| Step | File                                   | What to add                                                          |
| ---- | -------------------------------------- | -------------------------------------------------------------------- |
| 1    | `src/backend/wasm/routines/<name>.ts`  | Size-specialized wasmblr implementation (sizes as compile-time args) |
| 2    | `src/backend/wasm/routines/index.ts`   | Export the build function                                            |
| 3    | `src/backend/wasm/routine-provider.ts` | Add builder to `routineBuilders` map with size key generation        |
| 4    | `src/routine.ts`                       | Add to `Routines` enum                                               |
| 5    | `src/frontend/core.ts`                 | Add to `routinePrimitives` map                                       |
| 6    | `src/backend/wasm.ts`                  | Add dispatch case with size params from `ScanRoutineInfo`            |
| 7    | `src/frontend/jit.ts`                  | Add to `supportedRoutines`, populate `ScanRoutineInfo.sizeParams`    |
| 8    | `src/backend/wasm.ts`                  | Add codegen case in `codegenNativeScanGeneral()`                     |
| opt  | `src/routine.ts`                       | Add CPU fallback in `runCpuRoutine()`                                |
| opt  | `src/frontend/jvp.ts`                  | Add JVP rule if autodiff needed                                      |
| opt  | `src/frontend/linearize.ts`            | Add transpose rule if grad needed                                    |

**Size key convention:** Cache keys include dtype and all size dimensions, e.g., `cholesky_f32_4` or
`triangular_solve_f64_8_16_lower_unit`.

### wasmblr routine patterns

Routines are **size-specialized**: matrix dimensions are compile-time constants, enabling loop
unrolling and constant propagation. The `routine-provider.ts` caches compiled modules by a size key
(dtype + dimensions).

```typescript
import { CodeGenerator } from "../wasmblr";
import { WasmHl } from "../wasmblr-hl";

// Size-specialized: n is a compile-time constant, not a runtime parameter
function genRoutine(cg: CodeGenerator, hl: WasmHl, dtype: "f32" | "f64", n: number): number {
  const ty = dtype === "f32" ? cg.f32 : cg.f64;

  // Function signature: (inPtr: i32, outPtr: i32) - no size params at runtime
  return cg.function([cg.i32, cg.i32], [], () => {
    const inPtr = 0; // Function param indices
    const outPtr = 1;

    const i = cg.local.declare(cg.i32);
    const val = cg.local.declare(ty);

    // n is a compile-time constant - WASM compiler can unroll/optimize
    hl.forLoop(i, 0, n, () => {
      hl.load(dtype, inPtr, hl.getExpr(i));
      cg.local.set(val);

      hl.store(dtype, outPtr, hl.getExpr(i), () => {
        cg.local.get(val);
        hl.const(dtype, 2);
        hl.binOp(dtype, "mul");
      });
    });
  });
}

// Builder function called by routine-provider.ts with specific sizes
export function buildRoutineModule(dtype: "f32" | "f64", n: number): Uint8Array<ArrayBuffer> {
  const cg = new CodeGenerator();
  const hl = new WasmHl(cg);
  cg.memory.import("env", "memory");

  const func = genRoutine(cg, hl, dtype, n);
  cg.export(func, "routine"); // Single export, no dtype suffix

  return cg.finish();
}
```

**Key WasmHl helpers:**

- `forLoop(i, start, end, body)` — for loop with expression start/end
- `forLoopDown(i, start, end, body)` — downward for loop
- `forLoopUnrolled(n, body, threshold?)` — fully unrolls small fixed-size loops (default ≤8
  iterations)
- `whileLoop(cond, body)` — while loop with condition callback
- `ifElse(resultType, then, else?)` — conditional with optional else
- `load(dtype, base, indexExpr)` — load from base + index \* elemSize
- `store(dtype, base, indexExpr, valueExpr)` — store to memory
- `index2D(row, cols, col)` — compute row \* cols + col
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

The `simdReductionF32`/`F64` helpers take base pointers (locals containing byte addresses) and
handle both the SIMD main loop and scalar tail automatically. The `k` local is used as the loop
counter (element index, not byte offset).

**When to use SIMD:**

| Matrix Size | f32x4 Speedup | f64x2 Speedup |
| ----------- | ------------- | ------------- |
| n < 32      | ~0.8x (skip)  | ~0.9x (skip)  |
| n = 32      | ~1.1x         | ~1.0x         |
| n = 64      | ~1.7x         | ~1.3x         |
| n = 128     | ~3.0x         | ~1.8x         |
| n = 256     | ~3.8x         | ~1.9x         |

SIMD is automatically selected for Cholesky when `dtype === "f32" && n >= 32`.

### Autodiff of routines (example: Cholesky)

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
WASM. The gradient computation expresses the math in terms of these primitives.

---

## Codegen Architecture

Expression translation and shader generation share common code between regular kernels and scan.

**WASM Backend:**

| Function                               | Role                                                |
| -------------------------------------- | --------------------------------------------------- |
| `translateExpCore()`                   | Shared core handling all `AluOp` cases              |
| `TranslateExpContext` interface        | Callbacks for `getVariable` and `handleGlobalIndex` |
| `translateExp()`                       | Wrapper with bounds-check GlobalIndex               |
| `translateExpWithGeneralScanContext()` | Wrapper with const/carry/xs/internal classification |
| `codegenWasmKernel()`                  | Entry point, dispatches based on `isMultiOutput`    |
| `codegenWasmSinglePath()`              | Single-output kernel (supports reduction)           |
| `codegenWasmMultiPath()`               | Multi-output kernel (no reduction)                  |
| `codegenNativeScanGeneral()`           | Full scan loop codegen with direct-write analysis   |

**WebGPU Backend:**

| Function                        | Role                                                    |
| ------------------------------- | ------------------------------------------------------- |
| `translateAluOpToWgsl()`        | Binary/unary ops, comparisons, casts, ternary           |
| `translateErfToWgsl()`          | Erf/Erfc with f32 precision wrapper                     |
| `gen()` in `pipelineSource`     | CSE (common subexpression elimination) + special cases  |
| `genScanExpressionWithRidx`     | Scan-specific GlobalIndex + inline generation           |
| `createShaderEmitter()`         | Returns `{emit, pushIndent, popIndent, getCode}` helper |
| `nativeScanShaderSource()`      | Wrapper delegating to multi version                     |
| `nativeScanMultiShaderSource()` | Full scan shader implementation                         |

**Backend Interface:**

The `Backend` interface uses unified methods for single and multi-output kernels:

- `prepareKernel()` / `prepareKernelSync()` — each backend checks `kernel.isMultiOutput` internally
- WebGPU expands multi-output kernels into separate shader dispatches
- WebGL throws on multi-output (not supported)

### Multi-output Kernel in Native Scan

The `Kernel` class supports fusing multiple outputs via `Kernel.multi()`. Both regular JIT and scan
body compilation produce multi-output kernels when beneficial. Native scan on both WASM and WebGPU
supports multi-output kernel codegen.

**Example:** Mandelbrot body with 3 carry outputs compiles to 2-3 multi-output kernel steps instead
of 6 separate kernel steps, and runs via native scan on both WASM and WebGPU.

See test: `test/lax-scan.test.ts` → "Scan body multi-output: uses native scan with multi-output
kernel"

---

## Known Limitations & Future Work

### Current limitations

| Limitation                            | Workaround                           | Backend |
| ------------------------------------- | ------------------------------------ | ------- |
| `numCarry ≠ numY` on WebGPU           | Falls back to JS loop                | WebGPU  |
| WebGPU internal buffer deps in scan   | Falls back to JS loop                | WebGPU  |
| Mixed kernel+routine bodies on WebGPU | Falls back to JS loop                | WebGPU  |
| `grad(scan)` ~2× compute overhead     | Use `{ checkpoint: false }` for O(N) | All     |
| Sort in scan body on WebGPU           | Uses JS loop (uniforms)              | WebGPU  |
| Mixed-dtype carries on WebGPU         | Use WASM backend or same-dtype carry | WebGPU  |

**WebGPU preencoded-routine requirements:** WebGPU can only use `preencoded-routine` for scan bodies
that are:

1. **Exactly one routine** (no kernels before or after)
2. **numCarry === numY** (passthrough pattern)
3. **Routine doesn't use uniforms** (excludes Sort)

In practice, this means only simple Cholesky-passthrough patterns like:

```ts
const step = (carry, x) => {
  const L = lax.linalg.cholesky(x);
  return [L.ref, L]; // L is both carry and y
};
```

**Why most linalg patterns fall back on WebGPU:**

- **TriangularSolve**: The `lax.linalg.triangularSolve` API handles `leftSide`/`lower` via transpose
  operations (kernels), so the body has multiple steps.
- **LU**: Returns `[lu, pivots, permutation]` — three outputs, so numCarry ≠ numY.
- **lstsq/solve**: Combines Cholesky + TriangularSolve + TriangularSolve — multiple routines.
- **Kalman filters**: Mix matmul (kernel) + routines in one body.

**This is a minor limitation** because:

1. WASM `compiled-loop` handles all these cases natively via imports
2. Complex linalg patterns (Kalman, Newton, etc.) fall back regardless
3. WebGPU fallback still keeps data on GPU — the overhead is command encoding, not data transfer

**Note on Sort in scan body:** Sort already uses a uniform buffer for its configuration, which
conflicts with the scan offset uniform.

**Mixed-dtype carry on WebGPU:** The `nativeScanMultiShaderSource()` shader generator uses
`steps[0].kernel.dtype` for all buffer bindings. If a scan body has mixed dtypes (e.g., f32 carry +
i32 counter), the shader would produce incorrect results. WASM compiled-loop handles mixed dtypes
correctly since each buffer is typed independently.

**Length-0 scans:** Supported. Returns `(init, empty_ys)` matching JAX behavior.

### Code quality notes

These are not bugs but areas where the implementation uses pragmatic shortcuts that future
contributors should be aware of:

- **JVP detection heuristic:** `linearize.ts` uses `numCarry % 2 === 0 && numY % 2 === 0` to detect
  JVP-transformed scans during partial evaluation and transposition. This works because JVP always
  doubles carries/outputs, and this code only runs during autodiff. However, it could theoretically
  misclassify a user scan with even counts. Consider adding an explicit `isJvpTransformed` flag to
  `ScanParams` if this causes issues.

- **`_yTreedef` side-channel:** In `lax-scan.ts`, the Y treedef is stashed on the `flatF` function
  object via `(flatF as any)._yTreedef = yTreedef`. This is invisible to TypeScript and could be
  replaced with a closure variable.

- **Fallback Y stacking with `preallocateY`:** The `scanRunner` in `array.ts` supports two Y
  stacking strategies. Without `preallocateY`, it accumulates Y slices and stacks them via chunked
  `coreConcatenate` (groups of 6). With `preallocateY: true`, it preallocates the final stacked-Y
  buffer and writes each iteration's Y directly using `copyBufferToBuffer` (when 4-byte aligned) or
  the WGSL copy shader `COPY_SHADER_CODE` (when unaligned, e.g., f16 arrays with odd element
  counts). The WGSL copy shader indexes by destination word to avoid read-modify-write races between
  threads. See `test/scan-preallocate.test.ts` for edge-case coverage (duplicate-slot, passthrough,
  reverse, length-0). The preencoded scan path (`dispatchPreencodedScan`) also uses the WGSL copy
  shader for ys stacking when carry sizes are not 4-byte aligned.

### Future work

| Priority | Feature                         | Notes                                                         |
| -------- | ------------------------------- | ------------------------------------------------------------- |
| Medium   | Mixed-dtype WebGPU scan shader  | Per-binding dtype in `nativeScanMultiShaderSource`            |
| Medium   | Auto-detect preallocateY        | Infer when preallocateY is beneficial without explicit opt-in |
| Low      | `lax.scatter` / `dynamic_slice` | Would enable Jaxpr-based routines                             |

#### Completed: Preallocated Y stacking

The following planned steps are now implemented:

1. ✅ `dynamic_update_slice(dst, src, offset)` primitive and `arr.at(i).set(src)` frontend API.
2. ✅ CPU/WASM fast-path (`copyBufferToBuffer`) and WGSL copy shader fallback.
3. ✅ WebGPU WGSL copy kernel (`COPY_SHADER_CODE`) with pipeline caching.
4. ✅ `scanRunner` preallocateY path — direct-write into preallocated output buffers.
5. ✅ Preencoded scan (`dispatchPreencodedScan`) uses shader copy for unaligned ys stacking.
6. ✅ Tests: passthrough, duplicate-slot, reverse, length-0 (`test/scan-preallocate.test.ts`).

### WASM feature opportunities (assessed Feb 2026)

| Priority | Feature            | Browser risk       | Impact      | Notes                                                                                |
| -------- | ------------------ | ------------------ | ----------- | ------------------------------------------------------------------------------------ |
| Medium   | i64 in wasmblr     | None (MVP)         | Medium-High | Unlocks proper f64 builtins (exp/log/sin/erf) and simplifies Threefry PRNG           |
| Medium   | Relaxed SIMD (FMA) | Safari unsupported | High        | `f32x4.relaxed_madd` for 2× dot-product throughput; needs runtime feature detection  |
| Low      | Threads / atomics  | Needs COOP/COEP    | Very High   | SharedArrayBuffer + Workers for parallel matmul/routines; major architectural change |
| Low      | Sign extension ops | None               | Low         | `i32.extend8_s` etc.; marginal for float-focused workloads                           |

---

## Test Coverage Summary

### Test files

| File                                            | Purpose                                       |
| ----------------------------------------------- | --------------------------------------------- |
| `test/lax-scan.test.ts`                         | Main scan test suite (~3000 lines)            |
| `test/scan-preallocate.test.ts`                 | preallocateY edge cases (dup/passthrough/rev) |
| `test/jit-scan-dlm.test.ts`                     | Kalman filter integration tests               |
| `test/deno/webgpu.test.ts`                      | Headless WebGPU tests via Deno                |
| `test/deno/preencoded-scan.test.ts`             | Preencoded scan integration                   |
| `test/deno/preencoded-scan-integration.test.ts` | Multi-kernel WebGPU scan                      |

### Deno WebGPU test guidelines

**Critical: Avoid creating multiple `GPUDevice` instances**

- **Always reuse jax-js's WebGPU device** instead of calling `navigator.gpu.requestAdapter()` +
  `adapter.requestDevice()`.
- Creating a second `GPUDevice` can destabilize Deno's WebGPU runtime and cause flakiness, memory
  leaks, or segfaults across test files.
- Use the `getJaxJsWebGPUDevice()` helper pattern (see `test/deno/preencoded-scan.test.ts`) to
  access the backend's device.
- **Never call `device.destroy()`** on the shared backend device — let the backend manage its
  lifecycle.

**Import from `dist/` not `src/`**

- Deno tests MUST import from `../../dist/index.js` to share backend instances across test modules.
- Mixed `src/` vs `dist/` imports create separate module graphs with separate backend instances,
  causing leak detection failures.

**Buffer cleanup**

- Track all created `GPUBuffer`s in an array: `const createdBuffers: GPUBuffer[] = []`.
- Destroy them in `finally` blocks: `for (const b of createdBuffers) b.destroy()`.
- Call `await device.queue.onSubmittedWorkDone()` before destroying buffers to ensure GPU work is
  complete.

**Module parallelism**

- Deno test runner supports parallel module execution via `--parallel` flag or `DENO_JOBS`
  environment variable.
- The `test:deno` script chains separate `deno test` commands with `&&`, running each file in its
  own process for proper GPU state isolation.

### Memory leak detection (Deno)

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

### Scan path verification

See [Debugging scan paths](#debugging-scan-paths) for full `acceptPath` usage and examples. Always
use `acceptPath: ["compiled-loop", "preencoded-routine"]` in tests to ensure native compilation
doesn't silently regress to JS fallback.

### Test coverage by category

| Category                    | Backend | Path               | Purpose                          |
| --------------------------- | ------- | ------------------ | -------------------------------- |
| `scan basic`                | CPU     | fallback           | Core correctness                 |
| `native scan`               | WASM    | compiled-loop      | Verify fusion works              |
| `native scan > with consts` | WASM    | compiled-loop      | Constants in body                |
| `matmul in body (routine)`  | WASM    | compiled-loop      | Routine bodies                   |
| `Cholesky in body`          | WebGPU  | preencoded-routine | Preencoded-routine with routines |
| `KNOWN LIMITATIONS`         | WebGPU  | fallback           | Verify graceful fallback         |

Tests under "KNOWN LIMITATIONS" PASS when the limitation exists. If you fix a limitation, the test
will FAIL with instructions to update this documentation.

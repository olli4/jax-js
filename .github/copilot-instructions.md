# GitHub Copilot instructions for jax-js

These notes help AI coding agents be immediately productive. The document has two parts:

1. **Repository Overview** — General jax-js knowledge for any development work
2. **Scan Feature Reference** — `lax.scan` implementation details and backend-specific behavior

---

# Part 1: Repository Overview

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

**Before any commit**, run the full CI validation:

```bash
pnpm build                         # Build the project
pnpm check                         # TypeScript type checking
pnpm lint --max-warnings 0         # ESLint with zero warnings tolerance
pnpm exec playwright install chromium-headless-shell  # (if not already installed)
pnpm test                          # Run all tests
pnpm format:check                  # Verify Prettier formatting (or `pnpm format` to fix)
```

These match the checks in `.github/workflows/ci.yaml`.

### Temporary files

Use `tmp/` in the project root for temporary/scratch files. This directory is gitignored and allows
file operations without manual approval in VS Code.

### Debug logging

**IMPORTANT:** Do NOT use environment variables like `DEBUG=1`. Use the runtime function:

```typescript
import { setDebug } from "@jax-js/jax";
setDebug(1); // Enable debug logging BEFORE any jit compilation
```

| Level | Output                                    |
| ----- | ----------------------------------------- |
| 0     | No debug output (default)                 |
| 1     | JIT compile logs, scan path selection     |
| 2     | Shader code (WGSL/WASM), detailed tracing |
| 3     | Expressions and metadata                  |
| 4     | JIT programs, tuning details              |
| 5     | Most verbose operation traces             |

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

1. **Array creation** — `np.array(...)` allocates a backend `Slot` with refcount = 1.
2. **`.ref` accessor** — increments the Array object's `#rc`; same underlying Slot.
3. **Function call** — passing an Array decrements `#rc` by 1 (ownership transfer).
4. **`.data()` / `.dataSync()`** — reads buffer, then calls `dispose()` internally.
5. **`.dispose()`** — decrements `#rc`; when 0, calls `backend.decRef(slot)`.
6. **Pending ops** — `PendingExecute` holds refs on Slots until `submit()`.

### Backend memory comparison

| Aspect        | Wasm (`src/backend/wasm.ts`)              | WebGPU (`src/backend/webgpu.ts`)                      |
| ------------- | ----------------------------------------- | ----------------------------------------------------- |
| Allocation    | `WasmAllocator` over `WebAssembly.Memory` | `device.createBuffer()` with `GPUBufferUsage.STORAGE` |
| Slot tracking | `Map<Slot, {ptr, size, ref}>`             | `Map<Slot, {buffer, size, ref}>`                      |
| Sync read     | Direct memory view                        | `SyncReader` with staging buffer + `mapAsync`         |
| Dispatch      | Instantiate Wasm module, call exported fn | `commandEncoder.dispatchWorkgroups()`, queue submit   |

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
- Testing utilities: `setScanPathCallback`, `ScanPath` (type)

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

1. **Tracing** – `makeJaxpr(f)` traces a function to produce a `Jaxpr` (IR in ANF form)
2. **Simplification** – `jaxpr.flatten().simplify()` canonicalizes the graph
3. **Graph splitting** – `splitGraphDataflow()` identifies "black nodes" vs fusable ops
4. **Kernel fusion** – Consecutive elementwise ops merge into a single `Kernel` (multi-output if
   needed)
5. **Compilation** – `jitCompile(backend, jaxpr)` emits a `JitProgram` (list of `JitStep`s)
6. **Execution** – `JitProgram.execute(slots)` runs steps, managing memory lifetime

**Multi-output kernel fusion:**

When multiple black nodes have the same size and no reductions, they are batched into a multi-output
`Kernel`. This reduces kernel dispatch overhead for functions with multiple outputs.

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
  nodes when finding used input buffers
- **JIT pending ops before scan**: Flush pending ops before scan step execution

## Known flaky tests

- **LU JVP finite-differences** (`test/lax-linalg.test.ts`): Occasionally fails with precision
  errors at the edge of f32 machine epsilon. Not a bug — inherent to finite-difference verification.
- **Deno WebGPU tests** (`test/deno/`): When running all Deno test files together in a single
  `deno test` invocation, GPU state pollution between files causes memory leak detection failures.
  The `test:deno` script runs each file as a separate process to avoid this.

> ⚠️ **IMPORTANT: Deno WebGPU test isolation** - Due to Deno's module caching and GPU state
> persistence between test files, running all Deno tests together in a single process causes
> spurious memory leak failures. The `test:deno` script runs each test file as a separate Deno
> invocation to ensure proper isolation:
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
2. Update documentation when adding new features or APIs
3. Add/adjust tests exercising `.ref` and `.dispose()` for new behavior
4. Export new public symbols from `src/index.ts`
5. Update `FEATURES.md` for user-visible changes

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
- `requirePath?: ScanPath | ScanPath[]` — Enforce scan path; throws if path doesn't match

**Scan paths (`ScanPath` type):**

- `"fused"` — Loop runs in native code (`native-scan` or `batched-scan` step)
- `"fallback"` — JS loop with per-iteration dispatch (`scan` step)

Use `requirePath: "fused"` in tests to ensure native compilation doesn't regress.

**xs=null and Y=null (jax-js extensions):**

Unlike Python JAX, jax-js supports null inputs and outputs for efficiency:

- **xs=null:** When xs is null, you must provide `length` option. Body receives null as x.
- **Y=null:** Body can return `[newCarry, null]` to skip output stacking entirely.

```ts
// xs=null: carry-only scan with no input arrays
const [carry, ys] = await lax.scan(f, init, null, { length: 100 });
// f: (carry, null) => [newCarry, y]

// Y=null: skip output stacking (saves memory)
const [carry, nullYs] = await lax.scan(f, init, xs);
// f: (carry, x) => [newCarry, null]
// nullYs is null, not an empty array
```

**Use cases:**

- Cumulative sum/product
- RNN/LSTM forward pass
- Kalman filter (forward and backward passes)
- Any sequential state machine

**Key files:**

| File                                 | Role                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| `src/library/lax-scan.ts`            | Public API                                            |
| `src/frontend/core.ts`               | `Primitive.Scan` enum + params type                   |
| `src/frontend/jaxpr.ts`              | Abstract eval rule                                    |
| `src/frontend/array.ts`              | Scan impl rule with `scanRunner` callback             |
| `src/frontend/jit.ts`                | JIT step types: `scan`, `native-scan`, `batched-scan` |
| `src/frontend/linearize.ts`          | JVP + transpose rules for autodiff                    |
| `src/frontend/vmap.ts`               | Scan vmap rule (batches independent scans)            |
| `src/backend/wasm.ts`                | Native scan codegen                                   |
| `src/backend/webgpu.ts`              | Native scan + batched-scan for routines               |
| `src/backend/webgpu/scan-wrapper.ts` | WGSL shader transformer for uniform offsets           |

---

## Feature Status by Backend

### CPU Backend

The CPU backend uses JavaScript-interpreted evaluation. It serves as the reference implementation
for correctness testing.

| Feature / Test             | Status                           | Notes                |
| -------------------------- | -------------------------------- | -------------------- |
| `scan basic`               | [✅ Pass](test/lax-scan.test.ts) |                      |
| `scan with pytree carry`   | [✅ Pass](test/lax-scan.test.ts) |                      |
| `reverse scan`             | [✅ Pass](test/lax-scan.test.ts) |                      |
| `jit + scan`               | [✅ Pass](test/lax-scan.test.ts) |                      |
| `JVP (forward-mode)`       | [✅ Pass](test/lax-scan.test.ts) |                      |
| `VJP (reverse-mode)`       | [✅ Pass](test/lax-scan.test.ts) |                      |
| `vmap`                     | [✅ Pass](test/lax-scan.test.ts) |                      |
| `vmap` > `jit(vmap(scan))` | [✅ Pass](test/lax-scan.test.ts) |                      |
| `vmap` > `vmap(jit(scan))` | [✅ Pass](test/lax-scan.test.ts) |                      |
| `scan over views`          | [✅ Pass](test/lax-scan.test.ts) | sliced/transposed xs |

### WASM Backend

The WASM backend supports **native scan**: the entire scan loop is compiled into a WebAssembly
module, eliminating JS/WASM boundary overhead per iteration.

| Feature / Test                      | Status                           | Notes                        |
| ----------------------------------- | -------------------------------- | ---------------------------- |
| `scan basic`                        | [✅ Pass](test/lax-scan.test.ts) |                              |
| `scan with pytree carry`            | [✅ Pass](test/lax-scan.test.ts) |                              |
| `reverse scan`                      | [✅ Pass](test/lax-scan.test.ts) |                              |
| `jit + scan`                        | [✅ Pass](test/lax-scan.test.ts) |                              |
| `JVP (forward-mode)`                | [✅ Pass](test/lax-scan.test.ts) |                              |
| `VJP (reverse-mode)`                | [✅ Pass](test/lax-scan.test.ts) |                              |
| `vmap`                              | [✅ Pass](test/lax-scan.test.ts) |                              |
| `vmap` > `jit(vmap(scan))`          | [✅ Pass](test/lax-scan.test.ts) |                              |
| `vmap` > `vmap(jit(scan))`          | [✅ Pass](test/lax-scan.test.ts) |                              |
| `scan over views`                   | [✅ Pass](test/lax-scan.test.ts) | sliced/transposed xs         |
| `native scan`                       | [✅ Pass](test/lax-scan.test.ts) |                              |
| `native scan` > `with constants`    | [✅ Pass](test/lax-scan.test.ts) |                              |
| Native scan `reverse=true`          | ✅ Pass                          | all variants support reverse |
| `scan with routine body`            | [✅ Pass](test/lax-scan.test.ts) |                              |
| `routine in scan body`              | [✅ Pass](test/lax-scan.test.ts) | uses native scan via imports |
| `grad` through `scan` with routines | [✅ Pass](test/lax-scan.test.ts) | works via native path        |

**Performance benchmarks:**

- Fallback (JS loop, Kalman filter): ~308 iter/sec
- Native scan (general, Kalman filter): ~1.5M iter/sec

**Scan vs jit(loop) overhead:**

| Matrix Size | Overhead | Notes                                 |
| ----------- | -------- | ------------------------------------- |
| 16×16       | **-73%** | Scan FASTER (fewer JS→WASM crossings) |
| 32×32       | **-21%** | Scan FASTER                           |
| 64×64       | +28%     | Loop faster                           |
| 128×128     | +51%     | Loop faster                           |

Crossover point: ~48×48 matrices.

### WebGPU Backend

The WebGPU backend keeps data on GPU between iterations. Supports **native scan** for elementwise
kernels, **multi-kernel scan** for bodies with multiple independent kernels, and **batched-scan**
for routine bodies (Cholesky, LU, TriangularSolve).

| Feature / Test                          | Status                           | Notes                                    |
| --------------------------------------- | -------------------------------- | ---------------------------------------- |
| `scan basic`                            | [✅ Pass](test/lax-scan.test.ts) | uses native scan on WebGPU               |
| `scan with pytree carry`                | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `reverse scan`                          | [✅ Pass](test/lax-scan.test.ts) | uses native scan with dataIdx            |
| `jit + scan`                            | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `JVP (forward-mode)`                    | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `VJP (reverse-mode)`                    | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `vmap`                                  | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `vmap` > `jit(vmap(scan))`              | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `vmap` > `vmap(jit(scan))`              | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `scan over views`                       | [✅ Pass](test/lax-scan.test.ts) | sliced/transposed xs                     |
| `native scan`                           | [✅ Pass](test/lax-scan.test.ts) | kernel gids reindexed to scan layout     |
| `native scan` > `with reduction`        | [✅ Pass](test/lax-scan.test.ts) | e.g., `carry += sum(x)` or matmul        |
| `native scan` > `with reverse`          | [✅ Pass](test/lax-scan.test.ts) | uses dataIdx like WASM                   |
| `native scan` > `with constants`        | [✅ Pass](test/lax-scan.test.ts) | captured constants bound as storage      |
| `multi-kernel scan`                     | ✅ Pass                          | derives output mapping from body outputs |
| `batched-scan` (Cholesky, LU, TriSolve) | ✅ Pass                          | uses fused path with uniform offsets     |
| `batched-scan` (Sort)                   | ⚠️ Fallback                      | Sort already uses uniforms (conflict)    |

**Note on numCarry ≠ numY:** WebGPU native scan requires `numCarry === numY`. When they differ,
WebGPU falls back to JS loop. WASM's general scan handles this case.

**Tested on:** NVIDIA RTX 4070 Ti SUPER via Deno WebGPU (headless, no X11)

### WebGL Backend

The WebGL backend has **no native scan support**. All scans use the JS fallback path, which executes
the body program per iteration. This works correctly but lacks the native scan optimization.

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

### Why native-scan vs batched-scan vs fallback?

| Approach         | How it works                                  | When used                                        |
| ---------------- | --------------------------------------------- | ------------------------------------------------ |
| **Native-scan**  | Entire scan loop in native code (WASM/shader) | Elementwise kernels (WASM+WebGPU), WASM routines |
| **Batched-scan** | Pre-encode dispatches with uniform offsets    | WebGPU routines (Cholesky, LU, TriangularSolve)  |
| **Fallback**     | JS loop calling body program per iteration    | Unsupported patterns, Sort (uniform conflict)    |

**Rationale:** Native-scan is preferred because:

1. Eliminates JS↔native boundary per iteration (~5000× speedup for WASM)
2. Enables compiler optimizations across iterations
3. Single WASM module instantiation vs N calls

Batched-scan is used for WebGPU routines that can't be inlined into a shader. It transforms routine
shaders to accept per-iteration offsets via uniforms, enabling fused dispatch.

### Why wasmblr for WASM routines?

**Problem:** Hand-writing WASM bytecode is error-prone and unmaintainable.

**Solution:** wasmblr — a custom WASM bytecode assembler with a high-level helper layer (WasmHl).

**Benefits:**

- Runtime compilation (no separate build step)
- Single TypeScript syntax throughout the codebase
- Ergonomic helpers for control flow (`forLoop`, `whileLoop`, `ifElse`) and memory access
- SIMD-ready (v128, i32x4, f32x4 types available)
- Small output (~1KB per routine)
- **Size specialization**: Matrix dimensions baked at compile time enable loop unrolling and
  constant propagation
- **LRU caching**: 64-entry cache in `routine-provider.ts` amortizes compilation cost across calls

**Key files:**

| File                                   | Purpose                                              |
| -------------------------------------- | ---------------------------------------------------- |
| `src/backend/wasm/wasmblr.ts`          | Low-level WASM bytecode assembler                    |
| `src/backend/wasm/wasmblr-hl.ts`       | High-level helper layer (WasmHl class)               |
| `src/backend/wasm/routines/*.ts`       | Size-specialized routine implementations             |
| `src/backend/wasm/routine-provider.ts` | Module caching with 64-entry LRU cache (by size key) |

### Why 3 routine implementations (CPU/WASM/WebGPU)?

| Backend    | Implementation          | Location                         | Algorithm Style            |
| ---------- | ----------------------- | -------------------------------- | -------------------------- |
| **CPU**    | JavaScript (TypedArray) | `src/routine.ts`                 | Sequential (for debugging) |
| **WASM**   | wasmblr (runtime gen)   | `src/backend/wasm/routines/*.ts` | Sequential (optimized)     |
| **WebGPU** | Hand-written WGSL       | `src/backend/webgpu/routines.ts` | Parallel (GPU-optimized)   |

**Why 3 routine implementations (CPU/WASM/WebGPU)?**

1. **CPU backend assumes WASM unavailable** — exists for environments without WebAssembly
2. **WebGPU uses different algorithms** — GPU parallelism requires fundamentally different
   approaches:
   - Sort: Bitonic sort (parallel) vs merge sort (sequential)
   - Cholesky: Column-parallel Cholesky-Crout vs row-by-row Cholesky-Banachiewicz

### Why WASM imports for routines in scan?

**Problem:** How to call routines (Cholesky, Sort, etc.) from inside a compiled scan loop?

**Options considered:**

1. **Duplicate routine code in scan module** — Rejected: code bloat, maintenance burden
2. **Call out to JS** — Rejected: defeats purpose of compiled loop
3. **WASM imports** — Chosen: scan module imports pre-compiled routine functions

**Implementation:**

```typescript
// At codegen time:
cg.importFunction("routines", "cholesky_f32", [Type.i32, Type.i32, Type.i32]);

// At instantiation:
dispatchNativeScanGeneral(module, {
  routines: { cholesky_f32: routineInstance.exports.cholesky_f32 },
});
```

**Result:** Full native performance (~1.5M iter/sec) with zero code duplication.

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

**Body function — borrowed references:**

```ts
const f = (carry, x) => {
  // carry and x are BORROWED — do NOT dispose them
  // Return NEW arrays for newCarry and y
  const result = np.add(carry.ref, x.ref); // .ref because we use them
  return [result, result.ref]; // .ref for dual-use (passthrough pattern)
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
fallback path — a JS loop that calls `bodyProgram.execute()` per iteration. Native paths bypass
`scanRunner` entirely by running the loop in WASM/GPU code.

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
   - Native path (fused): Single native-scan or batched-scan step
   - Fallback path: JS loop calling bodyProgram.execute() per iteration
```

**With JIT wrapper:** `jit(() => lax.scan(f, init, xs))()`

```
1. Trace outer function → outerJaxpr containing Primitive.Scan
2. JIT-compile outerJaxpr → outerProgram containing scan step
3. Trace body function f → bodyJaxpr (nested inside scan step compilation)
4. JIT-compile bodyJaxpr → bodyProgram
5. Execute outerProgram:
   - Native path: outerProgram has native-scan step, runs entire loop in native code
   - Fallback path: outerProgram has scan step, calls scanRunner with bodyProgram
```

**Key insight:** The body function is **always** JIT-compiled into a `bodyProgram`. The difference
is whether the **outer loop** runs natively (native-scan/batched-scan) or via JS (fallback).

**Why use `jit()` wrapper:**

- **Enables native loop execution** — Without jit, scan uses `scanRunner` (JS loop). With jit, the
  compiler can emit `native-scan` or `batched-scan` steps that run entirely in WASM/GPU.
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

### Debugging scan paths

**Check which path was used:**

```ts
import { setScanPathCallback } from "@jax-js/jax";

setScanPathCallback((path, backend) => {
  console.log(`Scan used ${path} path on ${backend}`);
});

// Run your scan...
const [carry, ys] = await lax.scan(f, init, xs);

setScanPathCallback(null); // Cleanup
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

| ScanPath   | Meaning                              | Internal Step Types           |
| ---------- | ------------------------------------ | ----------------------------- |
| `fused`    | Loop runs in native code (fast path) | `native-scan`, `batched-scan` |
| `fallback` | JS loop with per-iteration dispatch  | `scan`                        |

### Body composition types

Scan bodies are classified by what operations they contain:

| Body Type                | Description                           | Example                          |
| ------------------------ | ------------------------------------- | -------------------------------- |
| **kernel-only**          | Only elementwise/reduction kernels    | `carry + x`, `sum(x)`            |
| **routine body**         | Single routine operation              | `cholesky(x)`, `sort(x)`         |
| **mixed kernel+routine** | Both kernels and routines in one body | `scale * x` then `cholesky(...)` |

**Execution path by body type and backend:**

| Body Type                | WASM        | WebGPU                      |
| ------------------------ | ----------- | --------------------------- |
| kernel-only (simple)     | native-scan | native-scan                 |
| kernel-only (with deps¹) | native-scan | **fallback**                |
| routine body (single)    | native-scan | batched-scan (or fallback²) |
| mixed kernel+routine     | native-scan | **fallback**                |
| multiple routines        | native-scan | **fallback**                |

¹ "With deps" = internal buffer dependencies between steps, or carry passthrough pattern. ² Sort
uses fallback due to uniform buffer conflict.

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

WASM's unified `codegenNativeScanGeneral` handles all body types natively. WebGPU has more
constraints and falls back to JS loop for complex patterns.

### Terminology glossary

The documentation uses descriptive terms that map to code constructs:

| Doc Term         | Code Step Type | Backend      | Description                                |
| ---------------- | -------------- | ------------ | ------------------------------------------ |
| **native-scan**  | `native-scan`  | WASM, WebGPU | Entire scan loop compiled to native code   |
| **batched-scan** | `batched-scan` | WebGPU       | Routine body with uniform offsets per iter |
| **fallback**     | `scan`         | All          | JS loop calling body program per iteration |

Note: `batched-scan` transforms routine shaders to use uniform-based offsets for xs buffers, then
dispatches all iterations with pre-encoded commands. Both `native-scan` and `batched-scan` implement
the "fused" scan path.

### Native scan routing

The `tryPrepareNativeScan()` dispatcher routes to backend-specific implementations:

- **WebGPU kernel-only** → `tryPrepareWebGPUNativeScan()` → uses `prepareNativeScan()` or
  `prepareNativeScanMulti()`
- **WebGPU routine body** → `tryPrepareBatchedScan()` → uses `prepareBatchedScan()`
- **WASM (kernels + routines)** → `tryPrepareWasmNativeScan()` → uses `prepareNativeScanGeneral()`

### Native scan eligibility

**WASM native-scan** (via `tryPrepareWasmNativeScan`):

- All body steps are Kernels or supported Routines
- Constants allowed, reductions allowed
- Any `numCarry`/`numY` combination
- Y outputs can be: carry passthrough, xs passthrough, or internal buffer
- Internal buffer dependencies between steps: **supported**
- Supported routines: Cholesky, Sort, TriangularSolve, LU, Argsort

**WebGPU native-scan (single kernel)** (via `prepareNativeScan`):

- Exactly 1 Kernel step (single-output)
- `numCarry === 1` and `numY === 1`
- Constants supported, reverse supported

**WebGPU native-scan (multi-kernel)** (via `prepareNativeScanMulti`):

- Multiple Kernel steps (kernel-only body)
- `numCarry === numY` (or `numY === 0`)
- **No internal buffer dependencies** between steps (falls back otherwise)
- **No carry passthrough** (every carry must be produced by a kernel)

**WebGPU batched-scan** (via `tryPrepareBatchedScan`):

- Exactly 1 Routine step (single routine body)
- `numCarry === numY` (passthrough pattern)
- Routine must not already use uniforms (excludes Sort)

### WASM native-scan details

All WASM scan variants use `codegenNativeScanGeneral`:

1. Allocate WASM locals: iteration counter, data index, element indices
2. Allocate internal temporary buffers for intermediate results
3. Copy initCarry to carryOut (working buffer)
4. Generate outer loop (iter = 0..length):
   - Compute dataIdx (reverse-aware)
   - For each step: evaluate kernel or call imported routine
   - Copy Y outputs from source (carry passthrough, xs passthrough, or internal buffer)
   - Copy carry outputs for next iteration
5. Free internal buffers, return WebAssembly.Module

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

### WebGPU native-scan details

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

### WebGPU batched-scan details (routine body)

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

1. Forward pass stores all intermediate carries in `allCarries`
2. Backward pass iterates from `length-1` to `0`
3. `evalJaxprTransposed` propagates "known" status for residuals

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

**WebGPU Backend:**

| Function                        | Role                                                    |
| ------------------------------- | ------------------------------------------------------- |
| `translateAluOpToWgsl()`        | Binary/unary ops, comparisons, casts, ternary           |
| `translateErfToWgsl()`          | Erf/Erfc with f32 precision wrapper                     |
| `gen()` in `pipelineSource`     | CSE + special cases (inverseSqrt, NaN, Threefry)        |
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
of 6 separate kernel steps, and runs via the fused native scan path on both WASM and WebGPU.

See test: `test/lax-scan.test.ts` → "Scan body multi-output: uses native scan with multi-output
kernel"

---

## Known Limitations & Future Work

### Current limitations

| Limitation                          | Workaround                | Backend |
| ----------------------------------- | ------------------------- | ------- |
| `numCarry ≠ numY` on WebGPU         | Falls back to JS loop     | WebGPU  |
| WebGPU internal buffer deps in scan | Falls back to JS loop     | WebGPU  |
| `grad(scan)` memory O(N)            | None (stores all carries) | All     |
| Sort in scan body on WebGPU         | Uses JS loop (uniforms)   | WebGPU  |

**Note on WebGPU internal buffer dependencies:** When a scan body has steps that depend on each
other (e.g., Mandelbrot: `Asq = A*A` then `newA = Asq - Bsq + X`), WebGPU falls back to JS loop
because its shader codegen doesn't support intermediate buffers between steps. WASM handles this
case natively.

**Note on Sort in scan body:** Sort already uses a uniform buffer for its configuration, which
conflicts with the scan offset uniform. This causes Sort-in-scan to fall back to JS loop on WebGPU.
Cholesky, LU, and TriangularSolve now use the fused batched-scan path.

### Future work

| Priority | Feature                              | Notes                                              |
| -------- | ------------------------------------ | -------------------------------------------------- |
| High     | Sqrt(N) checkpointing for grad(scan) | Reduce memory from O(N) to O(√N) with 2× recompute |
| Low      | `lax.scatter` / `dynamic_slice`      | Would enable Jaxpr-based routines                  |

---

## Test Coverage Summary

### Test files

| File                                         | Purpose                            |
| -------------------------------------------- | ---------------------------------- |
| `test/lax-scan.test.ts`                      | Main scan test suite (~2000 lines) |
| `test/jit-scan-dlm.test.ts`                  | Kalman filter integration tests    |
| `test/deno/webgpu.test.ts`                   | Headless WebGPU tests via Deno     |
| `test/deno/batched-scan.test.ts`             | Batched scan integration           |
| `test/deno/batched-scan-integration.test.ts` | Multi-kernel WebGPU scan           |

### Deno WebGPU test guidelines

**Critical: Avoid creating multiple `GPUDevice` instances**

- **Always reuse jax-js's WebGPU device** instead of calling `navigator.gpu.requestAdapter()` +
  `adapter.requestDevice()`.
- Creating a second `GPUDevice` can destabilize Deno's WebGPU runtime and cause flakiness, memory
  leaks, or segfaults across test files.
- Use the `getJaxJsWebGPUDevice()` helper pattern (see `test/deno/batched-scan.test.ts`) to access
  the backend's device.
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
- Current `test:deno` script uses `DENO_JOBS=1` to run test files sequentially, reducing GPU-related
  flakiness.

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

### Scan path tracking

```ts
import { setScanPathCallback, type ScanPath } from "../../dist/index.js";

function trackScanPaths() {
  const paths: { path: ScanPath; backend: string }[] = [];
  setScanPathCallback((path, backend) => paths.push({ path, backend }));
  return {
    paths,
    cleanup: () => setScanPathCallback(null),
    expectPath: (expected: ScanPath) => {
      if (!paths.some((p) => p.path === expected)) {
        throw new Error(`Expected ${expected}, got: ${paths.map((p) => p.path)}`);
      }
    },
  };
}
```

**Why this matters:** Tests named "native scan" may silently fall back to JS loop. Use
`requirePath: "fused"` to ensure tests fail if fusion regresses.

### Test coverage by category

| Category                    | Backend | Path     | Purpose                    |
| --------------------------- | ------- | -------- | -------------------------- |
| `scan basic`                | CPU     | fallback | Core correctness           |
| `native scan`               | WASM    | fused    | Verify fusion works        |
| `native scan > with consts` | WASM    | fused    | Constants in body          |
| `matmul in body (routine)`  | WASM    | fused    | Routine bodies             |
| `Cholesky in body`          | WebGPU  | fused    | Batched-scan with routines |
| `KNOWN LIMITATIONS`         | WebGPU  | fallback | Verify graceful fallback   |

Tests under "KNOWN LIMITATIONS" PASS when the limitation exists. If you fix a limitation, the test
will FAIL with instructions to update this documentation.

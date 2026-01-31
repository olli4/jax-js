# GitHub Copilot instructions for jax-js

These notes help AI coding agents be immediately productive: what to read, how to run the project, and non-obvious conventions.

## Architecture overview
- **Core library** (`@jax-js/jax`, root `src/`): array API, autodiff (`grad`, `jvp`, `vjp`), JIT compilation, device placement.
  - Frontend modules in `src/frontend/`: `array.ts` (Array class), `jit.ts` (kernel fusion), `jvp.ts`/`linearize.ts` (forward/reverse AD), `vmap.ts` (vectorization), `convolution.ts`.
  - Library namespaces in `src/library/`: `numpy.ts`, `lax.ts`, `nn.ts`, `random.ts`, `scipy-special.ts`, `numpy-linalg.ts`, `numpy-fft.ts`.
- **Backends** (`src/backend/`): `cpu.ts` (debug only), `wasm.ts` + `wasm/`, `webgl.ts` + `webgl/`, `webgpu.ts` + `webgpu/` (ML compiler & shader codegen).
- **Aux packages**: `packages/loaders` (safetensors, OPFS cache, BPE tokenizers), `packages/onnx` (ONNX model → native ops), `packages/optax` (optimizers).
- **Website & demos**: `website/` — live examples that double as integration tests.

## Developer workflows
```bash
pnpm install                       # requires pnpm ≥ 10
pnpm run build                     # tsdown → dist/*.js, dist/*.d.ts
pnpm run build:watch               # watch mode
pnpm exec playwright install       # one-time: install Chromium for WebGPU tests
pnpm test                          # Vitest + Playwright (browser + node)
pnpm test                          # Vitest + Playwright (browser + node)
pnpm test test/conv.test.ts        # single file
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm -C website dev                # local dev server for demos
```

### Temporary files
Use `tmp/` in the project root for temporary/scratch files instead of `/tmp`. This directory is gitignored and allows file operations without manual approval in VS Code. Create it if needed: `mkdir -p tmp`.

### WebGPU testing on headless servers
For GPU tests on a headless dev server (no display), Chrome requires specific flags:
```bash
google-chrome --headless=new --use-angle=vulkan --enable-features=Vulkan \
  --disable-vulkan-surface --enable-unsafe-webgpu --disable-software-rasterizer
```

**Prerequisites:**
- NVIDIA Vulkan ICD: `sudo apt install libnvidia-gl-<version>` (e.g., `libnvidia-gl-565`)
- Verify Vulkan: `vulkaninfo --summary` should show your GPU

**Known limitation:** Chrome's headless mode may not expose `navigator.gpu` on some systems even with correct Vulkan setup. In that case, WebGPU tests will be skipped; cpu and wasm backends still run. To test WebGPU, use a machine with a display or run headed Chrome via xvfb:
```bash
sudo apt-get install -y xvfb
xvfb-run -a pnpm test run
```

**Alternative: Deno for headless hardware WebGPU**
Deno's WebGPU implementation (based on wgpu-rs) works headless without X11:
```bash
# Install Deno
curl -fsSL https://deno.land/install.sh | sh

# Verify hardware GPU access (no display required)
deno eval --unstable-webgpu 'const a = await navigator.gpu.requestAdapter(); console.log(a?.info)'
# Should print: { description: "NVIDIA GeForce RTX 4070 Ti SUPER", ... }

# Run Deno-based tests
pnpm run test:deno                # runs test/deno/*.test.ts
```

**Deno WebGPU workaround:** Deno's `createComputePipelineAsync` has a WebIDL binding bug where the `compute` field is not recognized. jax-js automatically uses the synchronous `createComputePipeline` when running in Deno, enabling full WebGPU support on hardware GPUs headlessly.

## Reference-counting & ownership (critical)
Function arguments are **consumed by default** (refcount −1). Reuse an array by accessing `.ref` (refcount +1). The `.data()` method also **consumes the array** after reading.
```ts
// BAD: consumes x twice
function foo(x) { return x.add(x.mul(y)); }

// GOOD: .ref keeps x alive for the second use
function foo(x) { return x.ref.add(x.mul(y)); }

// .data() consumes - don't dispose after!
const result = await arr.data();  // arr is now consumed
// arr.dispose(); // WRONG - would double-free
```
Canonical examples: `test/refcount.test.ts`, `test/conv.test.ts`, `test/deno/webgpu.test.ts`.

### Memory lifecycle in detail
1. **Array creation** — `np.array(...)` allocates a backend `Slot` (buffer) with refcount = 1.
2. **`.ref` accessor** — increments the *Array object's* `#rc`; same underlying Slot.
3. **Function call** — passing an Array decrements `#rc` by 1 (ownership transfer).
4. **`.data()` / `.dataSync()`** — reads the buffer, then calls `dispose()` internally (consumes the array).
5. **`.dispose()`** — decrements `#rc`; when it hits 0:
   - Cancels any pending `PendingExecute` not yet submitted.
   - Calls `backend.decRef(slot)` → frees GPU/Wasm memory when slot refcount = 0.
6. **Pending ops** — `PendingExecute` holds refs on input/output Slots until `submit()` to prevent premature free.

### Backend memory (Wasm vs WebGPU)
| Aspect | Wasm (`src/backend/wasm.ts`) | WebGPU (`src/backend/webgpu.ts`) |
|--------|------------------------------|----------------------------------|
| Allocation | `WasmAllocator` over `WebAssembly.Memory` | `device.createBuffer()` with `GPUBufferUsage.STORAGE` |
| Slot tracking | `Map<Slot, {ptr, size, ref}>` | `Map<Slot, {buffer, size, ref}>` |
| Sync read | Direct memory view | `SyncReader` with staging buffer + `mapAsync` |
| Dispatch | Instantiate Wasm module, call exported fn | `commandEncoder.dispatchWorkgroups()`, queue submit |

- **cpu** backend (`src/backend/cpu.ts`) is JS-interpreted, no native buffers — for debugging only.
- **webgl** backend uses textures as storage; less optimized than WebGPU.

### Autodiff (grad/vjp) and ownership
- `grad(f)` internally calls `vjp(f, primals)`, producing a *linear* backward function.
- The backward function **captures** references to intermediate arrays from the forward pass (stored in a `ClosedJaxpr`).
- Calling `dispose()` on the returned `OwnedFunction` (e.g., `vjpFn.dispose()`) releases those intermediates.
- **Pattern**: always call `.dispose()` on `linearize`/`vjp` results when finished, or memory leaks on GPU.

```ts
const [y, vjpFn] = vjp(f, [x]);
const dx = vjpFn(dy);
vjpFn.dispose();  // free captured forward-pass intermediates
```

## Exports & public API
All public symbols must be exported from `src/index.ts`. Key exports:
- Transformations: `jit`, `grad`, `valueAndGrad`, `jvp`, `vjp`, `vmap`, `jacfwd`, `jacrev`, `hessian`, `linearize`, `makeJaxpr`
- Device control: `init`, `defaultDevice`, `devicePut`, `blockUntilReady`, `devices`
- Namespaces: `numpy` (aliased `np`), `lax`, `nn`, `random`, `scipySpecial`, `tree`

## Extending the codebase
| Area | Key files | Notes |
|------|-----------|-------|
| New numpy/lax op | `src/library/{numpy,lax}.ts` | Follow existing function signatures; add to exports if public. |
| Backend kernel | `src/backend/webgpu/builtins.ts`, shader templates | Mirror existing patterns; test on Chromium via Playwright. |
| Loader/tokenizer | `packages/loaders/src/` | See `safetensors.ts`, `tokenizers.ts`. |
| ONNX op | `packages/onnx/src/ops.ts` | Implement lowering; wire in `index.ts`. |

## Testing
- Tests run in a **headless Chromium** (Playwright) to cover WebGPU.
- Always exercise `.ref` / `.dispose()` semantics in new tests.
- Use `website/src/routes/repl/*` and `website/src/routes/mobileclip/*` as integration smoke tests.

### GPU device permissions (Linux)
To run WebGPU tests on Linux, the user must have access to GPU render devices:
```bash
# Check current groups
id

# Add user to render and video groups (requires sudo)
sudo usermod -a -G render,video $USER

# Log out and back in for changes to take effect
# Or use `sg render -c 'command'` to run a single command with the new group
```
The render devices (`/dev/dri/renderD*`) are owned by the `render` group.

## Common pitfalls
- Forgetting `.ref` → double-consume → `ReferenceError` in tests.
- Exporting a symbol from a library file but not from `src/index.ts` → missing from published types.
- Changing WebGPU shaders without running browser tests → silent breakage.

## JIT compiler internals
The JIT system lives in `src/frontend/jit.ts` and `src/frontend/jaxpr.ts`.

**Pipeline:**
1. **Tracing** – `makeJaxpr(f)` traces a function to produce a `Jaxpr` (JAX expression), an IR of primitives in ANF form.
2. **Simplification** – `jaxpr.flatten().simplify()` canonicalizes the graph.
3. **Graph splitting** – `splitGraphDataflow()` identifies "black nodes" (ops that must dispatch immediately, e.g. routines/reductions) vs fusable elementwise ops.
4. **Kernel fusion** – Consecutive elementwise ALU ops are merged into a single `Kernel` via lazy `AluExp` composition.
5. **Compilation** – `jitCompile(backend, jaxpr)` emits a `JitProgram` — a list of `JitStep`s (malloc, execute, free, scan).
6. **Execution** – `JitProgram.execute(slots)` runs steps on the backend, managing memory lifetime.

**Key types:**
| Type | File | Role |
|------|------|------|
| `Jaxpr`, `JaxprEqn`, `Var`, `Lit` | `jaxpr.ts` | IR nodes and bindings |
| `JitProgram`, `JitStep` | `jit.ts` | Compiled program + step types |
| `Kernel` | `alu.ts` | Fused elementwise kernel descriptor |
| `Routine` | `routine.ts` | Backend-specific op (matmul, conv, etc.) |

**Adding a new primitive:**
1. Declare in `Primitive` enum (`src/frontend/core.ts`).
2. Add tracing rule in `implRules` / `jvpRules` / `transposeRules` as needed.
3. If it's fusable elementwise, add an ALU lowering in `jit.ts` switch.
4. If it needs a dedicated kernel (routine), register in `routinePrimitives` and implement in `src/backend/*`.

## PR checklist
1. Add/adjust tests exercising `.ref` and `.dispose()` for new behavior.
2. Export new public symbols from `src/index.ts`; run `pnpm run check`.
3. Update `FEATURES.md` and website examples for user-visible changes.
4. For backend work, add a micro-benchmark in `bench/` where relevant.

## Scan Primitive (`lax.scan`)
`lax.scan` applies a function over the leading axis of arrays, threading carry state. Located in `src/library/lax-scan.ts`.

**Signature:**
```ts
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs);
// f: (carry, x) => [newCarry, y]
```

**Architecture:**
```
lax.scan(f, init, xs)
  → Trace f → bodyJaxpr (once)
  → Primitive.Scan(jaxpr, numCarry, numConsts, length)
  → scanRunner (JS loop):
      for i in 0..length:
        xSlice = xs[i] via ShapeTracker (zero-copy view)
        [carry, y] = bodyProgram.execute(carry, xSlice)
      return [carry, stack(ys)]
```

**Key files:**
| File | Role |
|------|------|
| `src/frontend/core.ts` | `Primitive.Scan` enum + params type |
| `src/frontend/jaxpr.ts` | Abstract eval rule |
| `src/frontend/array.ts` | Scan impl rule with `scanRunner` callback; pending op submission |
| `src/frontend/jit.ts` | `JitStep.scan` (JS loop), `JitStep["native-scan"]` (WASM/WebGPU), `JitStep["batched-scan"]` (routines) |
| `src/backend/wasm.ts` | `codegenNativeScan()`, `translateExpWithScanContext()` |
| `src/backend/webgpu.ts` | `NativeScanParams`, `BatchedScanParams`, `PreparedBatchedScan` |
| `src/backend/webgpu/scan-wrapper.ts` | WGSL transformer for uniform-based offsets |
| `src/library/lax-scan.ts` | Public API |

**Argument layout:**
```
Primitive args:   [...consts, ...initCarry, ...xs]
Body jaxpr input: [...consts, ...carry, ...x_slice]
```

**Critical patterns:**
1. **Pending ops before sync reads** — call `_realizeSource()` on args, submit pending BEFORE `jp.execute()`.
2. **Slot refcount on extraction** — `backend.incRef(slot)` before disposing the Array wrapper.
3. **WASM instance caching** — `WeakMap<WebAssembly.Module, WebAssembly.Instance>` gives ~2× speedup.
4. **`.ref` for dual-use values** — when a value appears in both carry and output:
   ```ts
   const newSum = carry.sum.ref.add(x.ref);
   return [{ sum: newSum.ref }, newSum];  // .ref keeps it alive
   ```

**Current status (Jan 2026):**
- ✅ CPU + WASM + WebGPU backends tested
- ✅ WebGPU tested via Deno on NVIDIA RTX 4070 Ti SUPER (headless, no X11)
- ✅ **WASM native scan loop** — ~130× faster than JS loop for single-kernel scans
- ✅ **WASM multi-kernel native scan** — ~4,000× faster for multi-kernel bodies (e.g., Kalman filter with 2 matmuls)
- The JS loop runs on CPU, but `bodyProgram.execute()` runs on the active backend (WebGPU/WASM)
- Data stays on GPU between iterations (zero-copy slicing via ShapeTracker)

**WASM native scan — single kernel (implemented):**
Eligible scans (single elementwise body kernel, no reduction, numCarry === numY)
are compiled to a single WASM module with the outer loop inlined. **Constants are now supported** —
captured variables (like scale factors, offsets) are passed as pointers and used without iteration offsets.

Key implementation (single kernel):
| File | Component |
|------|-----------|
| `src/backend/wasm.ts` | `NativeScanParams`, `prepareNativeScan()`, `dispatchNativeScan()`, `codegenNativeScan()` |
| `src/frontend/jit.ts` | `JitStep["native-scan"]`, `tryPrepareNativeScan()` |
| `src/frontend/array.ts` | Submit input pending ops before `jp.execute()` |

**Critical pattern — pending ops submission:**
Native scan reads synchronously from WASM memory. Input arrays created via lazy ops
(e.g., `arange().reshape().astype()`) have pending `PendingExecute` objects that must be
submitted before the native scan dispatches. Two flush points are required:
1. In `array.ts`: submit input args' pending ops before calling `jp.execute()`
2. In `jit.ts`: flush accumulated pending ops before `"native-scan"` step dispatch

**Native scan status:**
| Backend | Status | Constraint |
|---------|--------|------------|
| CPU | JS loop only | — |
| WASM | ✅ Native loop (single) | Single kernel body (elementwise or reduction) |
| WASM | ✅ Native loop (multi) | Multiple independent kernels (e.g., 2 matmuls) |
| WebGPU | ✅ Native loop | Elementwise bodies only (no reductions) |
| WebGPU | ✅ Batched scan | Routine bodies (TriangularSolve, Cholesky, LU) |
| WASM | ❌ Batched scan | Not needed — reductions work in native loop |

**Note:** WASM native scan supports reduction kernels (e.g., matmul = Mul→Sum).
Kalman filter with matrix ops runs entirely in WASM with inlined loop.

### WASM Multi-Kernel Native Scan (Jan 2026)

When a scan body compiles to **multiple independent kernels** (e.g., Kalman filter with 2 matmuls),
the single-kernel native scan path fails. Multi-kernel native scan compiles all kernels into
a single WASM module with the outer loop inlined.

**Performance:** ~4,000× faster than JS loop (8.5M iterations/sec vs 2K iterations/sec for 2×2 matmuls).

**Architecture:**
```
tryPrepareNativeScanMulti():
  1. Detect: body has N execute steps, each with a Kernel (not Routine)
  2. Verify: each kernel writes exactly one carry output (1:1 mapping)
  3. Extract: kernel expressions, input slot mappings, output sizes
  4. Return: NativeScanMultiParams with steps[] describing each kernel

codegenNativeScanMulti():
  1. Allocate WASM locals: iteration counter, element index, slot pointers
  2. Generate outer loop (i = 0..length):
     - Compute xs/ys offsets for iteration i
     - For each kernel step:
       - inner loop (elem = 0..outputSize):
         - Compute each input value (carry, xs, consts)
         - Evaluate kernel ALU expression
         - Store to carry output buffer
  3. Return WebAssembly.Module with inlined loop
```

**Key types:**
| Type | File | Role |
|------|------|------|
| `NativeScanMultiParams` | `wasm.ts` | Multi-kernel scan parameters |
| `NativeScanStep` | `wasm.ts` | Per-kernel step descriptor |
| `tryPrepareNativeScanMulti()` | `jit.ts` | Detection and preparation |
| `codegenNativeScanMulti()` | `wasm.ts` | WASM bytecode generation |
| `prepareNativeScanMulti()` | `wasm.ts` | Backend method |

**NativeScanStep interface:**
```ts
interface NativeScanStep {
  kernel: Kernel;           // The ALU kernel to execute
  inputs: number[];         // Jaxpr input indices for this kernel
  outputCarryIdx: number;   // Which carry slot this kernel writes (0-based)
  outputSize: number;       // Number of elements in output
}
```

**Eligibility constraints:**
- WASM backend only
- Body must compile to multiple execute steps (single-kernel handled separately)
- Each execute step must be a Kernel (not Routine)
- Each kernel must write exactly one carry output
- `numCarry === numY` (carry and output are the same values)
- Number of kernels must equal numCarry

**Input classification in multi-kernel:**
Each kernel's inputs are classified by their jaxpr index:
- `idx < numConsts`: constant (no offset, fixed pointer)
- `idx < numConsts + numCarry`: carry input (read from carry buffer)
- `idx >= numConsts + numCarry`: xs input (offset by `iteration * size`)

**Output handling:**
Each kernel writes to a specific carry buffer. The `outputCarryIdx` field determines
which carry slot the kernel updates. Carry buffers are reused across iterations
(no ping-pong needed since kernels are independent).

**WebGPU native scan design:**
For elementwise body kernels (no reductions), each element's scan is completely independent:
- Element 0: `carry[0] → f(carry[0], xs[0,0]) → f(..., xs[1,0]) → ...`
- Element 1: `carry[1] → f(carry[1], xs[0,1]) → f(..., xs[1,1]) → ...`

Since thread `i` only reads/writes `carry[i]` and `xs[:,i]`, there's no cross-element
communication. Multiple workgroups can run independent scan loops without synchronization.

**No barriers needed:** Each thread runs its own for-loop over iterations. No
`workgroupBarrier()` is required because there are no cross-element dependencies.

**Eligibility constraints:**
- Single elementwise kernel body (rejects routines like matmul, conv)
- No reductions in body (rejects sum, max, etc. which have cross-element dependencies)
- ✅ Constants now supported (captured variables like scale factors)
- `numCarry === numY` (carry and output are the same values)

**Kernel gid reindexing (critical):**
When extracting the body kernel for native scan, the kernel's GlobalIndex gids may not match
the jaxpr input order (due to lazy expression building order). In `tryPrepareNativeScan()`,
we reindex the kernel expression using `execStep.inputs` to ensure gids match jaxpr input indices.
This is essential for correct constant/carry/xs slot mapping.

**WebGPU batched scan for routines (verified Jan 2026):**
For routine bodies (triangular solve, cholesky, LU), we use uniform-based offsets.
**Numerically verified** via Deno WebGPU tests on NVIDIA RTX 4070 Ti SUPER.

| File | Component |
|------|-----------|
| `src/backend/webgpu/scan-wrapper.ts` | WGSL shader transformer for uniform-based offsets |
| `src/backend/webgpu/scan-wrapper.test.ts` | Unit tests (8 tests) |
| `src/backend/webgpu.ts` | `PreparedBatchedScan`, `prepareBatchedScan()`, `dispatchBatchedScan()` |
| `src/frontend/jit.ts` | `tryPrepareBatchedScan()`, `JitStep["batched-scan"]` handler |
| `test/deno/batched-scan.test.ts` | End-to-end GPU verification (5 tests) |

**Batched scan eligibility constraints:**
- WebGPU backend only
- Body must be single execute step with a Routine (not Kernel)
- Routine must not already use uniforms (excludes Sort/Argsort)
- No constants in scan body (MVP)
- `numCarry === numY` (carry and output are the same values)

**Note:** Matmul/Dot is NOT a Routine — it's lowered to Mul→Reduce Kernel. Common routines
that qualify: TriangularSolve, Cholesky, LU (but complex bodies with reshapes don't qualify).

**Why uniform-based offsets (not buffer offsets):**
- `minStorageBufferOffsetAlignment` is 256 bytes on most GPUs
- Buffer offset bindings fail for typical data (e.g., 30×4 = 120 bytes stride)
- Solution: Bind entire buffers, add offset as uniform variable in shader
- Shader reads `buffer[offset + idx]` where offset comes from uniform

**Critical: Dynamic uniform buffer offsets require explicit layout:**
Using `layout: "auto"` doesn't work for dynamic offsets. Must create
`GPUBindGroupLayout` with `hasDynamicOffset: true` for the uniform binding.

**Architecture:**
```
prepareBatchedScan():
  1. Wrap each routine shader with uniform offset variables (group 1)
     - Only xs/ys bindings get offsets (not carry/consts)
     - Uses scan signature (numConsts, numCarry, numX, numY) to identify bindings
  2. Create combined uniform buffer with all iteration offsets (aligned)
  3. Return PreparedBatchedScan with wrapped shaders + offset buffer

dispatchBatchedScan():
  1. Create ping-pong buffers for carry state
  2. Create bind groups with full buffer bindings (group 0)
  3. Use dynamic uniform offsets for iteration-specific offsets (group 1)
  4. Encode all iterations in single command buffer
  5. Submit once → zero JS roundtrip overhead
```

**Scan signature layout:**
```
Inputs:  [consts..., carry_in..., x...]
Outputs: [carry_out..., y...]
```
- `consts`: unchanged across iterations (no offset)
- `carry`: ping-pong buffers (no offset)
- `xs/ys`: need per-iteration offset (uniform-based)

## Where to start reading
- Entry & exports: [src/index.ts](src/index.ts)
- Memory model: README.md § "Reference counting", [test/refcount.test.ts](test/refcount.test.ts)
- Backends: [src/backend/webgpu/](src/backend/webgpu/), [src/backend/wasm/](src/backend/wasm/)
- Demos: [website/src/routes/repl/](website/src/routes/repl/), [website/src/routes/mobileclip/](website/src/routes/mobileclip/)
- Deno WebGPU tests: [test/deno/webgpu.test.ts](test/deno/webgpu.test.ts) — headless hardware GPU testing
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
pnpm test -- test/conv.test.ts     # single file
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm -C website dev                # local dev server for demos
```

## Reference-counting & ownership (critical)
Function arguments are **consumed by default** (refcount −1). Reuse an array by accessing `.ref` (refcount +1). Call `.dispose()` when done.
```ts
// BAD: consumes x twice
function foo(x) { return x.add(x.mul(y)); }

// GOOD: .ref keeps x alive for the second use
function foo(x) { return x.ref.add(x.mul(y)); }
```
Canonical examples: `test/refcount.test.ts`, `test/conv.test.ts`.

### Memory lifecycle in detail
1. **Array creation** — `np.array(...)` allocates a backend `Slot` (buffer) with refcount = 1.
2. **`.ref` accessor** — increments the *Array object's* `#rc`; same underlying Slot.
3. **Function call** — passing an Array decrements `#rc` by 1 (ownership transfer).
4. **`.dispose()`** — decrements `#rc`; when it hits 0:
   - Cancels any pending `PendingExecute` not yet submitted.
   - Calls `backend.decRef(slot)` → frees GPU/Wasm memory when slot refcount = 0.
5. **Pending ops** — `PendingExecute` holds refs on input/output Slots until `submit()` to prevent premature free.

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

## Where to start reading
- Entry & exports: [src/index.ts](src/index.ts)
- Memory model: README.md § "Reference counting", [test/refcount.test.ts](test/refcount.test.ts)
- Backends: [src/backend/webgpu/](src/backend/webgpu/), [src/backend/wasm/](src/backend/wasm/)
- Demos: [website/src/routes/repl/](website/src/routes/repl/), [website/src/routes/mobileclip/](website/src/routes/mobileclip/)
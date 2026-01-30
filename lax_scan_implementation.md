# LAX Scan Implementation

## Overview

This document outlines the implementation status and architectural roadmap for `lax.scan()` in jax-js. The `scan` primitive is a fundamental building block for sequential computations in JAX, enabling efficient loops over arrays with automatic differentiation support.

In JAX, `scan` is implemented as a special primitive that:
- Takes a function and applies it sequentially over elements
- Carries state forward through iterations
- Supports automatic differentiation (gradients flow through the loop)
- Uses backend-native loop constructs (`WhileOp` in XLA) for efficiency

## Current Status

### Phase 1: Compile Scan Body Once ✅ **COMPLETE**

**Status:** Fully implemented

The scan body function is now compiled once and reused across all iterations. This eliminates redundant compilation overhead that would otherwise occur on each iteration.

### Current Infrastructure

The current implementation uses a hybrid approach:

- **Frontend (JavaScript):** Loop control flow is managed in JavaScript in `src/frontend/array.ts`
- **Compilation:** The scan body function is compiled once into a `JitProgram` (similar to how `jit()` works)
- **Execution:** Each iteration dispatches the pre-compiled `bodyProgram` to the backend
- **Backend (WebGPU/WASM):** Individual operations within the body execute natively

This is conceptually similar to:

```typescript
// Simplified conceptual model
const bodyProgram = compileBodyFunction(fn); // Compile once

for (let i = 0; i < length; i++) {
  carry = scanRunner.execute(bodyProgram, carry, xs[i]); // Reuse compiled program
}
```

**Architecture components:**
- `scanRunner`: Manages iteration state and dispatch
- `bodyProgram`: Pre-compiled `JitProgram` representing the scan body
- Backend execution: Each iteration requires a separate dispatch

## Roadmap: Backend-Native Loops

### Current Architecture (Baseline)

The current hybrid approach (Option A) has a fundamental limitation:

- **Loop in JavaScript, body in backend:** The loop control flow (`for`, `while`) lives in JavaScript
- **Per-iteration dispatch:** Each iteration requires crossing the JavaScript→Backend boundary
- **O(N) overhead:** With N iterations, we pay N boundary-crossing costs

While Phase 1 eliminated redundant compilation, we still have **O(N) boundary transitions** rather than the ideal **O(1)**.

### Phase 3: Backend-Native Loop Primitive

**Goal:** Move the entire loop construct to the backend, matching JAX's `WhileOp` design.

**Why this is architecturally necessary:**

In JAX/XLA, `scan` compiles to a native `WhileOp` primitive that:
1. **Single dispatch:** The entire loop is dispatched once to the backend as a single operation
2. **Native iteration:** The backend (XLA, TPU, GPU) manages loop control internally
3. **O(1) transitions:** Only one JavaScript→Backend boundary crossing, regardless of iteration count

**Architectural changes required:**

1. **Jaxpr representation:**
   - Add `scan` as a primitive operation in the Jaxpr IR
   - Represent the loop body as a sub-Jaxpr (nested computation)
   - Track loop-carried state and sequence inputs

2. **Backend lowering:**
   - WebGPU: Lower to compute shader with loop constructs
   - WASM: Generate loop-based code in the compiled module
   - Both backends must handle iteration internally

3. **Autodiff support:**
   - Forward-mode (JVP): Propagate tangents through loop iterations
   - Reverse-mode (VJP): Unroll or use custom gradient rules

**Benefits of backend-native loops:**

- **Reduced overhead:** O(1) instead of O(N) boundary crossings
- **Better optimization:** Backend can optimize across iterations (loop unrolling, vectorization)
- **True parity with JAX:** Matches the semantic model of JAX's control flow primitives
- **Enables complex patterns:** Nested scans, scans in JIT, scans in gradients

**Implementation complexity:**

This requires significant architectural work:
- Extending the Jaxpr IR to support sub-computations
- Backend code generation for loop constructs
- Gradient rules for differentiation through loops
- Testing across all backends (WebGPU, WASM, WebGL)

## Architecture Context

### Why Loops Matter

Control flow primitives like `scan`, `while_loop`, and `cond` are essential for:
- Recurrent neural networks (RNNs, LSTMs, GRUs)
- Iterative algorithms (optimization, root finding)
- Sequence processing (time series, text generation)
- Differentiable dynamic programming

Without backend-native loops, these patterns require manual unrolling or inefficient Python/JavaScript loops.

### Integration with Existing Infrastructure

The scan implementation builds on jax-js's existing architecture:

- **JitProgram:** Compiled representation of operations (used for scan body compilation)
- **Jaxpr IR:** Intermediate representation for tracing and transformation
- **Backend abstraction:** Unified interface for WebGPU/WASM/WebGL execution
- **Autodiff system:** Forward and reverse-mode differentiation support

Moving loops to the backend requires extending each of these components to handle nested computations and iteration control.

## Next Steps

1. **Phase 2 (Future):** Implement basic `scan` API with current hybrid approach
2. **Phase 3 (Long-term):** Implement backend-native loop primitive
   - Design Jaxpr extensions for sub-computations
   - Implement backend lowering for WebGPU and WASM
   - Add gradient rules and autodiff support
   - Extend to other control flow primitives (`while_loop`, `cond`)

## Related Work

- **JAX documentation:** [Control flow primitives](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators)
- **XLA WhileOp:** Backend primitive for loops in JAX
- **jax-js architecture:** See `src/frontend/jaxpr.ts` (IR), `src/frontend/jit.ts` (compilation), `src/backend.ts` (execution)

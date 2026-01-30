# LAX Scan Implementation - Architecture Design

## Overview

This document outlines the architectural design and implementation roadmap for `lax.scan()` in jax-js. The `scan` primitive is a fundamental building block for sequential computations in JAX, enabling efficient loops over arrays with automatic differentiation support.

**Note:** This is a design document. The scan functionality is not yet implemented in jax-js.

In JAX, `scan` is implemented as a special primitive that:
- Takes a function and applies it sequentially over elements
- Carries state forward through iterations
- Supports automatic differentiation (gradients flow through the loop)
- Uses backend-native loop constructs (`WhileOp` in XLA) for efficiency

## Design Status

### Phase 1: Compile Scan Body Once ✅ **DESIGN COMPLETE**

**Status:** Design finalized, implementation pending

The planned approach is to compile the scan body function once and reuse it across all iterations. This will eliminate redundant compilation overhead that would otherwise occur on each iteration.

### Planned Infrastructure (Phase 1)

The Phase 1 design uses a hybrid approach:

- **Frontend (JavaScript):** Loop control flow would be managed in JavaScript (similar to patterns in `src/frontend/array.ts`)
- **Compilation:** The scan body function would be compiled once into a `JitProgram` (similar to how `jit()` works in `src/frontend/jit.ts`)
- **Execution:** Each iteration would dispatch the pre-compiled `bodyProgram` to the backend
- **Backend (WebGPU/WASM):** Individual operations within the body would execute natively

This is conceptually similar to:

```typescript
// Simplified conceptual model
const bodyProgram = compileBodyFunction(fn); // Compile once

for (let i = 0; i < length; i++) {
  carry = scanRunner.execute(bodyProgram, carry, xs[i]); // Reuse compiled program
}
```

**Planned architecture components:**
- `scanRunner`: Would manage iteration state and dispatch
- `bodyProgram`: Pre-compiled `JitProgram` representing the scan body
- Backend execution: Each iteration would require a separate dispatch

## Roadmap: Backend-Native Loops

### Phase 1 Design (Baseline)

The Phase 1 hybrid approach (Option A) would have a fundamental limitation:

- **Loop in JavaScript, body in backend:** The loop control flow (`for`, `while`) would live in JavaScript
- **Per-iteration dispatch:** Each iteration would require crossing the JavaScript→Backend boundary
- **O(N) overhead:** With N iterations, we would pay N boundary-crossing costs

While Phase 1 would eliminate redundant compilation, we would still have **O(N) boundary transitions** rather than the ideal **O(1)**.

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
   - WebGL (compatibility): Similar approach with shader-based loops
   - All backends must handle iteration internally

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
- Testing across all backends (WebGPU, WASM, WebGL for compatibility)

## Architecture Context

### Why Loops Matter

Control flow primitives like `scan`, `while_loop`, and `cond` are essential for:
- Recurrent neural networks (RNNs, LSTMs, GRUs)
- Iterative algorithms (optimization, root finding)
- Sequence processing (time series, text generation)
- Differentiable dynamic programming

Without backend-native loops, these patterns require manual unrolling or inefficient JavaScript loops.

### Integration with Existing Infrastructure

A future scan implementation would build on jax-js's existing architecture:

- **JitProgram:** Compiled representation of operations (used for scan body compilation)
- **Jaxpr IR:** Intermediate representation for tracing and transformation
- **Backend abstraction:** Unified interface for WebGPU/WASM/WebGL execution
- **Autodiff system:** Forward and reverse-mode differentiation support

Moving loops to the backend requires extending each of these components to handle nested computations and iteration control.

## Next Steps

1. **Phase 1 (Near-term):** Implement basic `scan` API with hybrid approach
   - JavaScript loop control with compiled body function
   - Reuse `JitProgram` compilation infrastructure
   - Test with simple sequential operations

2. **Phase 3 (Long-term):** Implement backend-native loop primitive
   - Design Jaxpr extensions for sub-computations
   - Implement backend lowering for WebGPU and WASM
   - Add gradient rules and autodiff support
   - Extend to other control flow primitives (`while_loop`, `cond`)

## Related Work

- **JAX documentation:** [Control flow primitives](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators)
- **XLA WhileOp:** Backend primitive for loops in JAX
- **jax-js architecture:** See `src/frontend/jaxpr.ts` (IR), `src/frontend/jit.ts` (compilation), `src/backend.ts` (execution)

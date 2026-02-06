/**
 * Benchmark wasmblr-compiled routines (size-specialized).
 *
 * This validates that the wasmblr-compiled WASM is performant.
 * Run with: pnpm vitest bench bench/routines.bench.ts
 */

import { bench, describe } from "vitest";

import {
  getArgsortModule,
  getCholeskyModule,
  getLUModule,
  getSortModule,
  getTriangularSolveModule,
} from "../src/backend/wasm/routine-provider";

describe("wasmblr routines (size-specialized)", () => {
  // Shared memory for all tests
  const memory = new WebAssembly.Memory({ initial: 16 }); // 1MB

  describe("cholesky", () => {
    const n = 32;
    const batch = 100;

    const module = getCholeskyModule({ n, dtype: "f32" });
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      cholesky: (inPtr: number, outPtr: number) => void;
      cholesky_batched: (inPtr: number, outPtr: number, batch: number) => void;
    };

    const view = new Float32Array(memory.buffer);

    // Generate random positive-definite matrices: A = B @ B^T + I
    for (let b = 0; b < batch; b++) {
      const offset = b * n * n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let sum = i === j ? 1.0 : 0.0; // Start with identity
          for (let k = 0; k < n; k++) {
            const bi = Math.random();
            const bj = Math.random();
            sum += bi * bj;
          }
          view[offset + i * n + j] = sum;
        }
      }
    }

    const inPtr = 0;
    const outPtr = batch * n * n * 4;

    bench(`cholesky ${n}x${n} single`, () => {
      exports.cholesky(inPtr, outPtr);
    });

    bench(`cholesky ${n}x${n} x${batch} batched`, () => {
      exports.cholesky_batched(inPtr, outPtr, batch);
    });
  });

  describe("triangular-solve", () => {
    const n = 32;
    const batchRows = 100;

    const module = getTriangularSolveModule({
      n,
      batchRows,
      dtype: "f32",
      unitDiagonal: false,
      lower: false,
    });
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      triangular_solve: (aPtr: number, bPtr: number, xPtr: number) => void;
      triangular_solve_batched: (
        aPtr: number,
        bPtr: number,
        xPtr: number,
        numBatches: number,
      ) => void;
    };

    const view = new Float32Array(memory.buffer);

    // Create upper-triangular matrix
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        view[i * n + j] = j >= i ? Math.random() + 0.1 : 0;
      }
    }

    // Create batch of RHS vectors
    const aPtr = 0;
    const bPtr = n * n * 4;
    const xPtr = bPtr + batchRows * n * 4;
    for (let b = 0; b < batchRows; b++) {
      for (let i = 0; i < n; i++) {
        view[(bPtr >> 2) + b * n + i] = Math.random();
      }
    }

    // Also build a single-row version for single-bench
    const singleModule = getTriangularSolveModule({
      n,
      batchRows: 1,
      dtype: "f32",
      unitDiagonal: false,
      lower: false,
    });
    const singleInstance = new WebAssembly.Instance(singleModule, {
      env: { memory },
    });
    const singleExports = singleInstance.exports as {
      triangular_solve: (aPtr: number, bPtr: number, xPtr: number) => void;
    };

    bench(`triangular-solve ${n}x${n} single`, () => {
      singleExports.triangular_solve(aPtr, bPtr, xPtr);
    });

    bench(`triangular-solve ${n}x${n} x${batchRows} batched`, () => {
      exports.triangular_solve_batched(aPtr, bPtr, xPtr, 1);
    });
  });

  describe("lu", () => {
    const n = 32;
    const batch = 100;

    const module = getLUModule({ m: n, n, dtype: "f32" });
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      lu: (
        aPtr: number,
        luPtr: number,
        pivPtr: number,
        permPtr: number,
      ) => void;
      lu_batched: (
        aPtr: number,
        luPtr: number,
        pivPtr: number,
        permPtr: number,
        batch: number,
      ) => void;
    };

    const view = new Float32Array(memory.buffer);

    // Fill with random data
    for (let i = 0; i < batch * n * n; i++) {
      view[i] = Math.random();
    }

    const aPtr = 0;
    const luPtr = batch * n * n * 4;
    const pivPtr = luPtr + batch * n * n * 4;
    const permPtr = pivPtr + batch * n * 4;

    bench(`lu ${n}x${n} single`, () => {
      exports.lu(aPtr, luPtr, pivPtr, permPtr);
    });

    bench(`lu ${n}x${n} x${batch} batched`, () => {
      exports.lu_batched(aPtr, luPtr, pivPtr, permPtr, batch);
    });
  });

  describe("sort", () => {
    const n = 1024;
    const batch = 32; // Reduced from 100 to avoid memory pressure

    const module = getSortModule({ n, dtype: "f32" });
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      sort: (dataPtr: number, auxPtr: number) => void;
      sort_batched: (dataPtr: number, auxPtr: number, batch: number) => void;
    };

    const view = new Float32Array(memory.buffer);

    const dataPtr = 0;
    const auxPtr = batch * n * 4;

    // Pre-fill with random data
    for (let i = 0; i < batch * n; i++) {
      view[i] = Math.random();
    }

    bench(`sort n=${n} single`, () => {
      // Refill single array portion before sorting
      for (let i = 0; i < n; i++) {
        view[i] = Math.random();
      }
      exports.sort(dataPtr, auxPtr);
    });

    bench(`sort n=${n} x${batch} batched`, () => {
      // Refill all arrays before sorting
      for (let i = 0; i < batch * n; i++) {
        view[i] = Math.random();
      }
      exports.sort_batched(dataPtr, auxPtr, batch);
    });
  });

  describe("argsort", () => {
    const n = 1024;
    const batch = 32; // Reduced from 100

    const module = getArgsortModule({ n, dtype: "f32" });
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      argsort: (
        dataPtr: number,
        outPtr: number,
        idxPtr: number,
        auxPtr: number,
      ) => void;
      argsort_batched: (
        dataPtr: number,
        outPtr: number,
        idxPtr: number,
        auxPtr: number,
        batch: number,
      ) => void;
    };

    const view = new Float32Array(memory.buffer);

    const dataPtr = 0;
    const outPtr = batch * n * 4;
    const idxPtr = outPtr + batch * n * 4;
    const auxPtr = idxPtr + batch * n * 4;

    // Fill with random data
    for (let i = 0; i < batch * n; i++) {
      view[i] = Math.random();
    }

    bench(`argsort n=${n} single`, () => {
      exports.argsort(dataPtr, outPtr, idxPtr, auxPtr);
    });

    bench(`argsort n=${n} x${batch} batched`, () => {
      exports.argsort_batched(dataPtr, outPtr, idxPtr, auxPtr, batch);
    });
  });
});

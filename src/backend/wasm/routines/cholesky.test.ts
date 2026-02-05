import { expect, suite, test } from "vitest";

import { buildCholeskyModuleSized } from "./cholesky";

suite("wasmblr Cholesky", () => {
  test("cholesky (size-specialized, f32) decomposes 3x3 SPD matrix", async () => {
    const wasmBytes = buildCholeskyModuleSized(3, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(wasmBytes, {
      env: { memory },
    });
    const { cholesky } = instance.exports as {
      cholesky(inPtr: number, outPtr: number): void;
    };

    // SPD matrix: A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    // Expected L: [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    const A = new Float32Array(memory.buffer, 0, 9);
    A.set([4, 12, -16, 12, 37, -43, -16, -43, 98]);

    const L = new Float32Array(memory.buffer, 64, 9); // Output at offset 64

    cholesky(0, 64);

    // Check L (lower triangular)
    expect(L[0]).toBeCloseTo(2, 5); // L[0,0]
    expect(L[1]).toBeCloseTo(0, 5); // L[0,1]
    expect(L[2]).toBeCloseTo(0, 5); // L[0,2]
    expect(L[3]).toBeCloseTo(6, 5); // L[1,0]
    expect(L[4]).toBeCloseTo(1, 5); // L[1,1]
    expect(L[5]).toBeCloseTo(0, 5); // L[1,2]
    expect(L[6]).toBeCloseTo(-8, 5); // L[2,0]
    expect(L[7]).toBeCloseTo(5, 5); // L[2,1]
    expect(L[8]).toBeCloseTo(3, 5); // L[2,2]
  });

  test("cholesky (size-specialized, f64) decomposes 3x3 SPD matrix", async () => {
    const wasmBytes = buildCholeskyModuleSized(3, "f64");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(wasmBytes, {
      env: { memory },
    });
    const { cholesky } = instance.exports as {
      cholesky(inPtr: number, outPtr: number): void;
    };

    const A = new Float64Array(memory.buffer, 0, 9);
    A.set([4, 12, -16, 12, 37, -43, -16, -43, 98]);

    const L = new Float64Array(memory.buffer, 128, 9);

    cholesky(0, 128);

    expect(L[0]).toBeCloseTo(2, 10);
    expect(L[3]).toBeCloseTo(6, 10);
    expect(L[4]).toBeCloseTo(1, 10);
    expect(L[6]).toBeCloseTo(-8, 10);
    expect(L[7]).toBeCloseTo(5, 10);
    expect(L[8]).toBeCloseTo(3, 10);
  });

  test("cholesky_batched (size-specialized, f32) processes multiple matrices", async () => {
    const wasmBytes = buildCholeskyModuleSized(2, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(wasmBytes, {
      env: { memory },
    });
    const { cholesky_batched } = instance.exports as {
      cholesky_batched(inPtr: number, outPtr: number, batch: number): void;
    };

    // Two 2x2 SPD matrices
    // A1 = [[1, 0], [0, 1]] -> L1 = [[1, 0], [0, 1]]
    // A2 = [[4, 2], [2, 2]] -> L2 = [[2, 0], [1, 1]]
    const A = new Float32Array(memory.buffer, 0, 8);
    A.set([1, 0, 0, 1, 4, 2, 2, 2]);

    const L = new Float32Array(memory.buffer, 64, 8);

    cholesky_batched(0, 64, 2);

    // Check L1
    expect(L[0]).toBeCloseTo(1, 5);
    expect(L[1]).toBeCloseTo(0, 5);
    expect(L[2]).toBeCloseTo(0, 5);
    expect(L[3]).toBeCloseTo(1, 5);

    // Check L2
    expect(L[4]).toBeCloseTo(2, 5);
    expect(L[5]).toBeCloseTo(0, 5);
    expect(L[6]).toBeCloseTo(1, 5);
    expect(L[7]).toBeCloseTo(1, 5);
  });

  test("verifies L @ L^T = A", async () => {
    const n = 4;
    const wasmBytes = buildCholeskyModuleSized(n, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(wasmBytes, {
      env: { memory },
    });
    const { cholesky } = instance.exports as {
      cholesky(inPtr: number, outPtr: number): void;
    };

    // Random SPD matrix (computed as B @ B^T + I)
    const A = new Float32Array(memory.buffer, 0, n * n);
    A.set([5, 2, 1, 3, 2, 6, 2, 1, 1, 2, 4, 2, 3, 1, 2, 7]);

    const L = new Float32Array(memory.buffer, 128, n * n);
    cholesky(0, 128);

    // Compute L @ L^T and verify it equals A
    const LLT = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += L[i * n + k] * L[j * n + k];
        }
        LLT[i * n + j] = sum;
      }
    }

    for (let i = 0; i < n * n; i++) {
      expect(LLT[i]).toBeCloseTo(A[i], 4);
    }
  });
});

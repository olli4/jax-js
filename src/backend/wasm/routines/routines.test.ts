/**
 * Tests for all wasmblr-based routines (size-specialized).
 */

import { describe, expect, it } from "vitest";

import { buildArgsortModuleSized } from "./argsort";
import { buildLUModuleSized } from "./lu";
import { buildSortModuleSized } from "./sort";
import { buildTriangularSolveModuleSized } from "./triangular-solve";

describe("wasmblr triangular-solve (size-specialized)", () => {
  it("builds valid wasm module", () => {
    const bytes = buildTriangularSolveModuleSized(2, 1, "f32", false, false);
    expect(bytes.length).toBeGreaterThan(0);
    const mod = new WebAssembly.Module(bytes);
    expect(mod).toBeInstanceOf(WebAssembly.Module);
  });

  it("solve upper triangular 2x2", async () => {
    const bytes = buildTriangularSolveModuleSized(2, 1, "f32", false, false);
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { triangular_solve } = instance.exports as {
      triangular_solve: (a: number, b: number, x: number) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    // A = [[2, 1], [0, 3]] (upper triangular)
    // b = [4, 9]
    // x should be [0.5, 3] since 2*0.5 + 1*3 = 4, 3*3 = 9
    const aPtr = 0;
    const bPtr = 16; // 4 floats for A
    const xPtr = 24; // 2 floats for b

    f32[0] = 2;
    f32[1] = 1;
    f32[2] = 0;
    f32[3] = 3;
    f32[4] = 4;
    f32[5] = 9;

    triangular_solve(aPtr, bPtr, xPtr);

    expect(f32[6]).toBeCloseTo(0.5);
    expect(f32[7]).toBeCloseTo(3);
  });

  it("solve lower triangular 2x2", async () => {
    const bytes = buildTriangularSolveModuleSized(2, 1, "f32", false, true);
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { triangular_solve } = instance.exports as {
      triangular_solve: (a: number, b: number, x: number) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    // A = [[2, 0], [1, 3]] (lower triangular)
    // b = [4, 7]
    // x = [2, 5/3] since 2*2 = 4, 1*2 + 3*(5/3) = 7
    const aPtr = 0;
    const bPtr = 16;
    const xPtr = 24;

    f32[0] = 2;
    f32[1] = 0;
    f32[2] = 1;
    f32[3] = 3;
    f32[4] = 4;
    f32[5] = 7;

    triangular_solve(aPtr, bPtr, xPtr);

    expect(f32[6]).toBeCloseTo(2);
    expect(f32[7]).toBeCloseTo(5 / 3);
  });
});

describe("wasmblr LU (size-specialized)", () => {
  it("builds valid wasm module", () => {
    const bytes = buildLUModuleSized(2, 2, "f32");
    expect(bytes.length).toBeGreaterThan(0);
    const mod = new WebAssembly.Module(bytes);
    expect(mod).toBeInstanceOf(WebAssembly.Module);
  });

  it("decomposes 2x2 matrix", async () => {
    const bytes = buildLUModuleSized(2, 2, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { lu } = instance.exports as {
      lu: (a: number, lu: number, piv: number, perm: number) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    const i32 = new Int32Array(memory.buffer);

    // A = [[1, 2], [3, 4]]
    // After pivoting: P @ A = [[3, 4], [1, 2]]
    // L = [[1, 0], [1/3, 1]], U = [[3, 4], [0, 2/3]]
    const aPtr = 0;
    const luPtr = 16;
    const pivPtr = 32;
    const permPtr = 40;

    f32[0] = 1;
    f32[1] = 2;
    f32[2] = 3;
    f32[3] = 4;

    lu(aPtr, luPtr, pivPtr, permPtr);

    // LU should have U in upper and L factors below
    // With pivoting, first row of LU = [3, 4], second has factor and remainder
    expect(f32[4]).toBeCloseTo(3); // U[0,0]
    expect(f32[5]).toBeCloseTo(4); // U[0,1]
    expect(f32[6]).toBeCloseTo(1 / 3); // L[1,0]
    expect(f32[7]).toBeCloseTo(2 / 3); // U[1,1]

    // Pivot should be [1, 1] (row 1 was pivot for col 0)
    expect(i32[8]).toBe(1); // pivots
  });

  it("handles batched LU", async () => {
    const bytes = buildLUModuleSized(2, 2, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { lu_batched } = instance.exports as {
      lu_batched: (
        a: number,
        lu: number,
        piv: number,
        perm: number,
        batch: number,
      ) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    // i32 view not needed for this test but kept for reference
    const _i32 = new Int32Array(memory.buffer);

    // Two 2x2 matrices
    const aPtr = 0; // 8 floats
    const luPtr = 32; // 8 floats
    const pivPtr = 64; // 4 ints
    const permPtr = 80; // 4 ints

    // First matrix [[4, 3], [6, 3]]
    f32[0] = 4;
    f32[1] = 3;
    f32[2] = 6;
    f32[3] = 3;
    // Second matrix [[2, 1], [1, 3]]
    f32[4] = 2;
    f32[5] = 1;
    f32[6] = 1;
    f32[7] = 3;

    lu_batched(aPtr, luPtr, pivPtr, permPtr, 2);

    // Both should produce valid LU
    expect(f32[8]).not.toBeNaN();
    expect(f32[12]).not.toBeNaN();
  });
});

describe("wasmblr sort (size-specialized)", () => {
  it("builds valid wasm module", () => {
    const bytes = buildSortModuleSized(5, "f32");
    expect(bytes.length).toBeGreaterThan(0);
    const mod = new WebAssembly.Module(bytes);
    expect(mod).toBeInstanceOf(WebAssembly.Module);
  });

  it("sorts f32 array", async () => {
    const bytes = buildSortModuleSized(5, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { sort } = instance.exports as {
      sort: (data: number, aux: number) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    const dataPtr = 0;
    const auxPtr = 40; // space for aux

    f32[0] = 5;
    f32[1] = 2;
    f32[2] = 8;
    f32[3] = 1;
    f32[4] = 9;

    sort(dataPtr, auxPtr);

    expect(f32[0]).toBe(1);
    expect(f32[1]).toBe(2);
    expect(f32[2]).toBe(5);
    expect(f32[3]).toBe(8);
    expect(f32[4]).toBe(9);
  });

  it("handles NaN values", async () => {
    const bytes = buildSortModuleSized(4, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { sort } = instance.exports as {
      sort: (data: number, aux: number) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    const dataPtr = 0;
    const auxPtr = 40;

    f32[0] = 3;
    f32[1] = NaN;
    f32[2] = 1;
    f32[3] = 2;

    sort(dataPtr, auxPtr);

    // NaN should be at the end
    expect(f32[0]).toBe(1);
    expect(f32[1]).toBe(2);
    expect(f32[2]).toBe(3);
    expect(Number.isNaN(f32[3])).toBe(true);
  });

  it("sorts batched arrays", async () => {
    const bytes = buildSortModuleSized(4, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { sort_batched } = instance.exports as {
      sort_batched: (data: number, aux: number, batch: number) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    const dataPtr = 0;
    const auxPtr = 80;

    // Two arrays of 4 elements each
    f32[0] = 4;
    f32[1] = 1;
    f32[2] = 3;
    f32[3] = 2;
    f32[4] = 8;
    f32[5] = 5;
    f32[6] = 7;
    f32[7] = 6;

    sort_batched(dataPtr, auxPtr, 2);

    expect([f32[0], f32[1], f32[2], f32[3]]).toEqual([1, 2, 3, 4]);
    expect([f32[4], f32[5], f32[6], f32[7]]).toEqual([5, 6, 7, 8]);
  });
});

describe("wasmblr argsort (size-specialized)", () => {
  it("builds valid wasm module", () => {
    const bytes = buildArgsortModuleSized(5, "f32");
    expect(bytes.length).toBeGreaterThan(0);
    const mod = new WebAssembly.Module(bytes);
    expect(mod).toBeInstanceOf(WebAssembly.Module);
  });

  it("returns sorted indices", async () => {
    const bytes = buildArgsortModuleSized(5, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { argsort } = instance.exports as {
      argsort: (data: number, out: number, idx: number, aux: number) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    const i32 = new Int32Array(memory.buffer);

    const dataPtr = 0; // 5 floats = 20 bytes
    const outPtr = 20; // 5 floats = 20 bytes
    const idxPtr = 40; // 5 ints = 20 bytes
    const auxPtr = 60; // aux space

    f32[0] = 30;
    f32[1] = 10;
    f32[2] = 50;
    f32[3] = 20;
    f32[4] = 40;

    argsort(dataPtr, outPtr, idxPtr, auxPtr);

    // Sorted indices: [1, 3, 0, 4, 2] (values: 10, 20, 30, 40, 50)
    expect(i32[10]).toBe(1); // index of 10
    expect(i32[11]).toBe(3); // index of 20
    expect(i32[12]).toBe(0); // index of 30
    expect(i32[13]).toBe(4); // index of 40
    expect(i32[14]).toBe(2); // index of 50

    // Output should be sorted values
    expect(f32[5]).toBe(10);
    expect(f32[6]).toBe(20);
    expect(f32[7]).toBe(30);
    expect(f32[8]).toBe(40);
    expect(f32[9]).toBe(50);
  });

  it("handles batched argsort", async () => {
    const n = 3;
    const bytes = buildArgsortModuleSized(n, "f32");
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { instance } = await WebAssembly.instantiate(bytes, {
      env: { memory },
    });
    const { argsort_batched } = instance.exports as {
      argsort_batched: (
        data: number,
        out: number,
        idx: number,
        aux: number,
        batch: number,
      ) => void;
    };

    const f32 = new Float32Array(memory.buffer);
    const i32 = new Int32Array(memory.buffer);

    const batch = 2;
    const dataPtr = 0; // 6 floats
    const outPtr = 24; // 6 floats
    const idxPtr = 48; // 6 ints
    const auxPtr = 72;

    // First batch: [3, 1, 2]
    f32[0] = 3;
    f32[1] = 1;
    f32[2] = 2;
    // Second batch: [5, 6, 4]
    f32[3] = 5;
    f32[4] = 6;
    f32[5] = 4;

    argsort_batched(dataPtr, outPtr, idxPtr, auxPtr, batch);

    // First batch sorted: [1, 2, 3], indices [1, 2, 0]
    expect([f32[6], f32[7], f32[8]]).toEqual([1, 2, 3]);
    expect([i32[12], i32[13], i32[14]]).toEqual([1, 2, 0]);

    // Second batch sorted: [4, 5, 6], indices [2, 0, 1]
    expect([f32[9], f32[10], f32[11]]).toEqual([4, 5, 6]);
    expect([i32[15], i32[16], i32[17]]).toEqual([2, 0, 1]);
  });
});

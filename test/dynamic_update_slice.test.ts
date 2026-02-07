import { beforeAll, describe, expect, it } from "vitest";

import {
  defaultDevice,
  dynamicUpdateSlice,
  init,
  jit,
  numpy as np,
} from "../src";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

describe("dynamicUpdateSlice", () => {
  it("copies a 2x2 block into a 4x2 dst at offset 1", async () => {
    const dst = np.zeros([4, 2]);
    const src = np.array([
      [10, 11],
      [12, 13],
    ]);
    const out = dynamicUpdateSlice(dst, src, 1);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([
      [0, 0],
      [10, 11],
      [12, 13],
      [0, 0],
    ]);
  });

  it("Array.at(index).set(src) convenience works", async () => {
    const dst = np.zeros([3, 1]);
    const src = np.array([[5]]);
    const out = dst.at(1).set(src);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([[0], [5], [0]]);
  });

  it("throws on out-of-bounds offsets", async () => {
    const dst = np.zeros([2, 2]);
    const src = np.array([
      [1, 2],
      [3, 4],
    ]);
    expect(() => dynamicUpdateSlice(dst, src, 1)).toThrow();
  });

  it("uses WebGPU copy path when available", async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      // Skip if WebGPU not available
      return;
    }
    const dst = np.zeros([4, 1], { device: "webgpu" });
    const src = np.array([[7], [8]], { device: "webgpu" });
    const out = dynamicUpdateSlice(dst, src, 1);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([[0], [7], [8], [0]]);
  });
});

describe("dynamicUpdateSlice with jit", () => {
  const jitAny = jit as (...args: any[]) => any;

  it("jit: same-rank update at axis=0", async () => {
    const f = jitAny((dst: any, src: any) => dynamicUpdateSlice(dst, src, 1));
    const dst = np.zeros([4, 2]);
    const src = np.array([
      [10, 11],
      [12, 13],
    ]);
    const out = f(dst, src);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([
      [0, 0],
      [10, 11],
      [12, 13],
      [0, 0],
    ]);
    f.dispose();
  });

  it("jit: stacked mode (src.ndim = dst.ndim - 1)", async () => {
    const f = jitAny((dst: any, src: any) => dynamicUpdateSlice(dst, src, 2));
    const dst = np.zeros([5, 3]);
    const src = np.array([7, 8, 9]);
    const out = f(dst, src);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([
      [0, 0, 0],
      [0, 0, 0],
      [7, 8, 9],
      [0, 0, 0],
      [0, 0, 0],
    ]);
    f.dispose();
  });

  it("jit: chained updates build up a buffer", async () => {
    const f = jitAny((dst: any, a: any, b: any) => {
      const step1 = dynamicUpdateSlice(dst, a, 0);
      return dynamicUpdateSlice(step1, b, 1);
    });
    const dst = np.zeros([3, 2]);
    const a = np.array([[1, 2]]);
    const b = np.array([[3, 4]]);
    const out = f(dst, a, b);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([
      [1, 2],
      [3, 4],
      [0, 0],
    ]);
    f.dispose();
  });

  it("jit: update at non-zero axis", async () => {
    const f = jitAny((dst: any, src: any) =>
      dynamicUpdateSlice(dst, src, 1, 1),
    );
    const dst = np.zeros([2, 4]);
    const src = np.array([
      [10, 11],
      [12, 13],
    ]);
    const out = f(dst, src);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([
      [0, 10, 11, 0],
      [0, 12, 13, 0],
    ]);
    f.dispose();
  });

  it("jit: DUS combined with arithmetic", async () => {
    const f = jitAny((dst: any, src: any) => {
      const doubled = np.multiply(src, np.array([2]));
      return dynamicUpdateSlice(dst, doubled, 0);
    });
    const dst = np.zeros([3, 1]);
    const src = np.array([[5], [6]]);
    const out = f(dst, src);
    const js = await (out as any).jsAsync();
    expect(js).toEqual([[10], [12], [0]]);
    f.dispose();
  });

  it("jit: scalar stacked mode", async () => {
    const f = jitAny((dst: any, src: any) => dynamicUpdateSlice(dst, src, 1));
    const dst = np.zeros([4]);
    const src = np.array(99);
    const out = f(dst, src);
    const data = await (out as any).data();
    expect(Array.from(new Float32Array(data.buffer))).toEqual([0, 99, 0, 0]);
    f.dispose();
  });
});

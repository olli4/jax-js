import { beforeAll, describe, expect, it } from "vitest";

import { defaultDevice, dynamicUpdateSlice, init, numpy as np } from "../src";

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

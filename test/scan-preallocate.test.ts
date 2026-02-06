import { beforeAll, describe, expect, it } from "vitest";

import { defaultDevice, init, jit, lax, numpy as np } from "../src";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

describe("lax.scan preallocateY", () => {
  it("produces correct stacked outputs with preallocateY=true", async () => {
    const step = (carry: any, x: any): [any, any] => {
      const nc = carry.add(x);
      return [nc, nc];
    };
    const initCarry = np.zeros([1]);
    const xs = np.array([[1], [2], [3], [4]]);
    const f = jit(() => lax.scan(step, initCarry, xs, { preallocateY: true }));
    // Enable debug logging for scanRunner
    const { setDebug } = await import("../src");
    setDebug(1);

    const [final, ys] = f();
    console.log(
      "DEBUG OUT: final.shape=",
      final.shape,
      "ys.shape=",
      ys.shape,
      "ys.ref=",
      ys.refCount,
    );
    const yjs = await ys.jsAsync();
    expect(yjs).toEqual([[1], [3], [6], [10]]);
    console.log("DEBUG: final.refCount=", final.refCount);
    // Note: `ys.jsAsync()` consumes `ys` (readback), so don't call `ys.dispose()` after.
    final.dispose();
  });
});

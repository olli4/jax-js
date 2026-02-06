import { beforeAll, describe, expect, it } from "vitest";

import { defaultDevice, init, jit, lax, numpy as np } from "../src";

beforeAll(async () => {
  const devices = await init();
  if (devices.includes("cpu")) defaultDevice("cpu");
});

describe("lax.scan preallocated ys", () => {
  it("produces correct stacked outputs", async () => {
    const step = (carry: any, x: any): [any, any] => {
      const nc = carry.add(x);
      return [nc, nc];
    };
    const initCarry = np.zeros([1]);
    const xs = np.array([[1], [2], [3], [4]]);
    const f = jit(() => lax.scan(step, initCarry, xs));

    const [final, ys] = f();
    const yjs = await ys.jsAsync();
    expect(yjs).toEqual([[1], [3], [6], [10]]);
    // Note: `ys.jsAsync()` consumes `ys` (readback), so don't call `ys.dispose()` after.
    final.dispose();
  });

  it("handles duplicate-slot ys", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, any] => {
      const nc = np.add(carry, x);
      return [nc, { a: nc.ref, b: nc.ref }];
    };
    const initCarry = np.zeros([1]);
    const xs = np.array([[1], [2], [3]]);
    const f = jit(() => lax.scan(step, initCarry, xs));

    const [final, ys] = f();
    const ysObj = ys as { a: np.Array; b: np.Array };
    const a = await ysObj.a.jsAsync();
    const b = await ysObj.b.jsAsync();
    expect(a).toEqual([[1], [3], [6]]);
    expect(b).toEqual([[1], [3], [6]]);
    final.dispose();
  });

  it("handles passthrough ys", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const carryRef = carry.ref;
      const nc = np.add(carry, x);
      return [nc, carryRef];
    };
    const initCarry = np.zeros([1]);
    const xs = np.array([[1], [2], [3]]);
    const f = jit(() => lax.scan(step, initCarry, xs));

    const [final, ys] = f();
    const yjs = await ys.jsAsync();
    expect(yjs).toEqual([[0], [1], [3]]);
    final.dispose();
  });

  it("handles reverse=true", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const nc = np.add(carry, x);
      return [nc, nc];
    };
    const initCarry = np.zeros([1]);
    const xs = np.array([[1], [2], [3], [4]]);
    const f = jit(() => lax.scan(step, initCarry, xs, { reverse: true }));

    const [final, ys] = f();
    const yjs = await ys.jsAsync();
    // reverse: processes xs as [4,3,2,1], cumsum = [4,7,9,10]
    // stacked in original xs order: ys[0]=10, ys[1]=9, ys[2]=7, ys[3]=4
    expect(yjs).toEqual([[10], [9], [7], [4]]);
    final.dispose();
  });

  it("handles length=0", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const nc = np.add(carry, x);
      return [nc, nc];
    };
    const initCarry = np.zeros([1]);
    const xs = np.zeros([0, 1]);
    const f = jit(() => lax.scan(step, initCarry, xs));

    const [final, ys] = f();
    const yjs = await ys.jsAsync();
    expect(yjs).toEqual([]);
    const finalJs = await final.jsAsync();
    expect(finalJs).toEqual([0]);
  });
});

/**
 * Leak detection tests for scan fallback path.
 *
 * CPU backend always uses fallback (no compiled-loop support), making it
 * ideal for verifying that the fallback loop doesn't leak slots.
 *
 * NOTE on lazy constants: np.array([scalar]) and np.array([v,...v]) with
 * <128 elements create AluExp-backed arrays — no backend.malloc. Use
 * distinct-valued multi-element arrays to force eager allocation if you
 * need precise slot counting.
 *
 * Each test measures slotCount() before and after a scan + data read,
 * expecting zero delta (all scan internals cleaned up).
 */
import { defaultDevice, getBackend, init, lax, numpy as np } from "@jax-js/jax";
import { beforeAll, describe, expect, it } from "vitest";

/** Return the number of live backend slots. */
function slotCount(): number {
  return (getBackend() as any).slotCount();
}

describe("scan fallback leak detection (CPU)", () => {
  beforeAll(async () => {
    await init("cpu");
    defaultDevice("cpu");
  });

  it("cumsum body does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, null] => {
      return [np.add(carry, x), null];
    };

    const before = slotCount();
    const [carry] = lax.scan(step, initC.ref, xs.ref);
    await carry.data();
    expect(slotCount() - before).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("body with captured constant does not leak", async () => {
    const bias = np.array([100, 200]); // distinct → eager alloc
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, null] => {
      return [np.add(np.add(carry, x), bias.ref), null];
    };

    const before = slotCount();
    const [carry] = lax.scan(step, initC.ref, xs.ref);
    await carry.data();
    expect(slotCount() - before).toBe(0);

    xs.dispose();
    initC.dispose();
    bias.dispose();
  });

  it("passthrough body (carry = y) does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, any] => {
      const newC = np.add(carry, x);
      return [newC.ref, newC];
    };

    const before = slotCount();
    const [carry, ys] = lax.scan(step, initC.ref, xs.ref);
    await carry.data();
    await ys.data();
    expect(slotCount() - before).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("body with independent carry and y does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, any] => {
      const newC = np.add(carry, x.ref);
      return [newC, np.multiply(x, np.array([2, 3]))];
    };

    const before = slotCount();
    const [carry, ys] = lax.scan(step, initC.ref, xs.ref);
    await carry.data();
    await ys.data();
    expect(slotCount() - before).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("xs=null carry-only scan does not leak", async () => {
    const initC = np.array([10, 20]);

    const step = (carry: any, _x: any): [any, null] => {
      // np.array([1, 2]) is a lazy constant (same elements = no malloc)
      // but np.add will inline it during tracing
      const newC = np.add(carry, np.array([1, 2]));
      return [newC, null];
    };

    const before = slotCount();
    const [carry, ys] = lax.scan(step, initC.ref, null, { length: 5 });
    expect(ys).toBeNull();
    await carry.data();
    expect(slotCount() - before).toBe(0);

    initC.dispose();
  });

  it("pytree carry does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const initA = np.array([10, 20]);
    const initB = np.array([30, 40]);

    const step = (carry: { a: any; b: any }, x: any): [any, any] => {
      const newA = np.add(carry.a, x.ref);
      const newB = np.add(carry.b, np.array([1, 2]));
      return [{ a: newA, b: newB }, np.multiply(x, np.array([2, 3]))];
    };

    const before = slotCount();
    const [carry, ys] = lax.scan(step, { a: initA.ref, b: initB.ref }, xs.ref);
    await (carry as any).a.data();
    await (carry as any).b.data();
    await ys.data();
    expect(slotCount() - before).toBe(0);

    xs.dispose();
    initA.dispose();
    initB.dispose();
  });

  it("reverse scan does not leak", async () => {
    const xs = np.array([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, any] => {
      return [np.add(carry, x.ref), np.multiply(x, np.array([3, 4]))];
    };

    const before = slotCount();
    const [carry, ys] = lax.scan(step, initC.ref, xs.ref, {
      reverse: true,
    });
    await carry.data();
    await ys.data();
    expect(slotCount() - before).toBe(0);

    xs.dispose();
    initC.dispose();
  });

  it("longer scan (L=50) does not leak", async () => {
    const data = Array.from({ length: 50 }, (_, i) => [i * 2, i * 2 + 1]);
    const xs = np.array(data);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, any] => {
      return [np.add(carry, x), null];
    };

    const before = slotCount();
    const [carry] = lax.scan(step, initC.ref, xs.ref);
    await carry.data();
    expect(slotCount() - before).toBe(0);

    xs.dispose();
    initC.dispose();
  });
});

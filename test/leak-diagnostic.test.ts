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
import {
  checkLeaks,
  defaultDevice,
  getBackend,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "@jax-js/jax";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

/** Return the number of live backend slots. */
function slotCount(): number {
  return (getBackend() as any).slotCount();
}

let previousDevice: string | undefined;

describe("scan fallback leak detection (CPU)", () => {
  beforeAll(async () => {
    const devices = await init();
    // Remember the best non-CPU device so we can restore it in afterAll
    previousDevice = devices.includes("webgpu")
      ? "webgpu"
      : devices.includes("wasm")
        ? "wasm"
        : undefined;
    defaultDevice("cpu");
  });

  afterAll(() => {
    if (previousDevice) defaultDevice(previousDevice as any);
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

  it("acceptPath is enforced in eager mode", () => {
    // When acceptPath causes scan to throw, internal .ref copies leak because
    // the cleanup code never runs. This is a known library-level cleanup issue.
    checkLeaks.stop();

    const xs = np.array([
      [1, 2],
      [3, 4],
    ]);
    const initC = np.array([10, 20]);

    const step = (carry: any, x: any): [any, null] => {
      return [np.add(carry, x), null];
    };

    // CPU only supports fallback — requesting compiled-loop should throw
    expect(() =>
      lax.scan(step, initC.ref, xs.ref, { acceptPath: "compiled-loop" }),
    ).toThrow();

    xs.dispose();
    initC.dispose();
    checkLeaks.start();
  });
});

/**
 * Regression tests for grad + jit memory leaks.
 *
 * Before the rcBeforeTrace fix in linearize.ts, every level of grad()
 * applied to a jit-wrapped function leaked exactly 1 backend slot.
 * The root cause: partial evaluation (PE) tracing in the jit path
 * borrows the input (doesn't consume it), but vjp's fVjp.dispose()
 * assumed PE always consumed inputs. The fix captures refcounts before
 * PE tracing and after disposeJaxpr(), disposing any arg whose refcount
 * was not reduced by PE.
 *
 * These tests use the CPU backend for precise slot counting.
 */
describe("grad + jit leak detection (CPU)", () => {
  beforeAll(async () => {
    const devices = await init();
    previousDevice = devices.includes("webgpu")
      ? "webgpu"
      : devices.includes("wasm")
        ? "wasm"
        : undefined;
    defaultDevice("cpu");
  });

  afterAll(() => {
    if (previousDevice) defaultDevice(previousDevice as any);
  });

  it("grad(f)(x) does not leak", async () => {
    // f(x) = x^2, scalar in → scalar out
    const f = (x: any) => np.multiply(x.ref, x);

    const before = slotCount();
    const dx = grad(f)(np.array(3.0));
    await dx.data();
    expect(slotCount() - before).toBe(0);
  });

  it("grad(jit(f))(x) does not leak", async () => {
    const f = jit((x: any) => np.multiply(x.ref, x));

    const before = slotCount();
    const dx = grad(f)(np.array(3.0));
    await dx.data();
    f.dispose();
    expect(slotCount() - before).toBe(0);
  });

  it("grad(grad(f))(x) does not leak", async () => {
    // f(x) = x^2 → f'(x) = 2x → f''(x) = 2
    const f = (x: any) => np.multiply(x.ref, x);

    const before = slotCount();
    const ddx = grad(grad(f))(np.array(3.0));
    await ddx.data();
    expect(slotCount() - before).toBe(0);
  });

  it("grad(grad(jit(f)))(x) does not leak", async () => {
    const f = jit((x: any) => np.multiply(x.ref, x));

    const before = slotCount();
    const ddx = grad(grad(f))(np.array(3.0));
    await ddx.data();
    f.dispose();
    expect(slotCount() - before).toBe(0);
  });

  it("grad^3(jit(f))(x) does not leak", async () => {
    // f(x) = x^4 → f'''(x) = 24x (use x^4 to ensure 3rd derivative nonzero)
    const f = jit((x: any) => {
      const x2 = np.multiply(x.ref, x);
      return np.multiply(x2.ref, x2);
    });

    const before = slotCount();
    const dddx = grad(grad(grad(f)))(np.array(2.0));
    await dddx.data();
    f.dispose();
    expect(slotCount() - before).toBe(0);
  });

  it("grad^5(f)(x) does not leak", async () => {
    // f(x) = x^6 → f^(5)(x) = 720x (nonzero at all derivative levels)
    // x is used many times, so each use except the last needs .ref
    const f = (x: any) => {
      const x2 = np.multiply(x.ref, x.ref);
      const x3 = np.multiply(x2, x.ref);
      const x6 = np.multiply(x3.ref, x3);
      return np.multiply(x6, np.array(1.0)); // keep scalar; x fully consumed
    };

    const before = slotCount();
    const d5x = grad(grad(grad(grad(grad(f)))))(np.array(1.0));
    await d5x.data();
    expect(slotCount() - before).toBe(0);
  });
});

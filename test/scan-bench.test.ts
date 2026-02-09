/**
 * Scan performance benchmark — measures scan paths on the current backend.
 * Run with: pnpm vitest run test/scan-bench.test.ts
 */

import { defaultDevice, init, jit, lax, numpy as np } from "@jax-js/jax";
import { describe, expect, it } from "vitest";

async function timeMs(fn: () => void, warmup = 3, runs = 10): Promise<number> {
  for (let i = 0; i < warmup; i++) fn();
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const start = performance.now();
    fn();
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

describe("scan performance benchmarks", () => {
  it("cumsum: N=100, SIZE=64", async () => {
    await init();
    const device = defaultDevice();
    const N = 100;
    const SIZE = 64;
    const xs = np.ones([N, SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry];
        },
        np.zeros([SIZE]),
        xs,
      );
    });

    // Verify correctness
    const [c, _y] = scanAuto(xs);
    const data = await c.data();
    expect(data[0]).toBeCloseTo(N, 1);
    _y.dispose();

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `[${device}] cumsum          (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
    xs.dispose();
  });

  it("cumsum large: N=500, SIZE=256", async () => {
    await init();
    const device = defaultDevice();
    const N = 500;
    const SIZE = 256;
    const xs = np.ones([N, SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry];
        },
        np.zeros([SIZE]),
        xs,
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `[${device}] cumsum-large    (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
    xs.dispose();
  });

  it("carry-only scan: N=200, SIZE=32", async () => {
    await init();
    const device = defaultDevice();
    const N = 200;
    const SIZE = 32;

    const scanAuto = jit(() => {
      return lax.scan(
        (carry: any, _x: any) => {
          const newCarry = carry.add(np.ones([SIZE]));
          return [newCarry, null];
        },
        np.zeros([SIZE]),
        null,
        { length: N },
      );
    });

    const autoMs = await timeMs(() => {
      const [c, _] = scanAuto() as [any, any];
      c.dispose();
    });
    console.log(
      `[${device}] carry-only      (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
  });

  it("multi-carry scan: N=100, SIZE=32", async () => {
    await init();
    const device = defaultDevice();
    const N = 100;
    const SIZE = 32;
    const xs = np.ones([N, SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const [a, b] = carry;
          const newA = a.add(x);
          const newB = b.mul(np.array([0.99])).add(x.mul(np.array([0.01])));
          return [
            [newA, newB],
            [newA, newB],
          ] as any;
        },
        [np.zeros([SIZE]), np.zeros([SIZE])] as any,
        xs,
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      (c as any)[0].dispose();
      (c as any)[1].dispose();
      (y as any)[0].dispose();
      (y as any)[1].dispose();
    });
    console.log(
      `[${device}] multi-carry     (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
    xs.dispose();
  });

  it("scan with reduction: N=100, SIZE=64", async () => {
    await init();
    const device = defaultDevice();
    const N = 100;
    const SIZE = 64;
    const xs = np.ones([N, SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const s = carry.add(np.sum(x));
          return [s, s];
        },
        np.zeros([]),
        xs,
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `[${device}] reduction       (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
    xs.dispose();
  });

  it("reverse scan: N=200, SIZE=64", async () => {
    await init();
    const device = defaultDevice();
    const N = 200;
    const SIZE = 64;
    const xs = np.ones([N, SIZE]);

    const scanAuto = jit((xs: any) => {
      return lax.scan(
        (carry: any, x: any) => {
          const newCarry = carry.add(x);
          return [newCarry, newCarry];
        },
        np.zeros([SIZE]),
        xs,
        { reverse: true },
      );
    });

    const autoMs = await timeMs(() => {
      const [c, y] = scanAuto(xs) as [any, any];
      c.dispose();
      y.dispose();
    });
    console.log(
      `[${device}] reverse         (N=${N}, size=${SIZE}): ${autoMs.toFixed(2)}ms`,
    );

    scanAuto.dispose();
    xs.dispose();
  });
});

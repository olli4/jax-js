/**
 * Tests for the checkLeaks diagnostic API.
 *
 * checkLeaks uses backend slot count deltas (same as slotCount()) for accurate
 * leak detection, plus a tracking map for diagnostic details.
 */
import { checkLeaks, init, jit, numpy as np } from "@jax-js/jax";
import { describe, expect, it } from "vitest";

await init();

describe("checkLeaks", () => {
  it("reports zero leaks when all arrays disposed", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5, 6]);
    x.dispose();
    y.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
    expect(report.summary).toBe("No leaks detected.");
  });

  it("detects a single leaked array", () => {
    checkLeaks.start();
    const x = np.array([1, 2, 3]);
    // deliberately do NOT dispose x
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(1);
    expect(report.details.length).toBeGreaterThanOrEqual(1);
    expect(report.summary).toContain("1 slot(s) leaked");
    expect(report.summary).toContain("jit()");
    // Clean up for next test
    x.dispose();
  });

  it("detects multiple leaked arrays", () => {
    checkLeaks.start();
    const a = np.array([1, 2, 3]);
    const b = np.array([4, 5, 6]);
    const c = np.array([7, 8, 9]);
    a.dispose(); // only a is disposed
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(2);
    b.dispose();
    c.dispose();
  });

  it("jit intermediates do not leak", () => {
    checkLeaks.start();
    const f = jit((x: np.Array) => x.mul(2).add(3).sub(1));
    const x = np.array([10, 20]);
    const result = f(x);
    expect(result.js()).toEqual([22, 42]);
    result.dispose();
    x.dispose();
    f.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0);
  });

  it("active property reflects tracking state", () => {
    expect(checkLeaks.active).toBe(false);
    checkLeaks.start();
    expect(checkLeaks.active).toBe(true);
    checkLeaks.stop();
    expect(checkLeaks.active).toBe(false);
  });

  it("arrays created before start are not counted", () => {
    const preExisting = np.array([99, 100]);
    checkLeaks.start();
    const tracked = np.array([1, 2]);
    tracked.dispose();
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(0); // preExisting not counted (existed before start)
    preExisting.dispose();
  });

  it("report details include array description", () => {
    checkLeaks.start();
    const x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const report = checkLeaks.stop();
    expect(report.leaked).toBe(1);
    // Details should contain dtype and shape info from tracked arrays
    expect(report.details.length).toBeGreaterThanOrEqual(1);
    const hasArrayInfo = report.details.some(
      (d) => /float32|int32/.test(d) && /2,2|2, 2/.test(d),
    );
    expect(hasArrayInfo).toBe(true);
    x.dispose();
  });
});

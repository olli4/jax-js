// Tests for convolution-related operations.

import { devices, init, lax, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  test("1d convolution", () => {
    const x = np.array([[[1, 2, 3, 4, 5]]]);
    const y = np.array([[[2, 0.5, -1]]]);
    const result = lax.convGeneralDilated(x.ref, y.ref, [1], "VALID");
    expect(result.js()).toEqual([[[0, 1.5, 3]]]);

    const result2 = lax.convGeneralDilated(x, y, [1], "SAME");
    expect(result2.js()).toEqual([[[-1.5, 0, 1.5, 3, 10.5]]]);
  });

  test("padding 'SAME' and 'SAME_LOWER'", () => {
    const x = np.ones([1, 1, 5]);
    const y = np.ones([1, 1, 2]);
    const resultSame = lax.convGeneralDilated(x.ref, y.ref, [1], "SAME");
    expect(resultSame.slice(0, 0).js()).toEqual([2, 2, 2, 2, 1]);
    const resultSameLower = lax.convGeneralDilated(x, y, [1], "SAME_LOWER");
    expect(resultSameLower.slice(0, 0).js()).toEqual([1, 2, 2, 2, 2]);
  });
});

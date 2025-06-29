import { devices, grad, init, nn, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  suite("jax.nn.relu()", () => {
    test("should compute ReLU", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.relu(x);
      expect(y.js()).toEqual([0, 0, 1, 2]);
    });

    test("should compute ReLU gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.relu(x).sum());
      const gx = gradFn(x);
      expect(gx.js()).toEqual([0, 0, 1, 1]);
    });
  });

  suite("jax.nn.sigmoid()", () => {
    test("should compute sigmoid", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.sigmoid(x);
      expect(y).toBeAllclose([0.26894142, 0.5, 0.73105858, 0.88079708]);
    });

    test("should compute sigmoid gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.sigmoid(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([0.0000454, 0.19661193, 0.19661193, 0.10499359]);
    });
  });

  suite("jax.nn.softSign()", () => {
    test("should compute softsign", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.softSign(x);
      expect(y).toBeAllclose([-0.5, 0, 0.5, 2 / 3]);
    });

    test("should compute softsign gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.softSign(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([1 / 121, 1 / 4, 1 / 4, 1 / 9]);
    });
  });
});

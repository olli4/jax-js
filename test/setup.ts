import { numpy as np } from "@jax-js/core";
import { expect } from "vitest";

expect.extend({
  toBeAllclose(received: np.ArrayLike, expected: np.ArrayLike) {
    const { isNot } = this;
    return {
      pass: np.allclose(received, expected),
      message: () => `expected array to be${isNot ? " not" : ""} allclose`,
      actual: np.array(received).js(),
      expected: np.array(expected).js(),
    };
  },
});

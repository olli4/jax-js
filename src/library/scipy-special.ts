// Mirrors the `jax.scipy.special` module in JAX.

import { Array, log, subtract } from "./numpy";
import { jit } from "../frontend/jaxpr";

export { erf } from "./lax";
export { erfc } from "./lax";
export { logSoftmax } from "./nn";

/**
 * @function
 * The logit function, `logit(p) = log(p / (1-p))`.
 */
export const logit = jit(function logit(x: Array): Array {
  return log(x.div(subtract(1, x)));
});

export { logsumexp } from "./nn";
export { softmax } from "./nn";

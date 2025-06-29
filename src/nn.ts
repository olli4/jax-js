// Common functions for neural network libraries, mirroring `jax.nn` in JAX.

import { fudgeArray } from "./frontend/array";
import {
  absolute,
  Array,
  ArrayLike,
  clip,
  exp,
  log,
  maximum,
  negative,
  reciprocal,
} from "./numpy";

/**
 * Rectified Linear Unit (ReLU) activation function:
 * `relu(x) = max(x, 0)`.
 */
export function relu(x: ArrayLike): Array {
  return maximum(x, 0);
}

/**
 * Rectified Linear Unit 6 (ReLU6) activation function:
 * `relu6(x) = min(max(x, 0), 6)`.
 */
export function relu6(x: ArrayLike): Array {
  return clip(x, 0, 6);
}

/**
 * Sigmoid activation function, computed element-wise:
 * `sigmoid(x) = 1 / (1 + exp(-x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Sigmoid_function
 */
export function sigmoid(x: ArrayLike): Array {
  return reciprocal(exp(negative(x)).add(1));
}

/**
 * Softplus activation function:
 * `softplus(x) = log(1 + exp(x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Softplus
 */
export function softplus(x: ArrayLike): Array {
  return log(exp(x).add(1));
}

/**
 * Soft-sign activation function, computed element-wise:
 * `softsign(x) = x / (|x| + 1)`.
 */
export function softSign(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.div(absolute(x).add(1));
}

/**
 * Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
 * Swish, computed element-wise:
 * `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
 *
 * `swish()` and `silu()` are both aliases for the same function.
 *
 * Reference: https://en.wikipedia.org/wiki/Swish_function
 */
export function silu(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.mul(sigmoid(x));
}

/**
 * Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
 * Swish, computed element-wise:
 * `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
 *
 * `swish()` and `silu()` are both aliases for the same function.
 *
 * Reference: https://en.wikipedia.org/wiki/Swish_function
 */
export const swish = silu;

/**
 * Log-sigmoid activation function, computed element-wise:
 * `log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))`.
 */
export function logSigmoid(x: ArrayLike): Array {
  return negative(softplus(negative(x)));
}

/** Identity activation function. Returns the argument unmodified. */
export const identity = fudgeArray;

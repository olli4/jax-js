// Mirrors the `jax.lax` module in JAX.
//
// Unlike in JAX, this does not actually underpin `jax.numpy` as a more "core"
// set of operations, as they both build open the same foundations.

import { Array } from "./frontend/array";
import { conv as convPrimitive } from "./frontend/core";
import { rep, zipn } from "./utils";

type PaddingType = "VALID" | "SAME" | "SAME_LOWER" | [number, number][];

function padtypeToPads(
  inShape: number[],
  filterShape: number[],
  strides: number[],
  dilation: number[],
  padding: string,
): [number, number][] {
  const padType = padding.toUpperCase();
  switch (padType) {
    case "VALID":
      return rep<[number, number]>(inShape.length, [0, 0]);
    case "SAME":
    case "SAME_LOWER": {
      const outShape = inShape.map((size, i) => Math.ceil(size / strides[i]));
      const padSizes = zipn(
        outShape,
        strides,
        filterShape,
        dilation,
        inShape,
      ).map(([o, s, k, d, i]) =>
        Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i),
      );
      if (padType === "SAME") {
        return padSizes.map((size) => [size >> 1, size - (size >> 1)]);
      } else {
        return padSizes.map((size) => [size - (size >> 1), size >> 1]);
      }
    }
    default:
      throw new Error(`Unknown padding type: ${padType}`);
  }
}

/**
 * General n-dimensional convolution operator, with optional dilation.
 *
 * The semantics of this operation mimic the `jax.lax.conv_general_dilated`
 * function in JAX, which wraps XLA's general convolution operator.
 *
 * Grouped convolutions are not supported right now.
 */
export function convGeneralDilated(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
  {
    lhsDilation,
    rhsDilation,
  }: {
    lhsDilation?: number[];
    rhsDilation?: number[];
  } = {},
): Array {
  if (typeof padding === "string") {
    if (lhsDilation?.some((d) => d !== 1)) {
      throw new Error(
        "String padding is not supported for transposed convolutions",
      );
    }
    padding = padtypeToPads(
      lhs.shape.slice(2),
      rhs.shape.slice(2),
      windowStrides,
      rhsDilation ?? rep(rhs.ndim - 2, 1),
      padding,
    );
  }
  return convPrimitive(lhs, rhs, {
    strides: windowStrides,
    padding,
    lhsDilation,
    rhsDilation,
  }) as Array;
}

/** Convenience wrapper around `convGeneralDilated`. */
export function convWithGeneralPadding(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
  lhsDilation?: number[],
  rhsDilation?: number[],
) {
  return convGeneralDilated(lhs, rhs, windowStrides, padding, {
    lhsDilation,
    rhsDilation,
  });
}

/** Convenience wrapper around `convGeneralDilated`. */
export function conv(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
) {
  return convGeneralDilated(lhs, rhs, windowStrides, padding);
}

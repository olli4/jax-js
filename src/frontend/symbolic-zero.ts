import { zeros } from "./array";
import { AbstractValue, ShapedArray, Tracer } from "./core";

/**
 * Symbolic zero tangent/cotangent value for AD internals.
 *
 * Represents a mathematically-zero value without allocating backend storage
 * until materialization is required by a primitive rule.
 */
export class SymbolicZero {
  readonly aval: ShapedArray;

  constructor(aval: AbstractValue) {
    this.aval = ShapedArray.fromAval(aval);
  }

  materialize(): Tracer {
    return zeros(this.aval.shape, { dtype: this.aval.dtype });
  }

  toString(): string {
    return `SymbolicZero(${this.aval.toString()})`;
  }
}

export function isSymbolicZero(x: unknown): x is SymbolicZero {
  return x instanceof SymbolicZero;
}

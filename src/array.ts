import { AluExp, AluOp, DType } from "./alu";
import {
  accessorAluExp,
  accessorGlobal,
  Backend,
  BackendType,
  Executable,
  getBackend,
  Slot,
} from "./backend";
import { ShapeTracker } from "./shape";
import { deepEqual, isPermutation } from "./utils";

const JsArray = globalThis.Array;

class PendingExecute {
  prepared: Executable | null = null;
  submitted = false;
  #promise: Promise<void> | null = null; // for prepare

  constructor(
    readonly exp: AluExp,
    readonly inputs: Slot[],
    readonly outputs: Slot[],
  ) {}

  async prepare(backend: Backend) {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      this.prepared = await backend.prepare(this.inputs.length, this.exp);
    })();
    await this.#promise;
  }

  prepareSync(backend: Backend) {
    if (this.prepared) return;
    this.prepared = backend.prepareSync(this.inputs.length, this.exp);
  }

  submit(backend: Backend) {
    if (this.submitted) return;
    if (!this.prepared) throw new Error("Not prepared yet");
    backend.dispatch(this.prepared, this.inputs, this.outputs);
  }
}

export class Array {
  readonly shape: number[];
  readonly dtype: DType;

  #source: AluExp | Slot;
  #st: ShapeTracker;
  #backend: Backend;
  #pending: Set<PendingExecute> | null; // only if source is `Slot`

  constructor(
    source: AluExp | Slot,
    st: ShapeTracker,
    dtype: DType,
    backend: Backend,
    pending: Set<PendingExecute> | null = null,
  ) {
    this.shape = st.shape;
    this.dtype = dtype;

    this.#source = source;
    this.#st = st;
    this.#backend = backend;
    this.#pending = pending;
  }

  get backend(): BackendType {
    return this.#backend.type;
  }

  get ndim(): number {
    return this.shape.length;
  }

  static zeros(
    shape: number[],
    { dtype, backend }: { dtype?: DType; backend?: BackendType } = {},
  ) {
    dtype = dtype ?? DType.Float32;
    return new Array(
      AluExp.const(dtype, 0),
      ShapeTracker.fromShape(shape),
      dtype,
      getBackend(backend),
    );
  }

  static ones(
    shape: number[],
    { dtype, backend }: { dtype?: DType; backend?: BackendType } = {},
  ) {
    dtype = dtype ?? DType.Float32;
    return new Array(
      AluExp.const(dtype, 1),
      ShapeTracker.fromShape(shape),
      dtype,
      getBackend(backend),
    );
  }

  // Movement operations

  #reshape(st: ShapeTracker): Array {
    return new Array(
      this.#source,
      st,
      this.dtype,
      this.#backend,
      this.#pending,
    );
  }

  reshape(shape: number[]): Array {
    const autoIdx = shape.indexOf(-1);
    if (autoIdx !== -1) {
      const remaining = this.#st.size / shape.reduce((a, b) => a * b, -1);
      if (remaining % 1 !== 0 || remaining < 0) {
        throw new Error(
          `Invalid reshape: ${JSON.stringify(this.shape)} -> ${JSON.stringify(shape)}`,
        );
      }
      shape = shape.toSpliced(autoIdx, 1, remaining);
    }
    return this.#reshape(this.#st.reshape(shape));
  }

  flatten(): Array {
    return this.reshape([-1]);
  }
  ravel(): Array {
    return this.reshape([-1]);
  }

  transpose(axes?: number[]): Array {
    if (axes) {
      if (!isPermutation(axes, this.ndim))
        throw new Error(`Invalid axes for transpose: ${JSON.stringify(axes)}`);
    } else if (!axes) {
      axes = this.shape.map((_, i) => i).reverse();
    }
    return this.#reshape(this.#st.permute(axes));
  }
  get T(): Array {
    return this.transpose();
  }

  #binary(op: AluOp, other: Array) {
    if (!deepEqual(this.shape, other.shape)) {
      throw new Error(`Shape mismatch in ${op}`); // todo: broadcasting, maybe at the jax level
    }
    if (this.dtype !== other.dtype) {
      throw new Error(`Dtype mismatch in ${op}`); // todo: dtype casting
    }

    // Short circuit if both are already AluExp.
    if (this.#source instanceof AluExp && other.#source instanceof AluExp) {
      const exp = new AluExp(op, this.dtype, [this.#source, other.#source]);
      return new Array(exp, this.#st, this.dtype, this.#backend);
    }

    const gidx = AluExp.special(DType.Int32, "gidx", this.#st.size);

    const inputs: Slot[] = [];
    const src: AluExp[] = [];

    for (const ar of [this, other]) {
      if (ar.#source instanceof AluExp) {
        src.push(accessorAluExp(ar.#source, ar.#st, gidx));
      } else {
        src.push(accessorGlobal(inputs.length, ar.#st, gidx));
        inputs.push(ar.#source);
      }
    }

    const exp = new AluExp(op, this.dtype, src);
    const output = this.#backend.malloc(this.#st.size * 4);
    const pending = new Set([
      ...(this.#pending ?? []),
      ...(other.#pending ?? []),
      new PendingExecute(exp, inputs, [output]),
    ]);
    return new Array(
      output,
      ShapeTracker.fromShape(this.shape),
      this.dtype,
      this.#backend,
      pending,
    );
  }

  /**
   * Normalizes this array into one backed by a `Slot`.
   *
   * This mutates the array in-place, turning it into an equivalent array whose
   * source is actual, contiguous data on device.
   *
   * Calling this twice is a no-op.
   */
  #realize(): void {
    if (this.#source instanceof AluExp) {
      const output = this.#backend.malloc(this.#st.size * 4);
      const gidx = AluExp.special(DType.Int32, "gidx", this.#st.size);
      const exp = accessorAluExp(this.#source, this.#st, gidx);
      const pendingItem = new PendingExecute(exp, [], [output]);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pending = new Set([pendingItem]);
    } else {
      // Only realize if the ShapeTracker is non-contiguous.
      if (this.#st.contiguous) return;
      const output = this.#backend.malloc(this.#st.size * 4);
      const gidx = AluExp.special(DType.Int32, "gidx", this.#st.size);
      const exp = accessorGlobal(0, this.#st, gidx);
      const pendingItem = new PendingExecute(exp, [this.#source], [output]);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pending ??= new Set();
      this.#pending.add(pendingItem);
    }
  }

  // These will be evaluation rules in the future, not public API.
  add(other: Array) {
    return this.#binary(AluOp.Add, other);
  }
  sub(other: Array) {
    return this.#binary(AluOp.Sub, other);
  }
  mul(other: Array) {
    return this.#binary(AluOp.Mul, other);
  }

  async data(): Promise<Float32Array> {
    this.#realize();
    if (this.#pending) {
      // Compile all pending executables concurrently.
      await Promise.all(
        [...this.#pending].map((exe) => exe.prepare(this.#backend)),
      );
      for (const p of this.#pending) {
        p.submit(this.#backend);
      }
      this.#pending = null;
    }
    const buf = await this.#backend.read(this.#source as Slot);
    return new Float32Array(buf);
  }

  dataSync(): Float32Array {
    this.#realize();
    if (this.#pending) {
      for (const p of this.#pending) {
        p.prepareSync(this.#backend);
        p.submit(this.#backend);
      }
      this.#pending = null;
    }
    const buf = this.#backend.readSync(this.#source as Slot);
    return new Float32Array(buf);
  }
}

export function arrayFromData(
  data: Float32Array | Int32Array,
  {
    dtype,
    backend: backendType,
  }: { dtype?: DType; backend?: BackendType } = {},
): Array {
  const backend = getBackend(backendType);
  if (data instanceof Float32Array) {
    if (dtype && dtype !== DType.Float32) {
      throw new Error("Float32Array must have float32 type");
    }
    const slot = backend.malloc(data.byteLength, data.buffer);
    return new Array(
      slot,
      ShapeTracker.fromShape([data.length]),
      DType.Float32,
      backend,
    );
  } else if (data instanceof Int32Array) {
    if (dtype && dtype !== DType.Int32) {
      throw new Error("Int32Array must have int32 type");
    }
    const slot = backend.malloc(data.byteLength, data.buffer);
    return new Array(
      slot,
      ShapeTracker.fromShape([data.length]),
      DType.Int32,
      backend,
    );
  } else {
    throw new Error("Unsupported data type");
  }
}

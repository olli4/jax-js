import {
  accessorAluExp,
  accessorGlobal,
  AluExp,
  AluOp,
  AluVar,
  byteWidth,
  DataArray,
  DType,
  dtypedArray,
  dtypedJsArray,
  Kernel,
  Reduction,
} from "../alu";
import { Backend, Device, Executable, getBackend, Slot } from "../backend";
import { Routine } from "../routine";
import { Pair, ShapeTracker, unravelAlu } from "../shape";
import {
  DEBUG,
  deepEqual,
  generalBroadcast,
  isPermutation,
  prod,
  range,
  RecursiveArray,
  recursiveFlatten,
  rep,
} from "../utils";
import {
  checkConvShape,
  pool,
  poolTranspose,
  prepareConv,
} from "./convolution";
import {
  AbstractValue,
  CompareOp,
  exp as coreExp,
  mul as coreMul,
  dynamicUpdateSlice,
  getAval,
  ndim,
  newMain,
  Primitive,
  PrimitiveParams,
  promoteAvals,
  routinePrimitives,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
  UseAfterFreeError,
  where,
} from "./core";
import { concatenate as coreConcatenate } from "./core";
import { abstractEvalRules, evalJaxpr } from "./jaxpr";
import { jitCompile, ScanRunner } from "./jit";

const JsArray = globalThis.Array;

// Don't realize expression arrays smaller than this size.
const inlineArrayLimit = 128;

export type ArrayLike = Array | number | boolean;

/** Version of pureArray with fudged types. */
export const fudgeArray = pureArray as (x: ArrayLike) => Array;

/**
 * An executable operation that will be dispatched to the backend.
 *
 * This holds a reference to all input buffers used in the operation. After the
 * operation is dispatched, the references should be released.
 */
export class PendingExecute {
  prepared: Executable | null = null;
  submitted = false;
  #promise: Promise<void> | null = null; // for prepare
  #rc = 1; // since this could be held by multiple arrays, cancel when it hits 0

  constructor(
    readonly backend: Backend,
    readonly source: Kernel | Routine,
    readonly inputs: Slot[],
    readonly outputs: Slot[],
  ) {
    // Take a reference to all I/O buffers, while this execution is pending.
    // The reference is dropped after submit() or cancellation.
    for (const slot of inputs) this.backend.incRef(slot);
    for (const slot of outputs) this.backend.incRef(slot);
  }

  // Change the reference count of the PendingExecute object.
  // Used when copying the object to a new Array, or disposing an array.
  updateRc(delta: number) {
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      // Cancel operation, release the references held to all input buffers.
      for (const slot of this.inputs) this.backend.decRef(slot);
      for (const slot of this.outputs) this.backend.decRef(slot);
    }
  }

  async prepare() {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      if (this.source instanceof Routine) {
        this.prepared = await this.backend.prepareRoutine(this.source);
      } else {
        this.prepared = await this.backend.prepareKernel(this.source);
      }
    })();
    await this.#promise;
  }

  prepareSync() {
    if (this.prepared) return;
    if (this.source instanceof Routine) {
      this.prepared = this.backend.prepareRoutineSync(this.source);
    } else {
      this.prepared = this.backend.prepareKernelSync(this.source);
    }
  }

  submit() {
    if (this.submitted) return;
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    if (!this.prepared) throw new Error("internal: Not prepared yet");
    this.submitted = true;
    this.backend.dispatch(this.prepared, this.inputs, this.outputs);
    for (const slot of this.inputs) this.backend.decRef(slot);
    for (const slot of this.outputs) this.backend.decRef(slot);
  }
}

/** @inline */
export type DTypeAndDevice = { dtype?: DType; device?: Device };

type ArrayConstructorArgs = {
  source: AluExp | Slot;
  st: ShapeTracker;
  dtype: DType;
  weakType: boolean;
  backend: Backend;
  committed: boolean;
  pending?: Iterable<PendingExecute>;
};

/**
 * A multidimensional numeric array with data stored on CPU or GPU.
 *
 * This is the library's core data type. Equivalent to `jax.Array` from JAX, or
 * `torch.Tensor`.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing
 * this into your code's namespace if you're already using the JavaScript
 * "Array" type by name.
 */
export class Array extends Tracer {
  #dtype: DType;
  #weakType: boolean;
  #source: AluExp | Slot;
  #st: ShapeTracker;
  #backend: Backend;
  #committed: boolean; // if array is committed to device (passed explicitly)
  #rc: number; // reference count for this specific Array object

  #pendingSet: Set<PendingExecute> | null; // only if source is `Slot`

  /**
   * @ignore
   * Constructs an array from source, shape and backend. Note that if the source
   * is a backend `Slot`, this constructor _takes ownership_ of the slot. It
   * will be freed when the array is disposed.
   */
  constructor(args: ArrayConstructorArgs) {
    super(baseArrayTrace);
    this.#dtype = args.dtype;
    this.#weakType = args.weakType;
    this.#source = args.source;
    this.#st = args.st;
    this.#backend = args.backend;
    this.#committed = args.committed;
    this.#rc = 1;

    this.#pendingSet = new Set(args.pending);
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
    } else if (this.#source instanceof AluExp) {
      throw new Error("internal: AluExp source cannot have pending executes");
    }
  }

  /** @ignore */
  get aval() {
    return new ShapedArray(this.#st.shape, this.#dtype, this.#weakType);
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array:${this.#dtype}[${this.shape.join(",")}]`;
  }

  get device(): Device {
    return this.#backend.type;
  }

  #check() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
  }

  /** Construct an array, copying fields from `this`. */
  #newArrayFrom(args: Partial<ArrayConstructorArgs>) {
    return new Array({
      source: args.source ?? this.#source,
      st: args.st ?? this.#st,
      dtype: args.dtype ?? this.#dtype,
      weakType: this.#weakType,
      backend: args.backend ?? this.#backend,
      committed: args.committed ?? this.#committed,
      pending: args.pending ?? this.#pending ?? undefined,
    });
  }

  get ref() {
    this.#check();
    this.#rc++;
    return this;
  }

  /** Get the current reference count (for debugging memory management). */
  get refCount(): number {
    return this.#rc;
  }

  dispose() {
    this.#check();
    if (--this.#rc === 0) {
      // Free any pending executables that haven't been submitted yet.
      for (const exe of this.#pending) exe.updateRc(-1);
      // If this has an array source, free it from the backend.
      if (typeof this.#source === "number") {
        this.#backend.decRef(this.#source);
      }
    }
  }

  /** Get the pending executes as a list, trimming if already submitted. */
  get #pending(): PendingExecute[] {
    if (!this.#pendingSet) return [];
    for (const p of this.#pendingSet) {
      if (p.submitted) this.#pendingSet.delete(p);
    }
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
      return [];
    } else {
      return [...this.#pendingSet];
    }
  }

  /**
   * Convert this array into a primitive value.
   *
   * This only works for scalars (0-dimensional arrays). It lets you get values
   * "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
   * evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
   *
   * This method is also called for `==` equality.
   */
  [Symbol.toPrimitive](): any {
    if (this.ndim === 0) {
      return this.dataSync()[0];
    } else {
      throw new Error(
        `Cannot convert non-scalar array to primitive: ${this.toString()}`,
      );
    }
  }

  #reshape(st: ShapeTracker): Array {
    this.#check();
    const pending = this.#pending;
    for (const exe of pending) exe.updateRc(+1);
    if (typeof this.#source === "number") this.#backend.incRef(this.#source);
    const ar = this.#newArrayFrom({ st, pending });
    this.dispose(); // After constructing Array, so we don't free this.#source early.
    return ar;
  }

  /**
   * Underlying implementation of the Gather primitive. This indexes an array
   * and extracts slices based on indices in other integer arrays.
   */
  #gather(indices: Array[], axis: number[], outDim: number): Array {
    this.#check();

    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length) {
      throw new TypeError("Gather axis must not have duplicates");
    }

    if (indices.some((a) => a.#committed && a.#backend !== this.#backend)) {
      throw new TypeError(
        `Gather indices must have the same backend: ${this.#backend.type}`,
      );
    }
    indices = indices.map((ar) => ar._putSync(this.#backend));

    // First, broadcast each integer array in `indices`.
    indices = Array.#broadcastArrays(indices);
    const indexShape = indices[0].shape;
    const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);

    // Make variables for expression indices for gathered axes, and non-axis.
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const idxAxis = idxNonaxis.splice(outDim, indexShape.length);

    // Then, construct a kernel expression that gathers the data.
    const inputs: Slot[] = [];
    const src: AluExp[] = [...idxNonaxis];
    for (let i = 0; i < this.shape.length; i++) {
      // insert 'null' as axis placeholder, overwritten below as src[axis[i]].
      if (axisSet.has(i)) src.splice(i, 0, null as any);
    }
    for (const [i, ar] of indices.entries()) {
      if (ar.#source instanceof AluExp) {
        src[axis[i]] = AluExp.cast(
          DType.Int32,
          accessorAluExp(ar.#source, ar.#st, idxAxis),
        );
      } else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        src[axis[i]] = AluExp.cast(
          DType.Int32,
          AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis),
        );
      }
    }

    let exp: AluExp;
    if (this.#source instanceof AluExp) {
      // This is an AluExp, not an actual array, so turn it into an expression.
      exp = accessorAluExp(this.#source, this.#st, src);
    } else {
      let gid = inputs.indexOf(this.#source);
      if (gid === -1) {
        gid = inputs.length;
        inputs.push(this.#source);
      }
      exp = accessorGlobal(this.#dtype, gid, this.#st, src);
    }

    const kernel = Kernel.single(inputs.length, prod(finalShape), exp);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending, ...indices.flatMap((ar) => ar.#pending)];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));

    // Dispose of inputs after creating PendingExecute.
    this.dispose();
    for (const ar of indices) ar.dispose();

    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(finalShape),
      pending,
    });
  }

  /** Move axes to the rightmost dimension of the shape. */
  #moveAxesDown(axis: number[]): Array {
    this.#check();
    if (axis.length === 0) return this.reshape(this.shape.concat(1));
    const newShape: number[] = [];
    const keptAxes: number[] = [];
    const shiftedAxes: number[] = [];
    for (let i = 0; i < this.#st.shape.length; i++) {
      if (axis.includes(i)) {
        shiftedAxes.push(i);
      } else {
        keptAxes.push(i);
        newShape.push(this.#st.shape[i]);
      }
    }
    newShape.push(-1);
    return this.#transpose(keptAxes.concat(shiftedAxes)).reshape(newShape);
  }

  #transpose(perm: number[]): Array {
    this.#check();
    if (!isPermutation(perm, this.ndim))
      throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
    return this.#reshape(this.#st.permute(perm));
  }

  #unary(op: AluOp, dtypeOutput?: DType) {
    const weakType = !dtypeOutput && this.#weakType;
    dtypeOutput ??= this.#dtype; // Default to current dtype unless changed.

    this.#check();
    // Short circuit if the array is already AluExp.
    if (this.#source instanceof AluExp) {
      const exp = new AluExp(op, dtypeOutput, [this.#source]);
      this.dispose();
      return this.#newArrayFrom({
        source: exp.simplify(),
        dtype: dtypeOutput,
        weakType,
      });
    }

    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    const exp = new AluExp(op, dtypeOutput, [
      AluExp.globalView(this.#dtype, 0, this.#st, indices),
    ]);
    const kernel = Kernel.single(1, this.#st.size, exp);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(
      new PendingExecute(this.#backend, kernel, [this.#source], [output]),
    );

    this.dispose(); // Dispose of inputs after creating PendingExecute.
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(this.shape),
      dtype: dtypeOutput,
      weakType,
      pending,
    });
  }

  #binary(op: AluOp, other: Array): Array {
    const custom = (src: AluExp[]) => new AluExp(op, src[0].dtype, src);
    return Array.#naryCustom(op, custom, [this, other]);
  }

  static #naryCustom(
    name: string,
    custom: (src: AluExp[]) => AluExp,
    arrays: Array[],
    {
      dtypeOverride,
      strongTypeOutput,
      reduceAxis,
    }: {
      dtypeOverride?: (DType | undefined)[];
      strongTypeOutput?: boolean;
      reduceAxis?: boolean;
    } = {},
  ): Array {
    const n = arrays.length;
    if (n === 0) throw new TypeError(`No inputs for ${name}`);

    for (const ar of arrays) ar.#check();

    let castDtype: DType | undefined;
    let castWeakType = true;
    for (let i = 0; i < n; i++) {
      if (dtypeOverride?.[i]) {
        if (arrays[i].#dtype !== dtypeOverride[i]) {
          throw new TypeError(
            `Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`,
          );
        }
      } else {
        // Try to cast with dtype of other arguments in the operation.
        if (castDtype === undefined) {
          castDtype = arrays[i].#dtype;
          castWeakType = arrays[i].#weakType;
        } else {
          ({ dtype: castDtype, weakType: castWeakType } = promoteAvals(
            new ShapedArray([], castDtype, castWeakType),
            arrays[i].aval.scalar(),
          ));
        }
      }
    }
    const weakType = castWeakType && !strongTypeOutput;

    const { backend, committed } = Array.#computeBackend(name, arrays);
    arrays = arrays.map((ar) => ar._putSync(backend));
    arrays = Array.#broadcastArrays(arrays);
    const newShape = [...arrays[0].shape];

    // Short circuit if all are already AluExp.
    if (arrays.every((ar) => ar.#source instanceof AluExp) && !reduceAxis) {
      const sources = arrays.map((ar, i) => {
        if (!dtypeOverride?.[i]) {
          return AluExp.cast(castDtype!, ar.#source as AluExp);
        } else {
          return ar.#source as AluExp;
        }
      });
      if (arrays.every((ar) => deepEqual(ar.#st, arrays[0].#st))) {
        // All are AluExp and have the same shape tracker.
        const exp = custom(sources);
        arrays.forEach((ar) => ar.dispose());
        return new Array({
          source: exp.simplify(),
          st: arrays[0].#st,
          dtype: exp.dtype,
          weakType: weakType,
          backend,
          committed,
        });
      }
      // If their shape trackers are different, we need to normalize them.
      const exp = custom(
        arrays.map((ar, i) => {
          const src = sources[i];
          if (ar.#st.contiguous) return src;
          return accessorAluExp(src, ar.#st, unravelAlu(newShape, AluVar.idx));
        }),
      );
      const st = ShapeTracker.fromShape(newShape);
      arrays.forEach((ar) => ar.dispose());
      return new Array({
        source: exp.simplify(),
        st,
        dtype: exp.dtype,
        weakType,
        backend,
        committed,
      });
    }

    let indices: AluExp[];
    if (!reduceAxis) {
      indices = unravelAlu(newShape, AluVar.gidx);
    } else {
      const contractedShape = newShape.slice(0, -1);
      indices = [...unravelAlu(contractedShape, AluVar.gidx), AluVar.ridx];
    }

    const inputs: Slot[] = [];
    const src: AluExp[] = [];
    for (const [i, ar] of arrays.entries()) {
      let nextSrc: AluExp;
      if (ar.#source instanceof AluExp) {
        nextSrc = accessorAluExp(ar.#source, ar.#st, indices);
      } else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        nextSrc = AluExp.globalView(ar.#dtype, gid, ar.#st, indices);
      }
      if (!dtypeOverride?.[i]) nextSrc = AluExp.cast(castDtype!, nextSrc);
      src.push(nextSrc);
    }

    const exp = custom(src);
    let re: Reduction | undefined = undefined;
    if (reduceAxis) {
      const [axisSize] = newShape.splice(-1, 1); // Remove the contracted axis.
      re = new Reduction(exp.dtype, AluOp.Add, axisSize);
    }
    const kernel = Kernel.single(inputs.length, prod(newShape), exp, re);
    const output = backend.malloc(kernel.bytes);
    const pending = new Set([...arrays.flatMap((ar) => ar.#pending)]);
    for (const exe of pending) exe.updateRc(+1);
    pending.add(new PendingExecute(backend, kernel, inputs, [output]));

    arrays.forEach((ar) => ar.dispose()); // Dispose of inputs after creating PendingExecute.
    return new Array({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      dtype: kernel.dtype,
      weakType,
      backend,
      committed,
      pending,
    });
  }

  /** Reduce the last dimension of the array by an operation. */
  #reduce(op: AluOp): Array {
    const shape = this.shape;
    const reduction = new Reduction(this.#dtype, op, shape[shape.length - 1]);
    const newShape = shape.slice(0, -1); // first n-1 axes are in the shape
    const newSize = prod(newShape);

    const indices = [...unravelAlu(newShape, AluVar.gidx), AluVar.ridx];

    let exp: AluExp;
    const inputs: Slot[] = [];
    if (this.#source instanceof AluExp) {
      exp = accessorAluExp(this.#source, this.#st, indices);
    } else {
      inputs.push(this.#source);
      exp = accessorGlobal(this.#dtype, 0, this.#st, indices);
    }

    const kernel = Kernel.single(inputs.length, newSize, exp, reduction);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));

    this.dispose(); // Dispose of inputs after creating PendingExecute.
    return this.#newArrayFrom({
      source: output,
      st: ShapeTracker.fromShape(newShape),
      pending,
    });
  }

  /** Apply an operation with custom lowering to this array. */
  static #routine<P extends Primitive>(prim: P): ImplRule<P> {
    return (arrays: Array[], params: PrimitiveParams<P>) => {
      const { backend, committed } = Array.#computeBackend(prim, arrays);
      for (const ar of arrays) ar.#realize();

      const avals = arrays.map((ar) => ar.aval);
      const avalsOut = abstractEvalRules[prim](avals, params);
      const routine = new Routine(
        routinePrimitives.get(prim)!,
        {
          inputShapes: avals.map((a) => a.shape),
          inputDtypes: avals.map((a) => a.dtype),
          outputShapes: avalsOut.map((a) => a.shape),
          outputDtypes: avalsOut.map((a) => a.dtype),
        },
        params,
      );

      const inputs = arrays.map((ar) => ar.#source as Slot);
      const outputs = avalsOut.map((x) =>
        backend.malloc(byteWidth(x.dtype) * x.size),
      );
      const pending = arrays.flatMap((ar) => ar.#pending);
      for (const exe of pending) exe.updateRc(+outputs.length);
      pending.push(new PendingExecute(backend, routine, inputs, outputs));
      pending[pending.length - 1].updateRc(+outputs.length - 1);

      arrays.forEach((ar) => ar.dispose()); // Dispose of inputs after creating PendingExecute.
      return outputs.map(
        (output, i) =>
          new Array({
            source: output,
            st: ShapeTracker.fromShape(avalsOut[i].shape),
            dtype: avalsOut[i].dtype,
            weakType: avalsOut[i].weakType,
            backend,
            committed,
            pending,
          }),
      );
    };
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
    this.#check();
    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    if (this.#source instanceof AluExp) {
      const exp = accessorAluExp(this.#source, this.#st, indices);
      const kernel = Kernel.single(0, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.bytes);
      const pendingItem = new PendingExecute(
        this.#backend,
        kernel,
        [],
        [output],
      );
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet = new Set([pendingItem]);
    } else {
      // Only realize if the ShapeTracker is non-contiguous.
      if (this.#st.contiguous) return;
      const exp = accessorGlobal(this.#dtype, 0, this.#st, indices);
      const kernel = Kernel.single(1, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.bytes);
      const pendingItem = new PendingExecute(
        this.#backend,
        kernel,
        [this.#source],
        [output],
      );
      this.#backend.decRef(this.#source);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet ??= new Set();
      this.#pendingSet.add(pendingItem);
    }
  }

  #dataInline(): DataArray {
    this.#check();
    if (!(this.#source instanceof AluExp))
      throw new Error("internal: #dataInline called on non-AluExp source");
    const ar = this.#newArrayFrom({ backend: getBackend("cpu") });
    this.dispose();
    return ar.dataSync();
  }

  static #broadcastArrays(arrays: Array[]): Array[] {
    // Broadcast all arrays to the same shape.
    if (arrays.length === 0)
      throw new Error("Need at least one array to broadcast");
    if (arrays.length === 1) return arrays;
    const newShape = arrays.map((a) => a.shape).reduce(generalBroadcast);
    return arrays.map((ar) => {
      if (deepEqual(ar.shape, newShape)) return ar;
      return ar.#reshape(
        ar.#st.broadcast(newShape, range(newShape.length - ar.ndim)),
      );
    });
  }

  static #computeBackend(
    name: string,
    arrays: Array[],
  ): {
    backend: Backend;
    committed: boolean;
  } {
    // First, check if any arrays are committed.
    const committed = arrays.filter((ar) => ar.#committed);
    if (committed.length > 0) {
      const backend = committed[0].#backend;
      for (const ar of committed) {
        if (ar.#backend !== backend) {
          throw new Error(
            `Device mismatch in ${name} between committed arrays on ` +
              `(${backend.type}, ${ar.#backend.type}), ` +
              `please move to the same device with devicePut()`,
          );
        }
      }
      return { backend, committed: true };
    } else {
      // No committed arrays, pick the backend of the first operand.
      const backend = arrays.length > 0 ? arrays[0].#backend : getBackend();
      return { backend, committed: false };
    }
  }

  /** Realize the array and return it as data. */
  async data(): Promise<DataArray> {
    if (
      this.#source instanceof AluExp &&
      this.size < inlineArrayLimit &&
      this.device !== "cpu"
    ) {
      return this.#dataInline();
    }
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    // While the array is contiguous, it might not be the whole buffer.
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = await this.#backend.read(this.#source as Slot, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }

  /**
   * Wait for this array to finish evaluation.
   *
   * Operations and data loading in jax-js are lazy, so this function ensures
   * that pending operations are dispatched and fully executed before it
   * returns.
   *
   * If you are mapping from `data()` or `dataSync()`, it will also trigger
   * dispatch of operations as well.
   *
   * **Note:** `jax.blockUntilReady()` is a higher-level API, it calls this
   * asynchronously for multiple arrays.
   */
  async blockUntilReady(): Promise<Array> {
    this.#check();
    if (this.#source instanceof AluExp) return this;
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    await this.#backend.read(this.#source, 0, 0);
    return this;
  }

  /**
   * Realize the array and return it as data. This is a sync variant and not
   * recommended for performance reasons, as it will block rendering.
   */
  dataSync(): DataArray {
    if (
      this.#source instanceof AluExp &&
      this.size < inlineArrayLimit &&
      this.device !== "cpu"
    ) {
      return this.#dataInline();
    }
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync();
      p.submit();
    }
    // While the array is contiguous, it might not be the whole buffer.
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = this.#backend.readSync(this.#source as Slot, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }

  /**
   * Convert this array into a JavaScript object.
   *
   * This is a blocking operation that will compile all of the shaders and wait
   * for execution to complete, synchronously. No other JavaScript code on the
   * site will be run during shader execution.
   *
   * To avoid blocking, prefer `jsAsync()` when possible.
   */
  js() {
    return dataToJs(this.dtype, this.dataSync(), this.shape);
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return dataToJs(this.dtype, await this.data(), this.shape);
  }

  /**
   * Copy an element of an array to a numeric scalar and return it.
   *
   * Throws an error if the array does not have a single element. The array must
   * either be rank-0, or all dimensions of the shape are 1.
   */
  item(): number {
    if (this.size !== 1) {
      throw new Error(`item() can only be called on arrays of size 1`);
    }
    return this.dataSync()[0];
  }

  /**
   * Convenience accessor for index/update-like usage: `arr.at(i).set(src)`.
   * Currently supports a single `axis=0` offset and integer index. Returns
   * a new array where the slice at `index` along axis 0 is replaced by `src`.
   */
  at(index: number) {
    const that = this;
    return {
      set: (src: TracerValue) => dynamicUpdateSlice(that, src, index),
    };
  }

  static #stackScanYs(ySlices: Array[][], reverse: boolean): Array[] {
    return ySlices.map((slices) => {
      const reshaped = slices.map((s) => {
        const expanded = s.ref.#reshape(s.#st.reshape([1, ...s.shape]));
        s.dispose();
        return expanded;
      });
      let stacked = reshaped[0];
      for (let i = 1; i < reshaped.length; i += 6) {
        const chunk = reshaped.slice(i, i + 6);
        stacked = coreConcatenate([stacked, ...chunk], 0) as Array;
      }
      if (reverse) {
        const flipArg = rep(stacked.ndim, false);
        flipArg[0] = true;
        const flipped = stacked.ref.#reshape(stacked.#st.flip(flipArg));
        stacked.dispose();
        return flipped;
      }
      return stacked;
    });
  }

  static #runScanFallbackLoop(params: {
    length: number;
    reverse: boolean;
    numCarry: number;
    numY: number;
    xs: Array[];
    initCarry: Array[];
    bodyOutAvals: ShapedArray[];
    runBody: (carry: Array[], xSlice: Array[], iter: number) => Array[];
    writeY?: (
      writeIndex: number,
      ySlice: Array[],
      yStrideBytes: number[],
    ) => void;
    onBeforeCarryDispose?: (oldCarry: Array[], ySlice: Array[]) => void;
    disposeXSlices: boolean;
  }): { carry: Array[]; ySlices: Array[][]; usedDirectWrite: boolean } {
    const {
      length,
      reverse,
      numCarry,
      numY,
      xs,
      initCarry,
      bodyOutAvals,
      runBody,
      writeY,
      onBeforeCarryDispose,
      disposeXSlices,
    } = params;

    const canDirectWriteY = numY > 0 && !!writeY;
    const yStrideBytes = bodyOutAvals
      .slice(numCarry)
      .map((aval) => prod(aval.shape) * byteWidth(aval.dtype));

    const ySlices: Array[][] = [];
    if (!canDirectWriteY) {
      for (let j = 0; j < numY; j++) ySlices.push([]);
    }

    let carry = initCarry;

    for (let i = 0; i < length; i++) {
      const dataIdx = reverse ? length - 1 - i : i;
      const xSlice = xs.map((x) => {
        const slicePairs: Pair[] = x.shape.map((s, axis) =>
          axis === 0 ? [dataIdx, dataIdx + 1] : [0, s],
        );
        const squeezedShape = x.shape.slice(1);
        return x.ref.#reshape(x.#st.shrink(slicePairs).reshape(squeezedShape));
      });

      const outs = runBody(carry, xSlice, i);

      const newCarry = outs.slice(0, numCarry);
      const ySlice = outs.slice(numCarry);

      if (canDirectWriteY) {
        const writeIndex = reverse ? length - 1 - i : i;
        writeY!(writeIndex, ySlice, yStrideBytes);
      } else {
        for (let j = 0; j < numY; j++) {
          ySlices[j].push(ySlice[j]);
        }
      }

      if (i > 0 && onBeforeCarryDispose) {
        onBeforeCarryDispose(carry, ySlice);
      }

      if (i > 0) {
        carry.forEach((c) => c.dispose());
      }
      carry = newCarry;

      if (canDirectWriteY) {
        ySlice.forEach((y) => y.dispose());
      }

      if (disposeXSlices) {
        xSlice.forEach((x) => x.dispose());
      }
    }

    return { carry, ySlices, usedDirectWrite: canDirectWriteY };
  }

  //
  // Internal methods follow, not public API. Do not use.
  //

  /** @private Internal plumbing method for Array / Tracer ops. */
  static _implRules(): typeof implRules {
    return {
      [Primitive.Add]([x, y]) {
        return [x.#binary(AluOp.Add, y)];
      },
      [Primitive.Mul]([x, y]) {
        return [x.#binary(AluOp.Mul, y)];
      },
      [Primitive.Idiv]([x, y]) {
        return [x.#binary(AluOp.Idiv, y)];
      },
      [Primitive.Mod]([x, y]) {
        return [x.#binary(AluOp.Mod, y)];
      },
      [Primitive.Min]([x, y]) {
        return [x.#binary(AluOp.Min, y)];
      },
      [Primitive.Max]([x, y]) {
        return [x.#binary(AluOp.Max, y)];
      },
      [Primitive.Neg]([x]) {
        return [zerosLike(x.ref).#binary(AluOp.Sub, x)];
      },
      [Primitive.Reciprocal]([x]) {
        return [x.#unary(AluOp.Reciprocal)];
      },
      [Primitive.Floor]([x]) {
        return [x.#unary(AluOp.Floor)];
      },
      [Primitive.Ceil]([x]) {
        return [x.#unary(AluOp.Ceil)];
      },
      [Primitive.StopGradient]([x]) {
        return [x]; // Stop gradient is a no-op, just return the input.
      },
      [Primitive.Cast]([x], { dtype }) {
        return [x.#unary(AluOp.Cast, dtype)];
      },
      [Primitive.Bitcast]([x], { dtype }) {
        if (x.dtype === DType.Bool || dtype === DType.Bool) {
          throw new TypeError("Bitcast to/from bool is not allowed");
        }
        if (x.dtype === dtype) return [x];
        if (byteWidth(x.dtype) !== byteWidth(dtype)) {
          throw new TypeError(
            `Bitcast from ${x.dtype} to ${dtype} with different byte width`,
          );
        }
        if (x.#source instanceof AluExp) {
          return [x.#unary(AluOp.Bitcast, dtype)];
        } else {
          // Just keep the same data / source, but change the dtype.
          x.#backend.incRef(x.#source);
          const pending = x.#pending;
          for (const exe of pending) exe.updateRc(+1);
          const y = x.#newArrayFrom({ dtype, weakType: false, pending });
          x.dispose();
          return [y];
        }
      },
      [Primitive.Sin]([x]) {
        return [x.#unary(AluOp.Sin)];
      },
      [Primitive.Cos]([x]) {
        return [x.#unary(AluOp.Cos)];
      },
      [Primitive.Asin]([x]) {
        return [x.#unary(AluOp.Asin)];
      },
      [Primitive.Atan]([x]) {
        return [x.#unary(AluOp.Atan)];
      },
      [Primitive.Exp]([x]) {
        return [x.#unary(AluOp.Exp)];
      },
      [Primitive.Log]([x]) {
        return [x.#unary(AluOp.Log)];
      },
      [Primitive.Erf]([x]) {
        return [x.#unary(AluOp.Erf)];
      },
      [Primitive.Erfc]([x]) {
        return [x.#unary(AluOp.Erfc)];
      },
      [Primitive.Sqrt]([x]) {
        return [x.#unary(AluOp.Sqrt)];
      },
      [Primitive.Reduce]([x], { op, axis }) {
        if (axis.length === 0) return [x];
        return [x.#moveAxesDown(axis).#reduce(op)];
      },
      [Primitive.Pool]([x], { window, strides }) {
        const st = pool(x.#st, window, strides);
        return [x.#reshape(st)];
      },
      [Primitive.PoolTranspose]([x], { inShape, window, strides }) {
        const n = inShape.length;
        let st = poolTranspose(x.#st, inShape, window, strides);
        st = st.reshape([...st.shape.slice(0, n), prod(st.shape.slice(n))]);
        return [x.#reshape(st).#reduce(AluOp.Add)];
      },
      [Primitive.Dot]([x, y]) {
        return [
          Array.#naryCustom(
            "dot",
            ([x, y]: AluExp[]) => AluExp.mul(x, y),
            [x, y],
            { reduceAxis: true },
          ),
        ];
      },
      [Primitive.Conv]([x, y], params) {
        checkConvShape(x.shape, y.shape, params);
        const [stX, stY] = prepareConv(x.#st, y.#st, params);
        return [
          Array.#naryCustom(
            "conv",
            ([x, y]: AluExp[]) => AluExp.mul(x, y),
            [x.#reshape(stX), y.#reshape(stY)],
            { reduceAxis: true },
          ),
        ];
      },
      [Primitive.Compare]([x, y], { op }) {
        const custom = ([x, y]: AluExp[]) => aluCompare(x, y, op);
        return [
          Array.#naryCustom("compare", custom, [x, y], {
            strongTypeOutput: true, // outputs strongly typed bool
          }),
        ];
      },
      [Primitive.Where]([cond, x, y]) {
        const custom = ([cond, x, y]: AluExp[]) => AluExp.where(cond, x, y);
        return [
          Array.#naryCustom("where", custom, [cond, x, y], {
            dtypeOverride: [DType.Bool],
          }),
        ];
      },
      [Primitive.Concatenate](xs, { axis }) {
        // Lower to a pad of all the arrays onto the final size, then sum.
        const ndim = xs[0].ndim;
        const sizes = xs.map((x) => x.shape[axis]);
        const finalSize = sizes.reduce((a, b) => a + b, 0);
        const makePadAxis = (start: number, end: number): [number, number][] =>
          range(ndim).map((i) => (i === axis ? [start, end] : [0, 0]));
        let cum = 0;
        const xsPadded: Array[] = [];
        for (let i = 0; i < xs.length; i++) {
          const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
          xsPadded.push(xs[i].#reshape(xs[i].#st.pad(padding)));
          cum += sizes[i];
        }
        const custom = (exps: AluExp[]) => exps.reduce(AluExp.add);
        return [Array.#naryCustom("concatenate", custom, xsPadded)];
      },
      [Primitive.Split]([x], { axis, sizes }) {
        const outputs: Array[] = [];
        for (let i = 0, start = 0; i < sizes.length; i++) {
          const slice = range(x.ndim).map<Pair>((d) =>
            d === axis ? [start, start + sizes[i]] : [0, x.shape[d]],
          );
          outputs.push(x.ref.#reshape(x.#st.shrink(slice)));
          start += sizes[i];
        }
        x.dispose();
        return outputs;
      },
      [Primitive.RandomBits]([k0, k1], { shape, mode }) {
        const keyShape = k0.shape;
        const genShape = shape.slice(keyShape.length);
        // Arrays of size >2^32 won't fit into browser memory anyway, so it's
        // okay to take lazy iota this way for counters.
        const c0 = zeros(genShape, { dtype: DType.Uint32, device: k0.device });
        const c1 = arange(0, prod(genShape), 1, {
          dtype: DType.Uint32,
          device: k0.device,
        }).reshape(genShape);
        k0 = k0.#reshape(
          k0.#st.reshape(keyShape.concat(rep(genShape.length, 1))),
        );
        k1 = k1.#reshape(
          k1.#st.reshape(keyShape.concat(rep(genShape.length, 1))),
        );
        const custom = ([k0, k1, c0, c1]: AluExp[]) =>
          AluExp.threefry2x32(k0, k1, c0, c1, mode);
        return [Array.#naryCustom("random_bits", custom, [k0, k1, c0, c1])];
      },
      [Primitive.Gather]([x, ...indices], { axis, outDim }) {
        return [x.#gather(indices, axis, outDim)];
      },
      [Primitive.Transpose]([x], { perm }) {
        return [x.#transpose(perm)];
      },
      [Primitive.Broadcast]([x], { shape, axis }) {
        return [x.#reshape(x.#st.broadcast(shape, axis))];
      },
      [Primitive.Reshape]([x], { shape }) {
        return [x.#reshape(x.#st.reshape(shape))];
      },
      [Primitive.Flip]([x], { axis }) {
        const arg = rep(x.ndim, false);
        for (const ax of axis) arg[ax] = true;
        return [x.#reshape(x.#st.flip(arg))];
      },
      [Primitive.Shrink]([x], { slice }) {
        return [x.#reshape(x.#st.shrink(slice))];
      },
      [Primitive.Pad]([x], { width }) {
        return [x.#reshape(x.#st.pad(width))];
      },
      [Primitive.DynamicUpdateSlice]([dst, src], { offset, axis }) {
        const dstShape = dst.shape;
        const srcShape = src.shape;

        // Mode 1: same rank, axis in range
        if (dstShape.length === srcShape.length) {
          for (let i = 0; i < dstShape.length; i++) {
            if (i === axis) continue;
            if (dstShape[i] !== srcShape[i]) {
              throw new Error(
                "dynamicUpdateSlice: dst and src must match on non-updated axes",
              );
            }
          }
          if (offset + srcShape[axis] > dstShape[axis]) {
            throw new Error(
              "dynamicUpdateSlice: offset + src.shape[axis] out of bounds",
            );
          }
          const innerBefore = prod(dstShape.slice(axis + 1));
          const outerBefore = prod(dstShape.slice(0, axis));
          const dstData = dst.dataSync();
          const srcData = src.dataSync();
          for (let out = 0; out < outerBefore; out++) {
            for (let i = 0; i < srcShape[axis]; i++) {
              const srcStart = (out * srcShape[axis] + i) * innerBefore;
              const dstStart =
                (out * dstShape[axis] + offset + i) * innerBefore;
              dstData.set(
                srcData.subarray(srcStart, srcStart + innerBefore),
                dstStart,
              );
            }
          }
          return [
            array(dstData, {
              shape: dstShape,
              dtype: dst.dtype,
              device: dst.device,
            }),
          ];
        }

        // Mode 2: stacked dst at axis=0
        if (axis === 0 && dstShape.length === srcShape.length + 1) {
          for (let i = 0; i < srcShape.length; i++) {
            if (dstShape[i + 1] !== srcShape[i]) {
              throw new Error(
                "dynamicUpdateSlice: dst and src must match on non-updated axes (stacked mode)",
              );
            }
          }
          if (offset + 1 > dstShape[0]) {
            throw new Error(
              "dynamicUpdateSlice: offset out of bounds for stacked dst",
            );
          }

          // Fast path: WebGPU buffer copy when both src/dst are on WebGPU
          if (dst.device === "webgpu" && src.device === "webgpu") {
            const dstSlot = dst._realizeSource();
            const srcSlot = src._realizeSource();
            const innerSizeBytes = prod(srcShape) * byteWidth(dst.dtype);
            const dstOffsetBytes = offset * innerSizeBytes;
            const backend = getBackend("webgpu");
            const canBufferCopy =
              dstOffsetBytes % 4 === 0 && innerSizeBytes % 4 === 0;
            if (backend.copyBufferToBuffer && canBufferCopy) {
              backend.copyBufferToBuffer(
                srcSlot,
                0,
                dstSlot,
                dstOffsetBytes,
                innerSizeBytes,
              );
              backend.incRef(dstSlot);
              return [
                new Array({
                  source: dstSlot,
                  st: ShapeTracker.fromShape(dstShape),
                  dtype: dst.dtype,
                  weakType: dst.weakType,
                  backend: getBackend(dst.device),
                  committed: true,
                  pending: [],
                }),
              ];
            } else if (backend.copyBufferWithShader) {
              // Use on-device copy shader when direct buffer copy is unavailable
              backend.copyBufferWithShader(
                srcSlot,
                0,
                dstSlot,
                dstOffsetBytes,
                innerSizeBytes,
              );
              backend.incRef(dstSlot);
              return [
                new Array({
                  source: dstSlot,
                  st: ShapeTracker.fromShape(dstShape),
                  dtype: dst.dtype,
                  weakType: dst.weakType,
                  backend: getBackend(dst.device),
                  committed: true,
                  pending: [],
                }),
              ];
            }
          }

          const dstData = dst.dataSync();
          const srcData = src.dataSync();
          const innerSize = prod(srcShape);
          const dstStart = offset * innerSize;
          dstData.set(srcData as any, dstStart);
          return [
            array(dstData, {
              shape: dstShape,
              dtype: dst.dtype,
              device: dst.device,
            }),
          ];
        }

        throw new Error(
          "dynamicUpdateSlice: unsupported dst/src shapes for update",
        );
      },
      [Primitive.Sort]: Array.#routine(Primitive.Sort),
      [Primitive.Argsort]: Array.#routine(Primitive.Argsort),
      [Primitive.TriangularSolve]: Array.#routine(Primitive.TriangularSolve),
      [Primitive.Cholesky]: Array.#routine(Primitive.Cholesky),
      [Primitive.LU]: Array.#routine(Primitive.LU),
      [Primitive.Jit](args, { jaxpr }) {
        if (jaxpr.inBinders.length !== args.length) {
          throw new Error(
            `jit expects ${jaxpr.inBinders.length} args, got ${args.length}`,
          );
        }

        const { backend, committed } = Array.#computeBackend("jit", args);
        args = args.map((ar) => ar._putSync(backend));

        const jp = jitCompile(backend, jaxpr);

        // Create scanRunner callback that executes scan body using compiled bodyProgram.
        // This is the JS-loop fallback path. It has per-iteration overhead from:
        // - Creating Array wrappers for slots
        // - ShapeTracker slicing for xs
        // - Object allocation inside the loop
        // The fused paths (compiled-loop, preencoded-routine) avoid this overhead entirely
        // by running the entire loop in WASM or GPU shader code.
        const scanRunner: ScanRunner = (
          bodyProgram,
          _backend,
          bodyJaxpr,
          length,
          numCarry,
          _numConsts,
          _numX,
          numY,
          reverse,
          constSlots,
          initCarrySlots,
          xsSlots,
          xsAvals, // xs avals passed from scan step (correct after transforms like vmap)
          outputSlots,
        ) => {
          // Get avals from bodyJaxpr for wrapping slots
          const carryAvals = bodyJaxpr.inBinders
            .slice(constSlots.length, constSlots.length + numCarry)
            .map((v) => v.aval);

          // OPTIMIZATION: Const slots are already realized, use directly
          const constSlotsRealized = constSlots as number[];

          // Wrap xs slots as Arrays for ShapeTracker slicing
          // Use xsAvals from the scan step (these have the correct shapes after transforms)
          const xs = xsSlots.map(
            (slot, i) =>
              new Array({
                source: slot,
                st: ShapeTracker.fromShape(xsAvals[i].shape),
                dtype: xsAvals[i].dtype,
                weakType: xsAvals[i].weakType,
                backend,
                committed,
                pending: [],
              }),
          );

          // Initialize carry as Arrays from slots
          let carry = initCarrySlots.map(
            (slot, i) =>
              new Array({
                source: slot,
                st: ShapeTracker.fromShape(carryAvals[i].shape),
                dtype: carryAvals[i].dtype,
                weakType: carryAvals[i].weakType,
                backend,
                committed,
                pending: [],
              }),
          );

          const bodyOutAvals = bodyJaxpr.outs.map((v) => v.aval);
          const canDirectWriteY =
            numY > 0 &&
            outputSlots.length === numCarry + numY &&
            (backend.copyBufferToBuffer || backend.copyBufferWithShader);

          const writeY = canDirectWriteY
            ? (writeIndex: number, ySlice: Array[], yStrideBytes: number[]) => {
                for (let j = 0; j < numY; j++) {
                  const sizeBytes = yStrideBytes[j];
                  if (sizeBytes <= 0) continue;
                  const dstOffsetBytes = writeIndex * sizeBytes;
                  const ySlot = ySlice[j]._realizeSource();
                  const dstSlot = outputSlots[numCarry + j];
                  const canBufferCopy =
                    dstOffsetBytes % 4 === 0 && sizeBytes % 4 === 0;
                  if (backend.copyBufferToBuffer && canBufferCopy) {
                    backend.copyBufferToBuffer(
                      ySlot,
                      0,
                      dstSlot,
                      dstOffsetBytes,
                      sizeBytes,
                    );
                  } else if (backend.copyBufferWithShader) {
                    backend.copyBufferWithShader(
                      ySlot,
                      0,
                      dstSlot,
                      dstOffsetBytes,
                      sizeBytes,
                    );
                  }
                }
              }
            : undefined;

          const onBeforeCarryDispose = (oldCarry: Array[], ySlice: Array[]) => {
            const oldCarrySlots = new Set(
              oldCarry.map((c) => c._realizeSource()),
            );
            for (const y of ySlice) {
              const slot = y._realizeSource();
              if (oldCarrySlots.has(slot)) {
                backend.incRef(slot);
              }
            }
          };

          const runBody = (curCarry: Array[], xSlice: Array[], i: number) => {
            const carrySlots = curCarry.map((c) => c._realizeSource());
            const xSliceSlots = xSlice.map((x) => x._realizeSource());

            for (const x of xSlice) {
              for (const exe of x.#pending) {
                exe.prepareSync();
                exe.submit();
              }
            }

            if (DEBUG >= 2) {
              console.log(
                `[scanRunner] iter ${i}: carrySlots=${carrySlots}, xSliceSlots=${xSliceSlots}`,
              );
            }

            const { outputs: bodyOuts, pending } = bodyProgram.execute([
              ...constSlotsRealized,
              ...carrySlots,
              ...xSliceSlots,
            ]);

            if (DEBUG >= 2) {
              console.log(
                `[scanRunner] iter ${i}: bodyOuts=${bodyOuts}, pending.length=${pending.length}`,
              );
            }

            for (const exe of pending) {
              exe.prepareSync();
              exe.submit();
            }

            const seenSlots = new Set<Slot>();
            return bodyOuts.map((slot, j) => {
              if (seenSlots.has(slot)) {
                backend.incRef(slot);
              } else {
                seenSlots.add(slot);
              }
              return new Array({
                source: slot,
                st: ShapeTracker.fromShape(bodyOutAvals[j].shape),
                dtype: bodyOutAvals[j].dtype,
                weakType: bodyOutAvals[j].weakType,
                backend,
                committed,
                pending: [],
              });
            });
          };

          const loopResult = Array.#runScanFallbackLoop({
            length,
            reverse,
            numCarry,
            numY,
            xs,
            initCarry: carry,
            bodyOutAvals,
            runBody,
            writeY,
            onBeforeCarryDispose,
            disposeXSlices: true,
          });

          carry = loopResult.carry;

          let stackedYs: Array[] = [];
          if (!loopResult.usedDirectWrite) {
            stackedYs = Array.#stackScanYs(loopResult.ySlices, reverse);
          }

          // Realize outputs to get final slots
          const carryOutSlots = carry.map((c) => c._realizeSource());
          const yOutSlots = canDirectWriteY
            ? outputSlots.slice(numCarry)
            : stackedYs.map((y) => y._realizeSource());

          // CRITICAL: Collect pending ops from carry and stackedYs
          // The stacking via coreConcatenate creates new pending ops that are held by stackedYs.
          // We must include these in the returned pending, otherwise the concat kernels won't run.
          // Note: Body pending ops are already submitted, so we only need carry/ys pending.
          const carryPending = carry.flatMap((c) => c.#pending);
          const ysPending = stackedYs.flatMap((y) => y.#pending);
          const finalPending = [...carryPending, ...ysPending];

          // NOTE: We DON'T dispose carry or stackedYs because their pending ops would be cancelled,
          // causing double-decRef on output slots. The pending ops are returned in finalPending,
          // and the caller creates new Arrays from the returned slots.
          // The old Array objects will be garbage collected, and since they have pending ops
          // that are still referenced via finalPending, those ops won't be cancelled.

          const outputs = [...carryOutSlots, ...yOutSlots];
          if (outputSlots.length > 0) {
            const outputSet = new Set(outputs);
            for (const slot of outputSlots) {
              if (!outputSet.has(slot)) backend.decRef(slot);
            }
          }

          return {
            outputs,
            pending: finalPending,
          };
        };

        // Realize inputs and collect their pending operations
        const realizedInputs = args.map((x) => x._realizeSource());
        const prevPending = [...new Set(args.flatMap((x) => x.#pending))];

        // Submit input pending operations BEFORE executing JIT program
        // This is necessary because compiled-loop reads from buffers synchronously
        for (const exe of prevPending) {
          exe.prepareSync();
          exe.submit();
        }

        const { outputs, pending } = jp.execute(realizedInputs, scanRunner);
        for (const exe of pending) exe.updateRc(+outputs.length - 1);

        args.forEach((x) => x.dispose()); // Dispose of args after dispatch.

        return outputs.map((source, i) => {
          return new Array({
            source,
            st: ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
            dtype: jaxpr.outs[i].aval.dtype,
            weakType: jaxpr.outs[i].aval.weakType,
            backend,
            committed,
            pending,
          });
        });
      },
      [Primitive.Scan](args, { jaxpr, numCarry, numConsts, length, reverse }) {
        // Scan primitive: executes jaxpr in a loop, threading carry state
        // Args layout: [...consts, ...initCarry, ...xs]
        // jaxpr inputs: [...consts, ...carry, ...x_slice]
        // jaxpr outputs: [...newCarry, ...y_slice]

        const consts = args.slice(0, numConsts);
        const initCarry = args.slice(numConsts, numConsts + numCarry);
        const xs = args.slice(numConsts + numCarry);

        const numX = xs.length;
        const numY = jaxpr.outs.length - numCarry;

        // Validate jaxpr inputs match expected count
        if (jaxpr.inBinders.length !== numConsts + numCarry + numX) {
          throw new Error(
            `scan jaxpr expects ${jaxpr.inBinders.length} inputs, got ${numConsts + numCarry + numX}`,
          );
        }

        // PRE-CREATE: Const refs (created once, reused each iteration via .ref)
        // Using ref.#reshape with same ST is equivalent to creating a non-consuming copy
        const constViews = consts.map((c) => c.ref.#reshape(c.#st));

        const { backend } = Array.#computeBackend("scan", args);
        const bodyOutAvals = jaxpr.outs.map((v) => v.aval);
        const canDirectWriteY =
          numY > 0 &&
          (backend.copyBufferToBuffer || backend.copyBufferWithShader);

        const preallocatedYs: Array[] = [];
        const preallocatedSlots: Slot[] = [];
        if (canDirectWriteY) {
          const yOutAtoms = jaxpr.outs.slice(numCarry);
          for (const atom of yOutAtoms) {
            const aval = atom.aval;
            const ySizeBytes =
              length * prod(aval.shape) * byteWidth(aval.dtype);
            if (ySizeBytes === 0) {
              const stacked = zeros([length, ...aval.shape], {
                dtype: aval.dtype,
                device: backend.type,
              });
              preallocatedYs.push(stacked);
              preallocatedSlots.push(stacked._realizeSource());
              continue;
            }
            const slot = backend.malloc(ySizeBytes);
            preallocatedSlots.push(slot);
            preallocatedYs.push(
              new Array({
                source: slot,
                st: ShapeTracker.fromShape([length, ...aval.shape]),
                dtype: aval.dtype,
                weakType: aval.weakType,
                backend,
                committed: true,
                pending: [],
              }),
            );
          }
        }

        let carry = initCarry;

        const writeY = canDirectWriteY
          ? (writeIndex: number, ySlice: Array[], yStrideBytes: number[]) => {
              for (let j = 0; j < numY; j++) {
                const sizeBytes = yStrideBytes[j];
                if (sizeBytes <= 0) continue;
                const dstOffsetBytes = writeIndex * sizeBytes;
                const ySlot = ySlice[j]._realizeSource();
                for (const exe of ySlice[j].#pending) {
                  exe.prepareSync();
                  exe.submit();
                }
                const dstSlot = preallocatedSlots[j];
                const canBufferCopy =
                  dstOffsetBytes % 4 === 0 && sizeBytes % 4 === 0;
                if (backend.copyBufferToBuffer && canBufferCopy) {
                  backend.copyBufferToBuffer(
                    ySlot,
                    0,
                    dstSlot,
                    dstOffsetBytes,
                    sizeBytes,
                  );
                } else if (backend.copyBufferWithShader) {
                  backend.copyBufferWithShader(
                    ySlot,
                    0,
                    dstSlot,
                    dstOffsetBytes,
                    sizeBytes,
                  );
                }
              }
            }
          : undefined;

        const runBody = (curCarry: Array[], xSlice: Array[]) => {
          const jaxprInputs = [
            ...constViews.map((c) => c.ref),
            ...curCarry.map((c) => c.ref),
            ...xSlice,
          ];
          return evalJaxpr(jaxpr, jaxprInputs) as Array[];
        };

        const loopResult = Array.#runScanFallbackLoop({
          length,
          reverse,
          numCarry,
          numY,
          xs,
          initCarry: carry,
          bodyOutAvals,
          runBody,
          writeY,
          disposeXSlices: false,
        });

        carry = loopResult.carry;

        // Dispose inputs
        initCarry.forEach((c) => c.dispose());
        xs.forEach((x) => x.dispose());
        consts.forEach((c) => c.dispose());
        constViews.forEach((c) => c.dispose());

        if (loopResult.usedDirectWrite) {
          return [...carry, ...preallocatedYs];
        }

        const stackedYs = Array.#stackScanYs(loopResult.ySlices, reverse);
        return [...carry, ...stackedYs];
      },
    };
  }

  /** @private */
  _realizeSource() {
    this.#realize();
    return this.#source as number; // Because #realize() was called.
  }

  /** @private Put this array on a new backend, asynchronously. */
  async _put(backend: Backend): Promise<Array> {
    if (this.#backend === backend) return this;
    if (this.#source instanceof AluExp) {
      // Not realized yet, just dump the AluExp on the target backend.
      const ar = this.#newArrayFrom({ backend, committed: true });
      this.dispose();
      return ar;
    } else {
      // Realize the array and copy data to the new backend.
      const data = await this.data();
      return arrayFromData(
        data,
        this.shape,
        { dtype: this.#dtype, device: backend.type },
        this.#weakType,
      );
    }
  }

  /** @private Put this array on a new backend, synchronously. */
  _putSync(backend: Backend): Array {
    if (this.#backend === backend) return this;
    if (this.#source instanceof AluExp) {
      // Not realized yet, just dump the AluExp on the target backend.
      const ar = this.#newArrayFrom({ backend, committed: true });
      this.dispose();
      return ar;
    } else {
      // Realize the array and copy data to the new backend.
      const data = this.dataSync();
      return arrayFromData(
        data,
        this.shape,
        { dtype: this.#dtype, device: backend.type },
        this.#weakType,
      );
    }
  }
}

/** Constructor for creating a new array from data. */
export function array(
  values: Array | DataArray | RecursiveArray<number> | RecursiveArray<boolean>,
  { shape, dtype, device }: { shape?: number[] } & DTypeAndDevice = {},
): Array {
  if (values instanceof Tracer) {
    if (shape && !deepEqual(values.shape, shape)) {
      values = values.reshape(shape);
    }
    if (dtype && values.dtype !== dtype) {
      values = values.astype(dtype);
    }
    return values;
  } else if (ArrayBuffer.isView(values)) {
    return arrayFromData(values, shape ?? [values.length], {
      dtype,
      device,
    });
  } else {
    // Assume this is a nested array object, infer the shape.
    if (!shape) {
      shape = [];
      let cur = values;
      while (JsArray.isArray(cur)) {
        shape.push(cur.length);
        cur = cur[0];
      }
    }
    const size = prod(shape);
    const flat = recursiveFlatten(values);
    if (flat.length !== size) {
      throw new Error(
        `Jagged shape: ${JSON.stringify(shape)} vs ${flat.length}`,
      );
    }
    if (size === 0) return zeros(shape, { dtype, device });
    if (size === 1) return full(shape, flat[0], { dtype, device });
    if (typeof flat[0] === "boolean") {
      dtype = dtype ?? DType.Bool;
      const data = new Int32Array(flat.map((x) => (x ? 1 : 0)));
      return arrayFromData(data, shape, { dtype, device });
    } else {
      const weakType = dtype == undefined && shape.length === 0;
      dtype = dtype ?? DType.Float32;
      const data = dtypedJsArray(dtype, flat as number[]);
      return arrayFromData(data, shape, { dtype, device }, weakType);
    }
  }
}

function arrayFromData(
  data: DataArray,
  shape: number[],
  { dtype, device }: DTypeAndDevice,
  weakType = false,
): Array {
  if (data instanceof Float32Array) {
    if (dtype && dtype !== DType.Float32)
      throw new Error("Float32Array must have float32 type");
    dtype ??= DType.Float32;
  } else if (data instanceof Int32Array) {
    if (dtype && dtype !== DType.Int32 && dtype !== DType.Bool)
      throw new Error("Int32Array must have int32 or bool type");
    dtype ??= DType.Int32;
  } else if (data instanceof Uint32Array) {
    if (dtype && dtype !== DType.Uint32)
      throw new Error("Uint32Array must have uint32 type");
    dtype ??= DType.Uint32;
  } else if (data instanceof Float16Array) {
    if (dtype && dtype !== DType.Float16)
      throw new Error("Float16Array must have float16 type");
    dtype ??= DType.Float16;
  } else if (data instanceof Float64Array) {
    if (dtype && dtype !== DType.Float64)
      throw new Error("Float64Array must have float64 type");
    dtype ??= DType.Float64;
  } else {
    throw new Error(
      "Unsupported data array type: " + (data as any).constructor.name,
    );
  }

  if (data.length < inlineArrayLimit) {
    // Check if all elements are of the same value and short-circuit.
    let allEqual = true;
    for (let i = 1; i < data.length; i++) {
      if (data[i] !== data[0]) {
        allEqual = false;
        break;
      }
    }
    if (allEqual) {
      // If all elements are equal, we can use a constant expression.
      const sa = new ShapedArray(shape, dtype, weakType);
      return fullInternal(sa, data[0], device);
    }
  }

  const backend = getBackend(device);
  const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const slot = backend.malloc(data.byteLength, buf);
  return new Array({
    source: slot,
    st: ShapeTracker.fromShape(shape),
    dtype,
    weakType,
    backend,
    committed: device != undefined,
  });
}

function dataToJs(
  dtype: DType,
  data: DataArray,
  shape: number[],
): RecursiveArray<number> | RecursiveArray<boolean> | any {
  if (shape.length === 0) {
    return dtype === DType.Bool ? Boolean(data[0]) : data[0];
  }
  const [first, ...rest] = shape;
  const restSize = prod(rest);
  const ret: any[] = [];
  for (let i = 0; i < first; i++) {
    const subarray = data.slice(i * restSize, (i + 1) * restSize);
    ret.push(dataToJs(dtype, subarray, rest));
  }
  return ret;
}

/** If x is a value, lift it into an array, otherwise leave it be. */
export function pureArray(x: TracerValue): Tracer {
  if (x instanceof Tracer) {
    return x;
  } else {
    return array(x);
  }
}

class EvalTrace extends Trace {
  // No boxing in Tracers needed.
  pure = (x: TracerValue) => pureArray(x);
  lift = (x: Tracer) => x;

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: Array[],
    params: PrimitiveParams<P>,
  ): Tracer[] {
    return implRules[primitive](tracers, params);
  }
}

// Special bottom of the stack: must be level 0.
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));

type ImplRule<P extends Primitive> = (
  tracers: Array[],
  params: PrimitiveParams<P>,
) => Array[];
const implRules: { [P in Primitive]: ImplRule<P> } = Array._implRules();

export function fullInternal(
  aval: AbstractValue,
  fillValue: number | boolean,
  device?: Device,
) {
  return new Array({
    source: AluExp.const(aval.dtype, fillValue),
    st: ShapeTracker.fromShape(aval.shape),
    dtype: aval.dtype,
    weakType: aval.weakType,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

export function zerosLike(val: TracerValue, dtype?: DType): Array {
  return fullLike(val, 0, dtype);
}

export function onesLike(val: TracerValue, dtype?: DType): Array {
  return fullLike(val, 1, dtype);
}

export function fullLike(
  val: TracerValue,
  fillValue: number | boolean | Array,
  dtype?: DType,
): Array {
  const aval = getAval(val);
  if (val instanceof Tracer) val.dispose();
  if (fillValue instanceof Tracer) {
    // TODO: Full can also take an array as a fill value. This is equivalent to
    // expanding the array.
    throw new Error("numpy.fullLike() with array argument not implemented yet");
  }
  const sa = new ShapedArray(aval.shape, dtype ?? aval.dtype, aval.weakType);
  return fullInternal(sa, fillValue);
}

/** Return a new array of given shape and type, filled with zeros. */
export function zeros(
  shape: number[],
  { dtype, device }: DTypeAndDevice = {},
): Array {
  return full(shape, 0, { dtype, device });
}

/** Return a new array of given shape and type, filled with ones. */
export function ones(
  shape: number[],
  { dtype, device }: DTypeAndDevice = {},
): Array {
  return full(shape, 1, { dtype, device });
}

/** Return a new array of given shape and type, filled with `fill_value`. */
export function full(
  shape: number[],
  fillValue: number | boolean | Array,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  let weakType = dtype == undefined && shape.length === 0;
  if (typeof fillValue === "number") {
    dtype = dtype ?? DType.Float32;
  } else if (typeof fillValue === "boolean") {
    dtype = dtype ?? DType.Bool;
    weakType = false; // booleans are never weakly typed
  } else if (fillValue instanceof Tracer) {
    // TODO: Full can also take an array as a fill value. This is equivalent to
    // expanding the array.
    throw new Error("numpy.full() with array argument not implemented yet");
  } else {
    throw new TypeError(`Invalid type for full: ${fillValue}`);
  }
  return fullInternal(
    new ShapedArray(shape, dtype, weakType),
    fillValue,
    device,
  );
}

/**
 * Create an identity matrix.
 *
 * If numCols is not provided, it defaults to numRows, i.e., a square identity
 * matrix with ones on the diagonal.
 */
export function eye(
  numRows: number,
  numCols?: number,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  numCols = numCols ?? numRows;
  const weakType = dtype == undefined;
  dtype = dtype ?? DType.Float32;

  if (numCols < numRows) {
    // If less columns than rows, take the transpose since it's no longer a
    // simple modular arithmetic expression.
    const arr = eye(numCols, numRows, { dtype, device });
    return arr.transpose();
  }
  if (numRows === 0) {
    return zeros([0, numCols], { dtype, device });
  }

  const exp = AluExp.cmplt(
    AluExp.mod(AluVar.idx, AluExp.i32(numCols + 1)),
    AluExp.i32(1),
  );
  return new Array({
    source: AluExp.cast(dtype, exp),
    st: ShapeTracker.fromShape([numRows, numCols]),
    dtype,
    weakType,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

/** Return the identity matrix, with ones on the main diagonal. */
export function identity(
  n: number,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  return eye(n, n, { dtype, device });
}

/**
 * Return evenly spaced values within a given interval.
 *
 * This can be called with a varying number of arguments, just like the range()
 * builtin function in Python.
 *
 * - `arange(stop)` is equivalent to `arange(0, stop, 1)`.
 * - `arange(start, stop)` is equivalent to `arange(start, stop, 1)`.
 * - `arange(start, stop, step)` creates an array starting at `start`, ending
 *   before `stop`, with a step size of `step`.
 *
 * Defaults to an integer data type. This can produce unintended results when
 * using a non-integer step, so prefer linspace() in those cases.
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  { dtype, device }: DTypeAndDevice = {},
) {
  dtype = dtype ?? DType.Int32; // default to int32 for arange

  if (stop === undefined) {
    stop = start;
    start = 0;
  }
  if (step === 0) {
    throw new RangeError(
      `Invalid step for arange: ${step}. Step must be non-zero.`,
    );
  }
  const size = Math.max(0, Math.ceil((stop - start) / step));
  if (size === 0) {
    return zeros([0], { dtype, device });
  }

  const exp = AluExp.add(
    AluExp.const(dtype, start),
    AluExp.mul(AluExp.cast(dtype, AluVar.idx), AluExp.const(dtype, step)),
  );
  const st = ShapeTracker.fromShape([size]);
  return new Array({
    source: exp,
    st,
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

/**
 * Return an array with ones on and below the diagonal and zeros elsewhere.
 *
 * If `k` is provided, it specifies the sub-diagonal on and below which the
 * array is filled with ones. `k=0` is the main diagonal, `k<0` is below it, and
 * `k>0` is above it.
 */
export function tri(
  n: number,
  m?: number,
  k: number = 0,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  m ??= n;
  dtype ??= DType.Float32;
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`tri: n must be a non-negative integer, got ${n}`);
  }
  if (!Number.isInteger(m) || m < 0) {
    throw new Error(`tri: m must be a non-negative integer, got ${m}`);
  }
  if (!Number.isInteger(k)) {
    throw new Error(`tri: k must be an integer, got ${k}`);
  }
  const rows = arange(k, n + k, 1, { dtype: DType.Int32, device });
  const cols = arange(0, m, 1, { dtype: DType.Int32, device });
  return rows.reshape([n, 1]).greaterEqual(cols).astype(dtype);
}

/** Return the lower triangle of an array. Must be of dimension >= 2. */
export function tril(a: ArrayLike, k: number = 0): Array {
  if (ndim(a) < 2) {
    throw new Error(`tril: input array must be at least 2D, got ${ndim(a)}D`);
  }
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where(
    tri(n, m, k, { dtype: DType.Bool }),
    a.ref,
    zerosLike(a),
  ) as Array;
}

/** Return the upper triangle of an array. Must be of dimension >= 2. */
export function triu(a: ArrayLike, k: number = 0): Array {
  if (ndim(a) < 2) {
    throw new Error(`tril: input array must be at least 2D, got ${ndim(a)}D`);
  }
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where(
    tri(n, m, k - 1, { dtype: DType.Bool }),
    zerosLike(a.ref),
    a,
  ) as Array;
}

/**
 * Return evenly spaced numbers over a specified interval.
 *
 * Returns _num_ evenly spaced samples, calculated over the interval
 * [`start`, `stop`]. The endpoint `stop` is included in the result by default,
 * but this is controlled by the `endpoint` parameter.
 *
 * The default data type is Float32. Use arange() for integer steps.
 */
export function linspace(
  start: number,
  stop: number,
  num: number = 50,
  endpoint: boolean = true,
  { dtype, device }: DTypeAndDevice = {},
) {
  dtype = dtype ?? DType.Float32; // default to float32 for linspace

  if (num < 0 || !Number.isInteger(num)) {
    throw new RangeError(
      `Invalid num for linspace: ${num}. Must be non-negative integer.`,
    );
  } else if (num === 0) {
    return zeros([0], { dtype, device });
  } else if (num === 1) {
    return full([1], start, { dtype, device });
  } else if (start === stop) {
    return full([num], start, { dtype, device });
  }

  // Now num >= 2, there are at least 2 points.
  const delta = stop - start;
  const denom = endpoint ? num - 1 : num;
  const exp = AluExp.cast(
    dtype,
    AluExp.add(
      AluExp.f32(start),
      AluExp.mul(
        AluExp.f32(delta / denom),
        AluExp.cast(DType.Float32, AluVar.idx),
      ),
    ),
  );
  const st = ShapeTracker.fromShape([num]);
  return new Array({
    source: exp,
    st,
    dtype,
    weakType: false,
    backend: getBackend(device),
    committed: device != undefined,
  });
}

/**
 * Return numbers spaced evenly on a log scale.
 *
 * In linear space, the sequence starts at `base ** start` and ends at
 * `base ** stop` (see `endpoint` below).
 *
 * @param start - `base ** start` is the starting value of the sequence.
 * @param stop - `base ** stop` is the final value of the sequence, unless `endpoint` is false.
 * @param num - Number of samples to generate. Default is 50.
 * @param endpoint - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
 * @param base - The base of the log space. Default is 10.
 * @returns Array of evenly spaced values on a log scale.
 */
export function logspace(
  start: number,
  stop: number,
  num: number = 50,
  endpoint: boolean = true,
  base: number = 10,
  { dtype, device }: DTypeAndDevice = {},
) {
  const y = linspace(start, stop, num, endpoint, { dtype, device });
  // base ** y = exp(log(base) * y)
  const logBase = Math.log(base);
  return coreExp(coreMul(y, logBase)) as Array;
}

export function aluCompare(a: AluExp, b: AluExp, op: CompareOp): AluExp {
  // Translate a `CompareOp` into an `AluExp` on two sub-expressions.
  switch (op) {
    case CompareOp.Less:
      return AluExp.cmplt(a, b);
    case CompareOp.Equal:
      // `x == y` is always equivalent to `!(x != y)`, including NaN
      return AluExp.cmpne(a, b).not();
    case CompareOp.NotEqual:
      return AluExp.cmpne(a, b);
    case CompareOp.LessEqual:
      // `x <= y` is equivalent to `x < y || x == y`
      return AluExp.add(AluExp.cmplt(a, b), AluExp.cmpne(a, b).not());
  }
}

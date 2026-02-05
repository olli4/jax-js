// Utilities for working with tree-like container data structures ("pytrees").

import type { Array } from "./frontend/array";
import { Tracer } from "./frontend/core";
import { deepEqual, unzip2 } from "./utils";

const JsArray = globalThis.Array;

export enum NodeType {
  Array = "Array",
  Object = "Object",
  Leaf = "Leaf",
  None = "None",
}

/** Analog to the JAX "pytree" object, but for JavaScript. */
export type JsTree<T> = T | JsTree<T>[] | { [key: string]: JsTree<T> };

type Same<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2
    ? true
    : false;

type MappedJsTree<T, A, B> = T extends A
  ? B
  : T extends Array // Special case: Do not recurse into np.Array
    ? T
    : T extends globalThis.Array<infer U>
      ? number extends T["length"]
        ? MapJsTree<U, A, B>[] // plain array
        : { [K in keyof T]: MapJsTree<T[K], A, B> } // tuple: map each slot
      : { [K in keyof T]: MapJsTree<T[K], A, B> }; // object: map each slot

/** Convert a subtype of JsTree<A> into JsTree<B>, preserving structure. Used by jit/grad/vjp types. */
export type MapJsTree<T, A, B> =
  Same<A, B> extends true ? T : MappedJsTree<T, A, B>;

/** Represents the structure of a JsTree. */
export class JsTreeDef {
  static leaf = new JsTreeDef(NodeType.Leaf, null, []);
  static none = new JsTreeDef(NodeType.None, null, []);

  constructor(
    readonly nodeType: NodeType,
    readonly nodeMetadata: any, // Must be comparable with deepEqual.
    readonly childTreedefs: JsTreeDef[],
  ) {}

  /** Get the total number of leaves in the tree. */
  get size(): number {
    if (this.nodeType === NodeType.Leaf) return 1;
    if (this.nodeType === NodeType.None) return 0;
    return this.childTreedefs.reduce((a, b) => a + b.size, 0);
  }

  /** Returns a string representation of this tree definition. */
  toString(root = true): string {
    if (root) {
      return "JsTreeDef(" + this.toString(false) + ")";
    }
    switch (this.nodeType) {
      case NodeType.None:
        return "null";
      case NodeType.Leaf:
        return "*";
      case NodeType.Array:
        return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
      case NodeType.Object: {
        const parts: string[] = [];
        for (let i = 0; i < this.childTreedefs.length; i++) {
          parts.push(
            `${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`,
          );
        }
        return `{${parts.join(", ")}}`;
      }
    }
  }

  /** Compare this tree definition with another. */
  equals(other: JsTreeDef): boolean {
    return (
      this.nodeType === other.nodeType &&
      deepEqual(this.nodeMetadata, other.nodeMetadata) &&
      this.childTreedefs.length === other.childTreedefs.length &&
      this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]))
    );
  }
}

function quoteObjectKey(key: string): string {
  // Check if the key is a valid JavaScript identifier
  if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key)) {
    return key; // No need to quote
  }
  return JSON.stringify(key);
}

/** Flatten a structured object, returning the tree definition. */
export function flatten<T>(tree: JsTree<T>): [T[], JsTreeDef] {
  const leaves: T[] = [];
  const treedef = _flatten(tree, leaves);
  return [leaves, treedef];
}

function _flatten<T>(tree: JsTree<T>, leaves: T[]): JsTreeDef {
  // Handle null/undefined as empty node (like JAX's None)
  if (tree === null || tree === undefined) {
    return JsTreeDef.none;
  }
  if (JsArray.isArray(tree)) {
    const childTrees = tree.map((c) => _flatten(c, leaves));
    return new JsTreeDef(NodeType.Array, null, childTrees);
  } else if (
    typeof tree === "object" &&
    tree !== null &&
    tree.constructor === Object // Needed to avoid treating Array as an object.
  ) {
    const [keys, values] = unzip2(Object.entries(tree));
    const childTrees = values.map((c) => _flatten(c, leaves));
    return new JsTreeDef(NodeType.Object, keys, childTrees);
  } else {
    leaves.push(tree as T);
    return JsTreeDef.leaf;
  }
}

/** Get the leaves of a tree. */
export function leaves<T>(tree: JsTree<T>): T[] {
  return flatten<T>(tree)[0];
}

/** Get the treedef for a tree. */
export function structure<T>(tree: JsTree<T>): JsTreeDef {
  return flatten<T>(tree)[1];
}

/** Reconstruct a structured object from the flattened representation. */
export function unflatten<T>(
  treedef: JsTreeDef,
  leaves: Iterable<T>,
): JsTree<T> {
  return _unflatten(treedef, leaves[Symbol.iterator]());
}

function _unflatten<T>(treedef: JsTreeDef, leaves: Iterator<T>): JsTree<T> {
  switch (treedef.nodeType) {
    case NodeType.None:
      // None node type represents null/undefined - return null
      return null as unknown as JsTree<T>;
    case NodeType.Leaf: {
      const { value, done } = leaves.next();
      if (done) {
        throw new TypeError("Ran out of leaves while unflattening JsTree");
      }
      return value;
    }
    case NodeType.Array:
      return treedef.childTreedefs.map((c) => _unflatten(c, leaves));
    case NodeType.Object: {
      const obj: any = {};
      for (let i = 0; i < treedef.childTreedefs.length; i++) {
        obj[treedef.nodeMetadata[i]] = _unflatten(
          treedef.childTreedefs[i],
          leaves,
        );
      }
      return obj;
    }
  }
}

/** Options for {@link map}. */
export interface MapOptions<T> {
  /** Returns true if value should be treated as a leaf (not recursed into). */
  isLeaf?: (x: T) => boolean;
}

/**
 * Maps a function over pytree leaves. Equivalent to `jax.tree.map`.
 *
 * @param fn - Function to apply to corresponding leaves.
 * @param tree - First pytree (determines output structure).
 * @param rest - Additional trees (must match structure), optionally ending with `{ isLeaf }`.
 * @throws {TypeError} If trees have different structures.
 *
 * @example
 * ```ts
 * tree.map((x, y) => x + y, { a: 1 }, { a: 10 });  // { a: 11 }
 * tree.map((...v) => sum(v), ...trees);  // JAX: tree.map(fn, *trees)
 * tree.map(fn, tree, { isLeaf: (x) => Array.isArray(x) });  // custom leaves
 * ```
 */
// Overload: single tree with options
export function map<T, U, Tree extends JsTree<T>>(
  fn: (arg: T) => U,
  tree: Tree,
  options: MapOptions<T>,
): MapJsTree<Tree, T, U>;
// Overload: two trees with options
export function map<T, U, Tree extends JsTree<T>>(
  fn: (a: T, b: T) => U,
  tree: Tree,
  tree2: Tree,
  options: MapOptions<T>,
): MapJsTree<Tree, T, U>;
// Overload: multiple trees (no options) - main signature
export function map<T, U, Tree extends JsTree<T>>(
  fn: (...args: T[]) => U,
  tree: Tree,
  ...rest: Tree[]
): MapJsTree<Tree, T, U>;
// Implementation
export function map<T, U, Tree extends JsTree<T>>(
  fn: (...args: T[]) => U,
  tree: Tree,
  ...rest: unknown[]
): MapJsTree<Tree, T, U> {
  // Extract options if last argument has isLeaf
  let options: MapOptions<T> | undefined;
  let restTrees: Tree[];
  const last = rest[rest.length - 1];
  if (
    rest.length > 0 &&
    typeof last === "object" &&
    last !== null &&
    !JsArray.isArray(last) &&
    "isLeaf" in last
  ) {
    options = last as MapOptions<T>;
    restTrees = rest.slice(0, -1) as Tree[];
  } else {
    restTrees = rest as Tree[];
  }

  const isLeaf = options?.isLeaf;
  const [leaves, treedef] = isLeaf
    ? flattenWithIsLeaf(tree, isLeaf)
    : flatten<T>(tree);

  // Flatten rest trees and validate structure
  const restFlattened = restTrees.map((t, i) => {
    const [l, td] = isLeaf ? flattenWithIsLeaf(t, isLeaf) : flatten<T>(t);
    if (!td.equals(treedef)) {
      throw new TypeError(
        `tree.map: tree structure mismatch at argument ${i + 2}. ` +
          `Expected ${treedef.toString()}, got ${td.toString()}`,
      );
    }
    return l;
  });

  const resultLeaves: U[] = [];
  for (let i = 0; i < leaves.length; i++) {
    resultLeaves.push(fn(leaves[i], ...restFlattened.map((x) => x[i])));
  }
  return unflatten(treedef, resultLeaves) as MapJsTree<Tree, T, U>;
}

/** Flatten with custom isLeaf predicate. */
function flattenWithIsLeaf<T>(
  tree: JsTree<T>,
  isLeaf: (x: T) => boolean,
): [T[], JsTreeDef] {
  const leaves: T[] = [];
  const treedef = _flattenWithIsLeaf(tree, leaves, isLeaf);
  return [leaves, treedef];
}

function _flattenWithIsLeaf<T>(
  tree: JsTree<T>,
  leaves: T[],
  isLeaf: (x: T) => boolean,
): JsTreeDef {
  // Handle null/undefined as empty node (like JAX's None)
  if (tree === null || tree === undefined) {
    return JsTreeDef.none;
  }
  // Check custom isLeaf predicate first
  if (isLeaf(tree as T)) {
    leaves.push(tree as T);
    return JsTreeDef.leaf;
  }
  if (JsArray.isArray(tree)) {
    const childTrees = tree.map((c) => _flattenWithIsLeaf(c, leaves, isLeaf));
    return new JsTreeDef(NodeType.Array, null, childTrees);
  } else if (
    typeof tree === "object" &&
    tree !== null &&
    tree.constructor === Object
  ) {
    const [keys, values] = unzip2(Object.entries(tree));
    const childTrees = values.map((c) => _flattenWithIsLeaf(c, leaves, isLeaf));
    return new JsTreeDef(NodeType.Object, keys, childTrees);
  } else {
    leaves.push(tree as T);
    return JsTreeDef.leaf;
  }
}

/** Take a reference of every array in a tree. */
export function ref<Tree extends JsTree<any>>(tree: Tree): Tree {
  return map((x) => (x instanceof Tracer ? x.ref : x), tree) as unknown as Tree;
}

/** Dispose every array in a tree. */
export function dispose<Tree extends JsTree<any>>(
  tree: Tree | null | undefined,
): void {
  if (tree) {
    map((x) => (x instanceof Tracer ? x.dispose() : undefined), tree);
  }
}

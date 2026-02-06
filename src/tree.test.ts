import { expect, suite, test } from "vitest";

import { flatten, JsTreeDef, map, NodeType, unflatten } from "./tree";

suite("JsTreeDef.toString()", () => {
  test("should return '*' for a leaf", () => {
    expect(JsTreeDef.leaf.toString()).toBe("JsTreeDef(*)");
  });

  test("should return a string representation for an array node", () => {
    // Create a treedef representing an array of two leaves.
    const arrTreedef = new JsTreeDef(NodeType.Array, null, [
      JsTreeDef.leaf,
      JsTreeDef.leaf,
    ]);
    expect(arrTreedef.toString()).toBe("JsTreeDef([*, *])");
  });

  test("should return a string representation for an object node", () => {
    // Create an object treedef with keys "a" and "b" mapping to leaves.
    const objTreedef = new JsTreeDef(
      NodeType.Object,
      ["a", "b"],
      [JsTreeDef.leaf, JsTreeDef.leaf],
    );
    expect(objTreedef.toString()).toBe("JsTreeDef({a: *, b: *})");
  });

  test("should properly quote keys that are not valid identifiers", () => {
    // Keys with spaces or special characters should be quoted.
    const objTreedef = new JsTreeDef(
      NodeType.Object,
      ["not-valid", "validKey"],
      [JsTreeDef.leaf, JsTreeDef.leaf],
    );
    // "not-valid" is not a valid identifier, so it should be quoted (using JSON.stringify).
    const expected = `JsTreeDef({${JSON.stringify("not-valid")}: *, validKey: *})`;
    expect(objTreedef.toString()).toBe(expected);
  });
});

suite("flatten()", () => {
  test("should flatten a leaf value", () => {
    const [leaves, treedef] = flatten(42);
    expect(leaves).toEqual([42]);
    // A leaf treedef always returns '*' for non-root calls.
    expect(treedef.toString()).toBe("JsTreeDef(*)");
  });

  test("should flatten an array", () => {
    const input = [1, 2, 3];
    const [leaves, treedef] = flatten(input);
    expect(leaves).toEqual([1, 2, 3]);
    // treedef should be an array type with three children (all leaves)
    expect(treedef.nodeType).toBe(NodeType.Array);
    expect(treedef.childTreedefs.length).toBe(3);
    for (const child of treedef.childTreedefs) {
      expect(child.nodeType).toBe(NodeType.Leaf);
    }
    expect(treedef.toString()).toBe("JsTreeDef([*, *, *])");
  });

  test("should flatten an object", () => {
    const input = { a: 1, b: [2, 3] };
    const [leaves, treedef] = flatten(input);
    // The order of keys is determined by Object.entries order.
    // Assuming the natural order: "a", then "b".
    expect(leaves).toEqual([1, 2, 3]);

    expect(treedef.nodeType).toBe(NodeType.Object);
    expect(treedef.nodeMetadata).toEqual(["a", "b"]);
    expect(treedef.childTreedefs.length).toBe(2);

    // Check the first child: a leaf.
    expect(treedef.childTreedefs[0].nodeType).toBe(NodeType.Leaf);
    // Check the second child: an array with two leaves.
    const secondChild = treedef.childTreedefs[1];
    expect(secondChild.nodeType).toBe(NodeType.Array);
    expect(secondChild.childTreedefs.length).toBe(2);
    expect(secondChild.childTreedefs[0].nodeType).toBe(NodeType.Leaf);
    expect(secondChild.childTreedefs[1].nodeType).toBe(NodeType.Leaf);

    // We can also check the string representation.
    expect(treedef.toString()).toBe("JsTreeDef({a: *, b: [*, *]})");
  });
});

suite("unflatten()", () => {
  test("should reconstruct a leaf value", () => {
    const value = "test";
    const [leaves, treedef] = flatten(value);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toBe(value);
  });

  test("should reconstruct an array", () => {
    const input = [10, 20, 30];
    const [leaves, treedef] = flatten(input);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toEqual(input);
  });

  test("should reconstruct an object", () => {
    const input = { x: 100, y: [200, 300] };
    const [leaves, treedef] = flatten(input);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toEqual(input);
  });

  test("should reconstruct a nested tree", () => {
    const input = { a: [1, { b: 2 }], c: "hello" };
    const [leaves, treedef] = flatten<number | string>(input);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toEqual(input);
  });

  test("should throw an error when there are too few leaves", () => {
    const input = [1, 2, 3];
    const [leaves, treedef] = flatten(input);
    // Remove one leaf to simulate missing leaves.
    const insufficientLeaves = leaves.slice(0, leaves.length - 1);
    expect(() => {
      unflatten(treedef, insufficientLeaves);
    }).toThrow("Ran out of leaves while unflattening JsTree");
  });
});

suite("JsTreeDef.equals()", () => {
  test("marks identical trees as equal", () => {
    const tree1 = { a: [1, 2], b: { c: 3 } };
    const tree2 = { a: [-40, 222], b: { c: 31 } };

    const [, treedef1] = flatten(tree1);
    const [, treedef2] = flatten(tree2);

    expect(treedef1.equals(treedef2)).toBe(true);
  });

  test("marks different shapes or keys as not equal", () => {
    const tree1 = { a: [1, 2], b: { c: 3 } };
    const tree2 = { a: [1, 2, 3], b: { c: 3 } };
    const tree3 = { a: [1, 2], d: { c: 3 } };

    const [, treedef1] = flatten(tree1);
    const [, treedef2] = flatten(tree2);
    const [, treedef3] = flatten(tree3);

    expect(treedef1.equals(treedef2)).toBe(false);
    expect(treedef1.equals(treedef3)).toBe(false);
  });
});

suite("map()", () => {
  test("should map a single-argument function", () => {
    const myTree = { a: [1, 2], b: { c: 3 } };
    const result = map((x: number) => x * 2, myTree);
    const expected = { a: [2, 4], b: { c: 6 } };
    expect(result).toEqual(expected);
  });

  test("should map a multi-argument function", () => {
    const tree1 = { a: [1, 2], b: { c: 3 } };
    const tree2 = { a: [4, 5], b: { c: 6 } };
    const result = map((x: number, y: number) => x + y, tree1, tree2);
    const expected = { a: [5, 7], b: { c: 9 } };
    expect(result).toEqual(expected);
  });

  test("should support spread array of trees (JAX-style)", () => {
    // Equivalent to JAX's: jax.tree.map(lambda *v: sum(v), *trees)
    type Tree = { a: number; b: number };
    const trees: [Tree, Tree, Tree] = [
      { a: 1, b: 2 },
      { a: 10, b: 20 },
      { a: 100, b: 200 },
    ];
    const result = map(
      (...xs: number[]) => xs.reduce((a, b) => a + b, 0),
      ...trees,
    );
    expect(result).toEqual({ a: 111, b: 222 });
  });

  test("should throw on structure mismatch", () => {
    const tree1 = { a: 1, b: 2 };
    const tree2 = { a: 1, c: 2 }; // Different key
    // Use 'as any' to bypass TypeScript's compile-time check (we want runtime error)
    expect(() =>
      map((x: number, y: number) => x + y, tree1, tree2 as any),
    ).toThrow(/tree structure mismatch/);
  });

  test("should throw on structure mismatch (different nesting)", () => {
    const tree1 = { a: [1, 2] };
    const tree2 = { a: 1 }; // Not nested
    // Use 'as any' to bypass TypeScript's compile-time check (we want runtime error)
    expect(() =>
      map((x: number, y: number) => x + y, tree1, tree2 as any),
    ).toThrow(/tree structure mismatch/);
  });

  test("should support isLeaf option", () => {
    // Treat 2-element arrays as leaves (tuples)
    const myTree = { point: [1, 2], values: [3, 4, 5] };
    const result = map(
      (x: number | number[]) =>
        Array.isArray(x) ? x.reduce((a, b) => a + b, 0) : x * 10,
      myTree,
      { isLeaf: (x) => Array.isArray(x) && x.length === 2 },
    );
    // point is treated as leaf (sum=3), values is recursed (each * 10)
    expect(result).toEqual({ point: 3, values: [30, 40, 50] });
  });

  test("isLeaf should apply to all trees", () => {
    const tree1 = { data: [1, 2] };
    const tree2 = { data: [10, 20] };
    const result = map(
      (a: number[], b: number[]) => [...a, ...b],
      tree1,
      tree2,
      { isLeaf: (x: number | number[]) => Array.isArray(x) },
    );
    expect(result).toEqual({ data: [1, 2, 10, 20] });
  });
});

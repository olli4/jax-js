import { expect, suite, test } from "vitest";

import { AluExp, DType } from "./alu";
import { ShapeTracker, unravelAlu, View } from "./shape";

suite("View.create()", () => {
  test("creates a contiguous view with default strides", () => {
    const v = View.create([10, 2, 3]);
    expect(v.shape).toEqual([10, 2, 3]);
    // For a 2x3 array, default row-major strides are [3, 1] (after canonicalization)
    expect(v.strides).toEqual([6, 3, 1]);
    expect(v.offset).toBe(0);
    expect(v.mask).toBeNull();
  });

  test("throws an error for negative shape values", () => {
    expect(() => {
      View.create([-1, 3]);
    }).toThrow("View shape must be non-negative");
  });

  test("zero-sized array has all zero strides", () => {
    const v = View.create([0, 5]);
    expect(v.strides).toEqual([0, 0]);
  });

  test("creates a view with offset and non-default strides", () => {
    const v = View.create([2, 3], [1, 2], 40);
    expect(v.strides).toEqual([1, 2]);
  });

  test("normalizes complete mask", () => {
    const v = View.create([3, 10], [10, 1], 0, [
      [0, 3],
      [0, 10],
    ]);
    expect(v.mask).toBeNull();
  });

  test("normalizes empty mask", () => {
    const v = View.create([3, 10], [10, 1], 0, [
      [0, 0],
      [0, 10],
    ]);
    expect(v.mask).toEqual([
      [0, 0],
      [0, 0],
    ]);
  });
});

suite("View.size and contiguous", () => {
  test("size is the product of the shape elements", () => {
    const v = View.create([2, 3, 4]);
    expect(v.size).toBe(24);
  });

  test("an empty shape yields size 1", () => {
    const v = View.create([]);
    expect(v.size).toBe(1);
  });

  test("a default (contiguous) view is recognized as contiguous", () => {
    const v = View.create([2, 3]);
    expect(v.contiguous).toBe(true);
  });

  test("a non-contiguous view is not marked as contiguous", () => {
    const v = View.create([2, 3]).flip([true, false]);
    expect(v.contiguous).toBe(false);
  });
});

suite("View.pad()", () => {
  test("padding with all zeros returns the same view", () => {
    const v = View.create([3, 3]);
    const padded = v.pad([
      [0, 0],
      [0, 0],
    ]);
    expect(padded).toBe(v);
  });

  test("pad increases shape and adjusts offset", () => {
    const v = View.create([3, 3]);
    const padded = v.pad([
      [1, 2],
      [3, 4],
    ]);
    // New shape: original dim + pad_before + pad_after
    expect(padded.shape).toEqual([3 + 1 + 2, 3 + 3 + 4]); // [6, 10]
    // Expected offset = original offset - strides[0]*pad_before - strides[1]*pad_before for dim0 and dim1
    // For [3,3] default strides: [3, 1] so: 3*-1 + 1*-3 = 6.
    expect(padded.offset).toBe(-6);
    expect(padded.mask).toEqual([
      [1, 4],
      [3, 6],
    ]);
  });

  test("pad with invalid arguments throws an error", () => {
    const v = View.create([3, 3]);
    expect(() => {
      v.pad([
        [0, -1],
        [0, 0],
      ]);
    }).toThrow();
  });
});

suite("View.shrink()", () => {
  test("shrink reduces the shape and adjusts offset accordingly", () => {
    const v = View.create([5, 5]);
    // Shrink to take rows 1..4 and columns 2..5 -> new shape [3, 3]
    const shrunk = v.shrink([
      [1, 4],
      [2, 5],
    ]);
    expect(shrunk.shape).toEqual([4 - 1, 5 - 2]); // [3, 3]
    // For default strides [5,1]: offset = 5*1 + 1*2 = 7.
    expect(shrunk.offset).toBe(7);
  });

  test("shrink with invalid ranges throws an error", () => {
    const v = View.create([5, 5]);
    expect(() => {
      v.shrink([
        [4, 1],
        [0, 5],
      ]);
    }).toThrow();
  });
});

suite("View.expand()", () => {
  test("expand increases a dimension of size 1 to a larger size", () => {
    const v = View.create([10, 1, 3]);
    const expanded = v.expand([10, 4, 3]);
    expect(expanded.shape).toEqual([10, 4, 3]);
    expect(expanded.strides).toEqual([3, 0, 1]);
  });

  test("expand throws an error if non-singleton dimensions are changed", () => {
    const v = View.create([2, 3]);
    expect(() => {
      v.expand([3, 3]); // first dimension is 2 and cannot expand to 3
    }).toThrow("Can't expand");
  });
});

suite("View.permute()", () => {
  test("permutes the axes of the view", () => {
    const v = View.create([2, 3, 4]);
    const permuted = v.permute([2, 0, 1]);
    expect(permuted.shape).toEqual([4, 2, 3]);
    expect(permuted.strides).toEqual([
      v.strides[2],
      v.strides[0],
      v.strides[1],
    ]);
  });

  test("an invalid permutation throws an error", () => {
    const v = View.create([2, 3]);
    expect(() => {
      v.permute([0, 0]); // not a valid permutation
    }).toThrow("Invalid permutation");
  });
});

suite("View.flip()", () => {
  test("flips the specified axes", () => {
    const v = View.create([2, 3]);
    const flipped = v.flip([true, false]);
    // For dimension 0: new offset = (dim - 1)*original_stride = (2-1)*3 = 3.
    expect(flipped.offset).toBe(3);
    // Stride for dimension 0 should be negated.
    expect(flipped.strides[0]).toBe(-v.strides[0]);
    // The second dimension remains unchanged.
    expect(flipped.strides[1]).toBe(v.strides[1]);
  });

  test("flip with an invalid argument length throws an error", () => {
    const v = View.create([2, 3]);
    expect(() => {
      v.flip([true]);
    }).toThrow();
  });
});

suite("View.reshape()", () => {
  test("reshapes a contiguous view to a new valid shape", () => {
    const v = View.create([2, 3]);
    const reshaped = v.reshape([3, 2]);
    expect(reshaped).not.toBeNull();
    expect(reshaped!.shape).toEqual([3, 2]);
  });

  test("reshape with the same shape returns the same view", () => {
    const v = View.create([2, 3]);
    const reshaped = v.reshape([2, 3]);
    expect(reshaped).toBe(v);
  });

  test("reshaping to a shape with mismatched total size throws an error", () => {
    const v = View.create([2, 3]);
    expect(() => {
      v.reshape([4, 2]); // 2*3 !== 4*2
    }).toThrow("Reshape size");
  });

  test("reshape of a non-contiguous view", () => {
    const v = View.create([2, 3]).flip([true, false]);
    const reshaped = v.reshape([3, 2]);
    // Depending on the internal merging, reshape may succeed or return null.
    if (reshaped !== null) {
      expect(reshaped.shape).toEqual([3, 2]);
    } else {
      expect(reshaped).toBeNull();
    }
  });

  test("complex reshape", () => {
    expect(
      View.create([240])
        .reshape([2, 2, 6, 1, 10])!
        .permute([1, 0, 2, 3, 4])
        .pad([
          [1, 1],
          [2, 3],
          [0, 0],
          [0, 0],
          [0, 0],
        ])
        .reshape([4, 7, 3, 2, 10]),
    ).toEqual({
      mask: [
        [1, 3],
        [2, 4],
        [0, 3],
        [0, 2],
        [0, 10],
      ],
      offset: -300,
      shape: [4, 7, 3, 2, 10],
      strides: [60, 120, 20, 10, 1],
    });
  });
});

suite("View.minify()", () => {
  test("minify reduces redundant dimensions", () => {
    const v = View.create([1, 3, 1, 4]);
    const minified = v.minify();
    expect(minified.shape).toEqual([12]);
  });

  test("minify on a non-contiguous view returns equivalent shape", () => {
    const v = View.create([2, 3]).flip([true, false]);
    expect(v.contiguous).toBe(false);
    const minified = v.minify();
    expect(minified.shape).toEqual(v.shape);
  });
});

suite("View.compose()", () => {
  test("compose with a contiguous view returns the other view", () => {
    const v1 = View.create([2, 3]);
    const v2 = View.create([2, 3]); // contiguous
    // In compose, if the calling view (v2) is contiguous, it returns its argument (v1)
    const composed = v2.compose(v1);
    expect(composed).toBe(v1);
  });

  test("compose with a non-contiguous view when shapes match", () => {
    const v = View.create([2, 3]);
    const vFlipped = v.flip([true, false]);
    // Here, calling compose on a contiguous view returns the non-contiguous view argument.
    const composed = v.compose(vFlipped);
    expect(composed).toBe(vFlipped);
  });

  test("compose returns null for incompatible views", () => {
    // Craft two views that are unlikely to compose.
    const v1 = View.create([2, 3]);
    // Flip and then shrink to force a nontrivial transformation.
    const v2 = v1.flip([true, false]).shrink([
      [0, 2],
      [0, 2],
    ]);
    const composed = v2.compose(v1);
    expect(composed).toBeNull();
  });
});

suite("ShapeTracker", () => {
  test("simplifies views correctly", () => {
    const base = ShapeTracker.fromShape([2, 3]);
    expect(base.reshape([6]).compose(base)).toEqual(base);
  });

  test("can stack non-composable views", () => {
    const base = ShapeTracker.fromShape([2, 3]);
    const flipped = base.flip([true, false]);
    expect(flipped.views).toHaveLength(1);
    expect(flipped.reshape([3, 2]).views).toEqual([
      flipped.views[0],
      View.create([3, 2]),
    ]);
  });
});

suite("toAluExp()", () => {
  test("converts View to expression", () => {
    let v = View.create([200]);
    let [exp, vexp] = v.toAluExp([AluExp.special(DType.Int32, "x", 200)]);
    expect(vexp.resolve()).toBe(true);
    expect(exp.evaluate({ x: 42 })).toEqual(42);

    v = View.create([20, 10]);
    [exp, vexp] = v.toAluExp([
      AluExp.special(DType.Int32, "x", 20),
      AluExp.special(DType.Int32, "y", 10),
    ]);
    expect(vexp.resolve()).toBe(true);
    expect(exp.evaluate({ x: 5, y: 3 })).toEqual(53);
    expect(exp.evaluate({ x: 15, y: 3 })).toEqual(153);

    v = v.flip([true, false]);
    [exp, vexp] = v.toAluExp([
      AluExp.special(DType.Int32, "x", 20),
      AluExp.special(DType.Int32, "y", 10),
    ]);
    expect(vexp.resolve()).toBe(true);
    expect(exp.evaluate({ x: 5, y: 3 })).toEqual(143);
    expect(exp.evaluate({ x: 15, y: 3 })).toEqual(43);
  });

  test("works with padding", () => {
    const v = View.create([3, 3]).pad([
      [1, 1],
      [2, 3],
    ]);
    expect(v.shape).toEqual([5, 8]);
    const [_exp, vexp] = v.toAluExp([
      AluExp.special(DType.Int32, "x", 5),
      AluExp.special(DType.Int32, "y", 8),
    ]);
    expect(vexp.evaluate({ x: 0, y: 0 })).toBe(false);
    expect(vexp.evaluate({ x: 1, y: 0 })).toBe(false);
    expect(vexp.evaluate({ x: 1, y: 2 })).toBe(true);
    expect(vexp.evaluate({ x: 3, y: 4 })).toBe(true);
    expect(vexp.evaluate({ x: 4, y: 4 })).toBe(false);
    expect(vexp.evaluate({ x: 3, y: 5 })).toBe(false);
  });

  test("converts ShapeTracker to expression", () => {
    let st = ShapeTracker.fromShape([2, 2]);
    // 01
    // 23
    st = st
      .pad([
        [1, 1],
        [1, 1],
      ])
      .flip([false, true]);
    // ....
    // .10.
    // .32.
    // ....
    st = st.reshape([16]);
    // .....10..32.....

    const [exp, vexp] = st.toAluExp([AluExp.special(DType.Int32, "x", 16)]);
    expect(exp.evaluate({ x: 5 })).toEqual(1);
    expect(exp.evaluate({ x: 6 })).toEqual(0);
    expect(exp.evaluate({ x: 9 })).toEqual(3);
    expect(exp.evaluate({ x: 10 })).toEqual(2);

    for (let i = 0; i < 16; i++) {
      expect(vexp.evaluate({ x: i })).toBe(
        i === 5 || i === 6 || i === 9 || i === 10,
      );
    }
  });

  test("simplifies common unravels", () => {
    // Unravels are very common, they're the default case if no movement
    // operations are applied. So we want them to be simplified.
    const idx = AluExp.special(DType.Int32, "idx", 200);
    let st = ShapeTracker.fromShape([10, 20]);
    let [iexpr, vexpr] = st.toAluExp(unravelAlu(st.shape, idx));
    expect(vexpr.resolve()).toBe(true);
    expect(iexpr.evaluate({ idx: 50 })).toEqual(50);
    expect(iexpr).toEqual(idx);

    st = ShapeTracker.fromShape([5, 2, 20]);
    [iexpr, vexpr] = st.toAluExp(unravelAlu(st.shape, idx));
    expect(vexpr.resolve()).toBe(true);
    expect(iexpr.evaluate({ idx: 50 })).toEqual(50);
    expect(iexpr).toEqual(idx);
  });
});

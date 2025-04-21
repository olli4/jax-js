import { expect, test } from "vitest";

import { AluExp, DType } from "./alu";

test("AluExp can be evaluated", () => {
  const e = AluExp.i32(3);
  expect(e.evaluate({})).toEqual(3);

  const e2 = AluExp.add(AluExp.i32(3), AluExp.i32(4));
  expect(e2.evaluate({})).toEqual(7);

  const e3 = AluExp.add(AluExp.i32(3), e2);
  expect(e3.evaluate({})).toEqual(10);

  const e4 = AluExp.mul(AluExp.special(DType.Int32, "idx", 10), AluExp.i32(50));
  expect(e4.evaluate({ idx: 10 })).toEqual(500);
});

test("AluExp works with ternaries", () => {
  const x = AluExp.special(DType.Int32, "x", 100);

  const e = AluExp.where(
    AluExp.cmplt(x, AluExp.i32(70)),
    AluExp.i32(0),
    AluExp.i32(1),
  );
  expect(e.dtype).toBe(DType.Int32);
  expect(e.src).toHaveLength(3);
  expect(e.src[0].dtype).toBe(DType.Bool);
  expect(e.evaluate({ x: 50 })).toEqual(0);
  expect(e.evaluate({ x: 69 })).toEqual(0);
  expect(e.evaluate({ x: 70 })).toEqual(1);
  expect(e.evaluate({ x: 80 })).toEqual(1);
});

test("AluExp handles boolean ops", () => {
  const t = AluExp.bool(true);
  const f = AluExp.bool(false);

  expect(AluExp.mul(t, t).evaluate({})).toBe(true);
  expect(AluExp.mul(t, f).evaluate({})).toBe(false);

  expect(AluExp.add(t, f).evaluate({})).toBe(true);
  expect(AluExp.add(f, f).evaluate({})).toBe(false);
});

test("AluExp has .min and .max", () => {
  const e = AluExp.add(AluExp.i32(3), AluExp.i32(4));
  expect(e.min).toEqual(7);
  expect(e.max).toEqual(7);

  const e2 = AluExp.add(
    AluExp.special(DType.Int32, "x", 10),
    AluExp.special(DType.Int32, "y", 20),
  );
  expect(e2.min).toEqual(0);
  expect(e2.max).toEqual(28);
});

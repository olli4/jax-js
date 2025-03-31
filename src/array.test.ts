import { describe, expect, test as globalTest } from "vitest";
import { backendTypes, init } from "./backend";
import { Array, arrayFromData } from "./array";

const backendsAvailable = await init(...backendTypes);

describe.each(backendTypes)("Backend '%s'", (backend) => {
  const skipped = !backendsAvailable.includes(backend);
  const test = globalTest.skipIf(skipped);

  test("can construct Array.zeros()", async () => {
    const ar = Array.zeros([3, 3], { backend });
    expect(ar.shape).toEqual([3, 3]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(await ar.T.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(ar.T.dataSync()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
  });

  test("can construct Array.ones()", async () => {
    const ar = Array.ones([2, 2], { backend });
    expect(ar.shape).toEqual([2, 2]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  test("can add two arrays", async () => {
    const ar1 = Array.ones([2, 2], { backend });
    const ar2 = Array.ones([2, 2], { backend });
    const ar3 = ar1.add(ar2);
    expect(ar3.shape).toEqual([2, 2]);
    expect(ar3.dtype).toEqual("float32");
    expect(await ar3.data()).toEqual(new Float32Array([2, 2, 2, 2]));
  });

  test("can construct arrays from data", () => {
    const a = arrayFromData(new Float32Array([1, 2, 3, 4]), { backend });
    const b = arrayFromData(new Float32Array([10, 5, 2, -8.5]), { backend });
    const c = a.mul(b);
    expect(c.shape).toEqual([4]);
    expect(c.dtype).toEqual("float32");
    expect(c.dataSync()).toEqual(new Float32Array([10, 10, 6, -34]));
    expect(c.reshape([2, 2]).T.dataSync()).toEqual(
      new Float32Array([10, 6, 10, -34]),
    );
  });
});

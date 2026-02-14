import { RuleTester } from "eslint";

import requireRetainedRelease from "./rules/require-retained-release.mjs";
import requireTryFinallySymmetry from "./rules/require-try-finally-symmetry.mjs";
import requireWrapperDisposeSymmetry from "./rules/require-wrapper-dispose-symmetry.mjs";

const tester = new RuleTester({
  languageOptions: {
    ecmaVersion: 2022,
    sourceType: "module",
  },
});

tester.run(
  "@ownership/require-retained-release",
  requireRetainedRelease,
  {
    valid: [
      {
        code: `
          function f(x) {
            const y = x.ref;
            y.dispose();
          }
        `,
      },
      {
        code: `
          function f(x) {
            const y = x.ref;
            return y;
          }
        `,
      },
      {
        code: `
          function f(x) {
            const y = x.ref;
            consume(y);
          }
        `,
      },
    ],
    invalid: [
      {
        code: `
          function f(x) {
            const y = x.ref;
            y.shape;
          }
        `,
        errors: [{ messageId: "missingRelease", suggestions: 1 }],
      },
    ],
  },
);

tester.run(
  "@ownership/require-try-finally-symmetry",
  requireTryFinallySymmetry,
  {
    valid: [
      {
        code: `
          function f(x) {
            let t;
            try {
              t = x.ref;
              work(t);
            } finally {
              t.dispose();
            }
          }
        `,
      },
    ],
    invalid: [
      {
        code: `
          function f(x) {
            let t;
            try {
              t = x.ref;
              work(t);
            } catch (e) {
              handle(e);
            }
          }
        `,
        errors: [{ messageId: "missingFinally" }],
      },
      {
        code: `
          function f(x) {
            let t;
            try {
              t = x.ref;
              work(t);
            } finally {
              cleanup();
            }
          }
        `,
        errors: [{ messageId: "missingFinallyDispose" }],
      },
    ],
  },
);

tester.run(
  "@ownership/require-wrapper-dispose-symmetry",
  requireWrapperDisposeSymmetry,
  {
    valid: [
      {
        code: `
          class Wrapper {
            dispose() {
              this.cache.dispose();
              this.inner.dispose();
            }
          }
        `,
      },
      {
        code: `
          class Wrapper {
            dispose() {
              this.inner.dispose();
            }
          }
        `,
      },
    ],
    invalid: [
      {
        code: `
          class Wrapper {
            dispose() {
              this.inner.dispose();
              this.cache.dispose();
            }
          }
        `,
        errors: [{ messageId: "innerNotLast" }],
      },
    ],
  },
);

console.log("ownership lint rule tests passed");

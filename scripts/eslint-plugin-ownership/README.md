# Local ownership ESLint plugin

This in-repo plugin encodes jax-js ownership principles for static checking.

## Current rules

- `@ownership/require-retained-release`
  - Flags `const y = x.ref` when `y` has no explicit terminal action in scope.
  - Terminal actions recognized today:
    - `y.dispose()`
    - transfer (passed as call arg)
    - return/yield/assignment/container handoff

- `@ownership/require-try-finally-symmetry`
  - Flags `.ref` temporaries created inside `try` when cleanup is not in the corresponding
    `finally`.
  - Targets error-path parity in transform internals.

- `@ownership/require-wrapper-dispose-symmetry`
  - In wrapper `dispose()` methods, enforces `this.inner.dispose()` ordering.
  - If `this.inner.dispose()` appears, it must be the last `.dispose()` call.
  - Encodes "release retained state first, then dispose inner".

## Why local

The project’s ownership model evolves quickly in transforms and backend plumbing. Keeping rules
in-repo allows synchronized updates with internal principles and APIs.

## Next rule candidates

- Add trust-list + ownership contracts for core library functions.

## Issue-derived regression checklist

Patterns reported in the published plugin that are worth preserving in tests:

- Member-path fixed-point behavior (`obj.a.ref`, `obj.b.ref`) for sibling diagnostics.
- Mutually exclusive branch handling (`if / else if / else`) to avoid false positives.
- Setup/UX docs correctness (TS parser, `files` globs, `eslint.config.ts`, editor on-type feedback).

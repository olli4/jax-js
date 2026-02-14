# Merge Checklist: feat/non-consuming-ops → main

This file tracks items that must be resolved before merging this branch to main.

## Pre-merge tasks

- [ ] **Fix all KNOWN_BUG tests** — search for `KNOWN_BUG(` across `test/` to find them. Every
      `KNOWN_BUG` test should either pass (fix the underlying bug) or be intentionally removed with
      justification (document in copilot-instructions.md as a known limitation).

  ```bash
  grep -rn 'KNOWN_BUG(' test/
  ```

- [ ] **Restore strict pre-commit hook** — search `.husky/pre-commit` for `TODO(merge-to-main)` and
      remove the `|| true` suffixes so test failures block commits again.

  ```bash
  grep -n 'TODO(merge-to-main)' .husky/pre-commit
  ```

- [ ] **0 test failures** — `pnpm vitest run` must exit 0 with no failures.

- [ ] **Remove this file** — `MERGE_CHECKLIST.md` is branch-specific; delete it after merge.

## Current KNOWN_BUG inventory

| Tag                 | File                           | Description                               |
| ------------------- | ------------------------------ | ----------------------------------------- |
| `depth4-grad-leak`  | transform-compositions.test.ts | `grad⁴(f)` leaks intermediates            |
| `depth4-vjp-uaf`    | transform-compositions.test.ts | `vjp(grad³(f))` UAF at depth 4            |
| `makejaxpr-jvp`     | tracing.test.ts                | `makeJaxpr` does not compose with `jvp`   |
| `sort-grad`         | numpy.test.ts                  | `sort` grad needs scatter (not impl)      |

### Resolved KNOWN_BUGs

| Tag        | Resolution                                                                 |
| ---------- | -------------------------------------------------------------------------- |
| `sign-nan` | Fixed: NaN propagation via `notEqual(x, x)` + `where` in `numpy.ts sign()` |
| `bare-vmap-leak` | Fixed: wrapper-aware primal borrow balancing in transpose + explicit input ownership in test |
| `bare-jacfwd-leak` | Fixed: BatchTrace intermediate disposal + jacfwd primal-tree disposal in eager vmap contexts |
| `bare-jacrev-leak` | Fixed: wrapper-aware primal borrow balancing; test now owns input explicitly |
| `bare-hessian-leak` | Fixed via jacfwd/vmap ownership cleanup + input ownership in test |

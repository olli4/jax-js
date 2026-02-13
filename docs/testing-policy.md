# Testing policy: strict vs architectural mode

This repository uses two explicit test policies:

1. **Strict mode (default)** — zero failing tests, zero expected-failure debt.
2. **Architectural mode (opt-in)** — temporary failures are allowed only when
   explicitly declared in `.ci/expected-failures.json` with an owner, reason,
   and expiry date.

The goal is to unblock large refactors without normalizing ad-hoc debt.

## Why this exists

- Large restructures often break many tests transiently.
- A pure pass/fail count is misleading in this repo because environment-driven
  skips are common and "known failing" tests can unexpectedly flip to passing.
- We need debt to be explicit, reviewable, and time-bounded.

## Commands

- Strict test policy:

  ```bash
  pnpm run test:policy:strict
  ```

- Architectural mode test policy:

  ```bash
  pnpm run test:arch
  ```

## Pre-commit behavior

Default pre-commit uses strict policy. To opt in to architectural mode for a
refactor branch:

JAX_ARCH_MODE=1 git commit -m "your message"
```

Architectural mode still runs strict core invariant suites:

- `test/transform-compositions.test.ts`

## Expected-failure manifest

Manifest file: `.ci/expected-failures.json`
  
 Record expected failures from current Vitest run (easy path):

 ```bash
 pnpm run test:arch:record
 ```

 Record with custom metadata:

 ```bash
 pnpm run test:report
 node scripts/record-expected-failures.mjs \
   --report tmp/vitest-report.json \
   --manifest .ci/expected-failures.json \
   --owner @your-team \
   --reason "scan rewrite phase 1" \
   --expires 2026-03-01
 ```

Format:

```json
{
  "entries": [
    {
      "file": "test/some-suite.test.ts",
      "fullName": "suite name test name",
      "reason": "Refactor branch: temporary break while replacing ...",
      "owner": "@team-or-person",
      "expires": "2026-03-15"
    }
  ]
}
```

Rules:

- Every failing test in architectural mode must be listed.
- No expired entries are allowed.
- If a listed test starts passing, the check fails until the entry is removed.

## Guidance for reviewers
Notes on the recorder:

- It only keeps currently failing tests in the manifest (stale passing entries
  are automatically dropped on re-record).
- Existing entries keep their original owner/reason/expires metadata.
- New entries get defaults unless custom flags are provided.

- Request narrow, well-justified expected-failure entries.
- Require short expiry windows.
- Reject manifest growth without architectural rationale.
- Prefer deleting entries as soon as behavior stabilizes.

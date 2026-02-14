/**
 * Array memory leak detection for jax-js applications.
 *
 * jax-js uses a consuming ownership model: most operations dispose their
 * input arrays automatically. If you create an array and never pass it to
 * an operation or call `.dispose()`, the underlying backend memory leaks.
 * `checkLeaks` helps you find these leaks in tests or during development.
 *
 * ## Quick start
 *
 * ```ts
 * import * as jax from "@jax-js/jax";
 *
 * jax.checkLeaks.start();
 *
 * // ... your code ...
 *
 * const report = jax.checkLeaks.stop();
 * if (report.leaked > 0) {
 *   console.error(report.summary);
 * }
 * ```
 *
 * ## Example output
 *
 * ```
 * 5 slot(s) leaked (2 tracked array(s)):
 *
 *   test/numpy.test.ts:389:27 — 1× Array:float32[] rc=1  (via onesLike → fullLike)
 *   src/frontend/jvp.ts:107:39 — 1× Array:float32[] rc=1  (@jax-js/jax)
 *
 * 1 from user code, 1 in @jax-js/jax.
 *
 * 3 slot(s) not in tracked list (created before start() or by internal buffers).
 * rc=1 → never consumed. Call .dispose() or pass to an op.
 * ```
 *
 * ## Common leak patterns
 *
 * | Pattern | Fix |
 * | --- | --- |
 * | Intermediate result not consumed | Store in variable, pass to next op or `.dispose()` |
 * | `.ref.dataSync()` on last use | `.dataSync()` (drop `.ref`) |
 * | `.ref.js()` on last use | `.js()` (drop `.ref`) |
 * | Shape-only inspection (`x.shape`) | `x.dispose()` after |
 * | Optimizer state after training | `jax.tree.dispose(state)` |
 * | Generator early return | Wrap iterator body in `try/finally` |
 *
 * ## How it works
 *
 * `start()` records the current backend slot count as a baseline.
 * While active, every `Array` creation and `.dispose()` call is tracked in
 * a lightweight map. `stop()` flushes JIT caches, diffs the slot count, and
 * — if any slots leaked — resolves source-mapped stack traces to pinpoint
 * exactly where the leaking arrays were created.
 *
 * **Hot path** (every Array create/dispose): a boolean guard + `Map.set`/`delete`.
 * `new Error()` captures V8 frame pointers cheaply; the `.stack` string is
 * only formatted on first access inside `stop()` (cold path).
 *
 * **Cold path** (only when leaks exist): fetches each module's inline source
 * map via sync XHR (browser) or falls back to raw positions (Node/Deno),
 * decodes Base64-VLQ mappings, and resolves transpiled positions back to
 * original TypeScript source. Source maps are cached for the session.
 *
 * ## Vitest integration
 *
 * Add leak checking to every test with global hooks:
 *
 * ```ts
 * // test/setup.ts
 * import * as jax from "@jax-js/jax";
 * import { afterEach, beforeEach, expect } from "vitest";
 *
 * beforeEach(() => jax.checkLeaks.start());
 * afterEach(() => {
 *   const r = jax.checkLeaks.stop();
 *   expect(r.leaked, r.summary).toBe(0);
 * });
 * ```
 *
 * **Source-mapped stack traces** — By default, leak reports show locations
 * in the bundled output. To get original TypeScript file/line/col, add a
 * resolve alias in your `vitest.config.ts`:
 *
 * ```ts
 * // vitest.config.ts
 * import path from "node:path";
 *
 * export default defineConfig({
 *   resolve: {
 *     alias: {
 *       "@jax-js/jax": path.resolve(__dirname, "node_modules/@jax-js/jax/src/index.ts"),
 *     },
 *   },
 * });
 * ```
 *
 * **No concurrent tests** — `checkLeaks` uses global state (one active
 * session at a time). Do not use `test.concurrent()`. Sequential test
 * execution is the default in vitest.
 *
 * **Suppressing known leaks** — If a test exercises code with a known leak
 * you can't fix yet, temporarily pause tracking around that call:
 *
 * ```ts
 * test("known leaky path", () => {
 *   const a = jax.numpy.array([1, 2, 3]);
 *   jax.checkLeaks.stop();   // discard the current session
 *   leakyFunction(a);        // leaks here won't be counted
 *   jax.checkLeaks.start();  // restart a fresh session
 *   // ... rest of the test is still leak-checked ...
 * });
 * ```
 *
 * @module
 */

import { type Device, devices, getBackend } from "../backend";
import { _disposeAllJitCaches, ClosedJaxpr } from "./jaxpr";
// ── Tracking state (checked on the hot path) ──

/** True only between start() and stop(). Checked in Array ctor/dispose. */
export let _leakTrackingEnabled = false;

/**
 * Live tracked Arrays → Error captured at construction time.
 * `new Error()` records V8 frame pointers cheaply; the `.stack` string
 * is only formatted on first access (cold path, inside stop()).
 */
export const _leakTrackingMap = new Map<object, Error>();

/**
 * When `trackRefs` is enabled, records the last `.ref` call site per array.
 * Only populated when `start({ trackRefs: true })` — checked in array.ts.
 */
export let _trackRefsEnabled = false;
export const _lastRefMap = new Map<object, Error>();

// ── Snapshot state ──

const _startSlots = new Map<Device, number>();
let _active = false;

// ── ClosedJaxpr lifetime tracking (debug-only, enabled only while active) ──

let _closedHookInstalled = false;
let _closedIdCounter = 0;
const _closedIds = new WeakMap<ClosedJaxpr, number>();
const _closedCreated = new Map<
  number,
  {
    createdAt: Error;
    consts: number;
    ref: WeakRef<ClosedJaxpr>;
    disposed: boolean;
  }
>();

function ensureClosedJaxprHooks() {
  if (_closedHookInstalled) return;
  _closedHookInstalled = true;
  ClosedJaxpr._createHooks.push((closed, createdAt) => {
    if (!_active) return;
    const id = ++_closedIdCounter;
    _closedIds.set(closed, id);
    _closedCreated.set(id, {
      createdAt,
      consts: closed.consts.length,
      ref: new WeakRef(closed),
      disposed: false,
    });
  });
  ClosedJaxpr._disposeClosedHooks.push((closed) => {
    const id = _closedIds.get(closed);
    if (!id) return;
    const info = _closedCreated.get(id);
    if (info) info.disposed = true;
  });
}

// ── Cold-path helpers (only called when leaks are detected) ──

/**
 * Library source URL prefix, derived from this module's own URL.
 * Any stack frame whose URL starts with this is library-internal.
 * Computed once at load time (not on the hot path).
 */
const _libSrcPrefix = (() => {
  try {
    // This file is <root>/src/frontend/check-leaks.{ts,js}
    const url = import.meta.url;
    const i = url.lastIndexOf("/src/frontend/check-leaks.");
    if (i >= 0) return url.slice(0, i + 1) + "src/";
  } catch {
    // import.meta.url unavailable (e.g. CJS context)
  }
  return null;
})();

/**
 * Detect which package a raw frame URL belongs to.
 * Returns the package name (e.g. "@jax-js/jax", "some-lib") or null for user code.
 */
function detectPackage(rawUrl: string): string | null {
  // node_modules/<scope>/<pkg> or node_modules/<pkg>
  const nm = rawUrl.match(/node_modules\/((?:@[^/]+\/)?[^/]+)/);
  if (nm) return nm[1];
  // jax-js library src
  if (_libSrcPrefix !== null && rawUrl.startsWith(_libSrcPrefix))
    return "@jax-js/jax";
  return null;
}

// ── Cold-path source map resolution ──
// Only runs when leaks are detected. Uses sync XHR + minimal VLQ decoder
// to map transpiled positions back to original TypeScript source locations.

const _B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const _b64Lookup = new Uint8Array(128);
for (let i = 0; i < 64; i++) _b64Lookup[_B64.charCodeAt(i)] = i;

/** Decode a Base64-VLQ segment into signed deltas. */
function decodeVlq(encoded: string): number[] {
  const result: number[] = [];
  let shift = 0;
  let value = 0;
  for (let i = 0; i < encoded.length; i++) {
    let digit = _b64Lookup[encoded.charCodeAt(i)];
    const cont = digit & 32; // continuation bit
    digit &= 31;
    value += digit << shift;
    if (cont) {
      shift += 5;
    } else {
      result.push(value & 1 ? -(value >> 1) : value >> 1);
      shift = 0;
      value = 0;
    }
  }
  return result;
}

interface SourceMapData {
  sources: string[];
  mappingLines: string[];
}

/** url-without-query → parsed source map (or null if unavailable). */
const _smCache = new Map<string, SourceMapData | null>();

function fetchSourceMap(rawUrl: string): SourceMapData | null {
  const base = rawUrl.replace(/\?.*$/, "");
  if (_smCache.has(base)) return _smCache.get(base)!;
  try {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", rawUrl, false); // sync – cold path only
    xhr.send();
    if (xhr.status !== 200) {
      _smCache.set(base, null);
      return null;
    }
    const m = xhr.responseText.match(
      /\/\/[#@]\s*sourceMappingURL=data:[^;]+;base64,([A-Za-z0-9+/=]+)/,
    );
    if (!m) {
      _smCache.set(base, null);
      return null;
    }
    const map = JSON.parse(atob(m[1]));
    const sm: SourceMapData = {
      sources: map.sources ?? [],
      mappingLines: (map.mappings as string).split(";"),
    };
    _smCache.set(base, sm);
    return sm;
  } catch {
    _smCache.set(base, null);
    return null;
  }
}

/**
 * Resolve a generated (line, col) back to the original source position.
 * Returns null when no source map is available or the position can't be mapped.
 */
function resolvePosition(
  rawUrl: string,
  genLine: number,
  genCol: number,
): { file: string; line: number; col: number } | null {
  const sm = fetchSourceMap(rawUrl);
  if (!sm) return null;

  let gCol = 0;
  let srcFile = 0;
  let srcLine = 0;
  let srcCol = 0;
  let bestLine = -1;
  let bestCol = -1;
  let bestFile = -1;

  for (let i = 0; i < sm.mappingLines.length; i++) {
    gCol = 0; // generated column resets per line
    const segs = sm.mappingLines[i];
    if (!segs) continue;
    for (const seg of segs.split(",")) {
      if (!seg) continue;
      const d = decodeVlq(seg);
      gCol += d[0];
      if (d.length >= 4) {
        srcFile += d[1];
        srcLine += d[2];
        srcCol += d[3];
      }
      if (i === genLine - 1 && gCol <= genCol - 1) {
        bestFile = srcFile;
        bestLine = srcLine;
        bestCol = srcCol;
      }
    }
    if (i >= genLine - 1 && bestLine >= 0) break;
  }

  if (bestLine < 0) return null;

  // Resolve source path relative to the module URL directory
  let file = sm.sources[bestFile] ?? rawUrl;
  if (!file.startsWith("/") && !file.startsWith("http")) {
    try {
      const dir = rawUrl.substring(0, rawUrl.lastIndexOf("/") + 1);
      file = new URL(file, dir).href;
    } catch {
      /* keep file as-is */
    }
  }

  return {
    file,
    line: bestLine + 1,
    col: bestCol + 1,
  };
}

/** Convert a Vite/V8 stack location to a workspace-relative path. */
function toRelativePath(raw: string): string {
  let loc = raw.replace(/\?[^:]*(?=:\d+:\d+)/, ""); // strip Vite query params
  loc = loc.replace(/^https?:\/\/[^/]+\//, ""); // strip URL origin
  if (loc.startsWith("@fs/")) {
    loc = loc.slice(4);
    if (!loc.startsWith("/")) loc = "/" + loc;
  }
  loc = loc.replace(/^file:\/\//, "");
  for (const m of ["/src/", "/test/", "/packages/", "/bench/"]) {
    const i = loc.indexOf(m);
    if (i >= 0) return loc.slice(i + 1);
  }
  return loc;
}

/** Boilerplate frames to skip (Array constructor internals). */
const SKIP_FUNCS = ["new Array", "new PendingExecute", "Array2", "new Array2"];

/** Dep functions that appear in every creation stack — filtered from via chain. */
const VIA_SKIP = new Set([
  "bind",
  "bind1",
  "pureArray",
  "fullLower",
  "fullRaise",
]);

interface CreationSite {
  location: string;
  via: string | null;
  /** Package name if the creation site is inside a dependency, or null for user code. */
  pkg: string | null;
}

/**
 * Walk the Error's stack to find the creation site.
 * Accessing err.stack here triggers V8's lazy string formatting.
 */
function findCreationSite(err: Error): CreationSite {
  const stack = err.stack;
  if (!stack) return { location: "(unknown)", via: null, pkg: "(unknown)" };

  let lastDepLoc: string | null = null;
  let lastDepPkg: string | null = null;
  /** Dep function names (innermost-first) for building via chain. */
  const depChain: string[] = [];
  let prevClean: string | null = null;

  for (const line of stack.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("at ")) continue;

    // Parse "at FuncName (location)" or "at location"
    let rawLoc: string;
    let funcName: string | undefined;
    const m = trimmed.match(/^at\s+(.+?)\s+\((.+)\)$/);
    if (m) {
      funcName = m[1];
      rawLoc = m[2];
    } else {
      rawLoc = trimmed.slice(3);
    }

    // Try source map resolution for accurate line numbers
    let location: string;
    const lineCol = rawLoc.match(/:(\d+):(\d+)$/);
    if (lineCol) {
      const resolved = resolvePosition(
        rawLoc.replace(/:\d+:\d+$/, ""),
        +lineCol[1],
        +lineCol[2],
      );
      if (resolved) {
        location =
          toRelativePath(resolved.file) + `:${resolved.line}:${resolved.col}`;
      } else {
        location = toRelativePath(rawLoc);
      }
    } else {
      location = toRelativePath(rawLoc);
    }
    const pathOnly = location.replace(/:\d+:\d+$/, "");

    // Skip non-file frames
    if (
      pathOnly === "<anonymous>" ||
      pathOnly === "native" ||
      pathOnly.startsWith("eval ")
    )
      continue;
    // Skip constructor boilerplate
    if (funcName && SKIP_FUNCS.some((p) => funcName!.startsWith(p))) continue;

    // Dependency frame if URL matches a package, but test files are always user code
    const pkg = detectPackage(rawLoc);
    if (pkg && !/\.test\.[jt]sx?$/.test(pathOnly)) {
      lastDepLoc = location;
      lastDepPkg = pkg;
      // Collect unique, non-generic function names for the via chain
      const clean = funcName?.replace(/^(Module|Object)\./, "") ?? null;
      if (clean && clean !== prevClean && !VIA_SKIP.has(clean)) {
        depChain.push(clean);
        prevClean = clean;
      }
      continue;
    }

    // Found a user-code frame — build via chain (outermost → innermost call order)
    const chain = depChain.reverse().slice(0, 2);
    const via = chain.length > 0 ? chain.join(" → ") : null;
    return { location, via, pkg: null };
  }

  return {
    location: lastDepLoc ?? "(unknown)",
    via: null,
    pkg: lastDepPkg ?? "(unknown)",
  };
}

function safeDesc(arr: object): string {
  try {
    return (arr as { toString(): string }).toString();
  } catch {
    return "(unknown)";
  }
}

function safeRefCountNum(arr: object): number {
  try {
    const rc = (arr as { refCount: number }).refCount;
    return typeof rc === "number" ? rc : 0;
  } catch {
    return 0;
  }
}

function safeRefCount(arr: object): string {
  const rc = safeRefCountNum(arr);
  return rc > 0 ? ` rc=${rc}` : "";
}

// ── Public API ──

/** Result returned by {@link checkLeaks.stop}. */
export interface LeakReport {
  /** Number of unreleased backend slots (0 = no leaks). */
  leaked: number;
  /** Human-readable summary with source-mapped file:line:col locations. */
  summary: string;
}

/** A single entry from {@link checkLeaks.snapshot}. */
export interface SnapshotEntry {
  /** Array description (dtype, shape). */
  desc: string;
  /** Current reference count. */
  rc: number;
  /** Source-mapped creation location. */
  location: string;
  /** Package name, or null for user code. */
  pkg: string | null;
  /** Last .ref call site (only when trackRefs enabled). */
  lastRef: string | null;
}

export const checkLeaks = {
  /**
   * Begin tracking array allocations.
   *
   * Records the current backend slot count as a baseline and enables
   * per-allocation tracking. Call this before the code you want to check,
   * then call {@link stop} afterwards to get a leak report.
   *
   * Can be called once per test in a `beforeEach` hook, or manually around
   * any section of code.
   *
   * @param opts.trackRefs - When `true`, also records each `.ref` call site
   *   so the leak report can show `↳ last .ref at <location>` for rc≥2 leaks.
   *   Adds one `Map.set` per `.ref` call. Off by default.
   */
  start(opts?: { trackRefs?: boolean }): void {
    if (_active) {
      throw new Error(
        "checkLeaks.start() called while already active. " +
          "Concurrent leak tracking is not supported — do not use test.concurrent().",
      );
    }
    _leakTrackingMap.clear();
    _lastRefMap.clear();
    _closedCreated.clear();
    ensureClosedJaxprHooks();
    // _smCache is intentionally NOT cleared — source maps are session-level.
    _trackRefsEnabled = opts?.trackRefs ?? false;

    // Record baseline slot counts for ALL initialized backends, not just the
    // current default. This avoids false-clean results when the test's
    // beforeEach switches defaultDevice() after our global beforeEach.
    _startSlots.clear();
    for (const d of devices) {
      try {
        _startSlots.set(d, getBackend(d).slotCount());
      } catch {
        /* not initialized */
      }
    }
    _leakTrackingEnabled = true;
    _active = true;
  },

  /**
   * Return a snapshot of all currently-tracked arrays without stopping.
   *
   * Useful for mid-session debugging — you can inspect which arrays are
   * still alive at any point between `start()` and `stop()`.
   */
  snapshot(): SnapshotEntry[] {
    const entries: SnapshotEntry[] = [];
    for (const [arr, err] of _leakTrackingMap) {
      const site = findCreationSite(err);
      const refErr = _lastRefMap.get(arr);
      let lastRef: string | null = null;
      if (refErr) {
        const refSite = findCreationSite(refErr);
        lastRef = refSite.location;
      }
      entries.push({
        desc: safeDesc(arr),
        rc: safeRefCountNum(arr),
        location: site.location,
        pkg: site.pkg,
        lastRef,
      });
    }
    return entries;
  },

  /**
   * Stop tracking and return a leak report.
   *
   * Flushes all JIT caches (freeing their cached constants), computes the
   * backend slot delta since `start()`, and — when leaks are found — builds
   * a human-readable summary with source-mapped creation sites, reference
   * counts, and diagnostic tips.
   *
   * @returns `{ leaked: 0, summary }` when clean. When leaks exist,
   *   `leaked` is the number of unreleased backend slots and `summary`
   *   contains a formatted multi-line report.
   */
  stop(): LeakReport {
    if (!_active) return { leaked: 0, summary: "No leaks detected." };

    _leakTrackingEnabled = false;
    _trackRefsEnabled = false;
    _active = false;

    // Flush all jit caches so their const arrays are freed before counting.
    _disposeAllJitCaches();

    // Sum slot deltas across ALL backends to catch leaks regardless of which
    // device the test actually used.
    let leaked = 0;
    for (const [d, baseline] of _startSlots) {
      try {
        leaked += getBackend(d).slotCount() - baseline;
      } catch {
        /* backend gone */
      }
    }
    _startSlots.clear();

    if (leaked === 0) {
      _leakTrackingMap.clear();
      _lastRefMap.clear();
      _closedCreated.clear();
      return { leaked: 0, summary: "No leaks detected." };
    }

    // ── Cold path: build leak report ──

    const groups = new Map<
      string,
      {
        count: number;
        descs: string[];
        via: string | null;
        pkg: string | null;
        maxRc: number;
        lastRefLocs: string[];
      }
    >();
    /** Leak counts by package (null key = user code). */
    const pkgCounts = new Map<string | null, number>();

    const leakedArrays = new Set<object>();
    for (const [arr, err] of _leakTrackingMap) {
      leakedArrays.add(arr);
      const site = findCreationSite(err);
      let g = groups.get(site.location);
      if (!g) {
        g = {
          count: 0,
          descs: [],
          via: site.via,
          pkg: site.pkg,
          maxRc: 0,
          lastRefLocs: [],
        };
        groups.set(site.location, g);
      }
      g.count++;
      g.maxRc = Math.max(g.maxRc, safeRefCountNum(arr));
      const desc = safeDesc(arr) + safeRefCount(arr);
      if (g.descs.length < 4) g.descs.push(desc);
      // Collect last .ref sites for rc≥2 arrays
      const refErr = _lastRefMap.get(arr);
      if (refErr && safeRefCountNum(arr) >= 2) {
        const refSite = findCreationSite(refErr);
        if (
          g.lastRefLocs.length < 2 &&
          !g.lastRefLocs.includes(refSite.location)
        ) {
          g.lastRefLocs.push(refSite.location);
        }
      }
      pkgCounts.set(site.pkg, (pkgCounts.get(site.pkg) ?? 0) + 1);
    }

    const closedHoldingLeaks: Array<{
      id: number;
      consts: number;
      site: string;
    }> = [];
    for (const [id, info] of _closedCreated.entries()) {
      const closed = info.ref.deref();
      if (!closed) continue;
      // If any leaked Array object is directly referenced by this ClosedJaxpr.consts,
      // it's a strong indicator that the jaxpr (or a closure holding it) is the owner.
      let holds = false;
      for (const c of closed.consts) {
        if (leakedArrays.has(c as any)) {
          holds = true;
          break;
        }
      }
      if (holds) {
        const site = findCreationSite(info.createdAt).location;
        closedHoldingLeaks.push({ id, consts: info.consts, site });
      }
    }

    const undisposedClosed = [..._closedCreated.entries()].filter(
      ([, info]) => !info.disposed && info.consts > 0,
    );

    _leakTrackingMap.clear();
    _lastRefMap.clear();
    _closedCreated.clear();

    // User-code sites first (most actionable), then dependency, by count desc
    const sorted = [...groups.entries()].sort(
      (a, b) =>
        Number(a[1].pkg !== null) - Number(b[1].pkg !== null) ||
        b[1].count - a[1].count,
    );

    const total = [...pkgCounts.values()].reduce((a, b) => a + b, 0);
    const lines: string[] = [
      `${leaked} slot(s) leaked (${total} tracked array(s)):\n`,
    ];

    for (const [loc, g] of sorted) {
      const descs =
        g.descs.join(", ") + (g.count > g.descs.length ? ", …" : "");
      // Show both package and via when available
      const tagParts: string[] = [];
      if (g.pkg) tagParts.push(g.pkg);
      if (g.via) tagParts.push(`via ${g.via}`);
      const tag = tagParts.length > 0 ? `  (${tagParts.join(", ")})` : "";
      lines.push(`  ${loc} — ${g.count}× ${descs}${tag}`);
      // Show last .ref sites for rc≥2 leaks
      for (const refLoc of g.lastRefLocs) {
        lines.push(`    ↳ last .ref at ${refLoc}`);
      }
    }

    const summaryParts: string[] = [];
    for (const [pkg, count] of pkgCounts) {
      summaryParts.push(pkg ? `${count} in ${pkg}` : `${count} from user code`);
    }
    lines.push("", summaryParts.join(", ") + ".");

    // Diagnostic tips based on observed patterns
    const tips: string[] = [];
    if (leaked > total) {
      tips.push(
        `${leaked - total} slot(s) not in tracked list (created before start() or by internal buffers).`,
      );
    }
    if (total > leaked) {
      tips.push(
        `Some tracked arrays share backend storage (${total} tracked, ${leaked} slots).`,
      );
    }
    if (sorted.some(([, g]) => !g.pkg && g.maxRc === 1)) {
      tips.push("rc=1 → never consumed. Call .dispose() or pass to an op.");
    }
    if (sorted.some(([, g]) => g.maxRc >= 2)) {
      tips.push("rc≥2 → extra .ref without matching .dispose().");
    }
    if (pkgCounts.size > (pkgCounts.has(null) ? 1 : 0)) {
      tips.push("Package-tagged leaks are library bugs, not user error.");
    }
    if (tips.length === 0) {
      tips.push(
        "Call .dispose() on unused results, or use .ref to keep arrays alive.",
      );
    }

    if (undisposedClosed.length > 0) {
      tips.push(
        `${undisposedClosed.length} ClosedJaxpr(s) created during checkLeaks were not disposed (may retain const arrays):`,
      );
      for (const [id, info] of undisposedClosed.slice(0, 6)) {
        const site = findCreationSite(info.createdAt);
        tips.push(
          `  - ClosedJaxpr #${id} consts=${info.consts} at ${site.location}`,
        );
      }
      if (undisposedClosed.length > 6) {
        tips.push(`  - … (+${undisposedClosed.length - 6} more)`);
      }
    }

    if (closedHoldingLeaks.length > 0) {
      tips.push(
        `Leaked arrays are referenced by ${closedHoldingLeaks.length} ClosedJaxpr(s):`,
      );
      for (const e of closedHoldingLeaks.slice(0, 6)) {
        tips.push(
          `  - ClosedJaxpr #${e.id} consts=${e.consts} created at ${e.site}`,
        );
      }
      if (closedHoldingLeaks.length > 6) {
        tips.push(`  - … (+${closedHoldingLeaks.length - 6} more)`);
      }
    }
    lines.push("", ...tips);

    return { leaked, summary: lines.join("\n") };
  },

  get active(): boolean {
    return _active;
  },
};

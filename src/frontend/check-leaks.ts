/**
 * Leak detection diagnostic for jax-js arrays.
 *
 * Zero overhead when not active. When enabled via `checkLeaks.start()`,
 * snapshots backend slot counts and tracks Array creations with stack traces.
 * `checkLeaks.stop()` returns a report of leaked backend slots.
 *
 * The `leaked` count is based on backend slot count delta — the same metric
 * as `slotCount()` — so it exactly matches the old manual leak detection
 * pattern. The tracking map provides additional diagnostic details (array
 * descriptions and creation locations) for debugging.
 *
 * Usage:
 *   import { checkLeaks } from "@jax-js/jax";
 *   checkLeaks.start();
 *   // ... user code ...
 *   const report = checkLeaks.stop();
 *   console.log(report.summary);
 *
 * @module
 */

import { type Device, getBackend } from "../backend";

/** Whether leak tracking is currently enabled. Checked in Array constructor/dispose. */
export let _leakTrackingEnabled = false;

/** Map from live tracked Array → creation stack trace string. */
export const _leakTrackingMap = new Map<object, string>();

/** Snapshot of slot counts per backend device at start. */
const _startSlotCounts = new Map<Device, number>();

/** Devices to check, captured at start time. */
const _devices: Device[] = [];

/** Get slot count from a backend (all 3 implement slotCount()). */
function slotCount(device: Device): number {
  return (getBackend(device) as any).slotCount();
}

/** Parse a stack trace to find the first user-code frame (skip jax-js internals). */
function parseUserFrame(stack: string): string {
  const lines = stack.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    // Skip the Error line itself
    if (trimmed.startsWith("Error")) continue;
    // Skip jax-js internal frames (src/, dist/, node_modules)
    if (
      trimmed.includes("/src/frontend/") ||
      trimmed.includes("/src/backend/") ||
      trimmed.includes("/src/library/") ||
      trimmed.includes("/src/alu") ||
      trimmed.includes("/src/routine") ||
      trimmed.includes("/src/shape") ||
      trimmed.includes("/src/tree") ||
      trimmed.includes("/src/utils") ||
      trimmed.includes("/src/index") ||
      trimmed.includes("/dist/") ||
      trimmed.includes("node_modules")
    ) {
      continue;
    }
    // Found a user frame
    return trimmed;
  }
  // Fallback: return first "at" line
  return lines.find((l) => l.trim().startsWith("at"))?.trim() ?? "(unknown)";
}

export interface LeakReport {
  /** Number of leaked backend slots (delta from start). */
  leaked: number;
  /** Array descriptions with creation locations (best-effort from tracking map). */
  details: string[];
  /** Human-readable summary string. */
  summary: string;
}

export const checkLeaks = {
  /**
   * Start tracking array allocations. Takes a snapshot of backend slot counts
   * and enables the creation tracking map for diagnostic details.
   */
  start(): void {
    _leakTrackingMap.clear();
    _startSlotCounts.clear();
    _devices.length = 0;

    // Snapshot slot count on the default backend only.
    // This matches the behavior of the traditional slotCount() pattern:
    //   const before = (getBackend() as any).slotCount();
    try {
      const device = getBackend().type;
      const count = slotCount(device);
      _startSlotCounts.set(device, count);
      _devices.push(device);
    } catch {
      // Backend not initialized — skip.
    }

    _leakTrackingEnabled = true;
  },

  /**
   * Stop tracking and return a report of leaked backend slots.
   * The `leaked` count is the total slot count delta across all backends.
   * The `details` array provides descriptions of tracked arrays still alive.
   */
  stop(): LeakReport {
    _leakTrackingEnabled = false;

    // Compute total leaked slots across all backends.
    let leaked = 0;
    for (const device of _devices) {
      const before = _startSlotCounts.get(device) ?? 0;
      const after = slotCount(device);
      leaked += after - before;
    }

    // Build diagnostic details from the tracking map.
    const details: string[] = [];
    for (const [arr, stack] of _leakTrackingMap) {
      // Use .toString() directly — String(arr) triggers [Symbol.toPrimitive]
      // which throws for non-scalar arrays.
      const desc =
        typeof (arr as any).toString === "function"
          ? (arr as any).toString()
          : "(unknown array)";
      const frame = parseUserFrame(stack);
      details.push(`${desc} created at ${frame}`);
    }

    _leakTrackingMap.clear();
    _startSlotCounts.clear();
    _devices.length = 0;

    const summary =
      leaked === 0
        ? "No leaks detected."
        : `${leaked} slot(s) leaked:\n${details.length > 0 ? details.map((d) => `  - ${d}`).join("\n") : "  (no tracked arrays — leak may be from internal allocations)"}\n\nWrap computation in jit() or call .dispose().`;

    return { leaked, details, summary };
  },

  /** Whether leak tracking is currently active. */
  get active(): boolean {
    return _leakTrackingEnabled;
  },
};

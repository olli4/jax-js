import { JsTree, numpy as np, tree } from "@jax-js/jax";

import {
  GradientTransformation,
  identity,
  initEmptyState,
  ScalarOrSchedule,
  Schedule,
} from "./base";
import {
  treeBiasCorrection,
  treeNorm,
  treeUpdateMoment,
  treeZerosLike,
} from "./treeUtils";

function u32(x: number): np.Array {
  return np.array(x, { dtype: np.uint32 });
}

export type ScaleByAdamOptions = {
  b1?: number;
  b2?: number;
  eps?: number;
  epsRoot?: number;
  nesterov?: boolean;
};

export function scaleByAdam({
  b1 = 0.9,
  b2 = 0.999,
  eps = 1e-8,
  epsRoot = 0.0,
  nesterov = false,
}: ScaleByAdamOptions = {}): GradientTransformation {
  return {
    init(params) {
      const mu = treeZerosLike(params); // first moment
      const nu = treeZerosLike(params); // second moment
      return { count: u32(0), mu, nu };
    },
    update(updates, state, params) {
      let { count, mu, nu } = state as {
        count: np.Array;
        mu: JsTree<np.Array>;
        nu: JsTree<np.Array>;
      };
      mu = treeUpdateMoment(updates, mu, b1, 1);
      nu = treeUpdateMoment(updates, nu, b2, 2);
      count = count.add(1);
      let muHat: typeof mu;
      if (nesterov) {
        muHat = tree.map(
          (m: np.Array, g: np.Array) => m.mul(b1).add(g.mul(1 - b1)),
          treeBiasCorrection(mu, b1, count.add(1)),
          treeBiasCorrection(updates, b1, count),
        );
      } else {
        muHat = treeBiasCorrection(mu, b1, count);
      }
      const nuHat = treeBiasCorrection(nu, b2, count);
      updates = tree.map(
        (m: np.Array, v: np.Array) => m.div(np.sqrt(v.add(epsRoot)).add(eps)),
        muHat,
        nuHat,
      ) as typeof updates;
      return [updates, { count, mu, nu }];
    },
  };
}

/** Scale by a constant step size. */
export function scale(stepSize: number): GradientTransformation {
  return {
    init: initEmptyState,
    update(updates, state, params) {
      updates = tree.map((g: np.Array) => g.mul(stepSize), updates);
      return [updates, state];
    },
  };
}

/** Scale updates using a custom schedule for the step size. */
export function scaleBySchedule(stepSizeFn: Schedule): GradientTransformation {
  return {
    init(_params) {
      return { count: u32(0) }; // initial step
    },
    update(updates, state, _params) {
      const { count } = state as { count: np.Array };
      const countInt = count.item();
      const stepSize = stepSizeFn(countInt);
      updates = tree.map((g: np.Array) => g.mul(stepSize), updates);
      return [updates, { count: u32(countInt + 1) }];
    },
  };
}

/** Scale by the (negative) learning rate (either as scalar or as schedule). */
export function scaleByLearningRate(
  learningRate: ScalarOrSchedule,
  flipSign = true,
): GradientTransformation {
  if (learningRate === undefined) return identity();
  const m = flipSign ? -1 : 1;
  if (typeof learningRate === "function") {
    return scaleBySchedule((count) => m * learningRate(count));
  }
  return scale(m * learningRate);
}

/** Clip gradients by global norm. */
export function clipByGlobalNorm(maxNorm: number): GradientTransformation {
  return {
    init: initEmptyState,
    update(updates, state, _params) {
      const gNorm = treeNorm(updates);
      const trigger = np.less(gNorm, maxNorm);

      const clippedUpdates = tree.map(
        (t: np.Array) =>
          np.where(trigger, t, t.div(gNorm).mul(maxNorm)),
        updates,
      );

      trigger.dispose();
      gNorm.dispose();

      return [clippedUpdates, state];
    },
  };
}

export type MaskFn = (tree: JsTree<np.Array>) => JsTree<np.Array>;

export type AddDecayedWeightsOptions = {
  weightDecay?: ScalarOrSchedule;
  mask?: JsTree<np.Array> | MaskFn | null;
};

/** Add parameter scaled by weight decay. */
export function addDecayedWeights({
  weightDecay = 0.0,
  mask = null,
}: AddDecayedWeightsOptions = {}): GradientTransformation {
  const isSchedule = typeof weightDecay === "function";

  return {
    init(_params) {
      if (isSchedule) {
        return { count: u32(0) };
      } else {
        return [];
      }
    },
    update(updates, state, params) {
      if (!params) {
        throw new Error("addDecayedWeights requires params to be provided");
      }

      let newState: typeof state;
      let currentWeightDecay: number;

      if (isSchedule) {
        const { count } = state as { count: np.Array };
        const countInt = count.item();
        currentWeightDecay = (weightDecay as Schedule)(countInt);
        newState = { count: u32(countInt + 1) };
      } else {
        currentWeightDecay = weightDecay as number;
        newState = state;
      }

      if (currentWeightDecay === 0.0) {
        return [updates, newState];
      }

      let decayedParams: JsTree<np.Array>;
      if (mask) {
        const maskTree =
          typeof mask === "function" ? mask(updates) : mask;

        decayedParams = tree.map(
          (p: np.Array, m: np.Array) => p.mul(m).mul(currentWeightDecay),
          params,
          maskTree,
        );
      } else {
        decayedParams = tree.map(
          (p: np.Array) => p.mul(currentWeightDecay),
          params,
        );
      }

      updates = tree.map(
        (g: np.Array, d: np.Array) => g.add(d),
        updates,
        decayedParams,
      ) as typeof updates;

      return [updates, newState];
    },
  };
}

export type TraceOptions = {
  decay?: number;
  nesterov?: boolean;
};

/** Compute a trace of past updates. */
export function trace({
  decay = 0.9,
  nesterov = false,
}: TraceOptions = {}): GradientTransformation {
  return {
    init(params) {
      const trace = treeZerosLike(params);
      return { trace };
    },
    update(updates, state, _params) {
      const { trace: prevTrace } = state as { trace: JsTree<np.Array> };

      // new_trace = g + decay * t
      const newTrace = tree.map(
        (g: np.Array, t: np.Array) => g.add(t.mul(decay)),
        updates,
        prevTrace,
      );

      let finalUpdates: typeof updates;
      if (nesterov) {
        // Nesterov: updates = g + decay * new_trace
        finalUpdates = tree.map(
          (g: np.Array, t: np.Array) => g.add(t.mul(decay)),
          updates,
          newTrace,
        ) as typeof updates;
      } else {
        // Standard momentum: updates = new_trace
        finalUpdates = newTrace as typeof updates;
      }

      return [finalUpdates, { trace: newTrace }];
    },
  };
}

import { grad, jit, nn, numpy as np, random } from "@jax-js/jax";
import { applyUpdates, sgd } from "@jax-js/optax";

// Logistic regression on a sample dataset.
//   > Classifier: y = sigmoid(X @ w)
//   > Loss: Binary cross-entropy,
//           loss(w) = y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)

const wTrue = np.array([2.0, -1.0, 0.5, -1.5]);

const key = random.key(0);
const X = random.uniform(key, [500, 4], { minval: -1, maxval: 1 });
const y = np.dot(X, wTrue).greater(0).astype(np.float32);

// Define loss function (binary cross-entropy).
const lossFn = jit((w: np.Array) => {
  const logits = np.dot(X, w);
  const logP = nn.logSigmoid(logits);
  const logNotP = nn.logSigmoid(np.negative(logits));
  const loss = np
    .add(y.mul(logP), np.subtract(1, y).mul(logNotP))
    .mean()
    .neg();
  return loss;
});

// Try adding jit() to lossGrad to see the code get faster.
const lossGrad = grad(lossFn);

// Training loop.
const steps = 100;
const solver = sgd(0.2);

let w = np.zerosLike(wTrue);
let optState = solver.init(w);
let updates: typeof w;

for (let step = 0; step < steps; step++) {
  const grads = lossGrad(w);
  [updates, optState] = solver.update(grads, optState);
  w = applyUpdates(w, updates);
  if (step % 20 === 19) {
    const loss = await lossFn(w).jsAsync();
    console.log(`Step ${step + 1}: loss = ${loss}`);
  }
}

// Output learned weights.
console.log("Learned weights:", await w.jsAsync());

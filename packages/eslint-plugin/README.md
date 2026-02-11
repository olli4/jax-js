# @jax-js/eslint-plugin

ESLint rules for catching **array memory leaks** in [jax-js](https://github.com/ekzhang/jax-js)
applications at edit time.

jax-js uses a consuming ownership model: most operations dispose their input arrays automatically.
If you create an array and never pass it to an operation or call `.dispose()`, the underlying
backend memory leaks. These lint rules catch the most common leak patterns statically, so you get
red squiggles in your editor instead of discovering leaks at runtime with `checkLeaks`.

## Rules

### `@jax-js/no-unnecessary-ref`

Warns when `.ref` is used on a variable whose last usage is the `.ref` chain itself. This is a
**guaranteed leak**: `.ref` bumps the reference count, the chained consuming method decrements it
back, but the original variable still holds `rc=1` and nobody will ever dispose it.

```ts
// ❌ Bad — x leaks (rc stays at 1 after dataSync disposes the .ref copy)
const x = np.array([1, 2, 3]);
const data = x.ref.dataSync();

// ✅ Good — x is consumed directly
const x = np.array([1, 2, 3]);
const data = x.dataSync();

// ✅ Good — x.ref is needed because x is used again later
const x = np.array([1, 2, 3]);
const data = x.ref.dataSync();
x.dispose();
```

**Autofix:** Removes `.ref` from the chain.

### `@jax-js/require-consume`

Warns when an array stored in a variable is never consumed — never passed to a consuming operation,
returned, yielded, or explicitly disposed. Accessing only non-consuming properties like `.shape`,
`.dtype`, `.ndim`, `.size`, `.device`, or `.refCount` does not count as consumption.

```ts
// ❌ Bad — x is created but never consumed or disposed
const x = np.array([1, 2, 3]);
console.log(x.shape);

// ✅ Good
const x = np.array([1, 2, 3]);
console.log(x.shape);
x.dispose();
```

## Setup

Add the plugin to your flat ESLint config:

```ts
// eslint.config.ts
import jaxJs from "@jax-js/eslint-plugin";

export default [
  // ... your other config
  jaxJs.configs.recommended,
];
```

Or enable rules individually:

```ts
import jaxJs from "@jax-js/eslint-plugin";

export default [
  {
    plugins: { "@jax-js": jaxJs },
    rules: {
      "@jax-js/no-unnecessary-ref": "warn",
      "@jax-js/require-consume": "warn",
    },
  },
];
```

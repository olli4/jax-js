/**
 * @ownership/require-try-finally-symmetry
 *
 * Principle mapping:
 * - Principle 3: release retained handles symmetrically
 * - Principle 4: temporaries are local liabilities
 *
 * This rule focuses on a high-signal transform-internal pattern:
 * If a retained handle (`const t = x.ref`) is created inside a try-block,
 * cleanup should happen in that try's finally block.
 */

const TS_WRAPPERS = new Set([
  "TSAsExpression",
  "TSNonNullExpression",
  "TSTypeAssertion",
  "TSInstantiationExpression",
  "ParenthesizedExpression",
]);

function isIdentifier(node, name) {
  return node && node.type === "Identifier" && node.name === name;
}

function unwrap(node) {
  let current = node;
  while (current && TS_WRAPPERS.has(current.type)) {
    current = current.expression;
  }
  return current;
}

function walk(node, fn) {
  if (!node || typeof node !== "object") return;
  fn(node);
  for (const key of Object.keys(node)) {
    if (key === "parent") continue;
    const value = node[key];
    if (Array.isArray(value)) {
      for (const child of value) {
        if (child && typeof child.type === "string") walk(child, fn);
      }
    } else if (value && typeof value.type === "string") {
      walk(value, fn);
    }
  }
}

function retainedFromRef(declarator) {
  if (!declarator || declarator.type !== "VariableDeclarator") return null;
  if (!declarator.init || declarator.id.type !== "Identifier") return null;

  const init = unwrap(declarator.init);
  if (!init || init.type !== "MemberExpression") return null;
  if (init.computed) return null;
  if (!isIdentifier(init.property, "ref")) return null;

  return { name: declarator.id.name, idNode: declarator.id };
}

function retainedFromRefAssignment(node) {
  if (!node || node.type !== "AssignmentExpression") return null;
  if (!node.right || node.left.type !== "Identifier") return null;

  const right = unwrap(node.right);
  if (!right || right.type !== "MemberExpression" || right.computed) {
    return null;
  }
  if (!isIdentifier(right.property, "ref")) return null;

  return { name: node.left.name, idNode: node.left };
}

function collectRetainedInTryBlock(blockNode) {
  const retained = new Map();
  walk(blockNode, (node) => {
    const found = retainedFromRef(node);
    if (found) retained.set(found.name, found);

    const assigned = retainedFromRefAssignment(node);
    if (assigned) retained.set(assigned.name, assigned);
  });
  return [...retained.values()];
}

function hasDisposeCall(node, varName) {
  let found = false;
  walk(node, (n) => {
    if (found || n.type !== "CallExpression") return;
    const callee = unwrap(n.callee);
    if (!callee || callee.type !== "MemberExpression" || callee.computed) return;
    if (!isIdentifier(callee.object, varName)) return;
    if (!isIdentifier(callee.property, "dispose")) return;
    found = true;
  });
  return found;
}

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require `.ref` temporaries created in try blocks to be released in finally",
      recommended: true,
    },
    schema: [],
    messages: {
      missingFinally:
        "Retained handle `{{name}}` is created inside try without a finally block. Add finally cleanup for symmetric release.",
      missingFinallyDispose:
        "Retained handle `{{name}}` is created inside try but not disposed in finally. Move cleanup to finally for error-path parity.",
    },
  },

  create(context) {
    return {
      TryStatement(node) {
        const retained = collectRetainedInTryBlock(node.block);
        if (retained.length === 0) return;

        if (!node.finalizer) {
          for (const ret of retained) {
            context.report({
              node: ret.idNode,
              messageId: "missingFinally",
              data: { name: ret.name },
            });
          }
          return;
        }

        for (const ret of retained) {
          if (!hasDisposeCall(node.finalizer, ret.name)) {
            context.report({
              node: ret.idNode,
              messageId: "missingFinallyDispose",
              data: { name: ret.name },
            });
          }
        }
      },
    };
  },
};

export default rule;

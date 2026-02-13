/**
 * @ownership/require-wrapper-dispose-symmetry
 *
 * Principle mapping:
 * - Principle 3: release retained handles symmetrically
 * - Wrapper lifecycle: release retained state, then dispose wrapped inner
 *
 * Rule intent (heuristic):
 * - In a `dispose()` method, if `this.inner.dispose()` exists,
 *   it must be the LAST `.dispose()` call in that method body.
 */

function isIdentifier(node, name) {
  return node && node.type === "Identifier" && node.name === name;
}

function getDisposeTarget(node) {
  if (!node || node.type !== "CallExpression") return null;

  const callee = node.callee;
  if (!callee || callee.type !== "MemberExpression" || callee.computed) {
    return null;
  }
  if (!isIdentifier(callee.property, "dispose")) return null;

  const target = callee.object;
  if (
    target &&
    target.type === "MemberExpression" &&
    !target.computed &&
    target.object.type === "ThisExpression" &&
    target.property.type === "Identifier"
  ) {
    return target.property.name;
  }

  return null;
}

function collectTopLevelDisposeCalls(block) {
  const calls = [];
  if (!block || block.type !== "BlockStatement") return calls;

  for (const stmt of block.body) {
    if (stmt.type !== "ExpressionStatement") continue;
    const target = getDisposeTarget(stmt.expression);
    if (!target) continue;
    calls.push({ target, node: stmt.expression });
  }

  return calls;
}

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require wrapper dispose symmetry: retained cleanup before this.inner.dispose()",
      recommended: true,
    },
    schema: [],
    messages: {
      innerNotLast:
        "In dispose(), `this.inner.dispose()` should be last; clean retained state before disposing inner.",
    },
  },

  create(context) {
    return {
      MethodDefinition(node) {
        if (!isIdentifier(node.key, "dispose")) return;
        if (!node.value || node.value.type !== "FunctionExpression") return;

        const body = node.value.body;
        const calls = collectTopLevelDisposeCalls(body);
        if (calls.length === 0) return;

        const innerIndices = calls
          .map((call, index) => ({ ...call, index }))
          .filter((call) => call.target === "inner");

        if (innerIndices.length === 0) return;

        const lastIndex = calls.length - 1;
        for (const inner of innerIndices) {
          if (inner.index !== lastIndex) {
            context.report({
              node: inner.node,
              messageId: "innerNotLast",
            });
          }
        }
      },
    };
  },
};

export default rule;

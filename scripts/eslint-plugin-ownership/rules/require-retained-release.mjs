/**
 * @ownership/require-retained-release
 *
 * Principle mapping:
 * - Principle 2: Retention boundaries must be explicit (`const y = x.ref`)
 * - Principle 3: Retained handles must have a release path
 *
 * This rule flags retained handles created from `.ref` that never show an
 * explicit terminal action inside the current function scope.
 */

function isIdentifier(node, name) {
  return node && node.type === "Identifier" && node.name === name;
}

function isDisposeCallOnIdentifier(refIdentifier, varName) {
  const member = refIdentifier.parent;
  if (!member || member.type !== "MemberExpression") return false;
  if (member.computed || !isIdentifier(member.object, varName)) return false;
  if (!isIdentifier(member.property, "dispose")) return false;

  const call = member.parent;
  return !!(call && call.type === "CallExpression" && call.callee === member);
}

function isTransferredUsage(refIdentifier) {
  const p = refIdentifier.parent;
  if (!p) return false;

  if (p.type === "ReturnStatement" && p.argument === refIdentifier) return true;
  if (p.type === "YieldExpression" && p.argument === refIdentifier) return true;
  if (p.type === "AssignmentExpression" && p.right === refIdentifier) return true;

  if (p.type === "CallExpression" && p.arguments.includes(refIdentifier)) {
    return true;
  }

  if (p.type === "ArrayExpression" || p.type === "ObjectExpression") {
    return true;
  }

  return false;
}

const rule = {
  meta: {
    type: "problem",
    docs: {
      description:
        "Require retained `.ref` handles to have an explicit release/transfer path",
      recommended: true,
    },
    hasSuggestions: true,
    schema: [],
    messages: {
      missingRelease:
        "Retained handle `{{name}}` from `.ref` has no explicit release path in this scope.",
      suggestDispose: "Add `{{name}}.dispose()` at end of scope (if ownership stays local)",
    },
  },

  create(context) {
    const sourceCode = context.sourceCode;

    return {
      VariableDeclarator(node) {
        if (!node.init || node.id.type !== "Identifier") return;
        const varName = node.id.name;

        if (
          node.init.type !== "MemberExpression" ||
          node.init.computed ||
          !isIdentifier(node.init.property, "ref")
        ) {
          return;
        }

        const scope = sourceCode.getScope(node);
        const variable = scope.set.get(varName);
        if (!variable) return;

        const refs = variable.references.filter((r) => r.identifier !== node.id);

        let hasTerminal = false;
        for (const ref of refs) {
          const id = ref.identifier;
          if (isDisposeCallOnIdentifier(id, varName)) {
            hasTerminal = true;
            break;
          }
          if (isTransferredUsage(id)) {
            hasTerminal = true;
            break;
          }
        }

        if (!hasTerminal) {
          context.report({
            node: node.id,
            messageId: "missingRelease",
            data: { name: varName },
            suggest: [
              {
                messageId: "suggestDispose",
                data: { name: varName },
                fix: (fixer) => {
                  const statement = node.parent && node.parent.type === "VariableDeclaration"
                    ? node.parent
                    : node;
                  return fixer.insertTextAfter(statement, `\n${varName}.dispose();`);
                },
              },
            ],
          });
        }
      },
    };
  },
};

export default rule;

import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import * as eslintImport from "eslint-plugin-import";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores(["dist/", "website/"]),
  {
    files: ["**/*.{js,mjs,cjs,ts}"],
    plugins: { js },
    extends: ["js/recommended"],
  },
  {
    files: ["**/*.{js,mjs,cjs,ts}"],
    languageOptions: { globals: globals.browser },
  },
  tseslint.configs.recommended,
  {
    plugins: { import: eslintImport },
    rules: {
      "@typescript-eslint/no-array-constructor": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-this-alias": "off",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          varsIgnorePattern: "^_",
          argsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
      "import/newline-after-import": "warn",
      "import/order": [
        "warn",
        {
          alphabetize: {
            order: "asc",
          },
          groups: ["builtin", "external"],
          "newlines-between": "always",
        },
      ],
      "prefer-const": ["warn", { destructuring: "all" }],
      "sort-imports": [
        "warn",
        {
          allowSeparatedGroups: true,
          ignoreCase: true,
          ignoreDeclarationSort: true,
          ignoreMemberSort: false,
          memberSyntaxSortOrder: ["none", "all", "multiple", "single"],
        },
      ],
    },
  },
]);

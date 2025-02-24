<script lang="ts">
  import { SplitPane } from "@rich_harris/svelte-split-pane";
  import type { Plugin } from "@rollup/browser";
  import { ArrowRightIcon, PaletteIcon, PlayIcon } from "lucide-svelte";

  import ReplEditor from "$lib/repl/ReplEditor.svelte";

  const codeSamples: {
    title: string;
    code: string;
  }[] = [
    {
      title: "Arrays",
      code: String.raw`import { grad, numpy as np } from "@jax-js/core";

const f = (x: np.Array) => x.mul(x);
const df = grad(f);

const x = np.array(3);
console.log(f(x).js());
console.log(df(x).js());
`,
    },
    {
      title: "Logistic regression",
      code: String.raw`import { numpy as np } from "@jax-js/core";

const X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]);
const y = np.dot(X, np.array([1, 2])).add(3);

// TODO
`,
    },
    {
      title: "Mandelbrot set",
      code: String.raw`import { numpy as np } from "@jax-js/core";
// TODO
`,
    },
  ];

  let selected = $state(0);
  let replEditor: ReplEditor;

  async function handleFormat() {
    const { formatWithCursor } = await import("prettier");
    const prettierParserTypescript = await import("prettier/parser-typescript");
    const prettierPluginEstree = await import("prettier/plugins/estree");

    const code = replEditor.getText();
    try {
      const { formatted, cursorOffset } = await formatWithCursor(code, {
        parser: "typescript",
        plugins: [prettierParserTypescript, prettierPluginEstree as any],
        cursorOffset: replEditor.getCursorOffset(),
      });
      replEditor.setText(formatted);
      replEditor.setCursorOffset(cursorOffset);
    } catch (e: any) {
      // TODO: Display the error in the console.
      alert(e.toString());
    }
  }

  async function handleRun() {
    const jax = await import("@jax-js/core");
    const ts = await import("typescript");
    const { rollup } = await import("@rollup/browser");

    const userCode = replEditor.getText();

    // Create a simple virtual module plugin to resolve our in-memory modules.
    const virtualPlugin: Plugin = {
      name: "virtual",
      resolveId(id) {
        // We treat 'index.ts' as the user code entry point.
        if (id === "index.ts") {
          return id;
        } else {
          throw new Error("Module not found: " + id);
        }
      },
      load(id) {
        if (id === "index.ts") {
          return userCode;
        } else {
          return null;
        }
      },
    };

    const typescriptPlugin: Plugin = {
      name: "typescript",
      transform(code, id) {
        if (id.endsWith(".ts")) {
          return ts.transpileModule(code, {
            compilerOptions: {
              module: ts.ModuleKind.ESNext,
              target: ts.ScriptTarget.ES2022,
            },
          }).outputText;
        }
        return null;
      },
    };

    // Use @rollup/browser to bundle the code.
    const bundle = await rollup({
      input: "index.ts",
      plugins: [typescriptPlugin, virtualPlugin],
      external: ["@jax-js/core"],
    });

    const { output } = await bundle.generate({
      file: "bundle.js",
      format: "iife",
      globals: {
        "@jax-js/core": "JAX",
      },
    });

    const bundledCode = output[0].code;
    new Function("JAX", bundledCode)(jax);
  }
</script>

<div class="h-dvh">
  <SplitPane
    type="horizontal"
    pos="288px"
    min="240px"
    max="40%"
    --color="var(--color-gray-200)"
  >
    {#snippet a()}
      <div class="shrink-0 bg-gray-50 px-4 py-4">
        <h1 class="text-xl font-light mb-4">
          <a href="/"><span class="font-medium">jax-js</span> REPL</a>
        </h1>

        <hr class="mb-6 border-gray-200" />

        <p class="text-sm mb-4">
          Try out jax-js. Machine learning and numerical computing on the web!
        </p>
        <p class="text-sm mb-4">
          The goal is to <em>just use</em> NumPy and JAX in the browser, on WASM
          or WebGPU — with JIT and kernel fusion.
        </p>

        <pre class="mb-4 text-sm bg-gray-100 px-2 py-1 rounded"><code
            >npm i @jax-js/core</code
          ></pre>

        <h2 class="text-lg mt-8 mb-2">Examples</h2>
        <div class="text-sm flex flex-col">
          {#each codeSamples as { title, code }, i (i)}
            <button
              class="px-2 py-1 text-left rounded flex items-center hover:bg-gray-100 active:bg-gray-200 transition-colors"
              class:font-semibold={i === selected}
              onclick={() => {
                selected = i;
                replEditor.setText(code);
              }}
            >
              <span class="mr-2">
                <ArrowRightIcon size={16} />
              </span>
              {title}
            </button>
          {/each}
        </div>
      </div>
    {/snippet}
    {#snippet b()}
      <SplitPane
        type="vertical"
        pos="-240px"
        min="33%"
        max="-64px"
        --color="var(--color-gray-200)"
      >
        {#snippet a()}
          <div class="flex flex-col min-w-0">
            <div class="px-4 py-2 flex items-center gap-1">
              <button
                class="bg-emerald-100 hover:bg-emerald-200 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center"
                onclick={handleRun}
              >
                <PlayIcon size={14} class="mr-1.5" />
                Run
              </button>
              <button
                class="hover:bg-gray-100 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center"
                onclick={handleFormat}
              >
                <PaletteIcon size={14} class="mr-1.5" />
                Format
              </button>
            </div>
            <!-- <div class="ml-4 text-sm text-gray-700 pb-1 flex items-center">
              <FileIcon size={14} class="mr-1 text-sky-600" /> index.ts
            </div> -->
            <div class="flex-1 min-h-0">
              <ReplEditor
                initialText={codeSamples[selected].code}
                bind:this={replEditor}
                onformat={handleFormat}
                onrun={handleRun}
              />
            </div>
          </div>
        {/snippet}
        {#snippet b()}
          <section class="p-4 !overflow-y-auto">
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
            <div class="text-sm font-mono">hello</div>
          </section>
        {/snippet}
      </SplitPane>
    {/snippet}
  </SplitPane>
</div>

<style lang="postcss">
  @reference "$app.css";
</style>

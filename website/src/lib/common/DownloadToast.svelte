<script lang="ts">
  import { CheckCircle, Loader2, XCircle } from "lucide-svelte";
  import { fly } from "svelte/transition";

  import { formatBytes } from "$lib/chart/format";

  export interface Props {
    status: "downloading" | "success" | "error";
    loaded?: number;
    total?: number;
    errorMessage?: string;
  }

  let { status, loaded, total, errorMessage }: Props = $props();
</script>

<div
  class="fixed bottom-2 right-2 sm:bottom-4 sm:right-4 bg-white rounded-lg shadow-md sm:shadow-lg p-4 min-w-80 border border-gray-200 transition-all duration-300 z-20"
  transition:fly={{ y: 32, duration: 400, delay: 200 }}
>
  <div class="flex items-start gap-3">
    <div class="flex-shrink-0 mt-0.5">
      {#if status === "downloading"}
        <Loader2 size={16} class="stroke-[2.5] animate-spin text-blue-600" />
      {:else if status === "success"}
        <CheckCircle size={16} class="stroke-[2.5] text-green-600" />
      {:else}
        <XCircle size={16} class="stroke-[2.5] text-red-600" />
      {/if}
    </div>
    <div class="flex-1">
      {#if status === "downloading"}
        <div class="text-gray-700 font-medium text-sm mb-1">
          Downloading model weights…
        </div>
        {#if loaded !== undefined && total !== undefined}
          {@const percentage = Math.round((loaded / total) * 100)}
          <div class="flex items-center gap-2 mb-1.5">
            <div
              class="flex-1 bg-gradient-to-r from-gray-100 to-gray-200 rounded-full h-1.5 overflow-hidden"
            >
              <div
                class="bg-gradient-to-r from-blue-500 to-blue-600 h-1.5 transition-all duration-300 ease-out rounded-full"
                style:width="{percentage}%"
              ></div>
            </div>
            <span class="tabular-nums text-xs text-gray-600 whitespace-nowrap">
              {percentage}%
            </span>
          </div>
          <div class="text-xs text-gray-500 tabular-nums">
            {formatBytes(loaded)} / {formatBytes(total)}
          </div>
        {/if}
      {:else if status === "success"}
        <div class="text-green-700 font-medium text-sm mb-1">
          Model weights loaded!
        </div>
        <div class="text-xs text-gray-600 tabular-nums">
          {formatBytes(loaded!)}
        </div>
      {:else}
        <div class="text-red-700 font-medium text-sm mb-1">
          Failed to load model weights
        </div>
        <div class="text-xs text-gray-600">
          {errorMessage}
        </div>
      {/if}
    </div>
  </div>
</div>

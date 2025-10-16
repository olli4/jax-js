<script lang="ts">
  import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";

  import DownloadToast, {
    type Props as DownloadToastProps,
  } from "$lib/common/DownloadToast.svelte";

  const weightsUrl =
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/mobileclip2-s0.safetensors";
  let weights: safetensors.File | null = null;

  let isDownloading = $state(false);
  let downloadState = $state<DownloadToastProps | null>(null);

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (weights) return weights;
    isDownloading = true;
    try {
      // Show custom progress notification
      downloadState = {
        status: "downloading",
      };

      const data = await cachedFetch(weightsUrl, {}, (progress) => {
        downloadState = {
          status: "downloading",
          loaded: progress.loadedBytes,
          total: progress.totalBytes,
        };
      });

      const result = safetensors.parse(data);

      // Transition to success state
      downloadState = {
        status: "success",
        loaded: downloadState.loaded,
        total: downloadState.total,
      };
      // Hide after a delay
      setTimeout(() => {
        downloadState = null;
      }, 3000);

      return result;
    } catch (error) {
      // Show error notification
      downloadState = {
        status: "error",
        errorMessage:
          error instanceof Error ? error.message : "Download aborted",
      };
      // Hide after a delay
      setTimeout(() => {
        downloadState = null;
      }, 4000);
      throw error;
    } finally {
      isDownloading = false;
    }
  }

  async function main() {
    if (isDownloading) return;

    try {
      weights = await downloadClipWeights();
      console.log(weights);

      const tokenizer = await tokenizers.get("clip");
      console.log(tokenizer.encode("hello world"));
    } catch (error) {
      console.error("Error in main:", error);
    }
  }
</script>

{#if downloadState}
  <DownloadToast {...downloadState} />
{/if}

<button onclick={main} disabled={isDownloading}>Run Model</button>

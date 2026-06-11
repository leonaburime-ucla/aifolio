<template>
  <div class="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
    <main class="min-w-0 flex-1 py-10">
      <div class="mx-auto flex max-w-5xl flex-col gap-6 px-6">
        <p class="text-sm font-semibold uppercase tracking-widest text-zinc-500">
          Agentic Research
        </p>
        <AgenticResearchWorkspace @dataset-change="onDatasetChange" />
      </div>
    </main>

    <div class="sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden">
      <ChatSidebar mode="research" :dataset-id="activeDatasetId" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from "vue";
import AgenticResearchWorkspace from "~/features/agentic-research/components/AgenticResearchWorkspace.vue";
import ChatSidebar from "~/features/ai-chat/components/ChatSidebar.vue";

const activeDatasetId = ref<string | null>(null);

function onDatasetChange(id: string) {
  activeDatasetId.value = id;
}
</script>

<template>
  <div class="flex min-h-screen flex-row bg-zinc-50 text-zinc-900">
    <main class="min-w-0 flex-1 py-10">
      <div class="mx-auto flex max-w-5xl flex-col gap-8 px-6">
        <p class="text-sm font-semibold uppercase tracking-widest text-zinc-500">
          AI-driven Chart Dashboard
        </p>
        <ChartsWorkspace />
      </div>
    </main>

    <div class="sticky top-16 h-[calc(100vh-64px)] w-[360px] shrink-0 overflow-hidden">
      <ChatSidebar />
    </div>
  </div>
</template>

<script setup lang="ts">
import ChartsWorkspace from "~/features/recharts/components/ChartsWorkspace.vue";
import ChatSidebar from "~/features/ai-chat/components/ChatSidebar.vue";
</script>

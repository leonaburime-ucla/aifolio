<script setup lang="ts">
import { useChatSidebarOrchestrator } from "~/features/ai-chat/orchestrator";

const props = withDefaults(defineProps<{
  mode?: "direct" | "research";
  datasetId?: string | null;
}>(), {
  mode: "direct",
  datasetId: null,
});

const {
  messages,
  inputValue,
  isSending,
  modelOptions,
  selectedModelId,
  screenFeedback,
  messagesEl,
  submit,
  handleHistory,
} = useChatSidebarOrchestrator({
  baseUrl: "/api/ai",
  getMode: () => props.mode,
  getDatasetId: () => props.datasetId,
});
</script>

<template>
  <aside class="flex h-full flex-col border-l border-zinc-200 bg-white">
    <div class="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
      <p class="text-xs font-semibold uppercase tracking-wide text-zinc-500">AI Chat</p>
      <select
        v-if="modelOptions.length > 0"
        v-model="selectedModelId"
        aria-label="Select AI model"
        class="rounded border border-zinc-200 px-2 py-0.5 text-[11px] text-zinc-700"
      >
        <option v-for="m in modelOptions" :key="m.id" :value="m.id">{{ m.label }}</option>
      </select>
    </div>

    <div ref="messagesEl" class="flex-1 overflow-y-auto px-4 py-3">
      <div v-if="messages.length === 0" class="text-xs text-zinc-400">
        Ask a question to get started.
      </div>
      <div v-for="msg in messages" :key="msg.id" class="mb-3">
        <p
          :class="[
            'whitespace-pre-wrap rounded-lg px-3 py-2 text-sm',
            msg.role === 'user'
              ? 'ml-6 bg-zinc-900 text-white'
              : 'mr-6 bg-zinc-100 text-zinc-800',
          ]"
        >
          {{ msg.content }}
        </p>
      </div>
      <div v-if="isSending" class="mb-3 mr-6 flex items-center gap-2 rounded-lg bg-zinc-100 px-3 py-2 text-sm text-zinc-500">
        <svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <span>Thinking...</span>
      </div>
    </div>

    <div v-if="screenFeedback" class="border-t border-red-100 bg-red-50 px-4 py-2">
      <p class="text-xs text-red-700">{{ screenFeedback.message }}</p>
    </div>

    <form class="border-t border-zinc-200 px-4 py-3" @submit.prevent="submit">
      <div class="flex items-center gap-2">
        <div class="flex items-center gap-1">
          <button
            type="button"
            disabled
            class="flex h-7 w-7 items-center justify-center rounded-md border border-zinc-200 text-sm text-zinc-400"
            title="Disabled for now"
          >
            +
          </button>
          <span class="text-[10px] text-zinc-400">Disabled for now</span>
        </div>
        <input
          v-model="inputValue"
          type="text"
          placeholder="Ask anything"
          aria-label="Chat input"
          class="flex-1 rounded-md border border-zinc-300 px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-500 focus:outline-none"
          @keydown.up="handleHistory('up')"
          @keydown.down="handleHistory('down')"
        />
        <button
          type="submit"
          :disabled="!inputValue.trim() || isSending"
          class="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white disabled:bg-zinc-400"
        >
          Send
        </button>
      </div>
    </form>
  </aside>
</template>

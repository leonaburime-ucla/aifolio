<template>
  <div class="flex h-full flex-col">
    <div class="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
      <h2 class="text-sm font-semibold text-zinc-900">AIfolio Agent</h2>
      <select
        v-model="selectedModelId"
        :disabled="modelOptions.length === 0"
        class="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-700 shadow-sm focus:outline-none focus:ring-2 focus:ring-zinc-300 disabled:cursor-not-allowed disabled:bg-zinc-100"
        aria-label="Select AG-UI model"
      >
        <option v-if="modelOptions.length === 0" value="">
          {{ isModelsLoading ? 'Loading models...' : 'No models available' }}
        </option>
        <option v-for="m in modelOptions" :key="m.id" :value="m.id">{{ m.label }}</option>
      </select>
    </div>
    <CopilotKitProvider :self-managed-agents="{ 'agentic-research': agent }">
      <AgUiCopilotTools
        :active-tab="activeTab"
        :active-dataset-id="activeDatasetId"
        :selected-model-id="selectedModelId"
        @switch-tab="$emit('switchTab', $event)"
        @dataset-change="$emit('datasetChange', $event)"
      />
      <CopilotChat
        class="min-h-0 flex-1 overflow-y-auto"
        agent-id="agentic-research"
        :labels="labels"
      >
        <template #message-view="{ messages, isRunning }">
          <CopilotChatMessageView :messages="messages" :is-running="isRunning">
            <template #assistant-message="{ message, messages: allMessages, isRunning: running }">
              <CopilotChatAssistantMessage
                :message="{ ...message, content: displayAssistantContent(message.content, message.id) }"
                :messages="allMessages"
                :is-running="running"
              />
            </template>
          </CopilotChatMessageView>
        </template>
      </CopilotChat>
    </CopilotKitProvider>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import {
  CopilotChat,
  CopilotChatAssistantMessage,
  CopilotChatMessageView,
  CopilotKitProvider,
  HttpAgent,
} from "@copilotkit/vue/v2";
import {
  parseCopilotAssistantPayload,
  normalizeChartSpecInput,
  AG_UI_FALLBACK_MODELS,
  AG_UI_PREFERRED_MODEL_ID,
  resolveNextAgUiSelectedModelId,
} from "@aifolio/frontend-core/ag-ui";
import { useChartStore } from "~/composables/useChartStore";
import AgUiCopilotTools from "~/components/screens/AgUiCopilotTools.vue";

useHead({
  link: [{ rel: "stylesheet", href: "/copilotkit-vue.css" }],
});

const props = defineProps<{
  activeTab: string;
  activeDatasetId: string | null;
}>();

defineEmits<{
  switchTab: [tab: string];
  datasetChange: [id: string];
}>();

const chartStore = useChartStore();
const processedMessageIds = new Set<string>();

// --- Model selector state ---
const modelOptions = ref<{ id: string; label: string }[]>(AG_UI_FALLBACK_MODELS);
const selectedModelId = ref<string>(AG_UI_FALLBACK_MODELS[0]?.id ?? "");
const isModelsLoading = ref(false);

onMounted(async () => {
  isModelsLoading.value = true;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    const res = await fetch("/api/ai/llm/gemini-models", { signal: controller.signal });
    clearTimeout(timeoutId);
    if (res.ok) {
      const payload = await res.json();
      if (payload.status === "ok" && Array.isArray(payload.models)) {
        modelOptions.value = payload.models.map((m: any) => ({ id: m.id, label: m.label }));
        const resolved = resolveNextAgUiSelectedModelId({
          currentSelectedModelId: null,
          fetchedModels: modelOptions.value,
          apiCurrentModelId: payload.currentModel ?? null,
          preferredModelId: AG_UI_PREFERRED_MODEL_ID,
        });
        if (resolved) selectedModelId.value = resolved;
      }
    }
  } catch {
    // Keep fallback models
  } finally {
    isModelsLoading.value = false;
  }
});

const agent = new HttpAgent({
  agentId: "agentic-research",
  description: "AIfolio AG-UI backend",
  url: "/api/ai/agui",
});

const labels = {
  modalHeaderTitle: "AIfolio Agent",
  welcomeMessageText: "Ask AIfolio to analyze data, explain charts, or plan a workflow.",
  chatInputPlaceholder: "Ask AIfolio...",
} as any;

function displayAssistantContent(content: string | undefined, messageId?: string): string {
  if (!content) return "";

  try {
    const parsed = JSON.parse(content.trim());
    if (parsed && typeof parsed === "object" && typeof parsed.message === "string") {
      if (parsed.chartSpec && messageId && !processedMessageIds.has(messageId)) {
        processedMessageIds.add(messageId);
        const specs = normalizeChartSpecInput(parsed.chartSpec);
        if (specs) {
          const specArray = Array.isArray(specs) ? specs : [specs];
          specArray.forEach((spec) => chartStore.addChartSpec(spec));
        }
      }
      return parsed.message;
    }
  } catch {
    return content;
  }

  return content;
}
</script>

<style scoped>
:deep([class*="cpk:overflow-y-scroll"]) {
  padding: 0 1rem 6rem;
}
</style>

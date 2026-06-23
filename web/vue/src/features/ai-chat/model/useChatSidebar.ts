import { ref, nextTick, watch } from "vue";
import {
  sendChatMessageDirect,
  sendChatMessage,
} from "@aifolio/frontend-core/chat";
import type {
  ChatMessage,
  ChatModelOption,
  ChatHistoryMessage,
  ScreenFeedback,
} from "@aifolio/contracts/entities/chat";
import { createChatApi } from "../api";
import type { ChatApi } from "../api";

export type UseChatSidebarOptions = {
  baseUrl: string;
  getMode: () => "direct" | "research";
  getDatasetId: () => string | null;
  onChartSpec?: (spec: unknown) => void;
  api?: ChatApi;
};

export function useChatSidebar(options: UseChatSidebarOptions) {
  const api = options.api ?? createChatApi({ baseUrl: options.baseUrl });

  const messages = ref<ChatMessage[]>([]);
  const inputValue = ref("");
  const isSending = ref(false);
  const modelOptions = ref<ChatModelOption[]>([]);
  const selectedModelId = ref<string | null>(null);
  const inputHistory = ref<string[]>([]);
  const historyCursor = ref<number | null>(null);
  const screenFeedback = ref<ScreenFeedback | null>(null);
  const messagesEl = ref<HTMLElement | null>(null);

  function generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  }

  watch(messages, () => {
    nextTick(() => {
      if (messagesEl.value) {
        messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
      }
    });
  }, { deep: true });

  async function loadModels() {
    try {
      const { models, currentModel } = await api.fetchModels();
      modelOptions.value = models;
      selectedModelId.value = currentModel;
    } catch {
      // non-critical
    }
  }

  async function submit() {
    const text = inputValue.value.trim();
    if (!text || isSending.value) return;

    screenFeedback.value = null;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };
    messages.value.push(userMsg);
    inputHistory.value.push(text);
    historyCursor.value = null;
    inputValue.value = "";
    isSending.value = true;

    const history: ChatHistoryMessage[] = messages.value
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));

    try {
      const sender = options.getMode() === "research" ? sendChatMessage : sendChatMessageDirect;
      const datasetId = options.getDatasetId();
      const result = await sender(
        {
          value: text,
          model: selectedModelId.value,
          history,
          attachments: undefined,
        },
        {
          runtimeDeps: { resolveBaseUrl: () => options.baseUrl },
          ...(datasetId ? { datasetId } : {}),
        },
      );

      if (result) {
        messages.value.push({
          id: generateId(),
          role: "assistant",
          content: result.message,
          createdAt: Date.now(),
          chartSpec: result.chartSpec as any,
        });
        if (result.chartSpec) {
          const specs = Array.isArray(result.chartSpec) ? result.chartSpec : [result.chartSpec];
          specs.forEach((spec: any) => {
            if (spec && spec.data) {
              options.onChartSpec?.(spec);
            }
          });
        }
      } else {
        screenFeedback.value = {
          kind: "error",
          code: "CHAT_EMPTY_RESPONSE",
          message: "No response from the backend.",
        };
      }
    } catch (err) {
      screenFeedback.value = {
        kind: "error",
        code: "CHAT_REQUEST_FAILED",
        message: err instanceof Error ? err.message : "Request failed.",
        retryable: true,
      };
    } finally {
      isSending.value = false;
    }
  }

  function handleHistory(direction: "up" | "down") {
    if (inputHistory.value.length === 0) return;
    if (direction === "up") {
      const next =
        historyCursor.value === null
          ? inputHistory.value.length - 1
          : Math.max(0, historyCursor.value - 1);
      historyCursor.value = next;
      inputValue.value = inputHistory.value[next];
    } else {
      if (historyCursor.value === null) return;
      const next = historyCursor.value + 1;
      if (next >= inputHistory.value.length) {
        historyCursor.value = null;
        inputValue.value = "";
      } else {
        historyCursor.value = next;
        inputValue.value = inputHistory.value[next];
      }
    }
  }

  return {
    messages,
    inputValue,
    isSending,
    modelOptions,
    selectedModelId,
    inputHistory,
    historyCursor,
    screenFeedback,
    messagesEl,
    loadModels,
    submit,
    handleHistory,
    generateId,
  };
}

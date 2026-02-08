/**
 * Domain type for chat messages.
 */
import type { ChartSpec } from "@/features/ai/types/chart.types";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  chartSpec?: ChartSpec | null;
};

export type ChatModelOption = {
  id: string;
  label: string;
};

export type ChatAssistantPayload = {
  message: string;
  chartSpec: ChartSpec | ChartSpec[] | null;
};

export type ChatHistoryMessage = {
  role: "user" | "assistant";
  content: string;
  attachments?: ChatAttachment[];
};

export type ChatAttachment = {
  name: string;
  type: string;
  size: number;
  dataUrl: string;
};


/**
 * Direction for navigating input history.
 */
export type ChatHistoryDirection = "up" | "down";

/**
 * Reactive slice of chat state managed by the store.
 * Consumed by UI and orchestration layers.
 */
export type ChatState = {
  messages: ChatMessage[];
  inputHistory: string[];
  historyCursor: number | null;
  isSending: boolean;
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  activeDatasetId?: string | null;
};

/**
 * Actions that mutate chat state.
 * These are injected from the store to keep hooks decoupled.
 */
export type ChatStateActions = {
  addMessage: (message: ChatMessage) => void;
  addInputToHistory: (value: string) => void;
  moveHistoryCursor: (direction: ChatHistoryDirection) => string;
  resetHistoryCursor: () => void;
  setSending: (value: boolean) => void;
  setModelOptions: (value: ChatModelOption[]) => void;
  setSelectedModelId: (value: string | null) => void;
  setModelsLoading: (value: boolean) => void;
  addChartSpec: (spec: ChartSpec) => void;
  onMessageReceived: (payload: ChatAssistantPayload) => void;
};

/**
 * API dependencies injected into chat logic.
 */
export type ChatApiDeps = {
  /**
   * Sends a user message to the AI backend and returns assistant content.
   */
  sendMessage: (
    value: string,
    model: string | null,
    history: ChatHistoryMessage[],
    attachments: ChatHistoryMessage["attachments"]
  ) => Promise<ChatAssistantPayload | null>;
  fetchModels: () => Promise<{
    currentModel: string | null;
    models: ChatModelOption[];
  } | null>;
};

/**
 * Local UI state for the chat input.
 */
export type ChatUiState = {
  value: string;
  showTooltip: boolean;
  attachments: ChatAttachment[];
  setShowTooltip: (value: boolean) => void;
  setValue: (value: string) => void;
  resetValue: () => void;
  addAttachments: (files: ChatAttachment[]) => void;
  clearAttachments: () => void;
  removeAttachment: (index: number) => void;
};

/**
 * Chat actions that coordinate UI state with store mutations and API calls.
 */
export type ChatActions = {
  submit: () => Promise<void>;
  handleHistory: (direction: ChatHistoryDirection) => void;
  resetHistoryCursor: () => void;
  setSelectedModelId: (value: string | null) => void;
  /**
   * Manually triggers a model fetch.
   */
  refetchModels: () => Promise<void>;
};

/**
 * Combined interface exposed to UI components.
 */
export type ChatIntegration = ChatUiState & ChatState & ChatActions;

/**
 * Dependencies injected into chat hooks.
 */
export type ChatDeps = {
  state: ChatState;
  actions: ChatStateActions;
  api: ChatApiDeps;
};

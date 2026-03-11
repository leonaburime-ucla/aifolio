/**
 * Domain type for chat messages.
 */
import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";
import type {
  BuildChatHistoryWindowInput,
  BuildChatHistoryWindowOptions,
  CreateChatMessageInput,
  NormalizeSubmissionInput,
  ShouldRestoreDraftValueInput,
} from "@/features/ai-chat/__types__/typescript/logic/chatSubmission.types";
import type {
  FallbackSelectionInput,
  FallbackSelectionOptions,
  FetchedSelectionInput,
  ModelSelectionResult,
} from "@/features/ai-chat/__types__/typescript/logic/modelSelection.types";
import type {
  FetchChatModelsInput,
  FetchChatModelsOptions,
  FetchChatModelsResult,
  SendChatMessageInput,
  SendChatMessageOptions,
} from "@/features/ai-chat/__types__/typescript/api.types";

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

export type ChatCoreStateActions = Omit<
  ChatStateActions,
  "addChartSpec" | "onMessageReceived"
>;

export type ChatStatePort = {
  state: Omit<ChatState, "activeDatasetId">;
  actions: ChatCoreStateActions;
};

export type UseChatStatePort = () => ChatStatePort;

export type ChatChartActionsPort = {
  addChartSpec: (spec: ChartSpec) => void;
};

export type UseChatChartActionsPort = () => ChatChartActionsPort;

/**
 * API dependencies injected into chat logic.
 */
export type ChatApiDeps = {
  /**
   * Sends a user message to the AI backend and returns assistant content.
   */
  sendMessage: (
    input: SendChatMessageInput,
    options?: SendChatMessageOptions
  ) => Promise<ChatAssistantPayload | null>;
  fetchModels: (
    input: FetchChatModelsInput,
    options?: FetchChatModelsOptions
  ) => Promise<FetchChatModelsResult | null>;
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
 * Logic functions injected into chat hooks.
 * These come from `chatSubmission.logic` and `modelSelection.logic`.
 */
export type ChatLogicDeps = {
  normalizeSubmissionValue: (input: NormalizeSubmissionInput) => string | null;
  buildChatHistoryWindow: (
    input: BuildChatHistoryWindowInput,
    options?: BuildChatHistoryWindowOptions
  ) => ChatHistoryMessage[];
  createUserChatMessage: (input: CreateChatMessageInput) => ChatMessage;
  createAssistantChatMessage: (input: CreateChatMessageInput) => ChatMessage;
  shouldRestoreDraftValue: (input: ShouldRestoreDraftValueInput) => boolean;
  resolveFallbackModelSelection: (
    input: FallbackSelectionInput,
    options?: FallbackSelectionOptions
  ) => ModelSelectionResult;
  resolveFetchedModelSelection: (input: FetchedSelectionInput) => ModelSelectionResult;
};

/**
 * Dependencies injected into chat hooks.
 */
export type ChatDeps = {
  state: ChatState;
  actions: ChatStateActions;
  api: ChatApiDeps;
  logic: ChatLogicDeps;
};

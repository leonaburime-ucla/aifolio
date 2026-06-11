import type { ChartSpec } from "../../chart/index";
import type {
  SendChatMessageInput,
  SendChatMessageOptions,
  FetchChatModelsInput,
  FetchChatModelsOptions,
  FetchChatModelsResult,
} from "../api/types";

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

export type ChatAttachment = {
  name: string;
  type: string;
  size: number;
  dataUrl: string;
};

export type ChatHistoryMessage = {
  role: "user" | "assistant";
  content: string;
  attachments?: ChatAttachment[];
};

export type ChatHistoryDirection = "up" | "down";

export type ScreenFeedback = {
  kind: "error" | "warning" | "info";
  code: string;
  message: string;
  retryable?: boolean;
  actionLabel?: string;
};

export type ChatState = {
  messages: ChatMessage[];
  inputHistory: string[];
  historyCursor: number | null;
  isSending: boolean;
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  screenFeedback: ScreenFeedback | null;
  activeDatasetId?: string | null;
};

export type ChatStateActions = {
  addMessage: (message: ChatMessage) => void;
  addInputToHistory: (value: string) => void;
  moveHistoryCursor: (direction: ChatHistoryDirection) => string;
  resetHistoryCursor: () => void;
  setSending: (value: boolean) => void;
  setModelOptions: (value: ChatModelOption[]) => void;
  setSelectedModelId: (value: string | null) => void;
  setModelsLoading: (value: boolean) => void;
  setScreenFeedback: (value: ScreenFeedback | null) => void;
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

export type ChatChartActionsPort = {
  addChartSpec: (spec: ChartSpec) => void;
};

export type UseChatStatePort = () => ChatStatePort;

export type UseChatChartActionsPort = () => ChatChartActionsPort;

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

export type ChatActions = {
  submit: () => Promise<void>;
  retryLastSubmission: () => Promise<void>;
  handleHistory: (direction: ChatHistoryDirection) => void;
  resetHistoryCursor: () => void;
  setSelectedModelId: (value: string | null) => void;
  setScreenFeedback: (value: ScreenFeedback | null) => void;
  refetchModels: () => Promise<void>;
};

export type ChatIntegration = ChatUiState & ChatState & ChatActions;

export type ModelSelectionResult = {
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
};

export type FallbackSelectionInput = {
  selectedModelId: string | null;
};

export type FallbackSelectionOptions = {
  fallbackModels?: ChatModelOption[];
};

export type FetchedModelsResult = {
  currentModel: string | null;
  models: ChatModelOption[];
};

export type FetchedSelectionInput = {
  selectedModelId: string | null;
  result: FetchedModelsResult;
};

export type NormalizeSubmissionInput = {
  value: string;
};

export type BuildChatHistoryWindowInput = {
  messages: ChatMessage[];
  userContent: string;
  attachments: ChatAttachment[] | undefined;
};

export type BuildChatHistoryWindowOptions = {
  windowSize?: number;
};

export type CreateChatMessageInput = {
  id: string;
  content: string;
  createdAt: number;
};

export type ShouldRestoreDraftValueInput = {
  direction: ChatHistoryDirection;
  historyCursor: number | null;
  nextValue: string;
};

export type ChatStoreCoreState = {
  messages: ChatMessage[];
  inputHistory: string[];
  historyCursor: number | null;
  isSending: boolean;
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  screenFeedback: ScreenFeedback | null;
};

export type HistoryCursorResult = {
  nextCursor: number | null;
  value: string;
};

export type AppendMessageInput = {
  messages: ChatMessage[];
  message: ChatMessage;
};

export type AppendInputHistoryInput = {
  inputHistory: string[];
  value: string;
};

export type ResolveHistoryCursorInput = {
  inputHistory: string[];
  historyCursor: number | null;
  direction: ChatHistoryDirection;
};

export type MapChatStateWithDatasetInput = {
  state: Omit<ChatState, "activeDatasetId">;
  activeDatasetId: string | null;
};

export type CreateOnMessageReceivedInput = {
  addChartSpec: (spec: ChartSpec) => void;
};

export type ComposeChatStateActionsInput = {
  coreActions: ChatCoreStateActions;
  addChartSpec: ChatStateActions["addChartSpec"];
};

export type ChatApiDeps = {
  sendMessage: (
    input: SendChatMessageInput,
    options?: SendChatMessageOptions
  ) => Promise<ChatAssistantPayload | null>;
  fetchModels: (
    input: FetchChatModelsInput,
    options?: FetchChatModelsOptions
  ) => Promise<FetchChatModelsResult | null>;
};

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

export type ChatDeps = {
  state: ChatState;
  actions: ChatStateActions;
  api: ChatApiDeps;
  logic: ChatLogicDeps;
};

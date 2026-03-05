import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";
import type { ChatHistoryMessage } from "@/features/ai-chat/__types__/typescript/chat.types";

export type ChatApiResponse = {
  status: "ok" | "error";
  result?:
  | string
  | Array<{ type: string; text?: string }>
  | { message?: string; chartSpec?: ChartSpec | ChartSpec[] | null };
  error?: string;
  model?: string;
};

export type ModelsApiResponse = {
  status: "ok" | "error";
  currentModel?: string;
  models?: Array<{ id: string; label: string }>;
  error?: string;
};

export type SendChatMessageInput = {
  value: string;
  model: string | null;
  history: ChatHistoryMessage[];
  attachments: ChatHistoryMessage["attachments"];
};

export type SendChatMessageOptions = {
  datasetId?: string | null;
  runtimeDeps?: ChatApiRuntimeDeps;
};

export type SendChatMessageToEndpointInput = SendChatMessageInput & {
  endpoint: "/chat" | "/chat-research";
};

export type SendChatMessageToEndpointOptions = {
  datasetId?: string | null;
  runtimeDeps?: ChatApiRuntimeDeps;
};

export type FetchChatModelsInput = Record<string, never>;

export type FetchChatModelsOptions = {
  timeoutMs?: number;
  runtimeDeps?: ChatApiRuntimeDeps;
};

export type ChatApiRuntimeDeps = {
  fetchImpl?: typeof fetch;
  resolveBaseUrl?: () => string;
  createAbortController?: () => AbortController;
  setTimeoutImpl?: typeof setTimeout;
  clearTimeoutImpl?: typeof clearTimeout;
};

export type ChatApiError = {
  code: "MODEL_FETCH_TIMEOUT" | "MODEL_FETCH_FAILED";
  retryable: boolean;
  message: string;
};

export type FetchChatModelsSuccessResult = {
  status: "ok";
  currentModel: string | null;
  models: Array<{ id: string; label: string }>;
};

export type FetchChatModelsErrorResult = {
  status: "error";
  error: ChatApiError;
};

export type FetchChatModelsResult =
  | FetchChatModelsSuccessResult
  | FetchChatModelsErrorResult;

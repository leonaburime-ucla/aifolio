import type {
  ChatHistoryDirection,
  ChatMessage,
  ChatModelOption,
} from "@/features/ai-chat/__types__/typescript/chat.types";

export type ChatStoreCoreState = {
  messages: ChatMessage[];
  inputHistory: string[];
  historyCursor: number | null;
  isSending: boolean;
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
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

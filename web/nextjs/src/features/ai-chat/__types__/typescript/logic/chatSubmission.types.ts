import type {
  ChatAttachment,
  ChatHistoryDirection,
  ChatMessage,
} from "@/features/ai-chat/__types__/typescript/chat.types";

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

import type {
  ChatApiDeps,
  ChatState,
  ChatStateActions,
} from "@/features/ai-chat/__types__/typescript/chat.types";

export type CreateChatApiDepsInput = {
  sendMessage: ChatApiDeps["sendMessage"];
  fetchModels: ChatApiDeps["fetchModels"];
};

export type CreateChatDepsInput = {
  state: ChatState;
  actions: ChatStateActions;
  api: ChatApiDeps;
};

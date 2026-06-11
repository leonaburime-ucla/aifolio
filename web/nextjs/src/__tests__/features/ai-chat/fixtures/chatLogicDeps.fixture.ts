import {
  normalizeSubmissionValue,
  buildChatHistoryWindow,
  createUserChatMessage,
  createAssistantChatMessage,
  shouldRestoreDraftValue,
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
} from "@aifolio/frontend-core/chat";
import type { ChatLogicDeps } from "@aifolio/contracts/entities/chat";

export const DEFAULT_CHAT_LOGIC_DEPS: ChatLogicDeps = {
  normalizeSubmissionValue,
  buildChatHistoryWindow,
  createUserChatMessage,
  createAssistantChatMessage,
  shouldRestoreDraftValue,
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
};

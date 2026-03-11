/**
 * Shared test fixture that provides concrete logic deps for ChatDeps construction.
 *
 * Import and spread into any test that constructs inline ChatDeps objects.
 */
import {
  normalizeSubmissionValue,
  buildChatHistoryWindow,
  createUserChatMessage,
  createAssistantChatMessage,
  shouldRestoreDraftValue,
} from "@/features/ai-chat/typescript/logic/chatSubmission.logic";
import {
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
} from "@/features/ai-chat/typescript/logic/modelSelection.logic";
import type { ChatLogicDeps } from "@/features/ai-chat/__types__/typescript/chat.types";

/**
 * Default logic deps using real implementations.
 * Suitable for integration-style tests that verify real behavior.
 */
export const DEFAULT_CHAT_LOGIC_DEPS: ChatLogicDeps = {
  normalizeSubmissionValue,
  buildChatHistoryWindow,
  createUserChatMessage,
  createAssistantChatMessage,
  shouldRestoreDraftValue,
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
};

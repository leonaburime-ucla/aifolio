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
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
} from "@aifolio/frontend-core/chat";
import type { ChatLogicDeps } from "@aifolio/contracts/entities/chat";

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

import type {
  ChatHistoryMessage,
  ChatMessage,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import type {
  BuildChatHistoryWindowInput,
  BuildChatHistoryWindowOptions,
  CreateChatMessageInput,
  NormalizeSubmissionInput,
  ShouldRestoreDraftValueInput,
} from "@/features/ai-chat/__types__/typescript/logic/chatSubmission.types";

const DEFAULT_HISTORY_WINDOW_SIZE = 10;

/**
 * Trim user input and return null for empty submissions.
 *
 * @param input - Required submission normalization input.
 * @returns Trimmed text or null when empty.
 */
export function normalizeSubmissionValue(
  input: NormalizeSubmissionInput
): string | null {
  const trimmed = input.value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

/**
 * Build bounded chat history payload for API requests.
 *
 * @param input - Required history-window inputs.
 * @param options - Optional history-window configuration.
 * @returns Bounded history payload ending with the latest user message.
 */
export function buildChatHistoryWindow(
  input: BuildChatHistoryWindowInput,
  options?: BuildChatHistoryWindowOptions
): ChatHistoryMessage[] {
  const windowSize = options?.windowSize ?? DEFAULT_HISTORY_WINDOW_SIZE;
  const currentUserMessage: ChatHistoryMessage = {
    role: "user",
    content: input.userContent,
    attachments: input.attachments,
  };

  return [
    ...input.messages.map((message) => ({
      role: message.role,
      content: message.content,
    })),
    currentUserMessage,
  ].slice(-windowSize);
}

/**
 * Create a timestamped user chat message.
 *
 * @param input - Required user message inputs.
 * @returns User chat message.
 */
export function createUserChatMessage(
  input: CreateChatMessageInput
): ChatMessage {
  return {
    id: input.id,
    role: "user",
    content: input.content,
    createdAt: input.createdAt,
  };
}

/**
 * Create a timestamped assistant chat message.
 *
 * @param input - Required assistant message inputs.
 * @returns Assistant chat message.
 */
export function createAssistantChatMessage(
  input: CreateChatMessageInput
): ChatMessage {
  return {
    id: input.id,
    role: "assistant",
    content: input.content,
    chartSpec: null,
    createdAt: input.createdAt,
  };
}

/**
 * Determine whether history navigation should restore the draft input.
 *
 * @param input - Required draft-restore decision inputs.
 * @returns True when draft input should be restored.
 */
export function shouldRestoreDraftValue(
  input: ShouldRestoreDraftValueInput
): boolean {
  return (
    input.direction === "down" &&
    input.historyCursor !== null &&
    input.nextValue === ""
  );
}

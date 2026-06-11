import type {
  ChatHistoryMessage,
  ChatMessage,
  BuildChatHistoryWindowInput,
  BuildChatHistoryWindowOptions,
  CreateChatMessageInput,
  NormalizeSubmissionInput,
  ShouldRestoreDraftValueInput,
} from "@aifolio/contracts/entities/chat";

const DEFAULT_HISTORY_WINDOW_SIZE = 10;

/**
 * Trim user input and return null for empty submissions.
 *
 * @param input - Required: `value` (raw user input string).
 * @returns Trimmed string, or null when input is whitespace-only.
 * @throws Never.
 * @complexity O(n) — string length
 * @overallScore 100
 */
export function normalizeSubmissionValue(
  input: NormalizeSubmissionInput
): string | null {
  const trimmed = input.value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

/**
 * Build a bounded chat history window for API payloads.
 * Maps existing messages + current user message, then slices to windowSize.
 *
 * @param input - Required: `messages` (existing transcript), `userContent`, `attachments`.
 * @param options - Optional: `windowSize` (default 10).
 * @returns Bounded `ChatHistoryMessage[]` ending with the current user message.
 * @throws Never.
 * @complexity O(n) — messages array length
 * @overallScore 100
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
 * @param input - Required: `id`, `content`, `createdAt`.
 * @returns `ChatMessage` with role "user".
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
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
 * Create a timestamped assistant chat message with null chartSpec.
 *
 * @param input - Required: `id`, `content`, `createdAt`.
 * @returns `ChatMessage` with role "assistant" and chartSpec null.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
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
 * True only when navigating "down" past the end of history with an empty value.
 *
 * @param input - Required: `direction`, `historyCursor`, `nextValue`.
 * @returns True when draft input should be restored.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
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

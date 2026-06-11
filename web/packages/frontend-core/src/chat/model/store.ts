import type {
  ChatMessage,
  ChatStoreCoreState,
  HistoryCursorResult,
  AppendMessageInput,
  AppendInputHistoryInput,
  ResolveHistoryCursorInput,
} from "@aifolio/contracts/entities/chat";

/**
 * Create the default initial state for any chat store implementation.
 * Every field has a defined value (no undefined) per INV-06.
 *
 * @param _input - Empty object for signature consistency.
 * @returns Fresh `ChatStoreCoreState` with all fields initialized.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
 */
export function createInitialChatStoreCoreState(
  _input: Record<string, never>
): ChatStoreCoreState {
  return {
    messages: [],
    inputHistory: [],
    historyCursor: null,
    isSending: false,
    modelOptions: [],
    selectedModelId: null,
    isModelsLoading: false,
    screenFeedback: null,
  };
}

/**
 * Immutably append a message to a transcript array.
 *
 * @param input - Required: `messages` (existing array), `message` (to append).
 * @returns New array with the message appended.
 * @throws Never.
 * @complexity O(n) — array copy
 * @overallScore 100
 */
export function appendMessage(input: AppendMessageInput): ChatMessage[] {
  return [...input.messages, input.message];
}

/**
 * Immutably append a user input to history and reset the cursor.
 *
 * @param input - Required: `inputHistory` (existing array), `value` (to append).
 * @returns New history array and null cursor.
 * @throws Never.
 * @complexity O(n) — array copy
 * @overallScore 100
 */
export function appendInputHistory(
  input: AppendInputHistoryInput
): Pick<ChatStoreCoreState, "inputHistory" | "historyCursor"> {
  return {
    inputHistory: [...input.inputHistory, input.value],
    historyCursor: null,
  };
}

/**
 * Resolve the next history cursor position and the value at that position.
 * Bounds-safe: clamps cursor to valid range, returns null cursor when navigating past ends.
 *
 * @param input - Required: `inputHistory`, `historyCursor` (current or null), `direction`.
 * @returns `HistoryCursorResult` with `nextCursor` and `value`.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
 */
export function resolveHistoryCursor(
  input: ResolveHistoryCursorInput
): HistoryCursorResult {
  const { inputHistory, historyCursor, direction } = input;
  if (inputHistory.length === 0) {
    return { nextCursor: historyCursor, value: "" };
  }

  const normalizedCursor =
    historyCursor === null
      ? null
      : Math.min(Math.max(historyCursor, 0), inputHistory.length - 1);

  if (direction === "up") {
    const nextCursor =
      normalizedCursor === null
        ? inputHistory.length - 1
        : Math.max(0, normalizedCursor - 1);
    return {
      nextCursor,
      value: inputHistory[nextCursor] ?? "",
    };
  }

  if (normalizedCursor === null) {
    return { nextCursor: null, value: "" };
  }

  const nextCursor = normalizedCursor + 1;
  if (nextCursor >= inputHistory.length) {
    return { nextCursor: null, value: "" };
  }

  return {
    nextCursor,
    value: inputHistory[nextCursor] ?? "",
  };
}

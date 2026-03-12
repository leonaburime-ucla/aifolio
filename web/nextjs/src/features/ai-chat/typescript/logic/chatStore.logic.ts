import type { ChatMessage } from "@/features/ai-chat/__types__/typescript/chat.types";
import type {
  AppendInputHistoryInput,
  AppendMessageInput,
  ChatStoreCoreState,
  HistoryCursorResult,
  ResolveHistoryCursorInput,
} from "@/features/ai-chat/__types__/typescript/logic/chatStore.types";

/**
 * Create the default chat state shared by chat stores.
 *
 * @param _input - Required input object for signature consistency.
 * @returns Fresh default chat store state.
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
 * Append a message to a transcript.
 *
 * @param input - Required message append inputs.
 * @returns New ordered transcript with appended message.
 */
export function appendMessage(
  input: AppendMessageInput
): ChatMessage[] {
  return [...input.messages, input.message];
}

/**
 * Add a user input to history and reset cursor.
 *
 * @param input - Required input history append inputs.
 * @returns Updated history and reset cursor.
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
 * Resolve next history cursor position and selected value.
 *
 * @param input - Required history cursor inputs.
 * @returns Next cursor and the resolved input value.
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

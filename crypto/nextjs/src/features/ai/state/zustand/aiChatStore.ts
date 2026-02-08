import { create } from "zustand";
import type { ChatMessage, ChatModelOption } from "@/features/ai/types/chat.types";

/**
 * Zustand state and actions for AI chat.
 */
type AiChatState = {
  /**
   * Ordered chat transcript, including user and assistant messages.
   */
  messages: ChatMessage[];
  /**
   * List of previous user inputs used for history navigation.
   */
  inputHistory: string[];
  /**
   * Pointer into inputHistory for arrow-key navigation. Null means "no selection".
   */
  historyCursor: number | null;
  /**
   * True while a chat request is in flight.
   */
  isSending: boolean;
  /**
   * Model options returned by the backend.
   */
  modelOptions: ChatModelOption[];
  /**
   * Currently selected model id for chat requests.
   */
  selectedModelId: string | null;
  /**
   * True while model options are being fetched.
   */
  isModelsLoading: boolean;
  /**
   * Append a new chat message to the transcript.
   */
  addMessage: (message: ChatMessage) => void;
  /**
   * Store a submitted input and reset history navigation.
   */
  addInputToHistory: (value: string) => void;
  /**
   * Move the history cursor up or down and return the selected input.
   */
  moveHistoryCursor: (direction: "up" | "down") => string;
  /**
   * Clear history navigation selection.
   */
  resetHistoryCursor: () => void;
  /**
   * Toggle request-in-flight state.
   */
  setSending: (value: boolean) => void;
  /**
   * Replace available model options.
   */
  setModelOptions: (value: ChatModelOption[]) => void;
  /**
   * Update the active model id.
   */
  setSelectedModelId: (value: string | null) => void;
  /**
   * Toggle model options loading state.
   */
  setModelsLoading: (value: boolean) => void;
};

/**
 * Global AI chat store.
 */
export const useAiChatStore = create<AiChatState>((set, get) => ({
  messages: [],
  inputHistory: [],
  historyCursor: null,
  isSending: false,
  modelOptions: [],
  selectedModelId: null,
  isModelsLoading: false,
  /**
   * Append a new message to the chat transcript.
   */
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  /**
   * Persist the latest user input and reset history selection.
   */
  addInputToHistory: (value) =>
    set((state) => ({
      inputHistory: [...state.inputHistory, value],
      historyCursor: null,
    })),
  /**
   * Navigate through prior inputs using up/down arrows.
   */
  moveHistoryCursor: (direction) => {
    const { inputHistory, historyCursor } = get();
    // Nothing to navigate.
    if (inputHistory.length === 0) return "";

    // Move toward older entries.
    if (direction === "up") {
      const nextIndex =
        historyCursor === null
          ? inputHistory.length - 1
          : Math.max(0, historyCursor - 1);
      set({ historyCursor: nextIndex });
      return inputHistory[nextIndex] ?? "";
    }

    // Move toward newer entries or clear selection.
    if (historyCursor === null) return "";
    const nextIndex = historyCursor + 1;
    if (nextIndex >= inputHistory.length) {
      set({ historyCursor: null });
      return "";
    }
    set({ historyCursor: nextIndex });
    return inputHistory[nextIndex] ?? "";
  },
  /**
   * Reset the cursor so no history value is selected.
   */
  resetHistoryCursor: () => set({ historyCursor: null }),
  /**
   * Mark the chat request as in-flight or complete.
   */
  setSending: (value) => set({ isSending: value }),
  /**
   * Store the available LLM model options.
   */
  setModelOptions: (value) => set({ modelOptions: value }),
  /**
   * Choose which model id to send with requests.
   */
  setSelectedModelId: (value) => set({ selectedModelId: value }),
  /**
   * Mark whether model options are still loading.
   */
  setModelsLoading: (value) => set({ isModelsLoading: value }),
}));

export type { AiChatState };

import { create } from "zustand";
import type { ChatMessage, ChatModelOption } from "@/features/ai-chat/__types__/typescript/chat.types";
import type { ScreenFeedback } from "@/features/ai-chat/__types__/typescript/uiFeedback.types";
import {
  appendInputHistory,
  appendMessage,
  createInitialChatStoreCoreState,
  resolveHistoryCursor,
} from "@/features/ai-chat/typescript/logic/chatStore.logic";

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
   * Persistent inline feedback for chat failures or degraded states.
   */
  screenFeedback: ScreenFeedback | null;
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
  /**
   * Persist or clear inline feedback for the chat surface.
   */
  setScreenFeedback: (value: ScreenFeedback | null) => void;
};

/**
 * Global AI chat store.
 */
export const useAiChatStore = create<AiChatState>((set, get) => ({
  ...createInitialChatStoreCoreState({}),
  /**
   * Append a new message to the chat transcript.
   */
  addMessage: (message) =>
    set((state) => ({
      messages: appendMessage({ messages: state.messages, message }),
    })),
  /**
   * Persist the latest user input and reset history selection.
   */
  addInputToHistory: (value) =>
    set((state) =>
      appendInputHistory({ inputHistory: state.inputHistory, value })
    ),
  /**
   * Navigate through prior inputs using up/down arrows.
   */
  moveHistoryCursor: (direction) => {
    const { inputHistory, historyCursor } = get();
    const next = resolveHistoryCursor({
      inputHistory,
      historyCursor,
      direction,
    });
    set({ historyCursor: next.nextCursor });
    return next.value;
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
  /**
   * Persist or clear inline chat feedback.
   */
  setScreenFeedback: (value) => set({ screenFeedback: value }),
}));

export type { AiChatState };

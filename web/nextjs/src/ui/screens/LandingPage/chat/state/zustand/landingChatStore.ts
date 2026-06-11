import { create } from "zustand";
import type { ChatMessage, ChatModelOption, ScreenFeedback } from "@aifolio/contracts/entities/chat";
import {
  appendInputHistory,
  appendMessage,
  createInitialChatStoreCoreState,
  resolveHistoryCursor,
} from "@aifolio/frontend-core/chat";

type LandingChatState = {
  messages: ChatMessage[];
  inputHistory: string[];
  historyCursor: number | null;
  isSending: boolean;
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  screenFeedback: ScreenFeedback | null;
  addMessage: (message: ChatMessage) => void;
  addInputToHistory: (value: string) => void;
  moveHistoryCursor: (direction: "up" | "down") => string;
  resetHistoryCursor: () => void;
  setSending: (value: boolean) => void;
  setModelOptions: (value: ChatModelOption[]) => void;
  setSelectedModelId: (value: string | null) => void;
  setModelsLoading: (value: boolean) => void;
  setScreenFeedback: (value: ScreenFeedback | null) => void;
};

export const useLandingChatStore = create<LandingChatState>((set, get) => ({
  ...createInitialChatStoreCoreState({}),
  addMessage: (message) =>
    set((state) => ({
      messages: appendMessage({ messages: state.messages, message }),
    })),
  addInputToHistory: (value) =>
    set((state) =>
      appendInputHistory({ inputHistory: state.inputHistory, value })
    ),
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
  resetHistoryCursor: () => set({ historyCursor: null }),
  setSending: (value) => set({ isSending: value }),
  setModelOptions: (value) => set({ modelOptions: value }),
  setSelectedModelId: (value) => set({ selectedModelId: value }),
  setModelsLoading: (value) => set({ isModelsLoading: value }),
  setScreenFeedback: (value) => set({ screenFeedback: value }),
}));

export type { LandingChatState };

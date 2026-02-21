import { create } from "zustand";
import type { ChatMessage, ChatModelOption } from "@/features/ai/types/chat.types";

type LandingChatState = {
  messages: ChatMessage[];
  inputHistory: string[];
  historyCursor: number | null;
  isSending: boolean;
  modelOptions: ChatModelOption[];
  selectedModelId: string | null;
  isModelsLoading: boolean;
  addMessage: (message: ChatMessage) => void;
  addInputToHistory: (value: string) => void;
  moveHistoryCursor: (direction: "up" | "down") => string;
  resetHistoryCursor: () => void;
  setSending: (value: boolean) => void;
  setModelOptions: (value: ChatModelOption[]) => void;
  setSelectedModelId: (value: string | null) => void;
  setModelsLoading: (value: boolean) => void;
};

export const useLandingChatStore = create<LandingChatState>((set, get) => ({
  messages: [],
  inputHistory: [],
  historyCursor: null,
  isSending: false,
  modelOptions: [],
  selectedModelId: null,
  isModelsLoading: false,
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  addInputToHistory: (value) =>
    set((state) => ({
      inputHistory: [...state.inputHistory, value],
      historyCursor: null,
    })),
  moveHistoryCursor: (direction) => {
    const { inputHistory, historyCursor } = get();
    if (inputHistory.length === 0) return "";

    if (direction === "up") {
      const nextIndex =
        historyCursor === null
          ? inputHistory.length - 1
          : Math.max(0, historyCursor - 1);
      set({ historyCursor: nextIndex });
      return inputHistory[nextIndex] ?? "";
    }

    if (historyCursor === null) return "";
    const nextIndex = historyCursor + 1;
    if (nextIndex >= inputHistory.length) {
      set({ historyCursor: null });
      return "";
    }

    set({ historyCursor: nextIndex });
    return inputHistory[nextIndex] ?? "";
  },
  resetHistoryCursor: () => set({ historyCursor: null }),
  setSending: (value) => set({ isSending: value }),
  setModelOptions: (value) => set({ modelOptions: value }),
  setSelectedModelId: (value) => set({ selectedModelId: value }),
  setModelsLoading: (value) => set({ isModelsLoading: value }),
}));

export type { LandingChatState };


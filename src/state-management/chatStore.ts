import { create } from "zustand";

interface ChatState {
  isChatOpen: boolean;
  messages: { id: string; role: "user" | "agent"; content: string }[];
  toggleChat: () => void;
  addMessage: (message: ChatState["messages"][number]) => void;
}

export const createChatStore = () =>
  create<ChatState>((set) => ({
    isChatOpen: false,
    messages: [],
    toggleChat: () => set((state) => ({ isChatOpen: !state.isChatOpen })),
    addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  }));

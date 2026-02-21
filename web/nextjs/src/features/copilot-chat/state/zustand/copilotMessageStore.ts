import { create } from "zustand";
import { persist } from "zustand/middleware";

type CopilotMessageStoreState = {
  messages: unknown[];
  hasHydrated: boolean;
  setMessages: (messages: unknown[]) => void;
  clearMessages: () => void;
  setHasHydrated: (value: boolean) => void;
};

export const useCopilotMessageStore = create<CopilotMessageStoreState>()(
  persist(
    (set) => ({
      messages: [],
      hasHydrated: false,
      setMessages: (messages) =>
        set(() => ({
          messages,
        })),
      clearMessages: () => set(() => ({ messages: [] })),
      setHasHydrated: (value) => set(() => ({ hasHydrated: value })),
    }),
    {
      name: "copilot-message-store",
      partialize: (state) => ({ messages: state.messages }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    },
  ),
);

export type { CopilotMessageStoreState };

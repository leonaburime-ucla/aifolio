import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CopilotMessageStoreState } from "@/features/ag-ui-chat/__types__/typescript/react/state/copilotMessage.types";

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
      // Bump key to drop previously persisted transient/internal Copilot messages.
      name: "copilot-message-store-v2",
      partialize: (state) => ({ messages: state.messages }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    },
  ),
);

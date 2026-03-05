import { useCopilotMessageStore } from "@/features/ag-ui-chat/typescript/react/state/zustand/copilotMessageStore";
import type { CopilotMessageStatePort } from "@/features/ag-ui-chat/__types__/typescript/react/state/copilotMessage.types";
import { useShallow } from "zustand/react/shallow";

/**
 * Adapter that exposes Copilot message store via a neutral state port.
 * Views should use this adapter instead of importing the store directly.
 */
export function useCopilotMessageStateAdapter(): CopilotMessageStatePort {
  const { messages, hasHydrated, setMessages } = useCopilotMessageStore(
    useShallow((state) => ({
      messages: state.messages,
      hasHydrated: state.hasHydrated,
      setMessages: state.setMessages,
    }))
  );

  return {
    messages,
    hasHydrated,
    setMessages,
  };
}

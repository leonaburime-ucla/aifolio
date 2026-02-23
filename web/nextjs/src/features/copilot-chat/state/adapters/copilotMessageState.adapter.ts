import { useCopilotMessageStore } from "@/features/copilot-chat/state/zustand/copilotMessageStore";

/**
 * State port interface for copilot message persistence.
 */
export type CopilotMessageStatePort = {
  messages: unknown[];
  hasHydrated: boolean;
  setMessages: (messages: unknown[]) => void;
};

/**
 * Adapter that exposes Copilot message store via a neutral state port.
 * Views should use this adapter instead of importing the store directly.
 */
export function useCopilotMessageStateAdapter(): CopilotMessageStatePort {
  const messages = useCopilotMessageStore((state) => state.messages);
  const hasHydrated = useCopilotMessageStore((state) => state.hasHydrated);
  const setMessages = useCopilotMessageStore((state) => state.setMessages);

  return {
    messages,
    hasHydrated,
    setMessages,
  };
}

export type CopilotMessageStoreState = {
  messages: unknown[];
  hasHydrated: boolean;
  setMessages: (messages: unknown[]) => void;
  clearMessages: () => void;
  setHasHydrated: (value: boolean) => void;
};

export type CopilotMessageStatePort = {
  messages: unknown[];
  hasHydrated: boolean;
  setMessages: (messages: unknown[]) => void;
};

/**
 * Spec: copilot-chat.api.spec.ts
 * Version: 1.0.0
 */
export const COPILOT_CHAT_API_SPEC_VERSION = "1.0.0";

export const copilotChatApiSpec = {
  id: "copilot-chat.api",
  version: COPILOT_CHAT_API_SPEC_VERSION,
  endpoints: {
    runtimeProxy: "/api/copilotkit",
  },
  requirements: [
    "Copilot runtime/networking is configured in adapters/providers, not UI components.",
    "Sidebar components do not issue raw fetch calls.",
  ],
} as const;

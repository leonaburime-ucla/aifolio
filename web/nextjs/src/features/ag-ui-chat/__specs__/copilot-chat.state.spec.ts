/**
 * Spec: copilot-chat.state.spec.ts
 * Version: 1.0.0
 */
export const COPILOT_CHAT_STATE_SPEC_VERSION = "1.0.0";

export const copilotChatStateSpec = {
  id: "copilot-chat.state",
  version: COPILOT_CHAT_STATE_SPEC_VERSION,
  stores: ["copilotMessageStore"],
  requirements: [
    "State is feature-scoped and exposed through typed selectors/actions.",
    "No direct page-level mutations of internal store state.",
  ],
} as const;

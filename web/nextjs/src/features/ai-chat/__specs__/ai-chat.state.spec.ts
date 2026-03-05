/**
 * Spec: ai-chat.state.spec.ts
 * Version: 1.8.0
 * Purpose: machine-readable contract index for state scope.
 */
export const AI_CHAT_STATE_SPEC_VERSION = "1.8.0";

export const aiChatStateSpec = {
  id: "ai-chat.state",
  version: AI_CHAT_STATE_SPEC_VERSION,
  slices: ["aiChatStore"],
  requirementRefs: ["REQ-001", "REQ-002", "REQ-005", "DR-003", "ERR-003"],
} as const;

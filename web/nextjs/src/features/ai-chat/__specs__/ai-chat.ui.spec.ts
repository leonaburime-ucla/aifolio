/**
 * Spec: ai-chat.ui.spec.ts
 * Version: 1.8.0
 * Purpose: machine-readable contract index for ui scope.
 */
export const AI_CHAT_UI_SPEC_VERSION = "1.8.0";

export const aiChatUiSpec = {
  id: "ai-chat.ui",
  version: AI_CHAT_UI_SPEC_VERSION,
  components: ["ChatBar", "ChatSidebar", "CopilotChatSidebar"],
  requirementRefs: ["REQ-001", "REQ-002", "REQ-008", "ERR-004", "ERR-007"],
} as const;

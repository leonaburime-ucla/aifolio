/**
 * Spec: ai-chat.api.spec.ts
 * Version: 1.8.0
 * Purpose: machine-readable contract index for api scope.
 */
export const AI_CHAT_API_SPEC_VERSION = "1.8.0";

export const aiChatApiSpec = {
  id: "ai-chat.api",
  version: AI_CHAT_API_SPEC_VERSION,
  endpoints: {
    chat: "/chat",
    chatResearch: "/chat-research",
    geminiModels: "/llm/gemini-models",
  },
  requirementRefs: ["REQ-004", "REQ-009", "ERR-001", "ERR-002", "ERR-005"],
} as const;

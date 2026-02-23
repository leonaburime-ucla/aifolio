/**
 * Spec: ai-chat.api.spec.ts
 * Version: 1.2.0
 *
 * API dependency contract for AI chat orchestration.
 */
export const AI_CHAT_API_SPEC_VERSION = "1.2.0";

export const aiChatApiSpec = {
  id: "ai-chat.api",
  version: AI_CHAT_API_SPEC_VERSION,
  endpoints: {
    chat: "/chat",
    chatResearch: "/chat-research",
    modelList: "/llm/gemini-models",
  },
  requirements: [
    "Orchestrators inject API dependencies through ChatApiDeps.",
    "No orchestrator issues raw fetch calls.",
  ],
} as const;

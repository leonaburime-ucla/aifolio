/**
 * Spec: ai-chat.domain.spec.ts
 * Version: 1.2.0
 *
 * Domain contract for chat orchestration behavior.
 */
export const AI_CHAT_DOMAIN_SPEC_VERSION = "1.2.0";

export const aiChatDomainSpec = {
  id: "ai-chat.domain",
  version: AI_CHAT_DOMAIN_SPEC_VERSION,
  rules: [
    "A user submission is trimmed before processing.",
    "Empty or whitespace-only submissions are ignored.",
    "Chat history ordering is append-only by createdAt insertion order.",
    "Model fallback behavior is deterministic when fetch fails.",
  ],
} as const;

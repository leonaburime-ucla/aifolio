/**
 * Spec: ai-chat.state.spec.ts
 * Version: 1.2.0
 *
 * State and adapter boundary contract for AI chat.
 */
export const AI_CHAT_STATE_SPEC_VERSION = "1.2.0";

export const aiChatStateSpec = {
  id: "ai-chat.state",
  version: AI_CHAT_STATE_SPEC_VERSION,
  ports: [
    "ChatStatePort",
    "ChatChartActionsPort",
    "AgenticResearchStatePort",
  ],
  requirements: [
    "Orchestrators consume state via adapters only.",
    "Orchestrators do not import Zustand stores directly.",
    "Chart updates route through chart action ports.",
  ],
} as const;

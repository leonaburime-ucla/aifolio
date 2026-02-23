/**
 * Spec: ai-chat.ui.spec.ts
 * Version: 1.2.0
 *
 * UI integration contract for chat surfaces.
 */
export const AI_CHAT_UI_SPEC_VERSION = "1.2.0";

export const aiChatUiSpec = {
  id: "ai-chat.ui",
  version: AI_CHAT_UI_SPEC_VERSION,
  surfaces: [
    "ChatSidebar",
    "LandingChatSidebar",
    "CopilotChatSidebar",
  ],
  requirements: [
    "UI consumes ChatIntegration from orchestrators/hooks.",
    "UI behavior remains stable after adapter-boundary refactor.",
  ],
} as const;

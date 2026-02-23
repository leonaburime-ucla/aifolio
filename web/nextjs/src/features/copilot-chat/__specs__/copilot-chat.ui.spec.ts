/**
 * Spec: copilot-chat.ui.spec.ts
 * Version: 2.0.0
 */
export const COPILOT_CHAT_UI_SPEC_VERSION = "2.0.0";

export const copilotChatUiSpec = {
  id: "copilot-chat.ui",
  version: COPILOT_CHAT_UI_SPEC_VERSION,
  components: [
    "CopilotSidebar",
    "CopilotAssistantMessage",
    "CopilotAssistantMessageLegacy",
  ],
  providers: [
    "CopilotChatProvider",
    "CopilotEffectsProvider",
  ],
  requirements: [
    "Mode-based UI rendering is controlled by props and defaults.",
    "CopilotEffectsProvider consolidates all invisible side-effects (frontend tools, chart bridge, message persistence).",
    "No invisible components (components returning null) in views/components/.",
  ],
} as const;

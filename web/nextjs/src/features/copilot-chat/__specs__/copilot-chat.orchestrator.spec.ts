/**
 * Spec: copilot-chat.orchestrator.spec.ts
 * Version: 1.0.0
 */
export const COPILOT_CHAT_ORCHESTRATOR_SPEC_VERSION = "1.0.0";

export const copilotChatOrchestratorSpec = {
  id: "copilot-chat.orchestrator",
  version: COPILOT_CHAT_ORCHESTRATOR_SPEC_VERSION,
  orchestrators: ["frontendTools.orchestrator"],
  requirements: [
    "Frontend tool orchestration is encapsulated behind feature orchestrator functions.",
    "Orchestrator outputs are consumed by UI bridge components only.",
  ],
} as const;

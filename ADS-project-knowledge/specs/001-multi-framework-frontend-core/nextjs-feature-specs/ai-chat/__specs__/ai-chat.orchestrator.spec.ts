/**
 * Spec: ai-chat.orchestrator.spec.ts
 * Version: 1.8.0
 * Purpose: machine-readable contract index for orchestrator scope.
 */
export const AI_CHAT_ORCHESTRATOR_SPEC_VERSION = "1.8.0";

export const aiChatOrchestratorSpec = {
  id: "ai-chat.orchestrator",
  version: AI_CHAT_ORCHESTRATOR_SPEC_VERSION,
  orchestrators: ["chatOrchestrator"],
  requirementRefs: ["REQ-003", "REQ-006", "AB-001", "AB-002"],
} as const;

import type { UseChatChartActionsPort } from "@aifolio/contracts/entities/chat";

export type CopilotAssistantMessageOrchestratorDeps = {
  useChartActionsPort?: UseChatChartActionsPort;
};

export type CopilotAssistantMessageOrchestrator = {
  processAssistantPayload: (rawContent: string) => void;
};

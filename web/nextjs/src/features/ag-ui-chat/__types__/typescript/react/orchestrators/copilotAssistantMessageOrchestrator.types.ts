import type { UseChatChartActionsPort } from "@/features/ai-chat/__types__/typescript/chat.types";

export type CopilotAssistantMessageOrchestratorDeps = {
  useChartActionsPort?: UseChatChartActionsPort;
};

export type CopilotAssistantMessageOrchestrator = {
  processAssistantPayload: (rawContent: string) => void;
};

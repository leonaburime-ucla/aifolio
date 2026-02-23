"use client";

import type { AssistantMessageProps } from "@copilotkit/react-ui";
import { AssistantMessage as DefaultAssistantMessage } from "@copilotkit/react-ui";
import { extractCopilotDisplayMessage } from "@/features/copilot-chat/utils/copilotAssistantPayload.util";
import {
  useCopilotAssistantMessageOrchestrator,
  useCopilotAssistantPayloadEffect,
  type CopilotAssistantMessageOrchestratorDeps,
} from "@/features/copilot-chat/orchestrators/copilotAssistantMessage.orchestrator";

export type CopilotAssistantMessageLegacyProps = AssistantMessageProps &
  CopilotAssistantMessageOrchestratorDeps;

/**
 * Legacy assistant renderer for `/` AI Chat:
 * still parses assistant JSON payload and pushes `chartSpec` to chart store.
 *
 * Uses orchestrator pattern to maintain separation of concerns.
 * Chart actions are injected via dependency injection for testability.
 */
export default function CopilotAssistantMessageLegacy({
  message,
  useChartActionsPort,
  ...props
}: CopilotAssistantMessageLegacyProps) {
  const rawContent = message?.content || "";
  const orchestrator = useCopilotAssistantMessageOrchestrator({
    useChartActionsPort,
  });

  useCopilotAssistantPayloadEffect(rawContent, orchestrator);

  const nextMessage =
    message && typeof message.content === "string"
      ? { ...message, content: extractCopilotDisplayMessage(message.content) }
      : message;

  return <DefaultAssistantMessage {...props} message={nextMessage} />;
}

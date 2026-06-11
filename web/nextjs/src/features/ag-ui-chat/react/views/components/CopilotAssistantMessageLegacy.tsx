"use client";

import { AssistantMessage as DefaultAssistantMessage } from "@copilotkit/react-ui";
import type { CopilotAssistantMessageLegacyProps } from "@/features/ag-ui-chat/react/views/components/CopilotAssistantMessageLegacy.types";
import { toLegacyAssistantRenderMessage } from "@/features/ag-ui-chat/logic/copilotAssistantPayload.util";
import {
  useCopilotAssistantMessageOrchestrator,
  useCopilotAssistantPayloadEffect,
} from "@/features/ag-ui-chat/react/orchestrators/copilotAssistantMessage.orchestrator";

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
  const nextMessage = toLegacyAssistantRenderMessage(message);

  return <DefaultAssistantMessage {...props} message={nextMessage} />;
}

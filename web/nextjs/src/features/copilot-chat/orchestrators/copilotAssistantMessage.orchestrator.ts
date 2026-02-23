import { useEffect, useCallback } from "react";
import { parseCopilotAssistantPayload } from "@/features/copilot-chat/utils/copilotAssistantPayload.util";
import { useRechartsChartActionsAdapter } from "@/features/recharts/state/adapters/chartActions.adapter";
import type { ChatChartActionsPort, UseChatChartActionsPort } from "@/features/ai/types/chat.types";

export type CopilotAssistantMessageOrchestratorDeps = {
  useChartActionsPort?: UseChatChartActionsPort;
};

export type CopilotAssistantMessageOrchestrator = {
  processAssistantPayload: (rawContent: string) => void;
};

/**
 * Orchestrator hook for processing copilot assistant message payloads.
 *
 * Handles parsing assistant JSON payloads and adding chart specs to the store.
 * Uses dependency injection for chart actions to maintain testability.
 */
export function useCopilotAssistantMessageOrchestrator({
  useChartActionsPort = useRechartsChartActionsAdapter,
}: CopilotAssistantMessageOrchestratorDeps = {}): CopilotAssistantMessageOrchestrator {
  const { addChartSpec } = useChartActionsPort();

  const processAssistantPayload = useCallback(
    (rawContent: string) => {
      if (!rawContent) return;

      const payload = parseCopilotAssistantPayload(rawContent);
      if (!payload?.chartSpec) return;

      if (Array.isArray(payload.chartSpec)) {
        payload.chartSpec.forEach((spec) => addChartSpec(spec));
        console.log("[copilot-assistant-legacy] chart_specs_added", {
          count: payload.chartSpec.length,
          ids: payload.chartSpec.map((spec) => spec.id),
          types: payload.chartSpec.map((spec) => spec.type),
        });
        return;
      }

      addChartSpec(payload.chartSpec);
      console.log("[copilot-assistant-legacy] chart_spec_added", {
        id: payload.chartSpec.id,
        type: payload.chartSpec.type,
      });
    },
    [addChartSpec]
  );

  return { processAssistantPayload };
}

/**
 * Effect hook that processes assistant payload when content changes.
 *
 * Separates effect logic from the orchestrator for cleaner testing.
 */
export function useCopilotAssistantPayloadEffect(
  rawContent: string,
  orchestrator: CopilotAssistantMessageOrchestrator
): void {
  useEffect(() => {
    orchestrator.processAssistantPayload(rawContent);
  }, [rawContent, orchestrator]);
}

"use client";

import { useCopilotChartBridgeOrchestrator } from "@/features/ag-ui-chat/typescript/react/orchestrators/copilotChartBridge.orchestrator";
import { useCopilotMessagePersistenceOrchestrator } from "@/features/ag-ui-chat/typescript/react/orchestrators/copilotMessagePersistence.orchestrator";
import CopilotFrontendTools from "@/features/ag-ui-chat/typescript/react/views/components/CopilotFrontendTools";
import AgenticResearchAiTools from "@/features/agentic-research/typescript/react/ai/views/AgenticResearchAiTools";

/**
 * Single provider that consolidates all CopilotKit "invisible" side-effects.
 *
 * Previously these were scattered across multiple null-returning components:
 * - CopilotFrontendTools (useCopilotAction registrations)
 * - CopilotChartBridge (message to chart store bridge)
 * - CopilotMessagePersistence (localStorage persistence)
 *
 * This provider calls the orchestrator hooks that contain these effects,
 * eliminating the need for "fake components" that return null.
 */
function CopilotEffectsRuntime() {
  // Bridge CopilotKit messages to chart store
  useCopilotChartBridgeOrchestrator();

  // Sync CopilotKit messages to localStorage
  useCopilotMessagePersistenceOrchestrator();

  return null;
}

export function CopilotEffectsProvider({ children }: { children: React.ReactNode }) {
  return (
    <>
      <CopilotEffectsRuntime />
      <CopilotFrontendTools />
      <AgenticResearchAiTools />
      {children}
    </>
  );
}

export default CopilotEffectsProvider;

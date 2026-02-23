"use client";

import { useCopilotFrontendToolsOrchestrator } from "@/features/copilot-chat/orchestrators/copilotFrontendTools.orchestrator";
import { useCopilotChartBridgeOrchestrator } from "@/features/copilot-chat/orchestrators/copilotChartBridge.orchestrator";
import { useCopilotMessagePersistenceOrchestrator } from "@/features/copilot-chat/orchestrators/copilotMessagePersistence.orchestrator";

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
export function CopilotEffectsProvider({ children }: { children: React.ReactNode }) {
  // Register CopilotKit frontend tools (useCopilotAction calls)
  useCopilotFrontendToolsOrchestrator();

  // Bridge CopilotKit messages to chart store
  useCopilotChartBridgeOrchestrator();

  // Sync CopilotKit messages to localStorage
  useCopilotMessagePersistenceOrchestrator();

  return <>{children}</>;
}

export default CopilotEffectsProvider;

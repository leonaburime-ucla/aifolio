import type { ChatIntegration } from "@aifolio/contracts/entities/chat";
import { useChatSurfaceOrchestrator } from "@/features/ai-chat/react/compositions/useChatSurface.orchestrator";
import { useAiChatStateAdapter } from "@/features/ai-chat/react/state/adapters/aiChatState.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/react/state/adapters/agenticResearchState.adapter";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/react/state/adapters/chartActions.adapter";

function useAgenticResearchSelectedDatasetId(): string | null {
  const researchStatePort = useAgenticResearchStateAdapter();
  return researchStatePort.state.selectedDatasetId ?? null;
}

/**
 * Chat orchestrator scoped to Agentic Research charts.
 * Uses the shared AI chat state but writes chart payloads into the
 * Agentic Research chart store (not the global AI Chat chart store).
 */
export function useAgenticResearchChatOrchestrator(): ChatIntegration {
  return useChatSurfaceOrchestrator({
    useStatePort: useAiChatStateAdapter,
    useChartActionsPort: useAgenticResearchChartActionsAdapter,
    useActiveDatasetId: useAgenticResearchSelectedDatasetId,
    mode: "research",
  });
}

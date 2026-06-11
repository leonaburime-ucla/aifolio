import type { ChatApiDeps, ChatChartActionsPort } from "@aifolio/contracts/entities/chat";
import type { ChatIntegration } from "@aifolio/contracts/entities/chat";
import { useAiChatStateAdapter } from "@/features/ai-chat/react/state/adapters/aiChatState.adapter";
import { useChatSurfaceOrchestrator } from "@/features/ai-chat/react/compositions/useChatSurface.orchestrator";

/**
 * Orchestrator hook that wires state + API + logic dependencies into the chat integration hook.
 */
export type ChatOrchestratorInput = {
  chartActionsPort?: ChatChartActionsPort;
  apiAdapter?: ChatApiDeps;
};

export function useChatOrchestrator(
  input: ChatOrchestratorInput = {}
): ChatIntegration {
  return useChatSurfaceOrchestrator({
    useStatePort: useAiChatStateAdapter,
    activeDatasetId: null,
    useChartActionsPort: () => input.chartActionsPort ?? { addChartSpec: () => {} },
    apiAdapter: input.apiAdapter,
    mode: "research",
  });
}

export type { ChatIntegration as ChatOrchestrator };

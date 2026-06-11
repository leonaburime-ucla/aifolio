import type { ChatIntegration } from "@aifolio/contracts/entities/chat";
import { useChatSurfaceOrchestrator } from "@/features/ai-chat/react/compositions/useChatSurface.orchestrator";
import { useCopilotChartActionsAdapter } from "@/features/recharts/react/ai/state/adapters/chartActions.adapter";
import { useLandingChatStateAdapter } from "@/ui/screens/LandingPage/chat/state/adapters/landingChatState.adapter";

/**
 * Landing page chat orchestrator that uses /chat endpoint and isolated chat state.
 */
export function useLandingChatOrchestrator(): ChatIntegration {
  return useChatSurfaceOrchestrator({
    useStatePort: useLandingChatStateAdapter,
    useChartActionsPort: useCopilotChartActionsAdapter,
    mode: "direct",
  });
}

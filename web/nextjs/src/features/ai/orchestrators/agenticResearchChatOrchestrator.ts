import { useMemo } from "react";
import { fetchChatModels, sendChatMessage } from "@/features/ai/api/chatApi";
import { useChatIntegration } from "@/features/ai/hooks/useChat.hooks";
import type {
  ChatApiDeps,
  ChatDeps,
  ChatIntegration,
  ChatStateActions,
} from "@/features/ai/types/chat.types";
import { useAiChatStateAdapter } from "@/features/ai/state/adapters/aiChatState.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/state/adapters/agenticResearchState.adapter";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/state/adapters/chartActions.adapter";
import {
  composeChatStateActions,
  mapChatStateWithDataset,
} from "@/features/ai/orchestrators/chatOrchestrator.helpers";

/**
 * Chat orchestrator scoped to Agentic Research charts.
 * Uses the shared AI chat state but writes chart payloads into the
 * Agentic Research chart store (not the global AI Chat chart store).
 */
export function useAgenticResearchChatOrchestrator(): ChatIntegration {
  const chatStatePort = useAiChatStateAdapter();
  const chartActionsPort = useAgenticResearchChartActionsAdapter();
  const researchStatePort = useAgenticResearchStateAdapter();

  const state = useMemo(
    () =>
      mapChatStateWithDataset(
        chatStatePort.state,
        researchStatePort.state.selectedDatasetId ?? null
      ),
    [chatStatePort.state, researchStatePort.state.selectedDatasetId]
  );

  const actions = useMemo<ChatStateActions>(() => {
    return composeChatStateActions(
      chatStatePort.actions,
      chartActionsPort.addChartSpec
    );
  }, [chatStatePort.actions, chartActionsPort.addChartSpec]);

  const api = useMemo<ChatApiDeps>(
    () => ({
      sendMessage: sendChatMessage,
      fetchModels: fetchChatModels,
    }),
    [],
  );

  const deps = useMemo<ChatDeps>(() => ({ state, actions, api }), [state, actions, api]);
  return useChatIntegration(deps);
}

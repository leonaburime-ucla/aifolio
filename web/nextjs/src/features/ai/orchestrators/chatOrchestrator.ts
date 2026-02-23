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
import { useRechartsChartActionsAdapter } from "@/features/recharts/state/adapters/chartActions.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/state/adapters/agenticResearchState.adapter";
import {
  composeChatStateActions,
  mapChatStateWithDataset,
} from "@/features/ai/orchestrators/chatOrchestrator.helpers";

/**
 * Orchestrator hook that wires state + API dependencies into the chat integration hook.
 */
export function useChatOrchestrator(): ChatIntegration {
  const chatStatePort = useAiChatStateAdapter();
  const chartActionsPort = useRechartsChartActionsAdapter();
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
    []
  );

  const deps = useMemo<ChatDeps>(() => ({ state, actions, api }), [
    state,
    actions,
    api,
  ]);

  return useChatIntegration(deps);
}

export type { ChatIntegration as ChatOrchestrator };

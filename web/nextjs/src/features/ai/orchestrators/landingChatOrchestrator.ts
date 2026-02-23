import { useMemo } from "react";
import { fetchChatModels, sendChatMessageDirect } from "@/features/ai/api/chatApi";
import { useChatIntegration } from "@/features/ai/hooks/useChat.hooks";
import type {
  ChatApiDeps,
  ChatDeps,
  ChatIntegration,
  ChatStateActions,
} from "@/features/ai/types/chat.types";
import { useLandingChatStateAdapter } from "@/features/ai/state/adapters/landingChatState.adapter";
import { useRechartsChartActionsAdapter } from "@/features/recharts/state/adapters/chartActions.adapter";
import {
  composeChatStateActions,
  mapChatStateWithDataset,
} from "@/features/ai/orchestrators/chatOrchestrator.helpers";

/**
 * Landing page chat orchestrator that uses /chat endpoint and isolated chat state.
 */
export function useLandingChatOrchestrator(): ChatIntegration {
  const chatStatePort = useLandingChatStateAdapter();
  const chartActionsPort = useRechartsChartActionsAdapter();

  const state = useMemo(
    () => mapChatStateWithDataset(chatStatePort.state, null),
    [chatStatePort.state]
  );

  const actions = useMemo<ChatStateActions>(() => {
    return composeChatStateActions(
      chatStatePort.actions,
      chartActionsPort.addChartSpec
    );
  }, [chatStatePort.actions, chartActionsPort.addChartSpec]);

  const api = useMemo<ChatApiDeps>(
    () => ({
      sendMessage: sendChatMessageDirect,
      fetchModels: fetchChatModels,
    }),
    []
  );

  const deps = useMemo<ChatDeps>(() => ({ state, actions, api }), [state, actions, api]);
  return useChatIntegration(deps);
}

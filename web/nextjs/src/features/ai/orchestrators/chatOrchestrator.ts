import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { fetchChatModels, sendChatMessage } from "@/features/ai/api/chatApi";
import { useChatIntegration } from "@/features/ai/hooks/useChat.hooks";
import type {
  ChatApiDeps,
  ChatDeps,
  ChatIntegration,
  ChatState,
  ChatStateActions,
} from "@/features/ai/types/chat.types";
import { useAiChatStore } from "@/features/ai/state/zustand/aiChatStore";
import { useChartStore } from "@/features/recharts/state/zustand/chartStore";
import { useAgenticResearchState } from "@/features/agentic-research/state/zustand/agenticResearchStore";

/**
 * Orchestrator hook that wires state + API dependencies into the chat integration hook.
 */
export function useChatOrchestrator(): ChatIntegration {
  const researchState = useAgenticResearchState();
  const state = useAiChatStore(
    useShallow((store): ChatState => ({
      messages: store.messages,
      inputHistory: store.inputHistory,
      historyCursor: store.historyCursor,
      isSending: store.isSending,
      modelOptions: store.modelOptions,
      selectedModelId: store.selectedModelId,
      isModelsLoading: store.isModelsLoading,
      activeDatasetId: researchState.selectedDatasetId ?? null,
    }))
  );

  const actions = useMemo<ChatStateActions>(() => {
    const current = useAiChatStore.getState();
    const chartStore = useChartStore.getState();
    return {
      addMessage: current.addMessage,
      addInputToHistory: current.addInputToHistory,
      moveHistoryCursor: current.moveHistoryCursor,
      resetHistoryCursor: current.resetHistoryCursor,
      setSending: current.setSending,
      setModelOptions: current.setModelOptions,
      setSelectedModelId: current.setSelectedModelId,
      setModelsLoading: current.setModelsLoading,
      addChartSpec: chartStore.addChartSpec,
      onMessageReceived: (payload) => {
        // When we receive a message from the API, we need to sync 
        // the chart spec to the chart store so it shows up in the UI.
        if (!payload.chartSpec) return;
        if (Array.isArray(payload.chartSpec)) {
          payload.chartSpec.forEach((spec) => chartStore.addChartSpec(spec));
          return;
        }
        chartStore.addChartSpec(payload.chartSpec);
      },
    };
  }, []);

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

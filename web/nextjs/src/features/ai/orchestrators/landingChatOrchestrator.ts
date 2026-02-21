import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { fetchChatModels, sendChatMessageDirect } from "@/features/ai/api/chatApi";
import { useChatIntegration } from "@/features/ai/hooks/useChat.hooks";
import type {
  ChatApiDeps,
  ChatDeps,
  ChatIntegration,
  ChatState,
  ChatStateActions,
} from "@/features/ai/types/chat.types";
import { useLandingChatStore } from "@/features/ai/state/zustand/landingChatStore";
import { useChartStore } from "@/features/recharts/state/zustand/chartStore";

/**
 * Landing page chat orchestrator that uses /chat endpoint and isolated chat state.
 */
export function useLandingChatOrchestrator(): ChatIntegration {
  const state = useLandingChatStore(
    useShallow((store): ChatState => ({
      messages: store.messages,
      inputHistory: store.inputHistory,
      historyCursor: store.historyCursor,
      isSending: store.isSending,
      modelOptions: store.modelOptions,
      selectedModelId: store.selectedModelId,
      isModelsLoading: store.isModelsLoading,
      activeDatasetId: null,
    }))
  );

  const actions = useMemo<ChatStateActions>(() => {
    const current = useLandingChatStore.getState();
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
      sendMessage: sendChatMessageDirect,
      fetchModels: fetchChatModels,
    }),
    []
  );

  const deps = useMemo<ChatDeps>(() => ({ state, actions, api }), [state, actions, api]);
  return useChatIntegration(deps);
}


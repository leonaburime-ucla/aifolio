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
import { useAgenticResearchState } from "@/features/agentic-research/state/zustand/agenticResearchStore";
import { useAgenticResearchChartStore } from "@/features/agentic-research/state/zustand/agenticResearchChartStore";

/**
 * Chat orchestrator scoped to Agentic Research charts.
 * Uses the shared AI chat state but writes chart payloads into the
 * Agentic Research chart store (not the global AI Chat chart store).
 */
export function useAgenticResearchChatOrchestrator(): ChatIntegration {
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
    })),
  );

  const actions = useMemo<ChatStateActions>(() => {
    const current = useAiChatStore.getState();
    const chartStore = useAgenticResearchChartStore.getState();
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
      sendMessage: sendChatMessage,
      fetchModels: fetchChatModels,
    }),
    [],
  );

  const deps = useMemo<ChatDeps>(() => ({ state, actions, api }), [state, actions, api]);
  return useChatIntegration(deps);
}

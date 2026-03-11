import { useMemo } from "react";
import { createChatApiAdapter } from "@/features/ai-chat/typescript/api/chatApi.adapter";
import { useChatIntegration } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type {
  ChatApiDeps,
  ChatDeps,
  ChatIntegration,
  ChatLogicDeps,
  ChatStateActions,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import { useAiChatStateAdapter } from "@/features/ai-chat/typescript/react/state/adapters/aiChatState.adapter";
import { useAgenticResearchStateAdapter } from "@/features/agentic-research/typescript/react/state/adapters/agenticResearchState.adapter";
import { useAgenticResearchChartActionsAdapter } from "@/features/agentic-research/typescript/react/state/adapters/chartActions.adapter";
import {
  composeChatStateActions,
  mapChatStateWithDataset,
} from "@/features/ai-chat/typescript/logic/chatComposition.logic";
import {
  createChatApiDeps,
  createChatDeps,
} from "@/features/ai-chat/typescript/logic/chatOrchestrator.logic";
import {
  normalizeSubmissionValue,
  buildChatHistoryWindow,
  createUserChatMessage,
  createAssistantChatMessage,
  shouldRestoreDraftValue,
} from "@/features/ai-chat/typescript/logic/chatSubmission.logic";
import {
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
} from "@/features/ai-chat/typescript/logic/modelSelection.logic";

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
      mapChatStateWithDataset({
        state: chatStatePort.state,
        activeDatasetId: researchStatePort.state.selectedDatasetId ?? null,
      }),
    [chatStatePort.state, researchStatePort.state.selectedDatasetId]
  );

  const actions = useMemo<ChatStateActions>(() => {
    return composeChatStateActions({
      coreActions: chatStatePort.actions,
      addChartSpec: chartActionsPort.addChartSpec,
    });
  }, [chatStatePort.actions, chartActionsPort.addChartSpec]);

  const api = useMemo<ChatApiDeps>(
    () => createChatApiDeps({ ...createChatApiAdapter({ mode: "research" }) }),
    []
  );

  const logic = useMemo<ChatLogicDeps>(
    () => ({
      normalizeSubmissionValue,
      buildChatHistoryWindow,
      createUserChatMessage,
      createAssistantChatMessage,
      shouldRestoreDraftValue,
      resolveFallbackModelSelection,
      resolveFetchedModelSelection,
    }),
    []
  );

  const deps = useMemo<ChatDeps>(
    () => createChatDeps({ state, actions, api, logic }),
    [state, actions, api, logic]
  );
  return useChatIntegration(deps);
}

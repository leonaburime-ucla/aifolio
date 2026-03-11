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
import { useLandingChatStateAdapter } from "@/screens/LandingPage/chat/state/adapters/landingChatState.adapter";
import { useCopilotChartActionsAdapter } from "@/features/recharts/typescript/react/ai/state/adapters/chartActions.adapter";
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
 * Landing page chat orchestrator that uses /chat endpoint and isolated chat state.
 */
export function useLandingChatOrchestrator(): ChatIntegration {
  const chatStatePort = useLandingChatStateAdapter();
  const chartActionsPort = useCopilotChartActionsAdapter();

  const state = useMemo(
    () =>
      mapChatStateWithDataset({
        state: chatStatePort.state,
        activeDatasetId: null,
      }),
    [chatStatePort.state]
  );

  const actions = useMemo<ChatStateActions>(() => {
    return composeChatStateActions({
      coreActions: chatStatePort.actions,
      addChartSpec: chartActionsPort.addChartSpec,
    });
  }, [chatStatePort.actions, chartActionsPort.addChartSpec]);

  const api = useMemo<ChatApiDeps>(
    () => createChatApiDeps({ ...createChatApiAdapter({ mode: "direct" }) }),
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

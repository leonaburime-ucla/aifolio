import { useMemo } from "react";
import { createChatApiAdapter } from "@/features/ai-chat/typescript/api/chatApi.adapter";
import { useChatIntegration } from "@/features/ai-chat/typescript/react/hooks/useChat.hooks";
import type {
  ChatApiDeps,
  ChatChartActionsPort,
  ChatDeps,
  ChatIntegration,
  ChatLogicDeps,
  ChatStateActions,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import { useAiChatStateAdapter } from "@/features/ai-chat/typescript/react/state/adapters/aiChatState.adapter";
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
 * Orchestrator hook that wires state + API + logic dependencies into the chat integration hook.
 */
export type ChatOrchestratorInput = {
  chartActionsPort?: ChatChartActionsPort;
  apiAdapter?: ChatApiDeps;
};

export function useChatOrchestrator(
  input: ChatOrchestratorInput = {}
): ChatIntegration {
  const chatStatePort = useAiChatStateAdapter();

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
      addChartSpec: input.chartActionsPort?.addChartSpec ?? (() => { }),
    });
  }, [chatStatePort.actions, input.chartActionsPort]);

  const api = useMemo<ChatApiDeps>(
    () =>
      createChatApiDeps(
        input.apiAdapter ?? { ...createChatApiAdapter({ mode: "research" }) }
      ),
    [input.apiAdapter]
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
    [
      state,
      actions,
      api,
      logic,
    ]);

  return useChatIntegration(deps);
}

export type { ChatIntegration as ChatOrchestrator };

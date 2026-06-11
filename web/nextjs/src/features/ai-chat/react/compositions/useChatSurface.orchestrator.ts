import { useMemo } from "react";
import {
  createChatApiAdapter,
  type CreateChatApiAdapterInput,
} from "@/features/ai-chat/api/chatApi.adapter";
import type {
  ChatApiDeps,
  ChatChartActionsPort,
  ChatDeps,
  ChatLogicDeps,
  ChatStateActions,
} from "@aifolio/contracts/entities/chat";
import type {
  ChatIntegration,
  UseChatChartActionsPort,
  UseChatStatePort,
} from "@aifolio/contracts/entities/chat";
import {
  composeChatStateActions,
  mapChatStateWithDataset,
  createChatApiDeps,
  createChatDeps,
  buildChatHistoryWindow,
  createAssistantChatMessage,
  createUserChatMessage,
  normalizeSubmissionValue,
  shouldRestoreDraftValue,
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
} from "@aifolio/frontend-core/chat";
import { useChatIntegration } from "@/features/ai-chat/react/hooks/useChat.hooks";
import { useAiChatStateAdapter } from "@/features/ai-chat/react/state/adapters/aiChatState.adapter";

const EMPTY_CHART_ACTIONS_PORT: ChatChartActionsPort = {
  addChartSpec: () => {},
};

const useEmptyChartActionsPort: UseChatChartActionsPort = () =>
  EMPTY_CHART_ACTIONS_PORT;

const useNoActiveDataset = (): string | null => null;

export type UseChatSurfaceOptions = {
  useStatePort?: UseChatStatePort;
  useChartActionsPort?: UseChatChartActionsPort;
  activeDatasetId?: string | null;
  useActiveDatasetId?: () => string | null;
  mode?: CreateChatApiAdapterInput["mode"];
  apiAdapter?: ChatApiDeps;
};

export function useChatSurfaceOrchestrator({
  useStatePort = useAiChatStateAdapter,
  useChartActionsPort = useEmptyChartActionsPort,
  activeDatasetId,
  useActiveDatasetId = useNoActiveDataset,
  mode = "research",
  apiAdapter,
}: UseChatSurfaceOptions = {}): ChatIntegration {
  const chatStatePort = useStatePort();
  const chartActionsPort = useChartActionsPort();
  const hookDatasetId = useActiveDatasetId();
  const resolvedActiveDatasetId =
    activeDatasetId === undefined ? hookDatasetId : activeDatasetId;

  const state = useMemo(
    () =>
      mapChatStateWithDataset({
        state: chatStatePort.state,
        activeDatasetId: resolvedActiveDatasetId,
      }),
    [resolvedActiveDatasetId, chatStatePort.state]
  );

  const actions = useMemo<ChatStateActions>(
    () =>
      composeChatStateActions({
        coreActions: chatStatePort.actions,
        addChartSpec: chartActionsPort.addChartSpec,
      }),
    [chartActionsPort.addChartSpec, chatStatePort.actions]
  );

  const api = useMemo<ChatApiDeps>(
    () =>
      createChatApiDeps(
        apiAdapter ?? { ...createChatApiAdapter({ mode }) }
      ),
    [apiAdapter, mode]
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
    [actions, api, logic, state]
  );

  return useChatIntegration(deps);
}

import type {
  ChatAssistantPayload,
  ChatCoreStateActions,
  ChatState,
  ChatStateActions,
} from "@/features/ai/types/chat.types";
import type { ChartSpec } from "@/features/ai/types/chart.types";

export function mapChatStateWithDataset(
  state: Omit<ChatState, "activeDatasetId">,
  activeDatasetId: string | null
): ChatState {
  return {
    ...state,
    activeDatasetId,
  };
}

export function createOnMessageReceived(
  addChartSpec: (spec: ChartSpec) => void
): (payload: ChatAssistantPayload) => void {
  return (payload) => {
    if (!payload.chartSpec) return;
    if (Array.isArray(payload.chartSpec)) {
      payload.chartSpec.forEach((spec) => addChartSpec(spec));
      return;
    }
    addChartSpec(payload.chartSpec);
  };
}

export function composeChatStateActions(
  coreActions: ChatCoreStateActions,
  addChartSpec: ChatStateActions["addChartSpec"]
): ChatStateActions {
  return {
    ...coreActions,
    addChartSpec,
    onMessageReceived: createOnMessageReceived(addChartSpec),
  };
}

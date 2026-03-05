import type {
  ChatCoreStateActions,
  ChatState,
  ChatStateActions,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";

export type MapChatStateWithDatasetInput = {
  state: Omit<ChatState, "activeDatasetId">;
  activeDatasetId: string | null;
};

export type CreateOnMessageReceivedInput = {
  addChartSpec: (spec: ChartSpec) => void;
};

export type ComposeChatStateActionsInput = {
  coreActions: ChatCoreStateActions;
  addChartSpec: ChatStateActions["addChartSpec"];
};

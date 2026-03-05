import type {
  ChatAssistantPayload,
  ChatState,
  ChatStateActions,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import type {
  ComposeChatStateActionsInput,
  CreateOnMessageReceivedInput,
  MapChatStateWithDatasetInput,
} from "@/features/ai-chat/__types__/typescript/logic/chatComposition.types";

/**
 * Inject active dataset into the chat state model.
 *
 * @param input - Required dataset-state mapping inputs.
 * @returns Chat state with dataset context.
 */
export function mapChatStateWithDataset(
  input: MapChatStateWithDatasetInput
): ChatState {
  return {
    ...input.state,
    activeDatasetId: input.activeDatasetId,
  };
}

/**
 * Create assistant payload handler that forwards chart specs to chart actions.
 *
 * @param input - Required payload-handler inputs.
 * @returns Assistant payload handler.
 */
export function createOnMessageReceived(
  input: CreateOnMessageReceivedInput
): (payload: ChatAssistantPayload) => void {
  return (payload) => {
    if (!payload.chartSpec) return;
    if (Array.isArray(payload.chartSpec)) {
      payload.chartSpec.forEach((spec) => input.addChartSpec(spec));
      return;
    }
    input.addChartSpec(payload.chartSpec);
  };
}

/**
 * Compose full chat state actions from core actions + chart action.
 *
 * @param input - Required action-composition inputs.
 * @returns Complete chat state actions.
 */
export function composeChatStateActions(
  input: ComposeChatStateActionsInput
): ChatStateActions {
  return {
    ...input.coreActions,
    addChartSpec: input.addChartSpec,
    onMessageReceived: createOnMessageReceived({
      addChartSpec: input.addChartSpec,
    }),
  };
}

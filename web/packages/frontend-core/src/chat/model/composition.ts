import type {
  ChatAssistantPayload,
  ChatState,
  ChatStateActions,
  MapChatStateWithDatasetInput,
  CreateOnMessageReceivedInput,
  ComposeChatStateActionsInput,
} from "@aifolio/contracts/entities/chat";

/**
 * Inject the active dataset ID into a chat state object.
 *
 * @param input - Required: `state` (chat state without dataset), `activeDatasetId`.
 * @returns `ChatState` with `activeDatasetId` merged in.
 * @throws Never.
 * @complexity O(1) — shallow spread
 * @overallScore 100
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
 * Create an assistant payload handler that fans out chart specs to addChartSpec.
 * Handles single ChartSpec, ChartSpec[], or null/undefined payloads.
 *
 * @param input - Required: `addChartSpec` function reference.
 * @returns Handler function `(payload: ChatAssistantPayload) => void`.
 * @throws Never internally.
 * @sideEffects Calls `input.addChartSpec` for each chart spec in the payload.
 * @complexity O(n) — chart spec array length
 * @overallScore 100
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
 * Compose full ChatStateActions from core actions + chart action.
 * Wires `onMessageReceived` to fan out chart specs automatically.
 *
 * @param input - Required: `coreActions` (ChatCoreStateActions), `addChartSpec`.
 * @returns Complete `ChatStateActions` with all methods.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
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

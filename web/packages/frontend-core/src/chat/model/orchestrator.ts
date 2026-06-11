import type {
  ChatApiDeps,
  ChatAssistantPayload,
  ChatDeps,
  ChatLogicDeps,
  ChatState,
  ChatStateActions,
} from "@aifolio/contracts/entities/chat";
import type {
  SendChatMessageInput,
  SendChatMessageOptions,
  FetchChatModelsInput,
  FetchChatModelsOptions,
  FetchChatModelsResult,
} from "@aifolio/contracts/entities/chat/api";

export type CreateChatApiDepsInput = {
  sendMessage: (
    input: SendChatMessageInput,
    options?: SendChatMessageOptions
  ) => Promise<ChatAssistantPayload | null>;
  fetchModels: (
    input: FetchChatModelsInput,
    options?: FetchChatModelsOptions
  ) => Promise<FetchChatModelsResult | null>;
};

export type CreateChatDepsInput = {
  state: ChatState;
  actions: ChatStateActions;
  api: ChatApiDeps;
  logic: ChatLogicDeps;
};

/**
 * Assemble a ChatApiDeps bundle from individual API functions.
 *
 * @param input - Required: `sendMessage`, `fetchModels` function references.
 * @returns `ChatApiDeps` ready for injection into chat hooks.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
 */
export function createChatApiDeps(
  input: CreateChatApiDepsInput
): ChatApiDeps {
  return {
    sendMessage: input.sendMessage,
    fetchModels: input.fetchModels,
  };
}

/**
 * Assemble the full ChatDeps bundle consumed by chat integration hooks.
 *
 * @param input - Required: `state`, `actions`, `api`, `logic`.
 * @returns `ChatDeps` dependency bundle.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
 */
export function createChatDeps(input: CreateChatDepsInput): ChatDeps {
  return {
    state: input.state,
    actions: input.actions,
    api: input.api,
    logic: input.logic,
  };
}

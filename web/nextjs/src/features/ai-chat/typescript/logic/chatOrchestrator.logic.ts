import type {
  ChatApiDeps,
  ChatDeps,
} from "@/features/ai-chat/__types__/typescript/chat.types";
import type {
  CreateChatApiDepsInput,
  CreateChatDepsInput,
} from "@/features/ai-chat/__types__/typescript/logic/chatOrchestrator.types";

/**
 * Build chat API dependencies for orchestrators.
 *
 * @param input - Required API dependency inputs.
 * @returns Chat API dependency object.
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
 * Build chat dependency bundle consumed by chat integration hooks.
 *
 * @param input - Required chat dependency inputs.
 * @returns Chat dependency bundle.
 */
export function createChatDeps(
  input: CreateChatDepsInput
): ChatDeps {
  return {
    state: input.state,
    actions: input.actions,
    api: input.api,
  };
}

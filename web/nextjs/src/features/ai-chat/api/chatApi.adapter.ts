// Stays in Next.js app: wires app-local API functions (which bind process.env base URL)
// into the adapter shape consumed by React hooks — deployment + React coupling.
import {
  fetchChatModels,
  sendChatMessage,
  sendChatMessageDirect,
} from "@/features/ai-chat/api/chatApi";
import type { ChatApiDeps } from "@aifolio/contracts/entities/chat";

export type CreateChatApiAdapterInput = {
  mode: "research" | "direct";
};

export type CreateChatApiAdapterOptions = {
  sendResearchMessage?: ChatApiDeps["sendMessage"];
  sendDirectMessage?: ChatApiDeps["sendMessage"];
  fetchModels?: ChatApiDeps["fetchModels"];
};

/**
 * Build a feature-facing chat API adapter so orchestrators are decoupled from
 * transport-level endpoint functions and can be swapped later (for example,
 * TanStack Query-based adapters) without changing hook contracts.
 *
 * @param input - Required adapter mode configuration.
 * @param options - Optional dependency overrides for tests or alternate transports.
 * @returns Chat API dependency object consumed by chat orchestrators.
 */
export function createChatApiAdapter(
  input: CreateChatApiAdapterInput,
  options: CreateChatApiAdapterOptions = {}
): ChatApiDeps {
  const sendResearchMessage = options.sendResearchMessage ?? sendChatMessage;
  const sendDirectMessage = options.sendDirectMessage ?? sendChatMessageDirect;
  const fetchModelsAdapter = options.fetchModels ?? fetchChatModels;

  return {
    sendMessage:
      input.mode === "direct" ? sendDirectMessage : sendResearchMessage,
    fetchModels: fetchModelsAdapter,
  };
}

// Stays in Next.js app: binds getAiApiBaseUrl (process.env / browser proxy) and
// app-specific debug logging as defaults — deployment-specific wiring over pure
// fetch logic already extracted to @aifolio/frontend-core/chat.
import type { ChatAssistantPayload } from "@aifolio/contracts/entities/chat";
import type {
  ChatApiRuntimeDeps,
  FetchChatModelsInput,
  FetchChatModelsOptions,
  FetchChatModelsResult,
  SendChatMessageInput,
  SendChatMessageOptions,
} from "@aifolio/contracts/entities/chat/api";
import {
  fetchChatModels as fetchChatModelsCore,
  sendChatMessage as sendChatMessageCore,
  sendChatMessageDirect as sendChatMessageDirectCore,
} from "@aifolio/frontend-core/chat";
import { getAiApiBaseUrl } from "@/core/config/aiApi";

const DEBUG_AI_PROXY = process.env.NODE_ENV === "development";

function resolveNextRuntimeDeps(
  runtimeDeps?: ChatApiRuntimeDeps
): ChatApiRuntimeDeps {
  return {
    ...runtimeDeps,
    resolveBaseUrl: runtimeDeps?.resolveBaseUrl ?? getAiApiBaseUrl,
    debug: runtimeDeps?.debug ?? DEBUG_AI_PROXY,
  };
}

/**
 * Sends a chat message to the AI research endpoint using the Next app base URL config.
 *
 * @param input - Required message payload for the chat request.
 * @param options - Optional request options and test runtime overrides.
 * @returns Normalized assistant payload or null on invalid/non-ok responses.
 * @complexity O(n) local serialization/normalization over payload and response size, excluding network latency.
 * @overallScore 100
 */
export async function sendChatMessage(
  input: SendChatMessageInput,
  options?: SendChatMessageOptions
): Promise<ChatAssistantPayload | null> {
  return sendChatMessageCore(input, {
    ...options,
    runtimeDeps: resolveNextRuntimeDeps(options?.runtimeDeps),
  });
}

/**
 * Sends a chat message to the direct chat endpoint using the Next app base URL config.
 *
 * @param input - Required message payload for the base chat endpoint.
 * @param options - Optional request options and test runtime overrides.
 * @returns Normalized assistant payload or null on invalid/non-ok responses.
 * @complexity O(n) local serialization/normalization over payload and response size, excluding network latency.
 * @overallScore 100
 */
export async function sendChatMessageDirect(
  input: SendChatMessageInput,
  options?: SendChatMessageOptions
): Promise<ChatAssistantPayload | null> {
  return sendChatMessageDirectCore(input, {
    ...options,
    runtimeDeps: resolveNextRuntimeDeps(options?.runtimeDeps),
  });
}

/**
 * Fetches available chat models using the Next app base URL config.
 *
 * @param input - Required input object, empty by design for API-shape consistency.
 * @param options - Optional timeout and test runtime overrides.
 * @returns Current model and model options, a typed error result, or null on non-ok response.
 * @complexity O(n) over returned model count, excluding network latency.
 * @overallScore 100
 */
export async function fetchChatModels(
  input: FetchChatModelsInput,
  options?: FetchChatModelsOptions
): Promise<FetchChatModelsResult | null> {
  return fetchChatModelsCore(input, {
    ...options,
    runtimeDeps: resolveNextRuntimeDeps(options?.runtimeDeps),
  });
}

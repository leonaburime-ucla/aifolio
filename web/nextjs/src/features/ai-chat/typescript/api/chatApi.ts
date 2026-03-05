import type { ChatAssistantPayload } from "@/features/ai-chat/__types__/typescript/chat.types";
import type {
  ChatApiResponse,
  ChatApiRuntimeDeps,
  FetchChatModelsInput,
  FetchChatModelsOptions,
  FetchChatModelsResult,
  FetchChatModelsSuccessResult,
  ModelsApiResponse,
  SendChatMessageInput,
  SendChatMessageOptions,
  SendChatMessageToEndpointInput,
  SendChatMessageToEndpointOptions,
} from "@/features/ai-chat/__types__/typescript/api.types";
import { getAiApiBaseUrl } from "@/core/config/aiApi";
import {
  createModelFetchErrorResult,
  normalizeChatApiResult,
} from "@/features/ai-chat/typescript/logic/chatApiNormalization.logic";

const DEFAULT_MODELS_TIMEOUT_MS = 5000;

type ResolvedChatApiRuntimeDeps = {
  fetchImpl: typeof fetch;
  resolveBaseUrl: () => string;
  createAbortController: () => AbortController;
  setTimeoutImpl: typeof setTimeout;
  clearTimeoutImpl: typeof clearTimeout;
};

function resolveRuntimeDeps(
  runtimeDeps?: ChatApiRuntimeDeps
): ResolvedChatApiRuntimeDeps {
  const rawFetchImpl = runtimeDeps?.fetchImpl ?? globalThis.fetch;
  const fetchImpl: typeof fetch = (input, init) => rawFetchImpl(input, init);
  return {
    fetchImpl,
    resolveBaseUrl: runtimeDeps?.resolveBaseUrl ?? getAiApiBaseUrl,
    createAbortController:
      runtimeDeps?.createAbortController ?? (() => new AbortController()),
    setTimeoutImpl: runtimeDeps?.setTimeoutImpl ?? setTimeout,
    clearTimeoutImpl: runtimeDeps?.clearTimeoutImpl ?? clearTimeout,
  };
}

/**
 * Send a chat message to the AI backend.
 * @param input - Required message payload for the chat request.
 * @param options - Optional request options.
 * @returns The assistant payload or null on error.
 */
export async function sendChatMessage(
  input: SendChatMessageInput,
  options?: SendChatMessageOptions
): Promise<ChatAssistantPayload | null> {
  return sendChatMessageToEndpoint(
    {
      ...input,
      endpoint: "/chat-research",
    },
    options
  );
}

/**
 * Send a plain chat message to the base chat endpoint (/chat).
 *
 * @param input - Required message payload for the base chat endpoint.
 * @returns The assistant payload or null on error.
 */
export async function sendChatMessageDirect(
  input: SendChatMessageInput,
  options?: SendChatMessageOptions
): Promise<ChatAssistantPayload | null> {
  return sendChatMessageToEndpoint(
    {
      ...input,
      endpoint: "/chat",
    },
    { datasetId: null, runtimeDeps: options?.runtimeDeps }
  );
}

/**
 * Send a message payload to a specific chat endpoint and normalize the response.
 *
 * @param input - Required endpoint + message payload.
 * @param options - Optional endpoint options.
 * @returns Normalized assistant payload or null on non-ok/invalid responses.
 */
function sendChatMessageToEndpoint(
  input: SendChatMessageToEndpointInput,
  options?: SendChatMessageToEndpointOptions
): Promise<ChatAssistantPayload | null> {
  const runtime = resolveRuntimeDeps(options?.runtimeDeps);
  const baseUrl = runtime.resolveBaseUrl();
  return runtime.fetchImpl(`${baseUrl}${input.endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: input.value,
      attachments: input.attachments ?? [],
      model: input.model,
      messages: input.history,
      dataset_id: options?.datasetId ?? null,
    }),
  }).then(async (response) => {
    if (!response.ok) return null;
    try {
      const data = (await response.json()) as ChatApiResponse;
      const normalized = normalizeChatApiResult(data.result);
      return normalized ?? null;
    } catch {
      return null;
    }
  });
}

/**
 * Fetch the list of available models from the backend.
 *
 * @param _input - Required input object (empty by design for API-shape consistency).
 * @param options - Optional timeout configuration.
 * @returns The current model and available options, or null on error.
 */
export async function fetchChatModels(
  _input: FetchChatModelsInput,
  options?: FetchChatModelsOptions
): Promise<FetchChatModelsResult | null> {
  const runtime = resolveRuntimeDeps(options?.runtimeDeps);
  const controller = runtime.createAbortController();
  const timeoutMs = options?.timeoutMs ?? DEFAULT_MODELS_TIMEOUT_MS;
  const timeoutId = runtime.setTimeoutImpl(() => controller.abort(), timeoutMs);
  let result: FetchChatModelsResult | null;

  try {
    const response = await runtime.fetchImpl(
      `${runtime.resolveBaseUrl()}/llm/gemini-models`,
      {
      signal: controller.signal,
      }
    );
    if (!response.ok) {
      result = null;
    } else {
    const data = (await response.json()) as ModelsApiResponse;
      if (data.status !== "ok" || !data.models) {
        result = createModelFetchErrorResult({
          code: "MODEL_FETCH_FAILED",
          retryable: true,
          message: "Model endpoint returned an invalid payload.",
        });
      } else {
        result = {
          status: "ok",
          currentModel: data.currentModel ?? null,
          models: data.models,
        } satisfies FetchChatModelsSuccessResult;
      }
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      result = createModelFetchErrorResult({
        code: "MODEL_FETCH_TIMEOUT",
        retryable: true,
        message: "Model endpoint timed out.",
      });
    } else {
      result = createModelFetchErrorResult({
        code: "MODEL_FETCH_FAILED",
        retryable: true,
        message: "Model endpoint request failed.",
      });
    }
  }

  runtime.clearTimeoutImpl(timeoutId);
  return result;
}

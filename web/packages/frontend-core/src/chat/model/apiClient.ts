import type { ChatAssistantPayload } from "@aifolio/contracts/entities/chat";
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
} from "@aifolio/contracts/entities/chat/api";
import {
  createModelFetchErrorResult,
  normalizeChatApiResult,
} from "./normalization";

const DEFAULT_MODELS_TIMEOUT_MS = 5000;

type ChatRequestErrorCode =
  | "CHAT_REQUEST_HTTP_ERROR"
  | "CHAT_RESPONSE_PARSE_ERROR";

class ChatRequestError extends Error {
  code: ChatRequestErrorCode;
  status?: number;

  constructor(input: {
    code: ChatRequestErrorCode;
    message: string;
    status?: number;
    cause?: unknown;
  }) {
    super(input.message);
    this.name = "ChatRequestError";
    this.code = input.code;
    this.status = input.status;
    this.cause = input.cause;
  }
}

type ResolvedChatApiRuntimeDeps = {
  fetchImpl: typeof fetch;
  resolveBaseUrl: () => string;
  createAbortController: () => AbortController;
  setTimeoutImpl: typeof setTimeout;
  clearTimeoutImpl: typeof clearTimeout;
  debug: boolean;
  logger: Pick<Console, "warn">;
};

export function resolveRuntimeDeps(
  runtimeDeps?: ChatApiRuntimeDeps
): ResolvedChatApiRuntimeDeps {
  const rawFetchImpl = runtimeDeps?.fetchImpl ?? globalThis.fetch;
  const fetchImpl: typeof fetch = (input, init) => rawFetchImpl(input, init);
  return {
    fetchImpl,
    resolveBaseUrl: runtimeDeps?.resolveBaseUrl ?? (() => ""),
    createAbortController:
      runtimeDeps?.createAbortController ?? (() => new AbortController()),
    setTimeoutImpl: runtimeDeps?.setTimeoutImpl ?? setTimeout,
    clearTimeoutImpl: runtimeDeps?.clearTimeoutImpl ?? clearTimeout,
    debug: runtimeDeps?.debug ?? false,
    logger: runtimeDeps?.logger ?? console,
  };
}

export function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "AbortError"
  );
}

/**
 * Sends a chat message to the AI research endpoint.
 *
 * @param input - Required message payload for the chat request.
 * @param options - Optional request options and runtime dependencies.
 * @returns Normalized assistant payload or null on invalid/non-ok responses.
 * @complexity O(n) local serialization/normalization over payload and response size, excluding network latency.
 * @overallScore 100
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
 * Sends a chat message to the base chat endpoint.
 *
 * @param input - Required message payload for the base chat endpoint.
 * @param options - Optional request options and runtime dependencies.
 * @returns Normalized assistant payload or null on invalid/non-ok responses.
 * @complexity O(n) local serialization/normalization over payload and response size, excluding network latency.
 * @overallScore 100
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
 * Sends a message payload to a specific chat endpoint and normalizes the response.
 *
 * @param input - Required endpoint and message payload.
 * @param options - Optional dataset and runtime options.
 * @returns Normalized assistant payload or null on invalid/non-ok responses.
 * @complexity O(n) local serialization/normalization over payload and response size, excluding network latency.
 * @overallScore 100
 */
export function sendChatMessageToEndpoint(
  input: SendChatMessageToEndpointInput,
  options?: SendChatMessageToEndpointOptions
): Promise<ChatAssistantPayload | null> {
  const runtime = resolveRuntimeDeps(options?.runtimeDeps);
  const baseUrl = runtime.resolveBaseUrl();
  const requestUrl = `${baseUrl}${input.endpoint}`;

  if (runtime.debug) {
    runtime.logger.warn("[ai-chat] request", {
      endpoint: input.endpoint,
      url: requestUrl,
    });
  }

  return runtime.fetchImpl(requestUrl, {
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
    if (!response.ok) {
      if (runtime.debug) {
        runtime.logger.warn("[ai-chat] request non-ok response", {
          endpoint: input.endpoint,
          url: requestUrl,
          status: response.status,
        });
      }
      throw new ChatRequestError({
        code: "CHAT_REQUEST_HTTP_ERROR",
        message: `Chat request failed with status ${response.status}.`,
        status: response.status,
      });
    }
    try {
      const data = (await response.json()) as ChatApiResponse;
      const normalized = normalizeChatApiResult(data.result);
      if (!normalized && runtime.debug) {
        runtime.logger.warn("[ai-chat] request invalid payload", {
          endpoint: input.endpoint,
          url: requestUrl,
        });
      }
      return normalized ?? null;
    } catch (error) {
      if (runtime.debug) {
        runtime.logger.warn("[ai-chat] request response parse failed", {
          endpoint: input.endpoint,
          url: requestUrl,
          error,
        });
      }
      throw new ChatRequestError({
        code: "CHAT_RESPONSE_PARSE_ERROR",
        message: "Chat response body could not be parsed.",
        cause: error,
      });
    }
  });
}

/**
 * Fetches available chat models with timeout handling.
 *
 * @param _input - Required input object, empty by design for API-shape consistency.
 * @param options - Optional timeout and runtime dependency configuration.
 * @returns Current model and model options, a typed error result, or null on non-ok response.
 * @complexity O(n) over returned model count, excluding network latency.
 * @overallScore 100
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
    const requestUrl = `${runtime.resolveBaseUrl()}/llm/gemini-models`;
    if (runtime.debug) {
      runtime.logger.warn("[ai-chat] fetch-models", {
        url: requestUrl,
      });
    }

    const response = await runtime.fetchImpl(requestUrl, {
      signal: controller.signal,
    });
    if (!response.ok) {
      if (runtime.debug) {
        runtime.logger.warn("[ai-chat] fetch-models non-ok response", {
          url: requestUrl,
          status: response.status,
        });
      }
      result = null;
    } else {
      const data = (await response.json()) as ModelsApiResponse;
      if (data.status !== "ok" || !data.models) {
        if (runtime.debug) {
          runtime.logger.warn("[ai-chat] fetch-models invalid payload", {
            url: requestUrl,
            data,
          });
        }
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
    if (runtime.debug) {
      runtime.logger.warn("[ai-chat] fetch-models threw", {
        error,
      });
    }
    if (isAbortError(error)) {
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

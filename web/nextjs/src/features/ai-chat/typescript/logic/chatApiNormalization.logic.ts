import type { ChatAssistantPayload } from "@/features/ai-chat/__types__/typescript/chat.types";
import type { ChartSpec } from "@/features/ai-chat/__types__/typescript/chart.types";
import type {
  ChatApiError,
  ChatApiResponse,
  FetchChatModelsErrorResult,
} from "@/features/ai-chat/__types__/typescript/api.types";

/**
 * Build a deterministic model-fetch error result.
 *
 * @param input - Required error construction input.
 * @returns Normalized model-fetch error result.
 */
export function createModelFetchErrorResult(input: {
  code: ChatApiError["code"];
  retryable: boolean;
  message: string;
}): FetchChatModelsErrorResult {
  return {
    status: "error",
    error: {
      code: input.code,
      retryable: input.retryable,
      message: input.message,
    },
  };
}

/**
 * Attempt to parse a JSON string into the assistant payload.
 * Returns null if parsing fails or if the shape is invalid.
 *
 * @param raw - Raw text returned by the backend or model.
 * @returns Parsed assistant payload when valid JSON shape is detected; otherwise null.
 */
export function parseJsonPayload(raw: string): ChatAssistantPayload | null {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return null;
  try {
    const parsed = JSON.parse(trimmed) as {
      message?: string;
      chartSpec?: ChartSpec | ChartSpec[] | null;
    };
    if (parsed && typeof parsed.message === "string") {
      return {
        message: parsed.message,
        chartSpec: parsed.chartSpec ?? null,
      };
    }
  } catch {
    return null;
  }
  return null;
}

/**
 * Normalize a plain string or Gemini-style content parts into a payload.
 *
 * @param text - Raw text content.
 * @returns Assistant payload, parsing embedded JSON payloads when possible.
 */
export function normalizeTextResult(text: string): ChatAssistantPayload {
  const parsed = parseJsonPayload(text);
  return parsed ?? { message: text, chartSpec: null };
}

/**
 * Normalize the backend result into a structured payload for UI consumption.
 *
 * @param result - Backend result payload.
 * @returns Structured assistant payload, or null when payload is unusable.
 */
export function normalizeChatApiResult(
  result: ChatApiResponse["result"]
): ChatAssistantPayload | null {
  if (!result) return null;

  if (typeof result === "object" && !Array.isArray(result)) {
    const rawMessage =
      typeof result.message === "string" ? result.message : "";
    const parsedFromMessage = parseJsonPayload(rawMessage);
    if (parsedFromMessage) return parsedFromMessage;
    const chartSpec = result.chartSpec ?? null;
    if (!rawMessage && !chartSpec) return null;
    return { message: rawMessage, chartSpec };
  }

  if (typeof result === "string") {
    return normalizeTextResult(result);
  }

  if (Array.isArray(result)) {
    const textParts = result
      .map((part) => part.text ?? "")
      .filter(Boolean);
    if (!textParts.length) return null;
    return normalizeTextResult(textParts.join("\n"));
  }

  return null;
}

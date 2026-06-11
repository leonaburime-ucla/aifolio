import type { ChatAssistantPayload } from "@aifolio/contracts/entities/chat";
import { ChartSpecSchema } from "@aifolio/contracts/entities/chart";
import type {
  ChatApiError,
  ChatApiResponse,
  FetchChatModelsErrorResult,
} from "@aifolio/contracts/entities/chat/api";

const chartSpecOrArray = ChartSpecSchema.or(
  ChartSpecSchema.array()
).nullable();

function validateChartSpec(
  raw: unknown
): ChatAssistantPayload["chartSpec"] {
  const result = chartSpecOrArray.safeParse(raw ?? null);
  return result.success ? result.data : null;
}

/**
 * Build a typed model-fetch error result.
 *
 * @param input - Required: `code` (error code), `retryable`, `message`.
 * @returns `FetchChatModelsErrorResult` with status "error".
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
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
 * Parse a JSON string into a ChatAssistantPayload.
 * Returns null if the string is not JSON or lacks a `message` field.
 *
 * @param raw - Raw text to attempt JSON parsing on.
 * @returns Parsed `ChatAssistantPayload` or null on failure.
 * @throws Never — catches JSON.parse errors internally.
 * @complexity O(n) — string length (JSON.parse)
 * @overallScore 100
 */
export function parseJsonPayload(raw: string): ChatAssistantPayload | null {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return null;
  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    if (!parsed || typeof parsed.message !== "string") return null;
    return {
      message: parsed.message,
      chartSpec: validateChartSpec(parsed.chartSpec),
    };
  } catch {
    return null;
  }
}

/**
 * Normalize a plain text or embedded-JSON string into a ChatAssistantPayload.
 * Attempts JSON extraction first; falls back to wrapping as plain message.
 *
 * @param text - Raw text content from the backend.
 * @returns `ChatAssistantPayload` — always returns a value (never null).
 * @throws Never.
 * @complexity O(n) — string length
 * @overallScore 100
 */
export function normalizeTextResult(text: string): ChatAssistantPayload {
  const parsed = parseJsonPayload(text);
  return parsed ?? { message: text, chartSpec: null };
}

/**
 * Normalize a backend API result into a structured ChatAssistantPayload.
 * Handles object, string, and array (Gemini-style content parts) result shapes.
 *
 * @param result - Raw backend result (string | object | array | undefined).
 * @returns `ChatAssistantPayload` or null when the payload is unusable.
 * @throws Never.
 * @complexity O(n) — content part count for array results
 * @overallScore 100
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
    const chartSpec = validateChartSpec(result.chartSpec);
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

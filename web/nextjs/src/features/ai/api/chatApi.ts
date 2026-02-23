import type {
  ChatAssistantPayload,
  ChatHistoryMessage,
} from "@/features/ai/types/chat.types";
import type { ChartSpec } from "@/features/ai/types/chart.types";
import type {
  ChatApiResponse,
  ModelsApiResponse,
} from "@/features/ai/api/api.types";
import { getAiApiBaseUrl } from "@/core/config/aiApi";

/**
 * Attempt to parse a JSON string into the assistant payload.
 * Returns null if parsing fails or if the shape is invalid.
 * @param raw - Raw text returned by the backend or model.
 */
function parseJsonPayload(raw: string): ChatAssistantPayload | null {
  const trimmed = raw.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return null;
  // Avoid throwing in UI code if the model returns malformed JSON.
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
 * @param text - Raw text content.
 */
function normalizeTextResult(text: string): ChatAssistantPayload {
  const parsed = parseJsonPayload(text);
  return parsed ?? { message: text, chartSpec: null };
}

/**
 * Normalize the backend result into a structured payload for UI consumption.
 * Handles:
 * - Direct JSON payloads
 * - Raw text
 * - Gemini-style content parts
 * @param result - Backend result payload.
 */
function normalizeResult(
  result: ChatApiResponse["result"]
): ChatAssistantPayload | null {
  if (!result) return null;

  if (typeof result === "object" && !Array.isArray(result)) {
    const rawMessage =
      typeof result.message === "string" ? result.message : "";
    // Some models return JSON as a string inside `message`.
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

/**
 * Send a chat message to the AI backend.
 * @param value - The user message text.
 * @param model - Selected model ID, or null to use the backend default.
 * @param history - Recent conversation context (user/assistant pairs).
 * @returns The assistant payload or null on error.
 */
export async function sendChatMessage(
  value: string,
  model: string | null,
  history: ChatHistoryMessage[],
  attachments: ChatHistoryMessage["attachments"],
  datasetId?: string | null
): Promise<ChatAssistantPayload | null> {
  return sendChatMessageToEndpoint(
    "/chat-research",
    value,
    model,
    history,
    attachments,
    datasetId
  );
}

/**
 * Send a plain chat message to the base chat endpoint (/chat).
 */
export async function sendChatMessageDirect(
  value: string,
  model: string | null,
  history: ChatHistoryMessage[],
  attachments: ChatHistoryMessage["attachments"]
): Promise<ChatAssistantPayload | null> {
  return sendChatMessageToEndpoint(
    "/chat",
    value,
    model,
    history,
    attachments,
    null
  );
}

function sendChatMessageToEndpoint(
  endpoint: "/chat" | "/chat-research",
  value: string,
  model: string | null,
  history: ChatHistoryMessage[],
  attachments: ChatHistoryMessage["attachments"],
  datasetId?: string | null
): Promise<ChatAssistantPayload | null> {
  const baseUrl = getAiApiBaseUrl();
  return fetch(`${baseUrl}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: value,
      attachments: attachments ?? [],
      model,
      messages: history,
      dataset_id: datasetId ?? null,
    }),
  }).then(async (response) => {
    if (!response.ok) return null;
    const data = (await response.json()) as ChatApiResponse;
    const normalized = normalizeResult(data.result);
    return normalized ?? null;
  });
}

/**
 * Fetch the list of available models from the backend.
 * @returns The current model and available options, or null on error.
 */
export async function fetchChatModels(): Promise<{
  currentModel: string | null;
  models: Array<{ id: string; label: string }>;
} | null> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${getAiApiBaseUrl()}/llm/gemini-models`, {
      signal: controller.signal,
    });
    if (!response.ok) return null;
    const data = (await response.json()) as ModelsApiResponse;
    if (data.status !== "ok" || !data.models) return null;
    return {
      currentModel: data.currentModel ?? null,
      models: data.models,
    };
  } catch {
    return null;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Fetch agent status metadata from the backend.
 * @returns Agent status or null on error.
 */

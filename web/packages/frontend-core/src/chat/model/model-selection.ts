import type {
  ChatModelOption,
  FallbackSelectionInput,
  FallbackSelectionOptions,
  FetchedSelectionInput,
  ModelSelectionResult,
} from "@aifolio/contracts/entities/chat";

/** Stable fallback model list used when the backend model fetch fails. */
export const FALLBACK_CHAT_MODELS: ChatModelOption[] = [
  { id: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview" },
  { id: "gemini-3.1-pro-preview", label: "Gemini 3.1 Pro Preview" },
  { id: "gemini-3-pro-preview", label: "Gemini 3 Pro Preview" },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
];

/**
 * Resolve model options and selected model when the backend model fetch fails.
 *
 * @param input - Required: `selectedModelId` (current selection or null).
 * @param options - Optional: `fallbackModels` override list.
 * @returns `ModelSelectionResult` with fallback model list and resolved selection.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
 */
export function resolveFallbackModelSelection(
  input: FallbackSelectionInput,
  options?: FallbackSelectionOptions
): ModelSelectionResult {
  const models = options?.fallbackModels ?? FALLBACK_CHAT_MODELS;
  return {
    modelOptions: models,
    selectedModelId: input.selectedModelId ?? models[0]?.id ?? null,
  };
}

/**
 * Resolve model options and selected model from a successful backend fetch.
 * Priority chain: existing selection → backend currentModel → first in list → null.
 *
 * @param input - Required: `selectedModelId`, `result` with `currentModel` and `models`.
 * @returns `ModelSelectionResult` with fetched model list and resolved selection.
 * @throws Never.
 * @complexity O(1)
 * @overallScore 100
 */
export function resolveFetchedModelSelection(
  input: FetchedSelectionInput
): ModelSelectionResult {
  return {
    modelOptions: input.result.models,
    selectedModelId:
      input.selectedModelId ??
      input.result.currentModel ??
      input.result.models[0]?.id ??
      null,
  };
}

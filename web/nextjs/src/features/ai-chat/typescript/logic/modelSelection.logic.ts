import type { ChatModelOption } from "@/features/ai-chat/__types__/typescript/chat.types";
import type {
  FallbackSelectionInput,
  FallbackSelectionOptions,
  FetchedSelectionInput,
  ModelSelectionResult,
} from "@/features/ai-chat/__types__/typescript/logic/modelSelection.types";

export const FALLBACK_CHAT_MODELS: ChatModelOption[] = [
  { id: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview" },
  { id: "gemini-3.1-pro-preview", label: "Gemini 3.1 Pro Preview" },
  { id: "gemini-3-pro-preview", label: "Gemini 3 Pro Preview" },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
];


/**
 * Resolve model options + selected model when model fetch fails.
 *
 * @param input - Required model selection inputs.
 * @param options - Optional fallback override options.
 * @returns Deterministic model options and selected model ID for fallback behavior.
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
 * Resolve model options + selected model when model fetch succeeds.
 *
 * @param input - Required model selection inputs.
 * @returns Deterministic model options and selected model ID for fetched behavior.
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

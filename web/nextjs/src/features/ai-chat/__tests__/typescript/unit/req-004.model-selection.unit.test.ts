import { describe, expect, it } from "vitest";
import {
  FALLBACK_CHAT_MODELS,
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
} from "@/features/ai-chat/typescript/logic/modelSelection.logic";

/**
 * Spec-first sample test.
 * Requirement: REQ-004 (ai-chat.spec.md v1.8.0)
 * "Model-loading behavior MUST select in this order:
 *  existing selected model, API currentModel, first API model, fallback first model."
 */
describe("REQ-004 model selection precedence", () => {
  it("applies selection precedence exactly as specified", () => {
    const keepExisting = resolveFetchedModelSelection({
      selectedModelId: "already-selected",
      result: {
        currentModel: "api-current",
        models: [{ id: "m1", label: "Model 1" }],
      },
    });
    expect(keepExisting.selectedModelId).toBe("already-selected");

    const useCurrentModel = resolveFetchedModelSelection({
      selectedModelId: null,
      result: {
        currentModel: "api-current",
        models: [{ id: "m1", label: "Model 1" }],
      },
    });
    expect(useCurrentModel.selectedModelId).toBe("api-current");

    const useFirstApiModel = resolveFetchedModelSelection({
      selectedModelId: null,
      result: {
        currentModel: null,
        models: [{ id: "m1", label: "Model 1" }, { id: "m2", label: "Model 2" }],
      },
    });
    expect(useFirstApiModel.selectedModelId).toBe("m1");

    const emptyApiModels = resolveFetchedModelSelection({
      selectedModelId: null,
      result: {
        currentModel: null,
        models: [],
      },
    });
    expect(emptyApiModels.selectedModelId).toBeNull();

    const useFallbackFirstModel = resolveFallbackModelSelection({
      selectedModelId: null,
    });
    expect(useFallbackFirstModel.selectedModelId).toBe(
      FALLBACK_CHAT_MODELS[0]?.id ?? null,
    );

    const useProvidedFallbackModels = resolveFallbackModelSelection(
      { selectedModelId: null },
      { fallbackModels: [{ id: "custom-1", label: "Custom 1" }] }
    );
    expect(useProvidedFallbackModels.modelOptions).toEqual([
      { id: "custom-1", label: "Custom 1" },
    ]);
    expect(useProvidedFallbackModels.selectedModelId).toBe("custom-1");

    const keepSelectedOnFallback = resolveFallbackModelSelection(
      { selectedModelId: "kept" },
      { fallbackModels: [] }
    );
    expect(keepSelectedOnFallback.selectedModelId).toBe("kept");

    const emptyFallbackAndNoSelection = resolveFallbackModelSelection(
      { selectedModelId: null },
      { fallbackModels: [] }
    );
    expect(emptyFallbackAndNoSelection.selectedModelId).toBeNull();
  });
});

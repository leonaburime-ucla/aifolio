import { describe, it, expect } from "vitest";
import {
  resolveFallbackModelSelection,
  resolveFetchedModelSelection,
  FALLBACK_CHAT_MODELS,
} from "../../../src/chat/index";

describe("FALLBACK_CHAT_MODELS", () => {
  it("has gemini-3-flash-preview as first model", () => {
    expect(FALLBACK_CHAT_MODELS[0].id).toBe("gemini-3-flash-preview");
  });

  it("has 4 models in stable order", () => {
    expect(FALLBACK_CHAT_MODELS).toHaveLength(4);
  });
});

describe("resolveFallbackModelSelection", () => {
  it("returns fallback models and selects first when selectedModelId is null", () => {
    const result = resolveFallbackModelSelection({ selectedModelId: null });
    expect(result.modelOptions).toBe(FALLBACK_CHAT_MODELS);
    expect(result.selectedModelId).toBe("gemini-3-flash-preview");
  });

  it("preserves existing selectedModelId when non-null", () => {
    const result = resolveFallbackModelSelection({
      selectedModelId: "custom-model",
    });
    expect(result.selectedModelId).toBe("custom-model");
  });

  it("uses custom fallback models when provided", () => {
    const custom = [{ id: "m1", label: "M1" }];
    const result = resolveFallbackModelSelection(
      { selectedModelId: null },
      { fallbackModels: custom }
    );
    expect(result.modelOptions).toBe(custom);
    expect(result.selectedModelId).toBe("m1");
  });

  it("returns null selectedModelId when fallback list is empty", () => {
    const result = resolveFallbackModelSelection(
      { selectedModelId: null },
      { fallbackModels: [] }
    );
    expect(result.selectedModelId).toBeNull();
  });
});

describe("resolveFetchedModelSelection", () => {
  it("selects currentModel when selectedModelId is null", () => {
    const result = resolveFetchedModelSelection({
      selectedModelId: null,
      result: {
        currentModel: "model-x",
        models: [{ id: "model-x", label: "X" }, { id: "model-y", label: "Y" }],
      },
    });
    expect(result.selectedModelId).toBe("model-x");
    expect(result.modelOptions).toHaveLength(2);
  });

  it("preserves existing selectedModelId over currentModel", () => {
    const result = resolveFetchedModelSelection({
      selectedModelId: "already-set",
      result: {
        currentModel: "model-x",
        models: [{ id: "model-x", label: "X" }],
      },
    });
    expect(result.selectedModelId).toBe("already-set");
  });

  it("falls back to first model when both selectedModelId and currentModel are null", () => {
    const result = resolveFetchedModelSelection({
      selectedModelId: null,
      result: {
        currentModel: null,
        models: [{ id: "first", label: "First" }],
      },
    });
    expect(result.selectedModelId).toBe("first");
  });

  it("returns null when everything is null/empty", () => {
    const result = resolveFetchedModelSelection({
      selectedModelId: null,
      result: { currentModel: null, models: [] },
    });
    expect(result.selectedModelId).toBeNull();
  });
});

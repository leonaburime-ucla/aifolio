import { describe, expect, it } from "vitest";
import { resolveNextAgUiSelectedModelId } from "@/features/ag-ui-chat/typescript/logic/agUiContext.logic";

describe("resolveNextAgUiSelectedModelId", () => {
  const models = [
    { id: "gemini-3.1-pro-preview", label: "Gemini 3.1 Pro Preview" },
    { id: "m1", label: "Model 1" },
    { id: "m2", label: "Model 2" },
  ];

  it("keeps current selection when still available", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: "m2",
      fetchedModels: models,
      apiCurrentModelId: "m1",
    });
    expect(result).toBe("m2");
  });

  it("falls back to API current model when current selection is invalid", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: "missing",
      fetchedModels: models,
      apiCurrentModelId: "m1",
    });
    expect(result).toBe("gemini-3.1-pro-preview");
  });

  it("prefers Gemini 3.1 Pro when no valid current selection exists", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: null,
      fetchedModels: models,
      apiCurrentModelId: "missing",
    });
    expect(result).toBe("gemini-3.1-pro-preview");
  });

  it("falls back to API current model when Gemini 3.1 Pro is unavailable", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: null,
      fetchedModels: [{ id: "m1", label: "Model 1" }],
      apiCurrentModelId: "m1",
    });
    expect(result).toBe("m1");
  });

  it("returns null for empty model lists", () => {
    const result = resolveNextAgUiSelectedModelId({
      currentSelectedModelId: "m1",
      fetchedModels: [],
      apiCurrentModelId: "m1",
    });
    expect(result).toBeNull();
  });
});

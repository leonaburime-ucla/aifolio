import { describe, expect, it } from "vitest";
import {
  resolveNextAgUiSelectedModelId,
  toReadableModelOptions,
  toReadableDatasetOptions,
} from "../../../src/ag-ui/model/context";

describe("resolveNextAgUiSelectedModelId", () => {
  const models = [
    { id: "gemini-3-flash", label: "Gemini 3 Flash" },
    { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
  ];

  it("returns null when no models are available", () => {
    expect(
      resolveNextAgUiSelectedModelId({
        currentSelectedModelId: "gemini-3-flash",
        fetchedModels: [],
        apiCurrentModelId: null,
        preferredModelId: "gemini-3-flash",
      })
    ).toBeNull();
  });

  it("keeps current selection when it exists in fetched models", () => {
    expect(
      resolveNextAgUiSelectedModelId({
        currentSelectedModelId: "gemini-2.5-pro",
        fetchedModels: models,
        apiCurrentModelId: null,
        preferredModelId: "gemini-3-flash",
      })
    ).toBe("gemini-2.5-pro");
  });

  it("falls back to preferred model when current is not in list", () => {
    expect(
      resolveNextAgUiSelectedModelId({
        currentSelectedModelId: "removed-model",
        fetchedModels: models,
        apiCurrentModelId: null,
        preferredModelId: "gemini-3-flash",
      })
    ).toBe("gemini-3-flash");
  });

  it("falls back to API current model when preferred is not in list", () => {
    expect(
      resolveNextAgUiSelectedModelId({
        currentSelectedModelId: null,
        fetchedModels: models,
        apiCurrentModelId: "gemini-2.5-pro",
        preferredModelId: "not-in-list",
      })
    ).toBe("gemini-2.5-pro");
  });

  it("falls back to first model when nothing else matches", () => {
    expect(
      resolveNextAgUiSelectedModelId({
        currentSelectedModelId: null,
        fetchedModels: models,
        apiCurrentModelId: "also-not-in-list",
        preferredModelId: "not-in-list",
      })
    ).toBe("gemini-3-flash");
  });

  it("returns null when fetched model entries are malformed", () => {
    expect(
      resolveNextAgUiSelectedModelId({
        currentSelectedModelId: null,
        fetchedModels: [{} as never],
        apiCurrentModelId: null,
        preferredModelId: "not-in-list",
      })
    ).toBeNull();
  });
});

describe("toReadableModelOptions", () => {
  it("maps model options to id/label pairs", () => {
    const result = toReadableModelOptions([
      { id: "a", label: "Model A" },
      { id: "b", label: "Model B" },
    ]);
    expect(result).toEqual([
      { id: "a", label: "Model A" },
      { id: "b", label: "Model B" },
    ]);
  });
});

describe("toReadableDatasetOptions", () => {
  it("maps dataset options to id/label pairs", () => {
    const result = toReadableDatasetOptions([
      { id: "iris", label: "Iris Dataset" },
    ]);
    expect(result).toEqual([{ id: "iris", label: "Iris Dataset" }]);
  });
});

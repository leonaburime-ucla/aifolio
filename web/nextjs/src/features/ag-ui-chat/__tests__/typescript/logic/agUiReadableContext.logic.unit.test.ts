import { describe, expect, it } from "vitest";
import {
  toReadableDatasetOptions,
  toReadableModelOptions,
} from "@/features/ag-ui-chat/typescript/logic/agUiContext.logic";

describe("toReadableModelOptions", () => {
  it("maps options to id/label pairs", () => {
    const result = toReadableModelOptions([
      { id: "m1", label: "Model 1" },
      { id: "m2", label: "Model 2" },
    ]);

    expect(result).toEqual([
      { id: "m1", label: "Model 1" },
      { id: "m2", label: "Model 2" },
    ]);
  });
});

describe("toReadableDatasetOptions", () => {
  it("maps dataset options to id/label pairs", () => {
    const result = toReadableDatasetOptions([
      { id: "d1", label: "Dataset 1" },
      { id: "d2", label: "Dataset 2" },
    ]);

    expect(result).toEqual([
      { id: "d1", label: "Dataset 1" },
      { id: "d2", label: "Dataset 2" },
    ]);
  });
});

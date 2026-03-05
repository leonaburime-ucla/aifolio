import { describe, expect, it } from "vitest";
import {
  formatPercentLabel,
  hasModelArtifacts,
} from "@/features/ml/typescript/logic/mlTrainingModals.logic";

describe("mlTrainingModals.logic", () => {
  it("formats numeric percentages", () => {
    expect(
      formatPercentLabel({
        value: 12.3456,
        fallback: "(n/a)",
      })
    ).toBe("(12.35%)");
  });

  it("returns fallback when percentage is missing", () => {
    expect(
      formatPercentLabel({
        value: null,
        fallback: "(n/a)",
      })
    ).toBe("(n/a)");
    expect(
      formatPercentLabel({
        value: Number.NaN,
        fallback: "(n/a)",
      })
    ).toBe("(n/a)");
  });

  it("detects model artifact visibility", () => {
    expect(
      hasModelArtifacts({
        modelId: "m1",
        modelPath: null,
      })
    ).toBe(true);
    expect(
      hasModelArtifacts({
        modelId: null,
        modelPath: "/tmp/m1",
      })
    ).toBe(true);
    expect(
      hasModelArtifacts({
        modelId: null,
        modelPath: null,
      })
    ).toBe(false);
  });
});

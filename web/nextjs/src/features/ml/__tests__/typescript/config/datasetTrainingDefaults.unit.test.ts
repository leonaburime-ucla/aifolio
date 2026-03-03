import { describe, expect, it } from "vitest";
import { getTrainingDefaults } from "@/features/ml/typescript/config/datasetTrainingDefaults";

describe("datasetTrainingDefaults", () => {
  it("returns dataset-specific defaults when known id is provided", () => {
    const defaults = getTrainingDefaults("customer_churn_telco.csv");
    expect(defaults.targetColumn).toBe("Churn");
    expect(defaults.task).toBe("classification");
  });

  it("returns fallback defaults when dataset id is unknown", () => {
    const defaults = getTrainingDefaults("unknown.csv");
    expect(defaults).toEqual({
      targetColumn: "",
      excludeColumns: [],
      dateColumns: [],
      task: "auto",
      epochs: 60,
    });
  });

  it("returns fallback defaults when dataset id is null", () => {
    const defaults = getTrainingDefaults(null);
    expect(defaults.task).toBe("auto");
    expect(defaults.epochs).toBe(60);
  });
});

import { describe, expect, it } from "vitest";
import {
  DEFAULT_ML_DATASET_ID,
  getTrainingDefaults,
  resolveDefaultTrainingDatasetId,
} from "@aifolio/frontend-core/ml-training";

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

  it("exports customer churn as the shared default dataset id", () => {
    expect(DEFAULT_ML_DATASET_ID).toBe("customer_churn_telco.csv");
  });

  it("resolves the default dataset with customer churn preference", () => {
    expect(
      resolveDefaultTrainingDatasetId({
        selectedDatasetId: "existing.csv",
        datasets: [{ id: DEFAULT_ML_DATASET_ID }],
      })
    ).toBe("existing.csv");

    expect(
      resolveDefaultTrainingDatasetId({
        selectedDatasetId: null,
        datasets: [{ id: "fraud.csv" }, { id: DEFAULT_ML_DATASET_ID }],
      })
    ).toBe(DEFAULT_ML_DATASET_ID);

    expect(
      resolveDefaultTrainingDatasetId({
        selectedDatasetId: null,
        datasets: [{ id: "fallback.csv" }],
      })
    ).toBe("fallback.csv");
  });
});

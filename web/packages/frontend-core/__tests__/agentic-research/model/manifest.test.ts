import { describe, expect, it } from "vitest";
import {
  resolveDefaultDatasetId,
  toDatasetOptions,
} from "@aifolio/frontend-core/agentic-research";

describe("agenticResearchManifest.logic", () => {
  it("resolveDefaultDatasetId prefers selected id then customer churn then first dataset then null", () => {
    const datasets = [
      { id: "iris", label: "Iris", description: "Iris dataset" },
      { id: "customer_churn_telco.csv", label: "Churn", description: "Customer churn dataset" },
    ];

    expect(
      resolveDefaultDatasetId({ selectedDatasetId: "iris", datasets })
    ).toBe("iris");
    expect(
      resolveDefaultDatasetId({ selectedDatasetId: null, datasets })
    ).toBe("customer_churn_telco.csv");
    expect(
      resolveDefaultDatasetId({
        selectedDatasetId: null,
        datasets: [{ id: "iris", label: "Iris", description: "Iris dataset" }],
      })
    ).toBe("iris");
    expect(
      resolveDefaultDatasetId({ selectedDatasetId: null, datasets: [] })
    ).toBeNull();
  });

  it("maps dataset options while preserving manifest order", () => {
    const datasetManifest = [
      { id: "b", label: "B", description: "second" },
      { id: "a", label: "A", description: "first" },
    ];

    expect(toDatasetOptions({ datasetManifest })).toEqual([
      { id: "b", label: "B", description: "second" },
      { id: "a", label: "A", description: "first" },
    ]);
  });
});

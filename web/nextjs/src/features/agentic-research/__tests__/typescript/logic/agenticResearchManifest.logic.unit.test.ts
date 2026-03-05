import { describe, expect, it } from "vitest";
import {
  resolveDefaultDatasetId,
  toDatasetOptions,
} from "@/features/agentic-research/typescript/logic/agenticResearchManifest.logic";

describe("agenticResearchManifest.logic", () => {
  it("resolveDefaultDatasetId prefers selected id then first dataset then null", () => {
    const datasets = [
      { id: "iris", label: "Iris", description: "Iris dataset" },
      { id: "wine", label: "Wine", description: "Wine dataset" },
    ];

    expect(
      resolveDefaultDatasetId({ selectedDatasetId: "wine", datasets })
    ).toBe("wine");
    expect(
      resolveDefaultDatasetId({ selectedDatasetId: null, datasets })
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

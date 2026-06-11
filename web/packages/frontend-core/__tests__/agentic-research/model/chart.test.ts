import { describe, expect, it } from "vitest";
import { buildPcaChartSpec } from "../../../src/agentic-research";

describe("agentic-research chart helpers", () => {
  it("builds a PCA scatter chart from transformed points", () => {
    expect(
      buildPcaChartSpec({
        transformed: [
          [1.25, -0.5],
          [0.5, 1.1],
        ],
        explained_variance_ratio: [0.6, 0.3, 0.1],
      })
    ).toEqual({
      id: "agentic-research-pca",
      title: "PCA Projection",
      description: "Explained variance: PC1 60.0%, PC2 30.0%, PC3 10.0%",
      type: "scatter",
      xKey: "pc1",
      yKeys: ["pc2"],
      xLabel: "PC1",
      yLabel: "PC2",
      data: [
        { id: "pca-1", pc1: 1.25, pc2: -0.5 },
        { id: "pca-2", pc1: 0.5, pc2: 1.1 },
      ],
    });
  });

  it("uses coordinate and description fallbacks for sparse PCA output", () => {
    expect(
      buildPcaChartSpec({
        transformed: [[undefined, undefined], [2]],
        explained_variance_ratio: [0.7],
      })
    ).toEqual(
      expect.objectContaining({
        description: undefined,
        data: [
          { id: "pca-1", pc1: 0, pc2: 0 },
          { id: "pca-2", pc1: 2, pc2: 0 },
        ],
      })
    );
  });

  it("omits variance description when variance data is missing", () => {
    const result = buildPcaChartSpec({
      transformed: [[1, 2]],
    });

    expect(result).toEqual(
      expect.objectContaining({
        description: undefined,
        data: [{ id: "pca-1", pc1: 1, pc2: 2 }],
      })
    );
  });

  it("returns null when PCA output has no transformed rows", () => {
    expect(buildPcaChartSpec({ transformed: [] })).toBeNull();
    expect(buildPcaChartSpec({ transformed: undefined })).toBeNull();
  });
});

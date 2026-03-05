import { describe, expect, it } from "vitest";
import {
  formatToolName,
  resolveActiveChartSpec,
} from "@/features/agentic-research/typescript/logic/agenticResearchChart.logic";

describe("agenticResearchChart.logic", () => {
  it("resolveActiveChartSpec uses deterministic precedence", () => {
    const chartA = {
      id: "a",
      title: "A",
      type: "scatter" as const,
      xKey: "x",
      yKeys: ["y"],
      data: [],
    };
    const chartB = { ...chartA, id: "b" };

    expect(
      resolveActiveChartSpec({
        pcaChartSpec: chartA,
        chartSpecs: [chartB],
      })
    ).toEqual(chartA);

    expect(
      resolveActiveChartSpec({
        pcaChartSpec: null,
        chartSpecs: [chartB],
      })
    ).toEqual(chartB);

    expect(
      resolveActiveChartSpec({
        pcaChartSpec: null,
        chartSpecs: [],
      })
    ).toBeNull();
  });

  it("formatToolName resolves acronym tokens and title-cases fallback words", () => {
    expect(formatToolName({ name: "pca_transform" })).toBe("PCA Transform");
    expect(formatToolName({ name: "kmeans_clustering" })).toBe(
      "K-Means Clustering"
    );
    expect(formatToolName({ name: "custom_tool" })).toBe("Custom Tool");
    expect(
      formatToolName({ name: "pca_transform" }, { acronyms: { pca: "PC-A" } })
    ).toBe("PC-A Transform");
  });
});

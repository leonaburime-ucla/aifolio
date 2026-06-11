import { describe, it, expect } from "vitest";
import { buildChartOption } from "~/features/recharts/lib";
import { getEChartsOption } from "@aifolio/frontend-core/recharts";
import type { ChartSpec } from "~/composables/useChartStore";

const PCA_SPEC: ChartSpec = {
  id: "agentic-research-pca-loadings-f624caa6",
  title: "PCA Loadings",
  type: "scatter",
  xKey: "pc1",
  yKeys: ["pc2"],
  xLabel: "PC1",
  yLabel: "PC2",
  data: [
    { id: "loading-1", feature: "Contract=Month-to-month", pc1: -0.0723, pc2: -0.2889 },
    { id: "loading-2", feature: "Contract=One year", pc1: 0.0146, pc2: 0.1013 },
    { id: "loading-3", feature: "Contract=Two year", pc1: 0.0703, pc2: 0.2398 },
    { id: "loading-4", feature: "Dependents=No", pc1: -0.0569, pc2: -0.1408 },
    { id: "loading-5", feature: "Dependents=Yes", pc1: 0.0569, pc2: 0.1408 },
    { id: "loading-6", feature: "TotalCharges", pc1: 0.3012, pc2: 0.0812 },
    { id: "loading-7", feature: "tenure", pc1: 0.2845, pc2: 0.1523 },
  ],
};

describe("buildChartOption — scatter (PCA)", () => {
  it("delegates to getEChartsOption for scatter type and adds encode", () => {
    const result = buildChartOption(PCA_SPEC);
    const shared = getEChartsOption({ spec: PCA_SPEC });

    expect(shared).not.toBeNull();
    expect((result.xAxis as any).type).toBe((shared!.xAxis as any).type);
    expect((result.series as any[])[0].type).toBe((shared!.series as any[])[0].type);
    expect((result.series as any[])[0].encode).toEqual({ x: "pc1", y: "pc2" });
    expect(result.dataset).toBeDefined();
  });

  it("uses value axes (not category) for scatter", () => {
    const result = buildChartOption(PCA_SPEC);
    const xAxis = result.xAxis as any;
    const yAxis = result.yAxis as any;

    expect(xAxis.type).toBe("value");
    expect(yAxis.type).toBe("value");
  });

  it("sets axis labels from spec xLabel/yLabel", () => {
    const result = buildChartOption(PCA_SPEC);
    const xAxis = result.xAxis as any;
    const yAxis = result.yAxis as any;

    expect(xAxis.name).toBe("PC1");
    expect(yAxis.name).toBe("PC2");
  });

  it("uses dataset + encode for scatter data mapping", () => {
    const result = buildChartOption(PCA_SPEC);
    const series = result.series as any[];
    const dataset = result.dataset as any;

    expect(series).toHaveLength(1);
    expect(series[0].type).toBe("scatter");
    expect(series[0].encode).toEqual({ x: "pc1", y: "pc2" });
    expect(dataset.source).toBe(PCA_SPEC.data);
    expect(dataset.source).toHaveLength(7);
  });

  it("includes dataZoom for interactivity", () => {
    const result = buildChartOption(PCA_SPEC);
    const dataZoom = result.dataZoom as any[];

    expect(dataZoom).toBeDefined();
    expect(dataZoom.length).toBeGreaterThan(0);
  });

  it("enables feature labels when data has 'feature' field", () => {
    const result = buildChartOption(PCA_SPEC);
    const series = result.series as any[];

    expect(series[0].label).toBeDefined();
    expect(series[0].label.show).toBe(true);
  });

  it("getEChartsOption returns null for line type (fallback path)", () => {
    const lineSpec: ChartSpec = {
      id: "line1",
      title: "Line",
      type: "line",
      xKey: "year",
      yKeys: ["value"],
      data: [{ year: "2020", value: 100 }],
    };
    const shared = getEChartsOption({ spec: lineSpec });
    expect(shared).toBeNull();

    const result = buildChartOption(lineSpec);
    expect(result.series).toBeDefined();
    const xAxis = result.xAxis as any;
    expect(xAxis.type).toBe("category");
  });
});

import { describe, it, expect } from "vitest";
import { ChartSpecSchema } from "../src/entities/chart/index.ts";

describe("ChartSpecSchema", () => {
  const validChartSpec = {
    id: "chart-1",
    title: "Revenue Over Time",
    type: "line" as const,
    xKey: "date",
    yKeys: ["revenue"],
    data: [{ date: "2026-01", revenue: 1000 }],
  };

  it("validates a minimal valid ChartSpec", () => {
    const result = ChartSpecSchema.parse(validChartSpec);
    expect(result.id).toBe("chart-1");
    expect(result.type).toBe("line");
  });

  it("validates all 15 chart types", () => {
    const types = [
      "line", "area", "bar", "scatter", "histogram", "density",
      "roc", "pr", "errorbar", "heatmap", "box", "violin",
      "biplot", "dendrogram", "surface",
    ] as const;

    for (const type of types) {
      const result = ChartSpecSchema.parse({ ...validChartSpec, type });
      expect(result.type).toBe(type);
    }
  });

  it("rejects missing id field", () => {
    const { id: _, ...noId } = validChartSpec;
    expect(() => ChartSpecSchema.parse(noId)).toThrow();
  });

  it("rejects missing title field", () => {
    const { title: _, ...noTitle } = validChartSpec;
    expect(() => ChartSpecSchema.parse(noTitle)).toThrow();
  });

  it("rejects invalid chart type", () => {
    expect(() =>
      ChartSpecSchema.parse({ ...validChartSpec, type: "pie" })
    ).toThrow();
  });

  it("accepts all optional fields", () => {
    const full = {
      ...validChartSpec,
      description: "A chart",
      xLabel: "Date",
      yLabel: "Revenue",
      zKey: "depth",
      colorKey: "category",
      errorKeys: { revenue: "revenue_err" },
      unit: "USD",
      currency: "USD",
      timeframe: { start: "2026-01-01", end: "2026-12-31" },
      source: { provider: "internal", url: "https://example.com" },
      meta: { datasetLabel: "Q1", queryTimeMs: 120 },
    };
    const result = ChartSpecSchema.parse(full);
    expect(result.description).toBe("A chart");
    expect(result.timeframe?.start).toBe("2026-01-01");
    expect(result.meta?.queryTimeMs).toBe(120);
  });

  it("rejects non-string/number values in data records", () => {
    expect(() =>
      ChartSpecSchema.parse({
        ...validChartSpec,
        data: [{ date: true }],
      })
    ).toThrow();
  });
});

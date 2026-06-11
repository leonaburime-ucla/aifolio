import { describe, it, expect } from "vitest";
import { buildChartOption } from "~/features/recharts/lib";

describe("buildChartOption", () => {
  it("builds line chart with explicit keys", () => {
    const result = buildChartOption({
      id: "1",
      type: "line",
      title: "Test",
      data: [{ month: "Jan", sales: 100 }, { month: "Feb", sales: 200 }],
      xKey: "month",
      yKeys: ["sales"],
    });

    expect(result.xAxis).toEqual({ type: "category", data: ["Jan", "Feb"] });
    expect(result.yAxis).toEqual({ type: "value" });
    expect(result.series).toEqual([{ name: "sales", type: "line", data: [100, 200] }]);
  });

  it("builds area chart as line with areaStyle", () => {
    const result = buildChartOption({
      id: "4",
      type: "area",
      title: "Area",
      data: [{ x: "1", y: 5 }, { x: "2", y: 10 }],
      xKey: "x",
      yKeys: ["y"],
    });

    const series = result.series as any[];
    expect(series[0].type).toBe("line");
    expect(series[0].areaStyle).toEqual({});
  });

  it("builds scatter chart with value axes", () => {
    const result = buildChartOption({
      id: "5",
      type: "scatter",
      title: "Scatter",
      data: [{ x: 1, y: 2 }, { x: 3, y: 4 }],
      xKey: "x",
      yKeys: ["y"],
    });

    const xAxis = result.xAxis as any;
    expect(xAxis.type).toBe("value");
    const series = result.series as any[];
    expect(series[0].type).toBe("scatter");
  });

  it("builds bar chart with categories", () => {
    const result = buildChartOption({
      id: "bar1",
      type: "bar",
      title: "Bar",
      data: [{ x: "A", y: 10 }, { x: "B", y: 20 }],
      xKey: "x",
      yKeys: ["y"],
    });

    const xAxis = result.xAxis as any;
    expect(xAxis.type).toBe("category");
    expect(xAxis.data).toEqual(["A", "B"]);
  });

  it("handles empty data gracefully", () => {
    const result = buildChartOption({
      id: "6",
      type: "line",
      title: "Empty",
      data: [],
      xKey: "x",
      yKeys: ["y"],
    });

    expect(result.xAxis).toEqual({ type: "category", data: [] });
    expect(result.series).toEqual([]);
  });

  it("builds multi-series line chart", () => {
    const result = buildChartOption({
      id: "9",
      type: "line",
      title: "City rents",
      data: [
        { year: "2000", Manhattan: 2500, London: 1800, Paris: 1600 },
        { year: "2010", Manhattan: 3200, London: 2400, Paris: 2100 },
      ],
      xKey: "year",
      yKeys: ["Manhattan", "London", "Paris"],
    });

    expect(result.legend).toEqual({ data: ["Manhattan", "London", "Paris"] });
    expect(result.series).toEqual([
      { name: "Manhattan", type: "line", data: [2500, 3200] },
      { name: "London", type: "line", data: [1800, 2400] },
      { name: "Paris", type: "line", data: [1600, 2100] },
    ]);
  });

  it("single series does not include legend", () => {
    const result = buildChartOption({
      id: "10",
      type: "line",
      title: "Single",
      data: [{ x: "A", y: 10 }],
      xKey: "x",
      yKeys: ["y"],
    });

    expect(result.legend).toBeUndefined();
  });
});

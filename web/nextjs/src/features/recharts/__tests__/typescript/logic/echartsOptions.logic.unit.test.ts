import { describe, expect, it } from "vitest";
import { getEChartsOption } from "@/features/recharts/typescript/logic/echartsOptions.logic";

const specBase = {
  id: "chart",
  title: "Chart",
  xKey: "x",
  yKeys: ["y"],
};

describe("echartsOptions.logic", () => {
  it("builds heatmap option and coerces invalid cells to zero", () => {
    const option = getEChartsOption({
      spec: {
        ...specBase,
        type: "heatmap",
        xLabel: "X",
        yLabel: "Y",
        yKeys: ["a", "b"],
        data: [{ x: "r1", a: 1, b: "x" }],
      },
    });

    expect(option?.series?.[0]).toMatchObject({ type: "heatmap" });
    expect((option?.series?.[0] as { data: number[][] }).data).toEqual([
      [0, 0, 1],
      [0, 1, 0],
    ]);
  });

  it("builds boxplot and bar/histogram options", () => {
    const box = getEChartsOption({
      spec: {
        ...specBase,
        type: "box",
        yKeys: ["q1", "q2"],
        data: [{ x: "A", q1: 1, q2: "2" }],
      },
    });
    expect(box?.series?.[0]).toMatchObject({ type: "boxplot", data: [[1, 2]] });

    const bar = getEChartsOption({
      spec: {
        ...specBase,
        type: "bar",
        data: [{ x: "A", y: 1 }, { x: "B", y: "2" }],
      },
    });
    expect(bar?.series?.[0]).toMatchObject({ type: "bar", data: [1, 2] });

    const histogram = getEChartsOption({
      spec: {
        ...specBase,
        type: "histogram",
        data: [{ x: "A", y: 3 }],
      },
    });
    expect(histogram?.series?.[0]).toMatchObject({ type: "bar", data: [3] });
  });

  it("builds tree option for dendrogram with rooted and parent-child nodes", () => {
    const option = getEChartsOption({
      spec: {
        ...specBase,
        type: "dendrogram",
        data: [
          { id: "root" },
          { id: "child", parent: "root" },
        ],
      },
    });

    expect(option?.series?.[0]).toMatchObject({ type: "tree" });
    const treeData = (option?.series?.[0] as { data: Array<{ name: string; children?: unknown[] }> }).data;
    expect(treeData[0]?.name).toBe("root");
  });

  it("builds tree fallback when no explicit root exists", () => {
    const option = getEChartsOption({
      spec: {
        ...specBase,
        type: "dendrogram",
        data: [{ id: "child", parent: "ghost" }],
      },
    });

    const treeData = (option?.series?.[0] as { data: Array<{ name: string; children?: unknown[] }> }).data;
    expect(treeData.some((node) => node.name === "ghost")).toBe(true);
  });

  it("ignores dendrogram rows without id/label/name", () => {
    const option = getEChartsOption({
      spec: {
        ...specBase,
        type: "dendrogram",
        data: [{ parent: "ghost" }],
      },
    });

    const treeData = (option?.series?.[0] as { data: Array<{ name: string }> }).data;
    expect(treeData).toEqual([]);
  });

  it("builds scatter/biplot options including tooltip and feature labels", () => {
    const scatter = getEChartsOption({
      spec: {
        ...specBase,
        type: "scatter",
        xLabel: "PC1",
        yLabel: "PC2",
        data: [{ x: "1.23", y: 2.5, feature: "feat-a" }],
      },
    });

    expect(scatter?.series?.[0]).toMatchObject({ type: "scatter" });
    const tooltipFormatter = (scatter?.tooltip as { formatter?: (params: unknown) => string }).formatter;
    expect(
      tooltipFormatter?.({ data: { x: "abc", y: 2.5, feature: "feat-a" } })
    ).toContain("feat-a");
    expect(
      tooltipFormatter?.({ data: { x: "4.2", y: "3.1" } })
    ).toContain("PC1: 4.2");
    expect(tooltipFormatter?.(undefined as unknown as { data: Record<string, unknown> })).toContain(
      "PC1: 0"
    );
    expect(
      tooltipFormatter?.({ data: { x: 5 } })
    ).toContain("PC2: 0");

    const labelFormatter = (
      scatter?.series?.[0] as {
        label?: { formatter?: (params: { data?: { feature?: string } }) => string };
      }
    ).label?.formatter;
    expect(labelFormatter?.({ data: { feature: "feat-a" } })).toBe("feat-a");
    expect(labelFormatter?.({ data: {} })).toBe("");

    const biplot = getEChartsOption({
      spec: {
        ...specBase,
        type: "biplot",
        data: [{ x: 1, y: 2 }],
      },
    });
    expect(biplot?.series?.[0]).toMatchObject({ type: "scatter" });
    const biplotTooltipFormatter = (biplot?.tooltip as { formatter?: (params: unknown) => string }).formatter;
    expect(biplotTooltipFormatter?.({ data: { x: 1, y: 2 } })).toContain("x: 1");
    expect(biplotTooltipFormatter?.({ data: { x: 1, y: 2 } })).toContain("y: 2");
  });

  it("returns null for unsupported chart types", () => {
    expect(
      getEChartsOption({
        spec: {
          ...specBase,
          type: "line",
          data: [{ x: 1, y: 2 }],
        },
      })
    ).toBeNull();
  });

  it("rotates bar labels when category count is large", () => {
    const manyRows = Array.from({ length: 11 }, (_, i) => ({ x: `row-${i}`, y: i + 1 }));
    const option = getEChartsOption({
      spec: {
        ...specBase,
        type: "bar",
        data: manyRows,
      },
    });

    expect((option?.xAxis as { axisLabel?: { rotate?: number } }).axisLabel?.rotate).toBe(35);
  });
});

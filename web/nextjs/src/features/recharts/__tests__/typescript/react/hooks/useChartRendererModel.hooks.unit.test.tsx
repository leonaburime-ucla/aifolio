import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useChartRendererModel } from "@/features/recharts/typescript/react/hooks/useChartRendererModel.hooks";

describe("useChartRendererModel", () => {
  it("derives yKeys, scatter label key, coerced data and chart props", () => {
    const spec = {
      id: "s1",
      title: "Scatter",
      type: "scatter" as const,
      xKey: "x",
      yKeys: ["y", "z"],
      data: [
        { x: 1, y: "2.5", z: "x", feature: "f1" },
        { x: 2, y: 3, z: "4" },
      ],
    };

    const { result } = renderHook(() => useChartRendererModel({ spec }));

    expect(result.current.isExpanded).toBe(false);
    expect(result.current.yKeys).toEqual(["y", "z"]);
    expect(result.current.scatterLabelKey).toBe("feature");
    expect(result.current.data).toEqual([
      { x: 1, y: 2.5, z: "x", feature: "f1" },
      { x: 2, y: 3, z: 4 },
    ]);
    expect(result.current.chartProps.margin).toEqual({
      top: 12,
      right: 20,
      left: 28,
      bottom: 40,
    });

    act(() => {
      result.current.setIsExpanded(true);
    });
    expect(result.current.isExpanded).toBe(true);
  });

  it("resolves scatter label key fallback order and handles missing yKeys/data", () => {
    const { result: withLabel } = renderHook(() =>
      useChartRendererModel({
        spec: {
          id: "l1",
          title: "Label",
          type: "scatter",
          xKey: "x",
          yKeys: ["y"],
          data: [{ x: 1, y: 2, label: "row" }],
        },
      })
    );
    expect(withLabel.current.scatterLabelKey).toBe("label");

    const { result: withName } = renderHook(() =>
      useChartRendererModel({
        spec: {
          id: "n1",
          title: "Name",
          type: "scatter",
          xKey: "x",
          yKeys: ["y"],
          data: [{ x: 1, y: 2, name: "row" }],
        },
      })
    );
    expect(withName.current.scatterLabelKey).toBe("name");

    const { result: fallback } = renderHook(() =>
      useChartRendererModel({
        spec: {
          id: "f1",
          title: "Fallback",
          type: "line",
          xKey: "x",
          yKeys: undefined as unknown as string[],
          data: undefined as unknown as Array<Record<string, unknown>>,
        },
      })
    );
    expect(fallback.current.yKeys).toEqual([]);
    expect(fallback.current.scatterLabelKey).toBeNull();
    expect(fallback.current.data).toEqual([]);

    const { result: noLabelKey } = renderHook(() =>
      useChartRendererModel({
        spec: {
          id: "plain1",
          title: "Plain",
          type: "scatter",
          xKey: "x",
          yKeys: ["y"],
          data: [{ x: 1, y: 2 }],
        },
      })
    );
    expect(noLabelKey.current.scatterLabelKey).toBeNull();
  });
});

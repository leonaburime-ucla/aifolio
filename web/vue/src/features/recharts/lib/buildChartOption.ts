import type { ChartSpec } from "~/composables/useChartStore";
import { getEChartsOption } from "@aifolio/frontend-core/recharts";

export type EChartsOption = Record<string, unknown>;

export function buildChartOption(spec: ChartSpec): EChartsOption {
  if (!spec.data || spec.data.length === 0) {
    return { tooltip: {}, xAxis: { type: "category", data: [] }, yAxis: { type: "value" }, series: [] };
  }

  const shared = getEChartsOption({ spec });
  if (shared) {
    if (spec.type === "scatter" || spec.type === "biplot") {
      const yKey = spec.yKeys[0];
      return {
        ...shared,
        dataset: { source: spec.data },
        series: [
          {
            ...(shared as any).series?.[0],
            data: undefined,
            encode: { x: spec.xKey, y: yKey },
          },
        ],
      } as EChartsOption;
    }
    return shared as EChartsOption;
  }

  const xKey = spec.xKey || Object.keys(spec.data[0])[0] || "x";
  const xData = spec.data.map((d) => String(d[xKey] ?? ""));

  if (spec.type === "line" || spec.type === "area") {
    const seriesKeys = spec.yKeys;
    const seriesType = spec.type === "area" ? "line" : "line";

    const series = seriesKeys.map((key) => ({
      name: key,
      type: seriesType,
      data: spec.data.map((d) => Number(d[key] ?? 0)),
      ...(spec.type === "area" ? { areaStyle: {} } : {}),
    }));

    return {
      tooltip: { trigger: "axis" },
      legend: seriesKeys.length > 1 ? { data: seriesKeys } : undefined,
      xAxis: { type: "category", data: xData },
      yAxis: { type: "value" },
      series,
    };
  }

  const yKey = spec.yKeys[0] || "y";
  return {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: xData },
    yAxis: { type: "value" },
    series: [{ name: yKey, type: spec.type, data: spec.data.map((d) => Number(d[yKey] ?? 0)) }],
  };
}

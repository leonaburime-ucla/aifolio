import type { EChartsOption } from "echarts";
import type {
  EChartsOptionBuilder,
  ScatterFormatterParams,
  TreeNode,
} from "@/features/recharts/__types__/typescript/logic/echartsOptions.types";

function toNumber(value: unknown): number {
  if (typeof value === "number") return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

/**
 * Build an ECharts option for heatmap charts.
 * @param params - Required parameters.
 * @param params.spec - ChartSpec with type `heatmap`.
 * @returns ECharts option object.
 */
function buildHeatmapOption({ spec }: EChartsOptionBuilder): EChartsOption {
  const xLabels = spec.data.map((row) => String(row[spec.xKey]));
  const yLabels = spec.yKeys;
  const values: Array<[number, number, number]> = [];

  spec.data.forEach((row, xIndex) => {
    yLabels.forEach((key, yIndex) => {
      const raw = row[key];
      const value = typeof raw === "number" ? raw : Number(raw);
      values.push([xIndex, yIndex, Number.isFinite(value) ? value : 0]);
    });
  });

  return {
    tooltip: { position: "top" },
    grid: { left: 60, right: 20, top: 20, bottom: 110 },
    xAxis: {
      type: "category",
      data: xLabels,
      axisLabel: { fontSize: 10 },
      name: spec.xLabel,
    },
    yAxis: {
      type: "category",
      data: yLabels,
      axisLabel: { fontSize: 10 },
      name: spec.yLabel,
    },
    visualMap: {
      min: Math.min(...values.map((v) => v[2])),
      max: Math.max(...values.map((v) => v[2])),
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      itemHeight: 12,
    },
    series: [
      {
        type: "heatmap",
        data: values,
        label: { show: false },
      },
    ],
  };
}

/**
 * Build an ECharts option for boxplot charts.
 * @param params - Required parameters.
 * @param params.spec - ChartSpec with type `box`.
 * @returns ECharts option object.
 */
function buildBoxplotOption({ spec }: EChartsOptionBuilder): EChartsOption {
  const categories = spec.data.map((row) => String(row[spec.xKey]));
  const seriesData = spec.data.map((row) =>
    spec.yKeys.map((key) => {
      const value = row[key];
      return typeof value === "number" ? value : Number(value);
    })
  );

  return {
    tooltip: { trigger: "item" },
    grid: { left: 60, right: 20, top: 20, bottom: 40 },
    xAxis: {
      type: "category",
      data: categories,
      name: spec.xLabel,
    },
    yAxis: {
      type: "value",
      name: spec.yLabel,
    },
    series: [
      {
        type: "boxplot",
        data: seriesData,
      },
    ],
  };
}

/**
 * Convert flat rows with id/parent into a tree structure.
 * @param params - Required parameters.
 * @param params.spec - ChartSpec containing flat tree rows.
 * @returns Hierarchical tree nodes.
 */
function buildTree({ spec }: EChartsOptionBuilder): TreeNode[] {
  const nodes: Record<string, TreeNode> = {};
  const parents: Record<string, string | null> = {};

  spec.data.forEach((row) => {
    const id = String(row.id ?? row.label ?? row.name ?? "");
    if (!id) return;
    nodes[id] = nodes[id] ?? { name: id };
    parents[id] = row.parent ? String(row.parent) : null;
  });

  const roots: TreeNode[] = [];
  Object.entries(nodes).forEach(([id, node]) => {
    const parentId = parents[id];
    if (!parentId) {
      roots.push(node);
      return;
    }
    const parent = nodes[parentId] ?? { name: parentId };
    nodes[parentId] = parent;
    parent.children = parent.children ?? [];
    parent.children.push(node);
  });

  return roots.length ? roots : Object.values(nodes);
}

/**
 * Build an ECharts option for dendrogram-like tree charts.
 * @param params - Required parameters.
 * @param params.spec - ChartSpec with type `dendrogram`.
 * @returns ECharts option object.
 */
function buildDendrogramOption({ spec }: EChartsOptionBuilder): EChartsOption {
  return {
    tooltip: { trigger: "item", triggerOn: "mousemove" },
    series: [
      {
        type: "tree",
        data: buildTree({ spec }),
        top: "5%",
        left: "12%",
        bottom: "5%",
        right: "20%",
        symbolSize: 8,
        orient: "LR",
        label: {
          position: "left",
          verticalAlign: "middle",
          align: "right",
          fontSize: 10,
        },
        leaves: {
          label: {
            position: "right",
            verticalAlign: "middle",
            align: "left",
          },
        },
        expandAndCollapse: true,
        animationDuration: 550,
        animationDurationUpdate: 750,
      },
    ],
  };
}

/**
 * Build an ECharts scatter option with built-in zoom controls.
 * @param params - Required parameters.
 * @param params.spec - ChartSpec with type `scatter` or `biplot`.
 * @returns ECharts option object.
 */
function buildScatterOption({ spec }: EChartsOptionBuilder): EChartsOption {
  const yKey = spec.yKeys[0];
  const hasFeatureLabels =
    Array.isArray(spec.data) && spec.data.length > 0 && "feature" in spec.data[0];

  return {
    tooltip: {
      trigger: "item",
      formatter: (params: ScatterFormatterParams) => {
        const point = params?.data ?? {};
        const label = point.feature ? `${point.feature}<br/>` : "";
        return (
          `${label}${spec.xLabel ?? spec.xKey}: ${parseFloat(toNumber(point[spec.xKey]).toFixed(7))}<br/>` +
          `${spec.yLabel ?? yKey}: ${parseFloat(toNumber(point[yKey]).toFixed(7))}`
        );
      },
    },
    grid: { left: 70, right: 30, top: 20, bottom: 70 },
    xAxis: {
      type: "value",
      name: spec.xLabel ?? spec.xKey,
      axisLabel: { fontSize: 11 },
    },
    yAxis: {
      type: "value",
      name: spec.yLabel ?? yKey,
      axisLabel: { fontSize: 11 },
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0, yAxisIndex: 0 },
      { type: "slider", xAxisIndex: 0, bottom: 24, height: 16 },
      { type: "slider", yAxisIndex: 0, right: 4, width: 14 },
    ],
    series: [
      {
        type: "scatter",
        symbolSize: 10,
        data: spec.data,
        label: hasFeatureLabels
          ? {
            show: true,
            position: "right",
            color: "#52525b",
            fontSize: 11,
            formatter: (params: ScatterFormatterParams) => {
              const feature = params.data?.feature;
              return typeof feature === "string" ? feature : "";
            },
          }
          : { show: false },
      },
    ],
  };
}

/**
 * Build an ECharts bar option with horizontal zoom for many features.
 * @param params - Required parameters.
 * @param params.spec - ChartSpec with type `bar` or `histogram`.
 * @returns ECharts option object.
 */
function buildBarOption({ spec }: EChartsOptionBuilder): EChartsOption {
  const yKey = spec.yKeys[0];
  const categories = spec.data.map((row) => String(row[spec.xKey]));
  const values = spec.data.map((row) => toNumber(row[yKey]));

  return {
    tooltip: { trigger: "axis" },
    grid: { left: 70, right: 30, top: 20, bottom: 90 },
    xAxis: {
      type: "category",
      data: categories,
      name: spec.xLabel ?? spec.xKey,
      axisLabel: { interval: 0, rotate: categories.length > 10 ? 35 : 0, fontSize: 11 },
    },
    yAxis: {
      type: "value",
      name: spec.yLabel ?? yKey,
      axisLabel: { fontSize: 11 },
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0 },
      { type: "slider", xAxisIndex: 0, bottom: 24, height: 16 },
    ],
    series: [
      {
        type: "bar",
        data: values,
      },
    ],
  };
}

/**
 * Select the appropriate ECharts option builder for the given chart type.
 * @param params - Required parameters.
 * @param params.spec - ChartSpec to map into an ECharts option.
 * @returns ECharts option or `null` when unsupported.
 */
export function getEChartsOption({ spec }: EChartsOptionBuilder): EChartsOption | null {
  if (spec.type === "heatmap") return buildHeatmapOption({ spec });
  if (spec.type === "box") return buildBoxplotOption({ spec });
  if (spec.type === "dendrogram") return buildDendrogramOption({ spec });
  if (spec.type === "scatter" || spec.type === "biplot") return buildScatterOption({ spec });
  if (spec.type === "bar" || spec.type === "histogram") return buildBarOption({ spec });
  return null;
}
